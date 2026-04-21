# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np
import importlib.util
import os


def _load_augmented_webshop_env():
    """Load the generated env2scaffold WebShop wrapper without patching WebShop."""
    module_path = os.environ.get("ENV2SCAFFOLD_WEBSHOP_AUG_PATH")
    if module_path is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../..")
        )
        module_path = os.path.join(
            repo_root,
            "env2scaffold",
            "generated",
            "webshop_codex",
            "augmentation",
            "augmented_env.py",
        )

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            "WebShop obs-aug requested, but generated wrapper was not found: "
            f"{module_path}"
        )

    spec = importlib.util.spec_from_file_location("env2scaffold_webshop_augmented_env", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AugmentedWebShopEnv


def _native_env(env):
    return getattr(env, "env", env)


def _is_success(raw_reward):
    return float(raw_reward) == 1.0

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs = dict(env_kwargs or {})
        self.use_augmented_env = bool(env_kwargs.pop('use_augmented_env', False))
        self.use_progress_reward = bool(env_kwargs.pop('use_progress_reward', False))
        self.max_progress_reward = float(env_kwargs.pop('progress_reward_scale', 3.0))

        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        if self.use_augmented_env:
            self.env = _load_augmented_webshop_env()(self.env)
        self._reset_progress_state()

    def _reset_progress_state(self):
        self.progress_accumulated = 0.0
        self.progress_flags = {
            'searched': False,
            'opened_product': False,
            'selected_option_groups': set(),
        }

    def _session_state(self):
        env = _native_env(self.env)
        server = getattr(env, 'server', None)
        session_id = getattr(env, 'session', None)
        if server is None or session_id is None:
            return {}
        return getattr(server, 'user_sessions', {}).get(session_id, {}) or {}

    def _progress_step(self):
        """Non-leaking shaping from public state the agent has already reached."""
        if not self.use_progress_reward:
            return 0.0, []

        session = self._session_state()
        delta = 0.0
        fired = []

        if session.get('keywords') and not self.progress_flags['searched']:
            delta += 0.25
            fired.append('searched')
            self.progress_flags['searched'] = True

        if session.get('asin') and not self.progress_flags['opened_product']:
            delta += 0.50
            fired.append('opened_product')
            self.progress_flags['opened_product'] = True

        selected_options = set((session.get('options') or {}).keys())
        new_option_groups = selected_options - self.progress_flags['selected_option_groups']
        if new_option_groups:
            option_delta = 0.25 * len(new_option_groups)
            delta += option_delta
            fired.extend(f'selected_option:{name}' for name in sorted(new_option_groups))
            self.progress_flags['selected_option_groups'].update(new_option_groups)

        remaining = max(0.0, self.max_progress_reward - self.progress_accumulated)
        delta = min(delta, remaining)
        self.progress_accumulated += delta
        return delta, fired
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, raw_reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = raw_reward

        progress_reward, fired_progress = self._progress_step()
        info['progress_reward'] = progress_reward
        info['progress_accumulated'] = self.progress_accumulated
        info['progress_fired'] = fired_progress

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and _is_success(raw_reward):
            info['won'] = True
            if self.use_progress_reward:
                reward = max(0.0, 10.0 - self.progress_accumulated)
            else:
                reward = 10.0
        else:
            info['won'] = False
            reward = progress_reward if self.use_progress_reward else 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        self._reset_progress_state()
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info['progress_reward'] = 0.0
        info['progress_accumulated'] = 0.0
        info['progress_fired'] = []
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
