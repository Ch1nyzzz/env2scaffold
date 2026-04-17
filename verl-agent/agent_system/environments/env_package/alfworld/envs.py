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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

import sys
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

# Import augmented environment from env2scaffold
_AUGMENTED_ENV_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'env2scaffold', 'augmentation')
_AUGMENTED_ENV_DIR = os.path.abspath(_AUGMENTED_ENV_DIR)
if _AUGMENTED_ENV_DIR not in sys.path:
    sys.path.insert(0, _AUGMENTED_ENV_DIR)
from augmented_env import AugmentedAlfWorldEnv

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

class AugmentedAlfworldWorker:
    """
    Ray remote actor with AugmentedAlfWorldEnv wrapper.
    Provides enhanced feedback for failed actions instead of 'Nothing happens.'
    """

    def __init__(self, config, seed, base_env):
        raw_env = base_env.init_env(batch_size=1)
        raw_env.seed(seed)
        self.env = AugmentedAlfWorldEnv(raw_env, verbose=False)

    def _rewrap_infos(self, infos, obs):
        """Match the vanilla AlfworldWorker batch_size=1 info shape."""
        wrapped_infos = {}
        for key, value in infos.items():
            # The downstream env manager unpacks batch_size=1 info values with
            # `info[k] = info[k][0]`. Keep every field wrapped in an outer list
            # so structured values such as admissible_commands/facts survive
            # that flattening step intact.
            wrapped_infos[key] = [value]
        wrapped_infos["observation_text"] = [obs]
        return wrapped_infos

    def step(self, action):
        obs, score, done, infos = self.env.step(action)
        infos = self._rewrap_infos(infos, obs)
        return [obs], [score], [done], infos

    def reset(self):
        obs, infos = self.env.reset()
        infos = self._rewrap_infos(infos, obs)
        return [obs], infos

    def getobs(self):
        return None


class AugmentedAlfworldEnvs(AlfworldEnvs):
    """AlfworldEnvs variant that uses AugmentedAlfWorldEnv for enhanced feedback."""

    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        # Skip AlfworldEnvs.__init__ and do our own setup
        gym.Env.__init__(self)

        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        self.use_progress_reward = env_kwargs.get('use_progress_reward', False)
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        env_worker = ray.remote(**resources_per_worker)(AugmentedAlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        text_obs_list, image_obs_list, rewards_list, dones_list, info_list = super().step(actions)
        if self.use_progress_reward:
            # Plan-driven dense shaping + return normalised to vanilla scale.
            # Non-terminal steps: emit per-step progress (pipeline C delta).
            # Terminal success step: emit (10 - accumulated) so trajectory
            # return equals 10 regardless of how many milestones fired,
            # matching vanilla / obs-aug success scale.
            for i, info in enumerate(info_list):
                progress = float(info.get('progress_reward', 0.0) or 0.0)
                won = bool(info.get('won', False))
                if won:
                    accumulated = float(info.get('progress_accumulated', 0.0) or 0.0)
                    # clip: if milestones already exceed 10 the terminal bonus is 0
                    rewards_list[i] = max(0.0, 10.0 - accumulated)
                else:
                    rewards_list[i] = progress
        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list


def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


def build_augmented_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AugmentedAlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
