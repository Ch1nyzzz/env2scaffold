from __future__ import annotations

import glob
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from verl.interactions.base import BaseInteraction


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "env2scaffold" / "augmentation" / "augmented_env.py"
        if candidate.exists():
            return parent
    raise RuntimeError("Could not locate repo root containing env2scaffold/augmentation/augmented_env.py.")


REPO_ROOT = _find_repo_root()
AUGMENTATION_DIR = REPO_ROOT / "env2scaffold" / "augmentation"
if str(AUGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(AUGMENTATION_DIR))

from augmented_env import AugmentedAlfWorldEnv  # noqa: E402


PARSE_ERROR_REWARD = -0.25
EXECUTION_ERROR_REWARD = -0.5
DEFAULT_SUCCESS_REWARD = 3.0

ALFWORLD_CACHE = Path(os.path.expanduser("~/.cache/alfworld/json_2.1.1"))


def _scan_game_files(split: str = "train") -> List[str]:
    """Scan ALFWorld cache for all game.tw-pddl files in a given split."""
    split_dir = ALFWORLD_CACHE / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ALFWorld data not found at {split_dir}. Run `alfworld-download` first.")
    files = sorted(str(p) for p in split_dir.glob("*/trial_*/game.tw-pddl"))
    if not files:
        raise FileNotFoundError(f"No game files found in {split_dir}")
    return files


@dataclass
class AlfWorldInstanceState:
    env: Any
    game_file: str
    initial_observation: str
    latest_observation: str
    latest_infos: Dict[str, Any]
    max_episode_steps: int
    use_augmented_env: bool
    step_count: int = 0
    done: bool = False
    won: bool = False
    step_rewards: List[float] = field(default_factory=list)
    step_extras: List[Dict[str, Any]] = field(default_factory=list)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _flatten_infos(infos: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: (value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value)
        for key, value in infos.items()
    }


def _normalize_reset(result: Tuple[Any, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    obs, infos = result
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    if isinstance(infos, dict):
        infos = _flatten_infos(infos)
    return obs, infos


def _normalize_step(result: Tuple[Any, Any, Any, Dict[str, Any]]) -> Tuple[str, float, bool, Dict[str, Any]]:
    obs_raw, score_raw, done_raw, infos_raw = result
    obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
    score = score_raw[0] if isinstance(score_raw, (list, tuple)) else score_raw
    done = done_raw[0] if isinstance(done_raw, (list, tuple)) else done_raw
    infos = _flatten_infos(infos_raw) if isinstance(infos_raw, dict) else infos_raw
    return obs, float(score), bool(done), infos


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _sanitize_action(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = text.replace("```", "").strip()
    for line in text.splitlines():
        candidate = line.strip().strip("`").strip('"').strip("'")
        if not candidate:
            continue
        candidate = re.sub(r"^(action|command)\s*:\s*", "", candidate, flags=re.I)
        return candidate
    return ""


def _latest_assistant_action(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        return _sanitize_action(_extract_text_content(message.get("content")))
    return ""


class AlfWorldInteraction(BaseInteraction):
    """
    Multi-turn ALFWorld interaction for verl/GRPO.

    Supports two modes:
    1. Static: game_file provided in interaction_kwargs (from parquet)
    2. Dynamic: no game_file → randomly sample from ALFWorld cache

    Config fields:
      - max_episode_steps: int, default 50
      - use_augmented_env: bool, default true
      - verbose: bool, default false
      - train_split: str, default "train" (for dynamic game selection)
      - val_split: str, default "valid_seen" (for dynamic game selection)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", "alfworld_augmented_env")
        self.default_max_episode_steps = int(config.get("max_episode_steps", 50))
        self.default_use_augmented_env = _coerce_bool(config.get("use_augmented_env", True), default=True)
        self.default_verbose = _coerce_bool(config.get("verbose", False), default=False)
        self._instance_dict: Dict[str, AlfWorldInstanceState] = {}

        # Scan available game files for dynamic selection
        train_split = config.get("train_split", "train")
        val_split = config.get("val_split", "valid_seen")
        self._train_games = _scan_game_files(train_split)
        self._val_games = _scan_game_files(val_split)
        self._rng = random.Random(config.get("seed", None))

    def _pick_random_game(self, split: str = "train") -> str:
        games = self._train_games if split == "train" else self._val_games
        return self._rng.choice(games)

    def _make_base_env(self, game_file: str, max_episode_steps: int):
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            facts=True,
            extras=["gamefile"],
        )
        env_id = textworld.gym.register_games(
            [str(game_file)],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_episode_steps,
            wrappers=[AlfredDemangler, AlfredInfos],
        )
        return textworld.gym.make(env_id)

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        if isinstance(interaction_kwargs, dict):
            merged_kwargs = {**interaction_kwargs, **kwargs}
        else:
            merged_kwargs = dict(kwargs)

        game_file = merged_kwargs.get("game_file")
        # Dynamic mode: pick a random game if not specified
        if not game_file:
            split = merged_kwargs.get("split", "train")
            game_file = self._pick_random_game(split)

        max_episode_steps = int(merged_kwargs.get("max_episode_steps", self.default_max_episode_steps))
        use_augmented_env = _coerce_bool(
            merged_kwargs.get("use_augmented_env", self.default_use_augmented_env),
            default=self.default_use_augmented_env,
        )
        verbose = _coerce_bool(merged_kwargs.get("verbose", self.default_verbose), default=self.default_verbose)

        base_env = self._make_base_env(str(game_file), max_episode_steps)
        env = AugmentedAlfWorldEnv(base_env, verbose=verbose) if use_augmented_env else base_env

        if use_augmented_env:
            obs, infos = env.reset()
        else:
            obs, infos = _normalize_reset(env.reset())

        self._instance_dict[instance_id] = AlfWorldInstanceState(
            env=env,
            game_file=str(game_file),
            initial_observation=obs,
            latest_observation=obs,
            latest_infos=infos,
            max_episode_steps=max_episode_steps,
            use_augmented_env=use_augmented_env,
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        state = self._instance_dict[instance_id]
        if state.done:
            return True, state.latest_observation, 0.0, self._build_extra(state, action="", env_score=0.0)

        action = _latest_assistant_action(messages)
        if not action:
            return (
                False,
                "Reply with exactly one ALFWorld action command and nothing else.",
                PARSE_ERROR_REWARD,
                self._build_extra(state, action="", env_score=0.0, error_type="parse_error"),
            )

        try:
            if state.use_augmented_env:
                next_obs, env_score, done, infos = state.env.step(action)
            else:
                next_obs, env_score, done, infos = _normalize_step(state.env.step([action]))
        except Exception as exc:
            return (
                False,
                f"Environment execution failed for action `{action}`: {exc}",
                EXECUTION_ERROR_REWARD,
                self._build_extra(state, action=action, env_score=0.0, error_type="execution_error"),
            )

        progress_reward = float(infos.get("progress_reward", 0.0))
        if bool(infos.get("won", False)) and progress_reward == 0.0:
            progress_reward = DEFAULT_SUCCESS_REWARD

        state.step_count += 1
        state.latest_observation = next_obs
        state.latest_infos = infos
        state.done = bool(done)
        state.won = bool(infos.get("won", False))
        state.step_rewards.append(progress_reward)

        extra = self._build_extra(state, action=action, env_score=env_score)
        state.step_extras.append(extra)
        return state.done, next_obs, progress_reward, extra

    def _build_extra(
        self,
        state: AlfWorldInstanceState,
        action: str,
        env_score: float,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        infos = state.latest_infos or {}
        extra = {
            "game_file": state.game_file,
            "step_count": state.step_count,
            "action": action,
            "env_score": float(env_score),
            "won": bool(infos.get("won", False)),
            "progress_events": list(infos.get("progress_events", []) or []),
            "progress_reward": float(infos.get("progress_reward", 0.0) or 0.0),
            "progress_score": float(infos.get("progress_score", 0.0) or 0.0),
            "progress_milestones": list(infos.get("progress_milestones", []) or []),
            "admissible_commands": list(infos.get("admissible_commands", []) or []),
        }
        if error_type is not None:
            extra["error_type"] = error_type
        if state.use_augmented_env and hasattr(state.env, "get_augmentation_log"):
            augmentation_log = state.env.get_augmentation_log()
            if augmentation_log:
                extra["last_augmentation"] = augmentation_log[-1]
        return extra

    async def calculate_score(self) -> float:
        return 0.0

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        if not instance_id:
            return
        state = self._instance_dict.pop(instance_id, None)
        if state is None:
            return
        try:
            state.env.close()
        except Exception:
            pass
