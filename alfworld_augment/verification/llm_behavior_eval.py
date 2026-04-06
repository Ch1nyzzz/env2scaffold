#!/usr/bin/env python3
"""
Behavioral A/B evaluation for original vs augmented ALFWorld environments.

Runs the same fixed LLM policy on the same task set in two conditions:
  1. Original environment
  2. Augmented environment

Metrics:
  - completion_rate
  - avg_steps
  - invalid_action_count
  - recovery_after_first_failure_rate

Outputs:
  verification/llm_behavior_eval_results.json

Example:
  python verification/llm_behavior_eval.py --max-steps 25 --max-games 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import textworld
import textworld.gym
from openai import OpenAI
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from augmented_env import AugmentedAlfWorldEnv  # noqa: E402


DATA_ROOT = Path("/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen")
TRAJECTORY_DIR = REPO_ROOT / "probing" / "trajectories"
RESULTS_PATH = REPO_ROOT / "verification" / "llm_behavior_eval_results.json"

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
MAX_HISTORY_ITEMS = 6

SYSTEM_PROMPT = """You are controlling an ALFWorld household agent.

Your job is to solve the task one text action at a time.

Rules:
- Reply with exactly one action command and nothing else.
- Prefer exact commands from the admissible command list when possible.
- Use the latest observation carefully, especially after failures.
- Avoid passive loops. Do not repeat look/help/inventory unless the situation changed.
- If nothing useful is visible, explore by going to a new location.
- If an action fails, use the feedback to repair the plan on the next step.
- Never invent object ids or location ids. Copy names exactly from the observation or admissible list.
- If the target object is visible and a matching take/move/use/open action is available, do that instead of looking around.
- If you reach a closed container that may contain useful items, prefer opening it before leaving.
- If feedback says you are already holding an object, do not try to take it again. Do the next task-relevant action.
- Valid action styles include:
  go to <location>
  take <object> from <receptacle>
  move <object> to <receptacle>
  open <receptacle>
  close <receptacle>
  examine <object>
  use <object>
  heat <object> with <appliance>
  cool <object> with <appliance>
  clean <object> with <appliance>
  inventory
  look
"""


def load_together_api_key() -> str:
    for key in ("TOGETHER_API_KEY", "TOGETHER_AI_API_KEY", "together_ai_api"):
        value = os.environ.get(key)
        if value:
            return value

    env_candidates = [
        Path("/home/yuhan/nested-agent/.env"),
        Path("/home/yuhan/optpilot/.env"),
    ]
    for path in env_candidates:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in {"TOGETHER_API_KEY", "TOGETHER_AI_API_KEY", "together_ai_api"}:
                continue
            value = value.strip().strip("'").strip('"')
            if value:
                return value
    raise RuntimeError("Together API key not found in env or known .env files.")


def discover_game_files(max_games: Optional[int] = None) -> List[Path]:
    games: List[Path] = []
    for path in sorted(TRAJECTORY_DIR.glob("*.json")):
        stem = path.stem
        if "_trial_" not in stem:
            continue
        _, rest = stem.split("_trial_", 1)
        parts = rest.split("_")
        if len(parts) < 4:
            continue
        trial_id = "_".join(parts[:3])
        task_dir = "_".join(parts[3:])
        game_file = DATA_ROOT / task_dir / f"trial_{trial_id}" / "game.tw-pddl"
        if game_file.exists():
            games.append(game_file)
    if max_games is not None:
        games = games[:max_games]
    if not games:
        raise RuntimeError("No ALFWorld game files discovered from probing trajectories.")
    return games


def make_base_env(game_file: Path, max_episode_steps: int):
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


def unwrap_reset(result: Tuple[Any, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    obs, infos = result
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    if isinstance(infos, dict):
        infos = {
            k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
            for k, v in infos.items()
        }
    return obs, infos


def unwrap_step(result: Tuple[Any, Any, Any, Dict[str, Any]]) -> Tuple[str, float, bool, Dict[str, Any]]:
    obs_raw, score_raw, done_raw, infos_raw = result
    obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
    score = score_raw[0] if isinstance(score_raw, (list, tuple)) else score_raw
    done = done_raw[0] if isinstance(done_raw, (list, tuple)) else done_raw
    infos = infos_raw
    if isinstance(infos, dict):
        infos = {
            k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
            for k, v in infos.items()
        }
    return obs, score, done, infos


def sanitize_command(text: str) -> str:
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
    return "look"


def build_user_prompt(
    task_obs: str,
    current_obs: str,
    candidate_commands: Sequence[str],
    history: Sequence[Dict[str, str]],
    step_idx: int,
) -> str:
    history_lines = []
    for item in history[-MAX_HISTORY_ITEMS:]:
        history_lines.append(f"- action: {item['command']}")
        history_lines.append(f"  observation: {item['observation']}")
    history_block = "\n".join(history_lines) if history_lines else "- none"
    candidate_block = "\n".join(f"- {cmd}" for cmd in candidate_commands) if candidate_commands else "- none"
    task_line = extract_task_line(task_obs)
    return f"""Step {step_idx}

Task:
{task_line}

Initial task observation:
{task_obs}

Current observation:
{current_obs}

Recent interaction history:
{history_block}

Candidate commands to choose from:
{candidate_block}

Reply with exactly one command from the candidate list."""


def extract_task_line(task_obs: str) -> str:
    match = re.search(r"Your task is to:\s*(.+)", task_obs)
    return match.group(1).strip() if match else task_obs.strip()


def extract_base_type(name: str) -> str:
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def parse_task_info(task_obs: str) -> Dict[str, Optional[str]]:
    task = extract_task_line(task_obs).lower()
    info = {
        "task_kind": None,
        "target_object_type": None,
        "target_destination_type": None,
        "modifier": None,
    }
    match = re.search(r"put (?:some |a )?(?:(hot|cool|clean) )?([a-z]+) (?:in/on|on) (?:the )?([a-z]+)", task)
    if match:
        info["task_kind"] = "put"
        info["modifier"] = match.group(1)
        info["target_object_type"] = match.group(2)
        info["target_destination_type"] = match.group(3)
        return info
    match = re.search(r"look at (?:some |a )?([a-z]+) under (?:the )?([a-z]+)", task)
    if match:
        info["task_kind"] = "look_under_light"
        info["target_object_type"] = match.group(1)
        info["target_destination_type"] = match.group(2)
        return info
    return info


def extract_visible_entities(obs: str) -> List[str]:
    entities: List[str] = []
    for match in re.finditer(r"you see (.+?)(?:\.|$)", obs, flags=re.I):
        chunk = match.group(1).strip()
        if "nothing" in chunk.lower():
            continue
        chunk = re.sub(r"\band\b", ",", chunk, flags=re.I)
        for piece in chunk.split(","):
            piece = piece.strip()
            if not piece:
                continue
            piece = re.sub(r"^(a|an)\s+", "", piece, flags=re.I).strip()
            if piece:
                entities.append(piece)
    return entities


def infer_current_location(obs: str, history: Sequence[Dict[str, str]]) -> Optional[str]:
    match = re.search(r"You arrive at ([^.]+)\.", obs)
    if match:
        return match.group(1).strip()
    match = re.search(r"You are facing (?:the )?([^.]+)\.", obs)
    if match:
        raw = match.group(1).strip()
        if "," not in raw and " and " not in raw:
            return raw
    for item in reversed(history):
        cmd = item["command"]
        if cmd.startswith("go to "):
            return cmd[6:]
    return None


def extract_recently_picked_object(obs: str) -> Optional[str]:
    match = re.match(r"You pick up the (.+?) from the .+\.", obs)
    return match.group(1).strip() if match else None


def infer_held_object(history: Sequence[Dict[str, str]], current_obs: str) -> Optional[str]:
    current_pick = extract_recently_picked_object(current_obs)
    if current_pick:
        return current_pick

    holding_match = re.search(r"already holding ([^.]+?)\.", current_obs, flags=re.I)
    if holding_match:
        return holding_match.group(1).strip()

    held: Optional[str] = None
    for item in history:
        action = item["command"]
        observation = item["observation"]
        take_match = re.match(r"take (.+?) from .+$", action)
        if take_match and "Nothing happens." not in observation:
            held = take_match.group(1).strip()
        move_match = re.match(r"move (.+?) to .+$", action)
        if move_match and held == move_match.group(1).strip() and "Nothing happens." not in observation:
            held = None
    return held


def recent_entities_for_location(
    current_location: Optional[str],
    current_obs: str,
    history: Sequence[Dict[str, str]],
) -> List[str]:
    if not current_location:
        return []
    current_entities = extract_visible_entities(current_obs)
    if current_entities:
        return current_entities
    for item in reversed(history):
        if current_location not in item["observation"]:
            continue
        entities = extract_visible_entities(item["observation"])
        if entities:
            return entities
    return []


def generate_candidate_commands(
    task_obs: str,
    current_obs: str,
    admissible_commands: Sequence[str],
    history: Sequence[Dict[str, str]],
) -> List[str]:
    task_info = parse_task_info(task_obs)
    target_object_type = task_info["target_object_type"]
    target_destination_type = task_info["target_destination_type"]
    modifier = task_info["modifier"]
    task_kind = task_info["task_kind"]
    current_location = infer_current_location(current_obs, history)
    visible_entities = recent_entities_for_location(current_location, current_obs, history)
    picked_obj = extract_recently_picked_object(current_obs)
    held_object = infer_held_object(history, current_obs) or picked_obj
    target_held = (
        held_object is not None
        and target_object_type is not None
        and extract_base_type(held_object).lower() == target_object_type
    )

    target_objects = [
        entity for entity in visible_entities
        if target_object_type and extract_base_type(entity).lower() == target_object_type
    ]
    visible_destinations = [
        entity for entity in visible_entities
        if target_destination_type and extract_base_type(entity).lower() == target_destination_type
    ]

    generated: List[str] = []
    already_holding_feedback = "already holding" in current_obs.lower()
    closed_match = re.search(r"The ([^.]+) is closed\.", current_obs)
    if closed_match:
        generated.append(f"open {closed_match.group(1).strip()}")

    if target_destination_type and target_held:
        generated.extend(
            cmd for cmd in admissible_commands
            if cmd.startswith("go to ")
            and extract_base_type(cmd[6:]).lower() == target_destination_type
        )

    for obj in target_objects[:2]:
        if current_location and not already_holding_feedback and not target_held:
            generated.append(f"take {obj} from {current_location}")
        if target_destination_type and target_held and task_kind == "put":
            for dest in visible_destinations[:2]:
                generated.append(f"move {held_object or obj} to {dest}")

        if task_kind == "look_under_light":
            for entity in visible_entities:
                if extract_base_type(entity).lower() == target_destination_type:
                    generated.append(f"use {entity}")
                    break

        if modifier == "hot":
            generated.append(f"heat {obj} with microwave 1")
        elif modifier == "cool":
            generated.append(f"cool {obj} with fridge 1")
        elif modifier == "clean":
            generated.append(f"clean {obj} with sinkbasin 1")

    if picked_obj or target_held:
        active_obj = picked_obj or held_object
        if task_kind == "look_under_light":
            for entity in visible_entities:
                if target_destination_type and extract_base_type(entity).lower() == target_destination_type:
                    generated.append(f"use {entity}")
                    break
        if modifier == "hot":
            generated.append(f"heat {active_obj} with microwave 1")
        elif modifier == "cool":
            generated.append(f"cool {active_obj} with fridge 1")
        elif modifier == "clean":
            generated.append(f"clean {active_obj} with sinkbasin 1")
        if target_destination_type and task_kind == "put":
            for dest in visible_destinations[:2]:
                generated.append(f"move {active_obj} to {dest}")

    filtered_admissible = list(admissible_commands)
    if generated:
        filtered_admissible = [
            cmd for cmd in filtered_admissible
            if cmd not in {"look", "help", "inventory"}
        ]
    if current_location:
        filtered_admissible = [
            cmd for cmd in filtered_admissible
            if cmd != f"go to {current_location}"
        ]

    candidate_commands: List[str] = []
    seen = set()
    for cmd in list(generated) + filtered_admissible:
        if cmd not in seen:
            seen.add(cmd)
            candidate_commands.append(cmd)
    return candidate_commands


class LLMPolicy:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def choose_action(
        self,
        task_obs: str,
        current_obs: str,
        candidate_commands: Sequence[str],
        history: Sequence[Dict[str, str]],
        step_idx: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=32,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        task_obs=task_obs,
                        current_obs=current_obs,
                        candidate_commands=candidate_commands,
                        history=history,
                        step_idx=step_idx,
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        command = sanitize_command(raw)
        return command if command else "look"


def choose_fallback_action(
    proposed: str,
    candidate_commands: Sequence[str],
    history: Sequence[Dict[str, str]],
    current_obs: str,
) -> str:
    if not candidate_commands:
        return proposed or "look"

    if proposed not in candidate_commands:
        proposed = ""

    last_command = history[-1]["command"] if history else None
    last_obs = history[-1]["observation"] if history else None
    lowered_obs = current_obs.lower()

    low_value = {"look", "help", "inventory"}
    repeated_low_value = proposed in low_value and proposed == last_command
    repeated_stall = (
        last_command is not None
        and proposed == last_command
        and last_obs is not None
        and last_obs.strip() == current_obs.strip()
    )
    if not (repeated_low_value or repeated_stall):
        return proposed or candidate_commands[0]

    tried_locations = {
        item["command"][6:]
        for item in history
        if item["command"].startswith("go to ")
    }
    go_candidates = [
        cmd for cmd in candidate_commands
        if cmd.startswith("go to ") and cmd[6:] not in tried_locations
    ]
    if go_candidates:
        return go_candidates[0]

    non_idle = [cmd for cmd in candidate_commands if cmd not in low_value]
    if non_idle:
        return non_idle[0]
    return candidate_commands[0]


@dataclass
class EpisodeMetrics:
    game_file: str
    success: bool
    won: bool
    steps: int
    invalid_actions: int
    first_invalid_step: Optional[int]
    recovered_after_first_failure: bool
    trajectory: List[Dict[str, Any]]


def run_episode(
    policy: LLMPolicy,
    game_file: Path,
    condition: str,
    max_steps: int,
) -> EpisodeMetrics:
    base_env = make_base_env(game_file, max_steps)
    env = AugmentedAlfWorldEnv(base_env, verbose=False) if condition == "augmented" else base_env

    try:
        if condition == "augmented":
            obs, infos = env.reset()
        else:
            obs, infos = unwrap_reset(env.reset())

        task_obs = obs
        history: List[Dict[str, str]] = []
        trajectory: List[Dict[str, Any]] = []
        invalid_actions = 0
        first_invalid_step: Optional[int] = None
        won = False

        for step_idx in range(1, max_steps + 1):
            admissible = infos.get("admissible_commands", []) or []
            candidate_commands = generate_candidate_commands(task_obs, obs, admissible, history)
            proposed_action = policy.choose_action(task_obs, obs, candidate_commands, history, step_idx)
            action = choose_fallback_action(proposed_action, candidate_commands, history, obs)

            prev_aug_log_len = len(env.get_augmentation_log()) if condition == "augmented" else 0
            if condition == "augmented":
                next_obs, score, done, infos = env.step(action)
                aug_log = env.get_augmentation_log()
                invalid = False
                invalid_rule = None
                original_obs = None
                if len(aug_log) > prev_aug_log_len:
                    latest = aug_log[-1]
                    invalid = latest["original_obs"].strip() == "Nothing happens."
                    invalid_rule = latest["rule_applied"]
                    original_obs = latest["original_obs"]
            else:
                next_obs, score, done, infos = unwrap_step(env.step([action]))
                invalid = next_obs.strip() == "Nothing happens."
                invalid_rule = None
                original_obs = next_obs if invalid else None

            if invalid:
                invalid_actions += 1
                if first_invalid_step is None:
                    first_invalid_step = step_idx

            won = bool(infos.get("won", False))
            trajectory.append(
                {
                    "step": step_idx,
                    "action": action,
                    "observation": next_obs,
                    "invalid_action": invalid,
                    "invalid_rule": invalid_rule,
                    "original_obs": original_obs,
                    "score": score,
                    "done": done,
                    "won": won,
                }
            )
            history.append({"command": action, "observation": next_obs})
            obs = next_obs
            if done:
                break

        steps = len(trajectory)
        success = won
        recovered_after_first_failure = first_invalid_step is not None and success
        return EpisodeMetrics(
            game_file=str(game_file),
            success=success,
            won=won,
            steps=steps,
            invalid_actions=invalid_actions,
            first_invalid_step=first_invalid_step,
            recovered_after_first_failure=recovered_after_first_failure,
            trajectory=trajectory,
        )
    finally:
        env.close()


def summarize(episodes: Sequence[EpisodeMetrics]) -> Dict[str, Any]:
    n = len(episodes)
    completion_count = sum(1 for ep in episodes if ep.success)
    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_actions for ep in episodes)
    episodes_with_failure = [ep for ep in episodes if ep.first_invalid_step is not None]
    recovered = sum(1 for ep in episodes_with_failure if ep.recovered_after_first_failure)

    return {
        "episodes": n,
        "completion_rate": completion_count / n if n else 0.0,
        "avg_steps": total_steps / n if n else 0.0,
        "invalid_action_count": total_invalid,
        "avg_invalid_actions": total_invalid / n if n else 0.0,
        "episodes_with_first_failure": len(episodes_with_failure),
        "recovery_after_first_failure_rate": (
            recovered / len(episodes_with_failure) if episodes_with_failure else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-games", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    api_key = load_together_api_key()
    game_files = discover_game_files(max_games=args.max_games)
    policy = LLMPolicy(model=args.model, api_key=api_key, base_url=args.base_url)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "base_url": args.base_url,
        "max_steps": args.max_steps,
        "games": [str(p) for p in game_files],
        "conditions": {},
    }

    for condition in ("original", "augmented"):
        print(f"\n=== Running condition: {condition} ===")
        episodes: List[EpisodeMetrics] = []
        for idx, game_file in enumerate(game_files, start=1):
            print(f"[{condition}] {idx}/{len(game_files)} {game_file.parent.parent.name}")
            episode = run_episode(policy, game_file, condition, args.max_steps)
            episodes.append(episode)
            print(
                f"  success={episode.success} steps={episode.steps} "
                f"invalid={episode.invalid_actions} first_invalid={episode.first_invalid_step}"
            )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        results["conditions"][condition] = {
            "summary": summarize(episodes),
            "episodes": [ep.__dict__ for ep in episodes],
        }

    original = results["conditions"]["original"]["summary"]
    augmented = results["conditions"]["augmented"]["summary"]
    results["delta"] = {
        "completion_rate": augmented["completion_rate"] - original["completion_rate"],
        "avg_steps": augmented["avg_steps"] - original["avg_steps"],
        "invalid_action_count": augmented["invalid_action_count"] - original["invalid_action_count"],
        "avg_invalid_actions": augmented["avg_invalid_actions"] - original["avg_invalid_actions"],
        "recovery_after_first_failure_rate": (
            None
            if original["recovery_after_first_failure_rate"] is None
            or augmented["recovery_after_first_failure_rate"] is None
            else augmented["recovery_after_first_failure_rate"]
            - original["recovery_after_first_failure_rate"]
        ),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote results to {RESULTS_PATH}")
    print(json.dumps(results["conditions"], indent=2, ensure_ascii=False))
    print("\nDelta:")
    print(json.dumps(results["delta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
