"""
verify_runner.py — Verification suite for AugmentedAlfWorldEnv.

Three test groups:
  1. A/B Comparison  : random heuristic agent on ORIGINAL vs AUGMENTED env
  2. Error Recovery  : deliberately trigger each rule, check recovery hint validity
  3. No-Regression   : 5 full episodes on augmented env to confirm reward/done/commands unchanged

Run from repo root:
    python3 verification/verify_runner.py

Writes results to verification/verify_results.json.
"""

import sys
import os
import json
import random
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "analysis")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, REPO_ROOT)

import textworld.gym
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from augmented_env import AugmentedAlfWorldEnv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Game file registry (one per task type where available)
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen"

GAME_FILES = {
    "pick_and_place_simple": os.path.join(
        DATA_ROOT,
        "pick_and_place_simple-Book-None-SideTable-329",
        "trial_T20190908_050633_745514",
        "game.tw-pddl",
    ),
    "pick_heat_then_place": os.path.join(
        DATA_ROOT,
        "pick_heat_then_place_in_recep-Apple-None-DiningTable-26",
        "trial_T20190907_060234_011675",
        "game.tw-pddl",
    ),
    "look_at_obj_in_light": os.path.join(
        DATA_ROOT,
        "look_at_obj_in_light-AlarmClock-None-DeskLamp-323",
        "trial_T20190909_044715_250790",
        "game.tw-pddl",
    ),
    "pick_cool_then_place": os.path.join(
        DATA_ROOT,
        "pick_cool_then_place_in_recep-Apple-None-CounterTop-14",
        "trial_T20190909_044933_815840",
        "game.tw-pddl",
    ),
    "pick_clean_then_place": os.path.join(
        DATA_ROOT,
        "pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8",
        "trial_T20190909_105559_983897",
        "game.tw-pddl",
    ),
    "pick_two_obj_and_place": os.path.join(
        DATA_ROOT,
        "pick_two_obj_and_place-AlarmClock-None-Dresser-305",
        "trial_T20190907_165826_194855",
        "game.tw-pddl",
    ),
}

MAX_EPISODE_STEPS = 50
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Environment factory helpers
# ---------------------------------------------------------------------------

def _make_base_env(game_file: str):
    """Create the raw TextWorld/ALFWorld gym env (no augmentation)."""
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=MAX_EPISODE_STEPS,
        wrappers=[AlfredDemangler, AlfredInfos],
    )
    return textworld.gym.make(env_id)


def make_original_env(game_file: str):
    """Return the raw (un-augmented) gym env."""
    return _make_base_env(game_file)


def make_augmented_env(game_file: str, verbose: bool = False) -> AugmentedAlfWorldEnv:
    """Return the AugmentedAlfWorldEnv wrapper."""
    base = _make_base_env(game_file)
    return AugmentedAlfWorldEnv(base, verbose=verbose)


# ---------------------------------------------------------------------------
# Batch-unwrap helpers (original env returns batch lists)
# ---------------------------------------------------------------------------

def _unwrap_reset(result):
    obs, infos = result
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    if isinstance(infos, dict):
        infos = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                 for k, v in infos.items()}
    return obs, infos


def _unwrap_step(result):
    obs_raw, scores_raw, dones_raw, infos_raw = result
    obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
    score = scores_raw[0] if isinstance(scores_raw, (list, tuple)) else scores_raw
    done = dones_raw[0] if isinstance(dones_raw, (list, tuple)) else dones_raw
    infos = infos_raw
    if isinstance(infos, dict):
        infos = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                 for k, v in infos.items()}
    return obs, score, done, infos


# ---------------------------------------------------------------------------
# Random heuristic agent
# ---------------------------------------------------------------------------

def random_agent_step(admissible_commands: List[str], rng: random.Random) -> str:
    """Choose a random command from admissible_commands."""
    if not admissible_commands:
        return "look"
    return rng.choice(admissible_commands)


# ---------------------------------------------------------------------------
# 1. A/B Comparison Test
# ---------------------------------------------------------------------------

def run_ab_comparison(n_steps: int = MAX_EPISODE_STEPS) -> Dict[str, Any]:
    """
    For each task type, run a random agent on both original and augmented env.
    Compare: how often does "Nothing happens." appear vs augmented feedback.
    """
    print("\n" + "=" * 60)
    print("A/B COMPARISON TEST")
    print("=" * 60)

    results = {}

    for task_type, game_file in GAME_FILES.items():
        print(f"\n  Task type: {task_type}")
        rng = random.Random(RANDOM_SEED)

        # ---- Original env ----
        orig_env = make_original_env(game_file)
        orig_obs, orig_infos = _unwrap_reset(orig_env.reset())
        nothing_happens_count_orig = 0
        orig_steps = 0
        orig_done = False

        for _ in range(n_steps):
            admissible = orig_infos.get("admissible_commands", []) or []
            cmd = random_agent_step(admissible, rng)
            obs, score, done, orig_infos = _unwrap_step(orig_env.step([cmd]))
            orig_steps += 1
            if obs.strip() == "Nothing happens.":
                nothing_happens_count_orig += 1
            if done:
                orig_done = True
                break
        orig_env.close()

        # ---- Augmented env ----
        rng = random.Random(RANDOM_SEED)  # same seed → same random choices
        aug_env = make_augmented_env(game_file)
        aug_obs, aug_infos = aug_env.reset()
        aug_steps = 0
        aug_done = False
        augmentations_triggered = 0
        unique_rules = set()

        for _ in range(n_steps):
            admissible = aug_infos.get("admissible_commands", []) or []
            cmd = random_agent_step(admissible, rng)
            obs, score, done, aug_infos = aug_env.step(cmd)
            aug_steps += 1
            if done:
                aug_done = True
                break

        aug_log = aug_env.get_augmentation_log()
        augmentations_triggered = len(aug_log)
        unique_rules = set(e["rule_applied"] for e in aug_log)
        aug_env.close()

        results[task_type] = {
            "game_file": os.path.basename(os.path.dirname(game_file)),
            "steps_original": orig_steps,
            "steps_augmented": aug_steps,
            "nothing_happens_original": nothing_happens_count_orig,
            "augmentations_triggered": augmentations_triggered,
            "unique_rules_triggered": sorted(unique_rules),
            "aug_log_sample": aug_log[:3],  # first 3 entries as sample
        }

        print(f"    Original  — steps: {orig_steps}, 'Nothing happens.' count: {nothing_happens_count_orig}")
        print(f"    Augmented — steps: {aug_steps}, augmentations: {augmentations_triggered}, "
              f"unique rules: {sorted(unique_rules)}")

    return results


# ---------------------------------------------------------------------------
# 2. Error Recovery Test
# ---------------------------------------------------------------------------

# For each rule, define: game_file, setup_commands, trigger_command,
# and a list of keywords that should appear in admissible_commands to
# confirm the recovery hint is valid.
RECOVERY_TEST_CASES = [
    {
        "rule_id": "R01",
        "name": "put_while_hands_empty",
        "game_file": GAME_FILES["pick_and_place_simple"],
        # Navigate to a location first so "take" commands are admissible
        "setup_commands": ["go to bed 1"],
        "trigger_command": "move book 1 to sidetable 1",
        # Recovery: need to pick up something — so admissible should have "take ..."
        "recovery_keyword_in_admissible": "take",
        "expected_obs_contains": "not holding",
    },
    {
        "rule_id": "R02",
        "name": "take_from_closed_container",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": ["go to drawer 1"],
        "trigger_command": "take cellphone 1 from drawer 1",
        # Recovery: need to open drawer → admissible should have "open drawer 1"
        "recovery_keyword_in_admissible": "open drawer",
        "expected_obs_contains": "closed",
    },
    {
        "rule_id": "R03",
        "name": "take_object_not_at_location",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": ["go to sidetable 1"],
        "trigger_command": "take pillow 1 from sidetable 1",
        # Recovery: explore other locations → "go to" should be in admissible
        "recovery_keyword_in_admissible": "go to",
        "expected_obs_contains": "not there",
    },
    {
        "rule_id": "R04",
        "name": "open_already_open",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": ["go to drawer 1", "open drawer 1"],
        "trigger_command": "open drawer 1",
        # Recovery: container is open, can take from it or close it
        "recovery_keyword_in_admissible": "take",
        "expected_obs_contains": "already open",
    },
    {
        "rule_id": "R05",
        "name": "close_already_closed",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": ["go to drawer 1"],
        "trigger_command": "close drawer 1",
        # Recovery: container is closed, open it first
        "recovery_keyword_in_admissible": "open drawer",
        "expected_obs_contains": "already closed",
    },
    {
        "rule_id": "R06",
        "name": "heat_without_holding",
        "game_file": GAME_FILES["pick_heat_then_place"],
        "setup_commands": ["go to microwave 1"],
        "trigger_command": "heat apple 1 with microwave 1",
        # Recovery: pick up an object first
        "recovery_keyword_in_admissible": "go to",  # need to navigate to find object
        "expected_obs_contains": "not holding",
    },
    {
        "rule_id": "R07",
        "name": "pick_up_second_object",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": ["go to bed 1", "take book 1 from bed 1"],
        "trigger_command": "take book 2 from bed 1",
        # Recovery: put down current object → "move ... to" should be in admissible
        "recovery_keyword_in_admissible": "move",
        "expected_obs_contains": "already holding",
    },
    {
        "rule_id": "R08",
        "name": "use_without_holding",
        "game_file": GAME_FILES["look_at_obj_in_light"],
        "setup_commands": ["go to desklamp 1"],
        "trigger_command": "use desklamp 1",
        # Recovery: pick up an object first → "go to" or "take"
        "recovery_keyword_in_admissible": "go to",
        "expected_obs_contains": "not holding",
    },
    {
        "rule_id": "R09",
        "name": "invalid_command",
        "game_file": GAME_FILES["pick_and_place_simple"],
        "setup_commands": [],
        "trigger_command": "fly to mars",
        # Recovery: valid commands in admissible → "go to" should exist
        "recovery_keyword_in_admissible": "go to",
        "expected_obs_contains": "not recognized",
    },
]


def run_error_recovery_test() -> Dict[str, Any]:
    """
    For each rule, set up the triggering condition, receive augmented feedback,
    then check if admissible_commands supports the recovery hint.
    """
    print("\n" + "=" * 60)
    print("ERROR RECOVERY TEST")
    print("=" * 60)

    results = []

    for tc in RECOVERY_TEST_CASES:
        rule_id = tc["rule_id"]
        print(f"\n  Rule {rule_id}: {tc['name']}")

        result = {
            "rule_id": rule_id,
            "name": tc["name"],
            "trigger_command": tc["trigger_command"],
            "expected_obs_contains": tc["expected_obs_contains"],
            "actual_obs": "",
            "feedback_triggered": False,
            "recovery_keyword": tc["recovery_keyword_in_admissible"],
            "recovery_action_in_admissible": False,
            "recovery_hint_valid": False,
            "error": None,
        }

        try:
            env = make_augmented_env(tc["game_file"])
            obs, infos = env.reset()

            for cmd in tc["setup_commands"]:
                obs, score, done, infos = env.step(cmd)

            # Execute trigger command
            obs, score, done, infos = env.step(tc["trigger_command"])
            result["actual_obs"] = obs

            # Check feedback triggered
            result["feedback_triggered"] = (
                tc["expected_obs_contains"].lower() in obs.lower()
            )

            # Check admissible_commands contains recovery action keyword
            admissible = infos.get("admissible_commands", []) or []
            keyword = tc["recovery_keyword_in_admissible"].lower()
            recovery_matches = [c for c in admissible if keyword in c.lower()]
            result["recovery_action_in_admissible"] = len(recovery_matches) > 0
            result["recovery_examples"] = recovery_matches[:3]
            result["recovery_hint_valid"] = (
                result["feedback_triggered"] and result["recovery_action_in_admissible"]
            )

            env.close()

            status = "PASS" if result["recovery_hint_valid"] else "FAIL"
            print(f"    {status}: feedback_triggered={result['feedback_triggered']}, "
                  f"recovery_in_admissible={result['recovery_action_in_admissible']}")
            if result["recovery_examples"]:
                print(f"    Recovery actions: {result['recovery_examples'][:2]}")

        except Exception as exc:
            result["error"] = str(exc)
            print(f"    ERROR: {exc}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 3. No-Regression Test
# ---------------------------------------------------------------------------

def run_no_regression_test(n_episodes: int = 5) -> Dict[str, Any]:
    """
    Run N episodes on BOTH original and augmented env with the same random seed.
    Compare: reward, done signals, and admissible_commands must be identical.
    Only observations may differ.
    """
    print("\n" + "=" * 60)
    print("NO-REGRESSION TEST")
    print("=" * 60)

    # Use a single game for all regression episodes
    game_file = GAME_FILES["pick_and_place_simple"]

    reward_unchanged_list = []
    done_unchanged_list = []
    admissible_unchanged_list = []
    episodes_completed = 0
    mismatches = []

    for episode_idx in range(n_episodes):
        rng = random.Random(RANDOM_SEED + episode_idx)

        # Run original env
        orig_env = make_original_env(game_file)
        _, orig_infos = _unwrap_reset(orig_env.reset())
        orig_trajectory = []  # list of (cmd, score, done, admissible)

        for _ in range(MAX_EPISODE_STEPS):
            admissible = orig_infos.get("admissible_commands", []) or []
            cmd = random_agent_step(admissible, rng)
            obs, score, done, orig_infos = _unwrap_step(orig_env.step([cmd]))
            admissible_after = orig_infos.get("admissible_commands", []) or []
            orig_trajectory.append({
                "cmd": cmd,
                "score": score,
                "done": done,
                "admissible": sorted(admissible_after),
            })
            if done:
                break
        orig_env.close()

        # Run augmented env with same seed
        rng = random.Random(RANDOM_SEED + episode_idx)
        aug_env = make_augmented_env(game_file)
        _, aug_infos = aug_env.reset()
        aug_trajectory = []

        for _ in range(MAX_EPISODE_STEPS):
            admissible = aug_infos.get("admissible_commands", []) or []
            cmd = random_agent_step(admissible, rng)
            obs, score, done, aug_infos = aug_env.step(cmd)
            admissible_after = aug_infos.get("admissible_commands", []) or []
            aug_trajectory.append({
                "cmd": cmd,
                "score": score,
                "done": done,
                "admissible": sorted(admissible_after),
            })
            if done:
                break
        aug_env.close()

        # Compare trajectories step by step
        min_len = min(len(orig_trajectory), len(aug_trajectory))
        episode_reward_ok = True
        episode_done_ok = True
        episode_admissible_ok = True

        for step_i in range(min_len):
            o = orig_trajectory[step_i]
            a = aug_trajectory[step_i]

            if o["score"] != a["score"]:
                episode_reward_ok = False
                mismatches.append(
                    f"Ep{episode_idx} step{step_i}: reward mismatch "
                    f"orig={o['score']} aug={a['score']}"
                )

            if o["done"] != a["done"]:
                episode_done_ok = False
                mismatches.append(
                    f"Ep{episode_idx} step{step_i}: done mismatch "
                    f"orig={o['done']} aug={a['done']}"
                )

            if o["admissible"] != a["admissible"]:
                episode_admissible_ok = False
                mismatches.append(
                    f"Ep{episode_idx} step{step_i}: admissible mismatch at step {step_i}"
                )

        # Also check trajectory lengths (should be equal since same commands)
        if len(orig_trajectory) != len(aug_trajectory):
            mismatches.append(
                f"Ep{episode_idx}: trajectory length mismatch "
                f"orig={len(orig_trajectory)} aug={len(aug_trajectory)}"
            )

        reward_unchanged_list.append(episode_reward_ok)
        done_unchanged_list.append(episode_done_ok)
        admissible_unchanged_list.append(episode_admissible_ok)
        episodes_completed += 1

        status = "PASS" if (episode_reward_ok and episode_done_ok and episode_admissible_ok) else "FAIL"
        print(f"  Episode {episode_idx+1}: {status} — "
              f"reward_ok={episode_reward_ok}, done_ok={episode_done_ok}, "
              f"admissible_ok={episode_admissible_ok}")

    all_reward_ok = all(reward_unchanged_list)
    all_done_ok = all(done_unchanged_list)
    all_admissible_ok = all(admissible_unchanged_list)

    print(f"\n  Summary: reward_unchanged={all_reward_ok}, "
          f"done_unchanged={all_done_ok}, admissible_unchanged={all_admissible_ok}")
    if mismatches:
        print(f"  Mismatches found ({len(mismatches)}):")
        for m in mismatches[:5]:
            print(f"    {m}")

    return {
        "reward_unchanged": all_reward_ok,
        "done_unchanged": all_done_ok,
        "admissible_unchanged": all_admissible_ok,
        "episodes_completed": episodes_completed,
        "episodes_requested": n_episodes,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AugmentedAlfWorldEnv — Full Verification Suite")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    ab_results = run_ab_comparison()
    recovery_results = run_error_recovery_test()
    regression_results = run_no_regression_test(n_episodes=5)

    # Aggregate summary
    recovery_pass = sum(1 for r in recovery_results if r.get("recovery_hint_valid"))
    recovery_total = len(recovery_results)

    total_augmentations = sum(
        v["augmentations_triggered"] for v in ab_results.values()
    )
    all_unique_rules = set()
    for v in ab_results.values():
        all_unique_rules.update(v["unique_rules_triggered"])

    summary = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "ab_comparison": ab_results,
        "error_recovery": recovery_results,
        "no_regression": regression_results,
        "summary": {
            "total_augmentations_in_ab": total_augmentations,
            "unique_rules_in_ab": sorted(all_unique_rules),
            "recovery_tests_passed": recovery_pass,
            "recovery_tests_total": recovery_total,
            "reward_unchanged": regression_results["reward_unchanged"],
            "done_unchanged": regression_results["done_unchanged"],
            "admissible_unchanged": regression_results["admissible_unchanged"],
            "regression_episodes_completed": regression_results["episodes_completed"],
        },
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print(f"  A/B: {total_augmentations} augmentations triggered across {len(ab_results)} games")
    print(f"  A/B: Unique rules triggered: {sorted(all_unique_rules)}")
    print(f"  Recovery: {recovery_pass}/{recovery_total} hints point to valid actions")
    print(f"  Regression: reward_ok={regression_results['reward_unchanged']}, "
          f"done_ok={regression_results['done_unchanged']}, "
          f"admissible_ok={regression_results['admissible_unchanged']}")
    print(f"  Results saved to: {output_path}")
    print("=" * 60)

    # Return exit code
    all_ok = (
        regression_results["reward_unchanged"]
        and regression_results["done_unchanged"]
        and regression_results["admissible_unchanged"]
        and recovery_pass == recovery_total
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
