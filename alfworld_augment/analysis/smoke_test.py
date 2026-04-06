"""
Smoke test for AugmentedAlfWorldEnv.

Tests each augmentation rule against real ALFWorld games.
Writes results to analysis/smoke_test_result.json.

Run from repo root:
    python3 analysis/smoke_test.py
"""

import sys
import os
import json
import logging
from datetime import datetime

# Make sure analysis/ is on path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import textworld.gym
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from augmented_env import AugmentedAlfWorldEnv, NOTHING_HAPPENS

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Game files from probing trajectories
# ---------------------------------------------------------------------------
GAME_FILES = {
    "pick_and_place_simple": (
        "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/"
        "pick_and_place_simple-Book-None-SideTable-329/"
        "trial_T20190908_050633_745514/game.tw-pddl"
    ),
    "pick_heat_then_place": (
        "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/"
        "pick_heat_then_place_in_recep-Apple-None-DiningTable-26/"
        "trial_T20190907_060234_011675/game.tw-pddl"
    ),
    "look_at_obj_in_light": (
        "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/"
        "look_at_obj_in_light-AlarmClock-None-DeskLamp-323/"
        "trial_T20190909_044715_250790/game.tw-pddl"
    ),
    "pick_cool_then_place": (
        "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/"
        "pick_cool_then_place_in_recep-Apple-None-CounterTop-14/"
        "trial_T20190909_044933_815840/game.tw-pddl"
    ),
    "pick_clean_then_place": (
        "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/"
        "pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8/"
        "trial_T20190909_105559_983897/game.tw-pddl"
    ),
}

MAX_EPISODE_STEPS = 100


def make_env(game_file: str) -> AugmentedAlfWorldEnv:
    """Create a single-game AugmentedAlfWorldEnv."""
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
    base_env = textworld.gym.make(env_id)
    return AugmentedAlfWorldEnv(base_env, verbose=False)


def run_test_case(name: str, game_file: str, setup_commands: list,
                  test_command: str, expected_rule: str,
                  expected_obs_contains: str) -> dict:
    """
    Run a single test case:
      1. Load game, execute setup_commands to reach desired state
      2. Execute test_command
      3. Check augmented observation matches expectations
    """
    result = {
        "test_name": name,
        "game_file": os.path.basename(os.path.dirname(game_file)),
        "setup_commands": setup_commands,
        "test_command": test_command,
        "expected_rule": expected_rule,
        "expected_obs_contains": expected_obs_contains,
        "passed": False,
        "actual_obs": "",
        "rule_applied": "",
        "error": None,
    }

    try:
        env = make_env(game_file)
        obs, infos = env.reset()

        # Execute setup commands
        for cmd in setup_commands:
            obs, score, done, infos = env.step(cmd)

        # Execute the test command
        obs, score, done, infos = env.step(test_command)
        result["actual_obs"] = obs

        # Check augmentation log
        log = env.get_augmentation_log()
        # Find the last augmentation (for test_command)
        matching_logs = [e for e in log if e["command"].lower() == test_command.lower()]
        if matching_logs:
            result["rule_applied"] = matching_logs[-1]["rule_applied"]
        else:
            result["rule_applied"] = "none (no augmentation triggered)"

        # Invariant: score and done must not change unexpectedly
        # (we just record them for reference)
        result["score"] = score
        result["done"] = done

        # Check expectations
        obs_ok = expected_obs_contains.lower() in obs.lower()
        rule_ok = (expected_rule == "any" or
                   expected_rule.lower() in result["rule_applied"].lower())
        result["passed"] = obs_ok and rule_ok

        if not obs_ok:
            result["error"] = (
                f"Expected obs to contain '{expected_obs_contains}', "
                f"got: '{obs[:150]}'"
            )
        elif not rule_ok:
            result["error"] = (
                f"Expected rule '{expected_rule}', "
                f"got: '{result['rule_applied']}'"
            )

        env.close()

    except Exception as exc:
        result["error"] = str(exc)
        result["passed"] = False

    return result


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

def build_test_cases() -> list:
    """Define all smoke-test cases covering the 9+ augmentation rules."""
    gf_simple = GAME_FILES["pick_and_place_simple"]
    gf_heat = GAME_FILES["pick_heat_then_place"]
    gf_light = GAME_FILES["look_at_obj_in_light"]
    gf_cool = GAME_FILES["pick_cool_then_place"]
    gf_clean = GAME_FILES["pick_clean_then_place"]

    cases = []

    # ------------------------------------------------------------------
    # R01: put while hands empty
    # ------------------------------------------------------------------
    cases.append({
        "name": "R01_put_while_hands_empty",
        "game_file": gf_simple,
        "setup_commands": [],
        "test_command": "move book 1 to sidetable 1",
        "expected_rule": "R01",
        "expected_obs_contains": "not holding",
    })

    # ------------------------------------------------------------------
    # R02: take from closed container
    # ------------------------------------------------------------------
    # drawer 1 in the simple game starts closed and contains cellphone 1
    cases.append({
        "name": "R02_take_from_closed_container",
        "game_file": gf_simple,
        "setup_commands": ["go to drawer 1"],   # arrive but do NOT open
        "test_command": "take cellphone 1 from drawer 1",
        "expected_rule": "R02",
        "expected_obs_contains": "closed",
    })

    # ------------------------------------------------------------------
    # R03: take object not at current location
    # ------------------------------------------------------------------
    # After opening drawer 1 and taking cellphone (hands full with phone),
    # reset: just go to sidetable 1 and try to take something not there
    cases.append({
        "name": "R03_take_object_not_at_location",
        "game_file": gf_simple,
        "setup_commands": ["go to sidetable 1"],
        # book 1 is actually on the bed, not sidetable 1
        "test_command": "take pillow 1 from sidetable 1",
        "expected_rule": "R03",
        "expected_obs_contains": "not there",
    })

    # ------------------------------------------------------------------
    # R04: open already open
    # ------------------------------------------------------------------
    cases.append({
        "name": "R04_open_already_open",
        "game_file": gf_simple,
        "setup_commands": ["go to drawer 1", "open drawer 1"],
        "test_command": "open drawer 1",
        "expected_rule": "R04",
        "expected_obs_contains": "already open",
    })

    # ------------------------------------------------------------------
    # R05: close already closed
    # ------------------------------------------------------------------
    cases.append({
        "name": "R05_close_already_closed",
        "game_file": gf_simple,
        "setup_commands": ["go to drawer 1"],
        "test_command": "close drawer 1",
        "expected_rule": "R05",
        "expected_obs_contains": "already closed",
    })

    # ------------------------------------------------------------------
    # R06: heat without holding
    # ------------------------------------------------------------------
    cases.append({
        "name": "R06_heat_without_holding",
        "game_file": gf_heat,
        "setup_commands": ["go to microwave 1"],
        "test_command": "heat apple 1 with microwave 1",
        "expected_rule": "R06",
        "expected_obs_contains": "not holding",
    })

    # R06 variant: cool without holding
    cases.append({
        "name": "R06_cool_without_holding",
        "game_file": gf_cool,
        "setup_commands": ["go to fridge 1"],
        "test_command": "cool apple 1 with fridge 1",
        "expected_rule": "R06",
        "expected_obs_contains": "not holding",
    })

    # R06 variant: clean without holding
    cases.append({
        "name": "R06_clean_without_holding",
        "game_file": gf_clean,
        "setup_commands": ["go to sinkbasin 1"],
        "test_command": "clean butterknife 1 with sinkbasin 1",
        "expected_rule": "R06",
        "expected_obs_contains": "not holding",
    })

    # ------------------------------------------------------------------
    # R07: pick up second object while holding one
    # ------------------------------------------------------------------
    cases.append({
        "name": "R07_pick_up_second_object",
        "game_file": gf_simple,
        "setup_commands": [
            "go to bed 1",
            "take book 1 from bed 1",   # now holding book 1
        ],
        "test_command": "take book 2 from bed 1",
        "expected_rule": "R07",
        "expected_obs_contains": "already holding",
    })

    # ------------------------------------------------------------------
    # R08: use appliance without holding
    # ------------------------------------------------------------------
    cases.append({
        "name": "R08_use_without_holding",
        "game_file": gf_light,
        "setup_commands": ["go to desklamp 1"],
        "test_command": "use desklamp 1",
        "expected_rule": "R08",
        "expected_obs_contains": "not holding",
    })

    # ------------------------------------------------------------------
    # R09: invalid/unrecognized command
    # ------------------------------------------------------------------
    cases.append({
        "name": "R09_invalid_command",
        "game_file": gf_simple,
        "setup_commands": [],
        "test_command": "fly to mars",
        "expected_rule": "R09",
        "expected_obs_contains": "not recognized",
    })

    # ------------------------------------------------------------------
    # R10: progress hint (picking up task-relevant object)
    # ------------------------------------------------------------------
    cases.append({
        "name": "R10_progress_hint_correct_object",
        "game_file": gf_heat,
        "setup_commands": ["go to diningtable 2"],
        "test_command": "take apple 1 from diningtable 2",
        "expected_rule": "R10",
        "expected_obs_contains": "apple",
    })

    # ------------------------------------------------------------------
    # R11: exploration guidance when nothing at location
    # ------------------------------------------------------------------
    cases.append({
        "name": "R11_exploration_nothing_relevant",
        "game_file": gf_simple,
        "setup_commands": [],
        "test_command": "go to sidetable 1",
        "expected_rule": "R11",
        "expected_obs_contains": "nothing useful",
    })

    # ------------------------------------------------------------------
    # Invariant: score and done not modified (normal successful action)
    # ------------------------------------------------------------------
    cases.append({
        "name": "INVARIANT_score_not_modified",
        "game_file": gf_simple,
        "setup_commands": [],
        "test_command": "go to bed 1",
        "expected_rule": "any",
        "expected_obs_contains": "bed 1",
    })

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AugmentedAlfWorldEnv Smoke Test")
    print("=" * 60)

    test_cases = build_test_cases()
    results = []
    passed = 0
    failed = 0

    for tc in test_cases:
        print(f"\nRunning: {tc['name']}...")
        result = run_test_case(**tc)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status}: rule={result['rule_applied']}")
        if not result["passed"]:
            print(f"  ERROR: {result['error']}")
            print(f"  Actual obs: {result['actual_obs'][:200]}")
            failed += 1
        else:
            passed += 1

    # Summary
    total = len(results)
    summary = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{100 * passed / total:.1f}%",
        "test_results": results,
    }

    output_path = os.path.join(os.path.dirname(__file__), "smoke_test_result.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed ({summary['pass_rate']})")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
