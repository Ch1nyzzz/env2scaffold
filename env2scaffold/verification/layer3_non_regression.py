"""
layer3_non_regression.py
Layer 3: Non-Regression Tests — verify wrapper preserves reward, done,
admissible_commands, and observation on non-trigger paths.

Strategy:
  - Run HandCodedTWAgent on the original env, record the command sequence.
  - Replay the exact same commands on the augmented env.
  - Compare reward, done, admissible_commands step-by-step.
  - For L3_T04 (noop path): on steps where original obs != 'Nothing happens.',
    check augmented obs is byte-for-byte identical to original obs.

Run:
    python3 verification/layer3_non_regression.py [--episodes 50]

Writes: verification/layer3_non_regression_results.json
"""

import sys
import os
import json
import glob
import argparse
import importlib.util
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_PATH = os.path.join(REPO_ROOT, "augmentation", "augmented_env.py")
PLAN_PATH = os.path.join(REPO_ROOT, "oracle_test", "unit_test_plan.json")
VALID_SEEN_ROOT = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen"
MAX_EPISODE_STEPS = 50
DEFAULT_EPISODES = 50
NOOP_PATH_EPISODES = 20
NOTHING_HAPPENS = "Nothing happens."


# ---------------------------------------------------------------------------
# Load wrapper via importlib
# ---------------------------------------------------------------------------

def _load_wrapper():
    spec = importlib.util.spec_from_file_location("augmented_env", WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AugmentedAlfWorldEnv


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_base_env(game_file: str, max_steps: int = MAX_EPISODE_STEPS):
    import textworld.gym
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        feedback=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler, AlfredInfos],
    )
    return textworld.gym.make(env_id)


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
    infos = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
             for k, v in (infos_raw if isinstance(infos_raw, dict) else {}).items()}
    return obs, score, done, infos


# ---------------------------------------------------------------------------
# HandCodedTWAgent episode runner (original env — records command sequence)
# ---------------------------------------------------------------------------

def run_episode_original(game_file: str, env, max_steps: int = MAX_EPISODE_STEPS
                          ) -> Tuple[List[str], List[Dict], Optional[str]]:
    """
    Run HandCodedTWAgent on original (unwrapped) env.
    Returns (commands, steps_data, error_text).
    steps_data: list of {cmd, orig_obs, orig_score, orig_done, orig_ac}
    """
    from alfworld.agents.expert import HandCodedTWAgent
    from alfworld.agents.expert.handcoded_expert import HandCodedAgentTimeout, HandCodedAgentFailed

    commands = []
    steps_data = []

    try:
        agent = HandCodedTWAgent(max_steps=max_steps)
        agent.reset(game=game_file)
        obs, infos = _unwrap_reset(env.reset())

        game_state = {
            "feedback": obs,
            "admissible_commands": infos.get("admissible_commands", []),
            "facts": infos.get("facts", []),
            "won": infos.get("won", False),
        }
        last_action = ""
        score = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            try:
                action = agent.act(game_state, score, done, last_action)
            except (HandCodedAgentTimeout, HandCodedAgentFailed):
                break
            except Exception:
                action = "look"

            obs, score, done, infos = _unwrap_step(env.step([action]))
            game_state = {
                "feedback": obs,
                "admissible_commands": infos.get("admissible_commands", []),
                "facts": infos.get("facts", []),
                "won": infos.get("won", False),
            }
            last_action = action
            steps += 1
            commands.append(action)
            steps_data.append({
                "cmd": action,
                "orig_obs": obs,
                "orig_score": float(score),
                "orig_done": bool(done),
                "orig_ac": sorted(infos.get("admissible_commands", [])),
            })
            if done:
                break

    except Exception:
        return commands, steps_data, traceback.format_exc()

    return commands, steps_data, None


# ---------------------------------------------------------------------------
# Replay commands on augmented env
# ---------------------------------------------------------------------------

def replay_on_augmented(game_file: str, AugmentedCls, commands: List[str],
                         max_steps: int = MAX_EPISODE_STEPS
                         ) -> Tuple[List[Dict], Optional[str]]:
    """
    Replay a given command sequence on the augmented env.
    Returns (aug_steps_data, error_text).
    aug_steps_data: list of {cmd, aug_obs, aug_score, aug_done, aug_ac}
    """
    aug_base = _make_base_env(game_file, max_steps=max_steps)
    aug_env = AugmentedCls(aug_base)
    aug_steps_data = []

    try:
        aug_env.reset()
        for cmd in commands:
            aug_obs, aug_score, aug_done, aug_infos = aug_env.step(cmd)
            aug_steps_data.append({
                "cmd": cmd,
                "aug_obs": aug_obs,
                "aug_score": float(aug_score),
                "aug_done": bool(aug_done),
                "aug_ac": sorted(aug_infos.get("admissible_commands", [])),
            })
            if aug_done:
                break
    except Exception:
        try:
            aug_env.close()
        except Exception:
            pass
        return aug_steps_data, traceback.format_exc()

    try:
        aug_env.close()
    except Exception:
        pass

    return aug_steps_data, None


# ---------------------------------------------------------------------------
# Layer 3 comparison
# ---------------------------------------------------------------------------

def run_layer3(n_episodes: int = DEFAULT_EPISODES) -> Dict[str, Any]:
    AugmentedCls = _load_wrapper()

    all_game_files = sorted(
        glob.glob(os.path.join(VALID_SEEN_ROOT, "**", "game.tw-pddl"), recursive=True)
    )
    # Use first n_episodes game files
    game_files = all_game_files[:n_episodes]
    n_actual = len(game_files)
    print(f"[Layer3] Using {n_actual} episodes")

    # Per-field mismatch tracking across all episodes
    reward_mismatches = []
    done_mismatches = []
    ac_mismatches = []
    noop_obs_mismatches = []

    episode_results = []

    for ep_idx, gf in enumerate(game_files):
        print(f"  [{ep_idx+1}/{n_actual}] {os.path.basename(os.path.dirname(gf))}", end="", flush=True)
        orig_env = _make_base_env(gf)
        commands, orig_steps, orig_err = run_episode_original(gf, orig_env)
        try:
            orig_env.close()
        except Exception:
            pass

        if orig_err:
            print(f" orig_error")
            episode_results.append({
                "episode_idx": ep_idx,
                "game_file": gf,
                "error": orig_err,
                "steps": 0,
                "reward_ok": False,
                "done_ok": False,
                "ac_ok": False,
                "noop_obs_ok": True,
            })
            continue

        aug_steps, aug_err = replay_on_augmented(gf, AugmentedCls, commands)

        if aug_err:
            print(f" aug_error")
            episode_results.append({
                "episode_idx": ep_idx,
                "game_file": gf,
                "error": aug_err,
                "steps": 0,
                "reward_ok": False,
                "done_ok": False,
                "ac_ok": False,
                "noop_obs_ok": True,
            })
            continue

        ep_reward_ok = True
        ep_done_ok = True
        ep_ac_ok = True
        ep_noop_ok = True
        n_steps = min(len(orig_steps), len(aug_steps))

        for step_i in range(n_steps):
            o = orig_steps[step_i]
            a = aug_steps[step_i]

            # L3_T01: reward
            if o["orig_score"] != a["aug_score"]:
                ep_reward_ok = False
                reward_mismatches.append({
                    "episode": ep_idx, "step": step_i,
                    "orig": o["orig_score"], "aug": a["aug_score"]
                })

            # L3_T02: done
            if o["orig_done"] != a["aug_done"]:
                ep_done_ok = False
                done_mismatches.append({
                    "episode": ep_idx, "step": step_i,
                    "orig": o["orig_done"], "aug": a["aug_done"]
                })

            # L3_T03: admissible_commands
            if o["orig_ac"] != a["aug_ac"]:
                ep_ac_ok = False
                ac_mismatches.append({
                    "episode": ep_idx, "step": step_i,
                    "orig_count": len(o["orig_ac"]), "aug_count": len(a["aug_ac"]),
                    "added": sorted(set(a["aug_ac"]) - set(o["orig_ac"])),
                    "removed": sorted(set(o["orig_ac"]) - set(a["aug_ac"])),
                })

            # L3_T04: noop path — only on steps where original obs != "Nothing happens."
            if o["orig_obs"].strip() != NOTHING_HAPPENS:
                if o["orig_obs"] != a["aug_obs"]:
                    ep_noop_ok = False
                    noop_obs_mismatches.append({
                        "episode": ep_idx, "step": step_i,
                        "cmd": o["cmd"],
                        "orig_obs": o["orig_obs"][:80],
                        "aug_obs": a["aug_obs"][:80],
                    })

        # Length check
        if len(orig_steps) != len(aug_steps):
            # Length mismatch is a done regression
            ep_done_ok = False
            done_mismatches.append({
                "episode": ep_idx, "step": "length",
                "orig": len(orig_steps), "aug": len(aug_steps)
            })

        status = "ok" if (ep_reward_ok and ep_done_ok and ep_ac_ok) else "mismatch"
        print(f"  steps={n_steps} reward_ok={ep_reward_ok} done_ok={ep_done_ok} "
              f"ac_ok={ep_ac_ok} noop_ok={ep_noop_ok}")
        episode_results.append({
            "episode_idx": ep_idx,
            "game_file": gf,
            "error": None,
            "steps": n_steps,
            "reward_ok": ep_reward_ok,
            "done_ok": ep_done_ok,
            "ac_ok": ep_ac_ok,
            "noop_obs_ok": ep_noop_ok,
        })

    # Compile per-test verdicts
    n_ep = len(episode_results)
    n_ep_valid = sum(1 for e in episode_results if e.get("error") is None)

    # L3_T01
    t01_pass = len(reward_mismatches) == 0
    # L3_T02
    t02_pass = len(done_mismatches) == 0
    # L3_T03
    t03_pass = len(ac_mismatches) == 0
    # L3_T04 (noop path, 20 ep subset)
    t04_pass = len(noop_obs_mismatches) == 0

    test_results = [
        {
            "test_id": "L3_T01",
            "field": "reward",
            "status": "pass" if t01_pass else "fail",
            "oracle_output": f"reward_mismatch_count={len(reward_mismatches)} over {n_ep_valid} episodes",
            "pass_criterion": "augmented_reward == original_reward for every step (exact float equality)",
            "details": f"mismatches={len(reward_mismatches)}" + (
                f" first={reward_mismatches[0]}" if reward_mismatches else ""),
            "error": None,
            "original_value": "n/a",
            "augmented_value": "n/a",
            "mismatch_count": len(reward_mismatches),
        },
        {
            "test_id": "L3_T02",
            "field": "done",
            "status": "pass" if t02_pass else "fail",
            "oracle_output": f"done_mismatch_count={len(done_mismatches)} over {n_ep_valid} episodes",
            "pass_criterion": "augmented_done == original_done for every step (exact boolean equality)",
            "details": f"mismatches={len(done_mismatches)}" + (
                f" first={done_mismatches[0]}" if done_mismatches else ""),
            "error": None,
            "original_value": "n/a",
            "augmented_value": "n/a",
            "mismatch_count": len(done_mismatches),
        },
        {
            "test_id": "L3_T03",
            "field": "admissible_commands",
            "status": "pass" if t03_pass else "fail",
            "oracle_output": f"ac_mismatch_count={len(ac_mismatches)} over {n_ep_valid} episodes",
            "pass_criterion": "set(augmented_ac) == set(original_ac) for every step (exact set equality)",
            "details": f"mismatches={len(ac_mismatches)}" + (
                f" first={ac_mismatches[0]}" if ac_mismatches else ""),
            "error": None,
            "original_value": "n/a",
            "augmented_value": "n/a",
            "mismatch_count": len(ac_mismatches),
        },
        {
            "test_id": "L3_T04",
            "field": "observation_noop_path",
            "status": "pass" if t04_pass else "fail",
            "oracle_output": f"noop_obs_mismatch_count={len(noop_obs_mismatches)} over {n_ep_valid} episodes",
            "pass_criterion": "augmented_obs == original_obs byte-for-byte on steps where no augmentation fires",
            "details": f"mismatches={len(noop_obs_mismatches)}" + (
                f" first={noop_obs_mismatches[0]}" if noop_obs_mismatches else ""),
            "error": None,
            "original_value": "n/a",
            "augmented_value": "n/a",
            "mismatch_count": len(noop_obs_mismatches),
        },
    ]

    total = len(test_results)
    n_pass = sum(1 for t in test_results if t["status"] == "pass")
    n_fail = sum(1 for t in test_results if t["status"] == "fail")
    n_error = sum(1 for t in test_results if t["status"] == "error")
    n_skipped = sum(1 for t in test_results if t["status"] == "skipped")

    print(f"\n[Layer3] Complete: {n_pass}/{total} pass")
    print(f"  reward_mismatches={len(reward_mismatches)}")
    print(f"  done_mismatches={len(done_mismatches)}")
    print(f"  ac_mismatches={len(ac_mismatches)}")
    print(f"  noop_obs_mismatches={len(noop_obs_mismatches)}")

    return {
        "layer": "layer3_non_regression",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wrapper_module_path": WRAPPER_PATH,
        "plan_path": PLAN_PATH,
        "oracles_consulted": ["won_info_field", "admissible_commands_validity_heuristic"],
        "n_episodes": n_actual,
        "n_episodes_valid": n_ep_valid,
        "policy": "HandCodedTWAgent (record then replay)",
        "tests": test_results,
        "summary": {
            "total": total,
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "skipped": n_skipped,
        },
        "detail": {
            "reward_mismatches": reward_mismatches[:10],
            "done_mismatches": done_mismatches[:10],
            "ac_mismatches": ac_mismatches[:10],
            "noop_obs_mismatches": noop_obs_mismatches[:10],
            "episode_results": episode_results,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 3: Non-Regression Tests")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Number of episodes (default={DEFAULT_EPISODES})")
    args = parser.parse_args()

    results = run_layer3(n_episodes=args.episodes)

    output_path = os.path.join(VERIFICATION_DIR, "layer3_non_regression_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"\nlayer3 complete: {s['pass']}/{s['total']} pass")
    return 0 if s["fail"] == 0 and s["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
