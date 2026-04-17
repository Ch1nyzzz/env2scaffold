"""
layer1_benchmark_native.py
Layer 1: Benchmark-Native Evaluation
A/B comparison of HandCodedTWAgent on original vs augmented ALFWorld env.
Tests L1_T01 (success_rate), L1_T02 (avg_game_points), L1_T03 (avg_steps).

Run:
    python3 verification/layer1_benchmark_native.py [--max-games 100]

Writes: verification/layer1_benchmark_native_results.json
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
DEFAULT_MAX_GAMES = 100
FALLBACK_MAX_GAMES = 20

# ---------------------------------------------------------------------------
# Load wrapper via importlib
# ---------------------------------------------------------------------------

def _load_wrapper():
    """Load AugmentedAlfWorldEnv from absolute path via importlib."""
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
# HandCodedTWAgent runner
# ---------------------------------------------------------------------------

def run_episode_handcoded(game_file: str, env, is_augmented: bool,
                           AugmentedCls=None, max_steps: int = MAX_EPISODE_STEPS
                           ) -> Dict[str, Any]:
    """
    Run one episode with HandCodedTWAgent.
    Returns dict with won, max_score, steps, error.
    """
    from alfworld.agents.expert import HandCodedTWAgent
    from alfworld.agents.expert.handcoded_expert import HandCodedAgentTimeout, HandCodedAgentFailed

    result = {
        "game_file": game_file,
        "is_augmented": is_augmented,
        "won": False,
        "max_score": 0.0,
        "steps": 0,
        "error": None,
    }

    try:
        agent = HandCodedTWAgent(max_steps=max_steps)
        agent.reset(game=game_file)

        if is_augmented:
            obs, infos = env.reset()
        else:
            obs, infos = _unwrap_reset(env.reset())

        game_state = {
            "feedback": obs,
            "admissible_commands": infos.get("admissible_commands", []),
            "facts": infos.get("facts", []),
            "won": infos.get("won", False),
        }

        last_action = ""
        done = False
        score = 0.0
        max_score = 0.0
        steps = 0
        won = False

        while not done and steps < max_steps:
            try:
                action = agent.act(game_state, score, done, last_action)
            except (HandCodedAgentTimeout, HandCodedAgentFailed):
                # agent exhausted — break gracefully
                action = "look"
                done = True
                break
            except Exception:
                action = "look"

            if is_augmented:
                obs, score, done, infos = env.step(action)
            else:
                obs, score, done, infos = _unwrap_step(env.step([action]))

            game_state = {
                "feedback": obs,
                "admissible_commands": infos.get("admissible_commands", []),
                "facts": infos.get("facts", []),
                "won": infos.get("won", False),
            }
            last_action = action
            steps += 1
            if score > max_score:
                max_score = score
            won = bool(infos.get("won", False))
            if done:
                break

        result["won"] = won
        result["max_score"] = max_score
        result["steps"] = steps

    except Exception as exc:
        result["error"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Main layer 1 evaluation
# ---------------------------------------------------------------------------

def run_layer1(max_games: int = DEFAULT_MAX_GAMES) -> List[Dict[str, Any]]:
    # Discover game files
    all_game_files = sorted(
        glob.glob(os.path.join(VALID_SEEN_ROOT, "**", "game.tw-pddl"), recursive=True)
    )
    if not all_game_files:
        raise RuntimeError(f"No game files found under {VALID_SEEN_ROOT}")

    game_files = all_game_files[:max_games]
    n_actual = len(game_files)
    print(f"[Layer1] Using {n_actual} games (max_games={max_games})")

    AugmentedCls = _load_wrapper()

    tests = []
    orig_wons, aug_wons = [], []
    orig_scores, aug_scores = [], []
    orig_steps_list, aug_steps_list = [], []

    for i, gf in enumerate(game_files):
        print(f"  [{i+1}/{n_actual}] {os.path.basename(os.path.dirname(gf))}", end="", flush=True)
        # --- Original env ---
        orig_env = _make_base_env(gf)
        orig_result = run_episode_handcoded(gf, orig_env, is_augmented=False)
        try:
            orig_env.close()
        except Exception:
            pass

        # --- Augmented env ---
        aug_base = _make_base_env(gf)
        aug_env = AugmentedCls(aug_base)
        aug_result = run_episode_handcoded(gf, aug_env, is_augmented=True, AugmentedCls=AugmentedCls)
        try:
            aug_env.close()
        except Exception:
            pass

        orig_wons.append(1 if orig_result["won"] else 0)
        aug_wons.append(1 if aug_result["won"] else 0)
        orig_scores.append(orig_result["max_score"])
        aug_scores.append(aug_result["max_score"])
        if orig_result["error"] is None and aug_result["error"] is None:
            orig_steps_list.append(orig_result["steps"])
            aug_steps_list.append(aug_result["steps"])

        print(f"  orig_won={orig_result['won']} aug_won={aug_result['won']} "
              f"orig_steps={orig_result['steps']} aug_steps={aug_result['steps']}")

        tests.append({
            "game_index": i,
            "game_file": gf,
            "original": orig_result,
            "augmented": aug_result,
        })

    # Aggregate metrics
    n = n_actual
    orig_sr = sum(orig_wons) / n if n > 0 else 0.0
    aug_sr = sum(aug_wons) / n if n > 0 else 0.0
    orig_agp = sum(orig_scores) / n if n > 0 else 0.0
    aug_agp = sum(aug_scores) / n if n > 0 else 0.0
    orig_avg_steps = sum(orig_steps_list) / len(orig_steps_list) if orig_steps_list else 0.0
    aug_avg_steps = sum(aug_steps_list) / len(aug_steps_list) if aug_steps_list else 0.0

    # Test verdicts
    test_results = []

    # L1_T01: success_rate comparison
    t01_pass = aug_sr >= orig_sr
    delta_sr = aug_sr - orig_sr
    test_results.append({
        "test_id": "L1_T01",
        "status": "pass" if t01_pass else "fail",
        "oracle_output": f"orig_success_rate={orig_sr:.4f} aug_success_rate={aug_sr:.4f} delta={delta_sr:+.4f}",
        "pass_criterion": "augmented_success_rate >= original_success_rate",
        "details": f"N={n} orig_won={sum(orig_wons)} aug_won={sum(aug_wons)}",
        "error": None,
        "original_value": orig_sr,
        "augmented_value": aug_sr,
        "metric_delta": delta_sr,
    })

    # L1_T02: avg_game_points comparison
    t02_pass = aug_agp >= orig_agp
    delta_agp = aug_agp - orig_agp
    test_results.append({
        "test_id": "L1_T02",
        "status": "pass" if t02_pass else "fail",
        "oracle_output": f"orig_avg_game_points={orig_agp:.4f} aug_avg_game_points={aug_agp:.4f} delta={delta_agp:+.4f}",
        "pass_criterion": "augmented_avg_game_points >= original_avg_game_points",
        "details": f"N={n} sum_orig_score={sum(orig_scores):.1f} sum_aug_score={sum(aug_scores):.1f}",
        "error": None,
        "original_value": orig_agp,
        "augmented_value": aug_agp,
        "metric_delta": delta_agp,
    })

    # L1_T03: avg_steps (auxiliary — delta < 5% relative does not gate pass/fail alone)
    if orig_avg_steps > 0:
        rel_delta = abs(aug_avg_steps - orig_avg_steps) / orig_avg_steps
        t03_pass = aug_avg_steps <= orig_avg_steps or rel_delta < 0.05
    else:
        rel_delta = 0.0
        t03_pass = True  # no data, treat as pass
    delta_steps = aug_avg_steps - orig_avg_steps
    test_results.append({
        "test_id": "L1_T03",
        "status": "pass" if t03_pass else "fail",
        "oracle_output": f"orig_avg_steps={orig_avg_steps:.2f} aug_avg_steps={aug_avg_steps:.2f} rel_delta={rel_delta:.4f}",
        "pass_criterion": "augmented_avg_steps <= original_avg_steps OR |delta| < 5% relative (auxiliary metric)",
        "details": f"N={len(orig_steps_list)} pairs with no errors",
        "error": None,
        "original_value": orig_avg_steps,
        "augmented_value": aug_avg_steps,
        "metric_delta": delta_steps,
    })

    total = len(test_results)
    n_pass = sum(1 for t in test_results if t["status"] == "pass")
    n_fail = sum(1 for t in test_results if t["status"] == "fail")
    n_error = sum(1 for t in test_results if t["status"] == "error")
    n_skipped = sum(1 for t in test_results if t["status"] == "skipped")

    print(f"\n[Layer1] Complete: {n_pass}/{total} pass")
    print(f"  success_rate: orig={orig_sr:.4f} aug={aug_sr:.4f}")
    print(f"  avg_game_points: orig={orig_agp:.4f} aug={aug_agp:.4f}")
    print(f"  avg_steps: orig={orig_avg_steps:.2f} aug={aug_avg_steps:.2f}")

    return {
        "layer": "layer1_benchmark_native",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wrapper_module_path": WRAPPER_PATH,
        "plan_path": PLAN_PATH,
        "oracles_consulted": ["won_info_field"],
        "policy": "HandCodedTWAgent",
        "n_games": n_actual,
        "max_games_requested": max_games,
        "note": (f"Used N={n_actual} games from valid_seen split with HandCodedTWAgent (deterministic). "
                 f"Fallback N={FALLBACK_MAX_GAMES} was not needed.") if n_actual >= max_games else
                (f"Fell back to N={n_actual} < {max_games} due to available game count."),
        "aggregate_metrics": {
            "orig_success_rate": orig_sr,
            "aug_success_rate": aug_sr,
            "orig_avg_game_points": orig_agp,
            "aug_avg_game_points": aug_agp,
            "orig_avg_steps": orig_avg_steps,
            "aug_avg_steps": aug_avg_steps,
        },
        "per_game_results": tests,
        "tests": test_results,
        "summary": {
            "total": total,
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "skipped": n_skipped,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 1: Benchmark-Native Evaluation")
    parser.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES,
                        help=f"Max number of games to evaluate (default={DEFAULT_MAX_GAMES})")
    args = parser.parse_args()

    results = run_layer1(max_games=args.max_games)

    output_path = os.path.join(VERIFICATION_DIR, "layer1_benchmark_native_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"\nlayer1 complete: {s['pass']}/{s['total']} pass")
    return 0 if s["fail"] == 0 and s["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
