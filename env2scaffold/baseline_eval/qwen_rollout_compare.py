#!/usr/bin/env python3
"""
Qwen rollout comparison — original ALFWorld env vs AugmentedAlfWorldEnv.

Connects to a vllm OpenAI-compatible endpoint (default: Qwen3.5-35B-A3B-FP8 at
localhost:8000). For each sampled game file, runs one episode on the original
env and one on the augmented env. Same game set, same LLM, same temperature —
the only change is the wrapper.

Full trajectories are saved so trace_evaluator.py can score them post-hoc.

Usage:
    python qwen_rollout_compare.py --n 64 --concurrency 64
    python qwen_rollout_compare.py --n 4 --concurrency 2 --phase original  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

# ALFWorld / TextWorld runtime
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

# Augmented wrapper via sys.path — matches the pattern in
# AWorld-RL/EnvTuning/env_tuning/interaction/alfworld_interaction.py
SCRIPT_DIR = Path(__file__).resolve().parent
AUG_DIR = SCRIPT_DIR.parent / "augmentation"
if str(AUG_DIR) not in sys.path:
    sys.path.insert(0, str(AUG_DIR))
from augmented_env import AugmentedAlfWorldEnv  # noqa: E402


# Global lock: TextWorld's PDDL grammar derivation uses tatsu parser which
# is not thread-safe. We serialize env.step / env.reset / env.close calls
# across all episodes; LLM calls remain fully concurrent.
ENV_LOCK = asyncio.Lock()


SYSTEM_PROMPT = """You are controlling an ALFWorld household agent. Solve the task one text action at a time.

HARD RULES:
- Every reply MUST be exactly one command string, copied verbatim from the "Admissible actions" list shown in the user message.
- Do NOT output any prefix, explanation, reasoning, quotes, or punctuation beyond the command itself.
- Do NOT invent commands that are not in the list.
- If the environment reports "Nothing happens." or equivalent, pick a different admissible command rather than repeating.
- Explore, pick up the target object(s), transform (clean/heat/cool) if required, then place at the target receptacle.
"""

DATA_ROOT_BASE = Path.home() / ".cache" / "alfworld" / "json_2.1.1"
DEFAULT_SPLITS = ["valid_seen", "valid_unseen", "train"]
DEFAULT_MODEL = "Qwen3.5-35B-A3B-FP8"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


# ─── Game selection ──────────────────────────────────────────────────────────
def sample_game_files(n: int, seed: int, splits: list[str]) -> list[Path]:
    all_games: list[Path] = []
    for split in splits:
        split_dir = DATA_ROOT_BASE / split
        if split_dir.is_dir():
            all_games.extend(split_dir.glob("*/trial_*/game.tw-pddl"))
    all_games = sorted(all_games)
    if not all_games:
        raise RuntimeError(f"No game files across splits {splits}")
    rng = random.Random(seed)
    rng.shuffle(all_games)
    if n > len(all_games):
        raise ValueError(f"Requested {n} games but only {len(all_games)} available across {splits}")
    return all_games[:n]


def task_type_of(game_file: Path) -> str:
    # folder name like 'pick_heat_then_place_in_recep-Apple-None-DiningTable-26'
    stem = game_file.parent.parent.name
    # task type is everything before first dash followed by a capital letter
    m = re.match(r"^([a-z_]+)-", stem)
    return m.group(1) if m else stem


# ─── Env factory ─────────────────────────────────────────────────────────────
def _make_base_env(game_file: Path, max_episode_steps: int):
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


def make_env(game_file: Path, use_augmented: bool, max_episode_steps: int):
    base = _make_base_env(game_file, max_episode_steps)
    return AugmentedAlfWorldEnv(base) if use_augmented else base


def _unbatch(obs, infos):
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    if isinstance(infos, dict):
        infos = {k: (v[0] if isinstance(v, list) else v) for k, v in infos.items()}
    return obs, infos


# ─── LLM interaction ─────────────────────────────────────────────────────────
def sanitize_action(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(Action\s*:\s*|>\s*)", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


async def llm_step(client: AsyncOpenAI, messages: list[dict], model: str,
                   temperature: float, max_tokens: int = 48) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content or ""
    return sanitize_action(raw)


def format_user_turn(obs: str, admissible: list[str]) -> str:
    """Build a user message that shows obs plus the admissible-command menu."""
    if not admissible:
        return obs
    menu = "\n".join(f"- {cmd}" for cmd in admissible)
    return f"{obs}\n\nAdmissible actions:\n{menu}"


# ─── Rollout ─────────────────────────────────────────────────────────────────
async def rollout_one(
    game_file: Path,
    use_augmented: bool,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_episode_steps: int,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        t_start = time.time()
        async with ENV_LOCK:
            env = await asyncio.to_thread(make_env, game_file, use_augmented, max_episode_steps)
            reset_result = await asyncio.to_thread(env.reset)
        if use_augmented:
            obs, infos = reset_result
        else:
            obs, infos = _unbatch(*reset_result)

        trajectory = [{
            "step": 0,
            "action": None,
            "observation": obs,
            "admissible_commands": list(infos.get("admissible_commands", []) or []),
            "score": 0.0,
            "done": False,
            "won": False,
            "was_admissible": None,
        }]

        initial_admissible = list(infos.get("admissible_commands", []) or [])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_turn(obs, initial_admissible)},
        ]

        final_score = 0.0
        final_won = False
        final_done = False
        llm_error = None

        for step_idx in range(1, max_episode_steps + 1):
            try:
                action = await llm_step(client, messages, model, temperature)
            except Exception as e:  # noqa: BLE001 — surface it as a per-episode error, not a global crash
                llm_error = f"{type(e).__name__}: {e}"
                break

            if not action:
                action = "look"

            was_admissible = action in (trajectory[-1]["admissible_commands"] or [])

            messages.append({"role": "assistant", "content": action})

            async with ENV_LOCK:
                if use_augmented:
                    step_result = await asyncio.to_thread(env.step, action)
                    obs, score, done, infos = step_result
                else:
                    # textworld batch-API expects a list even with batch_size=1
                    step_result = await asyncio.to_thread(env.step, [action])
                    obs_raw, score_raw, done_raw, infos_raw = step_result
                    obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
                    score = score_raw[0] if isinstance(score_raw, (list, tuple)) else score_raw
                    done = done_raw[0] if isinstance(done_raw, (list, tuple)) else done_raw
                    infos = {k: (v[0] if isinstance(v, list) else v)
                             for k, v in (infos_raw or {}).items()}

            trajectory.append({
                "step": step_idx,
                "action": action,
                "observation": obs,
                "admissible_commands": list(infos.get("admissible_commands", []) or []),
                "score": float(score),
                "done": bool(done),
                "won": bool(infos.get("won", False)),
                "was_admissible": was_admissible,
            })

            new_admissible = list(infos.get("admissible_commands", []) or [])
            messages.append({"role": "user", "content": format_user_turn(obs, new_admissible)})

            final_score = float(score)
            final_done = bool(done)
            final_won = bool(infos.get("won", False))

            if final_done:
                break

        # best-effort close
        try:
            async with ENV_LOCK:
                await asyncio.to_thread(env.close)
        except Exception:
            pass

        return {
            "game_file": str(game_file),
            "task_type": task_type_of(game_file),
            "use_augmented_env": use_augmented,
            "won": final_won,
            "final_score": final_score,
            "done": final_done,
            "steps": len(trajectory) - 1,
            "llm_error": llm_error,
            "wall_seconds": round(time.time() - t_start, 2),
            "trajectory": trajectory,
        }


async def run_phase(
    game_files: list[Path],
    use_augmented: bool,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_episode_steps: int,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        rollout_one(gf, use_augmented, client, model, temperature, max_episode_steps, sem)
        for gf in game_files
    ]
    results: list[dict] = []
    completed = 0
    total = len(tasks)
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        completed += 1
        marker = "W" if r.get("won") else ("." if not r.get("llm_error") else "E")
        print(f"  [{completed:3d}/{total}] {marker} steps={r.get('steps'):3d} "
              f"score={r.get('final_score'):.2f} task={r.get('task_type')}")
    return results


# ─── Reporting ───────────────────────────────────────────────────────────────
def summarize(results: list[dict], label: str) -> dict:
    ok = [r for r in results if not r.get("llm_error")]
    won = [r for r in ok if r["won"]]
    by_task: dict[str, dict] = {}
    for r in ok:
        b = by_task.setdefault(r["task_type"], {"won": 0, "total": 0, "steps_sum": 0})
        b["total"] += 1
        b["steps_sum"] += r["steps"]
        if r["won"]:
            b["won"] += 1
    return {
        "label": label,
        "total": len(results),
        "errors": len(results) - len(ok),
        "won": len(won),
        "success_rate": (len(won) / len(ok)) if ok else 0.0,
        "avg_steps": (sum(r["steps"] for r in ok) / len(ok)) if ok else 0.0,
        "avg_final_score": (sum(r["final_score"] for r in ok) / len(ok)) if ok else 0.0,
        "avg_wall_seconds": (sum(r["wall_seconds"] for r in ok) / len(ok)) if ok else 0.0,
        "by_task_type": {
            t: {
                "won": v["won"],
                "total": v["total"],
                "success_rate": v["won"] / v["total"] if v["total"] else 0.0,
                "avg_steps": v["steps_sum"] / v["total"] if v["total"] else 0.0,
            }
            for t, v in by_task.items()
        },
    }


def write_report(orig: list[dict], aug: list[dict], args: argparse.Namespace, out_dir: Path) -> None:
    orig_s = summarize(orig, "original")
    aug_s = summarize(aug, "augmented")

    (out_dir / "results_original.json").write_text(json.dumps(orig, indent=2, default=str))
    (out_dir / "results_augmented.json").write_text(json.dumps(aug, indent=2, default=str))
    (out_dir / "summary.json").write_text(
        json.dumps({"original": orig_s, "augmented": aug_s,
                    "config": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items()}},
                   indent=2)
    )

    def delta(a, b): return f"{b - a:+.3f}"
    md = f"""# Qwen Rollout Comparison — original vs augmented ALFWorld

Generated: {datetime.now(timezone.utc).isoformat()}
Model: `{args.model}` via `{args.base_url}`
Games: {args.n} (seed={args.seed}), max_episode_steps={args.max_steps},
concurrency={args.concurrency}, temperature={args.temperature}

## Overall

| Metric | Original | Augmented | Delta |
|---|---:|---:|---:|
| N episodes | {orig_s['total']} | {aug_s['total']} | |
| LLM errors | {orig_s['errors']} | {aug_s['errors']} | |
| Success rate | {orig_s['success_rate']:.3f} | {aug_s['success_rate']:.3f} | {delta(orig_s['success_rate'], aug_s['success_rate'])} |
| Avg steps | {orig_s['avg_steps']:.2f} | {aug_s['avg_steps']:.2f} | {delta(orig_s['avg_steps'], aug_s['avg_steps'])} |
| Avg final score | {orig_s['avg_final_score']:.3f} | {aug_s['avg_final_score']:.3f} | {delta(orig_s['avg_final_score'], aug_s['avg_final_score'])} |
| Avg wall-seconds | {orig_s['avg_wall_seconds']:.1f} | {aug_s['avg_wall_seconds']:.1f} | |

## Per task type

| Task type | Orig won/total | Orig avg steps | Aug won/total | Aug avg steps |
|---|---:|---:|---:|---:|
"""
    all_types = sorted(set(orig_s["by_task_type"]) | set(aug_s["by_task_type"]))
    for t in all_types:
        o = orig_s["by_task_type"].get(t, {"won": 0, "total": 0, "avg_steps": 0.0})
        a = aug_s["by_task_type"].get(t, {"won": 0, "total": 0, "avg_steps": 0.0})
        md += (
            f"| {t} | {o['won']}/{o['total']} | {o['avg_steps']:.1f} | "
            f"{a['won']}/{a['total']} | {a['avg_steps']:.1f} |\n"
        )

    md += "\n## Interpretation\n\n"
    if aug_s["success_rate"] > orig_s["success_rate"]:
        md += ("- Augmented env **improved** success rate by "
               f"{100 * (aug_s['success_rate'] - orig_s['success_rate']):.1f} percentage points.\n")
    elif aug_s["success_rate"] < orig_s["success_rate"]:
        md += ("- Augmented env **reduced** success rate by "
               f"{100 * (orig_s['success_rate'] - aug_s['success_rate']):.1f} percentage points.\n")
    else:
        md += "- Success rate unchanged.\n"
    md += ("\nFull per-episode trajectories are saved to `results_original.json` and "
           "`results_augmented.json` and can be scored by `../evaluation/trace_evaluator.py`.\n")

    (out_dir / "comparison_report.md").write_text(md)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen rollout: original vs augmented env")
    parser.add_argument("--n", type=int, default=64, help="number of games to sample")
    parser.add_argument("--max-steps", type=int, default=50, help="max_episode_steps per game")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--phase", choices=["both", "original", "augmented"], default="both")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                        help="comma-separated splits to sample from (default: "
                             "valid_seen,valid_unseen,train — together 3827 games)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    game_files = sample_game_files(args.n, args.seed, splits)
    split_counts: dict[str, int] = {}
    for g in game_files:
        split_name = g.parents[2].name  # .../json_2.1.1/<split>/<task>/trial_*/game.tw-pddl
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
    print(f"Sampled {len(game_files)} games from splits {splits} (seed={args.seed})")
    print(f"Split distribution: {split_counts}")
    print(f"First 3: {[g.parent.parent.name for g in game_files[:3]]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    async def go() -> None:
        orig: list[dict] | None = None
        aug: list[dict] | None = None
        t0 = time.time()

        if args.phase in ("both", "original"):
            print(f"\n=== Phase 1: original env ({len(game_files)} games, "
                  f"concurrency={args.concurrency}) ===")
            orig = await run_phase(game_files, False, client, args.model,
                                    args.temperature, args.max_steps, args.concurrency)
            (args.out_dir / "results_original.json").write_text(
                json.dumps(orig, indent=2, default=str))
            print(f"Phase 1 done in {time.time() - t0:.1f}s")

        if args.phase in ("both", "augmented"):
            t1 = time.time()
            print(f"\n=== Phase 2: augmented env ({len(game_files)} games, "
                  f"concurrency={args.concurrency}) ===")
            aug = await run_phase(game_files, True, client, args.model,
                                   args.temperature, args.max_steps, args.concurrency)
            (args.out_dir / "results_augmented.json").write_text(
                json.dumps(aug, indent=2, default=str))
            print(f"Phase 2 done in {time.time() - t1:.1f}s")

        if orig is not None and aug is not None:
            write_report(orig, aug, args, args.out_dir)
            print(f"\nReport written to {args.out_dir}/comparison_report.md")

    asyncio.run(go())


if __name__ == "__main__":
    main()
