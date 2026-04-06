#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY_DIR = DEFAULT_REPO_ROOT / "alfworld_augment" / "probing" / "trajectories"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SYSTEM_PROMPT = """You are controlling an ALFWorld household agent.

Solve the task one text action at a time.

Rules:
- Reply with exactly one action command and nothing else.
- Use the environment feedback to repair failed actions.
- Avoid loops and redundant exploration.
- Do not invent object ids or location ids.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EnvTuning-compatible ALFWorld parquet datasets.")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default="alfworld_train.parquet")
    parser.add_argument("--val-name", default="alfworld_val.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--use-augmented-env", action="store_true", default=True)
    parser.add_argument("--no-use-augmented-env", dest="use_augmented_env", action="store_false")
    return parser.parse_args()


def load_trajectory(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def make_entry(traj: Dict[str, Any], max_episode_steps: int, use_augmented_env: bool) -> Dict[str, Any]:
    game_id = traj.get("game_id") or Path(traj["game_file"]).parent.name
    initial_obs = traj["initial_obs"]
    task_type = traj.get("task_type", "alfworld")

    return {
        "data_source": f"alfworld::{task_type}",
        "prompt": [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": initial_obs,
            },
        ],
        "ability": "alfworld_multi_turn",
        "reward_model": {
            "style": "interaction",
            "ground_truth": [],
        },
        "extra_info": {
            "split": "train",
            "index": game_id,
            "original_id": game_id,
            "dataset_type": "alfworld",
            "interaction_kwargs": {
                "name": "alfworld_augmented_env",
                "game_file": traj["game_file"],
                "game_id": game_id,
                "task_type": task_type,
                "max_episode_steps": max_episode_steps,
                "use_augmented_env": use_augmented_env,
            },
        },
    }


def collect_entries(trajectory_dir: Path, max_episode_steps: int, use_augmented_env: bool) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(trajectory_dir.glob("*.json")):
        traj = load_trajectory(path)
        if "game_file" not in traj or "initial_obs" not in traj:
            continue
        entries.append(make_entry(traj, max_episode_steps=max_episode_steps, use_augmented_env=use_augmented_env))
    if not entries:
        raise RuntimeError(f"No usable ALFWorld trajectories found in {trajectory_dir}.")
    return entries


def split_entries(entries: Sequence[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(entries)
    random.Random(seed).shuffle(items)
    val_count = max(1, int(len(items) * val_ratio))
    val_entries = items[:val_count]
    train_entries = items[val_count:]
    if not train_entries:
        raise RuntimeError("Validation split consumed the entire dataset; lower --val-ratio.")
    for entry in train_entries:
        entry["extra_info"]["split"] = "train"
    for entry in val_entries:
        entry["extra_info"]["split"] = "val"
    return train_entries, val_entries


def write_parquet(entries: Sequence[Dict[str, Any]], output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to export parquet datasets.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(entries))
    pq.write_table(table, output_path)


def main() -> None:
    args = parse_args()
    entries = collect_entries(
        trajectory_dir=args.trajectory_dir,
        max_episode_steps=args.max_episode_steps,
        use_augmented_env=args.use_augmented_env,
    )
    train_entries, val_entries = split_entries(entries, val_ratio=args.val_ratio, seed=args.seed)

    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    write_parquet(train_entries, train_path)
    write_parquet(val_entries, val_path)

    summary = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "num_total": len(entries),
        "num_train": len(train_entries),
        "num_val": len(val_entries),
        "use_augmented_env": args.use_augmented_env,
        "max_episode_steps": args.max_episode_steps,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
