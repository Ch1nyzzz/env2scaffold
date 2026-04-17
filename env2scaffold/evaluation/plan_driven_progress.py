"""
PlanDrivenProgressTracker — runtime adapter that turns Pipeline C's static
trace_unit_test_plan.json (agent-produced) into incremental per-step
progress_reward for training.

This module is **infrastructure / glue**: it does not define any milestone,
weight, or detector itself. All of the following come from
trace_evaluator agent outputs and are read-only here:
  - which unit tests exist per task_type          (plan JSON)
  - weights per unit test                         (plan JSON)
  - the detector functions that decide pass/fail  (trace_evaluator._TASK_RUNNERS)

Only two numbers are adapter concerns, chosen outside of the plan:
  - max_progress_per_episode  — scale factor (default 10.0 to align with 10×won)
  - the decision that "first time a UT passes, emit weight × scale"

Usage:
    tracker = PlanDrivenProgressTracker(
        task_type="pick_and_place_simple",
        game_file="/path/to/game.tw-pddl",
        max_progress_per_episode=10.0,
    )
    tracker.reset()
    for step_record in trajectory:
        delta, newly_fired = tracker.step(step_record)
        # delta is the per-step progress_reward
        # newly_fired is the list of UT ids that fired on this exact step

A step_record dict must have at least:
  step (int), action (str or None), observation (str),
  admissible_commands (list[str]), score (float), done (bool), won (bool)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
import trace_evaluator as _te  # noqa: E402


def extract_task_type_from_gamefile(game_file: Optional[str]) -> Optional[str]:
    """Task-type id lives in the task directory name, e.g.
    '.../valid_seen/pick_and_place_simple-Book-None-SideTable-329/trial_.../game.tw-pddl'
    Returns 'pick_and_place_simple' or None if the path does not match.
    """
    if not game_file:
        return None
    try:
        task_folder = Path(game_file).parent.parent.name
    except Exception:
        return None
    m = re.match(r"^([a-z_]+)-", task_folder)
    return m.group(1) if m else None


class PlanDrivenProgressTracker:
    """Per-episode stateful tracker. One instance per running env."""

    def __init__(
        self,
        task_type: str,
        game_file: Optional[str] = None,
        plan_path: Optional[str] = None,
        max_progress_per_episode: float = 10.0,
    ):
        self.task_type = task_type
        self.game_file = game_file
        self.max_progress_per_episode = float(max_progress_per_episode)

        if plan_path is None:
            plan_path = _EVAL_DIR / "trace_unit_test_plan.json"
        plan = json.loads(Path(plan_path).read_text())

        task_entry = next(
            (tt for tt in plan["task_types"] if tt["task_type"] == task_type),
            None,
        )
        if task_entry is None:
            raise ValueError(
                f"task_type '{task_type}' not present in plan "
                f"{Path(plan_path).name}; available: "
                f"{[tt['task_type'] for tt in plan['task_types']]}"
            )
        self._plan_uts = task_entry["unit_tests"]

        # Relative weights sum from the plan (per-task normalisation → 1.0).
        # Scale factor maps plan's 1.0 ceiling onto training reward scale.
        plan_weight_sum = sum(ut["weight"] for ut in self._plan_uts) or 1.0
        self._scale = self.max_progress_per_episode / plan_weight_sum

        # Detector dispatch — comes from the trace_evaluator agent module.
        if task_type not in _te._TASK_RUNNERS:
            raise ValueError(
                f"task_type '{task_type}' has no registered detector in trace_evaluator"
            )
        self._runner = _te._TASK_RUNNERS[task_type]

        # Task parameters from traj_data.json (best effort).
        # Reuse TraceEvaluator._load_task_params so we don't duplicate the
        # reading logic. The method only reads self.game_file via its arg,
        # so a duck-typed shim works.
        _shim = type("_Shim", (), {})()
        params, limitation_msg = _te.TraceEvaluator._load_task_params(_shim, game_file)
        self._params = params or {}
        self._task_params_limitation = limitation_msg

        # Mutable episode state
        self._steps_so_far: list[dict] = []
        self._fired_uts: set[str] = set()
        self._accumulated: float = 0.0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at episode start. Adapter is reused across episodes by
        trainers to avoid re-reading the plan JSON every game."""
        self._steps_so_far = []
        self._fired_uts = set()
        self._accumulated = 0.0

    def step(self, step_record: dict) -> tuple[float, list[str]]:
        """Append step_record to the internal trace, run detectors, emit delta
        for any UT that newly passes. Returns (progress_reward, fired_ids)."""
        self._steps_so_far.append(step_record)
        results = self._runner(self._steps_so_far, self._params)
        # results layout: {ut_id: (passed: bool, weight: float, detail: str)}

        delta = 0.0
        newly_fired: list[str] = []
        for ut_id, outcome in results.items():
            passed, weight, _detail = outcome
            if passed and ut_id not in self._fired_uts:
                self._fired_uts.add(ut_id)
                delta += weight * self._scale
                newly_fired.append(ut_id)

        self._accumulated += delta
        return delta, newly_fired

    # ------------------------------------------------------------------

    @property
    def accumulated(self) -> float:
        return self._accumulated

    @property
    def fired_uts(self) -> set[str]:
        return set(self._fired_uts)

    @property
    def task_params_limitation(self) -> Optional[str]:
        return self._task_params_limitation


# ----------------------------------------------------------------------
# Small CLI for smoke-testing: feed in a results_*.json from baseline_eval
# and show per-episode accumulated progress over the full trajectory.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test plan-driven progress tracker on existing rollout trajectories")
    parser.add_argument("--results", type=Path, required=True,
                        help="e.g. env2scaffold/baseline_eval/results_original.json")
    parser.add_argument("--max-progress", type=float, default=10.0)
    parser.add_argument("--max-episodes", type=int, default=5)
    args = parser.parse_args()

    records = json.loads(args.results.read_text())
    print(f"Loaded {len(records)} episodes from {args.results}")
    print(f"scale = {args.max_progress} / 1.0 (plan sum per task)")
    print()
    for r in records[: args.max_episodes]:
        tt = r.get("task_type")
        gf = r.get("game_file")
        if not tt or tt not in _te._TASK_RUNNERS:
            print(f"skip: task_type={tt}")
            continue

        tracker = PlanDrivenProgressTracker(
            task_type=tt, game_file=gf,
            max_progress_per_episode=args.max_progress,
        )
        tracker.reset()
        steps = [s for s in r["trajectory"] if s.get("action") is not None]
        per_step = []
        for s in steps:
            delta, fired = tracker.step(s)
            if delta > 0:
                per_step.append(f"step{s['step']} +{delta:.2f} ({','.join(fired)})")
        won = r.get("won")
        bonus = max(0.0, args.max_progress - tracker.accumulated) if won else 0.0
        print(f"[{tt}] won={won} steps={len(steps)}")
        print(f"  mid-progress: {' | '.join(per_step) or '(no UT fired)'}")
        print(f"  accumulated = {tracker.accumulated:.2f}, terminal_bonus = {bonus:.2f}, trajectory_return = {tracker.accumulated + bonus:.2f}")
        print()
