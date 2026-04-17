"""
ALFWorld Trace Evaluator — Pipeline C (trace_evaluator)

Deterministic, offline trajectory-level scorer for ALFWorld episodes.
Consumes a sequence of step records and a unit-test plan; emits a ScoreReport.

No external dependencies beyond stdlib (json, dataclasses, pathlib, re).
Never imports from env2scaffold/augmentation/ or env2scaffold/oracle_test/.
Never mutates the input trajectory.
"""

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    """Result of scoring one episode trace."""
    task_type: str
    total_score: float
    per_unit_test: dict  # {unit_test_id: {passed, weight, contribution, detail}}
    success_bonus_applied: bool
    failure_penalty_applied: bool
    limitations_hit: list  # list[str]


# ---------------------------------------------------------------------------
# Helper predicates (pure functions over step lists)
# ---------------------------------------------------------------------------

def _obj_in_action(action: str, obj_lower: str) -> bool:
    """True if obj_lower appears as a substring in the action (case-insensitive)."""
    return obj_lower in action.lower()


def _detect_pickup(steps: list, obj_lower: str) -> int:
    """Return index of first step where agent takes the target object, or -1."""
    for i, s in enumerate(steps):
        a = s["action"].lower()
        if a.startswith("take ") and obj_lower in a:
            return i
    return -1


def _count_pickups(steps: list, obj_lower: str) -> int:
    """Count steps where agent takes the target object."""
    return sum(
        1 for s in steps
        if s["action"].lower().startswith("take ") and obj_lower in s["action"].lower()
    )


def _count_places(steps: list, obj_lower: str, recep_lower: str) -> int:
    """Count steps where agent places target at the target receptacle."""
    count = 0
    for s in steps:
        a = s["action"].lower()
        if (a.startswith("move ") or a.startswith("put ")) \
                and obj_lower in a and recep_lower in a:
            count += 1
    return count


def _detect_place(steps: list, obj_lower: str, recep_lower: str) -> int:
    """Return index of first step where agent places target at receptacle, or -1."""
    for i, s in enumerate(steps):
        a = s["action"].lower()
        if (a.startswith("move ") or a.startswith("put ")) \
                and obj_lower in a and recep_lower in a:
            return i
    return -1


def _detect_transform(steps: list, verb_lower: str, obj_lower: str) -> int:
    """Return index of first step where agent applies verb (clean/heat/cool) to target, or -1."""
    prefix = verb_lower + " "
    for i, s in enumerate(steps):
        a = s["action"].lower()
        if a.startswith(prefix) and obj_lower in a:
            return i
    return -1


def _count_invalid_actions(steps: list) -> tuple:
    """Return (invalid_count, checkable_count) for steps 1..N.

    action[i] is invalid if it was not in admissible_commands[i-1].
    Step 0 is excluded because we lack a prior admissible_commands list.
    """
    invalid = 0
    total = max(0, len(steps) - 1)
    for i in range(1, len(steps)):
        if steps[i]["action"] not in steps[i - 1]["admissible_commands"]:
            invalid += 1
    return invalid, total


def _invalid_rate_below(steps: list, threshold: float = 0.30) -> tuple:
    """Return (passed: bool, detail: str) for invalid-action-rate check."""
    invalid, total = _count_invalid_actions(steps)
    if total == 0:
        return True, "no checkable steps (single-step trace)"
    rate = invalid / total
    passed = rate < threshold
    detail = f"{invalid}/{total} actions invalid ({rate:.1%}); threshold {threshold:.0%}"
    return passed, detail


def _agent_holding_at_step(steps: list, step_idx: int, obj_lower: str) -> bool:
    """Infer whether agent holds obj at the START of step_idx.

    Uses admissible_commands[step_idx - 1]: if any command starts with
    'move <obj_lower>' the agent must be holding that object.
    """
    if step_idx <= 0:
        return False
    prev_admissible = steps[step_idx - 1]["admissible_commands"]
    move_prefix = "move " + obj_lower
    return any(cmd.lower().startswith(move_prefix) for cmd in prev_admissible)


def _visit_receptacle(steps: list, recep_lower: str) -> int:
    """Return index of first step where location contains recep_lower, or -1."""
    for i, s in enumerate(steps):
        if recep_lower in s.get("location", "").lower():
            return i
    return -1


# ---------------------------------------------------------------------------
# Per-task-type unit test suites
# ---------------------------------------------------------------------------

def _run_pas(steps: list, params: dict) -> dict:
    """Unit tests for pick_and_place_simple."""
    results = {}
    obj = params.get("object_target", "").lower()
    recep = params.get("parent_target", "").lower()
    no_params = not obj or not recep

    # UT_PAS_01 — pickup target
    if no_params:
        results["UT_PAS_01"] = (False, 0.30, "traj_data params unavailable")
    else:
        idx = _detect_pickup(steps, obj)
        passed = idx != -1
        detail = f"took '{obj}' at step {idx}" if passed else f"never took '{obj}'"
        results["UT_PAS_01"] = (passed, 0.30, detail)

    # UT_PAS_02 — visit target receptacle
    if no_params:
        results["UT_PAS_02"] = (False, 0.20, "traj_data params unavailable")
    else:
        idx = _visit_receptacle(steps, recep)
        passed = idx != -1
        detail = f"at '{recep}' at step {idx}" if passed else f"never visited '{recep}'"
        results["UT_PAS_02"] = (passed, 0.20, detail)

    # UT_PAS_03 — place target at receptacle
    if no_params:
        results["UT_PAS_03"] = (False, 0.35, "traj_data params unavailable")
    else:
        idx = _detect_place(steps, obj, recep)
        passed = idx != -1
        detail = (f"placed '{obj}' at '{recep}' at step {idx}"
                  if passed else f"never placed '{obj}' at '{recep}'")
        results["UT_PAS_03"] = (passed, 0.35, detail)

    # UT_PAS_04 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_PAS_04"] = (passed, 0.15, detail)

    return results


def _run_loil(steps: list, params: dict) -> dict:
    """Unit tests for look_at_obj_in_light."""
    results = {}
    obj = params.get("object_target", "").lower()
    lamp = params.get("toggle_target", "").lower()
    no_params = not obj or not lamp

    # UT_LOIL_01 — pickup target
    if no_params:
        results["UT_LOIL_01"] = (False, 0.25, "traj_data params unavailable")
    else:
        idx = _detect_pickup(steps, obj)
        passed = idx != -1
        detail = f"took '{obj}' at step {idx}" if passed else f"never took '{obj}'"
        results["UT_LOIL_01"] = (passed, 0.25, detail)

    # UT_LOIL_02 — lamp activated
    if no_params:
        results["UT_LOIL_02"] = (False, 0.20, "traj_data params unavailable")
    else:
        lamp_used_steps = [
            i for i, s in enumerate(steps)
            if s["action"].lower().startswith("use ") and lamp in s["action"].lower()
        ]
        passed = len(lamp_used_steps) > 0
        detail = (f"used '{lamp}' at step {lamp_used_steps[0]}"
                  if passed else f"never used '{lamp}'")
        results["UT_LOIL_02"] = (passed, 0.20, detail)

    # UT_LOIL_03 — lamp used while holding target
    if no_params:
        results["UT_LOIL_03"] = (False, 0.35, "traj_data params unavailable")
    else:
        compound_passed = False
        compound_detail = f"no 'use {lamp}' step found while holding '{obj}'"
        for i, s in enumerate(steps):
            a = s["action"].lower()
            if a.startswith("use ") and lamp in a:
                if _agent_holding_at_step(steps, i, obj):
                    compound_passed = True
                    compound_detail = (
                        f"used '{lamp}' at step {i} while holding '{obj}'"
                    )
                    break
        results["UT_LOIL_03"] = (compound_passed, 0.35, compound_detail)

    # UT_LOIL_04 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_LOIL_04"] = (passed, 0.20, detail)

    return results


def _run_pclean(steps: list, params: dict) -> dict:
    """Unit tests for pick_clean_then_place_in_recep."""
    results = {}
    obj = params.get("object_target", "").lower()
    recep = params.get("parent_target", "").lower()
    no_params = not obj or not recep

    # UT_PCLEAN_01 — pickup
    if no_params:
        results["UT_PCLEAN_01"] = (False, 0.20, "traj_data params unavailable")
    else:
        idx = _detect_pickup(steps, obj)
        passed = idx != -1
        detail = f"took '{obj}' at step {idx}" if passed else f"never took '{obj}'"
        results["UT_PCLEAN_01"] = (passed, 0.20, detail)

    # UT_PCLEAN_02 — clean
    if no_params:
        results["UT_PCLEAN_02"] = (False, 0.25, "traj_data params unavailable")
    else:
        idx = _detect_transform(steps, "clean", obj)
        passed = idx != -1
        detail = f"cleaned '{obj}' at step {idx}" if passed else f"never cleaned '{obj}'"
        results["UT_PCLEAN_02"] = (passed, 0.25, detail)

    # UT_PCLEAN_03 — place
    if no_params:
        results["UT_PCLEAN_03"] = (False, 0.30, "traj_data params unavailable")
    else:
        idx = _detect_place(steps, obj, recep)
        passed = idx != -1
        detail = (f"placed '{obj}' at '{recep}' at step {idx}"
                  if passed else f"never placed '{obj}' at '{recep}'")
        results["UT_PCLEAN_03"] = (passed, 0.30, detail)

    # UT_PCLEAN_04 — correct pipeline order
    if no_params:
        results["UT_PCLEAN_04"] = (False, 0.10, "traj_data params unavailable")
    else:
        p_i = _detect_pickup(steps, obj)
        c_i = _detect_transform(steps, "clean", obj)
        pl_i = _detect_place(steps, obj, recep)
        if p_i == -1 or c_i == -1 or pl_i == -1:
            passed = False
            detail = (
                f"missing milestone(s): pickup={p_i}, clean={c_i}, place={pl_i}"
            )
        else:
            passed = p_i < c_i < pl_i
            detail = (
                f"pickup={p_i} < clean={c_i} < place={pl_i}: {'OK' if passed else 'BAD ORDER'}"
            )
        results["UT_PCLEAN_04"] = (passed, 0.10, detail)

    # UT_PCLEAN_05 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_PCLEAN_05"] = (passed, 0.15, detail)

    return results


def _run_pheat(steps: list, params: dict) -> dict:
    """Unit tests for pick_heat_then_place_in_recep."""
    results = {}
    obj = params.get("object_target", "").lower()
    recep = params.get("parent_target", "").lower()
    no_params = not obj or not recep

    # UT_PHEAT_01 — pickup
    if no_params:
        results["UT_PHEAT_01"] = (False, 0.20, "traj_data params unavailable")
    else:
        idx = _detect_pickup(steps, obj)
        passed = idx != -1
        detail = f"took '{obj}' at step {idx}" if passed else f"never took '{obj}'"
        results["UT_PHEAT_01"] = (passed, 0.20, detail)

    # UT_PHEAT_02 — heat
    if no_params:
        results["UT_PHEAT_02"] = (False, 0.25, "traj_data params unavailable")
    else:
        idx = _detect_transform(steps, "heat", obj)
        passed = idx != -1
        detail = f"heated '{obj}' at step {idx}" if passed else f"never heated '{obj}'"
        results["UT_PHEAT_02"] = (passed, 0.25, detail)

    # UT_PHEAT_03 — place
    if no_params:
        results["UT_PHEAT_03"] = (False, 0.30, "traj_data params unavailable")
    else:
        idx = _detect_place(steps, obj, recep)
        passed = idx != -1
        detail = (f"placed '{obj}' at '{recep}' at step {idx}"
                  if passed else f"never placed '{obj}' at '{recep}'")
        results["UT_PHEAT_03"] = (passed, 0.30, detail)

    # UT_PHEAT_04 — correct pipeline order
    if no_params:
        results["UT_PHEAT_04"] = (False, 0.10, "traj_data params unavailable")
    else:
        p_i = _detect_pickup(steps, obj)
        h_i = _detect_transform(steps, "heat", obj)
        pl_i = _detect_place(steps, obj, recep)
        if p_i == -1 or h_i == -1 or pl_i == -1:
            passed = False
            detail = (
                f"missing milestone(s): pickup={p_i}, heat={h_i}, place={pl_i}"
            )
        else:
            passed = p_i < h_i < pl_i
            detail = (
                f"pickup={p_i} < heat={h_i} < place={pl_i}: {'OK' if passed else 'BAD ORDER'}"
            )
        results["UT_PHEAT_04"] = (passed, 0.10, detail)

    # UT_PHEAT_05 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_PHEAT_05"] = (passed, 0.15, detail)

    return results


def _run_pcool(steps: list, params: dict) -> dict:
    """Unit tests for pick_cool_then_place_in_recep."""
    results = {}
    obj = params.get("object_target", "").lower()
    recep = params.get("parent_target", "").lower()
    no_params = not obj or not recep

    # UT_PCOOL_01 — pickup
    if no_params:
        results["UT_PCOOL_01"] = (False, 0.20, "traj_data params unavailable")
    else:
        idx = _detect_pickup(steps, obj)
        passed = idx != -1
        detail = f"took '{obj}' at step {idx}" if passed else f"never took '{obj}'"
        results["UT_PCOOL_01"] = (passed, 0.20, detail)

    # UT_PCOOL_02 — cool
    if no_params:
        results["UT_PCOOL_02"] = (False, 0.25, "traj_data params unavailable")
    else:
        idx = _detect_transform(steps, "cool", obj)
        passed = idx != -1
        detail = f"cooled '{obj}' at step {idx}" if passed else f"never cooled '{obj}'"
        results["UT_PCOOL_02"] = (passed, 0.25, detail)

    # UT_PCOOL_03 — place
    if no_params:
        results["UT_PCOOL_03"] = (False, 0.30, "traj_data params unavailable")
    else:
        idx = _detect_place(steps, obj, recep)
        passed = idx != -1
        detail = (f"placed '{obj}' at '{recep}' at step {idx}"
                  if passed else f"never placed '{obj}' at '{recep}'")
        results["UT_PCOOL_03"] = (passed, 0.30, detail)

    # UT_PCOOL_04 — correct pipeline order
    if no_params:
        results["UT_PCOOL_04"] = (False, 0.10, "traj_data params unavailable")
    else:
        p_i = _detect_pickup(steps, obj)
        c_i = _detect_transform(steps, "cool", obj)
        pl_i = _detect_place(steps, obj, recep)
        if p_i == -1 or c_i == -1 or pl_i == -1:
            passed = False
            detail = (
                f"missing milestone(s): pickup={p_i}, cool={c_i}, place={pl_i}"
            )
        else:
            passed = p_i < c_i < pl_i
            detail = (
                f"pickup={p_i} < cool={c_i} < place={pl_i}: {'OK' if passed else 'BAD ORDER'}"
            )
        results["UT_PCOOL_04"] = (passed, 0.10, detail)

    # UT_PCOOL_05 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_PCOOL_05"] = (passed, 0.15, detail)

    return results


def _run_ptwo(steps: list, params: dict) -> dict:
    """Unit tests for pick_two_obj_and_place."""
    results = {}
    obj = params.get("object_target", "").lower()
    recep = params.get("parent_target", "").lower()
    no_params = not obj or not recep

    n_pickups = _count_pickups(steps, obj) if not no_params else 0
    n_places = _count_places(steps, obj, recep) if not no_params else 0

    # UT_PTWO_01 — first pickup
    if no_params:
        results["UT_PTWO_01"] = (False, 0.20, "traj_data params unavailable")
    else:
        passed = n_pickups >= 1
        detail = f"picked up '{obj}' {n_pickups} time(s)"
        results["UT_PTWO_01"] = (passed, 0.20, detail)

    # UT_PTWO_02 — first placement
    if no_params:
        results["UT_PTWO_02"] = (False, 0.25, "traj_data params unavailable")
    else:
        passed = n_places >= 1
        detail = f"placed '{obj}' at '{recep}' {n_places} time(s)"
        results["UT_PTWO_02"] = (passed, 0.25, detail)

    # UT_PTWO_03 — second pickup
    if no_params:
        results["UT_PTWO_03"] = (False, 0.20, "traj_data params unavailable")
    else:
        passed = n_pickups >= 2
        detail = f"picked up '{obj}' {n_pickups} time(s); need >= 2"
        results["UT_PTWO_03"] = (passed, 0.20, detail)

    # UT_PTWO_04 — second placement
    if no_params:
        results["UT_PTWO_04"] = (False, 0.25, "traj_data params unavailable")
    else:
        passed = n_places >= 2
        detail = f"placed '{obj}' at '{recep}' {n_places} time(s); need >= 2"
        results["UT_PTWO_04"] = (passed, 0.25, detail)

    # UT_PTWO_05 — low invalid action rate
    passed, detail = _invalid_rate_below(steps, 0.30)
    results["UT_PTWO_05"] = (passed, 0.10, detail)

    return results


# Map task_type → runner function
_TASK_RUNNERS = {
    "pick_and_place_simple": _run_pas,
    "look_at_obj_in_light": _run_loil,
    "pick_clean_then_place_in_recep": _run_pclean,
    "pick_heat_then_place_in_recep": _run_pheat,
    "pick_cool_then_place_in_recep": _run_pcool,
    "pick_two_obj_and_place": _run_ptwo,
}


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class TraceEvaluator:
    """Deterministic offline scorer for ALFWorld episode traces.

    Usage
    -----
    evaluator = TraceEvaluator()                      # loads plan from sibling file
    report = evaluator.score_trajectory(trajectory)   # trajectory: list[dict] or full dict
    print(report.total_score, report.per_unit_test)
    """

    def __init__(self, plan_path: Optional[str] = None):
        if plan_path is None:
            plan_path = pathlib.Path(__file__).parent / "trace_unit_test_plan.json"
        self.plan = json.loads(pathlib.Path(plan_path).read_text())
        self._task_map = {tt["task_type"]: tt for tt in self.plan["task_types"]}
        self._rubric = self.plan["scoring_rubric"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_trajectory(
        self,
        trajectory: Union[list, dict],
        task_type: Optional[str] = None,
    ) -> ScoreReport:
        """Score one episode trajectory.

        Parameters
        ----------
        trajectory:
            Either a list of step-dicts (fields: step, action, observation,
            score, done, won, admissible_commands, location, …) or a full
            trajectory record dict with keys 'steps', 'task_type', 'game_file'.
        task_type:
            Task type identifier string.  If None, inferred from the
            trajectory record's 'task_type' field.

        Returns
        -------
        ScoreReport
        """
        # --- Normalise input --------------------------------------------------
        if isinstance(trajectory, dict):
            game_file = trajectory.get("game_file")
            steps = list(trajectory.get("steps", []))
            if task_type is None:
                task_type = trajectory.get("task_type")
        else:
            steps = list(trajectory)  # copy; never mutate input
            game_file = None

        if not steps:
            return ScoreReport(
                task_type=task_type or "unknown",
                total_score=0.0,
                per_unit_test={},
                success_bonus_applied=False,
                failure_penalty_applied=False,
                limitations_hit=["empty trajectory"],
            )

        if task_type is None:
            raise ValueError(
                "task_type could not be inferred from trajectory; pass explicitly."
            )

        if task_type not in _TASK_RUNNERS:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Known types: {list(_TASK_RUNNERS.keys())}"
            )

        # --- Load task params -------------------------------------------------
        limitations_hit: list = []
        params, limitation_msg = self._load_task_params(game_file)
        if limitation_msg:
            limitations_hit.append(limitation_msg)

        # --- Run unit tests ---------------------------------------------------
        raw_results = _TASK_RUNNERS[task_type](steps, params)

        # --- Build per_unit_test dict -----------------------------------------
        per_unit_test: dict = {}
        base_score = 0.0
        for uid, (passed, weight, detail) in raw_results.items():
            contribution = weight * (1.0 if passed else 0.0)
            base_score += contribution
            per_unit_test[uid] = {
                "passed": passed,
                "weight": weight,
                "contribution": contribution,
                "detail": detail,
            }

        # --- Success bonus / failure penalty ----------------------------------
        episode_won = any(s.get("won", False) for s in steps)
        last_step = steps[-1]
        episode_timed_out = (
            last_step.get("done", False) and not episode_won
        )

        total_score = base_score
        if episode_won:
            total_score += self._rubric["success_bonus"]["additive_value"]
        if episode_timed_out:
            total_score += self._rubric["failure_penalty"]["additive_value"]

        return ScoreReport(
            task_type=task_type,
            total_score=round(total_score, 6),
            per_unit_test=per_unit_test,
            success_bonus_applied=episode_won,
            failure_penalty_applied=episode_timed_out,
            limitations_hit=limitations_hit,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_task_params(self, game_file: Optional[str]) -> tuple:
        """Load pddl_params from traj_data.json adjacent to game_file.

        Returns (params_dict, limitation_message_or_None).
        params_dict is empty if loading fails.
        """
        if game_file is None:
            return {}, "traj_data.json not loaded: no game_file in trajectory metadata"

        traj_data_path = pathlib.Path(game_file).parent / "traj_data.json"
        if not traj_data_path.exists():
            return {}, f"traj_data.json not found at {traj_data_path}"

        try:
            td = json.loads(traj_data_path.read_text())
            return td.get("pddl_params", {}), None
        except Exception as exc:
            return {}, f"traj_data.json load error: {exc}"


# ---------------------------------------------------------------------------
# Smoke-test self-runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    traj_dir = pathlib.Path(__file__).parent.parent / "probing" / "trajectories"
    evaluator = TraceEvaluator()

    header = (
        f"{'Trajectory':<72} {'Task Type':<35} {'Score':>6} "
        f"{'Won':>4} {'Penalty':>7} {'#UT_pass':>8}"
    )
    print(header)
    print("-" * len(header))

    for traj_file in sorted(traj_dir.glob("*.json")):
        data = json.loads(traj_file.read_text())
        report = evaluator.score_trajectory(data)
        n_pass = sum(1 for v in report.per_unit_test.values() if v["passed"])
        n_total = len(report.per_unit_test)
        print(
            f"{traj_file.name:<72} {report.task_type:<35} {report.total_score:>6.3f} "
            f"{'Y' if report.success_bonus_applied else 'N':>4} "
            f"{'Y' if report.failure_penalty_applied else 'N':>7} "
            f"{n_pass}/{n_total:>2}"
        )

    print()
    print("Per-test breakdown for first trajectory of each task type:")
    seen_types = set()
    for traj_file in sorted(traj_dir.glob("*.json")):
        data = json.loads(traj_file.read_text())
        tt = data.get("task_type", "")
        if tt in seen_types:
            continue
        seen_types.add(tt)
        report = evaluator.score_trajectory(data)
        print(f"\n  [{tt}] — {traj_file.name}")
        for uid, v in report.per_unit_test.items():
            status = "PASS" if v["passed"] else "FAIL"
            print(f"    {uid:<15} {status}  w={v['weight']:.2f}  contrib={v['contribution']:.3f}  {v['detail']}")
        if report.limitations_hit:
            for lim in report.limitations_hit:
                print(f"    LIMITATION: {lim}")
