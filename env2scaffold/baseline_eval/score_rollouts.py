#!/usr/bin/env python3
"""
Score qwen_rollout_compare.py trajectories using Pipeline C's TraceEvaluator.

Consumes:
  results_original.json   (original env rollouts)
  results_augmented.json  (augmented env rollouts)

Produces:
  scored_original.json    per-episode ScoreReport (compact form)
  scored_augmented.json
  scored_comparison.md    aggregated delta table
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
from trace_evaluator import TraceEvaluator  # noqa: E402


LABELS = ["original", "augmented"]


def load_rollouts(label: str) -> list[dict]:
    return json.loads((SCRIPT_DIR / f"results_{label}.json").read_text())


def score_label(evaluator: TraceEvaluator, records: list[dict]) -> tuple[list[dict], dict]:
    scored: list[dict] = []
    errors: list[str] = []
    for r in records:
        if r.get("llm_error"):
            continue
        # qwen_rollout stores step 0 as the reset record (action=None);
        # trace_evaluator expects every step to have a concrete action.
        # Filter pre-action records to match probing trajectory schema.
        steps = [s for s in r["trajectory"] if s.get("action") is not None]
        if not steps:
            continue
        try:
            report = evaluator.score_trajectory(
                {
                    "steps": steps,
                    "task_type": r["task_type"],
                    "game_file": r["game_file"],
                }
            )
        except Exception as exc:  # noqa: BLE001 — surface per-record failures
            errors.append(f"{r['game_file']}: {exc}")
            continue
        scored.append({
            "game_file": r["game_file"],
            "task_type": r["task_type"],
            "won": r["won"],
            "steps": r["steps"],
            "total_score": report.total_score,
            "success_bonus_applied": report.success_bonus_applied,
            "failure_penalty_applied": report.failure_penalty_applied,
            "per_unit_test": report.per_unit_test,
            "limitations_hit": report.limitations_hit,
        })

    # Aggregations
    agg_task = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "won": 0})
    ut_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "pass": 0, "contribution_sum": 0.0})
    for s in scored:
        agg_task[s["task_type"]]["n"] += 1
        agg_task[s["task_type"]]["score_sum"] += s["total_score"]
        if s["won"]:
            agg_task[s["task_type"]]["won"] += 1
        for uid, info in s["per_unit_test"].items():
            ut_stats[uid]["n"] += 1
            if info["passed"]:
                ut_stats[uid]["pass"] += 1
            ut_stats[uid]["contribution_sum"] += info["contribution"]

    summary = {
        "n": len(scored),
        "errors": len(errors),
        "avg_score": (sum(s["total_score"] for s in scored) / len(scored)) if scored else 0.0,
        "by_task_type": {
            t: {
                "n": v["n"],
                "avg_score": v["score_sum"] / v["n"] if v["n"] else 0.0,
                "success_rate": v["won"] / v["n"] if v["n"] else 0.0,
            } for t, v in agg_task.items()
        },
        "by_unit_test": {
            uid: {
                "n_applicable": v["n"],
                "pass_rate": v["pass"] / v["n"] if v["n"] else 0.0,
                "avg_contribution": v["contribution_sum"] / v["n"] if v["n"] else 0.0,
            } for uid, v in ut_stats.items()
        },
        "scoring_errors": errors[:20],
    }
    return scored, summary


def render_markdown(
    orig_scored: list[dict], orig_summ: dict,
    aug_scored: list[dict], aug_summ: dict,
) -> str:
    def delta(a: float, b: float) -> str:
        return f"{b - a:+.3f}"

    md = [
        "# Trace-Level Scoring — original vs augmented rollouts\n",
        "Scored with `evaluation/trace_evaluator.py` (Pipeline C, 28 unit tests across 6 task types).\n",
        "## Overall\n",
        "| Metric | Original | Augmented | Delta |",
        "|---|---:|---:|---:|",
        f"| Episodes scored | {orig_summ['n']} | {aug_summ['n']} | |",
        f"| Scoring errors  | {orig_summ['errors']} | {aug_summ['errors']} | |",
        f"| Avg total_score | {orig_summ['avg_score']:.3f} | {aug_summ['avg_score']:.3f} | {delta(orig_summ['avg_score'], aug_summ['avg_score'])} |",
        "",
        "## Per task type (avg_score · success_rate)\n",
        "| Task type | Orig avg | Aug avg | Δ avg | Orig succ | Aug succ | Δ succ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tasks = sorted(set(orig_summ["by_task_type"]) | set(aug_summ["by_task_type"]))
    for t in tasks:
        o = orig_summ["by_task_type"].get(t, {"avg_score": 0.0, "success_rate": 0.0})
        a = aug_summ["by_task_type"].get(t, {"avg_score": 0.0, "success_rate": 0.0})
        md.append(
            f"| {t} | {o['avg_score']:.3f} | {a['avg_score']:.3f} | "
            f"{delta(o['avg_score'], a['avg_score'])} | "
            f"{o['success_rate']:.3f} | {a['success_rate']:.3f} | "
            f"{delta(o['success_rate'], a['success_rate'])} |"
        )

    md.extend([
        "",
        "## Per unit test (pass rate)\n",
        "Sorted by |Δ pass_rate| to surface which unit tests the wrapper "
        "moved most. Positive Δ means more pass under augmented.\n",
        "| unit_test_id | N | Orig pass | Aug pass | Δ | Orig avg contrib | Aug avg contrib |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    all_uids = sorted(set(orig_summ["by_unit_test"]) | set(aug_summ["by_unit_test"]))
    rows = []
    for uid in all_uids:
        o = orig_summ["by_unit_test"].get(uid, {"n_applicable": 0, "pass_rate": 0.0, "avg_contribution": 0.0})
        a = aug_summ["by_unit_test"].get(uid, {"n_applicable": 0, "pass_rate": 0.0, "avg_contribution": 0.0})
        rows.append((uid, o, a, abs(a["pass_rate"] - o["pass_rate"])))
    rows.sort(key=lambda x: -x[3])
    for uid, o, a, _ in rows:
        n_app = max(o["n_applicable"], a["n_applicable"])
        md.append(
            f"| {uid} | {n_app} | {o['pass_rate']:.3f} | {a['pass_rate']:.3f} | "
            f"{delta(o['pass_rate'], a['pass_rate'])} | "
            f"{o['avg_contribution']:.3f} | {a['avg_contribution']:.3f} |"
        )
    md.append("")
    return "\n".join(md)


def main() -> None:
    evaluator = TraceEvaluator()
    print(f"Loaded TraceEvaluator with {sum(len(tt['unit_tests']) for tt in evaluator.plan['task_types'])} unit tests")

    scored_by_label: dict[str, list[dict]] = {}
    summ_by_label: dict[str, dict] = {}
    for label in LABELS:
        records = load_rollouts(label)
        print(f"\n=== Scoring {label} ({len(records)} episodes) ===")
        scored, summ = score_label(evaluator, records)
        (SCRIPT_DIR / f"scored_{label}.json").write_text(json.dumps(scored, indent=2))
        print(f"  scored {summ['n']} episodes, {summ['errors']} errors")
        print(f"  avg total_score: {summ['avg_score']:.3f}")
        scored_by_label[label] = scored
        summ_by_label[label] = summ

    md = render_markdown(
        scored_by_label["original"], summ_by_label["original"],
        scored_by_label["augmented"], summ_by_label["augmented"],
    )
    (SCRIPT_DIR / "scored_comparison.md").write_text(md)
    print(f"\nReport: {SCRIPT_DIR / 'scored_comparison.md'}")

    # Also dump the summary JSON for machine consumption
    (SCRIPT_DIR / "scored_summary.json").write_text(json.dumps({
        "original": summ_by_label["original"],
        "augmented": summ_by_label["augmented"],
    }, indent=2))


if __name__ == "__main__":
    main()
