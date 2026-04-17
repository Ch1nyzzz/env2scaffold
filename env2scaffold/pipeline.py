#!/usr/bin/env python3
"""
env2scaffold pipeline orchestrator.

Runs six Claude Code sub-agents in headless mode, in the order prescribed by
docs/framework_architecture.md. Pipeline A (augmentation_builder) and
Pipeline B (oracle_designer) run in parallel within a single stage.

Stages:
  1. probing                                   — explore the environment, collect feedback patterns
  2. benchmark_reader                          — spec the benchmark, enumerate oracle candidates
  3. feedback_auditor ∥ trace_evaluator        — audit ambiguity (Pipeline C-prep) + design trace-level scoring (Pipeline C)
  4. augmentation_builder ∥ oracle_designer    — Pipeline A (wrapper) and Pipeline B (wrapper tests), parallel
  5. verify_runner                             — execute three-layer wrapper verification, aggregate report

Pipeline C (trace_evaluator) is decoupled from Pipelines A/B: it runs as early as possible
(with feedback_auditor) so its outputs are ready for trainers independently of wrapper design.
Pipeline C does not read A's or B's artifacts; it does not produce wrapper text.

Usage:
    python pipeline.py                              # run full pipeline
    python pipeline.py --agent feedback_auditor     # run a single agent
    python pipeline.py --resume augmentation_builder  # run this agent and everything after
    python pipeline.py --serial                     # disable stage-level parallelism

Failure in one agent within a parallel stage does NOT short-circuit siblings;
the stage as a whole fails, and subsequent stages do not run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROBING_DIR = ROOT / "probing"
BENCHMARK_SPEC_DIR = ROOT / "benchmark_spec"
AUDIT_DIR = ROOT / "audit"
AUGMENTATION_DIR = ROOT / "augmentation"
ORACLE_TEST_DIR = ROOT / "oracle_test"
EVALUATION_DIR = ROOT / "evaluation"
VERIFICATION_DIR = ROOT / "verification"
PROMPTS_DIR = ROOT / "prompts"

# ─── Pipeline definition ─────────────────────────────────────────────────────
# STAGES is the source of truth for ordering. Each inner list runs in parallel;
# stages run sequentially.
STAGES: list[list[str]] = [
    ["probing"],
    ["benchmark_reader"],
    ["feedback_auditor", "trace_evaluator"],
    ["augmentation_builder", "oracle_designer"],
    ["verify_runner"],
]
AGENTS: list[str] = [name for stage in STAGES for name in stage]

AGENT_CONFIG: dict[str, dict] = {
    "probing": {
        "prompt_file": "probing_agent.md",
        "log_file": "probing_agent.log",
        "expected_outputs": [
            PROBING_DIR / "trajectories",
            PROBING_DIR / "feedback_catalog.json",
        ],
        "prerequisites": [],
    },
    "benchmark_reader": {
        "prompt_file": "benchmark_reader.md",
        "log_file": "benchmark_reader.log",
        "expected_outputs": [
            BENCHMARK_SPEC_DIR / "benchmark_spec.json",
            BENCHMARK_SPEC_DIR / "benchmark_analysis.md",
        ],
        "prerequisites": ["probing"],
    },
    "feedback_auditor": {
        "prompt_file": "feedback_auditor.md",
        "log_file": "feedback_auditor.log",
        "expected_outputs": [
            AUDIT_DIR / "augmentation_candidates.json",
            AUDIT_DIR / "feedback_audit.md",
        ],
        "prerequisites": ["probing", "benchmark_reader"],
    },
    "trace_evaluator": {
        "prompt_file": "trace_evaluator.md",
        "log_file": "trace_evaluator.log",
        "expected_outputs": [
            EVALUATION_DIR / "trace_unit_test_plan.json",
            EVALUATION_DIR / "trace_evaluator.py",
            EVALUATION_DIR / "evaluation_report.md",
        ],
        # Deliberately does NOT depend on feedback_auditor: Pipeline C is decoupled
        # from Pipelines A and B. See docs/framework_architecture.md separation
        # of text augmentation (A) from trace-level evaluation (C).
        "prerequisites": ["probing", "benchmark_reader"],
    },
    "augmentation_builder": {
        "prompt_file": "augmentation_builder.md",
        "log_file": "augmentation_builder.log",
        "expected_outputs": [
            AUGMENTATION_DIR / "augmented_env.py",
            AUGMENTATION_DIR / "augmentation_plan.json",
            AUGMENTATION_DIR / "leakage_review.md",
        ],
        "prerequisites": ["benchmark_reader", "feedback_auditor"],
    },
    "oracle_designer": {
        "prompt_file": "oracle_designer.md",
        "log_file": "oracle_designer.log",
        "expected_outputs": [
            ORACLE_TEST_DIR / "oracle_plan.json",
            ORACLE_TEST_DIR / "unit_test_plan.json",
            ORACLE_TEST_DIR / "verification_matrix.md",
        ],
        # Deliberately does NOT depend on augmentation_builder: Pipeline A and B are
        # parallel and independent. See docs/framework_architecture.md.
        "prerequisites": ["benchmark_reader", "feedback_auditor"],
    },
    "verify_runner": {
        "prompt_file": "verify_runner.md",
        "log_file": "verify_runner.log",
        "expected_outputs": [
            VERIFICATION_DIR / "layer1_benchmark_native.py",
            VERIFICATION_DIR / "layer1_benchmark_native_results.json",
            VERIFICATION_DIR / "layer2_diagnostic_unit.py",
            VERIFICATION_DIR / "layer2_diagnostic_unit_results.json",
            VERIFICATION_DIR / "layer3_non_regression.py",
            VERIFICATION_DIR / "layer3_non_regression_results.json",
            VERIFICATION_DIR / "verify_report.md",
        ],
        "prerequisites": ["augmentation_builder", "oracle_designer"],
    },
}


# ─── Logging ─────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"INFO": "●", "OK": "✓", "FAIL": "✗", "WAIT": "…"}.get(level, "●")
    print(f"[{timestamp}] {prefix} {msg}")


# ─── Prerequisite / output checks ────────────────────────────────────────────
def _path_is_satisfied(path: Path) -> bool:
    """A file path is satisfied if it exists and is non-empty; a directory path is
    satisfied if it exists and contains at least one entry."""
    if path.suffix:
        return path.exists() and path.stat().st_size > 0
    return path.exists() and any(path.iterdir())


def check_agent_ready(agent_name: str) -> bool:
    for prereq in AGENT_CONFIG[agent_name]["prerequisites"]:
        for path in AGENT_CONFIG[prereq]["expected_outputs"]:
            if not _path_is_satisfied(path):
                log(f"{agent_name}: missing prerequisite {path.relative_to(ROOT)}", "FAIL")
                return False
    return True


def check_agent_outputs(agent_name: str) -> bool:
    all_ok = True
    for path in AGENT_CONFIG[agent_name]["expected_outputs"]:
        if _path_is_satisfied(path):
            if path.suffix:
                log(f"  Output OK: {path.relative_to(ROOT)}", "OK")
            else:
                count = len(list(path.glob("*")))
                log(f"  Output OK: {path.relative_to(ROOT)} ({count} entries)", "OK")
        else:
            log(f"  Output MISSING: {path.relative_to(ROOT)}", "FAIL")
            all_ok = False
    return all_ok


# ─── Subprocess runner ───────────────────────────────────────────────────────
def run_claude_agent(agent_name: str) -> int:
    """Run a Claude Code sub-agent in headless mode. Streams stdout to both the
    terminal (prefixed) and to the agent's log file. Returns the exit code."""
    log(f"Launching {agent_name}", "WAIT")

    cfg = AGENT_CONFIG[agent_name]
    prompt_file = PROMPTS_DIR / cfg["prompt_file"]
    if not prompt_file.exists():
        log(f"{agent_name}: prompt file not found: {prompt_file}", "FAIL")
        return 127

    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--model", "sonnet",
        "--add-dir", str(ROOT),
        "--system-prompt-file", str(prompt_file),
        f"Execute the task described in your system prompt. Work in {ROOT}. "
        f"Do not ask questions — just do it.",
    ]

    log_path = ROOT / cfg["log_file"]
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"  [{agent_name}] {line}")
            lf.write(line)
        proc.wait()

    log(f"{agent_name} exited with code {proc.returncode}")
    return proc.returncode


def run_agent(agent_name: str) -> bool:
    log("=" * 60)
    log(f"Agent: {agent_name}")
    log("=" * 60)

    if not check_agent_ready(agent_name):
        return False

    returncode = run_claude_agent(agent_name)
    if returncode != 0:
        log(f"{agent_name} failed (exit code {returncode})", "FAIL")
        return False

    log(f"Checking {agent_name} outputs:")
    if not check_agent_outputs(agent_name):
        return False

    log(f"{agent_name} completed", "OK")
    return True


# ─── Stage execution ─────────────────────────────────────────────────────────
def run_stage(stage: list[str], serial: bool) -> bool:
    """Run every agent in the stage. Parallel by default, serial if requested or
    if the stage has only one agent. Returns True iff every agent succeeded."""
    if len(stage) == 1 or serial:
        return all(run_agent(name) for name in stage)

    log("=" * 60)
    log(f"Parallel stage: {stage}")
    log("=" * 60)

    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=len(stage)) as pool:
        futures = {pool.submit(run_agent, name): name for name in stage}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001 — we want the raw message in the log
                log(f"{name} raised: {exc}", "FAIL")
                results[name] = False

    return all(results.values())


# ─── Stage selection ─────────────────────────────────────────────────────────
def select_stages(single: str | None, resume: str | None) -> list[list[str]]:
    if single:
        return [[single]]
    if resume:
        for i, stage in enumerate(STAGES):
            if resume in stage:
                return STAGES[i:]
        raise ValueError(f"resume agent not found in any stage: {resume}")
    return STAGES


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="env2scaffold pipeline orchestrator")
    parser.add_argument("--agent", choices=AGENTS, help="Run only this agent")
    parser.add_argument(
        "--resume",
        choices=AGENTS,
        help="Resume from the stage containing this agent",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable stage-level parallelism (run Pipeline A and B sequentially)",
    )
    args = parser.parse_args()

    if args.agent and args.resume:
        parser.error("--agent and --resume are mutually exclusive")

    log("env2scaffold pipeline")
    log(f"Project root: {ROOT}")

    stages_to_run = select_stages(args.agent, args.resume)
    log(f"Stages to run: {stages_to_run}")

    for stage in stages_to_run:
        ok = run_stage(stage, serial=args.serial)
        if not ok:
            log(f"Pipeline stopped at stage {stage}", "FAIL")
            sys.exit(1)

    log("=" * 60)
    log("Pipeline completed", "OK")
    log(f"Artifacts under: {ROOT}")


if __name__ == "__main__":
    main()
