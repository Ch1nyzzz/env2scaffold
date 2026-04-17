#!/usr/bin/env python3
"""
ALFWorld Environment Augmentation Pipeline

Orchestrates 4 sub-agents via Claude Code CLI (headless mode):
  1. Probing Agent   — explore environment, collect feedback patterns
  2. Analysis Agent  — read source + trajectories, design augmentations, implement wrapper
  3. Progress Mining Agent — infer progress milestones from state transitions
  4. Verify Agent    — check leakage, validate effectiveness

Usage:
    python pipeline.py                          # run full pipeline
    python pipeline.py --agent probing          # run only probing
    python pipeline.py --agent analysis         # run only analysis (requires probing output)
    python pipeline.py --agent progress_mining  # run only progress mining
    python pipeline.py --agent verify           # run only verify (requires analysis + progress mining output)
    python pipeline.py --resume analysis        # resume from analysis onward

NOTE: pipeline stages and agent names are being refactored in M2/M3 to match
docs/framework_architecture.md. This file still references the legacy "analysis"
agent pointing at env2scaffold/augmentation/ — the full new config is coming.
    python pipeline.py --budget 10.0            # adjust per-agent budget
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROBING_DIR = ROOT / "probing"
AUGMENTATION_DIR = ROOT / "augmentation"
PROGRESS_DIR = ROOT / "progress"
VERIFY_DIR = ROOT / "verification"
PROMPTS_DIR = ROOT / "prompts"

# Agent definitions: order matters (pipeline sequence)
AGENTS = ["probing", "analysis", "progress_mining", "verify"]

AGENT_CONFIG = {
    "probing": {
        "prompt_file": "probing_agent_prompt.md",
        "log_file": "probing_agent.log",
        "expected_outputs": [
            PROBING_DIR / "trajectories",           # directory with json files
            PROBING_DIR / "feedback_catalog.json",
        ],
        "prerequisites": [],                        # no prerequisites
    },
    "analysis": {
        "prompt_file": "analysis_agent_prompt.md",
        "log_file": "analysis_agent.log",
        "expected_outputs": [
            AUGMENTATION_DIR / "source_analysis.md",
            AUGMENTATION_DIR / "augmentation_plan.json",
            AUGMENTATION_DIR / "augmented_env.py",
            AUGMENTATION_DIR / "smoke_test_result.json",
        ],
        "prerequisites": ["probing"],               # needs probing output
    },
    "progress_mining": {
        "prompt_file": "progress_mining_agent_prompt.md",
        "log_file": "progress_mining_agent.log",
        "expected_outputs": [
            PROGRESS_DIR / "mine_progress_rules.py",
            PROGRESS_DIR / "progress_rules.json",
            PROGRESS_DIR / "progress_mining_report.md",
        ],
        "prerequisites": ["probing", "analysis"],   # needs enriched trajectories + analysis context
    },
    "verify": {
        "prompt_file": "verify_agent_prompt.md",
        "log_file": "verify_agent.log",
        "expected_outputs": [
            VERIFY_DIR / "verify_report.md",
        ],
        "prerequisites": ["analysis", "progress_mining"],  # needs analysis + progress outputs
    },
}


def log(msg: str, level: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"INFO": "●", "OK": "✓", "FAIL": "✗", "WAIT": "…"}
    print(f"[{timestamp}] {prefix.get(level, '●')} {msg}")


def load_prompt(agent_name: str) -> str:
    """Load the prompt file for a given agent."""
    prompt_file = PROMPTS_DIR / AGENT_CONFIG[agent_name]["prompt_file"]
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text()


def check_agent_ready(agent_name: str) -> bool:
    """Check if prerequisite agents' outputs exist."""
    for prereq in AGENT_CONFIG[agent_name]["prerequisites"]:
        for path in AGENT_CONFIG[prereq]["expected_outputs"]:
            if path.suffix:  # it's a file
                if not path.exists():
                    log(f"Missing prerequisite: {path}", "FAIL")
                    return False
            else:  # it's a directory
                if not path.exists() or not any(path.iterdir()):
                    log(f"Missing or empty prerequisite dir: {path}", "FAIL")
                    return False
    return True


def check_agent_outputs(agent_name: str) -> bool:
    """Check if the expected outputs of an agent were produced."""
    all_ok = True
    for path in AGENT_CONFIG[agent_name]["expected_outputs"]:
        if path.suffix:  # file
            if path.exists() and path.stat().st_size > 0:
                log(f"  Output OK: {path.relative_to(ROOT)}", "OK")
            else:
                log(f"  Output MISSING: {path.relative_to(ROOT)}", "FAIL")
                all_ok = False
        else:  # directory
            if path.exists() and any(path.iterdir()):
                count = len(list(path.glob("*.json")))
                log(f"  Output OK: {path.relative_to(ROOT)} ({count} files)", "OK")
            else:
                log(f"  Output MISSING/EMPTY: {path.relative_to(ROOT)}", "FAIL")
                all_ok = False
    return all_ok


def run_claude_agent(agent_name: str) -> int:
    """Run a Claude Code sub-agent in headless mode."""
    log(f"Launching {agent_name} agent", "WAIT")

    prompt_file = PROMPTS_DIR / AGENT_CONFIG[agent_name]["prompt_file"]

    cmd = [
        "claude",
        "-p",                           # print mode (non-interactive)
        "--dangerously-skip-permissions",# auto-approve all tool calls
        "--model", "sonnet",             # use sonnet for cost efficiency
        "--add-dir", str(ROOT),          # give access to project dir
        "--system-prompt-file", str(prompt_file),
        f"Execute the task described in your system prompt. Work in {ROOT}. Do not ask questions — just do it.",
    ]

    log_file = ROOT / AGENT_CONFIG[agent_name]["log_file"]

    with open(log_file, "w") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )

        for line in process.stdout:
            sys.stdout.write(f"  [{agent_name}] {line}")
            lf.write(line)

        process.wait()

    log(f"{agent_name} agent exited with code {process.returncode}")
    return process.returncode


def run_agent(agent_name: str) -> bool:
    """Run a single agent with pre/post checks."""
    log(f"{'='*60}")
    log(f"Agent: {agent_name}")
    log(f"{'='*60}")

    # Pre-check
    if not check_agent_ready(agent_name):
        log(f"{agent_name} agent prerequisites not met. Run prerequisite agents first.", "FAIL")
        return False

    # Run agent
    returncode = run_claude_agent(agent_name)

    if returncode != 0:
        log(f"{agent_name} agent failed (exit code {returncode})", "FAIL")
        return False

    # Post-check
    log(f"Checking {agent_name} agent outputs:")
    if not check_agent_outputs(agent_name):
        log(f"{agent_name} agent outputs incomplete", "FAIL")
        return False

    log(f"{agent_name} agent completed successfully!", "OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="ALFWorld Env Augmentation Pipeline")
    parser.add_argument("--agent", choices=AGENTS,
                        help="Run only this agent")
    parser.add_argument("--resume", choices=AGENTS,
                        help="Resume from this agent (runs this agent and all after)")
    args = parser.parse_args()

    log("ALFWorld Environment Augmentation Pipeline")
    log(f"Project root: {ROOT}")

    if args.agent:
        agents_to_run = [args.agent]
    elif args.resume:
        start_idx = AGENTS.index(args.resume)
        agents_to_run = AGENTS[start_idx:]
    else:
        agents_to_run = AGENTS

    for agent_name in agents_to_run:
        success = run_agent(agent_name)
        if not success:
            log(f"Pipeline stopped at {agent_name} agent", "FAIL")
            sys.exit(1)

    log(f"{'='*60}")
    log("Pipeline completed!", "OK")
    log(f"Results in: {ROOT}")


if __name__ == "__main__":
    main()
