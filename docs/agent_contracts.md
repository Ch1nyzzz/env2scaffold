# Agent Contracts

## Overview

This document defines the responsibilities, inputs, outputs, and decision boundaries for the two-agent framework:

- `Qwen`: exploration agent
- `Claude Code`: analysis, augmentation, oracle, and verification agent

The framework assumes `Claude Code` is the governing agent. `Qwen` is a specialized rollout worker.

## Shared Rules

- Both agents must treat the benchmark as the source of truth.
- Neither agent may silently redefine official scoring semantics.
- Both agents must preserve reproducibility by logging commands, configs, seeds, and benchmark versions where possible.
- All artifacts must be written to files, not only described in logs.

## Qwen Contract

### Role

`Qwen` is the exploration and coverage agent.

### Primary Objective

Generate broad and useful interaction data for downstream analysis.

### Allowed Responsibilities

- run large numbers of rollouts
- sample diverse trajectories
- trigger valid and invalid actions
- collect observations, infos, rewards, dones, and metadata
- maximize coverage over states, failure modes, and recovery attempts
- cluster raw failures if useful

### Forbidden Responsibilities

- choosing benchmark oracle sources
- defining official correctness
- deciding which augmentation is acceptable to ship
- making final leakage judgments
- editing benchmark evaluator semantics

### Required Outputs

- trajectories
- coverage metrics
- failure buckets
- feedback clusters
- reproducibility metadata

### Exploration Targets

Qwen should attempt to cover:

- successful progress states
- common failures
- ambiguous returns
- recovery attempts after failure
- repeated loops
- no-op or low-information transitions
- benchmark-specific boundary cases discovered by Claude Code

### Output Quality Bar

Qwen output must be:

- broad rather than optimized for benchmark score
- labeled with enough context for Claude Code to interpret
- sufficiently diverse to support ambiguity analysis

## Claude Code Contract

### Role

`Claude Code` is the benchmark reader, analysis agent, augmentation designer, oracle selector, and verifier.

### Primary Objective

Turn benchmark understanding plus exploration data into justified augmentation and evaluation decisions.

### Allowed Responsibilities

- read benchmark code, docs, configs, and evaluator
- derive benchmark interface schema
- analyze ambiguity and feedback utility
- design text augmentations
- choose ground-truth or oracle sources
- design unit tests and scenario tests
- run verification and summarize evidence

### Required Justifications

Whenever Claude Code chooses an oracle strategy or test strategy, it must explicitly record:

- candidate oracle sources
- why each candidate is or is not usable
- chosen oracle combination
- why the chosen plan best matches benchmark intent

### Forbidden Responsibilities

- modifying benchmark official evaluator semantics
- smuggling answer information into textual hints
- declaring a hint useful without evidence
- using unit tests that only prove wrapper self-consistency while ignoring benchmark-native evaluation

## Mandatory Separation of Duties

Claude Code must keep these two deliverables separate:

### A. Text Augmentation Deliverables

- `feedback_audit.md`
- `augmentation_plan.json`
- wrapper implementation
- leakage review

### B. Oracle / Test Deliverables

- `oracle_plan.json`
- `unit_test_plan.json`
- `verification_matrix.md`
- test runner outputs

The same file should not mix augmentation rule definitions with oracle selection policy.

## Required Claude Code Decision Format

When choosing oracle sources, Claude Code should emit a structure equivalent to:

```json
{
  "oracle_candidates": [
    {
      "source": "official_evaluator",
      "usable": true,
      "reason": "aligned with benchmark objective"
    },
    {
      "source": "hidden_env_state",
      "usable": true,
      "reason": "supports intermediate assertions without replacing official score"
    },
    {
      "source": "reference_trajectory",
      "usable": false,
      "reason": "not present in benchmark assets"
    }
  ],
  "chosen_oracle": [
    "official_evaluator",
    "hidden_env_state"
  ],
  "test_strategy": "benchmark-native score comparison plus targeted state-transition unit tests"
}
```

## Escalation Rules

Claude Code must escalate from analysis to implementation only if:

- benchmark reading is complete enough
- exploration coverage is adequate
- at least one augmentation candidate passes utility and non-leakage screening
- a viable oracle plan exists

If any of these fail, Claude Code should stop at analysis and report the blocker.

## Success Criteria by Agent

### Qwen succeeds if

- it provides enough trajectory coverage for ambiguity analysis
- it surfaces representative failure and recovery traces
- its output is reproducible and structured

### Claude Code succeeds if

- benchmark mechanics are correctly understood
- useful augmentations are distinguished from cosmetic rewrites
- oracle choice is justified
- unit tests are benchmark-aligned
- verification covers utility, leakage, and regression

## Review Bias Guardrail

Claude Code must assume a conflict of interest between:

- proving an augmentation looks good
- proving it actually improves benchmark-relevant outcomes

To counter this, every final verification must include:

- benchmark-native metrics
- diagnostic unit tests
- non-regression checks

No augmentation may be considered validated from only one of these three.
