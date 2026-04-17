# Benchmark2Scaffold Framework Architecture

## Goal

Build a benchmark-agnostic framework that:

1. Reads a benchmark's code, docs, evaluator, and runtime interface.
2. Explores the benchmark to collect trajectories and failure cases.
3. Designs text-only environment feedback augmentations.
4. Separately designs oracle selection and test plans for evaluation.
5. Verifies that augmentations improve useful signal without corrupting benchmark semantics.

This framework explicitly separates:

- `Text Environment Augmentation`
- `Oracle / Unit Test Design`

They inform each other, but they are different pipelines and must not be merged into one module.

## Non-Goals

- Do not assume ALFWorld-specific APIs, task templates, or latent state names.
- Do not assume a fixed ground-truth source.
- Do not treat binary success as the only evaluation target.
- Do not let augmentation logic directly modify reward, done, transition dynamics, or official evaluator semantics.

## Core Principles

- `Benchmark-first`: understand the benchmark before designing augmentations.
- `Agent-delegated but justified`: Claude Code may choose oracles and test strategy, but must explain why.
- `Exploration-heavy`: coverage collection is a first-class artifact, not a side effect.
- `Text-only augmentation`: observation text may change; environment dynamics must not.
- `Evaluation-separated`: augmentation design and test/oracle design must remain modular.

## High-Level Architecture

### 1. Benchmark Reader

Purpose:

- Read benchmark source code, runtime wrappers, evaluator, docs, configs, and task assets.

Primary agent:

- `Claude Code`

Outputs:

- `benchmark_spec.json`
- `benchmark_analysis.md`

Required contents:

- environment API schema
- observation/action/info/reward/done interface
- official metrics and evaluator entrypoints
- candidate latent-state channels
- candidate ground-truth sources
- minimal runnable commands

### 2. Exploration Runner

Purpose:

- Collect broad trajectory coverage across valid, invalid, recovery, boundary, and looping behaviors.

Primary agent:

- `Qwen` for rollout generation

Outputs:

- `trajectory_corpus/`
- `feedback_catalog.json`
- `coverage_report.json`
- `failure_clusters.json`

Required coverage dimensions:

- normal progression paths
- failure paths
- repair/recovery paths
- repeated-state loops
- benchmark-specific corner cases

### 3. Feedback Analysis

Purpose:

- Analyze whether environment returns are ambiguous, weak, redundant, or improvable.

Primary agent:

- `Claude Code`

Outputs:

- `feedback_audit.md`
- `augmentation_candidates.json`

Key questions:

- Does one returned message correspond to multiple hidden causes?
- Is the feedback actionable?
- Is the feedback already inferable from public context?
- Would disambiguation help policy recovery?
- Would an augmented message leak solution-specific information?

### 4. Text Environment Augmentation Pipeline

Purpose:

- Turn approved augmentation candidates into text-only environment wrappers or middleware.

Primary agent:

- `Claude Code`

Outputs:

- `augmentation_plan.json`
- `augmented_env.py` or equivalent wrapper
- `augmentation_smoke_tests.json`

Strict constraints:

- may change only observation text or equivalent textual feedback channel
- may not change reward
- may not change done
- may not change admissible action space
- may not change transition semantics

### 5. Oracle and Unit Test Design Pipeline

Purpose:

- Independently choose the best benchmark-specific oracle sources and derive test plans.

Primary agent:

- `Claude Code`

Outputs:

- `oracle_plan.json`
- `unit_test_plan.json`
- `verification_matrix.md`

This pipeline is separate from text augmentation. Its job is not to invent hints. Its job is to decide how correctness, utility, regression, and score impact should be tested.

### 6. Verification Runner

Purpose:

- Run benchmark-native, unit-level, and regression-level verification.

Primary agent:

- `Claude Code`

Outputs:

- `verification_report.md`
- `ab_eval_results.json`
- `unit_test_results.json`
- `non_regression_results.json`

## Required Separation: Two Independent Pipelines

### Pipeline A: Text Environment Augmentation

Owns:

- ambiguity analysis
- augmentation rule design
- text wrapper implementation
- leakage analysis for textual hints

Must not own:

- benchmark scoring definition
- official evaluator semantics
- oracle selection policy

### Pipeline B: Oracle / Unit Test Design

Owns:

- ground-truth candidate discovery
- oracle selection
- unit test construction
- score-based and state-based assertions
- non-regression evaluation design

Must not own:

- generation of hint text
- modification of environment textual outputs

Reason for separation:

- A benchmark may need strong tests even if no text augmentation is useful.
- A benchmark may benefit from text augmentation even when score oracle construction is complex.
- Keeping them separate prevents the augmentation agent from designing overly convenient tests for itself.

## Standard Artifacts

For each benchmark, the framework should produce at least:

- `benchmark_analysis.md`
- `benchmark_spec.json`
- `feedback_catalog.json`
- `coverage_report.json`
- `feedback_audit.md`
- `augmentation_plan.json`
- `oracle_plan.json`
- `unit_test_plan.json`
- `verification_report.md`

## Verification Layers

Every benchmark should be verified at three layers:

### Layer 1. Benchmark-Native Evaluation

- official evaluator
- official score
- official success or completion criteria

### Layer 2. Diagnostic Unit Tests

- per-rule trigger correctness
- per-rule non-leakage checks
- per-scenario recovery checks
- oracle-driven state assertions

### Layer 3. Non-Regression Tests

- reward unchanged
- done unchanged
- action space unchanged
- transition semantics unchanged
- public API unchanged except approved text fields

## Default Control Flow

1. Claude Code reads benchmark and emits `benchmark_spec`.
2. Qwen explores benchmark and emits trajectory corpus plus coverage artifacts.
3. Claude Code audits feedback and emits augmentation candidates.
4. Claude Code independently selects oracles and emits test policy.
5. Claude Code implements text augmentation only if justified.
6. Claude Code generates and runs verification using the chosen oracle plan.

## Decision Rule for Shipping Augmentation

An augmentation should ship only if all of the following hold:

- `utility`: expected to improve recovery, exploration, or decision quality
- `novelty`: adds information not already trivially available in public context
- `non_leakage`: does not expose solution path, hidden target location, or exact action sequence
- `semantic_preservation`: does not alter benchmark dynamics or scoring semantics
- `testability`: can be evaluated under the benchmark's chosen oracle plan
