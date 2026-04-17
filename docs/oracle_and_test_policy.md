# Oracle And Test Policy

## Purpose

Define how ground truth, oracle selection, and unit test construction should work in the framework.

This policy intentionally does not hardcode a single ground-truth source. Instead, `Claude Code` must inspect the benchmark and choose the most appropriate oracle strategy.

This document governs:

- oracle discovery
- oracle selection
- unit test design
- score-based evaluation
- regression policy

It does not govern text augmentation rule wording.

## Core Policy

Ground truth is benchmark-dependent.

`Claude Code` must discover and evaluate candidate oracle sources after reading the benchmark. It may choose one source or combine several, but it must justify the choice.

## Candidate Oracle Sources

Typical candidates include:

- official evaluator output
- official scalar reward or score
- success or completion predicates
- hidden simulator state
- reference solutions or trajectories
- task asset annotations
- environment info fields
- deterministic milestone predicates derived from benchmark code

No candidate is automatically preferred except as constrained below.

## Selection Priorities

When available, prefer oracle sources in this order:

1. `official benchmark evaluator`
2. `official score / success definitions`
3. `benchmark-native state predicates directly implied by code`
4. `reference trajectories or task annotations`
5. `derived heuristics`

Heuristics should be a last resort and must be labeled as such.

## Mandatory Oracle Analysis

Before designing tests, `Claude Code` must produce:

- list of oracle candidates
- accessibility of each candidate
- fidelity of each candidate to benchmark goals
- determinism and reproducibility of each candidate
- misuse risk of each candidate

## Unit Test Categories

Unit tests must be separated into three groups.

### 1. Text Augmentation Correctness Tests

Goal:

- verify the text wrapper fires when it should and stays silent when it should

Examples:

- ambiguous failure X triggers augmentation rule Y
- non-ambiguous feedback remains unchanged
- hint text does not include forbidden leaked fields

These are not benchmark score tests.

### 2. Oracle-Driven Diagnostic Tests

Goal:

- use chosen ground truth to verify intermediate correctness

Examples:

- state predicate became true after a specific transition
- recovery action suggested by augmented feedback is actually available or valid
- a failure diagnosis matches hidden state

These are not the same as benchmark-native final evaluation.

### 3. Benchmark-Native Outcome Tests

Goal:

- verify that augmentation helps or at least does not harm benchmark-relevant outcomes

Examples:

- official score comparison
- success rate comparison
- average steps
- invalid action count
- recovery-after-failure rate

These must use the benchmark's own notion of performance whenever possible.

## Required Test Matrix

Every benchmark should have a matrix covering:

- `trigger correctness`
- `non-trigger correctness`
- `leakage checks`
- `recovery usefulness`
- `benchmark-native score impact`
- `non-regression of reward/done/action-space/transition semantics`

## Score Policy

Binary success alone is insufficient when richer benchmark-native scoring exists.

If the benchmark exposes a scalar score, partial-credit evaluator, or dense metric, that metric must be included in final evaluation.

If the benchmark only exposes binary success, Claude Code may derive additional diagnostic metrics, but it must clearly distinguish:

- official metric
- auxiliary metric

Auxiliary metrics may support analysis but may not replace the official metric.

## Test Design Rules

### Rule 1. Do Not Let Tests Serve The Wrapper

Tests must not be designed only to show that the wrapper produced longer text.

Acceptable:

- wrapper disambiguates a hidden failure and improves recovery

Unacceptable:

- wrapper changed output length and therefore "passed"

### Rule 2. Do Not Use Convenience Oracles Without Justification

If hidden state is easiest to inspect but not aligned with benchmark objectives, it may be used only for diagnostic tests, not as the sole success oracle.

### Rule 3. Keep Text And Test Pipelines Separate

- augmentation rules live in augmentation artifacts
- oracle choice and test logic live in oracle/test artifacts

The same JSON file should not define both textual hints and test oracle policy.

### Rule 4. Every Augmentation Rule Must Be Testable

For each shipped augmentation rule, there must be at least:

- one positive trigger test
- one negative non-trigger test
- one non-leakage check

### Rule 5. Every Benchmark Must Have Non-Regression Checks

At minimum:

- reward unchanged
- done unchanged
- action space unchanged
- transition semantics unchanged

If any of these change, the system is no longer doing text-only augmentation.

## Recommended Oracle Plan Schema

The framework should persist oracle decisions in a file shaped like:

```json
{
  "benchmark": "example_benchmark",
  "oracle_candidates": [
    {
      "name": "official_evaluator",
      "category": "official",
      "usable": true,
      "priority": 1,
      "reason": "primary benchmark metric"
    },
    {
      "name": "hidden_state_predicates",
      "category": "diagnostic",
      "usable": true,
      "priority": 2,
      "reason": "supports per-transition assertions"
    },
    {
      "name": "reference_trajectory",
      "category": "reference",
      "usable": false,
      "priority": 4,
      "reason": "not available"
    }
  ],
  "chosen_oracles": [
    "official_evaluator",
    "hidden_state_predicates"
  ],
  "intended_usage": {
    "official_evaluator": "final benchmark-native evaluation",
    "hidden_state_predicates": "diagnostic unit tests"
  }
}
```

## Recommended Unit Test Plan Schema

```json
{
  "test_groups": [
    {
      "name": "text_trigger_tests",
      "purpose": "verify augmentation triggers correctly"
    },
    {
      "name": "diagnostic_oracle_tests",
      "purpose": "verify failure diagnosis and recovery validity"
    },
    {
      "name": "benchmark_native_eval",
      "purpose": "measure score and success impact"
    },
    {
      "name": "non_regression",
      "purpose": "verify environment semantics unchanged"
    }
  ]
}
```

## Shipping Standard

An augmentation set should be considered acceptable only if:

- oracle choice is justified
- unit tests cover trigger, leakage, and recovery
- benchmark-native evaluation is included
- non-regression checks pass

If these are not satisfied, the framework may still output analysis artifacts, but it should not claim validated augmentation.
