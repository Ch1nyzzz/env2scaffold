# Trace-Level Scoring — original vs augmented rollouts

Scored with `evaluation/trace_evaluator.py` (Pipeline C, 28 unit tests across 6 task types).

## Overall

| Metric | Original | Augmented | Delta |
|---|---:|---:|---:|
| Episodes scored | 640 | 640 | |
| Scoring errors  | 0 | 0 | |
| Avg total_score | 1.017 | 1.080 | +0.063 |

## Per task type (avg_score · success_rate)

| Task type | Orig avg | Aug avg | Δ avg | Orig succ | Aug succ | Δ succ |
|---|---:|---:|---:|---:|---:|---:|
| look_at_obj_in_light | 0.894 | 0.950 | +0.056 | 0.567 | 0.600 | +0.033 |
| pick_and_place_simple | 1.091 | 1.103 | +0.011 | 0.819 | 0.833 | +0.014 |
| pick_clean_then_place_in_recep | 0.909 | 1.012 | +0.102 | 0.571 | 0.643 | +0.071 |
| pick_cool_then_place_in_recep | 0.908 | 0.899 | -0.009 | 0.596 | 0.596 | +0.000 |
| pick_heat_then_place_in_recep | 0.806 | 0.994 | +0.188 | 0.294 | 0.635 | +0.341 |
| pick_two_obj_and_place | 1.284 | 1.345 | +0.062 | 0.800 | 0.843 | +0.043 |

## Per unit test (pass rate)

Sorted by |Δ pass_rate| to surface which unit tests the wrapper moved most. Positive Δ means more pass under augmented.

| unit_test_id | N | Orig pass | Aug pass | Δ | Orig avg contrib | Aug avg contrib |
|---|---:|---:|---:|---:|---:|---:|
| UT_PCLEAN_03 | 112 | 0.589 | 0.679 | +0.089 | 0.177 | 0.204 |
| UT_PCLEAN_04 | 112 | 0.571 | 0.643 | +0.071 | 0.057 | 0.064 |
| UT_LOIL_03 | 60 | 0.333 | 0.400 | +0.067 | 0.117 | 0.140 |
| UT_PCLEAN_02 | 112 | 0.598 | 0.661 | +0.062 | 0.150 | 0.165 |
| UT_LOIL_04 | 60 | 0.950 | 1.000 | +0.050 | 0.190 | 0.200 |
| UT_PTWO_04 | 140 | 0.850 | 0.900 | +0.050 | 0.212 | 0.225 |
| UT_PHEAT_05 | 85 | 0.941 | 0.988 | +0.047 | 0.141 | 0.148 |
| UT_PTWO_02 | 140 | 0.893 | 0.936 | +0.043 | 0.223 | 0.234 |
| UT_PCOOL_01 | 99 | 0.646 | 0.606 | -0.040 | 0.129 | 0.121 |
| UT_PCLEAN_01 | 112 | 0.732 | 0.768 | +0.036 | 0.146 | 0.154 |
| UT_PTWO_03 | 140 | 0.907 | 0.943 | +0.036 | 0.181 | 0.189 |
| UT_PHEAT_01 | 85 | 0.765 | 0.729 | -0.035 | 0.153 | 0.146 |
| UT_PHEAT_04 | 85 | 0.588 | 0.553 | -0.035 | 0.059 | 0.055 |
| UT_PAS_04 | 144 | 0.951 | 0.986 | +0.035 | 0.143 | 0.148 |
| UT_LOIL_02 | 60 | 0.717 | 0.750 | +0.033 | 0.143 | 0.150 |
| UT_PTWO_05 | 140 | 0.950 | 0.979 | +0.029 | 0.095 | 0.098 |
| UT_PHEAT_02 | 85 | 0.659 | 0.635 | -0.024 | 0.165 | 0.159 |
| UT_PHEAT_03 | 85 | 0.706 | 0.682 | -0.024 | 0.212 | 0.205 |
| UT_PCLEAN_05 | 112 | 0.911 | 0.929 | +0.018 | 0.137 | 0.139 |
| UT_LOIL_01 | 60 | 0.817 | 0.800 | -0.017 | 0.204 | 0.200 |
| UT_PTWO_01 | 140 | 0.957 | 0.971 | +0.014 | 0.191 | 0.194 |
| UT_PCOOL_05 | 99 | 0.960 | 0.970 | +0.010 | 0.144 | 0.145 |
| UT_PCOOL_03 | 99 | 0.586 | 0.576 | -0.010 | 0.176 | 0.173 |
| UT_PCOOL_04 | 99 | 0.525 | 0.535 | +0.010 | 0.053 | 0.054 |
| UT_PAS_01 | 144 | 0.868 | 0.861 | -0.007 | 0.260 | 0.258 |
| UT_PAS_02 | 144 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 |
| UT_PAS_03 | 144 | 0.847 | 0.847 | +0.000 | 0.297 | 0.297 |
| UT_PCOOL_02 | 99 | 0.596 | 0.596 | +0.000 | 0.149 | 0.149 |

## Interpretation

### Matches binary success result
avg total_score +0.063 mirrors the binary success_rate delta +0.073 — trace-level
scoring corroborates the N=640 headline.

### pick_heat paradox
The task type with the largest success-rate gain (**+34.1 pp**) shows
**negative deltas on UT_PHEAT_01..04** (-0.024 to -0.035) but
**positive on UT_PHEAT_05** (+0.047). Likely explanation: augmented feedback
lets Qwen take shorter, more direct paths, skipping some "incidental"
milestone-detector signatures (e.g. alternate phrasing of `heat` command, or
intermediate pickups) while still reaching the actual goal. The won field is
the ground truth; UT_PHEAT pass rate is a less-direct proxy here.

### UT_PAS_02 = 0.000 on both sides
The `pick_and_place_simple` milestone UT_PAS_02 never fires in 144 episodes —
pass rate 0 on both original and augmented. Worth inspecting: either its
detector is too strict for how LLM agents phrase actions, or the milestone
itself is vacuous. This is a potential trace_evaluator revision for a future
Pipeline C iteration.

### Where the wrapper helps most at UT level
Top 5 UT gains (Δ pass_rate):

| UT | Task | Δ | Likely interpretation |
|---|---|---:|---|
| UT_PCLEAN_03 | pick_clean | +0.089 | disambiguation helps clean step |
| UT_PCLEAN_04 | pick_clean | +0.071 | efficiency / avoided error improvement |
| UT_LOIL_03 | look_at | +0.067 | lamp usage milestone |
| UT_PCLEAN_02 | pick_clean | +0.062 | clean-place pipeline step |
| UT_LOIL_04 | look_at | +0.050 | examine-under-lamp terminal milestone |

### Where the wrapper hurts (or is inert)
Five UTs show Δ ≤ -0.024, all in pick_heat / pick_cool / look_at_01 /
pick_and_place_simple_01. Combined absolute contribution change is small
(< 0.01 avg score), and the task-level success rate dominates in every
case. No task type regresses in success_rate except pick_cool_then_place
(flat at 0.596).
