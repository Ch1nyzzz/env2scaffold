# Trace Evaluator — Evaluation Report

**Pipeline role:** C — trace_evaluator  
**Benchmark:** ALFWorld (alfworld 0.4.2 / textworld 1.7.0)  
**Generated:** 2026-04-16  
**Unit tests:** 28 across 6 task types  
**Score range:** [−0.1, 1.5]

---

## Trace Schema

The scorer consumes fields from two sources:

### Top-level trajectory record fields (when passed as a full dict)

| Field | Type | Usage |
|---|---|---|
| `game_file` | `str` | Derives path to `traj_data.json` for task-parameter loading |
| `task_type` | `str` | Selects the per-task-type unit test suite |
| `steps` | `list[dict]` | The step sequence fed to predicates |

Verified present in all 12 probing trajectories (`probing/trajectories/*.json`).

### Per-step fields (from actual probing trajectories)

| Field | Type | Verified values | Usage |
|---|---|---|---|
| `step` | `int` | 0..44 (45-step probing) | Step index reference in detail messages |
| `action` | `str` | e.g. `"take butterknife 1 from countertop 2"` | All milestone predicates parse action verbs |
| `observation` | `str` | text feedback | **NOT USED** (observation-text forbidden per role spec) |
| `score` | `float` | 0 or 1 | Not used directly (won used instead) |
| `done` | `bool` | False in all probing samples | Failure-penalty condition |
| `won` | `bool` | False in all probing samples | Success-bonus condition |
| `admissible_commands` | `list[str]` | e.g. `["go to bed 1", "take book 1 from bed 1", ...]` | Invalid-action detection; held-object inference |
| `inventory_state` | `str` | Always `"empty"` in probing | NOT USED (unreliable in probing recordings) |
| `holding` | `str` | Always `""` in probing | NOT USED (unreliable in probing recordings) |
| `location` | `str` | e.g. `"sidetable 1"`, `"countertop 2"` | UT_PAS_02: receptacle-visit detection |

**Note on `holding`/`inventory_state`:** Both fields are always empty in the probing recordings even after successful `take` actions. Object-holding state is therefore inferred from `admissible_commands`: if the step's admissible list includes any command starting with `"move <obj>"`, the agent is currently holding that object. This inference is grounded in the PDDL backend's action set: `move` (place) is only admissible while the object is held.

### traj_data.json fields (via `traj_data_task_annotation` oracle)

Accessed from `pathlib.Path(game_file).parent / "traj_data.json"`.

| Field | Type | Example values |
|---|---|---|
| `pddl_params.object_target` | `str` | `"Book"`, `"Apple"`, `"ButterKnife"` |
| `pddl_params.parent_target` | `str` | `"SideTable"`, `"CounterTop"`, `"DiningTable"` |
| `pddl_params.toggle_target` | `str` | `"DeskLamp"` (look_at_obj_in_light only) |

Verified present in all six game directories used by probing trajectories.

---

## Oracle Source Rationale

### `won_info_field`

**Spec entry:** `oracle_candidates[0]` — `"category": "official_evaluator"`, `"accessibility": "direct_api"`, `"determinism": "deterministic"`.

Every step in the probing trajectories records the `won` field directly. It is the benchmark's authoritative binary success signal (PDDL goal check). It is used as the **success-bonus trigger** rather than as a unit-test predicate, to avoid double-counting with milestone tests. The misuse risk noted in the spec (using `won` as dense reward) does not apply here because it is used as an episode-level boolean, not a step-level reward.

### `admissible_commands_validity_heuristic`

**Spec entry:** `oracle_candidates[6]` — `"category": "derived_heuristic"`, `"accessibility": "direct_api"`, `"determinism": "deterministic"`.

The `admissible_commands` list is recorded at every step. Two uses:
1. **Invalid-action detection** (UT_*_04/05): `action[N] not in admissible_commands[N-1]` is the exact PDDL-engine criterion for "Nothing happens."
2. **Held-object inference** (UT_LOIL_03): `any(cmd.startswith("move <obj>") for cmd in admissible_commands[N-1])` implies the agent holds `<obj>` at the start of step N.

The spec's misuse risk (admissible_commands does not distinguish task-relevant actions) does not apply here: both uses are structural (action validity and hold-state inference), not task-relevance classification.

### `traj_data_task_annotation`

**Spec entry:** `oracle_candidates[5]` — `"category": "task_annotation"`, `"accessibility": "external_file"`, `"determinism": "deterministic"`.

Provides `pddl_params` (object_target, parent_target, toggle_target) which ground all task-specific action-matching predicates. Without these parameters, predicates would need to extract target names from the observation text, which is forbidden. The path-joining logic uses `pathlib.Path(game_file).parent / "traj_data.json"` which is robust to the directory structure used in `~/.cache/alfworld/json_2.1.1/`.

**Oracles deliberately NOT used:**

| Oracle | Reason not used |
|---|---|
| `pddl_facts_state` | Not present in probing trajectories (facts not enabled at recording time); deferred to future scorer version |
| `policy_commands_plan` | Not recorded in probing trajectories; would reveal solution path (high misuse risk) |
| `expert_plan_handcoded` | Not recorded in probing trajectories |
| `goal_condition_success_rate_thor` | Only available in ThorEnv path; not the text-only ALFWorld benchmark path |
| `score_info_field` | Numerically identical to `won`; using both would be redundant |

---

## Per-Task-Type Unit Test Design

### pick_and_place_simple

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_PAS_01 | target_object_picked_up | milestone | 0.30 | none |
| UT_PAS_02 | target_receptacle_visited | milestone | 0.20 | none |
| UT_PAS_03 | target_placed_at_receptacle | milestone | 0.35 | none |
| UT_PAS_04 | low_invalid_action_rate | avoided_error | 0.15 | none |

**Weight rationale:** The placement milestone (UT_PAS_03) receives the most weight because it is both the terminal milestone and the only one that is necessary but not sufficient for success (the PDDL goal also verifies the object's state). The receptacle visit (UT_PAS_02) gets less weight than pickup because visiting the wrong receptacle first is a recoverable error.

### look_at_obj_in_light

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_LOIL_01 | target_object_picked_up | milestone | 0.25 | none |
| UT_LOIL_02 | lamp_activated | milestone | 0.20 | none |
| UT_LOIL_03 | lamp_used_while_holding_target | milestone | 0.35 | low |
| UT_LOIL_04 | low_invalid_action_rate | avoided_error | 0.20 | none |

**Design note:** UT_LOIL_03 is the compound milestone that approximates the PDDL goal ("hold target AND lamp is on"). The `admissible_commands` inference for held-object is the only structural signal available without PDDL facts. The avoided_error weight is higher than other task types (0.20 vs 0.15) because this task has the most steps where the agent might attempt invalid `use lamp` actions.

### pick_clean_then_place_in_recep

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_PCLEAN_01 | target_object_picked_up | milestone | 0.20 | none |
| UT_PCLEAN_02 | target_object_cleaned | milestone | 0.25 | none |
| UT_PCLEAN_03 | cleaned_object_placed_at_receptacle | milestone | 0.30 | none |
| UT_PCLEAN_04 | correct_pipeline_order | efficiency | 0.10 | low |
| UT_PCLEAN_05 | low_invalid_action_rate | avoided_error | 0.15 | none |

**Design note:** UT_PCLEAN_04 assigns a small weight (0.10) to the ordering constraint because all three component milestones must already be detected. The ordering check is a discriminative efficiency signal: an agent that picks up, places (without cleaning), then tries to clean is meaningfully worse than one that follows the correct pipeline.

### pick_heat_then_place_in_recep

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_PHEAT_01 | target_object_picked_up | milestone | 0.20 | none |
| UT_PHEAT_02 | target_object_heated | milestone | 0.25 | none |
| UT_PHEAT_03 | heated_object_placed_at_receptacle | milestone | 0.30 | none |
| UT_PHEAT_04 | correct_pipeline_order | efficiency | 0.10 | low |
| UT_PHEAT_05 | low_invalid_action_rate | avoided_error | 0.15 | none |

**Design note:** Structurally identical to pick_clean_then_place_in_recep with the `heat` verb substituted. The `heat X with microwave/stoveburner N` action form is parsed by the same `_detect_transform` helper.

### pick_cool_then_place_in_recep

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_PCOOL_01 | target_object_picked_up | milestone | 0.20 | none |
| UT_PCOOL_02 | target_object_cooled | milestone | 0.25 | none |
| UT_PCOOL_03 | cooled_object_placed_at_receptacle | milestone | 0.30 | none |
| UT_PCOOL_04 | correct_pipeline_order | efficiency | 0.10 | low |
| UT_PCOOL_05 | low_invalid_action_rate | avoided_error | 0.15 | none |

**Design note:** Identical structure to the heat variant; `cool X with fridge N` is the transformative action.

### pick_two_obj_and_place

| unit_test_id | name | kind | weight | leakage_risk |
|---|---|---|---|---|
| UT_PTWO_01 | first_target_object_picked_up | milestone | 0.20 | none |
| UT_PTWO_02 | first_target_placed_at_receptacle | milestone | 0.25 | none |
| UT_PTWO_03 | second_target_object_picked_up | milestone | 0.20 | none |
| UT_PTWO_04 | second_target_placed_at_receptacle | milestone | 0.25 | none |
| UT_PTWO_05 | low_invalid_action_rate | avoided_error | 0.10 | none |

**Design note:** The four milestone tests form a monotone chain (UT_PTWO_01 ⊆ UT_PTWO_03, UT_PTWO_02 ⊆ UT_PTWO_04 by count). An agent that retrieves and places only one instance scores 0.45, while one that completes both scores 0.90 (before success bonus). The avoided_error weight is intentionally lowest (0.10) here because searching for two instances inherently produces more navigational steps, increasing the chance of some invalid attempts.

---

## Scoring Rubric Rationale

### Aggregation: `weighted_sum`

A weighted sum of per-unit-test pass/fail values is chosen over alternatives:
- **vs. count_of_passes**: Weights encode relative task-progress importance. Placing the object is harder and more informative than visiting the receptacle, so it should contribute more.
- **vs. mean_of_weights**: Identical to equal-weight sum; loses the ability to express that terminal milestones matter more than earlier ones.
- **vs. task_aware_partial**: Already achieved by having per-task-type suites with different weights.

### Weights sum to 1.0 per task type

This ensures the base score before bonuses/penalties lives in [0.0, 1.0], making it directly interpretable as "fraction of task progress captured". The success bonus (+0.5) adds a distinct reward tier for episodes that actually complete the task, distinguishing "nearly successful" (e.g., score 0.90) from "just successful" (score 1.5).

### Success bonus: +0.5

Added when `any(step['won'] for step in steps)`. The 0.5 magnitude was chosen so that an agent completing the task always scores above 1.0, creating a clear tier boundary. An agent that passes all unit tests but somehow does not trigger `won` (edge case) would score 1.0 baseline.

### Failure penalty: −0.1

Applied when `steps[-1]['done'] == True and not any(step['won'])`, i.e., the episode terminated via step-limit rather than success. The −0.1 value is small (one avoided_error weight at most) to avoid distorting the base score signal while still penalising timeout-truncated episodes more than incomplete-but-still-running ones.

### Score range: [−0.1, 1.5]

- **Minimum −0.1**: all unit tests fail AND done=True timeout → 0.0 − 0.1 = −0.1
- **Maximum 1.5**: all unit tests pass (1.0) + success bonus (0.5) = 1.5
- Note: the success bonus and full-pass milestone suite are correlated but not identical. A won episode with some failed unit tests still gets the bonus; an episode with all milestones passed but `won=False` does not.

### Invalid-action threshold: 30%

The 30% threshold was calibrated from the probing trajectories:
- Agents with heavy invalid-action rates (UT_LOIL_04: 63.6%, UT_PCLEAN_05: 70.5%) represent clearly stuck/confused behaviour.
- Agents navigating correctly but failing on task-specific steps typically produce <30% invalid actions (UT_PAS_04: 0/44 = 0.0%).
- The threshold avoids penalising agents for occasional exploration-related invalid commands.

---

## Self-Test Results

Output from `python3 trace_evaluator.py` against all 12 probing trajectories:

```
Trajectory                                                               Task Type                            Score  Won Penalty #UT_pass
-----------------------------------------------------------------------------------------------------------------------------------------
look_at_obj_in_light_1_trial_T20190909_044715_250790_look_at_obj_in_light-AlarmClock-None-DeskLamp-323.json look_at_obj_in_light                 0.200    N       N 1/ 4
look_at_obj_in_light_2_trial_T20190907_150418_457594_look_at_obj_in_light-Vase-None-DeskLamp-303.json look_at_obj_in_light                 0.000    N       N 0/ 4
pick_and_place_simple_1_trial_T20190908_050633_745514_pick_and_place_simple-Book-None-SideTable-329.json pick_and_place_simple                0.350    N       N 2/ 4
pick_and_place_simple_2_trial_T20190907_200154_378982_pick_and_place_simple-WineBottle-None-Shelf-7.json pick_and_place_simple                0.200    N       N 1/ 4
pick_clean_then_place_in_recep_1_trial_T20190909_105559_983897_pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8.json pick_clean_then_place_in_recep       0.200    N       N 1/ 5
pick_clean_then_place_in_recep_2_trial_T20190909_012550_586494_pick_clean_then_place_in_recep-Tomato-None-CounterTop-25.json pick_clean_then_place_in_recep       0.000    N       N 0/ 5
pick_cool_then_place_in_recep_1_trial_T20190909_044933_815840_pick_cool_then_place_in_recep-Apple-None-CounterTop-14.json pick_cool_then_place_in_recep        0.000    N       N 0/ 5
pick_cool_then_place_in_recep_2_trial_T20190908_024426_412044_pick_cool_then_place_in_recep-WineBottle-None-DiningTable-17.json pick_cool_then_place_in_recep        0.200    N       N 1/ 5
pick_heat_then_place_in_recep_1_trial_T20190907_060234_011675_pick_heat_then_place_in_recep-Apple-None-DiningTable-26.json pick_heat_then_place_in_recep        0.200    N       N 1/ 5
pick_heat_then_place_in_recep_2_trial_T20190908_033721_967359_pick_heat_then_place_in_recep-Tomato-None-Fridge-24.json pick_heat_then_place_in_recep        0.000    N       N 0/ 5
pick_two_obj_and_place_1_trial_T20190907_165826_194855_pick_two_obj_and_place-AlarmClock-None-Dresser-305.json pick_two_obj_and_place               0.100    N       N 1/ 5
pick_two_obj_and_place_2_trial_T20190907_182211_592010_pick_two_obj_and_place-Watch-None-Dresser-205.json pick_two_obj_and_place               0.100    N       N 1/ 5
```

**Per-test breakdown (first trajectory of each task type):**

```
  [look_at_obj_in_light] — look_at_obj_in_light_1_...AlarmClock-None-DeskLamp-323.json
    UT_LOIL_01      FAIL  w=0.25  contrib=0.000  never took 'alarmclock'
    UT_LOIL_02      PASS  w=0.20  contrib=0.200  used 'desklamp' at step 39
    UT_LOIL_03      FAIL  w=0.35  contrib=0.000  no 'use desklamp' step found while holding 'alarmclock'
    UT_LOIL_04      FAIL  w=0.20  contrib=0.000  28/44 actions invalid (63.6%); threshold 30%

  [pick_and_place_simple] — pick_and_place_simple_1_...Book-None-SideTable-329.json
    UT_PAS_01       FAIL  w=0.30  contrib=0.000  never took 'book'
    UT_PAS_02       PASS  w=0.20  contrib=0.200  at 'sidetable' at step 0
    UT_PAS_03       FAIL  w=0.35  contrib=0.000  never placed 'book' at 'sidetable'
    UT_PAS_04       PASS  w=0.15  contrib=0.150  0/44 actions invalid (0.0%); threshold 30%

  [pick_clean_then_place_in_recep] — ...ButterKnife-None-CounterTop-8.json
    UT_PCLEAN_01    PASS  w=0.20  contrib=0.200  took 'butterknife' at step 2
    UT_PCLEAN_02    FAIL  w=0.25  contrib=0.000  never cleaned 'butterknife'
    UT_PCLEAN_03    FAIL  w=0.30  contrib=0.000  never placed 'butterknife' at 'countertop'
    UT_PCLEAN_04    FAIL  w=0.10  contrib=0.000  missing milestone(s): pickup=2, clean=-1, place=-1
    UT_PCLEAN_05    FAIL  w=0.15  contrib=0.000  31/44 actions invalid (70.5%); threshold 30%

  [pick_cool_then_place_in_recep] — ...Apple-None-CounterTop-14.json
    UT_PCOOL_01     FAIL  w=0.20  contrib=0.000  never took 'apple'
    UT_PCOOL_02     FAIL  w=0.25  contrib=0.000  never cooled 'apple'
    UT_PCOOL_03     FAIL  w=0.30  contrib=0.000  never placed 'apple' at 'countertop'
    UT_PCOOL_04     FAIL  w=0.10  contrib=0.000  missing milestone(s): pickup=-1, cool=-1, place=-1
    UT_PCOOL_05     FAIL  w=0.15  contrib=0.000  23/44 actions invalid (52.3%); threshold 30%

  [pick_heat_then_place_in_recep] — ...Apple-None-DiningTable-26.json
    UT_PHEAT_01     PASS  w=0.20  contrib=0.200  took 'apple' at step 2
    UT_PHEAT_02     FAIL  w=0.25  contrib=0.000  never heated 'apple'
    UT_PHEAT_03     FAIL  w=0.30  contrib=0.000  never placed 'apple' at 'diningtable'
    UT_PHEAT_04     FAIL  w=0.10  contrib=0.000  missing milestone(s): pickup=2, heat=-1, place=-1
    UT_PHEAT_05     FAIL  w=0.15  contrib=0.000  27/44 actions invalid (61.4%); threshold 30%

  [pick_two_obj_and_place] — ...AlarmClock-None-Dresser-305.json
    UT_PTWO_01      FAIL  w=0.20  contrib=0.000  picked up 'alarmclock' 0 time(s)
    UT_PTWO_02      FAIL  w=0.25  contrib=0.000  placed 'alarmclock' at 'dresser' 0 time(s)
    UT_PTWO_03      FAIL  w=0.20  contrib=0.000  picked up 'alarmclock' 0 time(s); need >= 2
    UT_PTWO_04      FAIL  w=0.25  contrib=0.000  placed 'alarmclock' at 'dresser' 0 time(s); need >= 2
    UT_PTWO_05      PASS  w=0.10  contrib=0.100  8/44 actions invalid (18.2%); threshold 30%
```

**Self-test interpretation:** All 12 probing trajectories are failing episodes (won=False, done=False at step 44). The partial scores (0.0–0.35) correctly reflect partial task progress: the pick_and_place_simple agent visited a sidetable and had 0 invalid actions but never picked up a book; the butterknife agent picked up the correct object but stalled before cleaning. The scores are strictly deterministic: re-running produces identical output.

---

## Limitations and Deferred Tests

### Known limitations affecting scorer accuracy

| # | Situation | Effect |
|---|---|---|
| 1 | `game_file` absent from trajectory metadata | All traj_data-dependent UTs evaluate to False with "unavailable" detail; scorer still runs |
| 2 | `traj_data.json` file moved or deleted | Same effect as above; scorer logs path in limitations_hit |
| 3 | `holding`/`inventory_state` fields always empty in probing recordings | Cannot use these fields; held-object inferred from admissible_commands heuristic instead |
| 4 | Trajectory recording truncated before `done=True` | Failure-penalty not applied even if step limit was exceeded |
| 5 | Future grammar changes to action verbs | Action-verb predicates (`take`, `move`, `clean`, `heat`, `cool`, `use`) would silently misfire |
| 6 | `pick_two_obj_and_place`: pick-up counting | Re-picking the same instance counted twice; PDDL facts would enable exact instance-ID tracking |
| 7 | `look_at_obj_in_light` UT_LOIL_03 false positive | If admissible_commands lists `move <obj>` for an unrelated reason while lamp is used, test passes incorrectly |

### Tests considered but rejected

| Proposed test | Reason rejected |
|---|---|
| **`pddl_facts_state`-based sub-goal detection** (e.g., `isclean(obj)`, `ishot(obj)`) | `facts` not recorded in probing trajectories; oracle inaccessible without `request_infos.facts=True`; deferred to scorer v2 |
| **Step-count efficiency** (e.g., "completed in ≤ N steps") | Minimum-step baseline per game is unavailable without either `policy_commands` or `traj_data.plan.high_pddl`; would require running the PDDL planner at score time |
| **Receptacle correctness before placement** (e.g., "visited sinkbasin before cleaning") | Would require tracking location changes relative to sinkbasin entity names, which vary by game; overly fragile without facts oracle |
| **Repeated revisiting penalty** (e.g., "visited same receptacle >3 times") | Reasonable efficiency signal but threshold calibration requires data from successful episodes; deferred |
| **`expert_plan_handcoded` trajectory similarity** | Expert plan not recorded in probing trajectories; would reveal solution path at score time if feed forward |
| **`goal_condition_success_rate_thor`** | Only available in ThorEnv visual path; not applicable to text-only ALFWorld benchmark |
| **Object-type confusion penalty** (agent picks wrong object type) | Requires knowledge of wrong vs. right object; only available if `traj_data_task_annotation` is loaded; a test could be added in scorer v2 once the params loading is confirmed stable |
