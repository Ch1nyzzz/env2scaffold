# Feedback Audit — ALFWorld (alfworld 0.4.2 / textworld 1.7.0)

**Auditor:** feedback_auditor  
**Date:** 2026-04-16  
**Inputs:** benchmark_spec.json, benchmark_analysis.md, feedback_catalog.json, 12 trajectories (540 steps total, 7,957 error probes)

---

## Catalogue Summary

The probing agent collected feedback across 12 games spanning all 6 ALFWorld task types (2 games per type), with 45 steps per game and a comprehensive error-probe suite injected at each step.

| Metric | Value |
|--------|-------|
| Total distinct feedback strings | 344 |
| Feedback strings with > 1 distinct cause | **1** |
| Total ambiguity clusters identified | **6** |
| Error probes total | 7,957 |
| Occurrences of "Nothing happens." | 7,957 (100% of error probes) |
| Successful action feedback strings | 343 |
| Successful action feedback strings with > 1 cause | 0 |

**Coverage estimate: 0.93.** All 6 ALFWorld task types are represented. The 12-game sample covers all six feedback-generation points documented in the spec for the failure path. The single gap is the absence of clean/heat/cool/slice transformation success feedback (none of the 12 trajectories completed the transformation step in 45 moves), leaving estimated 7% of the feedback space unobserved. The success-feedback space (343 strings) is almost entirely unambiguous.

**Key finding:** The single hardcoded string `"Nothing happens."` (source: `textworld/envs/pddl/pddl.py:162`) is the sole multi-cause feedback in the entire ALFWorld text environment. It accounts for 15 distinct underlying causes and 7,957 of the 8,497 observed feedback events (93.7%). Every augmentation candidate below is a cluster within this single string.

---

## Ambiguity Clusters

### Cluster A (→ C01): Syntactically Unrecognized Command

**Shared text:** `Nothing happens.`  
**Distinct underlying cause:** 1 (`completely_invalid_command`)  
**Frequency:** 2,160  
**Cause detail:** The command string contains a verb or structure that does not correspond to any ALFWorld grammar rule (e.g., `"fly to mars"`, `"summon chair"`). The PDDL engine's `list.index()` call raises `ValueError` immediately because no `_valid_commands` entry matches.

**Why this is a cluster:** Although there is technically one cause, it is ambiguous with all other "Nothing happens." causes from the agent's perspective. An agent that submits a nonsensical command and gets `"Nothing happens."` cannot distinguish this from inventory-empty (C03) or entity-nonexistent (C02) failures without additional reasoning over the command syntax. This cluster is the highest-volume single cause.

**Trajectory evidence:**  
- `look_at_obj_in_light_1_...json#step=0`: probe action `"fly to mars"` → `"Nothing happens."`, `was_admissible=false`  
- `look_at_obj_in_light_2_...json#step=0`: same pattern in a different game  
- `pick_and_place_simple_1_...json#step=0`: nonsensical commands in simple-task context

---

### Cluster B (→ C02): Valid Syntax, Entity Does Not Exist

**Shared text:** `Nothing happens.`  
**Distinct underlying causes:** 5 (`take_nonexistent_object`, `open_nonexistent_container`, `go_to_nonexistent_location`, `use_nonexistent_lamp`, `put_nonexistent_object`)  
**Frequency:** 2,700 (540 per sub-cause)  
**Cause detail:** The command uses a recognized verb and correct syntactic structure, but the entity name (object, receptacle, location, or lamp) does not appear anywhere in the game world. Example: `"take watermelon 1 from floor"` — watermelon 1 does not exist in the game.

**Why these are clustered together:** All five sub-causes share the same diagnostic: the entity is absent from `pddl_facts` regardless of game state. They require the same recovery insight: the named entity cannot be found via exploration because it does not exist.

**Distinguishability from Cluster A:** The command verb IS a valid verb (take, open, go to, use, put), so the failure is semantic (entity lookup) rather than syntactic (verb parse). Separating A and B requires the wrapper to first validate the verb, then check entity existence.

**Trajectory evidence:**  
- `look_at_obj_in_light_1_...json#step=0`: `"take watermelon 1 from floor"` → `"Nothing happens."`, `was_admissible=false`  
- `look_at_obj_in_light_2_...json#step=0`: `"go to nonexistent 99"` → `"Nothing happens."`  
- `pick_and_place_simple_1_...json#step=0`: `"open nonexistent 99"` → `"Nothing happens."`

---

### Cluster C (→ C03): Precondition Failure — Not Holding Required Object

**Shared text:** `Nothing happens.`  
**Distinct underlying causes:** 4 (`put_while_empty_hands`, `clean_without_holding`, `heat_without_holding`, `cool_without_holding`)  
**Frequency:** 2,159 (540 + 540 + 540 + 539)  
**Cause detail:** The agent attempts an action that requires holding the target object (`put X in Y`, `clean X with Y`, `heat X with Y`, `cool X with Y`) but the agent's inventory does not contain `X`. The PDDL precondition `holds(agent_1, X)` is not satisfied.

**Why these are clustered together:** All four sub-causes share the PDDL precondition `holds(agent_1, X)` being false. The recovery in all cases is: find and pick up the target object first. The `pddl_facts` channel exposes `holds(agent_1, <obj>)` predicates that directly discriminate this cause.

**Interaction with C02:** If `X` does not exist (C02) AND the agent tries `put X / clean X / heat X / cool X`, C02 fires first (entity nonexistent). C03 only applies when the entity exists but is not held. The augmentation wrapper must check C02 first.

**Trajectory evidence:**  
- `look_at_obj_in_light_1_...json#step=0`: `"put bed 1 in/on bed 1"` → `"Nothing happens."`, context: `inventory_state: empty`, cause: `put_while_empty_hands`  
- `look_at_obj_in_light_2_...json#step=0`: `"heat bed 1 with microwave 1"` → `"Nothing happens."`, context: `holding: ""`, cause: `heat_without_holding`  
- `pick_and_place_simple_1_...json#step=0`: `"clean bed 1 with sinkbasin 1"` → `"Nothing happens."`, cause: `clean_without_holding`

---

### Cluster D (→ C04): Spatial / Accessibility Failure on Take

**Shared text:** `Nothing happens.`  
**Distinct underlying causes:** 2 (`take_from_wrong_location`, `take_from_closed_container`)  
**Frequency:** 656 (540 + 116)  
**Cause detail:**  
- `take_from_wrong_location` (540): Object exists in the game but is NOT at the receptacle named in the command. `inreceptacle(X, Y)` is false for the stated `Y`.  
- `take_from_closed_container` (116): Object exists and IS at the receptacle, but the receptacle is currently closed (`opened(Y)` is false in `pddl_facts`).

**Why these are clustered together:** Both result from the same command form (`take X from Y`) failing, but require completely different recovery actions. Recovery for wrong-location: explore to find X's actual location. Recovery for closed container: go to Y and open it first.

**Partial redundancy note:** For `take_from_closed_container`, after the failed command, `admissible_commands` will include `"open Y"` but NOT `"take X from Y"`. An attentive agent could infer the container-is-closed cause by scanning AC. For `take_from_wrong_location`, `admissible_commands` may include `"take X from Z"` where Z is the actual location—revealing X's true location by scanning AC. This partial redundancy is noted but multi-step AC reasoning is non-trivial for the agent to perform reliably.

**Leakage risk:** Reporting actual location of X ("object is at Z, not Y") would leak game world state the agent hasn't explored. Safe feedback indicates failure category only.

**Trajectory evidence:**  
- `look_at_obj_in_light_1_...json#step=5`: `"take drawer 1 from drawer 1"` → `"Nothing happens."`, cause: `take_from_closed_container`  
- `look_at_obj_in_light_2_...json#step=5`: `"take drawer 1 from drawer 1"` → `"Nothing happens."`, same cause  
- `pick_and_place_simple_1_...json#step=6`: `"take bed 1 from sidetable 2"` → `"Nothing happens."`, cause: `take_from_wrong_location`

---

### Cluster E (→ C05): Task-Specific Precondition — Lamp Not Active

**Shared text:** `Nothing happens.`  
**Distinct underlying causes:** 1 (`examine_without_lamp_on`)  
**Frequency:** 37  
**Cause detail:** Observed exclusively in `look_at_obj_in_light` task games. The probing agent labeled these failures as `examine_without_lamp_on` based on task context (no lamp had been activated). All 37 probes have `was_admissible=false`, meaning `examine X` was not in `admissible_commands` at the time. The PDDL-level reason for non-admissibility—whether it is specifically gated on `toggled(desklamp)` or is a generic entity-not-accessible cause—is **unresolved** (see open_question #3 in spec and Cross-Check section below).

**Important caveat:** Step-level inspection of the lamp probes shows that at probe time, the agent was typically NOT at the target object's location. For example, the step=43 probe tried `"examine laptop 1"` while `current_location` was `"bed 1"` but laptop 1 was not at bed 1. This is consistent with a generic wrong-location failure rather than a lamp-specific gate. The `examine_without_lamp_on` label is the probing agent's semantic classification, not a confirmed PDDL-level cause.

**Why included:** Despite the uncertainty, this is a real failure pattern unique to one task type. If the PDDL grammar for `look_at_obj_in_light` does gate examine-admissibility on lamp state, this candidate addresses a genuine missing diagnostic. The candidate is marked `hard_to_test` until the PDDL mechanism is confirmed.

**Trajectory evidence:**  
- `look_at_obj_in_light_1_...json#step=0`: `"examine dresser 1"` → `"Nothing happens."`, `was_admissible=false`, cause: `examine_without_lamp_on`  
- `look_at_obj_in_light_2_...json#step=0`: `"examine dresser 1"` → `"Nothing happens."`, same cause

---

### Cluster F (→ C06): Silent Episode Termination (Step Limit)

**Shared text:** `<absent>` — no special feedback emitted  
**Distinct underlying causes:** 1 (`step_limit_silent_termination`)  
**Frequency:** 0 directly observed (inferred from spec, see below)  
**Cause detail:** Per `benchmark_spec.json::done_signature` and `benchmark_analysis.md`, the `Limit` wrapper (`textworld/envs/wrappers/limit.py`) sets `done=True` after `max_episode_steps` without modifying the observation text. The agent's final observation is identical to any regular-step observation. This is a missing signal: the agent cannot distinguish terminal state caused by step exhaustion from any mid-episode step.

**Coverage gap note:** All 12 probed trajectories used a 45-step cutoff that did not reach the Limit wrapper's threshold (the spec's example uses `max_episode_steps=50`). No step in the trajectories is a confirmed Limit-wrapper termination. The closest evidence is `step=44` of any trajectory — the final step before probing stopped — which shows a normal action feedback with `won=False` and `done=False` in the trajectory context (probing stopped, not the Limit wrapper).

---

## Candidate Index

| candidate_id | kind | one-line intent |
|---|---|---|
| C01 | `disambiguate_failure` | Indicate that the command verb/structure is syntactically unrecognized by the ALFWorld grammar |
| C02 | `disambiguate_failure` | Indicate that the entity named in the command does not exist anywhere in the current game world |
| C03 | `disambiguate_failure` | Indicate that the action requires holding the target object first, and current inventory lacks it |
| C04 | `disambiguate_failure` | Distinguish "object not at that receptacle" from "receptacle is closed" for failed take commands |
| C05 | `disambiguate_failure` | For `look_at_obj_in_light` tasks: indicate that an active light source is required for examine to succeed |
| C06 | `emit_missing_signal` | Emit a distinct terminal observation when the episode ends due to step-limit exhaustion rather than task success |

---

## Redundancy and Deferrals

### Deferred: `open_already_open` + `close_already_closed` (245 occurrences)

**Reason for deferral:** Both causes are **reliably inferable from `admissible_commands`** at the step where the failure occurs:

- Agent sends `"open X"` → `"Nothing happens."` → checks updated `admissible_commands`:
  - If `"close X"` appears in AC: X is already open (can be closed)
  - If neither `"open X"` nor `"close X"` appears in AC: X does not exist (→ C02)
  
- Agent sends `"close X"` → `"Nothing happens."` → checks updated `admissible_commands`:
  - If `"open X"` appears in AC: X is already closed (can be opened)
  - Combined with "X mentioned elsewhere in AC": X exists but is in the desired-closed state already

This two-step inference is deterministic given admissible_commands and requires no pddl_facts access. Augmenting this case would be **redundant** with already-exposed public context. The 245 occurrences are noted but not made into candidates.

### Deferred: Clean/Heat/Cool/Toggle Success Feedback Observed Gap

**Reason for deferral:** The probing catalog contains no instances of `"You clean X with Y."`, `"You heat X with Y."`, `"You cool X with Y."`, or similar transformation-success feedback. All 12 probed trajectories failed to complete the transformation step within 45 moves. It is therefore unknown whether:

1. The grammar produces informative feedback confirming the state change (e.g., confirms `isclean(X)` updated)
2. The feedback is generic and does not confirm the new PDDL state

This cannot be assessed without live execution. Not made into a candidate; added to Cross-Check open questions.

### Deferred: GotoLocation and Other Success Feedback Strings

All 343 success feedback strings have exactly 1 cause. GotoLocation feedback already includes room name and visible objects (e.g., `"You arrive at sinkbasin 1. On the sinkbasin 1, you see a fork 2."`), which is informative. PickupObject and PutObject feedback include object and receptacle names. No `strengthen_success` candidates are warranted from observed data.

---

## Cross-Check Against Spec

### Claims Corroborated

| Spec Claim | Evidence |
|---|---|
| `"Nothing happens."` is the single hardcoded failure feedback string | All 7,957 error probes produced exactly `"Nothing happens."` |
| `admissible_commands` enumerates all PDDL-executable actions per step | Every error probe has `was_admissible=false`; successful steps always used commands from admissible_commands |
| GotoLocation feedback is grammar-derived and includes room description | 161 distinct GotoLocation strings observed; each includes `"You arrive at X. On/In the X, you see..."` |
| ToggleObject success produces distinct feedback | `"You turn on the desklamp 1."` and `"You turn on the desklamp 2."` observed in trajectories |
| `done=True` only fires on `won=True` or step-limit; `lost` always False | No trajectory observed `done=True` with `won=False` except by step-limit cutoff in probing |

### Claims Not Corroborated (Coverage Gaps)

1. **Clean/Heat/Cool/Slice success feedback grammar** — No instance of any transformation success was observed in 12 trajectories × 45 steps. The grammar rules for `CleanObject.feedback`, `HeatObject.feedback`, `CoolObject.feedback` remain unverified. Spec open_question #3 explicitly flags this. Recommend a targeted probe: register a game, pick up the target object, go to the appropriate appliance, execute the transformation, and record the observation.

2. **examine success feedback for look_at_obj_in_light after lamp-on** — No trajectory successfully turned on a lamp AND subsequently executed an examine command. The single trajectory that turned on a lamp (`look_at_obj_in_light_1`, step 39) did not attempt to examine the target object afterwards. It is unknown whether `"examine X"` becomes admissible after `"use desklamp N"` succeeds, or whether admissibility requires being at the object's location regardless of lamp state.

3. **Limit wrapper silent termination behavior** — No trajectory reached `max_episode_steps`. Spec open_question #5 directly asks whether the Limit wrapper changes the observation on the terminating step. Not confirmed.

4. **`state['lost']` ever becoming True** — No trajectory observed `done=True` with `won=False` via the game engine (as opposed to probing cutoff). Spec open_question #1 notes this is hardcoded `False`. Consistent with observations but not positively confirmed via a game that enters a losing state.

5. **Walkthrough field population** — Spec open_question #4 asks whether any game files contain pre-computed walkthroughs vs. requiring re-planning. Not verifiable from trajectory data alone; requires inspecting `game.tw-pddl` files directly.

### New Open Questions Raised by This Audit

1. **Is the `examine_without_lamp_on` PDDL cause distinct from `take_from_wrong_location`?** All 37 `examine_without_lamp_on` probes had `was_admissible=false`. Multiple probes occurred when the agent was not at the target object's location, consistent with a generic wrong-location failure. A targeted probe (go to object's exact location before lamp is on, then try examine) is needed to confirm whether lamp state gates examine-admissibility in the PDDL model.

2. **Does the examine grammar produce different feedback for the successful look_at_obj_in_light final step vs. a normal examine?** If the grammar rule for the task-completing examine produces a richer string (e.g., `"You examine the alarmclock 1 under the desklamp 2. It appears [color/property]."`) rather than `"There's nothing special about alarmclock 1."`, this would be a significant signal for the agent. This was not observed in the probing data.

3. **Is the `completely_invalid_command` cause truly structurally distinct from entity-nonexistent at the PDDL `list.index()` level?** Both result in `ValueError` from `list.index()`. The labeling is by the probing agent's classification heuristic, not a PDDL-level discriminant. Augmentation_builder must verify that a verb-level check is both possible and meaningful before implementing C01 separately from C02.
