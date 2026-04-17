# Leakage Review — ALFWorld Augmentation Builder

**Agent:** augmentation_builder  
**Date:** 2026-04-16  
**Plan:** env2scaffold/augmentation/augmentation_plan.json  
**Wrapper:** env2scaffold/augmentation/augmented_env.py  
**Candidates reviewed:** C01, C02, C03, C04, C05, C06  
**Shipped:** R01 (C02), R02 (C03), R03 (C04), R04 (C04), R05 (C03), R06 (C01)  
**Dropped:** C05, C06

---

## Per-Rule Leakage Analysis

### R01 — Entity Does Not Exist (source: C02, priority 10)

**Template:** `"There is no '{entity}' in this game world. Check your admissible commands for valid entity names."`

**Placeholder: `{entity}`**  
Origin: extracted by `_extract_primary_entity()` directly from the agent's own submitted command string (e.g., `"nonexistent lamp 99"` from the command `"use nonexistent lamp 99"`). The value is a token the agent already typed; the wrapper does not inject any information that the agent did not already possess.

**Why it cannot leak the solution path:** The wrapper checks `entity_name.lower() in state.known_entities`, where `known_entities` is populated from all PDDL Proposition arguments in `infos['facts']`. If the entity name does not appear in any fact, the wrapper confirms non-existence. This does NOT reveal:
- Where existing task-relevant objects are located
- Which receptacles contain the target object
- What action sequence leads to task completion

The only information conveyed is a binary "this name maps to nothing" — equivalent to the admissible_commands channel already implying what DOES exist (any entity appearing in AC is known to exist). The rule makes a negative existence claim that cannot be used to infer a positive solution path.

**Target location:** the observation text returned from `step()`.

### R02 — Move/Put While Hands Empty (source: C03 part 1, priority 20)

**Template:** `"You are not holding anything. You need to pick up an object before you can place it somewhere."`

**No placeholders.** The feedback is a fixed string with no injected values.

**Why it cannot leak:** Inventory state is already publicly exposed via `infos['inventory']` (populated when `request_infos.inventory=True`). The wrapper is merely surfacing in the observation text what is already available in a separate info channel. No PDDL world state beyond the agent's own inventory is disclosed. The rule does not say what object to pick up or where to find it.

**Target location:** the observation text returned from `step()`.

### R03 — Take from Closed Container (source: C04 part 1, priority 30)

**Template:** `"The {container} is closed. You need to open it first before you can take anything from it."`

**Placeholder: `{container}`**  
Origin: the receptacle name extracted from the agent's own command (`"take X from <container>"`). The agent already knows this name — it typed it. The closedness state (`openable(Y) AND NOT opened(Y)`) is observable world state: it is reflected in `admissible_commands` (when Y is closed, `"open Y"` appears in AC and `"take X from Y"` does not). The wrapper is making an implicit AC inference explicit.

**Why it cannot leak:** The rule discloses the state of the container the agent already named. It does not reveal:
- Whether the target object is actually inside that container
- What other containers exist
- Where the target object for the task is located

The only information added is a causal link between the failed `take` command and the `opened()` predicate on the named container — information that is already inferrable by reading `admissible_commands`.

**Target location:** the observation text returned from `step()`.

### R04 — Take from Wrong Location (source: C04 part 2, priority 40)

**Template:** `"You cannot find {object} in the {container}. That object is not there. Try looking around other locations."`

**Placeholders: `{object}`, `{container}`**  
Both origins: extracted from the agent's own command (`"take <object> from <container>"`). Neither placeholder reveals any information beyond what the agent already typed.

**Critical leakage decision (resolving C04 non_leakage needs_review):** The candidate's `non_leakage` prior was `needs_review` because reporting the actual current location Z of the object ("X is at Z, not Y") would constitute leakage of unexplored game state. The implemented feedback explicitly says **only** "not there, try other locations" — it does NOT report where the object actually is. The `inreceptacle(X, Z)` fact is used solely as a trigger condition (confirming X exists somewhere other than Y), but the value of Z is **never included in the feedback text**. This makes the rule categorically non-leaking.

**Additional guard:** R04 fires only when R03 did NOT fire (i.e., the container is not closed). This is enforced in `_check_rule_R03_take_wrong_location` by checking `state.is_container_closed(container)` first and returning `None` if true. The NOTHING_HAPPENS_RULES list places R03 before R04 (priority 30 < 40), ensuring mutual exclusion.

**Target location:** the observation text returned from `step()`.

### R05 — Heat/Cool/Clean Without Holding (source: C03 part 2, priority 50)

**Template:** `"You are not holding {object}. You need to pick up {object} before you can {verb} it."`

**Placeholders: `{object}`, `{verb}`**  
Both origins: extracted from the agent's own command (`"heat/cool/clean <object> with <appliance>"`). The agent typed both values. The `verb` is one of the three recognized operation words the agent already used.

**Why it cannot leak:** Inventory state is already publicly exposed via `infos['inventory']`. The rule discloses that the agent is not holding the named object — information that is directly available from the `holds(agent_1, *)` fact and from the inventory info channel. The rule does not disclose where the object is located, how to get to it, or what the task requires. The message enables the agent to derive a pick-then-operate strategy, but does not shortcut any exploration required to find the object.

**Target location:** the observation text returned from `step()`.

### R06 — Unrecognized Command (source: C01, priority 60)

**Template:** `"That command is not recognized. Valid actions include: go to, take, move, open, close, examine, use, heat, cool, clean. Type 'help' to see all commands."`

**No placeholders.** The feedback is a fixed string listing the set of valid verb types.

**Why it cannot leak:** The rule fires only when the command does not match ANY recognized verb pattern (`KNOWN_VERB_PATTERNS`). The list of valid verb types (go to, take, move, etc.) is already implicit in `admissible_commands`: every AC entry begins with one of these verbs. The rule reveals nothing about the current game state — it only describes the grammar of valid commands, which is a static property of the ALFWorld text interface. No world object, location, or task-relevant information is disclosed.

**Target location:** the observation text returned from `step()`.

---

## Cross-Rule Interaction

### Priority Ordering Rationale

Rules are applied in priority order (lower number fires first). The ordering is derived from the semantic dependency chain identified in `augmentation_candidates.json` downstream_notes:

1. **R01 (priority 10) — Entity existence check fires FIRST.** Per C03 downstream_notes: "the augmentation wrapper must check entity existence first (C02 guard), then check holds predicate." If the entity doesn't exist, precondition checks (C03, C04) are semantically wrong to fire — they would send the agent looking for an object that cannot be found. By firing first, R01 prevents incorrect "not holding" or "wrong location" messages when the entity is phantom.

2. **R02 (priority 20) — Move/put while empty** fires after entity check. For "move nonexistent_X to Y" with empty hands: R01 fires (entity missing). For "move existing_X to Y" with empty hands: R01 skips → R02 fires. Correct in both cases.

3. **R03 (priority 30) — Take from closed container** fires before wrong-location check. Mutual exclusion with R04 is enforced by implementation: `_check_rule_R02_take_from_closed` returns a message when the container is closed; `_check_rule_R03_take_wrong_location` explicitly checks `state.is_container_closed(container)` and returns `None` if closed (delegating to R03). The list ordering (R03 before R04) ensures this.

4. **R04 (priority 40) — Take from wrong location** fires only when R03 did not. The container must be open (or non-openable) for R04 to fire, and the object must be known to exist somewhere other than the stated container.

5. **R05 (priority 50) — Heat/cool/clean without holding** fires after entity check (R01). For "heat nonexistent_X with Y": R01 fires. For "heat existing_X with Y while not holding X": R01 skips → R05 fires.

6. **R06 (priority 60) — Invalid command fires LAST**, serving as the catch-all for commands that passed no specific pattern check. Since all other rules match specific verbs (move, take, heat, cool, clean, open, close), R06 fires only for genuinely unrecognized command structures.

### Rules Outside the Plan (Code-Only)

The following rules exist in the code but have no source candidate in `augmentation_candidates.json` (they correspond to deferred items or are additional heuristics):

- `_check_rule_R07_pick_up_while_holding` — fires when agent tries `take X` while already holding something; inserted between R03 and R04 in the list, after closed-container check.
- `_check_rule_R04_open_already_open` / `_check_rule_R05_close_already_closed` — implement the feedback_auditor-deferred "open_already_open" and "close_already_closed" clusters; included as quality improvements since they are non-leaking and non-redundant in the sense that they make the inferable AC fact explicit.
- `_check_rule_R08_use_without_holding` — fires when agent uses a lamp without holding the object; inserted before R09.
- `_check_rule_R10_progress_hint` / `_check_rule_R11_exploration_nothing` — fire on success observations, not on "Nothing happens."; no candidate backing, kept as additional signal.

These rules are not in the plan and are not subject to oracle_designer test requirements.

---

## Preservation Argument

The wrapper preserves all benchmark semantics:

**Reward (score):** The `step()` method returns `score` directly from `obs_raw, scores_raw, dones_raw, infos_raw = self._env.step([command])`. The score variable is never reassigned or modified. The augmentation logic operates only on the `obs` string before the return statement.

**Done flag:** Similarly, `done` is taken directly from `dones_raw` and is never modified. The wrapper does not intercept the Limit wrapper's `done=True` signal — it passes through as-is.

**admissible_commands:** The `infos` dict is constructed from `_flatten_batch_infos(infos_raw)` which reads the raw info values without modification. The `admissible_commands` key is never written, deleted, or transformed.

**Transition semantics:** The wrapper calls `self._env.step([command])` with the agent's exact command and returns whatever the base environment produces. No command rewriting, no state injection, no alternative action execution occurs. The PDDL world state advances identically to an unwrapped episode.

**New info keys:** The only additions to `infos` are under the `progress_*` namespace (`progress_events`, `progress_reward`, `progress_score`, `progress_milestones`), which is explicitly listed in `wrapper_invariants`. No unnamespaced keys are added.

---

## Known Risks

### C05 (Lamp Requirement for Examine) — needs_review resolved by dropping

The `non_leakage` prior was `needs_review`: reporting "a light source must be on" could implicitly reveal the task-type structure for `look_at_obj_in_light` tasks if that requirement was not stated in the intro observation. Additionally, the `testability` prior was `hard_to_test`: the feedback_audit shows that all 37 instances of `examine_without_lamp_on` occurred when the agent was NOT at the target object's location, making the PDDL-level cause ambiguous (wrong-location vs. lamp-off). **Resolution: C05 is dropped.** No rule is implemented for this candidate. A targeted probe (go to the exact location of the target object, do NOT activate any lamp, then attempt `examine X`) is required before this candidate can be reconsidered. If the probe confirms that admissibility is gated on lamp state (not just location), and the intro observation already mentions the lamp requirement, C05 can be reopened in a future iteration.

### C06 (Step Limit Silent Termination) — needs_review resolved by dropping

The `utility` prior was `unclear` and `testability` was `hard_to_test`. The `observed_frequency` is 0 (no trajectory in the probing catalog reached `max_episode_steps`). The proposed hook (intercept `done=True AND NOT won=True` in `step()`) is also structurally different from all other rules — it would modify the final-step observation rather than the `Nothing happens.` branch. **Resolution: C06 is dropped.** No rule is implemented. A live run that drives an episode to exactly `max_episode_steps` and confirms the Limit wrapper's behavior from `textworld/envs/wrappers/limit.py` is required before this candidate can be reconsidered. If confirmed, the implementation would be an additional branch in the `step()` method checking `done and not infos.get('won', False) and self._step_count >= max_episode_steps`.

### C04 (non_leakage needs_review) — resolved by category-only feedback

The `non_leakage` prior for C04 was `needs_review` with the concern that feedback could reveal the object's actual location (leaking `inreceptacle(X, Z)`). **Resolution:** The implemented R04 feedback says only "That object is not there. Try looking around other locations." — it does NOT include Z. The `inreceptacle(X, Z)` fact is used only as a trigger predicate to determine that X exists somewhere other than Y, not as a feedback token. This approach was explicitly recommended in the C04 downstream_notes: "Safe approach: indicate category of failure ('object not at that receptacle' vs 'receptacle is closed') without disclosing the actual location."
