# Verification Report — AugmentedAlfWorldEnv

Generated: 2026-04-16  
Wrapper: `env2scaffold/augmentation/augmented_env.py`  
Plans: `env2scaffold/oracle_test/unit_test_plan.json` · `oracle_plan.json` · `augmentation_plan.json`

---

## Run Summary

**Overall verdict: PASS** — all six shipped rules VALIDATED.

| Layer | Tests | Pass | Fail | Error | Skipped |
|-------|-------|------|------|-------|---------|
| Layer 1 — Benchmark Native | 3 | 3 | 0 | 0 | 0 |
| Layer 2 — Diagnostic Unit | 18 | 17 | 1 | 0 | 0 |
| Layer 3 — Non-Regression | 4 | 4 | 0 | 0 | 0 |
| **Total** | **25** | **24** | **1** | **0** | **0** |

The single fail is `L2_C06_trigger`, which tests a **dropped candidate** (C06) that intentionally has no augmentation in the wrapper. It does not block any shipped rule.

---

## Layer 1: Benchmark-Native Results

**Policy:** `HandCodedTWAgent` (deterministic; no timeouts observed)  
**Episode set:** `valid_seen` split, N=100 matched games  
**Max steps per episode:** 50  
**Method:** A/B — same game files, same policy, only the wrapper changes.

| Test ID | Description | Original | Augmented | Delta | Status |
|---------|-------------|----------|-----------|-------|--------|
| L1_T01 | success\_rate (won=True fraction) | 0.9800 | 0.9800 | 0.0000 | **PASS** |
| L1_T02 | avg\_game\_points (max score/ep) | 0.9800 | 0.9800 | 0.0000 | **PASS** |
| L1_T03 | avg\_steps (steps until done) | 14.97 | 14.97 | 0.00 | **PASS** |

**Pass criteria met:**
- L1_T01: `aug_success_rate (0.98) >= orig_success_rate (0.98)` ✓
- L1_T02: `aug_avg_game_points (0.98) >= orig_avg_game_points (0.98)` ✓
- L1_T03: `|delta| / orig = 0.0% < 5% threshold` ✓ (auxiliary — does not gate pass/fail alone)

**Notes:**
- 98/100 games won by HandCodedTWAgent on both original and augmented envs.
- 2 games exceeded 50 steps on both envs — agent could not solve those games regardless of augmentation.
- Augmented obs text change did not affect HandCodedTWAgent decisions because the agent primarily uses `infos['facts']` (PDDL predicates) for subgoal tracking; the wrapper never modifies facts.

---

## Layer 2: Diagnostic Unit Results

**Oracles used:** `pddl_facts_state` (infos['facts']), `admissible_commands_validity_heuristic`  
**Game files:** pick_and_place_simple, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, look_at_obj_in_light

### Candidate C01 — Invalid Command Verb (→ Rule R06)

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C01_trigger | trigger | **PASS** | `cmd_in_pre_ac=False` (fly to mars not in AC) · `aug_emitted=True` (R09 fired "That command is not recognized…") |
| L2_C01_non_trigger | non_trigger | **PASS** | `was_in_ac=True` (go to bed 1 was admissible) · `c01_fired=False` |
| L2_C01_non_leakage | non_leakage | **PASS** | `leaked=[]` — C01 signal references only command structure, no world-state entity names |

### Candidate C02 — Entity Non-Existence (→ Rule R01)

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C02_trigger | trigger | **PASS** | `entity_fact_count=0` (ladle_nonexistent_99 absent from all PDDL propositions) · `c02_signal=True` ("There is no 'ladle_nonexistent_99'…") |
| L2_C02_non_trigger | non_trigger | **PASS** | `entity='dishsponge 1' fact_count=5` (entity IS in facts) · `c02_fired=False` (C04/fallback fired instead) |
| L2_C02_non_leakage | non_leakage | **PASS** | `leaked=[]` — C02 signal names only the command token, never reveals real entity locations |

### Candidate C03 — Missing Precondition: move/put while empty + heat/cool/clean without holding (→ Rules R02, R05)

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C03_trigger | trigger | **PASS** | `holds_apple=False` · `apple_in_any_fact=True` · `c03_signal=True` ("You are not holding anything. You need to pick up apple 1…") |
| L2_C03_non_trigger | non_trigger | **PASS** | `holds_apple_before=True` (picked up apple 1) · `c03_fired=False` (fallback fired, not C03) |
| L2_C03_non_leakage | non_leakage | **PASS** | `actual_receptacle_leaked=False` · `leaked=[]` — "not holding" signal names only what agent commanded, not where apple 1 is stored |

### Candidate C04 — Take from Closed Container / Wrong Location (→ Rules R03, R04)

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C04_trigger | trigger | **PASS** | `inrec=True` (bread 3 in fridge 1) · `opened=False` · `c04_signal=True` ("The fridge 1 is closed. You need to open it first…") |
| L2_C04_non_trigger | non_trigger | **PASS** | `inrec=True` · `opened=True` (fridge opened before take) · `holds_after=True` (take succeeded) · `c04_fired=False` |
| L2_C04_non_leakage | non_leakage | **PASS** | `actual_recep='microwave 1'` (apple's true location) · `leaked=False` — wrong-location signal does not name the actual receptacle |

### Candidate C05 — Dropped Candidate (examine without lamp on)

> **Note:** C05 was dropped in `augmentation_plan.json` (fails testability and non-leakage rubric).
> Tests in `unit_test_plan.json` are marked speculative. See §Upstream Issues Surfaced.

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C05_trigger | trigger | PASS ⚠ | `toggled=False` · `examine_in_ac=False` · `aug_emitted=True` — **pass by fallback**: wrapper fallback fired ("Nothing happens. The action could not be performed…"), not a C05-specific signal. Criterion permits "any non-empty text beyond Nothing happens." |
| L2_C05_non_trigger | non_trigger | **PASS** | `c05_fired=False` — no C05 augmentation signal exists in wrapper |
| L2_C05_non_leakage | non_leakage | **PASS** | `leaked=[]` — fallback signal has no entity names |

### Candidate C06 — Dropped Candidate (step-limit termination)

> **Note:** C06 was dropped in `augmentation_plan.json` (fails utility and testability rubric).
> Tests in `unit_test_plan.json` are marked speculative.

| Test ID | Kind | Status | Oracle Evidence |
|---------|------|--------|-----------------|
| L2_C06_trigger | trigger | **FAIL** | `done=True` · `won=False` (limit fired correctly) · `step_limit_signal=False` — no step-limit augmentation in wrapper (expected for dropped candidate) |
| L2_C06_non_trigger | non_trigger | **PASS** | `done=False` · `step_limit_signal=False` — no spurious signal before limit |
| L2_C06_non_leakage | non_leakage | **PASS** | augmentation_log empty at step 5 → vacuously no leakage (no signal emitted) |

**Layer 2 summary:** 17/18 pass. The single fail (`L2_C06_trigger`) is expected for a dropped candidate.

---

## Layer 3: Non-Regression Results

**Policy:** HandCodedTWAgent on original env (record commands) → replay on augmented env  
**Episodes:** 50 matched game pairs  
**Zero tolerance** on all discrete fields.

| Test ID | Field | Episodes | Mismatches | Status |
|---------|-------|----------|------------|--------|
| L3_T01 | reward (score) | 50 | 0 | **PASS** |
| L3_T02 | done | 50 | 0 | **PASS** |
| L3_T03 | admissible\_commands | 50 | 0 | **PASS** |
| L3_T04 | observation noop-path | 50 | 0 | **PASS** |

**All four tests pass with zero mismatches across 50 episodes.**

- L3_T01: `reward` is never modified by the wrapper (score 0.0→0.0 or 1.0→1.0 preserved exactly).
- L3_T02: `done` flag passes through unchanged from the base env's Limit/terminal logic.
- L3_T03: `infos['admissible_commands']` is never altered; wrapper only adds `progress_*` keys to infos.
- L3_T04: On steps where the base env returns obs ≠ "Nothing happens.", the augmented obs is byte-for-byte identical to the original — no spurious augmentation on successful actions.

---

## Shipping Verdict

Layer 1 pass = L1_T01 and L1_T02 both pass (global, applies to all rules).  
Layer 3 pass = all 4 L3 tests pass (global, applies to all rules).  
Layer 2 triplet pass = all three tests (trigger + non_trigger + non_leakage) pass for the candidate(s) sourced by each rule.

| rule\_id | kind | source\_candidate | layer1\_pass | layer2\_triplet\_pass | layer3\_pass | verdict |
|----------|------|-------------------|--------------|-----------------------|--------------|---------|
| R01 | disambiguate\_failure | C02 (entity non-exist) | ✓ | ✓ (C02 triplet: 3/3) | ✓ | **VALIDATED** |
| R02 | disambiguate\_failure | C03 (move while empty) | ✓ | ✓ (C03 triplet: 3/3) | ✓ | **VALIDATED** |
| R03 | disambiguate\_failure | C04 (take from closed) | ✓ | ✓ (C04 triplet: 3/3) | ✓ | **VALIDATED** |
| R04 | disambiguate\_failure | C04 (take wrong location) | ✓ | ✓ (C04 triplet: 3/3) | ✓ | **VALIDATED** |
| R05 | disambiguate\_failure | C03 (heat/cool/clean w/o holding) | ✓ | ✓ (C03 triplet: 3/3) | ✓ | **VALIDATED** |
| R06 | disambiguate\_failure | C01 (invalid verb) | ✓ | ✓ (C01 triplet: 3/3) | ✓ | **VALIDATED** |

**All 6 shipped rules are VALIDATED.** No rule is BLOCKED.

---

## Upstream Issues Surfaced

The following contract issues were detected in upstream artifacts. They are reported here and **not modified** per the verify_runner boundary contract.

### Issue 1 — unit_test_plan.json contains tests for dropped candidates C05 and C06

`unit_test_plan.json` defines triplets for C05 and C06, but `augmentation_plan.json::dropped_candidates` records both as deliberately excluded from the shipped wrapper. The wrapper contains no augmentation logic for either candidate.

**Impact:**
- `L2_C06_trigger` fails because no step-limit signal is emitted (expected for dropped candidate). This failure is not tied to any shipped rule and does not affect the Shipping Verdict.
- `L2_C05_trigger` passes only because the generic fallback augmentation ("Nothing happens. The action could not be performed…") technically satisfies the criterion "any non-empty text beyond Nothing happens." This is the fallback firing on any unmatched "Nothing happens." case, not a C05-specific disambiguation signal. The pass criterion as written is too permissive for detecting whether a genuine C05 signal exists.

**Recommendation (upstream):** Either remove C05/C06 triplets from `unit_test_plan.json` or mark them `"expect_fail": true` since the candidates were dropped. The current plan causes a misleading pass for C05_trigger.

### Issue 2 — C05 toggled-absence ↔ examine non-admissibility correlation unconfirmed

`unit_test_plan.json::L2_C05_trigger` notes: *"verify_runner should flag if toggled absence does not correlate with examine non-admissibility."*

In the executed test: `toggled=False` (lamp is off) AND `examine alarmclock 2` is NOT in admissible_commands. However, the non-admissibility is likely caused by wrong location (agent is in the middle of the room, not at desk 1 where alarmclock 2 resides), not by lamp-off. The correlation between toggled absence and examine non-admissibility is **unconfirmed** — this is precisely the open question cited in `dropped_candidates::C05::reason`. **Flagged as requested.**

### Issue 3 — Layer 1 plan specifies "valid_eval split"; valid_seen used instead

`unit_test_plan.json::L1_T01` states "Minimum episode count N=100 drawn from valid_eval split." The ALFWorld dataset cache contains only a `valid_seen` split (140 games available); no `valid_eval` directory exists at the cache path. The `valid_seen` split was used (100 games selected). This does not affect test correctness but is a naming discrepancy between the plan and the available data.

### Issue 4 — oracle_plan.json rejects expert_plan_handcoded as oracle; invocation used it as evaluation policy

`oracle_plan.json` rejects `expert_plan_handcoded` as an oracle due to HandCodedAgentTimeout risk. Per the invocation instruction, `HandCodedTWAgent` was used as the **evaluation policy** for Layer 1 A/B comparison (not as an oracle). No timeouts were observed across 100 games with max_steps=50. This is consistent: the plan's rejection applies to oracle use; policy use is a separate concern not addressed in oracle_plan.json.
