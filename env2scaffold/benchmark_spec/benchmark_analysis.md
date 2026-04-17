# ALFWorld Benchmark Analysis

**Benchmark:** ALFWorld (text-only, PDDL backend)  
**Version:** alfworld 0.4.2 / textworld 1.7.0 / fast-downward-textworld 20.6.4  
**Analysis date:** 2026-04-16  
**Analyst:** benchmark_reader pipeline role

---

## Source Layout

ALFWorld's runtime spans two installed packages and a verl-agent wrapper:

### `alfworld` package — task wrappers and evaluators
Location: `/data/home/yuhan/cyh_dev/lib/python3.12/site-packages/alfworld/`

- **`alfworld/env/tasks.py`** — six `BaseTask` subclasses corresponding to the six ALFWorld task types. Each implements `goal_satisfied(state)` and `goal_conditions_met(state)`, which check ALFRED-specific object-state predicates (hot, cold, clean, toggled, in-receptacle) against the AI2-THOR metadata. These are the ground-truth task evaluators for the **THOR** (visual) path; the text path relies on the PDDL goal checker instead.
- **`alfworld/agents/environment/alfred_tw_env.py`** — primary text-world interface. Contains `AlfredTWEnv` (game file collector, env initialiser), `AlfredDemangler` (name mapper), `AlfredInfos` (gamefile injector), and `AlfredExpert` (handcoded/planner expert wrapper).
- **`alfworld/agents/environment/alfred_thor_env.py`** — visual (THOR) environment path; not the primary ALFWorld text benchmark path but shares task evaluation logic.
- **`alfworld/agents/eval/evaluate_dagger.py`** (and `evaluate_dqn.py`) — official evaluation loops; compute success rate, goal-condition rate, and average steps.
- **`alfworld/agents/expert/`** — `HandCodedTWAgent` plus per-task-type policy classes (`PickAndPlaceSimpleTWPolicy`, etc.) that use PDDL facts and admissible commands to derive next actions.

### `textworld` package — runtime engine
Location: `/data/home/yuhan/cyh_dev/lib/python3.12/site-packages/textworld/`

- **`textworld/envs/pddl/pddl.py`** — `PddlEnv`: the core text environment. Implements `load()`, `reset()`, `step()`, and `_gather_infos()`. This is where feedback is emitted, the PDDL state is queried, and `won`/`score`/`done` are set.
- **`textworld/core.py`** — `Environment` base class, `EnvInfos` (info request configuration), `GameState` (dict-like state object), `Wrapper`.
- **`textworld/gym/`** — gym-compatible wrappers: `TextworldGymEnv` (single game), `TextworldBatchGymEnv` (batched), `register_games()`, `make()`.
- **`textworld/envs/wrappers/filter.py`** — `Filter` wrapper: selects only requested `EnvInfos` fields to expose in the `infos` dict returned to the agent.
- **`textworld/envs/wrappers/limit.py`** — `Limit` wrapper: enforces `max_episode_steps`.

### `verl-agent` wrappers (consumer)
Location: `/data/home/yuhan/env-aug/verl-agent/agent_system/environments/env_package/alfworld/envs.py`

- `AlfworldEnvs` and `AlfworldWorker`: Ray-based parallelisation on top of `AlfredTWEnv.init_env()`.
- `AugmentedAlfworldEnvs` and `AugmentedAlfworldWorker`: use `AugmentedAlfWorldEnv` (from `env2scaffold/augmentation/`) to replace "Nothing happens." with diagnostic feedback.
- `alfworld_projection()` in `projection.py`: strips `<think>...</think>` and `<action>...</action>` XML tags from LLM outputs before passing to `env.step()`.

### Assets
Location: `~/.cache/alfworld/json_2.1.1/`

Each game lives in a directory like `{task_type}-{object}-{mrecep}-{parent}-{scene_id}/trial_{id}/` containing:
- `game.tw-pddl` — JSON with PDDL domain, PDDL problem, grammar rules, solvability flag.
- `traj_data.json` — task metadata: `task_type`, `pddl_params` (object/parent/toggle targets), `plan` (high-level PDDL + low-level AI2-THOR actions), `turk_annotations`, scene info.

---

## Environment Interface

### Instantiation and reset/step cycle

The standard way to construct an ALFWorld text environment is via the TextWorld gym API:

```python
import textworld, textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
env_id = textworld.gym.register_games(
    gamefiles,                     # list of *.tw-pddl paths
    request_infos,
    batch_size=1,
    asynchronous=False,
    max_episode_steps=50,
    wrappers=[AlfredDemangler, AlfredInfos]
)
env = textworld.gym.make(env_id)
obs, infos = env.reset()           # returns (str, dict) for batch_size=1
obs, score, done, infos = env.step("go to sidetable 1")
```

When `batch_size > 1`, `env.reset()` returns `(List[str], Dict[str, List])` and `env.step(commands)` takes a list.

### Internal call chain

`env.step(command)` → `TextworldGymEnv.step()` → `TextworldBatchGymEnv.step()` → `SyncBatchEnv/AsyncBatchEnv.step()` → per-env `Filter.step()` → wraps back to `PddlEnv.step()`.

Inside `PddlEnv.step(command)`:
1. `command.strip()` is looked up in `self.prev_state["_valid_commands"]` via `list.index()`.
2. If found: the matching PDDL action is applied (`_pddl_state.apply(action)`), and the grammar-derived feedback string is produced via `self._logic.grammar.derive(action.feedback_rule, context)`.
3. If not found (ValueError): `state.feedback = "Nothing happens."` — no state change occurs.
4. `_gather_infos()` is called: populates `won`, `admissible_commands`, `facts` (if requested), `policy_commands` (if requested), etc.
5. Returns `(state, score=1 if won else 0, done=won or lost)`.

The `Limit` wrapper adds a hard episode cap: when `max_episode_steps` is reached, `done` is set to True regardless of task status.

### Reward and done

- **Reward**: sparse binary. `score = 1.0` only when `won = True` for the first time; all other steps return `0.0`.
- **Done**: True when `won or lost or max_episode_steps_exceeded`. In the text (PDDL) backend, `lost` is hardcoded to `False` (noted as a TODO in `pddl.py:77`), so done is effectively `won or step_limit`.

---

## Feedback Generation

### Where feedback strings originate

There are two feedback generation sites in `PddlEnv.step()`:

**Site 1 — Successful action** (`textworld/envs/pddl/pddl.py:158`):
```python
self.state.feedback = self._logic.grammar.derive(self._last_action.feedback_rule, context)
```
Each PDDL action type has a `feedback_rule` (a grammar rule name, e.g., `"GotoLocation.feedback"`). The grammar is a per-game JSON block embedded in `game.tw-pddl`. Rules are context-dependent: they reference entity names and sometimes check PDDL predicates. Example rules observed:
- `GotoLocation.feedback` → `"You arrive at {r.name}. #examineReceptacle.feedback#"` (templated)
- `OpenObject.feedback` → `"You open the {r.name}. #examineReceptacle.feedback#"` (templated)
- `CloseObject.feedback` → `"You close the {r.name}."` (templated)
- `PickupObject.feedback` → `"You pick up the {o.name} from the {r.name}."` (templated)
- `PutObject.feedback` → `"You move the {o.name} to the {r.name}."` (templated)
- `inventory.feedback` → conditional: mentions objects held or "You are not carrying anything." (both templated and hardcoded branches)

**Site 2 — Invalid command** (`textworld/envs/pddl/pddl.py:162`):
```python
self.state.feedback = "Nothing happens."
```
This is the only hardcoded string. It fires on _every_ cause of command invalidity without discrimination.

**Site 3 — Reset intro** (`textworld/envs/pddl/pddl.py:126`):
```python
self.state.feedback = self._logic.grammar.derive("#intro#", context)
```
The `#intro#` rule typically expands to `"-= Welcome to TextWorld, ALFRED! =-\n\n{room_description}\n\n{task_description}"`. Both room and task descriptions are game-specific and templated.

### The critical ambiguity cluster: "Nothing happens."

Probing data (`env2scaffold/probing/feedback_catalog.json`) confirms that the single string `"Nothing happens."` is emitted for **15 structurally distinct failure causes** across 7,957 observed occurrences. These include:
- Completely nonsensical commands (2,160 occurrences)
- `put` while holding nothing (540)
- Operations on nonexistent objects (540 each for put/take/open/use)
- Navigation to nonexistent locations (540)
- `take` from wrong location (540)
- `clean/heat/cool` without holding target (540 each)
- `open` already-open container (129)
- `take` from closed container (116)
- `close` already-closed container (116)
- `examine` without lamp on for `look_at_obj_in_light` task (37)

This is a severe ambiguity cluster: the agent receives identical feedback regardless of whether it failed because it hallucinated an object name, is holding the wrong item, is at the wrong location, or is violating a task precondition. The `feedback_auditor` will need to map which PDDL-fact conditions distinguish each cause cluster.

---

## Oracle Candidate Discussion

### 1. `won_info_field` (official_evaluator)
The `won` flag in the `infos` dict is populated at every step by `PddlEnv._gather_infos()` via `self._pddl_state.check_goal()`. It is the official, authoritative binary success signal for the ALFWorld text benchmark. It is deterministic given the PDDL world state. Access requires `request_infos.won=True` (this is standard in all observed wrappers). This is the primary ground-truth signal downstream roles should reference when defining Layer 1 validation checks.

### 2. `score_info_field` (official_score)
`info["score"]` is numerically equivalent to `int(won)` — set at `pddl.py:170`. It is the float form of the same signal. In the gym-level API, this is returned as the second element of the step tuple (not just in `infos`). The distinction from `won` is only typing (float vs bool) and position (tuple element vs dict key); both are computed from the same PDDL check.

### 3. `pddl_facts_state` (hidden_state_predicate)
When `request_infos.facts=True`, every step returns `info["facts"]` as a list of `Proposition` objects representing all currently true PDDL facts. This is the richest symbolic oracle: a downstream role could define arbitrary goal predicates over this set — e.g., checking `isclean(obj)`, `inreceptacle(obj, recep)`, `holds(agent, obj)`, `opened(container)` — to build fine-grained intermediate-progress detectors. The facts use game-specific entity IDs unless `AlfredDemangler` is applied (which renames entity `info.name` fields). Downstream callers reading raw facts without demangling should be aware of this.

### 4. `policy_commands_plan` (reference_trajectory)
With `request_infos.policy_commands=True`, the PDDL planner (`PddlState.replan()`) generates an optimal action sequence from the current state at each step. This is a fresh re-plan (not a cached sequence), so it is always valid for the current world state. It provides the full remaining solution path. This is a highly faithful oracle for measuring deviation from optimal, but exposing it to the agent trivialises the task. It requires `fast-downward-textworld` to be installed.

### 5. `expert_plan_handcoded` (reference_trajectory)
The `AlfredExpert` wrapper (with `expert_type=HANDCODED`) runs a handcoded `HandCodedTWAgent` to recommend a single next action at each step, stored in `info["extra.expert_plan"]` as a one-element list. The handcoded expert uses PDDL facts and admissible commands via task-type-specific policy classes (e.g., `PickAndPlaceSimpleTWPolicy`). Unlike the PDDL planner, the handcoded expert can time out (`HandCodedAgentTimeout`) on complex states, falling back to `["look"]`. This is an approximate oracle — it may be wrong in edge cases — but does not require the planner.

### 6. `traj_data_task_annotation` (task_annotation)
Each game's `traj_data.json` file (co-located with `game.tw-pddl`) contains `pddl_params` (the target object, receptacle, and toggle entities for the task), `task_type`, and `plan` (both high-level PDDL actions and low-level AI2-THOR primitive sequences). These are accessible offline via `info["extra.gamefile"]` to reconstruct the game's file path. This provides ground-truth about *what* the task requires without needing to run the PDDL checker, useful for task-type-conditional oracle design.

### 7. `admissible_commands_validity_heuristic` (derived_heuristic)
`info["admissible_commands"]` lists every command that the PDDL engine considers executable in the current state (sorted, deduplicated). A command not in this list will always produce "Nothing happens.". This field can serve as a binary validity oracle for the action-parse layer. However, membership in `admissible_commands` only guarantees PDDL executability — it does not distinguish task-advancing actions from task-neutral or task-regressing ones.

### 8. `goal_condition_success_rate_thor` (official_score)
This metric is computed in `alfred_thor_env.py:178` as `pcs[0] / float(pcs[1])` where `pcs` comes from the relevant `BaseTask.goal_conditions_met(state)`. It provides fractional sub-goal progress (e.g., 2/3 goal conditions satisfied). It is only available in the visual (THOR) execution path; the text-only (PDDL) path has no equivalent exposed in `infos`. Downstream agents on the text path cannot access this without switching to ThorEnv or independently computing sub-goal predicates from `pddl_facts_state`.

---

## Latent State Accessibility

### Via `infos` dict (runtime-accessible)

**`pddl_facts` channel** — enabled by `request_infos.facts=True`. At each `reset()` and `step()`, `PddlEnv._gather_infos()` (pddl.py:80-81) calls:
```python
self.state["facts"] = list(map(self._get_human_readable_fact, self.state["_facts"]))
```
where `_facts = list(self._pddl_state.facts)`. The human-readable mapping uses `AlfredDemangler`-modified entity names if that wrapper was applied. The raw `_facts` field is always populated but not exposed via the Filter wrapper unless `facts=True`.

**`policy_commands` channel** — enabled by `request_infos.policy_commands=True`. At each step, `_gather_infos()` calls `self._pddl_state.replan(self._entity_infos)` (pddl.py:108), which invokes the Fast Downward planner to re-plan from the current PDDL state.

**`extra.walkthrough` channel** — enabled by `extras=["walkthrough"]` in `EnvInfos`. The stored walkthrough from `game.tw-pddl` is used if present; otherwise the planner re-plans (pddl.py:131). On steps after reset, the walkthrough is carried forward unchanged (pddl.py:168), so it represents the solution from the start of the episode rather than from the current state.

**`extra.expert_plan` channel** — enabled by applying the `AlfredExpert` wrapper. This wrapper intercepts `step()` and `reset()`, runs the `HandCodedTWAgent`, and writes the recommended action to `state["extra.expert_plan"]` (alfred_tw_env.py:76-88).

### Via external files (offline-accessible)

**`traj_data.json`** — every `extra.gamefile` path `path/to/game.tw-pddl` has a co-located `path/to/traj_data.json` with full task annotation. Code path: `alfred_tw_env.py:166-170` reads `traj_data.json` to validate task type during game collection.

**`game.tw-pddl` PDDL problem** — the per-game PDDL problem definition encodes the initial state and goal condition, accessible by parsing the `pddl_problem` key in the JSON. This can be used to extract goal predicates without running the PDDL checker.

### What cannot be accessed without introspection

The internal `_pddl_state` object is not exposed via infos. To access it directly, the caller would need to traverse the wrapper chain and reach the underlying `PddlEnv` instance. The per-task `goal_conditions_met()` progress ratio (available in the THOR path) is not computed in the text path. The `heated_objects`, `cooled_objects`, and `cleaned_objects` sets tracked in `AlfredThorEnv` have no text-path equivalent in infos.

---

## Open Questions

1. **Does the PDDL backend ever set `state['lost'] = True` in practice, or is it always False as the comment in `pddl.py:77` suggests?**  
   The code comment says `# TODO: ask planner if the game is lost.` and hardcodes `self.state["lost"] = False`. If the game is never losable in the PDDL path, the only termination conditions are win or step limit. Confirmation requires either running the environment or reading `PddlState` internals.

2. **Is `goal_condition_success_rate` available in the `AlfredTWEnv` (text-only) path, or only in the `AlfredThorEnv` (visual) path?**  
   Source inspection shows it is only computed at `alfred_thor_env.py:178`. The text evaluators in `evaluate_dagger.py` guard its use with `if "goal_condition_success_rate" in infos`. This strongly implies it is not present in the text path, but cannot be confirmed without a live run.

3. **What is the exact set of grammar feedback rules for `CleanObject`, `HeatObject`, `CoolObject`, `SliceObject`, and `ToggleObject` actions?**  
   The grammar JSON in `game.tw-pddl` was partially inspected. The rules for these action types were not fully printed. The general pattern follows `ActionName.feedback` key names, but the exact template strings (and whether they vary by task type) are unresolved.

4. **Are there game files where the `walkthrough` key in `game.tw-pddl` is populated vs. requiring re-planning?**  
   One game file was inspected and showed `"walkthrough": null` (absent key, accessed via `data.get("walkthrough", None)` at pddl.py:60). It is unclear whether any game files in the 2.1.1 dataset include precomputed walkthroughs.

5. **Does the `Limit` wrapper trigger `done=True` silently without any feedback text change, i.e., does the agent receive a final observation different from a normal step?**  
   The `Limit` wrapper's implementation was not fully read. If it simply overrides `done` without changing `obs`, the agent would see the same (likely "Nothing happens." or a normal action feedback) on the step where the episode is force-terminated. This matters for reward shaping and training signal design.
