# ALFWorld Source Code Analysis

## 1. Where "Nothing happens." Is Generated

**File**: `/data/home/yuhan/cyh_dev/lib/python3.12/site-packages/textworld/envs/pddl/pddl.py`
**Layer**: TextWorld PDDL engine (lowest layer, below ALFWorld wrappers)
**Line**: ~162

```python
def step(self, command: str):
    command = command.strip()
    ...
    try:
        # Find the action corresponding to the command.
        idx = self.prev_state["_valid_commands"].index(command)
        ...
    except ValueError:
        # We assume nothing happened in the game.
        self.state.feedback = "Nothing happens."
```

The mechanism: `_valid_commands` contains all currently admissible commands (actions that are physically possible in the current game state). Any command NOT in this list throws a `ValueError`, and the feedback is hardcoded to `"Nothing happens."`.

There is no distinction between:
- Completely nonsensical commands ("fly to mars")
- Contextually wrong commands ("put X" when hands are empty)
- Structurally correct but state-invalid commands ("open drawer 1" when already open)
- Object-not-at-location errors ("take X from Y" when X is not in Y)

All produce the same feedback: `"Nothing happens."`

## 2. Information Available Internally But NOT Exposed to the Agent

The `_valid_commands` list in `prev_state` tells us exactly what actions ARE valid. By comparing the attempted command against this list, we can determine:

### Facts available via `request_infos.facts=True`:
- `holds(agent1, <object>)` — what the agent is currently holding
- `inreceptacle(<object>, <receptacle>)` — what objects are in what containers
- `opened(<receptacle>)` — which containers are currently open
- `openable(<receptacle>)` — which containers CAN be opened
- `atlocation(agent1, <location>)` — agent's current location
- `receptacleatlocation(<receptacle>, <location>)` — where receptacles are
- `objectatlocation(<object>, <location>)` — where objects are (by location grid cell)
- `pickupable(<object>)` — which objects can be picked up
- `heatable(<object>)` — which objects can be heated
- `coolable(<object>)` — which objects can be cooled
- `cleanable(<object>)` — which objects can be cleaned
- `isreceptacleobject(<object>)` — objects that are also receptacles (e.g., mugs, boxes)
- `receptacletype(<receptacle>, <type>)` / `objecttype(<object>, <type>)` — type info

### Available via `admissible_commands` (no extra request needed):
- Full list of valid actions in current state. This is the primary source for detecting errors.

### Key internal state readable from facts but not directly told to agent:
1. **Held object**: `holds(agent1, X)` fact
2. **Container open/closed**: `opened(X)` present = open; `openable(X)` present but `opened(X)` absent = closed
3. **Object locations**: `inreceptacle(obj, recep)` + `receptacleatlocation(recep, loc)` + `atlocation(agent1, loc)`
4. **Whether agent is "near" a receptacle**: A receptacle is considered "at current location" if it appears in current admissible commands (e.g., `open <recep>`, `take <obj> from <recep>`)

## 3. The `infos` Dictionary (Gym Interface)

When using `textworld.gym` with `batch_size=1`, `env.reset()` returns `(obs_str, infos_dict)` and `env.step(cmd)` returns `(obs_str, score, done, infos_dict)`.

Key fields in `infos`:
```
{
  'admissible_commands': list[str],    # all valid actions right now
  'facts': list[Proposition],          # PDDL facts about world state  
  'won': bool,                         # whether task is complete
  'extra.gamefile': str,               # path to game file
  # (if requested):
  'inventory': str,                    # text output of 'inventory' command
  'description': str,                  # text output of 'look' command
  'location': str,                     # current room name
}
```

Note: `inventory`, `description`, and `location` return `None` unless explicitly requested AND they are actively computed. In ALFRED PDDL games, `location` is always `None` (not implemented).

## 4. Wrapper Access to Internal State

A gym wrapper around `TextworldGymEnv` (or batch version) receives:
- `obs` (str): The text observation
- `infos` (dict): All requested info including `admissible_commands` and `facts`

The wrapper can:
1. **Detect "Nothing happens."** by checking `obs == "Nothing happens."`
2. **Read current state** from `infos['facts']` to understand WHY it happened
3. **Parse the attempted command** to identify the specific error type
4. **Generate targeted feedback** without modifying `score` or `done`

The wrapper wraps the gym env (not the TextWorld core env), so it operates at the gym API level: `reset() -> (obs, infos)` and `step(cmd) -> (obs, score, done, infos)`.

## 5. Command Grammar in ALFWorld

ALFWorld (via AlfredDemangler) uses human-readable commands:
```
go to <receptacle>
take <object> from <receptacle>
move <object> to <receptacle>   (i.e., put)
open <receptacle>
close <receptacle>
examine <receptacle/object>
use <lamp>                       (for look_at_obj task)
heat <object> with <receptacle>
cool <object> with <receptacle>
clean <object> with <receptacle>
slice <object> with <object>
inventory
look
help
```
