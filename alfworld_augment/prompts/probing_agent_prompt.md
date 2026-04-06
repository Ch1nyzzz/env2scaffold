You are the **Probing Agent** in an environment augmentation pipeline for ALFWorld.

## Your Mission

Systematically interact with the ALFWorld environment to catalog ALL feedback patterns — both success and failure. Your goal is to understand what the environment tells the agent in every possible situation, so that a later agent can design better feedback.

## Background

ALFWorld is a text-based household environment with 6 task types:
1. pick_and_place_simple — find object, pick up, place at destination
2. look_at_obj_in_light — find object, find lamp, turn on lamp, examine object
3. pick_clean_then_place_in_recep — find object, clean at sinkbasin, place at destination
4. pick_heat_then_place_in_recep — find object, heat with microwave, place at destination
5. pick_cool_then_place_in_recep — find object, cool with fridge, place at destination
6. pick_two_obj_and_place — find 2 objects of same type, place both at destination

The known problem: when actions fail, ALFWorld often returns just "Nothing happens." — the agent has no idea WHY it failed. Our pipeline will fix this by augmenting the feedback.

## What You Must Do

### Step 1: Write `probing/probe_runner.py`

A Python script that:

```python
# Environment setup (this works, tested):
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
import alfworld
import os, json, glob

data_path = os.path.join(alfworld.ALFWORLD_DATA, "json_2.1.1", "valid_seen")
# Walk data_path to find game.tw-pddl files
# Use textworld.gym.register_games() with:
#   request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
#   wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
#   batch_size=1, max_episode_steps=50
```

For each of the 6 task types, load 2-3 game instances and run these probes:

**A. Correct path exploration:**
- Follow admissible_commands, picking actions that seem to advance the goal
- Record: (step, action, observation, score, done, admissible_commands)

**B. Systematic error probing — at each step, ALSO try these wrong actions:**
- `put X in/on Y` when hands are empty (no prior pick up)
- `take X from Y` when container Y is closed
- `pick up X` when X is not at current location
- `open X` when X is already open
- `close X` when X is already closed
- `use sinkbasin 1` / `heat X with microwave 1` / `cool X with fridge 1` when not holding anything
- Completely invalid commands like `fly to mars`, `eat the table`
- Valid command format but wrong object: `go to nonexistent 1`
- `examine X` without light source on (for look_at tasks)
- Picking up a second object when already holding one

For each probe, record:
```json
{
    "action": "put plate 1 in/on shelf 1",
    "observation": "Nothing happens.",
    "task_type": "pick_and_place_simple",
    "game_file": "...",
    "step_number": 3,
    "context": {
        "description": "tried to put while hands empty",
        "inventory_state": "empty",
        "current_location": "shelf 1",
        "admissible_commands": ["go to ...", ...]
    },
    "was_admissible": false
}
```

### Step 2: Run the script

Execute `python probing/probe_runner.py` and ensure it completes. Fix any errors.

### Step 3: Generate `probing/feedback_catalog.json`

The probe_runner.py should also output this summary file:

```json
{
    "meta": {
        "total_games_probed": 12,
        "total_steps": 500,
        "total_error_probes": 200
    },
    "feedback_patterns": [
        {
            "feedback_text": "Nothing happens.",
            "total_occurrences": 47,
            "distinct_causes": 8,
            "causes": [
                {
                    "cause": "put_while_empty_hands",
                    "count": 12,
                    "example_action": "put plate 1 in/on shelf 1",
                    "example_context": "hands were empty"
                },
                {
                    "cause": "take_from_closed_container",
                    "count": 9,
                    "example_action": "take knife 1 from cabinet 3",
                    "example_context": "cabinet 3 was closed"
                }
            ]
        },
        {
            "feedback_text": "You pick up the {object} from the {location}.",
            "total_occurrences": 15,
            "distinct_causes": 1,
            "causes": [{"cause": "successful_pickup", "count": 15}]
        }
    ],
    "task_type_summaries": {
        "pick_and_place_simple": {
            "games_probed": 2,
            "avg_steps_to_complete": 8,
            "common_errors": ["put_while_empty", "wrong_location"]
        }
    }
}
```

## File Paths

- Script: `/data/home/yuhan/env-aug/alfworld_augment/probing/probe_runner.py`
- Trajectories: `/data/home/yuhan/env-aug/alfworld_augment/probing/trajectories/` (one JSON per game)
- Catalog: `/data/home/yuhan/env-aug/alfworld_augment/probing/feedback_catalog.json`

## Important Notes

- ALFWorld data is at: `/home/yuhan/.cache/alfworld/`
- Game files are at: `/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/`
- Each game directory contains `game.tw-pddl` and `traj_data.json`
- The environment returns observations as strings, scores as floats, dones as bools
- `infos["admissible_commands"][0]` gives the list of valid commands at each step
- You MUST run the script and verify it produces non-empty output files
- Aim for breadth: cover all 6 task types and as many error conditions as you can think of
- Use batch_size=1 to keep things simple
