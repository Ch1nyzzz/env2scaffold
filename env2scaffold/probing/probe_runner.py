#!/usr/bin/env python3
"""
ALFWorld Probing Runner

Systematically explores the ALFWorld environment to catalog all feedback patterns,
both success and failure. Produces:
  - probing/trajectories/{game_id}.json  — per-game step logs
  - probing/feedback_catalog.json        — aggregated pattern catalog
"""

import json
import os
import re
import glob
from collections import defaultdict
from pathlib import Path

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
import alfworld

# ─── Paths ─────────────────────────────────────────────────────────────────────
ALFWORLD_DATA = alfworld.ALFWORLD_DATA
DATA_PATH = os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_seen")
PROBING_DIR = Path(__file__).resolve().parent
TRAJECTORIES_DIR = PROBING_DIR / "trajectories"
CATALOG_PATH = PROBING_DIR / "feedback_catalog.json"

TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Task type definitions ──────────────────────────────────────────────────────
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

GAMES_PER_TASK = 2   # probe 2 games per task type
MAX_STEPS = 45       # max steps per game
PROBE_EVERY_N = 1    # probe errors every N correct steps


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_task_type(game_path: str) -> str:
    """Infer task type from the game file path."""
    for tt in TASK_TYPES:
        if tt in game_path:
            return tt
    if "pick_and_place_with_movable_recep" in game_path:
        return "pick_and_place_with_movable_recep"
    return "unknown"


def discover_games() -> dict:
    """Walk DATA_PATH and group game.tw-pddl files by task type."""
    games_by_type = defaultdict(list)
    for game_file in sorted(glob.glob(os.path.join(DATA_PATH, "**", "game.tw-pddl"), recursive=True)):
        tt = get_task_type(game_file)
        games_by_type[tt].append(game_file)
    return dict(games_by_type)


def make_env(game_files: list, env_name: str):
    """Register and create a TextWorld gym environment."""
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        extras=["gamefile"],
    )
    # max_episode_steps must be large enough to cover error probes (~50/step)
    # plus the actual correct steps (~40). 50 probes × 40 steps + 200 buffer = 2200
    env_id = textworld.gym.register_games(
        game_files,
        request_infos=request_infos,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        batch_size=1,
        max_episode_steps=3000,
        name=env_name,
    )
    return textworld.gym.make(env_id)


def extract_objects_from_obs(obs: str) -> list:
    """
    Extract simple 'word number' object references from an observation.
    e.g., 'plate 1', 'knife 2', 'cabinet 3'.
    Only matches single lowercase word + integer patterns.
    """
    # Match e.g. "a plate 1" or "the cabinet 3" – capture the word and number only
    matches = re.findall(r'\b([a-z]+)\s+(\d+)\b', obs.lower())
    # Filter out common non-object words that appear with numbers
    noise_words = {"step", "you", "the", "a", "an", "at", "in", "on", "is", "of",
                   "your", "task", "to", "see", "are", "room"}
    return [f"{w} {n}" for w, n in matches if w not in noise_words]


def extract_location_from_obs(obs: str) -> str:
    """Try to extract current location from an observation."""
    # "You arrive at X Y" or "You are in X Y"
    m = re.search(r'You arrive at ([a-z]+ \d+)', obs, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'at the ([a-z]+ \d+)', obs, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return ""


def serialize_fact(fact) -> dict:
    """Convert a TextWorld proposition into a JSON-serializable record."""
    args = [arg.name for arg in getattr(fact, "arguments", [])]
    fact_text = f"{fact.name}({', '.join(args)})" if args else fact.name
    return {
        "name": fact.name,
        "arguments": args,
        "text": fact_text,
    }


def serialize_facts(facts: list) -> list:
    """Serialize and sort facts for stable JSON output."""
    serialized = [serialize_fact(fact) for fact in (facts or [])]
    return sorted(serialized, key=lambda item: item["text"])


def compute_fact_delta(before_facts: list, after_facts: list) -> dict:
    """Return added/removed fact records between two serialized fact lists."""
    before_map = {fact["text"]: fact for fact in before_facts}
    after_map = {fact["text"]: fact for fact in after_facts}
    added = [after_map[key] for key in sorted(after_map.keys() - before_map.keys())]
    removed = [before_map[key] for key in sorted(before_map.keys() - after_map.keys())]
    return {
        "added": added,
        "removed": removed,
    }


def parse_inventory(admissible: list) -> tuple:
    """
    Determine inventory state from admissible commands.
    'put X in/on Y' appears only when holding X.
    Returns (inventory_state, holding_object).
    """
    for cmd in admissible:
        m = re.match(r'put (.+?) in/on', cmd)
        if m:
            return "holding", m.group(1)
    return "empty", ""


def build_state_snapshot(obs: str, admissible: list, facts: list) -> dict:
    """Capture the observable and latent state around a transition."""
    inventory_state, holding = parse_inventory(admissible)
    return {
        "location": extract_location_from_obs(obs),
        "visible_entities": extract_objects_from_obs(obs),
        "inventory_state": inventory_state,
        "holding": holding,
        "admissible_commands": list(admissible),
        "facts": serialize_facts(facts),
    }


def find_closed_containers(obs: str) -> list:
    """Return list of closed container names visible in observation."""
    # Pattern: "The cabinet 1 is closed." or "a drawer 2 is closed"
    return re.findall(r'\b([a-z]+ \d+) is closed', obs.lower())


def find_open_containers(obs: str) -> list:
    """Return list of open container names visible in observation."""
    return re.findall(r'\b([a-z]+ \d+) is open', obs.lower())


def get_destination_from_admissible(admissible: list) -> list:
    """Extract all go-to destination names from admissible commands."""
    dests = []
    for cmd in admissible:
        m = re.match(r'go to (.+)', cmd)
        if m:
            dests.append(m.group(1))
    return dests


# ─── Error probe builders ───────────────────────────────────────────────────────

def build_error_probes(obs: str, admissible: list, inventory_state: str,
                       current_location: str, holding: str, task_type: str) -> list:
    """
    Build (action, cause, description) tuples for error probing at current state.
    Each tuple is a wrong action that should fail with 'Nothing happens.' or similar.
    """
    probes = []

    # Extract objects visible in this observation
    visible_objs = extract_objects_from_obs(obs)
    # Remove location-word objects (words that are typically locations not items)
    location_words = {"cabinet", "drawer", "countertop", "sidetable", "shelf", "fridge",
                      "microwave", "sinkbasin", "stoveburner", "bed", "sofa", "toilet",
                      "bathtubbasin", "garbagecan", "coffeetable", "armchair", "cart",
                      "desk", "safe", "ottoman", "diningtable", "coffeemachine",
                      "desklamp", "floorlamp", "toiletpaperhanger"}
    item_objs = [o for o in visible_objs if o.split()[0] not in location_words]
    all_objs = visible_objs

    # Pick representative objects for probes
    sample_item = item_objs[0] if item_objs else (all_objs[0] if all_objs else "plate 1")
    sample_item2 = item_objs[1] if len(item_objs) > 1 else "knife 1"
    sample_container = all_objs[0] if all_objs else "cabinet 1"

    # Also extract containers/destinations from admissible commands
    dests = get_destination_from_admissible(admissible)
    sample_dest = dests[0] if dests else "cabinet 1"

    closed_containers = find_closed_containers(obs)
    open_containers = find_open_containers(obs)

    # ── Group A: Inventory errors ──────────────────────────────────────────────

    if inventory_state == "empty":
        # A1. put while hands empty
        probes.append((
            f"put {sample_item} in/on {sample_dest}",
            "put_while_empty_hands",
            f"tried to put {sample_item!r} while holding nothing"
        ))
        # A2. put a different fake object
        probes.append((
            f"put knife 99 in/on {sample_dest}",
            "put_nonexistent_object",
            "tried to put a nonexistent object"
        ))
    else:
        # A3. pick up when already holding something
        probes.append((
            f"take {sample_item2} from {sample_dest}",
            "take_when_already_holding",
            f"tried to pick up {sample_item2!r} while already holding {holding!r}"
        ))
        # A4. take from current location but wrong object
        probes.append((
            f"take nonexistent 99 from {current_location}",
            "take_nonexistent_when_holding",
            "tried to take a nonexistent object while already holding something"
        ))

    # ── Group B: Container state errors ───────────────────────────────────────

    if closed_containers:
        closed = closed_containers[0]
        # B1. take from closed container
        probes.append((
            f"take {sample_item} from {closed}",
            "take_from_closed_container",
            f"tried to take from {closed!r} which is currently closed"
        ))
        # B2. close already closed
        probes.append((
            f"close {closed}",
            "close_already_closed",
            f"tried to close {closed!r} which is already closed"
        ))

    if open_containers:
        opened = open_containers[0]
        # B3. open already open container
        probes.append((
            f"open {opened}",
            "open_already_open",
            f"tried to open {opened!r} which is already open"
        ))

    # B4. open a container that isn't here
    probes.append((
        "open nonexistent 99",
        "open_nonexistent_container",
        "tried to open a container that doesn't exist"
    ))

    # ── Group C: Location errors ───────────────────────────────────────────────

    # C1. take object not at this location
    probes.append((
        "take watermelon 1 from floor",
        "take_nonexistent_object",
        "tried to take a completely nonexistent object"
    ))

    # C2. navigate to nonexistent location
    probes.append((
        "go to nonexistent 99",
        "go_to_nonexistent_location",
        "tried to navigate to a location that doesn't exist"
    ))

    # C3. take from wrong receptacle (object exists but not there)
    if dests:
        alt_dest = dests[-1]  # pick last one (likely different from sample)
        probes.append((
            f"take {sample_item} from {alt_dest}",
            "take_from_wrong_location",
            f"tried to take {sample_item!r} from {alt_dest!r} which probably doesn't have it"
        ))

    # ── Group D: Clean/Heat/Cool without holding ───────────────────────────────

    if inventory_state == "empty":
        probes.append((
            f"clean {sample_item} with sinkbasin 1",
            "clean_without_holding",
            "tried to clean an object without holding it"
        ))
        probes.append((
            f"heat {sample_item} with microwave 1",
            "heat_without_holding",
            "tried to heat an object without holding it"
        ))
        probes.append((
            f"cool {sample_item} with fridge 1",
            "cool_without_holding",
            "tried to cool an object without holding it"
        ))
    else:
        # D4. clean/heat/cool wrong object (use receptacle name)
        probes.append((
            f"clean {sample_dest} with sinkbasin 1",
            "clean_wrong_object",
            f"tried to clean a receptacle {sample_dest!r} instead of held item"
        ))
        probes.append((
            f"heat {sample_dest} with microwave 1",
            "heat_wrong_object",
            f"tried to heat a receptacle {sample_dest!r} instead of held item"
        ))
        probes.append((
            f"cool {sample_dest} with fridge 1",
            "cool_wrong_object",
            f"tried to cool a receptacle {sample_dest!r} instead of held item"
        ))

    # ── Group E: Lamp / examine errors ────────────────────────────────────────

    # E1. examine without lamp turned on (relevant for look_at tasks)
    if task_type == "look_at_obj_in_light":
        probes.append((
            f"examine {sample_item}",
            "examine_without_lamp_on",
            "tried to examine object for look_at task without lamp turned on"
        ))

    # E2. use lamp not present here
    probes.append((
        "use desklamp 99",
        "use_nonexistent_lamp",
        "tried to use a desklamp that doesn't exist"
    ))

    # ── Group F: Completely invalid commands ──────────────────────────────────

    probes.append(("fly to mars", "completely_invalid_command", "nonsensical action"))
    probes.append(("eat the table", "completely_invalid_command", "nonsensical action"))
    probes.append(("teleport home", "completely_invalid_command", "nonsensical action"))
    probes.append(("attack robot 1", "completely_invalid_command", "nonsensical action"))

    return probes


# ─── Action selection ───────────────────────────────────────────────────────────

def parse_goal(task_goal: str, task_type: str) -> tuple:
    """Parse goal string to extract target object and receptacle names."""
    goal_lower = task_goal.lower()
    goal_obj = ""
    goal_recep = ""

    # "put some X on/in Y"
    m = re.search(r'put (?:some |a |two |the )?(\w+) (?:in|on) (\w+)', goal_lower)
    if m:
        goal_obj = m.group(1)
        goal_recep = m.group(2)

    # "look at / examine X" (for look_at)
    m2 = re.search(r'(?:look at|examine) (?:the |a |some )?(\w+)', goal_lower)
    if m2:
        goal_obj = m2.group(1)

    # "clean/heat/cool X and put it in Y"
    m3 = re.search(r'(?:clean|heat|cool) (?:some |a |the )?(\w+) and put (?:it )?in (\w+)', goal_lower)
    if m3:
        goal_obj = m3.group(1)
        goal_recep = m3.group(2)

    # "put a hot/cool/clean X in Y"
    m4 = re.search(r'put a (?:hot|cool|clean) (\w+) in (\w+)', goal_lower)
    if m4:
        goal_obj = m4.group(1)
        goal_recep = m4.group(2)

    # "find two X and put them in Y"
    m5 = re.search(r'find two (\w+) and put (?:them )?in (\w+)', goal_lower)
    if m5:
        goal_obj = m5.group(1)
        goal_recep = m5.group(2)

    return goal_obj, goal_recep


def choose_next_action(obs: str, admissible: list, task_goal: str,
                       task_type: str, step_count: int,
                       inventory_state: str, holding: str,
                       visited_actions: set) -> str:
    """
    Task-oriented action selector. Prioritizes actions that advance the goal.
    Falls back to exploration when goal-directed actions aren't available.
    """
    if not admissible:
        return None

    goal_obj, goal_recep = parse_goal(task_goal, task_type)
    obs_lower = obs.lower()

    def score_action(cmd: str) -> float:
        cl = cmd.lower()
        s = 0.0

        # === Goal-directed scoring ===

        if inventory_state == "empty":
            # Take target object
            if goal_obj and f"take {goal_obj}" in cl:
                s += 50
            # Take any object (might be goal object hiding elsewhere)
            elif "take" in cl:
                s += 8
            # Go to goal object location
            if goal_obj and f"go to {goal_obj}" in cl:
                s += 30
            # Open a container that might contain the goal
            if "open" in cl:
                s += 6

        elif inventory_state == "holding":
            held_lower = holding.lower()
            held_word = held_lower.split()[0] if held_lower else ""

            # Put held object at goal receptacle
            if goal_recep and "put" in cl and goal_recep in cl:
                s += 60
            # Put anywhere (fallback)
            elif "put" in cl:
                s += 5

            # Task-specific intermediate steps
            if task_type == "look_at_obj_in_light":
                # Use/turn on lamp
                if "use" in cl and ("lamp" in cl or "light" in cl):
                    s += 55
                # Examine object (after lamp is on)
                if "examine" in cl and goal_obj and goal_obj in cl:
                    s += 40

            elif task_type == "pick_clean_then_place_in_recep":
                # Clean held object at sinkbasin
                if "clean" in cl and held_word and held_word in cl:
                    s += 55
                # Go to sinkbasin
                if "go to sinkbasin" in cl:
                    s += 35

            elif task_type == "pick_heat_then_place_in_recep":
                # Heat held object in microwave
                if "heat" in cl and held_word and held_word in cl:
                    s += 55
                # Go to microwave
                if "go to microwave" in cl:
                    s += 35

            elif task_type == "pick_cool_then_place_in_recep":
                # Cool held object in fridge
                if "cool" in cl and held_word and held_word in cl:
                    s += 55
                # Go to fridge
                if "go to fridge" in cl:
                    s += 35

        # === Exploration scoring ===

        # Go to goal receptacle (to place)
        if goal_recep and f"go to {goal_recep}" in cl:
            s += 20

        # Navigate > examine > open > other
        if "go to" in cl:
            s += 3
        if "open" in cl:
            s += 2
        if "examine" in cl:
            s += 1

        # Penalize already-visited actions (still allow them, just last resort)
        if cmd in visited_actions:
            s -= 100

        return s

    scored = sorted(admissible, key=score_action, reverse=True)

    # Return best unvisited action; if all visited, return best overall
    for action in scored:
        if action not in visited_actions:
            return action

    return scored[0] if scored else None


# ─── Core probing logic ─────────────────────────────────────────────────────────

def probe_game(game_file: str, task_type: str, game_idx: int) -> dict:
    """
    Probe a single game:
    - Follow a task-oriented path through admissible commands
    - At each step, probe a set of error conditions
    Returns a trajectory dict.
    """
    game_id = Path(game_file).parent.name + "_" + Path(game_file).parent.parent.name
    print(f"\n  Probing: {game_id} [{task_type}]")

    env_name = f"probe_{task_type}_{game_idx}".replace("-", "_")
    env = make_env([game_file], env_name)

    obs_tuple, infos = env.reset()
    obs = obs_tuple[0]
    current_admissible = infos["admissible_commands"][0]
    current_facts = infos.get("facts", [[]])[0]
    current_snapshot = build_state_snapshot(obs, current_admissible, current_facts)

    trajectory = {
        "game_file": game_file,
        "game_id": game_id,
        "task_type": task_type,
        "initial_obs": obs,
        "initial_state": current_snapshot,
        "steps": [],
        "error_probes": [],
        "completed": False,
        "total_steps": 0,
    }

    goal_match = re.search(r'Your task is to: (.+)', obs)
    task_goal = goal_match.group(1).strip() if goal_match else ""
    trajectory["task_goal"] = task_goal
    print(f"    Goal: {task_goal}")

    current_obs = obs
    step_number = 0
    done = False

    inventory_state = current_snapshot["inventory_state"]
    holding = current_snapshot["holding"]
    current_location = current_snapshot["location"] or "start"
    visited_actions = set()

    while not done and step_number < MAX_STEPS:
        # ── Error probing phase ───────────────────────────────────────────────
        probes = build_error_probes(
            current_obs, current_admissible,
            inventory_state, current_location, holding, task_type
        )

        for probe_action, cause, description in probes:
            # Skip if action is actually admissible (it would be a valid/correct step)
            if probe_action in current_admissible:
                continue

            probe_pre_snapshot = build_state_snapshot(current_obs, current_admissible, current_facts)
            probe_obs_tuple, probe_scores, probe_dones, probe_infos = env.step([probe_action])
            probe_obs = probe_obs_tuple[0]
            probe_admissible = probe_infos["admissible_commands"][0]
            probe_facts = probe_infos.get("facts", [[]])[0]
            probe_post_snapshot = build_state_snapshot(probe_obs, probe_admissible, probe_facts)

            probe_record = {
                "action": probe_action,
                "observation": probe_obs,
                "task_type": task_type,
                "game_file": game_file,
                "step_number": step_number,
                "score": probe_scores[0],
                "done": probe_dones[0],
                "won": probe_infos.get("won", [False])[0],
                "state_before": probe_pre_snapshot,
                "state_after": probe_post_snapshot,
                "fact_delta": compute_fact_delta(
                    probe_pre_snapshot["facts"],
                    probe_post_snapshot["facts"],
                ),
                "context": {
                    "description": description,
                    "inventory_state": inventory_state,
                    "holding": holding,
                    "current_location": current_location,
                    "admissible_commands": list(current_admissible),
                    "cause": cause,
                },
                "was_admissible": False,
            }
            trajectory["error_probes"].append(probe_record)

            # Invalid actions should NOT advance the episode; if done fires, stop
            if probe_dones[0]:
                done = True
                break

        if done:
            break

        # ── Correct-path action selection ─────────────────────────────────────
        action = choose_next_action(
            current_obs, current_admissible, task_goal, task_type,
            step_number, inventory_state, holding, visited_actions
        )

        if action is None:
            print(f"    No action available at step {step_number}")
            break

        new_obs_tuple, scores, dones, new_infos = env.step([action])
        new_obs = new_obs_tuple[0]
        score = scores[0]
        done = dones[0]
        new_admissible = new_infos["admissible_commands"][0]
        new_facts = new_infos.get("facts", [[]])[0]
        won = new_infos.get("won", [False])[0]

        state_before = build_state_snapshot(current_obs, current_admissible, current_facts)
        state_after = build_state_snapshot(new_obs, new_admissible, new_facts)
        new_inv_state = state_after["inventory_state"]
        new_holding = state_after["holding"]
        new_location = state_after["location"] or current_location

        step_record = {
            "step": step_number,
            "action": action,
            "observation": new_obs,
            "score": score,
            "done": done,
            "won": won,
            "admissible_commands": list(new_admissible),
            "inventory_state": new_inv_state,
            "holding": new_holding,
            "location": new_location,
            "state_before": state_before,
            "state_after": state_after,
            "fact_delta": compute_fact_delta(
                state_before["facts"],
                state_after["facts"],
            ),
        }
        trajectory["steps"].append(step_record)

        print(f"    Step {step_number:2d}: {action!r} → {new_obs[:80]!r}")

        visited_actions.add(action)
        current_obs = new_obs
        current_admissible = new_admissible
        current_facts = new_facts
        inventory_state = new_inv_state
        holding = new_holding
        current_location = new_location
        step_number += 1

        if won:
            trajectory["completed"] = True
            print(f"    ✓ Task completed in {step_number} steps!")
            break

    trajectory["total_steps"] = step_number
    env.close()
    return trajectory


# ─── Catalog generation ─────────────────────────────────────────────────────────

def build_feedback_catalog(all_trajectories: list) -> dict:
    """Aggregate trajectory data into a structured feedback catalog."""

    # obs_text → cause → list of examples
    feedback_map: dict = defaultdict(lambda: defaultdict(list))
    total_steps = 0
    total_error_probes = 0
    task_summaries: dict = defaultdict(lambda: {
        "games_probed": 0,
        "steps_to_complete": [],
        "error_counts": defaultdict(int),
    })

    for traj in all_trajectories:
        tt = traj["task_type"]
        task_summaries[tt]["games_probed"] += 1

        if traj["completed"]:
            task_summaries[tt]["steps_to_complete"].append(traj["total_steps"])

        # Correct-path observations
        for step in traj["steps"]:
            total_steps += 1
            obs = step["observation"]
            feedback_map[obs]["successful_action"].append({
                "action": step["action"],
                "task_type": tt,
                "game_file": traj["game_file"],
                "step": step["step"],
            })

        # Error probe observations
        for probe in traj["error_probes"]:
            total_error_probes += 1
            obs = probe["observation"]
            cause = probe["context"]["cause"]
            feedback_map[obs][cause].append({
                "action": probe["action"],
                "task_type": tt,
                "game_file": traj["game_file"],
                "step": probe["step_number"],
                "context": probe["context"]["description"],
                "inventory_state": probe["context"]["inventory_state"],
                "holding": probe["context"].get("holding", ""),
            })
            task_summaries[tt]["error_counts"][cause] += 1

    # Build ordered pattern list (most frequent first)
    feedback_patterns = []
    for obs_text, causes_dict in sorted(
        feedback_map.items(),
        key=lambda x: -sum(len(v) for v in x[1].values())
    ):
        total_occurrences = sum(len(v) for v in causes_dict.values())
        causes_list = []
        for cause, examples in sorted(causes_dict.items(), key=lambda x: -len(x[1])):
            ex = examples[0]
            causes_list.append({
                "cause": cause,
                "count": len(examples),
                "example_action": ex["action"],
                "example_context": ex.get("context", ex.get("task_type", "")),
            })

        # Create template by replacing specific names with placeholders
        template = _make_template(obs_text)

        feedback_patterns.append({
            "feedback_text": obs_text,
            "feedback_template": template,
            "total_occurrences": total_occurrences,
            "distinct_causes": len(causes_dict),
            "causes": causes_list,
        })

    # Finalize task summaries
    task_summaries_final = {}
    for tt, summary in task_summaries.items():
        steps_list = summary["steps_to_complete"]
        avg_steps = round(sum(steps_list) / len(steps_list), 1) if steps_list else None
        sorted_errors = sorted(summary["error_counts"].items(), key=lambda x: -x[1])
        task_summaries_final[tt] = {
            "games_probed": summary["games_probed"],
            "games_completed": len(steps_list),
            "avg_steps_to_complete": avg_steps,
            "common_errors": [e[0] for e in sorted_errors[:5]],
            "error_counts": dict(sorted_errors),
        }

    return {
        "meta": {
            "total_games_probed": len(all_trajectories),
            "total_steps": total_steps,
            "total_error_probes": total_error_probes,
            "task_types_covered": list(task_summaries_final.keys()),
            "state_transition_fields": ["state_before", "state_after", "fact_delta"],
        },
        "feedback_patterns": feedback_patterns,
        "task_type_summaries": task_summaries_final,
    }


# ALFWorld object and location word lists for template normalization
_OBJECT_WORDS = (
    "alarmclock|apple|appleSliced|baseballbat|book|bottle|bowl|box|bread|breadSliced|"
    "butterknife|candle|cd|cellphone|cloth|creditcard|cup|desklamp|dishsponge|egg|"
    "floorlamp|fork|glassbottle|handtowel|kettle|keychain|knife|ladle|laptop|lettuce|"
    "lettuceSliced|mug|newspaper|pan|pen|pencil|pillow|plate|plunger|potato|potatoSliced|"
    "remotecontrol|soapbar|soapbottle|spatula|spoon|spraybottle|statue|tennisracket|"
    "tissuebox|toiletpaper|tomato|tomatoSliced|vase|watch|winebottle|wateringcan"
)
_LOCATION_WORDS = (
    "armchair|bathtub|bathtubbasin|bed|cabinet|cart|coffeetable|countertop|desk|"
    "diningtable|drawer|fridge|garbagecan|microwave|ottoman|safe|shelf|sidetable|"
    "sinkbasin|sofa|stoveburner|toilet|toiletpaperhanger|coffeemachine"
)


def _make_template(obs_text: str) -> str:
    """Replace specific object/location names+numbers with {object}/{location} placeholders."""
    t = re.sub(
        rf'\b({_OBJECT_WORDS.lower()})\s+(\d+)\b',
        r'{\1} \2',  # keep the name but mark it as template
        obs_text,
        flags=re.IGNORECASE
    )
    t = re.sub(
        rf'\b({_LOCATION_WORDS.lower()})\s+(\d+)\b',
        r'{\1} \2',
        t,
        flags=re.IGNORECASE
    )
    return t


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ALFWorld Probing Runner")
    print("=" * 60)

    # Discover games
    games_by_type = discover_games()
    print("\nDiscovered games by task type:")
    for tt, files in sorted(games_by_type.items()):
        print(f"  {tt}: {len(files)} games")

    # Select games to probe (take from different subdirectories for variety)
    selected_games = []
    for tt in TASK_TYPES:
        files = games_by_type.get(tt, [])
        if not files:
            print(f"  WARNING: No games found for task type: {tt}")
            continue
        # Pick first and last to get some variety
        if len(files) >= 2:
            selected = [files[0], files[-1]]
        else:
            selected = files[:GAMES_PER_TASK]
        for f in selected:
            selected_games.append((f, tt))
        print(f"  Selected {len(selected)} games for {tt}")

    print(f"\nTotal games to probe: {len(selected_games)}")

    # Run probes
    all_trajectories = []
    game_counter: dict = defaultdict(int)

    for game_file, task_type in selected_games:
        game_counter[task_type] += 1
        idx = game_counter[task_type]

        try:
            trajectory = probe_game(game_file, task_type, idx)
            all_trajectories.append(trajectory)

            # Save individual trajectory
            traj_name = f"{task_type}_{idx}_{trajectory['game_id']}.json"
            traj_path = TRAJECTORIES_DIR / traj_name
            with open(traj_path, "w") as f:
                json.dump(trajectory, f, indent=2)
            print(f"    Saved: {traj_path.name}")

        except Exception as e:
            print(f"  ERROR probing {game_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Probing complete. {len(all_trajectories)} games probed.")

    # Build and save catalog
    print("\nBuilding feedback catalog...")
    catalog = build_feedback_catalog(all_trajectories)

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Saved catalog: {CATALOG_PATH}")
    print(f"\nSummary:")
    print(f"  Total games probed:  {catalog['meta']['total_games_probed']}")
    print(f"  Total steps:         {catalog['meta']['total_steps']}")
    print(f"  Total error probes:  {catalog['meta']['total_error_probes']}")
    print(f"  Distinct patterns:   {len(catalog['feedback_patterns'])}")

    print("\nTop feedback patterns:")
    for pat in catalog["feedback_patterns"][:15]:
        causes_preview = [c["cause"] for c in pat["causes"][:3]]
        print(f"  [{pat['total_occurrences']:4d}x, {pat['distinct_causes']} causes] "
              f"{pat['feedback_text'][:65]!r}")
        print(f"          → {causes_preview}")

    print("\nTask type summaries:")
    for tt, summary in catalog["task_type_summaries"].items():
        print(f"  {tt}:")
        print(f"    games={summary['games_probed']}, "
              f"completed={summary['games_completed']}, "
              f"avg_steps={summary['avg_steps_to_complete']}")
        print(f"    top errors: {summary['common_errors'][:3]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
