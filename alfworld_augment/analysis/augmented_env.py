"""
AugmentedAlfWorldEnv: A gym wrapper around ALFWorld TextWorld environments
that replaces uninformative "Nothing happens." feedback (and other observations)
with targeted, diagnostic augmented feedback.

Design principles:
  - Wraps the gym-level interface (obs, infos) produced by textworld.gym.make()
  - Maintains internal state derived from PDDL facts in infos
  - Augments observation text only; never modifies score or done
  - Never leaks the solution path; only provides diagnostic hints
  - All augmentations are logged for analysis

Usage:
    import textworld.gym, textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from analysis.augmented_env import AugmentedAlfWorldEnv

    gamefiles = ["/path/to/game.tw-pddl"]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(gamefiles, request_infos, batch_size=1,
                                           asynchronous=False, max_episode_steps=50,
                                           wrappers=[AlfredDemangler, AlfredInfos])
    base_env = textworld.gym.make(env_id)
    env = AugmentedAlfWorldEnv(base_env)
    obs, infos = env.reset()
    obs, score, done, infos = env.step("some command")
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recognised command verb patterns (for R09 invalid command detection)
# ---------------------------------------------------------------------------
KNOWN_VERB_PATTERNS = [
    re.compile(r"^go to .+$"),
    re.compile(r"^take .+ from .+$"),
    re.compile(r"^move .+ to .+$"),
    re.compile(r"^put .+ in .+$"),          # alternate phrasing
    re.compile(r"^put .+ on .+$"),          # alternate phrasing
    re.compile(r"^open .+$"),
    re.compile(r"^close .+$"),
    re.compile(r"^examine .+$"),
    re.compile(r"^use .+$"),
    re.compile(r"^heat .+ with .+$"),
    re.compile(r"^cool .+ with .+$"),
    re.compile(r"^clean .+ with .+$"),
    re.compile(r"^slice .+ with .+$"),
    re.compile(r"^inventory$"),
    re.compile(r"^look$"),
    re.compile(r"^help$"),
]

NOTHING_HAPPENS = "Nothing happens."


# ---------------------------------------------------------------------------
# Helper: parse PDDL facts into a convenient InternalState
# ---------------------------------------------------------------------------
class InternalState:
    """
    Derived world state built from PDDL Proposition facts.
    All names are human-readable (post-AlfredDemangler).
    """

    def __init__(self, facts: List[Any], admissible_commands: List[str]):
        self.held_object: Optional[str] = None       # name of held object or None
        self.open_containers: set = set()            # receptacle names that are open
        self.openable_containers: set = set()        # receptacle names that can be opened
        # object -> receptacle mapping (where the object currently resides)
        self.object_in_receptacle: Dict[str, str] = {}
        # receptacle -> set of objects
        self.receptacle_contents: Dict[str, set] = {}
        self.admissible_commands = set(admissible_commands)

        self._parse_facts(facts)

    def _parse_facts(self, facts: List[Any]) -> None:
        for fact in facts:
            name = fact.name
            args = [v.name for v in fact.arguments]

            if name == "holds" and len(args) == 2:
                # holds(agent1, object_name)
                self.held_object = args[1]

            elif name == "opened" and len(args) == 1:
                self.open_containers.add(args[0])

            elif name == "openable" and len(args) == 1:
                self.openable_containers.add(args[0])

            elif name == "inreceptacle" and len(args) == 2:
                obj, recep = args[0], args[1]
                self.object_in_receptacle[obj] = recep
                if recep not in self.receptacle_contents:
                    self.receptacle_contents[recep] = set()
                self.receptacle_contents[recep].add(obj)

    def is_holding(self) -> bool:
        return self.held_object is not None

    def is_container_open(self, container: str) -> bool:
        return container in self.open_containers

    def is_container_openable(self, container: str) -> bool:
        return container in self.openable_containers

    def is_container_closed(self, container: str) -> bool:
        return (container in self.openable_containers and
                container not in self.open_containers)

    def object_location(self, obj: str) -> Optional[str]:
        return self.object_in_receptacle.get(obj)

    def objects_in_receptacle(self, recep: str) -> set:
        return self.receptacle_contents.get(recep, set())


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _parse_take_command(command: str) -> Optional[Tuple[str, str]]:
    """Parse 'take <obj> from <recep>' -> (obj, recep) or None."""
    m = re.match(r"^take (.+) from (.+)$", command)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _parse_move_command(command: str) -> Optional[Tuple[str, str]]:
    """Parse 'move <obj> to <recep>' -> (obj, recep) or None."""
    m = re.match(r"^move (.+) to (.+)$", command)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _parse_open_command(command: str) -> Optional[str]:
    """Parse 'open <recep>' -> recep or None."""
    m = re.match(r"^open (.+)$", command)
    return m.group(1).strip() if m else None


def _parse_close_command(command: str) -> Optional[str]:
    """Parse 'close <recep>' -> recep or None."""
    m = re.match(r"^close (.+)$", command)
    return m.group(1).strip() if m else None


def _parse_appliance_command(command: str) -> Optional[Tuple[str, str, str]]:
    """Parse 'heat/cool/clean <obj> with <appliance>' -> (verb, obj, appliance) or None."""
    m = re.match(r"^(heat|cool|clean) (.+) with (.+)$", command)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return None


def _parse_use_command(command: str) -> Optional[str]:
    """Parse 'use <obj>' -> obj or None."""
    m = re.match(r"^use (.+)$", command)
    return m.group(1).strip() if m else None


def _is_known_command_format(command: str) -> bool:
    """Return True if command matches any known verb pattern."""
    cmd_lower = command.lower().strip()
    return any(p.match(cmd_lower) for p in KNOWN_VERB_PATTERNS)


def _extract_object_type(obj_name: str) -> str:
    """Extract base object type from 'apple 1' -> 'apple'."""
    parts = obj_name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return obj_name


def _task_goal_mentions_object(task_obs: str, obj_name: str) -> bool:
    """Check if the task description mentions the object type."""
    if not task_obs:
        return False
    obj_type = _extract_object_type(obj_name).lower()
    return obj_type in task_obs.lower()


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def _check_rule_R01_put_while_empty(command: str, state: InternalState) -> Optional[str]:
    """R01: move X to Y while hands empty."""
    if _parse_move_command(command) is None:
        return None
    if state.is_holding():
        return None
    return "You are not holding anything. You need to pick up an object before you can place it somewhere."


def _check_rule_R02_take_from_closed(command: str, state: InternalState) -> Optional[str]:
    """R02: take X from closed container."""
    parsed = _parse_take_command(command)
    if parsed is None:
        return None
    obj, container = parsed
    if not state.is_container_closed(container):
        return None
    return f"The {container} is closed. You need to open it first before you can take anything from it."


def _check_rule_R03_take_wrong_location(command: str, state: InternalState) -> Optional[str]:
    """R03: take X from Y when X is not in Y (and container is not closed)."""
    parsed = _parse_take_command(command)
    if parsed is None:
        return None
    obj, container = parsed
    # Skip if container is closed (R02 handles that)
    if state.is_container_closed(container):
        return None
    # Skip if agent is already holding something (R07 handles that)
    if state.is_holding():
        return None
    # Check if object exists somewhere else
    actual_location = state.object_location(obj)
    if actual_location is not None and actual_location != container:
        return f"You cannot find {obj} in the {container}. That object is not there. Try looking around other locations."
    return None


def _check_rule_R04_open_already_open(command: str, state: InternalState) -> Optional[str]:
    """R04: open container that is already open or not openable."""
    container = _parse_open_command(command)
    if container is None:
        return None
    if state.is_container_open(container):
        return f"The {container} is already open."
    if not state.is_container_openable(container):
        return f"The {container} is not a container that can be opened or closed — its contents are already accessible."
    return None


def _check_rule_R05_close_already_closed(command: str, state: InternalState) -> Optional[str]:
    """R05: close container that is already closed or not openable."""
    container = _parse_close_command(command)
    if container is None:
        return None
    if state.is_container_closed(container):
        return f"The {container} is already closed."
    if not state.is_container_openable(container):
        return f"The {container} is not a container that can be opened or closed — its contents are already accessible."
    return None


def _check_rule_R06_heat_cool_clean_without_holding(command: str, state: InternalState) -> Optional[str]:
    """R06: heat/cool/clean X without holding X."""
    parsed = _parse_appliance_command(command)
    if parsed is None:
        return None
    verb, obj, appliance = parsed
    if not state.is_holding():
        return f"You are not holding anything. You need to pick up {obj} before you can {verb} it."
    if state.held_object != obj:
        return f"You are not holding {obj} (you are holding {state.held_object}). Pick up {obj} first."
    return None


def _check_rule_R07_pick_up_while_holding(command: str, state: InternalState) -> Optional[str]:
    """R07: take X while already holding something. Must check AFTER R02 to avoid overlap."""
    parsed = _parse_take_command(command)
    if parsed is None:
        return None
    obj, container = parsed
    # Skip if container is closed (R02 handles that)
    if state.is_container_closed(container):
        return None
    if state.is_holding():
        return (f"You are already holding {state.held_object}. "
                f"You can only carry one object at a time. Put it down first.")
    return None


def _check_rule_R08_use_without_holding(command: str, state: InternalState) -> Optional[str]:
    """R08: use <lamp> without holding anything (for look_at_obj tasks)."""
    lamp = _parse_use_command(command)
    if lamp is None:
        return None
    if not state.is_holding():
        return ("You are not holding anything. To examine an object under a lamp, "
                "you must first pick up the object, then use the lamp.")
    return None


def _check_rule_R09_invalid_command(command: str, state: InternalState) -> Optional[str]:
    """R09: completely unrecognized command format."""
    if _is_known_command_format(command):
        return None
    return ("That command is not recognized. Valid actions include: go to, take, move, "
            "open, close, examine, use, heat, cool, clean. Type 'help' to see all commands.")


def _check_rule_R10_progress_hint(obs: str, command: str,
                                   state: InternalState, task_obs: str) -> Optional[str]:
    """R10: agent picks up task-relevant object (positive reinforcement)."""
    m = re.match(r"^You pick up the (.+) from the .+\.$", obs)
    if m is None:
        return None
    obj_name = m.group(1).strip()
    obj_type = _extract_object_type(obj_name)
    if _task_goal_mentions_object(task_obs, obj_name):
        return f"{obs} [You are now holding the {obj_type} needed for this task. Good progress!]"
    return None


def _check_rule_R11_exploration_nothing(obs: str, command: str,
                                        state: InternalState, task_obs: str) -> Optional[str]:
    """R11: agent arrives somewhere with nothing useful."""
    if "you see nothing" not in obs.lower():
        return None
    if not obs.lower().startswith("you arrive"):
        return None
    if state.is_holding():
        return None
    return f"{obs} There is nothing useful here for your current task. Keep exploring."


NOTHING_HAPPENS_RULES = [
    _check_rule_R01_put_while_empty,
    _check_rule_R02_take_from_closed,
    _check_rule_R07_pick_up_while_holding,   # R07 before R03 (holding check first)
    _check_rule_R03_take_wrong_location,
    _check_rule_R04_open_already_open,
    _check_rule_R05_close_already_closed,
    _check_rule_R06_heat_cool_clean_without_holding,
    _check_rule_R08_use_without_holding,
    _check_rule_R09_invalid_command,
]

SUCCESS_OBS_RULES = [
    _check_rule_R10_progress_hint,
    _check_rule_R11_exploration_nothing,
]


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class AugmentedAlfWorldEnv:
    """
    Gym-level wrapper around a TextWorld/ALFWorld environment.

    Intercepts step() and reset() calls, augments uninformative observations
    with targeted diagnostic feedback, and logs all augmentations.

    Parameters
    ----------
    env : gym environment
        The base ALFWorld gym environment (from textworld.gym.make()).
        Must have been registered with request_infos including
        facts=True and admissible_commands=True.
    verbose : bool
        If True, print augmentation messages to stdout.
    """

    def __init__(self, env, verbose: bool = False):
        self._env = env
        self.verbose = verbose

        # Episode-level state
        self._task_description: str = ""
        self._current_state: Optional[InternalState] = None
        self._last_command: str = ""
        self._step_count: int = 0

        # Augmentation log
        self.augmentation_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Gym API passthrough methods
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        return self._env.seed(seed)

    def close(self):
        return self._env.close()

    def render(self, mode="human"):
        return self._env.render(mode)

    @property
    def unwrapped(self):
        return self._env

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Batch-vs-single interface detection
    # ------------------------------------------------------------------

    def _is_batch_env(self, obs) -> bool:
        """Return True if the underlying env uses the batch API (returns tuples/lists)."""
        return isinstance(obs, (list, tuple))

    def _flatten_batch_infos(self, infos: Dict) -> Dict:
        """For batch_size=1 envs, unwrap each info value from its single-element list."""
        return {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                for k, v in infos.items()}

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and return (obs, infos).

        Handles both single-game and batch_size=1 environments transparently.
        """
        result = self._env.reset()
        obs, infos = result

        # Detect and flatten batch interface
        if self._is_batch_env(obs):
            obs = obs[0]
            infos = self._flatten_batch_infos(infos)

        self._step_count = 0
        self._last_command = ""
        self._task_description = obs  # initial obs contains task description
        self.augmentation_log = []

        # Build internal state from facts
        facts = infos.get("facts", []) or []
        admissible = infos.get("admissible_commands", []) or []
        self._current_state = InternalState(facts, admissible)

        return obs, infos

    def step(self, command: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute command, augment observation if needed, return (obs, score, done, infos).

        Handles both single-game and batch_size=1 environments transparently.
        The score and done flag are NEVER modified.
        """
        self._last_command = command.strip().lower()
        self._step_count += 1

        # Save pre-step state for rule evaluation
        pre_step_state = self._current_state

        # Execute in base environment - detect batch API
        raw_result = self._env.step([command])   # batch API always takes a list
        obs_raw, scores_raw, dones_raw, infos_raw = raw_result

        # Unwrap batch results
        if self._is_batch_env(obs_raw):
            obs = obs_raw[0]
            score = scores_raw[0] if isinstance(scores_raw, (list, tuple)) else scores_raw
            done = dones_raw[0] if isinstance(dones_raw, (list, tuple)) else dones_raw
            infos = self._flatten_batch_infos(infos_raw)
        else:
            obs, score, done, infos = obs_raw, scores_raw, dones_raw, infos_raw

        # Update internal state from new facts
        facts = infos.get("facts", []) or []
        admissible = infos.get("admissible_commands", []) or []
        self._current_state = InternalState(facts, admissible)

        # Augment observation using pre-step state (more accurate for error detection)
        state_for_rules = pre_step_state if pre_step_state is not None else self._current_state
        augmented_obs = self._augment(obs, command, state_for_rules)

        return augmented_obs, score, done, infos

    # ------------------------------------------------------------------
    # Augmentation logic
    # ------------------------------------------------------------------

    def _augment(self, obs: str, command: str, state: InternalState) -> str:
        """
        Apply augmentation rules and return (possibly augmented) observation.
        """
        cmd_lower = command.strip().lower()

        # --- "Nothing happens." branch ---
        if obs.strip() == NOTHING_HAPPENS:
            for rule_fn in NOTHING_HAPPENS_RULES:
                result = rule_fn(cmd_lower, state)
                if result is not None:
                    rule_name = rule_fn.__name__.replace("_check_rule_", "")
                    self._log_augmentation(
                        step=self._step_count,
                        command=command,
                        original_obs=obs,
                        augmented_obs=result,
                        rule=rule_name,
                    )
                    return result

            # Fallback: no specific rule matched
            fallback = (
                "Nothing happens. The action could not be performed in the current state. "
                "Check your inventory and the state of nearby objects."
            )
            self._log_augmentation(
                step=self._step_count,
                command=command,
                original_obs=obs,
                augmented_obs=fallback,
                rule="fallback",
            )
            return fallback

        # --- Success observation branch ---
        for rule_fn in SUCCESS_OBS_RULES:
            result = rule_fn(obs, cmd_lower, state, self._task_description)
            if result is not None:
                rule_name = rule_fn.__name__.replace("_check_rule_", "")
                self._log_augmentation(
                    step=self._step_count,
                    command=command,
                    original_obs=obs,
                    augmented_obs=result,
                    rule=rule_name,
                )
                return result

        # No augmentation needed
        return obs

    def _log_augmentation(self, step: int, command: str,
                           original_obs: str, augmented_obs: str, rule: str) -> None:
        """Record an augmentation event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episode_step": step,
            "command": command,
            "original_obs": original_obs,
            "augmented_obs": augmented_obs,
            "rule_applied": rule,
        }
        self.augmentation_log.append(entry)
        if self.verbose:
            print(f"[AugmentedEnv] step={step} rule={rule}")
            print(f"  cmd : {command}")
            print(f"  orig: {original_obs}")
            print(f"  aug : {augmented_obs}")

    def get_augmentation_log(self) -> List[Dict[str, Any]]:
        """Return the full augmentation log for the current episode."""
        return list(self.augmentation_log)
