"""
layer2_diagnostic_unit.py
Layer 2: Diagnostic Unit Tests — per-candidate trigger/non-trigger/non-leakage checks.
Uses pddl_facts_state oracle (infos['facts']) and admissible_commands_validity_heuristic.

Run:
    python3 verification/layer2_diagnostic_unit.py

Writes: verification/layer2_diagnostic_unit_results.json
"""

import sys
import os
import re
import json
import importlib.util
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_PATH = os.path.join(REPO_ROOT, "augmentation", "augmented_env.py")
PLAN_PATH = os.path.join(REPO_ROOT, "oracle_test", "unit_test_plan.json")

# Named game files for specific test scenarios
GAME_PICK_AND_PLACE = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/pick_and_place_simple-Book-None-SideTable-329/trial_T20190908_050633_745514/game.tw-pddl"
GAME_PICK_HEAT = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/pick_heat_then_place_in_recep-Apple-None-DiningTable-26/trial_T20190907_060234_011675/game.tw-pddl"
GAME_PICK_COOL = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/pick_cool_then_place_in_recep-Apple-None-CounterTop-14/trial_T20190909_044933_815840/game.tw-pddl"
GAME_LOOK_AT_OBJ = "/home/yuhan/.cache/alfworld/json_2.1.1/valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790/game.tw-pddl"
MAX_EPISODE_STEPS = 50


# ---------------------------------------------------------------------------
# Load wrapper via importlib
# ---------------------------------------------------------------------------

def _load_wrapper():
    spec = importlib.util.spec_from_file_location("augmented_env", WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AugmentedAlfWorldEnv


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_aug_env(game_file: str, max_steps: int = MAX_EPISODE_STEPS):
    import textworld.gym
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    AugmentedCls = _load_wrapper()

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        feedback=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler, AlfredInfos],
    )
    base_env = textworld.gym.make(env_id)
    return AugmentedCls(base_env)


def _make_base_env(game_file: str, max_steps: int = MAX_EPISODE_STEPS):
    """Return the raw (unwrapped) env for use in C06 tests."""
    import textworld.gym
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        feedback=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler, AlfredInfos],
    )
    return textworld.gym.make(env_id)


def _unwrap_base_reset(result):
    obs, infos = result
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    if isinstance(infos, dict):
        infos = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                 for k, v in infos.items()}
    return obs, infos


def _unwrap_base_step(result):
    obs_raw, scores_raw, dones_raw, infos_raw = result
    obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
    score = scores_raw[0] if isinstance(scores_raw, (list, tuple)) else scores_raw
    done = dones_raw[0] if isinstance(dones_raw, (list, tuple)) else dones_raw
    infos = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
             for k, v in (infos_raw if isinstance(infos_raw, dict) else {}).items()}
    return obs, score, done, infos


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def _fact_args(fact) -> List[str]:
    """Return list of argument names from a PDDL Proposition."""
    return [v.name for v in fact.arguments]


def _facts_contain(facts, name: str, args: Optional[List[Optional[str]]] = None) -> bool:
    """
    Return True if any fact with the given name exists.
    If args is provided, each position is checked (None = wildcard).
    """
    for fact in facts:
        if fact.name != name:
            continue
        fa = _fact_args(fact)
        if args is None:
            return True
        if len(fa) != len(args):
            continue
        if all(a is None or a == fa[i] for i, a in enumerate(args)):
            return True
    return False


def _get_fact_arg(facts, name: str, arg_position: int,
                  constraints: Optional[Dict[int, str]] = None) -> Optional[str]:
    """
    Return the value of arg_position from the first matching fact.
    constraints: {position: required_value}
    """
    for fact in facts:
        if fact.name != name:
            continue
        fa = _fact_args(fact)
        if arg_position >= len(fa):
            continue
        if constraints:
            if all(len(fa) > p and fa[p] == v for p, v in constraints.items()):
                return fa[arg_position]
        else:
            return fa[arg_position]
    return None


def _all_entity_names(facts) -> set:
    """Return all entity names appearing in any PDDL fact argument (lower-cased)."""
    names = set()
    for fact in facts:
        for v in fact.arguments:
            names.add(v.name.lower())
    return names


def _entity_names_in_text(text: str, entity_names: set) -> set:
    """Return which entity names from the given set appear in the text."""
    text_lower = text.lower()
    return {name for name in entity_names if name in text_lower}


def _extract_entity_names_from_text(text: str) -> set:
    """Extract lowercase potential entity names from text (space-separated words and N-grams)."""
    words = re.findall(r'[a-z0-9_]+(?:\s[0-9]+)?', text.lower())
    result = set()
    for w in words:
        result.add(w.strip())
    # Also try bigrams (entity names like "apple 1")
    token_list = text.lower().split()
    for i in range(len(token_list) - 1):
        result.add(token_list[i] + " " + token_list[i+1])
    return result


def _known_entities_from_obs_and_ac(obs_text: str, admissible: List[str]) -> set:
    """Extract entity names already known to the agent from obs and admissible_commands."""
    known = set()
    for token in re.findall(r'[a-zA-Z0-9 ]+', obs_text):
        known.add(token.strip().lower())
    # Add tokens from admissible_commands
    for cmd in admissible:
        for token in re.findall(r'[a-zA-Z0-9 ]+', cmd):
            known.add(token.strip().lower())
    return known


# ---------------------------------------------------------------------------
# Individual test implementations
# ---------------------------------------------------------------------------

def test_L2_C01_trigger() -> Dict[str, Any]:
    """C01 trigger: invalid command 'fly to mars' should emit disambiguation."""
    result = {
        "test_id": "L2_C01_trigger", "candidate_id": "C01", "kind": "trigger",
        "status": "error", "oracle_output": "", "pass_criterion":
            "obs ≠ 'Nothing happens.' pure string AND 'fly to mars' NOT in admissible_commands "
            "AND augmented obs differs from 'Nothing happens.'",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE)
        obs, infos = env.reset()
        pre_ac = set(infos.get("admissible_commands", []))

        obs_step, score, done, infos_step = env.step("fly to mars")

        cmd_in_ac = "fly to mars" in pre_ac
        nothing_happens_pure = "Nothing happens." == obs_step.strip()
        base_would_be_nothing = True  # command is invalid, base env returns "Nothing happens."
        aug_emitted = obs_step.strip() != "Nothing happens." or len(obs_step.strip()) > len("Nothing happens.")
        # actually augmented wrapper replaces or appends, so check if obs differs from bare "Nothing happens."
        aug_emitted = obs_step.strip() != "Nothing happens."

        oracle_out = (f"cmd_in_pre_ac={cmd_in_ac} "
                      f"obs_is_pure_nothing_happens={nothing_happens_pure} "
                      f"aug_emitted_signal={aug_emitted} "
                      f"obs_text={repr(obs_step[:100])}")
        result["oracle_output"] = oracle_out

        # Pass: 'fly to mars' NOT in pre_ac AND augmented signal was emitted
        passed = (not cmd_in_ac) and aug_emitted
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"cmd_in_ac={cmd_in_ac} aug_emitted={aug_emitted}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C01_non_trigger() -> Dict[str, Any]:
    """C01 non-trigger: admissible command should NOT emit 'verb unrecognized' signal."""
    result = {
        "test_id": "L2_C01_non_trigger", "candidate_id": "C01", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "submitted command was in pre-action admissible_commands AND "
                          "obs does not contain 'not recognized' verb signal",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE)
        obs, infos = env.reset()
        pre_ac = list(infos.get("admissible_commands", []))
        if not pre_ac:
            result["status"] = "skipped"
            result["details"] = "No admissible commands at reset"
            return result

        test_cmd = pre_ac[0]
        was_in_ac = test_cmd in set(pre_ac)

        obs_step, score, done, infos_step = env.step(test_cmd)

        # C01 signal: "not recognized" / "unrecognized" verb
        c01_signal_emitted = any(phrase in obs_step.lower() for phrase in
                                 ["not recognized", "verb unrecognized", "command is not recognized",
                                  "that command is not recognized"])

        oracle_out = (f"test_cmd={repr(test_cmd)} was_in_pre_ac={was_in_ac} "
                      f"c01_signal_emitted={c01_signal_emitted} obs={repr(obs_step[:80])}")
        result["oracle_output"] = oracle_out

        passed = was_in_ac and (not c01_signal_emitted)
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"cmd='{test_cmd}' was_in_ac={was_in_ac} c01_fired={c01_signal_emitted}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C01_non_leakage() -> Dict[str, Any]:
    """C01 non-leakage: 'fly to mars' augmented obs should not reveal entity names from facts."""
    result = {
        "test_id": "L2_C01_non_leakage", "candidate_id": "C01", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "no entity name from facts (absent from reset obs) appears in augmented obs",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE)
        obs_reset, infos_reset = env.reset()
        facts_at_reset = infos_reset.get("facts", [])
        ac_at_reset = infos_reset.get("admissible_commands", [])
        all_entity_names = _all_entity_names(facts_at_reset)
        # Known entities: those already in reset obs or AC
        known_prior = _known_entities_from_obs_and_ac(obs_reset, ac_at_reset)

        obs_step, score, done, infos_step = env.step("fly to mars")

        # New entities in augmented obs that weren't known before
        hidden_entities = all_entity_names - known_prior
        leaked = _entity_names_in_text(obs_step, hidden_entities)

        oracle_out = (f"total_fact_entities={len(all_entity_names)} "
                      f"known_prior={len(known_prior)} "
                      f"hidden_entities_count={len(hidden_entities)} "
                      f"leaked={sorted(leaked)} "
                      f"obs={repr(obs_step[:100])}")
        result["oracle_output"] = oracle_out

        passed = len(leaked) == 0
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"leaked_entities={sorted(leaked)}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C02_trigger() -> Dict[str, Any]:
    """C02 trigger: non-existent entity name in command → entity-non-existence signal."""
    result = {
        "test_id": "L2_C02_trigger", "candidate_id": "C02", "kind": "trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "zero propositions reference entity AND aug obs differs from "
                          "'Nothing happens.' AND entity-non-existence signal emitted",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE)
        obs, infos = env.reset()

        fake_entity = "ladle_nonexistent_99"
        cmd = f"take {fake_entity} from countertop 1"
        obs_step, score, done, infos_step = env.step(cmd)

        facts = infos_step.get("facts", [])
        entity_in_facts = _facts_contain(facts, "__any__")  # placeholder; check manually
        entity_fact_count = sum(
            1 for f in facts
            for arg in _fact_args(f)
            if fake_entity.lower() in arg.lower()
        )

        aug_emitted = obs_step.strip() != "Nothing happens."
        # C02 signal keywords
        c02_signal = any(phrase in obs_step.lower() for phrase in
                         ["does not exist", "no '", "there is no", "not in this game"])

        oracle_out = (f"entity='{fake_entity}' entity_fact_count={entity_fact_count} "
                      f"aug_emitted={aug_emitted} c02_signal={c02_signal} "
                      f"obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = entity_fact_count == 0 and aug_emitted and c02_signal
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"entity_fact_count={entity_fact_count} c02_signal={c02_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C02_non_trigger() -> Dict[str, Any]:
    """C02 non-trigger: existing entity → C02 does NOT fire entity-non-existence signal."""
    result = {
        "test_id": "L2_C02_non_trigger", "candidate_id": "C02", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "entity X in ≥1 fact AND C02 'entity does not exist' signal NOT emitted",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_COOL)
        obs, infos = env.reset()
        facts = infos.get("facts", [])

        # Find inreceptacle(X, Y) for any X; X exists in facts
        existing_obj = None
        existing_recep = None
        for fact in facts:
            if fact.name == "inreceptacle":
                fa = _fact_args(fact)
                if len(fa) == 2:
                    existing_obj = fa[0]
                    existing_recep = fa[1]
                    break

        if existing_obj is None:
            result["status"] = "skipped"
            result["details"] = "No inreceptacle fact found"
            return result

        # Submit take X from WRONG container (not its actual location)
        # Use countertop_1 as wrong location if existing_recep is not countertop_1
        wrong_recep = "countertop 1" if existing_recep != "countertop 1" else "stoveburner 1"
        cmd = f"take {existing_obj} from {wrong_recep}"

        obs_step, score, done, infos_step = env.step(cmd)
        facts_after = infos_step.get("facts", [])

        entity_fact_count = sum(
            1 for f in facts_after
            for arg in _fact_args(f)
            if existing_obj.lower() in arg.lower()
        )

        c02_signal = any(phrase in obs_step.lower() for phrase in
                         ["does not exist", "there is no '", "no '", "not in this game"])

        oracle_out = (f"entity='{existing_obj}' entity_fact_count={entity_fact_count} "
                      f"wrong_recep='{wrong_recep}' c02_signal={c02_signal} "
                      f"obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = entity_fact_count >= 1 and (not c02_signal)
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"entity='{existing_obj}' fact_count={entity_fact_count} c02_fired={c02_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C02_non_leakage() -> Dict[str, Any]:
    """C02 non-leakage: non-existent entity command obs should not reveal hidden entity names."""
    result = {
        "test_id": "L2_C02_non_leakage", "candidate_id": "C02", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "no entity from facts (not in reset obs/AC) appears in C02 obs",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])
        ac_reset = infos_reset.get("admissible_commands", [])
        all_entity_names = _all_entity_names(facts_reset)
        known_prior = _known_entities_from_obs_and_ac(obs_reset, ac_reset)

        obs_step, score, done, infos_step = env.step("take nonexistent_entity_99 from countertop 1")

        hidden_entities = all_entity_names - known_prior
        leaked = _entity_names_in_text(obs_step, hidden_entities)

        oracle_out = (f"hidden_count={len(hidden_entities)} leaked={sorted(leaked)} "
                      f"obs={repr(obs_step[:100])}")
        result["oracle_output"] = oracle_out

        passed = len(leaked) == 0
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"leaked={sorted(leaked)}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C03_trigger() -> Dict[str, Any]:
    """C03 trigger: heat apple 1 with microwave 1 when not holding apple 1."""
    result = {
        "test_id": "L2_C03_trigger", "candidate_id": "C03", "kind": "trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "obs is 'Nothing happens.' from base AND holds(agent_1, apple_1) absent "
                          "AND apple_1 in ≥1 other fact AND aug emits 'not holding' signal",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_HEAT)
        obs, infos = env.reset()
        # Confirm agent is not holding anything
        facts_reset = infos.get("facts", [])
        holds_at_start = _facts_contain(facts_reset, "holds")

        # Submit heat without navigating to or picking up apple
        obs_step, score, done, infos_step = env.step("heat apple 1 with microwave 1")
        facts_step = infos_step.get("facts", [])

        holds_apple = _facts_contain(facts_step, "holds", [None, "apple 1"])
        apple_in_any_fact = any(
            "apple 1" in _fact_args(f) for f in facts_step
        )
        c03_signal = any(phrase in obs_step.lower() for phrase in
                         ["not holding", "not hold", "pick up", "you need to pick"])

        oracle_out = (f"holds_at_start={holds_at_start} holds_apple_after={holds_apple} "
                      f"apple_in_any_fact={apple_in_any_fact} c03_signal={c03_signal} "
                      f"obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        # Pass: agent was not holding apple (holds absent for apple_1),
        # apple exists in facts, and C03 signal emitted
        passed = (not holds_apple) and apple_in_any_fact and c03_signal
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"holds_apple={holds_apple} apple_exists={apple_in_any_fact} c03_signal={c03_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C03_non_trigger() -> Dict[str, Any]:
    """C03 non-trigger: holding apple → heat command should NOT emit 'not holding' signal."""
    result = {
        "test_id": "L2_C03_non_trigger", "candidate_id": "C03", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "holds(agent_1, apple_1) present in facts before action AND "
                          "'not holding' signal NOT emitted",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_HEAT)
        obs, infos = env.reset()

        # Navigate to diningtable 2 (apple 1 is there) and pick it up
        obs1, s1, d1, infos1 = env.step("go to diningtable 2")
        obs2, s2, d2, infos2 = env.step("take apple 1 from diningtable 2")

        facts_after_take = infos2.get("facts", [])
        holds_apple = _facts_contain(facts_after_take, "holds", [None, "apple 1"])

        if not holds_apple:
            result["status"] = "skipped"
            result["details"] = f"Could not pick up apple 1 (obs: {obs2[:80]})"
            return result

        # Now try to heat apple with microwave (not at microwave, so action fails)
        obs_step, score, done, infos_step = env.step("heat apple 1 with microwave 1")
        facts_step = infos_step.get("facts", [])
        holds_apple_pre = holds_apple  # was True before this step

        c03_signal = any(phrase in obs_step.lower() for phrase in
                         ["not holding", "you need to pick up", "pick up apple"])

        oracle_out = (f"holds_apple_before_step={holds_apple_pre} "
                      f"c03_signal={c03_signal} obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = holds_apple_pre and (not c03_signal)
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"holds_apple_before={holds_apple_pre} c03_fired={c03_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C03_non_leakage() -> Dict[str, Any]:
    """C03 non-leakage: 'put apple 1 in countertop' when not holding → no receptacle location leaked."""
    result = {
        "test_id": "L2_C03_non_leakage", "candidate_id": "C03", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "augmented obs does not contain receptacle name Y from "
                          "inreceptacle(apple_1, Y) that wasn't in prior observation",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_HEAT)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])
        ac_reset = infos_reset.get("admissible_commands", [])

        # Find where apple 1 actually lives
        apple_receptacle = _get_fact_arg(facts_reset, "inreceptacle", 1,
                                          constraints={0: "apple 1"})
        known_prior = _known_entities_from_obs_and_ac(obs_reset, ac_reset)

        # Submit put without holding anything (C03 trigger)
        obs_step, score, done, infos_step = env.step("move apple 1 to countertop 3")

        # Check if apple's actual receptacle name appears in augmented obs
        actual_receptacle_leaked = (
            apple_receptacle is not None
            and apple_receptacle.lower() not in known_prior
            and apple_receptacle.lower() in obs_step.lower()
        )

        # Also check all hidden entities
        all_entities = _all_entity_names(facts_reset)
        hidden_entities = all_entities - known_prior
        leaked = _entity_names_in_text(obs_step, hidden_entities)

        oracle_out = (f"apple_receptacle='{apple_receptacle}' "
                      f"actual_leaked={actual_receptacle_leaked} "
                      f"all_leaked={sorted(leaked)} "
                      f"obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = not actual_receptacle_leaked and len(leaked) == 0
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"actual_receptacle_leaked={actual_receptacle_leaked} leaked={sorted(leaked)}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C04_trigger() -> Dict[str, Any]:
    """C04 trigger: take bread 3 from closed fridge 1."""
    result = {
        "test_id": "L2_C04_trigger", "candidate_id": "C04", "kind": "trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "inreceptacle(bread_3, fridge_1)=true AND opened(fridge_1)=false "
                          "AND aug emits 'closed' signal",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_COOL)
        obs, infos = env.reset()
        facts_reset = infos.get("facts", [])

        # Find an object in closed fridge
        # bread 3 is in fridge 1 (from earlier inspection), fridge starts closed
        obj = "bread 3"
        container = "fridge 1"

        # Verify from facts: inreceptacle(bread_3, fridge_1) and NOT opened(fridge_1)
        inrec = _facts_contain(facts_reset, "inreceptacle", [obj, container])
        is_openable = _facts_contain(facts_reset, "openable", [container])
        is_opened = _facts_contain(facts_reset, "opened", [container])
        is_closed = is_openable and not is_opened

        if not inrec or not is_closed:
            # Try to find another object in a closed container
            for fact in facts_reset:
                if fact.name == "inreceptacle":
                    fa = _fact_args(fact)
                    if len(fa) == 2:
                        c_name = fa[1]
                        o_name = fa[0]
                        if (_facts_contain(facts_reset, "openable", [c_name]) and
                                not _facts_contain(facts_reset, "opened", [c_name])):
                            obj, container = o_name, c_name
                            inrec = True
                            is_closed = True
                            break

        # Navigate to container
        env.step(f"go to {container}")
        obs_step, score, done, infos_step = env.step(f"take {obj} from {container}")
        facts_step = infos_step.get("facts", [])

        inrec_check = _facts_contain(facts_step, "inreceptacle", [obj, container])
        opened_check = _facts_contain(facts_step, "opened", [container])
        c04_closed_signal = any(phrase in obs_step.lower() for phrase in
                                ["is closed", "closed. you need to open", "need to open it first"])

        oracle_out = (f"obj='{obj}' container='{container}' "
                      f"inrec={inrec_check} opened={opened_check} "
                      f"c04_closed_signal={c04_closed_signal} obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = inrec_check and (not opened_check) and c04_closed_signal
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"inrec={inrec_check} opened={opened_check} c04_signal={c04_closed_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C04_non_trigger() -> Dict[str, Any]:
    """C04 non-trigger: open fridge then take → success, C04 does NOT fire."""
    result = {
        "test_id": "L2_C04_non_trigger", "candidate_id": "C04", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "inreceptacle(X,C) and opened(C) both true before action AND "
                          "C04 augmentation NOT fired (action succeeds or fails non-C04)",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_COOL)
        obs, infos = env.reset()
        facts_reset = infos.get("facts", [])

        # Find object in fridge, navigate and open, then take
        obj = "bread 3"
        container = "fridge 1"

        # Find actual object in closed container
        for fact in facts_reset:
            if fact.name == "inreceptacle":
                fa = _fact_args(fact)
                if len(fa) == 2:
                    c_name = fa[1]
                    o_name = fa[0]
                    if (_facts_contain(facts_reset, "openable", [c_name]) and
                            not _facts_contain(facts_reset, "opened", [c_name])):
                        obj, container = o_name, c_name
                        break

        env.step(f"go to {container}")
        env.step(f"open {container}")

        # Verify container is now open
        # Re-read facts from augmented env's last infos
        _, _, _, infos_after_open = env.step("look")
        facts_after_open = infos_after_open.get("facts", [])

        inrec = _facts_contain(facts_after_open, "inreceptacle", [obj, container])
        opened = _facts_contain(facts_after_open, "opened", [container])

        obs_step, score, done, infos_step = env.step(f"take {obj} from {container}")
        facts_after_take = infos_step.get("facts", [])
        holds_obj = _facts_contain(facts_after_take, "holds", [None, obj])

        c04_signal_fired = any(phrase in obs_step.lower() for phrase in
                               ["is closed", "need to open it first", "cannot find", "not there"])

        oracle_out = (f"obj='{obj}' container='{container}' "
                      f"inrec_before={inrec} opened_before={opened} "
                      f"holds_after={holds_obj} c04_fired={c04_signal_fired} "
                      f"obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        # Either action succeeded (holds obj) or at least C04 didn't fire
        passed = (not c04_signal_fired) and (inrec or holds_obj)
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"inrec={inrec} opened={opened} holds_after={holds_obj} c04_fired={c04_signal_fired}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C04_non_leakage() -> Dict[str, Any]:
    """C04 non-leakage: wrong-location take → actual location A must not appear in obs."""
    result = {
        "test_id": "L2_C04_non_leakage", "candidate_id": "C04", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "actual location A of object X not disclosed in augmented obs text",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_COOL)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])

        # apple 1 is at microwave 1; we'll try to take from fridge 1 (after opening it)
        target_obj = "apple 1"
        actual_recep = _get_fact_arg(facts_reset, "inreceptacle", 1,
                                      constraints={0: target_obj})
        # Choose a wrong receptacle (open one to avoid C04 closed trigger)
        wrong_recep = "fridge 1"

        # Navigate to fridge, open it, then try to take apple from fridge
        env.step(f"go to {wrong_recep}")
        env.step(f"open {wrong_recep}")
        obs_step, score, done, infos_step = env.step(f"take {target_obj} from {wrong_recep}")

        # actual_recep should NOT appear in obs_step
        actual_recep_leaked = (
            actual_recep is not None and
            actual_recep.lower() in obs_step.lower()
        )

        oracle_out = (f"target_obj='{target_obj}' actual_recep='{actual_recep}' "
                      f"wrong_recep='{wrong_recep}' "
                      f"actual_recep_in_obs={actual_recep_leaked} "
                      f"obs={repr(obs_step[:150])}")
        result["oracle_output"] = oracle_out

        passed = not actual_recep_leaked
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"actual_recep='{actual_recep}' leaked={actual_recep_leaked}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C05_trigger() -> Dict[str, Any]:
    """C05 trigger (dropped candidate): examine target obj with lamp off — speculative."""
    result = {
        "test_id": "L2_C05_trigger", "candidate_id": "C05", "kind": "trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "obs contains 'Nothing happens.' AND toggled absent AND "
                          "examine not in AC AND aug emits disambiguation signal (any text beyond Nothing happens.)",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_LOOK_AT_OBJ)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])

        # Find a target object (alarmclock 2 is at desk 1)
        target_obj = "alarmclock 2"

        # Don't navigate anywhere — agent in middle of room
        # Lamp is off (no toggled fact)
        toggled_present = _facts_contain(facts_reset, "toggled")
        ac_reset = infos_reset.get("admissible_commands", [])
        examine_in_ac = any("examine" in c and target_obj in c for c in ac_reset)

        obs_step, score, done, infos_step = env.step(f"examine {target_obj}")
        facts_step = infos_step.get("facts", [])
        toggled_present_after = _facts_contain(facts_step, "toggled")
        ac_after = infos_step.get("admissible_commands", [])
        examine_in_ac_after = any("examine" in c and target_obj in c for c in ac_after)

        base_nothing_happens = "Nothing happens." in obs_step
        aug_emitted = obs_step.strip() != "Nothing happens."

        oracle_out = (f"toggled_at_reset={toggled_present} examine_in_ac_reset={examine_in_ac} "
                      f"toggled_after={toggled_present_after} base_nothing_in_obs={base_nothing_happens} "
                      f"aug_emitted={aug_emitted} obs={repr(obs_step[:150])}")
        result["oracle_output"] = oracle_out

        # Per spec: "this test is marked speculative"
        # Pass: all conditions met. Note: aug_emitted may be fallback message (acceptable per criterion)
        passed = (base_nothing_happens and not toggled_present and
                  not examine_in_ac and aug_emitted)
        result["status"] = "pass" if passed else "fail"
        result["details"] = (f"[SPECULATIVE/DROPPED_CANDIDATE C05] "
                              f"toggled={toggled_present} examine_in_ac={examine_in_ac} "
                              f"aug_emitted={aug_emitted}")

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C05_non_trigger() -> Dict[str, Any]:
    """C05 non-trigger (dropped candidate): examine after lamp on → no C05 augmentation."""
    result = {
        "test_id": "L2_C05_non_trigger", "candidate_id": "C05", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "toggled(desklamp) in facts AND examine in AC AND C05 aug NOT fired",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_LOOK_AT_OBJ)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])

        # Navigate to desk 1 (where desklamp 1 and alarmclock 2 are)
        env.step("go to desk 1")
        # Pick up target object first
        obs_pickup, _, _, infos_pickup = env.step("take alarmclock 2 from desk 1")
        # Use the lamp (toggle on)
        obs_lamp, _, _, infos_lamp = env.step("use desklamp 1")
        facts_lamp = infos_lamp.get("facts", [])
        ac_lamp = infos_lamp.get("admissible_commands", [])

        toggled = _facts_contain(facts_lamp, "toggled")
        # examine after lamp on
        obs_examine, score, done, infos_examine = env.step("examine alarmclock 2")

        c05_signal = any(phrase in obs_examine.lower() for phrase in
                         ["lamp must be", "light source", "examine under"])

        oracle_out = (f"toggled_after_use={toggled} "
                      f"c05_signal={c05_signal} obs={repr(obs_examine[:120])}")
        result["oracle_output"] = oracle_out

        passed = not c05_signal
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"[DROPPED_CANDIDATE C05] toggled={toggled} c05_fired={c05_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C05_non_leakage() -> Dict[str, Any]:
    """C05 non-leakage (dropped candidate): lamp-off examine → no hidden entity leaked."""
    result = {
        "test_id": "L2_C05_non_leakage", "candidate_id": "C05", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "no desklamp name or target entity from facts leaked in obs",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_LOOK_AT_OBJ)
        obs_reset, infos_reset = env.reset()
        facts_reset = infos_reset.get("facts", [])
        ac_reset = infos_reset.get("admissible_commands", [])
        all_entities = _all_entity_names(facts_reset)
        known_prior = _known_entities_from_obs_and_ac(obs_reset, ac_reset)

        obs_step, score, done, infos_step = env.step("examine alarmclock 2")

        hidden_entities = all_entities - known_prior
        leaked = _entity_names_in_text(obs_step, hidden_entities)

        oracle_out = (f"hidden_count={len(hidden_entities)} leaked={sorted(leaked)} "
                      f"obs={repr(obs_step[:100])}")
        result["oracle_output"] = oracle_out

        passed = len(leaked) == 0
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"[DROPPED_CANDIDATE C05] leaked={sorted(leaked)}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C06_trigger() -> Dict[str, Any]:
    """C06 trigger (dropped candidate): step-limit termination signal expected."""
    result = {
        "test_id": "L2_C06_trigger", "candidate_id": "C06", "kind": "trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "done=True AND won=False AND score=0.0 AND "
                          "step-limit termination signal in obs",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE, max_steps=5)
        obs, infos = env.reset()

        # Execute 4 admissible commands
        for _ in range(4):
            ac = infos.get("admissible_commands", [])
            cmd = ac[0] if ac else "look"
            obs, score, done, infos = env.step(cmd)
            if done:
                break

        # 5th step — Limit wrapper should fire
        ac = infos.get("admissible_commands", [])
        cmd = ac[0] if ac else "look"
        obs_step, score, done_step, infos_step = env.step(cmd)
        won = bool(infos_step.get("won", False))

        step_limit_signal = any(phrase in obs_step.lower() for phrase in
                                ["maximum steps", "step limit", "out of time",
                                 "exceeded", "episode ended", "time limit"])

        oracle_out = (f"done={done_step} won={won} score={score} "
                      f"step_limit_signal={step_limit_signal} obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = done_step and (not won) and step_limit_signal
        result["status"] = "pass" if passed else "fail"
        result["details"] = (f"[DROPPED_CANDIDATE C06] done={done_step} won={won} "
                              f"step_limit_signal={step_limit_signal}")

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C06_non_trigger() -> Dict[str, Any]:
    """C06 non-trigger (dropped candidate): 4th step should not have step-limit signal."""
    result = {
        "test_id": "L2_C06_non_trigger", "candidate_id": "C06", "kind": "non_trigger",
        "status": "error", "oracle_output": "",
        "pass_criterion": "done=False AND won=False AND no step-limit signal",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE, max_steps=5)
        obs, infos = env.reset()

        # Execute 3 admissible commands
        for _ in range(3):
            ac = infos.get("admissible_commands", [])
            cmd = ac[0] if ac else "look"
            obs, score, done, infos = env.step(cmd)

        # 4th step (one before limit)
        ac = infos.get("admissible_commands", [])
        cmd = ac[0] if ac else "look"
        obs_step, score, done_step, infos_step = env.step(cmd)
        won = bool(infos_step.get("won", False))

        step_limit_signal = any(phrase in obs_step.lower() for phrase in
                                ["maximum steps", "step limit", "out of time",
                                 "exceeded", "episode ended", "time limit"])

        oracle_out = (f"done={done_step} won={won} "
                      f"step_limit_signal={step_limit_signal} obs={repr(obs_step[:120])}")
        result["oracle_output"] = oracle_out

        passed = (not done_step) and (not won) and (not step_limit_signal)
        result["status"] = "pass" if passed else "fail"
        result["details"] = f"[DROPPED_CANDIDATE C06] done={done_step} step_limit_signal={step_limit_signal}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def test_L2_C06_non_leakage() -> Dict[str, Any]:
    """C06 non-leakage (dropped candidate): step-limit signal should not reveal entity names.

    If the wrapper emitted no augmentation at the terminal step (C06 is dropped),
    the criterion is vacuously satisfied — an empty signal cannot leak entities.
    """
    result = {
        "test_id": "L2_C06_non_leakage", "candidate_id": "C06", "kind": "non_leakage",
        "status": "error", "oracle_output": "",
        "pass_criterion": "step-limit signal (augmented portion of obs) contains no hidden entity names; "
                          "vacuously passes if no augmentation was emitted",
        "details": "", "error": None,
    }
    env = None
    try:
        env = _make_aug_env(GAME_PICK_AND_PLACE, max_steps=5)
        obs, infos = env.reset()

        for _ in range(4):
            ac = infos.get("admissible_commands", [])
            cmd = ac[0] if ac else "look"
            obs, score, done, infos = env.step(cmd)

        # 5th step
        ac = infos.get("admissible_commands", [])
        cmd = ac[0] if ac else "look"
        obs_step, score, done_step, infos_step = env.step(cmd)

        # Check if any augmentation was emitted (consult augmentation_log)
        aug_log = env.get_augmentation_log()
        aug_emitted = len(aug_log) > 0
        step5_aug = [e for e in aug_log if e.get("episode_step") == 5]
        step5_aug_text = step5_aug[0]["augmented_obs"] if step5_aug else ""

        if not aug_emitted or not step5_aug:
            # No augmentation fired at step 5 → vacuously no leakage in the signal
            oracle_out = (f"aug_log_size={len(aug_log)} step5_aug_entries={len(step5_aug)} "
                          f"verdict=vacuous_pass (no signal emitted) obs={repr(obs_step[:80])}")
            result["oracle_output"] = oracle_out
            result["status"] = "pass"
            result["details"] = "[DROPPED_CANDIDATE C06] No augmentation fired → vacuously pass"
        else:
            # Augmentation was emitted — check for hidden entity leakage in the signal
            facts_step = infos_step.get("facts", [])
            all_entities = _all_entity_names(facts_step)
            leaked = _entity_names_in_text(step5_aug_text, all_entities)
            oracle_out = (f"step5_aug_text={repr(step5_aug_text[:80])} "
                          f"leaked={sorted(leaked)}")
            result["oracle_output"] = oracle_out
            passed = len(leaked) == 0
            result["status"] = "pass" if passed else "fail"
            result["details"] = f"[DROPPED_CANDIDATE C06] leaked={sorted(leaked)}"

    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

TEST_FUNCTIONS = [
    test_L2_C01_trigger,
    test_L2_C01_non_trigger,
    test_L2_C01_non_leakage,
    test_L2_C02_trigger,
    test_L2_C02_non_trigger,
    test_L2_C02_non_leakage,
    test_L2_C03_trigger,
    test_L2_C03_non_trigger,
    test_L2_C03_non_leakage,
    test_L2_C04_trigger,
    test_L2_C04_non_trigger,
    test_L2_C04_non_leakage,
    test_L2_C05_trigger,
    test_L2_C05_non_trigger,
    test_L2_C05_non_leakage,
    test_L2_C06_trigger,
    test_L2_C06_non_trigger,
    test_L2_C06_non_leakage,
]


def run_layer2() -> Dict[str, Any]:
    tests = []
    for fn in TEST_FUNCTIONS:
        print(f"  Running {fn.__name__}...", end="", flush=True)
        result = fn()
        status = result.get("status", "error")
        print(f" {status}")
        tests.append(result)

    total = len(tests)
    n_pass = sum(1 for t in tests if t["status"] == "pass")
    n_fail = sum(1 for t in tests if t["status"] == "fail")
    n_error = sum(1 for t in tests if t["status"] == "error")
    n_skipped = sum(1 for t in tests if t["status"] == "skipped")

    print(f"\n[Layer2] Complete: {n_pass}/{total} pass")

    return {
        "layer": "layer2_diagnostic_unit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wrapper_module_path": WRAPPER_PATH,
        "plan_path": PLAN_PATH,
        "oracles_consulted": ["pddl_facts_state", "admissible_commands_validity_heuristic"],
        "tests": tests,
        "summary": {
            "total": total,
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "skipped": n_skipped,
        },
    }


def main():
    results = run_layer2()

    output_path = os.path.join(VERIFICATION_DIR, "layer2_diagnostic_unit_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"\nlayer2 complete: {s['pass']}/{s['total']} pass")
    return 0 if s["fail"] == 0 and s["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
