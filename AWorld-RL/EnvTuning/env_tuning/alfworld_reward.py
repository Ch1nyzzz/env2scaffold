from __future__ import annotations

from typing import Any, Dict, Iterable, List


SUCCESS_BONUS = 5.0
POSITIVE_CLIP = 3.0
NEGATIVE_CLIP = -1.0


def _coerce_float_list(values: Iterable[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _iter_metadata_candidates(reward_scores: Dict[str, Any], extra_info: Any) -> Iterable[Any]:
    for key in ("interaction_extras", "step_extras", "extras", "metadata"):
        value = reward_scores.get(key)
        if value is not None:
            yield value
    if extra_info is not None:
        yield extra_info


def _extract_success_flag(reward_scores: Dict[str, Any], extra_info: Any, step_rewards: List[float]) -> bool:
    for candidate in _iter_metadata_candidates(reward_scores, extra_info):
        if isinstance(candidate, dict):
            for key in ("won", "success", "task_completed"):
                if key in candidate:
                    return bool(candidate[key])
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict):
                    for key in ("won", "success", "task_completed"):
                        if key in item and item[key]:
                            return True
    return any(reward >= POSITIVE_CLIP for reward in step_rewards)


def compute_score(
    reward_scores: Dict[str, List[float]],
    ground_truth: list[list] | None,
    extra_info: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Aggregate ALFWorld interaction rewards into a single GRPO scalar.

    The interaction emits per-step shaped rewards derived from the augmented
    environment's internal `progress_reward`. This function sums the clipped
    shaped returns and adds a sparse terminal success bonus.
    """

    user_turn_rewards = _coerce_float_list(reward_scores.get("user_turn_rewards", []))
    clipped_turn_rewards = [
        max(NEGATIVE_CLIP, min(POSITIVE_CLIP, reward))
        for reward in user_turn_rewards
    ]

    progress_return = sum(clipped_turn_rewards)
    success = _extract_success_flag(reward_scores, extra_info, clipped_turn_rewards)
    success_bonus = SUCCESS_BONUS if success else 0.0
    error_turns = sum(1 for reward in clipped_turn_rewards if reward < 0)

    return {
        "score": progress_return + success_bonus,
        "progress_return": progress_return,
        "success_bonus": success_bonus,
        "success": float(success),
        "total_interaction_rounds": len(clipped_turn_rewards),
        "negative_reward_turns": error_turns,
    }
