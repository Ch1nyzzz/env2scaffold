"""EnvTuning ALFWorld reward: fine-grained progress shaping, normalized to [0, 1]."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


PROGRESS_WEIGHT = 0.5   # half from progress
SUCCESS_WEIGHT = 0.5    # half from final success
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
    Fine-grained progress reward, normalized to [0, 1].

    score = PROGRESS_WEIGHT * progress_ratio + SUCCESS_WEIGHT * success
    - progress_ratio: fraction of positive steps vs total steps, in [0, 1]
    - success: 1.0 if task completed, else 0.0
    Total score is in [0, 1], same scale as vanilla for fair comparison.
    """
    user_turn_rewards = _coerce_float_list(reward_scores.get("user_turn_rewards", []))
    clipped_turn_rewards = [
        max(NEGATIVE_CLIP, min(POSITIVE_CLIP, reward))
        for reward in user_turn_rewards
    ]

    success = _extract_success_flag(reward_scores, extra_info, clipped_turn_rewards)

    # Normalize progress to [0, 1]
    if clipped_turn_rewards:
        positive_sum = sum(max(0.0, r) for r in clipped_turn_rewards)
        max_possible = POSITIVE_CLIP * len(clipped_turn_rewards)
        progress_ratio = min(1.0, positive_sum / max_possible) if max_possible > 0 else 0.0
    else:
        progress_ratio = 0.0

    success_val = 1.0 if success else 0.0
    # Success = full reward; failure = progress only (scaled to [0, 1))
    if success:
        score = 1.0
    else:
        score = progress_ratio * PROGRESS_WEIGHT

    error_turns = sum(1 for reward in clipped_turn_rewards if reward < 0)

    return {
        "score": score,
        "progress_return": progress_ratio,
        "success_bonus": success_val,
        "success": float(success),
        "total_interaction_rounds": len(clipped_turn_rewards),
        "negative_reward_turns": error_turns,
    }
