"""
env2scaffold WebShop observation augmentation.

This module is intentionally standalone: it does not patch verl-agent or the
WebShop benchmark. Wrap a native WebAgentTextEnv-like object with
AugmentedWebShopEnv to append non-leaking diagnostic text to observations while
preserving reward, done, and transition semantics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


AvailableActions = Dict[str, Any]
StateSummary = Dict[str, Any]


def _parse_webshop_action(action: Any) -> Tuple[Optional[str], Optional[str]]:
    if action is None:
        return None, None
    match = re.fullmatch(
        r"\s*(search|click)\[(.*)\]\s*",
        str(action),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None, None
    return match.group(1).lower(), match.group(2).strip().lower()


def _normalized_clickables(available_actions: Optional[AvailableActions]) -> List[str]:
    if not available_actions:
        return []
    return [str(item).lower() for item in available_actions.get("clickables", [])]


def _has_search_bar(available_actions: Optional[AvailableActions]) -> bool:
    if not available_actions:
        return False
    return bool(available_actions.get("has_search_bar"))


def _format_options(options: Dict[str, Any]) -> str:
    if not options:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(options.items()))


def infer_page_name(url: Optional[str], done: bool = False) -> str:
    if done:
        return "done"
    if url is None:
        return "search"
    for page_name in ("search_results", "item_page", "item_sub_page", "done"):
        if page_name in url:
            return page_name
    return "search"


def build_action_feedback(
    action: Any,
    previous_available_actions: Optional[AvailableActions],
) -> Optional[str]:
    """Diagnose only the agent's own submitted action against visible actions."""
    if not action:
        return None

    action_name, action_arg = _parse_webshop_action(action)
    if action_name is None:
        return (
            "The previous action was malformed. Use search[query] when the "
            "search bar is visible, or click[label] with an exact visible label."
        )

    clickables = set(_normalized_clickables(previous_available_actions))
    has_search_bar = _has_search_bar(previous_available_actions)

    if action_name == "search":
        if not action_arg:
            return "The previous search was empty. Search actions need non-empty product keywords."
        if not has_search_bar:
            return (
                "The previous search was outside the displayed action set. The "
                "search bar was not visible on the previous page; use a listed "
                "click action such as click[Back to Search] before reformulating."
            )
        return None

    if action_name == "click":
        if not action_arg:
            return "The previous click target was empty. Use click[label] with an exact visible label."
        if action_arg == "search":
            return "The search button is not a click target. Use search[query] instead."
        if action_arg not in clickables:
            examples = ", ".join(sorted(clickables)[:5])
            if examples:
                return (
                    "The previous click target was not visible on the prior page. "
                    f"Choose an exact listed click label, for example: {examples}."
                )
            return "The previous click target was not visible on the prior page."

    return None


def build_state_feedback(
    available_actions: Optional[AvailableActions],
    state_summary: Optional[StateSummary],
) -> Optional[str]:
    """Surface page/action affordances already implied by the visible state."""
    state_summary = state_summary or {}
    available_actions = available_actions or {}
    page_name = state_summary.get("page_name", "search")
    clickables = _normalized_clickables(available_actions)

    if page_name == "done":
        return None

    if page_name == "search":
        if _has_search_bar(available_actions):
            return "Page cue: search page. Use search[query] with concise product terms from the instruction."
        return None

    if page_name == "search_results":
        product_count = sum(
            1
            for item in clickables
            if item not in {"back to search", "next >", "< prev"}
        )
        parts = ["Page cue: search results"]
        if state_summary.get("query"):
            parts.append(f"current query='{state_summary['query']}'")
        if state_summary.get("page"):
            parts.append(f"page={state_summary['page']}")
        parts.append(f"visible products={product_count}")
        parts.append("click a product id to inspect it, or use a listed navigation/search control")
        return "; ".join(parts) + "."

    if page_name in {"item_page", "item_sub_page"}:
        selected_options = state_summary.get("selected_options") or {}
        option_groups = state_summary.get("option_groups") or []
        missing_groups = [group for group in option_groups if group not in selected_options]
        parts = ["Page cue: product page"]
        if selected_options:
            parts.append(f"selected options: {_format_options(selected_options)}")
        elif option_groups:
            parts.append("selected options: none")
        if missing_groups:
            parts.append(f"unselected option groups: {', '.join(missing_groups)}")
        if "buy now" in clickables:
            parts.append("click[Buy Now] submits the current product and selected options")
        else:
            parts.append("use click[< Prev] to return to the product page before buying")
        return "; ".join(parts) + "."

    return None


def augment_observation(
    observation: str,
    available_actions: Optional[AvailableActions],
    state_summary: Optional[StateSummary] = None,
    previous_action: Any = None,
    previous_available_actions: Optional[AvailableActions] = None,
    done: bool = False,
) -> Tuple[str, List[str]]:
    feedback: List[str] = []
    action_feedback = build_action_feedback(previous_action, previous_available_actions)
    if action_feedback:
        feedback.append(action_feedback)

    if not done:
        state_feedback = build_state_feedback(available_actions, state_summary)
        if state_feedback:
            feedback.append(state_feedback)

    if not feedback:
        return observation, []

    return f"{observation} [SEP] Env feedback: {' '.join(feedback)}", feedback


class AugmentedWebShopEnv:
    """Composition wrapper for WebAgentTextEnv-compatible environments."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any):
        observation, info = self.env.reset(*args, **kwargs)
        available_actions = self._safe_available_actions()
        augmented, feedback = augment_observation(
            observation,
            available_actions,
            state_summary=self._state_summary(done=False),
        )
        info = self._with_feedback_info(info, feedback)
        return augmented, info

    def step(self, action: Any):
        previous_available_actions = self._safe_available_actions()
        observation, reward, done, info = self.env.step(action)
        available_actions = self._safe_available_actions()
        augmented, feedback = augment_observation(
            observation,
            available_actions,
            state_summary=self._state_summary(done=done),
            previous_action=action,
            previous_available_actions=previous_available_actions,
            done=done,
        )
        info = self._with_feedback_info(info, feedback)
        return augmented, reward, done, info

    def _safe_available_actions(self) -> AvailableActions:
        getter = getattr(self.env, "get_available_actions", None)
        if getter is None:
            return {"has_search_bar": False, "clickables": []}
        actions = getter()
        return dict(actions or {})

    def _state_summary(self, done: bool = False) -> StateSummary:
        browser = getattr(self.env, "browser", None)
        url = getattr(browser, "current_url", None)
        summary: StateSummary = {"page_name": infer_page_name(url, done=done)}
        if done:
            return summary

        server = getattr(self.env, "server", None)
        session_id = getattr(self.env, "session", None)
        sessions = getattr(server, "user_sessions", {}) if server is not None else {}
        session = sessions.get(session_id, {}) if session_id is not None else {}

        keywords = session.get("keywords")
        if keywords:
            summary["query"] = " ".join(str(item) for item in keywords)
        if session.get("page") is not None:
            summary["page"] = session["page"]

        selected_options = dict(session.get("options") or {})
        summary["selected_options"] = selected_options

        asin = session.get("asin")
        products = getattr(server, "product_item_dict", {}) if server is not None else {}
        if asin in products:
            product_info = products[asin]
            summary["option_groups"] = sorted((product_info.get("options") or {}).keys())
        return summary

    @staticmethod
    def _with_feedback_info(info: Any, feedback: Iterable[str]) -> Dict[str, Any]:
        info_dict = dict(info or {})
        info_dict["webshop_augmented"] = True
        info_dict["webshop_feedback"] = list(feedback)
        return info_dict


def wrap_webshop_env(env: Any) -> AugmentedWebShopEnv:
    return AugmentedWebShopEnv(env)
