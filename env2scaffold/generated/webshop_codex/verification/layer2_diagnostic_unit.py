#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "augmentation" / "augmented_env.py"
RESULT_PATH = Path(__file__).with_name("layer2_diagnostic_unit_results.json")


def load_module():
    spec = importlib.util.spec_from_file_location("webshop_augmented_env", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_tests():
    module = load_module()
    cases = []

    obs, feedback = module.augment_observation(
        "Instruction: find a mug",
        {"has_search_bar": True, "clickables": []},
        {"page_name": "search"},
        previous_action="nonsense",
        previous_available_actions={"has_search_bar": True, "clickables": []},
    )
    cases.append({
        "name": "malformed_action",
        "passed": "malformed" in " ".join(feedback).lower() and "Env feedback:" in obs,
    })

    obs, feedback = module.augment_observation(
        "Page 1 results",
        {"has_search_bar": False, "clickables": ["b001", "back to search"]},
        {"page_name": "search_results", "query": "mug", "page": 1},
        previous_action="click[b999]",
        previous_available_actions={"has_search_bar": False, "clickables": ["b001", "back to search"]},
    )
    text = " ".join(feedback).lower()
    cases.append({
        "name": "invalid_click_target",
        "passed": "not visible" in text and "b001" in text and "b999" not in text,
    })

    obs, feedback = module.augment_observation(
        "Product page",
        {"has_search_bar": False, "clickables": ["back to search", "< prev", "buy now"]},
        {
            "page_name": "item_page",
            "selected_options": {"color": "red"},
            "option_groups": ["color", "size"],
        },
        previous_action="search[new query]",
        previous_available_actions={"has_search_bar": False, "clickables": ["back to search", "< prev", "buy now"]},
    )
    text = " ".join(feedback).lower()
    cases.append({
        "name": "product_page_options_and_search_guard",
        "passed": "search bar was not visible" in text and "selected options" in text and "size" in text,
    })

    return {
        "status": "pass" if all(case["passed"] for case in cases) else "fail",
        "cases": cases,
    }


if __name__ == "__main__":
    result = run_tests()
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
