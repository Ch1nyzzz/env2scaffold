#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "augmentation" / "augmented_env.py"
RESULT_PATH = Path(__file__).with_name("layer3_non_regression_results.json")


def load_module():
    spec = importlib.util.spec_from_file_location("webshop_augmented_env", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeBrowser:
    current_url = "http://127.0.0.1:3000/item_page/session/B001/query/1/{}"


class FakeServer:
    product_item_dict = {
        "B001": {"options": {"color": ["red", "blue"], "size": ["small", "large"]}}
    }
    user_sessions = {
        "session": {
            "keywords": ["mug"],
            "page": 1,
            "asin": "B001",
            "options": {"color": "red"},
        }
    }


class FakeWebShopEnv:
    browser = FakeBrowser()
    server = FakeServer()
    session = "session"

    def reset(self):
        return "reset observation", {"native": "reset"}

    def step(self, action):
        return "native observation", 0.75, False, {"native": action}

    def get_available_actions(self):
        return {
            "has_search_bar": False,
            "clickables": ["back to search", "< prev", "buy now", "blue", "large"],
        }


def run_tests():
    module = load_module()
    native = FakeWebShopEnv()
    wrapped = module.AugmentedWebShopEnv(native)

    obs, info = wrapped.reset()
    reset_passed = "Env feedback:" in obs and info["native"] == "reset"

    obs, reward, done, info = wrapped.step("click[buy now]")
    step_passed = (
        reward == 0.75
        and done is False
        and info["native"] == "click[buy now]"
        and info["webshop_augmented"] is True
        and "Env feedback:" in obs
    )

    result = {
        "status": "pass" if reset_passed and step_passed else "fail",
        "cases": [
            {"name": "reset_info_preserved", "passed": reset_passed},
            {"name": "step_reward_done_info_preserved", "passed": step_passed},
        ],
    }
    return result


if __name__ == "__main__":
    result = run_tests()
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
