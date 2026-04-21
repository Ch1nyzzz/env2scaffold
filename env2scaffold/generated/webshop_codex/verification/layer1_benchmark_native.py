#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


RESULT_PATH = Path(__file__).with_name("layer1_benchmark_native_results.json")


def main():
    result = {
        "status": "deferred",
        "reason": (
            "Native WebShop rollout was not executed in this shell because the "
            "active environment is missing WebShop runtime dependencies. The "
            "generated wrapper is standalone and not yet integrated into verl-agent."
        ),
        "required_future_command": (
            "Run paired vanilla vs AugmentedWebShopEnv rollouts after installing "
            "WebShop dependencies, then compare native task_score and success_rate."
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
