#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "verify_report.md"


def run_script(name: str):
    proc = subprocess.run(
        [sys.executable, str(ROOT / name)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {"script": name, "returncode": proc.returncode, "output": proc.stdout}


def main():
    results = [
        run_script("layer1_benchmark_native.py"),
        run_script("layer2_diagnostic_unit.py"),
        run_script("layer3_non_regression.py"),
    ]
    report = ["# WebShop Verification Report", ""]
    for result in results:
        status = "pass" if result["returncode"] == 0 else "fail"
        report.append(f"- `{result['script']}`: {status}")
    report.append("")
    report.append("Layer 1 is intentionally deferred until WebShop runtime dependencies are available.")
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(json.dumps(results, indent=2))
    failed = [result for result in results if result["returncode"] != 0]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
