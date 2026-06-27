#!/usr/bin/env python3
"""Run all offline unit tests."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    ROOT / "tests" / "test_combo_and_reward.py",
    ROOT / "tests" / "test_mask_integration.py",
    ROOT / "tests" / "test_planning.py",
]


def main() -> int:
    for path in TESTS:
        print(f"Running {path.name}...")
        rc = subprocess.call([sys.executable, str(path)], cwd=str(ROOT))
        if rc != 0:
            return rc
    print("All offline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
