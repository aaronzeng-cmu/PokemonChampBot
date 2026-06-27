#!/usr/bin/env python3
"""Correlate raw top-1 collapse (action index 1) with trajectory depth in traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit_trace(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    battles = data if isinstance(data, list) else data.get("battles", [data])
    print(f"Audit: {path}")
    for battle in battles:
        tag = battle.get("battle_tag", "?")
        print(f"\n=== {tag} ===")
        for step in battle.get("decisions", []):
            if step.get("kind") != "inference":
                continue
            frames = step.get("trajectory_frames", [])
            nonempty = sum(1 for f in frames if "empty" not in f)
            raw0 = step.get("slot0", {}).get("raw_top1", {}).get("index")
            raw1 = step.get("slot1", {}).get("raw_top1", {}).get("index")
            collapsed = raw0 == 1 or raw1 == 1
            fb = step.get("any_fallback", False)
            print(
                f"  d{step.get('decision_index')} turn {step.get('turn')} "
                f"frames={nonempty}/3 raw=({raw0},{raw1}) "
                f"collapse_idx1={collapsed} fallback={fb}"
            )
            if frames:
                print(f"    {' | '.join(frames)}")


def main() -> None:
    traces = sorted(Path("logs/eval/inference_trace").glob("*/inference_trace_latest.json"))
    if not traces:
        print("No trace JSON found")
        return
    audit_trace(traces[-1])


if __name__ == "__main__":
    main()
