#!/usr/bin/env python3
"""
Parser sanity check — prints ground-truth actions for leak + target audit.

Fails if common moves resolve to impossible targets (e.g. Protect -> ally).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.doubles.data.action_codec import format_log_action_pair
from src.doubles.data.action_space_spec import TARGET_DEFAULT, log_action_space_decision
from src.doubles.data.replay_parser import parse_log_file, sample_audit_dict
from src.core.data.state_tokenizer import (
    N_FIELDS,
    STACKED_N_TOKENS,
    TEMPORAL_LAST_MOVE_ID,
    TEMPORAL_PROTECT_COUNTER,
    TEMPORAL_TURNS_ACTIVE,
    encode_log_state,
    human_readable_state,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_logs"
OUT_DIR = Path(__file__).resolve().parents[1] / "logs" / "parser_sanity"

# Replays known to contain target-sensitive moves in early turns.
TARGET_AUDIT_LOGS = [
    "gen9championsvgc2026regma-2616376406.log",  # Protect
    "gen9championsvgc2026regma-2616379577.log",  # Trick Room
    "gen9championsvgc2026regma-2616378822.log",  # Knock Off
]

# Fake Out turn 1 + Protect follow-up for temporal fields.
TEMPORAL_AUDIT_LOG = "gen9championsvgc2026regma-2620741830.log"


def _target_checks(action_text: str, move_hint: str) -> list[dict]:
    checks: list[dict] = []
    lower = action_text.lower()
    if "protect" in move_hint.lower() or "protect" in lower:
        checks.append(
            {
                "check": "protect_not_ally",
                "pass": "-> ally" not in lower or "protect" not in lower,
                "detail": action_text,
            }
        )
        checks.append(
            {
                "check": "protect_self_or_default",
                "pass": "-> default" in lower,
                "detail": action_text,
            }
        )
    if "trickroom" in move_hint.lower() or "trick room" in lower:
        checks.append(
            {
                "check": "trickroom_not_ally",
                "pass": "trickroom" not in lower or "-> ally" not in lower,
                "detail": action_text,
            }
        )
        checks.append(
            {
                "check": "trickroom_self_or_default",
                "pass": "trickroom" not in lower or "-> default" in lower,
                "detail": action_text,
            }
        )
    if "knockoff" in move_hint.lower() or "knock off" in lower:
        checks.append(
            {
                "check": "knockoff_targets_foe",
                "pass": "knockoff" not in lower.replace(" ", "")
                or "opp slot" in lower,
                "detail": action_text,
            }
        )
    return checks


def _print_sample_header(title: str, sample) -> None:
    view = sample.view_state
    if view is None:
        return
    gt = format_log_action_pair(view, sample.side, sample.action_slot0, sample.action_slot1)
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"replay={sample.replay_id} turn={sample.turn} side={sample.side}")
    print(f"Ground truth: {gt}")
    print(f"slot0={sample.action_slot0} slot1={sample.action_slot1}")


def main() -> None:
    decision_path = log_action_space_decision()
    print("=" * 72)
    print("ACTION OUTPUT SPACE DECISION")
    print("=" * 72)
    print(decision_path.read_text(encoding="utf-8"))
    print()

    log_files = [RAW_DIR / name for name in TARGET_AUDIT_LOGS if (RAW_DIR / name).is_file()]
    if not log_files:
        log_files = sorted(RAW_DIR.glob("*.log"))[:5]

    print("=" * 72)
    print(f"PARSER TARGET AUDIT — {len(log_files)} replays")
    print("=" * 72)
    for p in log_files:
        print(f"  - {p.name}")

    all_checks: list[dict] = []
    highlighted: list = []

    for path in log_files:
        samples = parse_log_file(path, keep_view_state=True)
        print(f"{path.name}: {len(samples)} samples")

        for sample in samples:
            if sample.view_state is None:
                continue
            gt = format_log_action_pair(
                sample.view_state,
                sample.side,
                sample.action_slot0,
                sample.action_slot1,
            )
            blob = gt.lower()
            move_hint = path.name
            if any(
                k in blob
                for k in ("protect", "trickroom", "trick room", "knockoff", "knock off")
            ):
                highlighted.append(sample)
                all_checks.extend(_target_checks(gt, move_hint))

    if not highlighted:
        raise SystemExit("No target-audit samples found in selected logs.")

    for sample in highlighted[:12]:
        _print_sample_header("TARGET AUDIT SAMPLE", sample)

    print()
    print("=" * 72)
    print("TARGET AUDIT CHECKS")
    print("=" * 72)
    failures = 0
    for c in all_checks:
        status = "PASS" if c["pass"] else "FAIL"
        if not c["pass"]:
            failures += 1
        print(f"  [{status}] {c['check']}: {c['detail']}")

    first = highlighted[0]
    audit = sample_audit_dict(first)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "target_audit.json"
    out_path.write_text(
        json.dumps(
            {
                "audit": audit,
                "checks": all_checks,
                "n_highlighted": len(highlighted),
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Saved: {out_path}")

    if failures:
        raise SystemExit(f"Target audit failed ({failures} check(s)).")

    print("All target audit checks passed.")

    print()
    print("=" * 72)
    print("TEMPORAL TOKEN AUDIT")
    print("=" * 72)
    temporal_path = RAW_DIR / TEMPORAL_AUDIT_LOG
    if not temporal_path.is_file():
        raise SystemExit(f"Missing temporal audit log: {temporal_path}")

    temporal_checks: list[dict] = []
    for sample in parse_log_file(temporal_path, keep_view_state=True):
        if sample.view_state is None:
            continue
        view = sample.view_state
        tokens = encode_log_state(view, sample.side)
        assert tokens.shape == (13, N_FIELDS), tokens.shape
        hr = human_readable_state(view, sample.side)
        for label, mon_hr, tok_idx in (
            ("our_a", hr["our_actives"][0], 1),
            ("our_b", hr["our_actives"][1], 2),
            ("opp_a", hr["opp_actives"][0], 3),
            ("opp_b", hr["opp_actives"][1], 4),
        ):
            if not mon_hr.get("present"):
                continue
            row = tokens[tok_idx]
            ta = int(row[TEMPORAL_TURNS_ACTIVE])
            pc = int(row[TEMPORAL_PROTECT_COUNTER])
            lm = int(row[TEMPORAL_LAST_MOVE_ID])
            temporal_checks.append(
                {
                    "replay": sample.replay_id,
                    "turn": sample.turn,
                    "side": sample.side,
                    "slot": label,
                    "turns_active": ta,
                    "protect_counter": pc,
                    "last_move_id": lm,
                    "hr_turns_active": mon_hr.get("turns_active"),
                    "hr_protect_counter": mon_hr.get("protect_counter"),
                    "hr_last_move_id": mon_hr.get("last_move_id"),
                }
            )

    # Turn 6 p1: Farigiraf on field since end T5 (turns_active>=1); Incineroar just switched in T6 actions but pre-turn still Meganium on b.
    t6_p1 = next(
        (
            c
            for c in temporal_checks
            if c["turn"] == 6 and c["side"] == "p1" and c["slot"] == "our_a"
        ),
        None,
    )
    if t6_p1 is None or t6_p1["turns_active"] < 1:
        raise SystemExit(f"Temporal audit failed: expected Farigiraf turns_active>=1 at T6, got {t6_p1}")

    print(f"  tensor field count: {N_FIELDS} (temporal at indices 12-14)")
    print(f"  temporal rows sampled: {len(temporal_checks)}")
    for row in temporal_checks[:8]:
        print(
            f"  T{row['turn']} {row['side']} {row['slot']}: "
            f"turns_active={row['turns_active']} "
            f"protect_counter={row['protect_counter']} "
            f"last_move_id={row['last_move_id']}"
        )
    print("Temporal token audit passed.")

    print()
    print("=" * 72)
    print("TRAJECTORY STACK AUDIT")
    print("=" * 72)
    stacked_samples = parse_log_file(temporal_path, skip_rating=True)
    if not stacked_samples:
        raise SystemExit("Trajectory audit failed: no samples")
    bad = [s for s in stacked_samples if s.tokens.shape != (STACKED_N_TOKENS, N_FIELDS)]
    if bad:
        raise SystemExit(
            f"Trajectory audit failed: {len(bad)} samples not shape ({STACKED_N_TOKENS}, {N_FIELDS})"
        )
    fs_count = sum(1 for s in stacked_samples if s.sample_kind == "force_switch")
    print(f"  samples: {len(stacked_samples)} | force_switch: {fs_count}")
    print(f"  stacked token shape: ({STACKED_N_TOKENS}, {N_FIELDS})")
    print("Trajectory stack audit passed.")


if __name__ == "__main__":
    main()
