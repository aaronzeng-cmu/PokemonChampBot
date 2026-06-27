#!/usr/bin/env python3
"""Audit doubles mega parsing: team limit, GT vs FLAG_CAN_MEGA, roster-key gaps."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.doubles.battle.move_order import is_mega_action
from src.doubles.data.action_codec import ACTION_UNKNOWN
from src.core.data.log_tracker import BattleLogState
from src.core.data.roster_profile import build_match_rosters, roster_species_key
from src.core.data.state_tokenizer import FIELD_FLAGS, FLAG_CAN_MEGA, TOKEN_OUR_ACTIVE, encode_log_state
from src.doubles.data.mega_state import compute_can_mega
from src.doubles.data.replay_parser import parse_log_file
from src.doubles.planning.meta_database import MetaDatabase

_MEGA_LINE = re.compile(r"^\|-mega\|(p[12][ab]?)", re.MULTILINE)


def _count_megas_per_log(text: str) -> dict[str, int]:
    counts: dict[str, int] = {"p1": 0, "p2": 0}
    for m in _MEGA_LINE.finditer(text):
        slot = m.group(1)[:2]
        counts[slot] += 1
    return counts


def audit_log_mega_limits(log_dir: Path, *, max_files: int | None = None) -> dict:
    paths = sorted(log_dir.glob("*.log"))
    if max_files is not None:
        paths = paths[:max_files]
    logs_with_mega = 0
    multi_mega_side = 0
    examples: list[dict] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        counts = _count_megas_per_log(text)
        if counts["p1"] or counts["p2"]:
            logs_with_mega += 1
        for side, n in counts.items():
            if n > 1:
                multi_mega_side += 1
                if len(examples) < 5:
                    examples.append({"file": path.name, "side": side, "count": n})
    return {
        "files_scanned": len(paths),
        "logs_with_mega": logs_with_mega,
        "sides_with_gt1_mega": multi_mega_side,
        "examples": examples,
    }


def _slot_flags(view: BattleLogState, side: str) -> dict[str, bool]:
    tokens = encode_log_state(view, side)
    return {
        "a": bool(tokens[TOKEN_OUR_ACTIVE, FIELD_FLAGS] & FLAG_CAN_MEGA),
        "b": bool(tokens[TOKEN_OUR_ACTIVE + 1, FIELD_FLAGS] & FLAG_CAN_MEGA),
    }


def _diagnose_mega_gap(
    view: BattleLogState,
    side: str,
    suffix: str,
    rosters,
) -> dict:
    slot = f"{side}{suffix}"
    mon = view.mons.get(slot)
    if mon is None:
        return {"reason": "missing_mon"}
    roster = rosters.for_side(side)
    entry = roster.get(mon.species)
    key = roster_species_key(mon.species)
    alt_keys = [
        k
        for k, e in roster.entries.items()
        if e.mega_capable and k != key
    ]
    return {
        "species": mon.species,
        "roster_key": key,
        "can_mega": mon.can_mega,
        "mega_capable": mon.mega_capable,
        "team_mega_used": view.team_mega_used.get(side, False),
        "compute_can_mega": compute_can_mega(
            mon, team_mega_used=bool(view.team_mega_used.get(side, False))
        ),
        "roster_entry_found": entry is not None,
        "roster_entry_mega_capable": bool(entry and entry.mega_capable),
        "other_mega_capable_roster_keys": alt_keys[:5],
    }


def audit_training_mega_alignment(
    log_dir: Path,
    *,
    max_files: int | None = None,
    seed: int = 42,
) -> dict:
    paths = sorted(log_dir.glob("*.log"))
    mega_paths = [p for p in paths if "|-mega|" in p.read_text(encoding="utf-8", errors="replace")]
    if max_files is not None and len(mega_paths) > max_files:
        rng = random.Random(seed)
        mega_paths = rng.sample(mega_paths, max_files)

    meta_db = MetaDatabase(live_fetch=False)
    mega_gt = 0
    flag_ok = 0
    flag_missing = 0
    gap_reasons: Counter[str] = Counter()
    gap_examples: list[dict] = []

    for path in mega_paths:
        rosters = build_match_rosters(
            [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        )
        samples = parse_log_file(
            path, skip_rating=True, keep_view_state=True, meta_db=meta_db
        )
        for sample in samples:
            if sample.side != "p1":
                continue
            view = sample.view_state
            if view is None:
                continue
            flags = _slot_flags(view, sample.side)
            for suffix, action in (
                ("a", sample.action_slot0),
                ("b", sample.action_slot1),
            ):
                if action == ACTION_UNKNOWN or not is_mega_action(action):
                    continue
                mega_gt += 1
                if flags[suffix]:
                    flag_ok += 1
                else:
                    flag_missing += 1
                    diag = _diagnose_mega_gap(view, sample.side, suffix, rosters)
                    if diag.get("mega_capable") and not diag.get("team_mega_used"):
                        if diag.get("other_mega_capable_roster_keys"):
                            reason = "roster_key_mismatch"
                        elif not diag.get("roster_entry_mega_capable"):
                            reason = "roster_missing_mega_capable"
                        else:
                            reason = "can_mega_false_despite_capable"
                    elif diag.get("team_mega_used"):
                        reason = "team_mega_already_used"
                    else:
                        reason = "not_mega_capable"
                    gap_reasons[reason] += 1
                    if len(gap_examples) < 8 and reason == "roster_key_mismatch":
                        gap_examples.append(
                            {
                                "replay": sample.replay_id or path.stem,
                                "turn": sample.turn,
                                "kind": sample.sample_kind,
                                "slot": suffix,
                                "action": action,
                                **diag,
                            }
                        )

    return {
        "mega_logs_parsed": len(mega_paths),
        "mega_gt_samples_p1": mega_gt,
        "with_flag_can_mega": flag_ok,
        "missing_flag_can_mega": flag_missing,
        "pct_with_flag": (100.0 * flag_ok / mega_gt) if mega_gt else 0.0,
        "gap_reasons": dict(gap_reasons),
        "roster_mismatch_examples": gap_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("data/raw_logs"))
    parser.add_argument("--max-log-scan", type=int, default=None)
    parser.add_argument("--max-mega-logs", type=int, default=2000)
    args = parser.parse_args()

    print("=== Doubles mega audit ===\n")
    limits = audit_log_mega_limits(args.logs, max_files=args.max_log_scan)
    print("1. Max one mega per team (raw logs)")
    print(f"   Files scanned: {limits['files_scanned']:,}")
    print(f"   Logs with any |-mega|: {limits['logs_with_mega']:,}")
    print(f"   Sides with >1 mega: {limits['sides_with_gt1_mega']}")
    if limits["examples"]:
        print(f"   Examples: {limits['examples']}")

    print("\n2. Training: mega GT vs FLAG_CAN_MEGA (p1 samples, mega logs)")
    align = audit_training_mega_alignment(
        args.logs, max_files=args.max_mega_logs
    )
    print(f"   Mega logs parsed: {align['mega_logs_parsed']:,}")
    print(f"   Mega GT slot actions: {align['mega_gt_samples_p1']:,}")
    print(f"   With FLAG_CAN_MEGA: {align['with_flag_can_mega']:,} ({align['pct_with_flag']:.1f}%)")
    print(f"   Missing FLAG_CAN_MEGA: {align['missing_flag_can_mega']:,}")
    if align["gap_reasons"]:
        print(f"   Gap reasons: {align['gap_reasons']}")
    if align["roster_mismatch_examples"]:
        print("   Roster-key mismatch examples:")
        for ex in align["roster_mismatch_examples"][:3]:
            print(f"     - {ex['replay']} turn {ex['turn']} slot {ex['slot']}: "
                  f"species={ex['species']} key={ex['roster_key']} "
                  f"alt_capable={ex['other_mega_capable_roster_keys']}")

    print("\n3. Eval context notes")
    print("   Live doubles: live_can_mega_for_pos(battle, pos) + mask_mega_actions")
    print("   Offline bc_examples: find_sample_view_state -> same project_first_person path")


if __name__ == "__main__":
    main()
