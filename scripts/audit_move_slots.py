#!/usr/bin/env python3
"""Audit canonical (alphabetical) vs poke-env move-slot ordering."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.data import to_id_str
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder import Teambuilder

from config.settings import BATTLE_FORMAT, RAW_LOGS_DIR, TEAM_PATH
from src.doubles.battle.move_order import (
    canonical_move_list,
    compare_move_orders,
    pokeenv_available_move_list,
    pokeenv_move_list,
    remap_canonical_action_to_pokeenv,
)
from src.doubles.data.action_codec import encode_log_move
from src.doubles.data.replay_parser import find_sample_view_state, parse_log_file
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.vgc_random_player import VGCRandomPlayer


def _audit_team_paste(team_path: Path) -> list[dict]:
    export = team_path.read_text(encoding="utf-8").strip()
    rows: list[dict] = []
    for mon in Teambuilder.parse_showdown_team(export):
        paste_moves = [m for m in mon.moves if m]
        canonical = canonical_move_list(paste_moves)
        pokeenv = [to_id_str(m) for m in paste_moves]
        rows.append(
            compare_move_orders(
                label=f"paste:{mon.nickname or mon.species}",
                canonical=canonical,
                pokeenv=pokeenv,
            )
        )
    return rows


def _audit_replay_samples(log_dir: Path, *, limit: int = 5) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(log_dir.glob("*.log"))[:limit]
    for path in paths:
        for sample in parse_log_file(path, skip_rating=True, keep_view_state=True):
            view = sample.view_state
            if view is None or sample.turn < 2:
                continue
            for suffix in ("a", "b"):
                slot = f"{sample.side}{suffix}"
                mon = view.mons.get(slot)
                if mon is None or not mon.active or not mon.moves:
                    continue
                canonical = canonical_move_list(mon.moves)
                rows.append(
                    compare_move_orders(
                        label=f"{path.stem} t{sample.turn} {slot}",
                        canonical=canonical,
                        pokeenv=canonical,
                    )
                )
            if len(rows) >= limit:
                return rows
    return rows


async def _audit_live_battle() -> list[dict]:
    team = TEAM_PATH.read_text(encoding="utf-8")
    agent = VGCRandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("AuditAgent", rand=True),
    )
    opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("AuditOpp", rand=True),
    )
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    rows: list[dict] = []
    for pos, suffix in enumerate(("a", "b")):
        mon = battle.active_pokemon[pos]
        if mon is None:
            continue
        canonical = canonical_move_list([m.id for m in mon.moves.values()])
        pokeenv = pokeenv_move_list(battle, pos)
        available = pokeenv_available_move_list(battle, pos)
        row = compare_move_orders(
            label=f"live:{battle.battle_tag} slot{pos} ({mon.species})",
            canonical=canonical,
            pokeenv=pokeenv,
            available=available,
        )
        row["available_moves_pokeenv"] = [
            {"index": i + 1, "id": mid} for i, mid in enumerate(available)
        ]
        for move_slot in range(1, min(5, len(canonical) + 1)):
            move_id = canonical[move_slot - 1]
            canonical_idx = 7 + (move_slot - 1) * 5 + 2
            remapped = remap_canonical_action_to_pokeenv(canonical_idx, battle, pos)
            row.setdefault("remap_samples", []).append(
                {
                    "move": move_id,
                    "canonical_action": canonical_idx,
                    "pokeenv_action": remapped,
                    "changed": canonical_idx != remapped,
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit move slot ordering")
    parser.add_argument("--team", type=Path, default=TEAM_PATH)
    parser.add_argument("--logs", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--live", action="store_true", help="Run one live battle audit")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {
        "team_paste_audit": _audit_team_paste(args.team),
        "parser_canonical_samples": _audit_replay_samples(args.logs, limit=8),
    }

    paste_mismatches = sum(1 for r in report["team_paste_audit"] if not r["orders_match"])
    report["summary"] = {
        "paste_mons_checked": len(report["team_paste_audit"]),
        "paste_mons_alphabetical_ne_pokeenv": paste_mismatches,
        "parser_uses_canonical": True,
        "inference_needs_remap": paste_mismatches > 0,
    }

    if args.live:
        try:
            report["live_battle_audit"] = asyncio.run(_audit_live_battle())
            live_mismatches = sum(
                1 for r in report["live_battle_audit"] if not r["orders_match"]
            )
            report["summary"]["live_slots_mismatch"] = live_mismatches
        except Exception as exc:
            report["live_battle_audit_error"] = str(exc)

    text_lines = ["=== Move slot audit ===", json.dumps(report["summary"], indent=2), ""]
    for section in ("team_paste_audit", "live_battle_audit"):
        rows = report.get(section) or []
        if not rows:
            continue
        text_lines.append(f"--- {section} ---")
        for row in rows:
            text_lines.append(row["label"])
            text_lines.append(f"  canonical: {row['canonical_order']}")
            text_lines.append(f"  poke-env:  {row['pokeenv_order']}")
            text_lines.append(f"  match: {row['orders_match']}")
            if row.get("available_moves"):
                text_lines.append(f"  available: {row['available_moves']}")
            for sm in row.get("remap_samples", []):
                if sm["changed"]:
                    text_lines.append(
                        f"  REMAP {sm['move']}: canonical={sm['canonical_action']} "
                        f"-> pokeenv={sm['pokeenv_action']}"
                    )
            text_lines.append("")

    text = "\n".join(text_lines)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON saved to {args.out}")


if __name__ == "__main__":
    main()
