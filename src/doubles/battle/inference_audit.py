"""Debug helpers for live inference index translation (0-106 -> poke-env)."""

from __future__ import annotations

from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv

from src.doubles.battle.move_order import (
    canonical_move_list,
    decode_move_action_index,
    format_live_canonical_action,
    pokeenv_action_mask_to_canonical,
    pokeenv_available_move_list,
    pokeenv_move_list,
    remap_canonical_action_to_pokeenv,
    remap_pokeenv_action_to_canonical,
)
from src.doubles.data.action_space_spec import ACTION_SIZE, target_offset_label


def decode_action_index(
    action_idx: int,
    *,
    move_ids: list[str],
) -> dict:
    """Decode a 0-106 index using an explicit move-id slot list (1-4)."""
    if action_idx == 0:
        return {"kind": "pass", "index": 0}
    if 1 <= action_idx <= 6:
        return {"kind": "switch", "index": action_idx, "bench_slot": action_idx}

    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    move_name = (
        move_ids[move_slot - 1] if 0 < move_slot <= len(move_ids) else f"slot{move_slot}?"
    )
    return {
        "kind": "move",
        "index": action_idx,
        "move_slot": move_slot,
        "move_id": move_name,
        "target_offset": target_offset,
        "target_label": target_offset_label(target_offset),
        "mega": mega,
        "tera": tera,
    }


def _label_index(battle: DoubleBattle, pos: int, idx: int) -> str:
    return format_live_canonical_action(battle, pos, idx)


def legal_mask_debug_report(battle: DoubleBattle, pos: int) -> dict:
    """
    Full mask audit for one active slot.
    Compares paste-order, available-order, and canonical-order decodings.
    """
    mon = battle.active_pokemon[pos]
    species = to_id_str(mon.species) if mon else "?"
    paste_moves = pokeenv_move_list(battle, pos)
    available = pokeenv_available_move_list(battle, pos)
    canonical_moves = canonical_move_list([m.id for m in mon.moves.values()]) if mon else []

    pe_mask = list(DoublesEnv.get_action_mask_individual(battle, pos))
    ca_mask = pokeenv_action_mask_to_canonical(battle, pos, pe_mask)

    pe_legal: list[dict] = []
    for pe_idx, legal in enumerate(pe_mask):
        if not legal:
            continue
        ca_idx = remap_pokeenv_action_to_canonical(pe_idx, battle, pos)
        pe_legal.append(
            {
                "pokeenv_index": pe_idx,
                "canonical_index": ca_idx,
                "paste_decode": decode_action_index(pe_idx, move_ids=paste_moves),
                "canonical_decode": decode_action_index(ca_idx, move_ids=canonical_moves),
                "canonical_label": _label_index(battle, pos, ca_idx),
                "remap_changed": pe_idx != ca_idx,
            }
        )

    ca_legal: list[dict] = []
    for ca_idx, legal in enumerate(ca_mask):
        if not legal:
            continue
        pe_idx = remap_canonical_action_to_pokeenv(ca_idx, battle, pos)
        ca_legal.append(
            {
                "canonical_index": ca_idx,
                "pokeenv_index": pe_idx,
                "canonical_decode": decode_action_index(ca_idx, move_ids=canonical_moves),
                "paste_decode": decode_action_index(pe_idx, move_ids=paste_moves),
                "canonical_label": _label_index(battle, pos, ca_idx),
                "remap_changed": pe_idx != ca_idx,
            }
        )

    # Highlight protect / tailwind indices in both orderings
    protect_tailwind: dict = {}
    for move_id in ("protect", "tailwind"):
        for order_name, moves in (
            ("paste", paste_moves),
            ("canonical", canonical_moves),
        ):
            if move_id not in moves:
                continue
            slot = moves.index(move_id) + 1
            default_idx = 7 + (slot - 1) * 5 + 2
            protect_tailwind[f"{move_id}_{order_name}_default_idx"] = default_idx
            protect_tailwind[f"{move_id}_{order_name}_slot"] = slot
            protect_tailwind[f"{move_id}_{order_name}_default_legal_ca"] = (
                ca_idx_legal(ca_mask, default_idx)
            )
            protect_tailwind[f"{move_id}_{order_name}_default_legal_pe"] = (
                pe_idx_legal(pe_mask, default_idx)
            )

    return {
        "pos": pos,
        "species": species,
        "paste_moves": paste_moves,
        "available_moves": available,
        "canonical_moves": canonical_moves,
        "orders_match": paste_moves == canonical_moves,
        "available_matches_paste": available == paste_moves,
        "protect_tailwind_indices": protect_tailwind,
        "pokeenv_legal_count": sum(pe_mask),
        "canonical_legal_count": sum(ca_mask),
        "pokeenv_legal": pe_legal,
        "canonical_legal": ca_legal,
    }


def ca_idx_legal(mask: list[bool], idx: int) -> bool:
    return 0 <= idx < len(mask) and bool(mask[idx])


def pe_idx_legal(mask: list[bool], idx: int) -> bool:
    return 0 <= idx < len(mask) and bool(mask[idx])


def format_legal_mask_lines(report: dict) -> list[str]:
    """Human-readable legal mask list for trace output."""
    lines = [
        f"Slot {report['pos']} ({report['species']})",
        f"  Paste moves:      {report['paste_moves']}",
        f"  Available moves:  {report['available_moves']}",
        f"  Canonical moves:  {report['canonical_moves']}",
        f"  Paste==Canonical: {report['orders_match']}",
        f"  Available==Paste: {report['available_matches_paste']}",
    ]
    pt = report.get("protect_tailwind_indices") or {}
    if pt:
        lines.append("  Key move default indices:")
        for k, v in sorted(pt.items()):
            lines.append(f"    {k}: {v}")

    lines.append(f"  Canonical legal ({report['canonical_legal_count']}):")
    for row in report["canonical_legal"]:
        changed = " REMAP" if row["remap_changed"] else ""
        lines.append(
            f"    [{row['canonical_index']}] {row['canonical_label']}"
            f" -> pe={row['pokeenv_index']}{changed}"
        )

    lines.append(f"  Poke-env legal ({report['pokeenv_legal_count']}):")
    for row in report["pokeenv_legal"]:
        changed = " REMAP" if row["remap_changed"] else ""
        lines.append(
            f"    pe[{row['pokeenv_index']}] {row['paste_decode']}"
            f" -> ca[{row['canonical_index']}]{changed}"
        )
    return lines


def translation_audit_for_decision(
    battle: DoubleBattle,
    *,
    raw0: int,
    raw1: int,
    ca0: int,
    ca1: int,
    pe0: int,
    pe1: int,
) -> dict:
    """Trace argmax -> canonical pick -> poke-env submission for both slots."""
    slots = []
    for pos, raw, ca, pe in ((0, raw0, ca0, pe0), (1, raw1, ca1, pe1)):
        mon = battle.active_pokemon[pos]
        paste = pokeenv_move_list(battle, pos)
        canonical = canonical_move_list([m.id for m in mon.moves.values()]) if mon else []
        slots.append(
            {
                "pos": pos,
                "raw_argmax": raw,
                "canonical_picked": ca,
                "pokeenv_submitted": pe,
                "raw_decode_paste": decode_action_index(raw, move_ids=paste),
                "raw_decode_canonical": decode_action_index(raw, move_ids=canonical),
                "picked_decode_paste": decode_action_index(pe, move_ids=paste),
                "picked_decode_canonical": decode_action_index(ca, move_ids=canonical),
                "raw_label": _label_index(battle, pos, raw),
                "picked_label": _label_index(battle, pos, ca),
                "fallback": raw != ca,
                "remap_changed": ca != pe,
            }
        )
    return {"turn": int(battle.turn), "slots": slots}
