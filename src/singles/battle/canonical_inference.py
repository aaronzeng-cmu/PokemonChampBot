"""Canonical singles inference via semantic move-id / bench-slot matching."""

from __future__ import annotations

import numpy as np
import torch
from poke_env.battle.battle import Battle
from poke_env.data import to_id_str
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from src.singles.battle.live_log_bridge import canonical_moves_for_battle_team
from src.core.data.roster_profile import roster_species_key
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.action_space_spec import decode_singles_action_index
from src.singles.bench_slots import (
    bench_switch_index_to_species_live,
    live_our_bench_mons,
)
from src.singles.battle.live_legality import legal_switch_indices
from src.singles.log_action_codec import MOVE_BASE, SWITCH_BASE


def canonical_moves_for_active(battle: Battle) -> list[str]:
    return canonical_moves_for_battle_team(battle)


def find_move_object_by_id(battle: Battle, move_id: str):
    move_id = to_id_str(move_id)
    active = battle.active_pokemon
    if active is None:
        return None

    avail = list(battle.available_moves)
    avail_ids = [to_id_str(m.id) for m in avail]
    known_moves = [m for m in active.moves.values() if m is not None]
    known_ids = [to_id_str(m.id) for m in known_moves]

    search_pool = (
        avail
        if len(avail_ids) == 1 and avail_ids[0] not in known_ids
        else known_moves
    )
    for move in search_pool:
        if to_id_str(move.id) == move_id:
            return move

    for move in avail:
        if to_id_str(move.id) == move_id:
            return move
    return None


def _order_in_valid_set(order: BattleOrder, battle: Battle) -> bool:
    return str(order) in [str(o) for o in battle.valid_orders]


def _pick_valid_switch(battle: Battle, canonical_idx: int) -> BattleOrder | None:
    if canonical_idx not in legal_switch_indices(battle):
        return None

    bench_idx = canonical_idx - SWITCH_BASE
    bench = live_our_bench_mons(battle)
    if bench_idx < 0 or bench_idx >= len(bench):
        return None

    target = bench[bench_idx]
    target_id = roster_species_key(target.species)
    for mon in battle.available_switches:
        if roster_species_key(mon.species) == target_id:
            order = Player.create_order(mon)
            if _order_in_valid_set(order, battle):
                return order
            return order
    return None


def canonical_index_to_battle_order(battle: Battle, canonical_idx: int) -> BattleOrder:
    decoded = decode_singles_action_index(canonical_idx)

    if decoded.is_switch:
        switch_order = _pick_valid_switch(battle, decoded.index)
        if switch_order is not None:
            return switch_order
        return DefaultBattleOrder()

    if decoded.move_slot is None:
        return DefaultBattleOrder()

    moves = canonical_moves_for_active(battle)
    slot = int(decoded.move_slot)
    if slot < 0 or slot >= len(moves):
        return DefaultBattleOrder()

    move_id = moves[slot]
    move_obj = find_move_object_by_id(battle, move_id)
    if move_obj is None:
        return DefaultBattleOrder()

    order = Player.create_order(
        move_obj,
        mega=decoded.mega,
        terastallize=decoded.tera,
    )
    if _order_in_valid_set(order, battle):
        return order

    move_id_norm = to_id_str(move_id)
    for candidate in battle.valid_orders:
        cand_str = str(candidate).lower()
        if move_id_norm in cand_str or move_id_norm.replace("-", "") in cand_str.replace("-", ""):
            return candidate
    return DefaultBattleOrder()


def pick_masked_canonical_index(
    logits: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
) -> int:
    if isinstance(logits, torch.Tensor):
        row = logits.detach().clone().float()
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=row.device)
    else:
        row = torch.as_tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask, dtype=bool))

    n = min(row.shape[0], mask_t.shape[0], SINGLES_ACTION_SIZE)
    row = row[:n]
    mask_t = mask_t[:n]
    if not bool(mask_t.any()):
        return SWITCH_BASE
    masked = row.clone()
    masked[~mask_t] = -float("inf")
    return int(masked.argmax().item())


def decode_canonical_submission(battle: Battle, canonical_idx: int) -> dict:
    decoded = decode_singles_action_index(canonical_idx)
    out: dict = {
        "canonical_index": canonical_idx,
        "decoded": {
            "is_switch": decoded.is_switch,
            "mega": decoded.mega,
            "zmove": decoded.zmove,
            "dynamax": decoded.dynamax,
            "tera": decoded.tera,
        },
    }
    if decoded.is_switch:
        bench_idx = decoded.index - SWITCH_BASE
        out["bench_slot"] = bench_idx + 1
        out["switch_species"] = bench_switch_index_to_species_live(battle, bench_idx)
        out["bench_species"] = [roster_species_key(p.species) for p in live_our_bench_mons(battle)]
        return out

    moves = canonical_moves_for_active(battle)
    if decoded.move_slot is not None and 0 <= decoded.move_slot < len(moves):
        out["move_id"] = moves[decoded.move_slot]
    out["canonical_moves"] = moves
    return out


def submission_debug(battle: Battle, canonical_idx: int) -> dict:
    semantic = decode_canonical_submission(battle, canonical_idx)
    order = canonical_index_to_battle_order(battle, canonical_idx)
    semantic["order"] = str(order)
    return semantic
