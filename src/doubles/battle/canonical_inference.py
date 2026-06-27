"""Canonical-index inference via semantic move-id matching (no poke-env index remap)."""

from __future__ import annotations

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player

from src.doubles.battle.move_order import (
    apply_joint_slot1_mask_torch,
    canonical_force_switch_mask,
    canonical_move_list,
    decode_move_action_index,
    effective_force_switch_flags,
    pokeenv_action_mask_to_canonical,
)
from src.doubles.data.action_space_spec import ACTION_PASS


def decode_canonical_tuple(canonical_idx: int) -> dict:
    """Decode a canonical 0-106 index into semantic components."""
    if canonical_idx == ACTION_PASS:
        return {"kind": "pass", "index": 0}
    if 1 <= canonical_idx <= 6:
        return {"kind": "switch", "index": canonical_idx, "bench_slot": canonical_idx}

    move_slot, target_offset, mega, tera = decode_move_action_index(canonical_idx)
    return {
        "kind": "move",
        "index": canonical_idx,
        "move_slot": move_slot,
        "target_offset": target_offset,
        "mega": mega,
        "tera": tera,
    }


def canonical_moves_for_slot(battle: DoubleBattle, pos: int) -> list[str]:
    mon = battle.active_pokemon[pos]
    if mon is None:
        return []
    return canonical_move_list([m.id for m in mon.moves.values()])


def find_move_object_by_id(
    battle: DoubleBattle,
    pos: int,
    move_id: str,
):
    """Find a Move object by Showdown id (matches poke-env selection rules)."""
    move_id = to_id_str(move_id)
    active_mon = battle.active_pokemon[pos]
    if active_mon is None:
        return None

    avail = battle.available_moves[pos]
    avail_ids = [to_id_str(m.id) for m in avail]
    known_moves = list(active_mon.moves.values())[:4]
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


def _order_in_valid_set(order: SingleBattleOrder, battle: DoubleBattle, pos: int) -> bool:
    return str(order) in [str(o) for o in battle.valid_orders[pos]]


def _pick_valid_switch(
    battle: DoubleBattle, pos: int, canonical_idx: int
) -> SingleBattleOrder | None:
    """Map canonical bench slot to a server-legal switch from available_switches."""
    team_list = list(battle.team.values())
    bench = canonical_idx - 1
    if bench < 0 or bench >= len(team_list):
        return None
    target = team_list[bench]
    target_id = to_id_str(target.base_species)
    for mon in battle.available_switches[pos]:
        if to_id_str(mon.base_species) == target_id:
            order = Player.create_order(mon)
            if _order_in_valid_set(order, battle, pos):
                return order
    for mon in battle.available_switches[pos]:
        order = Player.create_order(mon)
        if _order_in_valid_set(order, battle, pos):
            return order
    return None


def canonical_index_to_single_order(
    battle: DoubleBattle,
    pos: int,
    canonical_idx: int,
) -> SingleBattleOrder:
    """
    Translate a masked canonical argmax into a poke-env SingleBattleOrder
    by semantic move-id lookup (never indexes into paste-order slots).
    """
    if canonical_idx == ACTION_PASS:
        order = PassBattleOrder()
        if _order_in_valid_set(order, battle, pos):
            return order
        for candidate in battle.valid_orders[pos]:
            if "pass" in str(candidate).lower():
                return candidate
        return PassBattleOrder()

    if 1 <= canonical_idx <= 6:
        switch_order = _pick_valid_switch(battle, pos, canonical_idx)
        if switch_order is not None:
            return switch_order
        return PassBattleOrder()

    decoded = decode_canonical_tuple(canonical_idx)
    move_slot = int(decoded["move_slot"])
    target_offset = int(decoded["target_offset"])
    mega = bool(decoded["mega"])
    tera = bool(decoded["tera"])

    moves = canonical_moves_for_slot(battle, pos)
    if move_slot < 1 or move_slot > len(moves):
        return PassBattleOrder()

    move_id = moves[move_slot - 1]
    move_obj = find_move_object_by_id(battle, pos, move_id)
    if move_obj is None:
        return PassBattleOrder()

    order = Player.create_order(
        move_obj,
        move_target=target_offset,
        mega=mega,
        terastallize=tera,
    )
    if _order_in_valid_set(order, battle, pos):
        return order

    # Semantic move id correct but target/gimmick string mismatch — scan valid_orders.
    move_id_norm = to_id_str(move_id)
    for candidate in battle.valid_orders[pos]:
        cand_str = str(candidate).lower()
        if move_id_norm in cand_str or move_id_norm.replace("-", "") in cand_str.replace("-", ""):
            return candidate
    return PassBattleOrder()


def pick_masked_canonical_indices(
    battle: DoubleBattle,
    logits0: torch.Tensor,
    logits1: torch.Tensor,
) -> tuple[int, int]:
    """Masked argmax in canonical action space (no poke-env remap)."""
    force = any(effective_force_switch_flags(battle))
    masks: list[torch.Tensor] = []
    for pos in (0, 1):
        if force:
            ca = canonical_force_switch_mask(battle, pos)
        else:
            pe = DoublesEnv.get_action_mask_individual(battle, pos)
            ca = np.array(pokeenv_action_mask_to_canonical(battle, pos, pe), dtype=bool)
        masks.append(torch.as_tensor(ca, dtype=torch.bool, device=logits0.device))

    def _pick(row: torch.Tensor, mask: torch.Tensor) -> int:
        masked = row.clone()
        masked[~mask] = -float("inf")
        if not bool(mask.any()):
            return ACTION_PASS
        return int(masked.argmax().item())

    a0 = _pick(logits0, masks[0])
    mask1 = apply_joint_slot1_mask_torch(
        masks[1], a0_canonical=a0, force_switch=force
    )
    a1 = _pick(logits1, mask1)
    return a0, a1


def canonical_indices_to_battle_order(
    battle: DoubleBattle,
    ca0: int,
    ca1: int,
) -> DoubleBattleOrder | DefaultBattleOrder:
    """Build a joint DoubleBattleOrder from two canonical slot indices."""
    order0 = canonical_index_to_single_order(battle, 0, ca0)
    order1 = canonical_index_to_single_order(battle, 1, ca1)
    joined = DoubleBattleOrder.join_orders([order0], [order1])
    if joined:
        return joined[0]
    return DefaultBattleOrder()


def is_joint_order_valid(battle: DoubleBattle, order: BattleOrder) -> bool:
    """True when a doubles joint order is accepted by poke-env valid_orders."""
    if isinstance(order, DefaultBattleOrder):
        return False
    if not isinstance(order, DoubleBattleOrder):
        return False
    if not _order_in_valid_set(order.first_order, battle, 0):
        return False
    if not _order_in_valid_set(order.second_order, battle, 1):
        return False
    joints = DoubleBattleOrder.join_orders(
        [order.first_order], [order.second_order]
    )
    if not joints:
        return False
    return str(joints[0]) == str(order)


def submission_debug(
    battle: DoubleBattle,
    pos: int,
    canonical_idx: int,
) -> dict:
    """Trace helper: canonical index -> semantic tuple -> submitted order."""
    decoded = decode_canonical_tuple(canonical_idx)
    moves = canonical_moves_for_slot(battle, pos)
    move_id = None
    if decoded.get("kind") == "move" and decoded.get("move_slot"):
        slot = int(decoded["move_slot"])
        if 0 < slot <= len(moves):
            move_id = moves[slot - 1]
    order = canonical_index_to_single_order(battle, pos, canonical_idx)
    return {
        "canonical_index": canonical_idx,
        "decoded": decoded,
        "canonical_moves": moves,
        "move_id": move_id,
        "order": str(order),
    }
