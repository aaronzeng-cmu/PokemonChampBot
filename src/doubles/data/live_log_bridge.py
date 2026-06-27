"""Build BC-compatible log views and masks from live poke-env battles."""

from __future__ import annotations

import random

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle

from src.doubles.battle.move_order import (
    apply_joint_slot1_mask_numpy,
    canonical_force_switch_mask,
    canonical_move_list,
    effective_force_switch_flags,
    pokeenv_action_mask_to_canonical,
)
from src.doubles.data.action_codec import is_force_switch_decision
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.doubles.data.log_action_mask import (
    forced_switch_battle_flags,
    is_dual_force_view,
    log_force_switch_slot_masks,
    log_turn_slot_mask,
    pick_masked_argmax,
    pick_masked_log_actions,
)
from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.doubles.data.mega_state import live_can_mega_for_pos
from src.core.data.perspective import stable_seed_int
from src.core.data.roster_profile import build_match_rosters, roster_species_key
from src.core.data.state_tokenizer import TRAJECTORY_DEPTH, encode_log_state, stack_trajectory
from src.doubles.planning.meta_database import MetaDatabase
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv


def _empty_mask() -> np.ndarray:
    return np.zeros(ACTION_SIZE, dtype=bool)


def pokeenv_canonical_slot_mask(battle: DoubleBattle, pos: int) -> np.ndarray:
    """Per-slot legality mask in canonical index space from the live request."""
    if any(effective_force_switch_flags(battle)):
        return canonical_force_switch_mask(battle, pos)
    pe = DoublesEnv.get_action_mask_individual(battle, pos)
    return np.array(pokeenv_action_mask_to_canonical(battle, pos, pe), dtype=bool)


def _intersect_masks(log_mask: np.ndarray, live_mask: np.ndarray) -> np.ndarray:
    both = log_mask & live_mask
    if bool(both.any()):
        return both
    return live_mask


def sync_our_mons_from_battle(
    battle: DoubleBattle,
    view: BattleLogState,
    *,
    side: str = "p1",
) -> None:
    """
    Override meta-imputed move lists for our team with the known paste moves.

    Training logs impute p1 moves from meta; live battles know the real set.
    Without this, canonical action indices can decode to the wrong move at submit.
    """
    team = getattr(battle, "team", None)
    if not team:
        return
    team_by_species: dict[str, object] = {}
    for poke in team.values():
        team_by_species[roster_species_key(to_id_str(poke.species))] = poke

    for slot, mon in view.mons.items():
        if not slot.startswith(side):
            continue
        key = roster_species_key(mon.species)
        poke = team_by_species.get(key)
        if poke is None:
            continue
        mon.moves = canonical_move_list(
            [to_id_str(m.id) for m in poke.moves.values() if m]
        )

    for pos, suffix in enumerate(("a", "b")):
        slot = f"{side}{suffix}"
        if slot in view.mons and isinstance(battle, DoubleBattle):
            view.mons[slot].can_mega = live_can_mega_for_pos(battle, pos)


def _project_view(
    state: BattleLogState,
    *,
    side: str,
    turn: int,
    sample_kind: str,
    replay_id: str,
    rosters,
    meta_db: MetaDatabase,
) -> BattleLogState:
    rng = random.Random(stable_seed_int(replay_id, turn, side, sample_kind))
    return project_first_person(
        state,
        side,
        rosters=rosters,
        meta_db=meta_db,
        rng=rng,
    )


def replay_view_at_turn_start(
    lines: list[str],
    *,
    side: str,
    turn: int,
    replay_id: str,
    meta_db: MetaDatabase,
) -> BattleLogState | None:
    """Mirror replay_parser pre_turn_state for a normal turn-start decision."""
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    view_state: BattleLogState | None = None

    for line in lines:
        if line.startswith("|turn|"):
            parts = line.split("|")
            if len(parts) >= 3:
                t = int(parts[2])
                tracker.process_line(line)
                if t == turn:
                    view_state = tracker.state.clone()
                    break
        else:
            tracker.process_line(line)

    if view_state is None:
        return None
    return _project_view(
        view_state,
        side=side,
        turn=turn,
        sample_kind="turn",
        replay_id=replay_id,
        rosters=rosters,
        meta_db=meta_db,
    )


def replay_force_switch_views(
    lines: list[str],
    *,
    side: str,
    turn: int,
    replay_id: str,
    meta_db: MetaDatabase,
) -> list[BattleLogState]:
    """All pre-switch views for forced |switch| lines on a turn (parser order)."""
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    current_turn = 0
    turn_lines: list[str] = []
    views: list[BattleLogState] = []

    for line in lines:
        if line.startswith("|turn|"):
            parts = line.split("|")
            if len(parts) >= 3:
                current_turn = int(parts[2])
            turn_lines = []
            tracker.process_line(line)
            continue

        if current_turn >= 1:
            turn_lines.append(line)
            parts = line.split("|")
            if len(parts) >= 2 and current_turn == turn:
                if is_force_switch_decision(
                    parts, turn_lines, len(turn_lines) - 1
                ):
                    pre_switch = tracker.state.clone()
                    views.append(
                        _project_view(
                            pre_switch,
                            side=side,
                            turn=turn,
                            sample_kind="force_switch",
                            replay_id=replay_id,
                            rosters=rosters,
                            meta_db=meta_db,
                        )
                    )
        tracker.process_line(line)

    return views


def replay_view_at_force_switch(
    lines: list[str],
    *,
    side: str,
    turn: int,
    replay_id: str,
    meta_db: MetaDatabase,
) -> BattleLogState | None:
    """
    Mirror replay_parser pre_switch state for the next forced |switch| on this turn.

    When poke-env forces both slots at once, this returns the state before the
    first forced switch line (both actives fainted).
    """
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    current_turn = 0
    turn_lines: list[str] = []
    view_state: BattleLogState | None = None

    for line in lines:
        if line.startswith("|turn|"):
            parts = line.split("|")
            if len(parts) >= 3:
                current_turn = int(parts[2])
            turn_lines = []
            tracker.process_line(line)
            continue

        if current_turn >= 1:
            turn_lines.append(line)
            parts = line.split("|")
            if len(parts) >= 2 and current_turn == turn:
                if is_force_switch_decision(
                    parts, turn_lines, len(turn_lines) - 1
                ):
                    pre_switch = tracker.state.clone()
                    if view_state is None:
                        view_state = pre_switch
        tracker.process_line(line)

    if view_state is None:
        return None
    return _project_view(
        view_state,
        side=side,
        turn=turn,
        sample_kind="force_switch",
        replay_id=replay_id,
        rosters=rosters,
        meta_db=meta_db,
    )


def _history_without_same_turn_push(
    history: list[np.ndarray],
    *,
    turn: int,
    last_push_turn: int | None,
) -> list[np.ndarray]:
    if last_push_turn == turn and history:
        return history[:-1]
    return list(history)


def live_force_switch_slot_masks(
    battle: DoubleBattle,
    view: BattleLogState,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Force-switch masks from the log view (same as BC / training).

    Dual simultaneous faint uses per-slot switch masks from the view; live
    picks via two parser-style forward passes, not a single combined mask.
    """
    del battle
    return log_force_switch_slot_masks(view, side)


def _intersect_with_live_mask(
    log_mask: np.ndarray,
    battle: DoubleBattle,
    pos: int,
) -> np.ndarray:
    if not isinstance(battle, DoubleBattle):
        return log_mask
    try:
        live_mask = pokeenv_canonical_slot_mask(battle, pos)
    except Exception:
        return log_mask
    return _intersect_masks(log_mask, live_mask)


def pick_masked_live_log_actions(
    logits0: torch.Tensor,
    logits1: torch.Tensor,
    *,
    battle: DoubleBattle,
    view: BattleLogState,
    side: str,
    sample_kind: str,
) -> tuple[int, int]:
    """BC-compatible masked argmax using log view state (+ poke-env intersection)."""
    fs = list(effective_force_switch_flags(battle))
    if (
        sample_kind == "force_switch"
        and len(fs) >= 2
        and fs[0]
        and fs[1]
        and is_dual_force_view(view, side)
    ):
        raise ValueError(
            "Dual force-switch requires pick_masked_dual_force_actions "
            "(two parser-style forward passes)."
        )
    if sample_kind == "force_switch":
        mask0, mask1 = live_force_switch_slot_masks(battle, view, side)
    else:
        mask0 = log_turn_slot_mask(view, side, "a")
        mask1 = log_turn_slot_mask(view, side, "b")

    mask0 = _intersect_with_live_mask(mask0, battle, 0)
    a0 = pick_masked_argmax(logits0, mask0)
    mask1_final = apply_joint_slot1_mask_numpy(
        mask1,
        a0_canonical=a0,
        force_switch=(sample_kind == "force_switch"),
    )
    mask1_final = _intersect_with_live_mask(mask1_final, battle, 1)
    a1 = pick_masked_argmax(logits1, mask1_final)
    return a0, a1


def _pick_single_slot_force_switch(
    logits0: torch.Tensor,
    logits1: torch.Tensor,
    *,
    battle: DoubleBattle | None,
    view: BattleLogState,
    side: str,
) -> tuple[int, int]:
    """
    One parser-style force_switch pick (single forced slot in view).

    Uses the same masks as BC training; optionally intersects with poke-env
    when a live battle is provided.
    """
    if battle is None:
        return pick_masked_log_actions(
            logits0, logits1, view=view, side=side, sample_kind="force_switch"
        )

    class _FakeBattle:
        def __init__(self, tag: str, turn: int, force_switch: list[bool]):
            self.battle_tag = tag
            self.turn = turn
            self.force_switch = force_switch

    fake = _FakeBattle(
        battle.battle_tag,
        int(battle.turn),
        forced_switch_battle_flags(view, side),
    )
    return pick_masked_live_log_actions(
        logits0, logits1, battle=fake, view=view, side=side, sample_kind="force_switch"
    )


def pick_masked_dual_force_actions(
    model,
    *,
    battle: DoubleBattle,
    protocol_lines: list[str],
    side: str,
    meta_db: MetaDatabase,
    history: list[np.ndarray],
    last_push_turn: int | None,
    device: str,
) -> tuple[int, int]:
    """
    When poke-env forces both slots at once, run two BC force_switch forward
    passes (one per parser sample) and take one slot from each.
    """
    turn = int(battle.turn)
    hist = _history_without_same_turn_push(
        history, turn=turn, last_push_turn=last_push_turn
    )
    views = replay_force_switch_views(
        protocol_lines,
        side=side,
        turn=turn,
        replay_id=battle.battle_tag,
        meta_db=meta_db,
    )
    if len(views) < 2:
        encoded = encode_live_as_log(
            battle, protocol_lines=protocol_lines, side=side, meta_db=meta_db
        )
        assert encoded is not None
        _, view, kind = encoded
        snap = encoded[0]
        stacked = stack_trajectory(hist, snap, depth=TRAJECTORY_DEPTH)
        x = torch.as_tensor(stacked, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0, l1 = model(x)
        if kind == "force_switch":
            return _pick_single_slot_force_switch(
                l0[0], l1[0], battle=battle, view=view, side=side
            )
        return pick_masked_live_log_actions(
            l0[0], l1[0], battle=battle, view=view, side=side, sample_kind=kind
        )

    preds: list[int] = []
    for idx, view in enumerate(views[:2]):
        sync_our_mons_from_battle(battle, view, side=side)
        snap = encode_log_state(view, side)
        stacked = stack_trajectory(hist, snap, depth=TRAJECTORY_DEPTH)
        x = torch.as_tensor(stacked, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0, l1 = model(x)
        a0, a1 = _pick_single_slot_force_switch(
            l0[0], l1[0], battle=battle, view=view, side=side
        )
        preds.append(a0 if idx == 0 else a1)

    return preds[0], preds[1]


def encode_live_as_log(
    battle: DoubleBattle,
    *,
    protocol_lines: list[str],
    side: str = "p1",
    meta_db: MetaDatabase | None = None,
) -> tuple[np.ndarray, BattleLogState, str] | None:
    """
    Encode the current live decision point with encode_log_state (BC path).

    Returns (snapshot, view, sample_kind) or None when the view cannot be built.
    """
    if meta_db is None:
        meta_db = MetaDatabase(live_fetch=False)

    turn = int(battle.turn)
    replay_id = battle.battle_tag
    force = any(effective_force_switch_flags(battle))
    sample_kind = "force_switch" if force else "turn"

    if force:
        view = replay_view_at_force_switch(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
        )
    else:
        view = replay_view_at_turn_start(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
        )

    if view is None:
        return None
    sync_our_mons_from_battle(battle, view, side=side)
    return encode_log_state(view, side), view, sample_kind


def slot_mask_for_live(
    battle: DoubleBattle,
    view: BattleLogState,
    *,
    side: str,
    sample_kind: str,
    slot_suffix: str,
    slot0_pred: int | None = None,
) -> np.ndarray:
    """Per-slot mask for live trace display (mirrors slot_mask_for_eval)."""
    pos = 0 if slot_suffix == "a" else 1
    if sample_kind == "force_switch":
        mask0, mask1 = live_force_switch_slot_masks(battle, view, side)
        if slot_suffix == "a":
            return _intersect_with_live_mask(mask0, battle, pos)
        if slot0_pred is None:
            return _intersect_with_live_mask(mask1, battle, pos)
        mask1 = apply_joint_slot1_mask_numpy(
            mask1,
            a0_canonical=slot0_pred,
            force_switch=True,
        )
        return _intersect_with_live_mask(mask1, battle, pos)
    if slot_suffix == "a":
        mask = log_turn_slot_mask(view, side, "a")
        return _intersect_with_live_mask(mask, battle, pos)
    mask1 = log_turn_slot_mask(view, side, "b")
    if slot0_pred is None:
        return _intersect_with_live_mask(mask1, battle, pos)
    mask1 = apply_joint_slot1_mask_numpy(
        mask1,
        a0_canonical=slot0_pred,
        force_switch=False,
    )
    return _intersect_with_live_mask(mask1, battle, pos)
