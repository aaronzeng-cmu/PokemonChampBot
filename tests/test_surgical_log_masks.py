"""Surgical BC training masks: force-switch, targeting, mega, safety valve."""

from __future__ import annotations

from src.doubles.battle.move_order import (
    encode_move_action_index,
    is_mega_action,
)
from src.doubles.data.action_space_spec import (
    TARGET_ALLY_SLOT_A,
    TARGET_DEFAULT,
    TARGET_OPP_SLOT_A,
)
from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.doubles.data.log_action_mask import (
    log_force_switch_slot_masks,
    log_turn_slot_mask,
    training_slot_masks,
)


def _garchomp_mon(*, can_mega: bool = True) -> MonPerspective:
    return MonPerspective(
        species="garchomp",
        hp=100,
        max_hp=100,
        active=True,
        mega_capable=can_mega,
        can_mega=can_mega,
        moves=["earthquake", "protect", "rockslide", "dragonclaw"],
    )


def test_force_switch_forced_slot_has_no_moves_or_pass():
    view = BattleLogState(
        mons={
            "p1a": MonPerspective(species="fainted", hp=0, max_hp=100, active=True, fainted=True),
            "p1b": MonPerspective(species="whimsicott", hp=100, max_hp=100, active=True),
        },
        team_roster={"p1": ["fainted", "whimsicott", "incineroar", "rillaboom", "garchomp", "amoonguss"]},
    )
    mask0, mask1 = log_force_switch_slot_masks(view, "p1")
    assert not mask0[0]
    assert not mask0[7:].any()
    assert mask0[1:7].any()
    assert mask1[0]
    assert not mask1[1:].any()


def test_ally_targeting_masked_for_offensive_move():
    mon = _garchomp_mon()
    view = BattleLogState(mons={"p1a": mon})
    mask = log_turn_slot_mask(view, "p1", "a")
    moves = ["earthquake", "protect", "rockslide", "dragonclaw"]
    ally_idx = encode_move_action_index(moves, "dragonclaw", TARGET_ALLY_SLOT_A)
    opp_idx = encode_move_action_index(moves, "dragonclaw", TARGET_OPP_SLOT_A)
    assert not mask[ally_idx]
    assert mask[opp_idx]


def test_spread_move_only_default_target():
    mon = _garchomp_mon()
    view = BattleLogState(mons={"p1a": mon})
    mask = log_turn_slot_mask(view, "p1", "a")
    moves = ["earthquake", "protect", "rockslide", "dragonclaw"]
    default_idx = encode_move_action_index(moves, "earthquake", TARGET_DEFAULT)
    wrong_idx = encode_move_action_index(moves, "earthquake", TARGET_OPP_SLOT_A)
    assert mask[default_idx]
    assert not mask[wrong_idx]


def test_mega_masked_when_cannot_mega():
    mon = _garchomp_mon(can_mega=False)
    view = BattleLogState(mons={"p1a": mon})
    mask = log_turn_slot_mask(view, "p1", "a")
    mega_indices = [i for i in range(mask.shape[0]) if is_mega_action(i)]
    assert mega_indices
    assert not any(mask[i] for i in mega_indices)


def test_safety_valve_restores_masked_ground_truth():
    mon = _garchomp_mon(can_mega=False)
    view = BattleLogState(mons={"p1a": mon, "p1b": mon})
    illegal_mega = next(i for i in range(107) if is_mega_action(i))
    mask0, mask1 = training_slot_masks(
        view,
        "p1",
        "turn",
        ground_truth_a0=illegal_mega,
        ground_truth_a1=0,
    )
    assert mask0[illegal_mega]
    assert mask1[0]
