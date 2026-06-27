"""Surgical singles BC masks: gimmicks, mega, force-switch, safety valve."""

from __future__ import annotations

import numpy as np

from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.action_space_spec import (
    DYNAMAX_BASE,
    MEGA_BASE,
    Z_MOVE_BASE,
    decode_singles_action_index,
    is_mega_action,
    is_zmove_action,
)
from src.singles.log_action_mask import (
    singles_force_switch_mask,
    singles_turn_mask,
    training_singles_mask,
)


def _meowscarada_view(*, can_mega: bool = False) -> BattleLogState:
    view = BattleLogState(turn=2)
    view.team_roster = {
        "p1": ["meowscarada", "floette", "lucario", "volcarona", "rotomwash", "garchomp"],
    }
    view.brought_species = {"p1": set(view.team_roster["p1"][:3])}
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="meowscarada",
        hp=100,
        max_hp=155,
        active=True,
        moves=["flowertrick", "thunderpunch", "tripleaxel", "uturn"],
        can_mega=can_mega,
        mega_capable=can_mega,
    )
    view.mons["p1b"] = MonPerspective(
        slot="p1b", species="floette", hp=100, max_hp=100, active=False, moves=["moonblast"]
    )
    view.mons["p1c"] = MonPerspective(
        slot="p1c", species="lucario", hp=100, max_hp=100, active=False, moves=["closecombat"]
    )
    return view


def test_zmove_and_dynamax_indices_never_legal():
    view = _meowscarada_view()
    mask = singles_turn_mask(view, "p1")
    for idx in range(Z_MOVE_BASE, SINGLES_ACTION_SIZE):
        assert not mask[idx], f"index {idx} should be illegal ({decode_singles_action_index(idx)})"


def test_mega_masked_when_cannot_mega():
    view = _meowscarada_view(can_mega=False)
    mask = singles_turn_mask(view, "p1")
    mega_indices = [i for i in range(SINGLES_ACTION_SIZE) if is_mega_action(i)]
    assert mega_indices
    assert not any(mask[i] for i in mega_indices)
    from src.singles.log_action_codec import MOVE_BASE

    assert mask[MOVE_BASE]


def test_mega_allowed_when_can_mega():
    view = BattleLogState(turn=2)
    view.team_roster = {"p1": ["floette", "lucario", "volcarona"]}
    view.brought_species = {"p1": set(view.team_roster["p1"])}
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="floetteeternal",
        hp=149,
        max_hp=149,
        active=True,
        moves=["moonblast", "calmmind", "drainingkiss", "lightofruin"],
        can_mega=True,
        mega_capable=True,
    )
    mask = singles_turn_mask(view, "p1")
    assert mask[MEGA_BASE]


def test_force_switch_masks_all_moves():
    view = BattleLogState(turn=1)
    view.team_roster = {"p1": ["meowscarada", "floette", "lucario"]}
    view.brought_species = {"p1": set(view.team_roster["p1"])}
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="meowscarada",
        hp=0,
        max_hp=155,
        active=True,
        fainted=True,
        moves=["flowertrick", "thunderpunch", "tripleaxel", "uturn"],
        can_mega=True,
    )
    view.mons["p1b"] = MonPerspective(
        slot="p1b", species="floette", hp=100, max_hp=100, active=False, moves=["moonblast"]
    )
    view.mons["p1c"] = MonPerspective(
        slot="p1c", species="lucario", hp=100, max_hp=100, active=False, moves=["closecombat"]
    )
    mask = singles_force_switch_mask(view, "p1")
    from src.singles.log_action_codec import MOVE_BASE

    assert not mask[MOVE_BASE:].any()
    assert mask[:MOVE_BASE].any()


def test_safety_valve_keeps_illegal_gt_for_loss():
    view = _meowscarada_view(can_mega=False)
    illegal_mega_gt = MEGA_BASE  # mega when cannot mega
    mask = training_singles_mask(view, "p1", "turn", ground_truth=illegal_mega_gt)
    assert mask[illegal_mega_gt]
    assert not mask[MEGA_BASE + 1]
    assert is_zmove_action(Z_MOVE_BASE)
    assert not mask[Z_MOVE_BASE]


def test_safety_valve_on_force_switch_illegal_move_gt():
    view = BattleLogState(turn=1)
    view.team_roster = {"p1": ["meowscarada", "floette", "lucario"]}
    view.brought_species = {"p1": set(view.team_roster["p1"])}
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="meowscarada",
        hp=0,
        max_hp=155,
        active=True,
        fainted=True,
        moves=["flowertrick", "thunderpunch", "tripleaxel", "uturn"],
    )
    view.mons["p1b"] = MonPerspective(
        slot="p1b", species="floette", hp=100, max_hp=100, active=False, moves=["moonblast"]
    )
    view.mons["p1c"] = MonPerspective(
        slot="p1c", species="lucario", hp=100, max_hp=100, active=False, moves=["closecombat"]
    )
    from src.singles.log_action_codec import MOVE_BASE

    mask = training_singles_mask(view, "p1", "force_switch", ground_truth=MOVE_BASE + 2)
    assert mask[MOVE_BASE + 2]
    assert not singles_force_switch_mask(view, "p1")[MOVE_BASE + 2]
