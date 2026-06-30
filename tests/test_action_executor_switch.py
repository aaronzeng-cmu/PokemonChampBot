"""Switch tap planning: teampreview roster bench rows + relative popup offsets."""

from __future__ import annotations

from src.cv_bridge.action_executor import (
    ActionExecutor,
    _confirms_on_self,
    switch_popup_tap,
)
from src.doubles.battle.move_order import encode_move_action_index
from src.doubles.data.action_space_spec import TARGET_OPP_SLOT_A


def _executor() -> ActionExecutor:
    return ActionExecutor(battle_format="doubles")


def test_confirms_on_self_classifies_self_target_moves():
    # Target.SELF (Protect / Rage Powder / Follow Me / Ally Switch) and the locked
    # RANDOM_NORMAL moves (Outrage) confirm on our own mon.
    for mv in ("ragepowder", "followme", "protect", "spikyshield", "allyswitch", "outrage"):
        assert _confirms_on_self(mv) is True, mv
    # Foe / spread / ally-buff moves do not.
    for mv in ("earthquake", "icywind", "fakeout", "helpinghand"):
        assert _confirms_on_self(mv) is False, mv


def test_rage_powder_taps_self_even_with_foe_offset():
    ex = _executor()
    moves = ["ragepowder", "earthquake", "rockslide", "protect"]
    # Model (wrongly) encodes Rage Powder against an opponent slot.
    ca = encode_move_action_index(moves, "ragepowder", TARGET_OPP_SLOT_A)

    for slot, self_key in ((0, "ally_slot_a"), (1, "ally_slot_b")):
        seq = ex._plan_doubles_slot(ca, slot=slot, move_names=moves)
        assert seq.taps[-1].label == f"target.{self_key}"


def test_doubles_switch_uses_teampreview_roster_and_relative_popup():
    ex = _executor()
    seq = ex._plan_switch("switch_bench_3")

    assert [t.label for t in seq.taps] == [
        "switch.open",
        "switch.switch_bench_3.select",
        "switch.switch_bench_3.open_popup",
        "switch.switch_in",
    ]

    bench = seq.taps[1]
    confirm = seq.taps[3]
    assert (bench.x, bench.y) == (258, 463)
    # Select and open-popup taps land on the same row.
    assert (seq.taps[2].x, seq.taps[2].y) == (bench.x, bench.y)

    offset = ex.coords["shared"]["battle"]["switch"]["popup_offset"]
    expected_x, expected_y = switch_popup_tap(bench.x, bench.y, offset, "switch_in")
    assert (confirm.x, confirm.y) == (expected_x, expected_y)
    assert (confirm.x, confirm.y) == (406, 663)


def test_switch_popup_offset_varies_with_bench_row():
    offset = {"switch_in": [148, 200], "cancel": [154, 273]}

    row1 = switch_popup_tap(258, 211, offset, "switch_in")
    row6 = switch_popup_tap(258, 826, offset, "switch_in")

    assert row1 == (406, 411)
    assert row6 == (406, 1026)
    assert row1 != row6


def test_force_popup_tap_is_row_relative_not_clamped():
    # The Switch in / Cancel popup follows the selected row (verified rows 1-4:
    # switch_in == slot_N + [140, 91]). Row 4 must NOT be clamped to row 3's popup
    # position, which was the bug that tapped Cancel / missed the popup.
    ex = _executor()
    force = ex.coords["shared"]["battle"]["force_switch"]
    off = force["popup_offset"]["switch_in"]

    for row in (3, 4, 5, 6):
        tap = ex.switch_row_confirm_taps(row)[-1]
        sx, sy = force[f"slot_{row}"]
        assert (tap.x, tap.y) == (sx + off[0], sy + off[1]), row

    # Row 4 lands a full row below row 3 (popup is not pinned to row 3).
    row3 = ex.switch_row_confirm_taps(3)[-1]
    row4 = ex.switch_row_confirm_taps(4)[-1]
    assert row4.y > row3.y


def test_singles_switch_still_uses_mode_bench_coords():
    ex = ActionExecutor(battle_format="singles")
    seq = ex._plan_switch("switch_bench_1")

    bench = seq.taps[1]
    assert (bench.x, bench.y) == (258, 463)
