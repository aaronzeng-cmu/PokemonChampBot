"""Switch tap planning: teampreview roster bench rows + relative popup offsets."""

from __future__ import annotations

from src.cv_bridge.action_executor import ActionExecutor, switch_popup_tap


def _executor() -> ActionExecutor:
    return ActionExecutor(battle_format="doubles")


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


def test_singles_switch_still_uses_mode_bench_coords():
    ex = ActionExecutor(battle_format="singles")
    seq = ex._plan_switch("switch_bench_1")

    bench = seq.taps[1]
    assert (bench.x, bench.y) == (258, 463)
