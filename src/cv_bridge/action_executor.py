"""Translate model actions into emulator tap sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from poke_env.battle.move import Move
from poke_env.battle.target import Target
from poke_env.data import to_id_str

from src.doubles.battle.canonical_inference import decode_canonical_tuple
from src.doubles.data.action_space_spec import (
    TARGET_ALLY_SLOT_A,
    TARGET_ALLY_SLOT_B,
    TARGET_DEFAULT,
    TARGET_OPP_SLOT_A,
    TARGET_OPP_SLOT_B,
)
from src.singles.action_space_spec import decode_singles_action_index
from src.singles.log_action_codec import SWITCH_BASE

BattleFormat = Literal["singles", "doubles"]

_DEFAULT_COORDS = Path(__file__).with_name("ui_coordinates.json")

_SPREAD_OR_FIELD = frozenset(
    {
        Target.ALL,
        Target.ALL_ADJACENT,
        Target.ALL_ADJACENT_FOES,
        Target.ALLIES,
        Target.ALLY_SIDE,
        Target.ALLY_TEAM,
        Target.FOE_SIDE,
    }
)


def _target_key_for_move_name(move_name: str, *, commanding_slot: int = 0) -> str | None:
    """Infer overlay target from move semantics when offset alone is ambiguous."""
    try:
        move = Move(to_id_str(move_name), gen=9)
    except Exception:
        return "opp_slot_a"

    if move.deduced_target == Target.SELF:
        return "ally_slot_a" if commanding_slot == 0 else "ally_slot_b"
    if move.deduced_target in _SPREAD_OR_FIELD:
        return None
    if move.deduced_target in (Target.NORMAL, Target.ANY):
        return "opp_slot_a"
    return "opp_slot_a"


_TARGET_KEY_BY_OFFSET = {
    TARGET_ALLY_SLOT_B: "ally_slot_b",
    TARGET_ALLY_SLOT_A: "ally_slot_a",
    TARGET_OPP_SLOT_A: "opp_slot_a",
    TARGET_OPP_SLOT_B: "opp_slot_b",
}


@dataclass(frozen=True)
class Tap:
    x: int
    y: int
    label: str = ""


@dataclass
class TapSequence:
    taps: list[Tap] = field(default_factory=list)

    def extend(self, other: TapSequence) -> None:
        self.taps.extend(other.taps)


def load_ui_coordinates(path: Path | str | None = None) -> dict[str, Any]:
    coords_path = Path(path or _DEFAULT_COORDS)
    return json.loads(coords_path.read_text(encoding="utf-8"))


def _xy(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise KeyError(f"Expected [x, y] coordinate, got {value!r}")
    return int(value[0]), int(value[1])


def _xy_offset(value: Any) -> tuple[int, int]:
    return _xy(value)


def _bench_key_to_slot(bench_key: str) -> int:
    prefix = "switch_bench_"
    if not bench_key.startswith(prefix):
        raise KeyError(f"Unknown bench key {bench_key!r}")
    return int(bench_key[len(prefix) :])


def switch_popup_tap(
    bench_x: int,
    bench_y: int,
    popup_offset: dict[str, Any],
    kind: Literal["switch_in", "cancel"],
) -> tuple[int, int]:
    """Return absolute tap for a switch popup button relative to the bench row tap."""
    ox, oy = _xy_offset(popup_offset[kind])
    return bench_x + ox, bench_y + oy


class ActionExecutor:
    """Build tap plans for singles or doubles Champions battles.

    Doubles turn order is fixed: slot A (left active, ca0) is commanded first, then
    slot B (right active, ca1). The client advances between them; no slot-select tap.
    """

    def __init__(
        self,
        *,
        battle_format: BattleFormat = "doubles",
        coordinates: dict[str, Any] | None = None,
        coordinates_path: Path | str | None = None,
    ):
        self.battle_format: BattleFormat = battle_format
        self.coords = coordinates or load_ui_coordinates(coordinates_path)
        self.shared = self.coords["shared"]
        self.mode = self.coords[battle_format]

    def dismiss_tap(self) -> Tap:
        x, y = _xy(self.shared["dismiss_background"])
        return Tap(x, y, "dismiss_background")

    def _tap(self, section: dict[str, Any], key: str, label: str | None = None) -> Tap:
        x, y = _xy(section[key])
        return Tap(x, y, label or key)

    def _shared_battle(self) -> dict[str, Any]:
        return self.shared["battle"]

    def _overlay_target(self, target_key: str) -> Tap:
        overlay = self._shared_battle()["targets"]["overlay"]
        x, y = _xy(overlay[target_key])
        return Tap(x, y, f"target.{target_key}")

    def target_key_for_offset(
        self,
        target_offset: int,
        *,
        commanding_slot: int = 0,
        move_name: str | None = None,
    ) -> str | None:
        """Map poke-env target offset (and optional move name) to overlay key."""
        if target_offset in _TARGET_KEY_BY_OFFSET:
            return _TARGET_KEY_BY_OFFSET[target_offset]
        if target_offset == TARGET_DEFAULT:
            if move_name:
                return _target_key_for_move_name(move_name, commanding_slot=commanding_slot)
            return None
        return None

    def requires_target_tap(
        self,
        *,
        target_offset: int | None = None,
        move_name: str | None = None,
        commanding_slot: int = 0,
    ) -> bool:
        if target_offset is not None:
            return True
        if move_name:
            return True
        return False

    def plan_teampreview(self, roster_slots: list[int]) -> TapSequence:
        expected = int(self.mode["teampreview_pick_count"])
        if len(roster_slots) != expected:
            raise ValueError(
                f"{self.battle_format} teampreview expects {expected} slots, got {len(roster_slots)}"
            )

        preview = self.coords["teampreview"]
        seq = TapSequence()
        for slot in roster_slots:
            key = f"roster_slot_{slot}"
            if key not in preview:
                raise KeyError(f"Unknown roster slot {slot}")
            seq.taps.append(self._tap(preview, key, f"teampreview.{key}"))
        seq.taps.append(self._tap(preview, "confirm_selection", "teampreview.confirm"))
        return seq

    def plan_singles(self, action_index: int, *, move_names: list[str] | None = None) -> TapSequence:
        decoded = decode_singles_action_index(action_index)
        battle = self._shared_battle()

        if decoded.is_switch:
            bench_idx = decoded.index - SWITCH_BASE
            bench_key = f"switch_bench_{bench_idx + 1}"
            return self._plan_switch(bench_key)

        if decoded.move_slot is None:
            return TapSequence()

        move_key = f"move_{decoded.move_slot + 1}"
        seq = TapSequence()

        if decoded.mega:
            seq.taps.append(self._tap(battle["gimmicks"], "mega", "gimmick.mega"))

        seq.taps.append(self._tap(battle["main_menu"], "fight", "main.fight"))
        # The move row needs two taps: one to select (highlight) it, one to
        # confirm. A single tap only highlights and the turn never commits.
        seq.taps.append(self._tap(battle["moves"], move_key, f"move.{move_key}.select"))
        seq.taps.append(self._tap(battle["moves"], move_key, f"move.{move_key}.confirm"))

        move_name = None
        if move_names and 0 <= decoded.move_slot < len(move_names):
            move_name = move_names[decoded.move_slot]

        if move_name:
            target_key = _target_key_for_move_name(move_name, commanding_slot=0)
            if target_key:
                seq.taps.append(self._overlay_target(target_key))
        return seq

    def plan_doubles(
        self,
        ca0: int,
        ca1: int,
        *,
        move_names_slot0: list[str] | None = None,
        move_names_slot1: list[str] | None = None,
    ) -> TapSequence:
        seq = TapSequence()
        seq.extend(
            self._plan_doubles_slot(
                ca0, slot=0, move_names=move_names_slot0
            )
        )
        if ca1 != 0:
            seq.extend(
                self._plan_doubles_slot(
                    ca1, slot=1, move_names=move_names_slot1
                )
            )
        return seq

    def _plan_doubles_slot(
        self,
        canonical_index: int,
        *,
        slot: int,
        move_names: list[str] | None = None,
    ) -> TapSequence:
        decoded = decode_canonical_tuple(canonical_index)
        battle = self._shared_battle()
        seq = TapSequence()

        if decoded["kind"] == "pass":
            return seq

        if decoded["kind"] == "switch":
            bench = int(decoded["bench_slot"])
            return self._plan_switch(f"switch_bench_{bench}")

        move_slot = int(decoded["move_slot"])
        target_offset = int(decoded["target_offset"])
        move_key = f"move_{move_slot}"

        if decoded.get("mega"):
            seq.taps.append(self._tap(battle["gimmicks"], "mega", f"slot{slot}.gimmick.mega"))
        elif decoded.get("tera"):
            seq.taps.append(self._tap(battle["gimmicks"], "tera", f"slot{slot}.gimmick.tera"))

        seq.taps.append(self._tap(battle["main_menu"], "fight", f"slot{slot}.main.fight"))
        # Two taps on the move row: select (highlight), then confirm.
        seq.taps.append(self._tap(battle["moves"], move_key, f"slot{slot}.move.{move_key}.select"))
        seq.taps.append(self._tap(battle["moves"], move_key, f"slot{slot}.move.{move_key}.confirm"))

        move_name = None
        if move_names and 0 < move_slot <= len(move_names):
            move_name = move_names[move_slot - 1]

        target_key = self.target_key_for_offset(
            target_offset,
            commanding_slot=slot,
            move_name=move_name,
        )
        if target_key is not None:
            seq.taps.append(self._overlay_target(target_key))

        return seq

    def _bench_tap_xy(self, bench_key: str) -> tuple[int, int]:
        """Bench row tap on the left party column (same layout as teampreview)."""
        if self.battle_format == "doubles":
            slot = _bench_key_to_slot(bench_key)
            return _xy(self.coords["teampreview"][f"roster_slot_{slot}"])
        return _xy(self.mode[bench_key])

    def _switch_popup_tap(self, bench_x: int, bench_y: int, kind: Literal["switch_in", "cancel"]) -> Tap:
        popup_offset = self._shared_battle()["switch"]["popup_offset"]
        x, y = switch_popup_tap(bench_x, bench_y, popup_offset, kind)
        return Tap(x, y, f"switch.{kind}")

    def _plan_switch(self, bench_key: str) -> TapSequence:
        battle = self._shared_battle()
        seq = TapSequence()
        seq.taps.append(self._tap(battle["main_menu"], "pokemon_switch_menu", "switch.open"))
        bench_x, bench_y = self._bench_tap_xy(bench_key)
        # First tap selects the mon (shows its summary); the second opens the
        # Switch in / Cancel popup. Then confirm Switch in.
        seq.taps.append(Tap(bench_x, bench_y, f"switch.{bench_key}.select"))
        seq.taps.append(Tap(bench_x, bench_y, f"switch.{bench_key}.open_popup"))
        seq.taps.append(self._switch_popup_tap(bench_x, bench_y, "switch_in"))
        return seq

    def plan_force_switch(self, slot: int) -> TapSequence:
        """Replace a fainted Pokemon. The party list is already open, so we just
        tap the replacement's row and confirm via the switch popup (no menu tap).

        ``slot`` is the 1-based party-row index from ``perception.read_party_slots``.
        """
        force = self._shared_battle()["force_switch"]
        key = f"slot_{slot}"
        if key not in force:
            raise KeyError(f"Unknown force-switch slot {slot}")
        bench_x, bench_y = _xy(force[key])
        # The force-switch popup is anchored to the tapped row but at a different
        # delta than the voluntary-switch menu, so it has its own popup_offset.
        popup_offset = force.get("popup_offset") or self._shared_battle()["switch"]["popup_offset"]
        ox, oy = _xy_offset(popup_offset["switch_in"])
        seq = TapSequence()
        # First tap selects the mon (shows its summary); the second opens the
        # Switch in / Cancel popup. Then confirm Switch in.
        seq.taps.append(Tap(bench_x, bench_y, f"force_switch.{key}.select"))
        seq.taps.append(Tap(bench_x, bench_y, f"force_switch.{key}.open_popup"))
        seq.taps.append(Tap(bench_x + ox, bench_y + oy, "force_switch.switch_in"))
        return seq

    def plan_turn(
        self,
        action: int | tuple[int, int],
        *,
        move_names: list[str] | None = None,
        move_names_slot0: list[str] | None = None,
        move_names_slot1: list[str] | None = None,
    ) -> TapSequence:
        if self.battle_format == "singles":
            if not isinstance(action, int):
                raise TypeError("Singles plan_turn expects a single action index")
            return self.plan_singles(action, move_names=move_names)

        if not isinstance(action, tuple) or len(action) != 2:
            raise TypeError("Doubles plan_turn expects (ca0, ca1)")
        return self.plan_doubles(
            action[0],
            action[1],
            move_names_slot0=move_names_slot0,
            move_names_slot1=move_names_slot1,
        )
