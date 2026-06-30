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

# DEFAULT-offset moves that still hit the opponent(s): the game's target overlay
# expects a foe tap to confirm these. Self / ally-side / field / locked-random
# moves (e.g. Outrage) confirm on your own mon instead.
_DEFAULT_FOE_TARGETS = frozenset(
    {
        Target.ALL_ADJACENT,
        Target.ALL_ADJACENT_FOES,
        Target.FOE_SIDE,
        Target.SCRIPTED,
    }
)

# Moves whose target overlay is confirmed on *your own* mon regardless of any
# encoded target offset: Target.SELF (Protect, Rage Powder, Follow Me, Ally Switch,
# Swords Dance, ...) and the locked random-foe moves (Outrage / Thrash / Petal
# Dance, RANDOM_NORMAL), which the game also resolves by tapping the user. The model
# can emit a stale/foe offset for these, so semantics must win over the offset.
_SELF_CONFIRM_TARGETS = frozenset(
    {
        Target.SELF,
        Target.RANDOM_NORMAL,
    }
)


def _confirms_on_self(move_name: str | None) -> bool:
    """True when the doubles target overlay for this move is tapped on our own mon."""
    if not move_name:
        return False
    try:
        return Move(to_id_str(move_name), gen=9).deduced_target in _SELF_CONFIRM_TARGETS
    except Exception:
        return False


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

    def _resolve_move_button(
        self, canonical_slot: int, physical_moves: list[str] | None
    ) -> tuple[int, str | None]:
        """Map a canonical (alphabetical) move slot to the on-screen button index.

        The model emits move slots in canonical (alphabetical) order, but the
        battle UI lists moves in the mon's actual move order. We look up the move
        *name* at the canonical slot, then find its position in the physical move
        list so we tap the correct button (and infer the correct target).
        """
        if not physical_moves:
            return canonical_slot, None
        from src.core.data.move_utils import canonical_move_list

        canonical = canonical_move_list(list(physical_moves))
        if not (1 <= canonical_slot <= len(canonical)):
            return canonical_slot, None
        move_name = canonical[canonical_slot - 1]
        phys_ids = [to_id_str(m) for m in physical_moves]
        target = to_id_str(move_name)
        button = phys_ids.index(target) + 1 if target in phys_ids else canonical_slot
        return button, move_name

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

        button, move_name = self._resolve_move_button(decoded.move_slot + 1, move_names)
        move_key = f"move_{button}"
        seq = TapSequence()

        if decoded.mega:
            seq.taps.append(self._tap(battle["gimmicks"], "mega", "gimmick.mega"))

        seq.taps.append(self._tap(battle["main_menu"], "fight", "main.fight"))
        # The move row needs two taps: one to select (highlight) it, one to
        # confirm. A single tap only highlights and the turn never commits.
        seq.taps.append(self._tap(battle["moves"], move_key, f"move.{move_key}.select"))
        seq.taps.append(self._tap(battle["moves"], move_key, f"move.{move_key}.confirm"))

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
        switch_rows: dict[int, int] | None = None,
    ) -> TapSequence:
        switch_rows = switch_rows or {}
        seq = TapSequence()
        seq.extend(
            self._plan_doubles_slot(
                ca0, slot=0, move_names=move_names_slot0, switch_row=switch_rows.get(0)
            )
        )
        if ca1 != 0:
            seq.extend(
                self._plan_doubles_slot(
                    ca1, slot=1, move_names=move_names_slot1, switch_row=switch_rows.get(1)
                )
            )
        return seq

    def _plan_doubles_slot(
        self,
        canonical_index: int,
        *,
        slot: int,
        move_names: list[str] | None = None,
        switch_row: int | None = None,
    ) -> TapSequence:
        decoded = decode_canonical_tuple(canonical_index)
        battle = self._shared_battle()
        seq = TapSequence()

        if decoded["kind"] == "pass":
            return seq

        if decoded["kind"] == "switch":
            if switch_row is not None:
                return self._plan_switch_row(switch_row)
            bench = int(decoded["bench_slot"])
            return self._plan_switch(f"switch_bench_{bench}")

        move_slot = int(decoded["move_slot"])
        target_offset = int(decoded["target_offset"])
        button, move_name = self._resolve_move_button(move_slot, move_names)
        move_key = f"move_{button}"

        if decoded.get("mega"):
            seq.taps.append(self._tap(battle["gimmicks"], "mega", f"slot{slot}.gimmick.mega"))
        elif decoded.get("tera"):
            seq.taps.append(self._tap(battle["gimmicks"], "tera", f"slot{slot}.gimmick.tera"))

        seq.taps.append(self._tap(battle["main_menu"], "fight", f"slot{slot}.main.fight"))
        # Two taps on the move row: select (highlight), then confirm.
        seq.taps.append(self._tap(battle["moves"], move_key, f"slot{slot}.move.{move_key}.select"))
        seq.taps.append(self._tap(battle["moves"], move_key, f"slot{slot}.move.{move_key}.confirm"))

        # Doubles requires a target pick after confirming the move. Self-confirming
        # moves (Protect, Rage Powder, Follow Me, Outrage, ...) always tap our own
        # mon, even if the model encoded a foe/ally offset. Otherwise: explicit
        # foe/ally offsets tap that overlay cell; DEFAULT-offset moves are routed by
        # semantics (spread/hazards -> a foe, self/field -> self).
        if _confirms_on_self(move_name):
            target_key = "ally_slot_a" if slot == 0 else "ally_slot_b"
        elif target_offset == TARGET_DEFAULT:
            target_key = self._default_target_key(move_name, slot)
        else:
            target_key = self.target_key_for_offset(
                target_offset, commanding_slot=slot, move_name=move_name
            ) or self._default_target_key(move_name, slot)
        seq.taps.append(self._overlay_target(target_key))

        return seq

    def _default_target_key(self, move_name: str | None, slot: int) -> str:
        """Overlay cell for a DEFAULT-offset move (no explicit foe encoded).

        Spread / hazard / counter moves that hit foes confirm on an opponent;
        self / ally-side / field / locked-random moves (Outrage) confirm on self.
        """
        self_key = "ally_slot_a" if slot == 0 else "ally_slot_b"
        if not move_name:
            return self_key
        try:
            target = Move(to_id_str(move_name), gen=9).deduced_target
        except Exception:
            return self_key
        return "opp_slot_a" if target in _DEFAULT_FOE_TARGETS else self_key

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

    def switch_open_tap(self) -> Tap:
        """Tap that opens the Pokemon (switch) menu from the command menu."""
        return self._tap(self._shared_battle()["main_menu"], "pokemon_switch_menu", "switch.open")

    def _force_popup_tap(self, row: int, kind: str, label: str) -> Tap:
        """Switch in / Cancel popup tap for an open party screen (force or voluntary).

        The Switch in / Cancel popup is anchored to the *selected* row and slides
        down with it (verified rows 1-4: switch_in lands at slot_N + [140, 91]). So
        the tap is purely row-relative -- offsets are deltas from this row's coords.
        """
        force = self._shared_battle()["force_switch"]
        popup_offset = force.get("popup_offset") or self._shared_battle()["switch"]["popup_offset"]
        ox, oy = _xy_offset(popup_offset[kind])
        ax, ay = _xy(force[f"slot_{row}"])
        return Tap(ax + ox, ay + oy, label)

    def switch_row_confirm_taps(self, row: int) -> list[Tap]:
        """Row taps (select, open popup, confirm Switch in) for an open party screen.

        Excludes the menu-open tap so callers can re-perceive the (reordering)
        party list before choosing the row. Reuses ``force_switch.slot_N`` row
        coordinates + popup offset (same layout as the force-switch screen).
        """
        battle = self._shared_battle()
        force = battle["force_switch"]
        key = f"slot_{row}"
        if key not in force:
            raise KeyError(f"Unknown switch row {row}")
        bench_x, bench_y = _xy(force[key])
        # First tap selects the mon (shows its summary); the second opens the
        # Switch in / Cancel popup. Then confirm Switch in (popup anchor is clamped
        # for low rows so we don't tap Cancel).
        return [
            Tap(bench_x, bench_y, f"switch.row_{row}.select"),
            Tap(bench_x, bench_y, f"switch.row_{row}.open_popup"),
            self._force_popup_tap(row, "switch_in", "switch.switch_in"),
        ]

    def _plan_switch_row(self, row: int) -> TapSequence:
        """Voluntary switch by on-screen party row (1-based).

        The in-battle Pokemon switch screen has the same layout as the
        force-switch party screen, so we reuse ``force_switch.slot_N`` row
        coordinates + its popup offset. ``row`` is resolved by the caller from
        the team's preview selection order minus the currently-active mons.
        """
        seq = TapSequence()
        seq.taps.append(self.switch_open_tap())
        seq.taps.extend(self.switch_row_confirm_taps(row))
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
        seq = TapSequence()
        # First tap selects the mon (shows its summary); the second opens the
        # Switch in / Cancel popup. Then confirm Switch in (popup anchor is clamped
        # for low rows so we don't tap Cancel).
        seq.taps.append(Tap(bench_x, bench_y, f"force_switch.{key}.select"))
        seq.taps.append(Tap(bench_x, bench_y, f"force_switch.{key}.open_popup"))
        seq.taps.append(self._force_popup_tap(slot, "switch_in", "force_switch.switch_in"))
        return seq

    def plan_turn(
        self,
        action: int | tuple[int, int],
        *,
        move_names: list[str] | None = None,
        move_names_slot0: list[str] | None = None,
        move_names_slot1: list[str] | None = None,
        switch_rows: dict[int, int] | None = None,
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
            switch_rows=switch_rows,
        )
