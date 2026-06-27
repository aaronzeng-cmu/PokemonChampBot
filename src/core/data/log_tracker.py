"""Turn-by-turn game state tracker from Showdown protocol logs (first-person)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Literal

from poke_env.data import to_id_str

from src.core.data.perspective import MonPerspective, apply_reveal_move, move_vocab_id

_PROTECT_MOVE_IDS = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "kingsshield",
        "banefulbunker",
        "obstruct",
        "silktrap",
        "burningbulwark",
        "maxguard",
    }
)

PlayerSide = Literal["p1", "p2"]
SPECIES_FROM_DETAILS = re.compile(r"^([^,]+)")


@dataclass
class FieldState:
    weather: str = ""
    terrain: str = ""
    trick_room: bool = False
    tailwind_p1: int = 0
    tailwind_p2: int = 0
    reflect_p1: int = 0
    reflect_p2: int = 0
    light_screen_p1: int = 0
    light_screen_p2: int = 0
    aurora_veil_p1: int = 0
    aurora_veil_p2: int = 0


@dataclass
class BattleLogState:
    ratings: dict[str, int] = dc_field(default_factory=dict)
    team_roster: dict[str, list[str]] = dc_field(default_factory=dict)
    brought_species: dict[str, set[str]] = dc_field(default_factory=dict)
    mons: dict[str, MonPerspective] = dc_field(default_factory=dict)
    active: dict[str, str] = dc_field(default_factory=dict)
    field: FieldState = dc_field(default_factory=FieldState)
    turn: int = 0
    max_turn: int = 0
    finished: bool = False
    team_mega_used: dict[str, bool] = dc_field(
        default_factory=lambda: {"p1": False, "p2": False}
    )

    def clone(self) -> BattleLogState:
        import copy

        return copy.deepcopy(self)


def _hp_int(text: str) -> int:
    digits = []
    for ch in text.strip():
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    return int("".join(digits)) if digits else 0


def _parse_hp(hp_text: str) -> tuple[int, int, bool]:
    hp_text = hp_text.strip()
    if "fnt" in hp_text:
        parts = hp_text.replace(" fnt", "").split("/")
        if len(parts) == 2:
            return 0, _hp_int(parts[1]), True
        return 0, 100, True
    if "/" in hp_text:
        cur, mx = hp_text.split("/", 1)
        return _hp_int(cur), _hp_int(mx), False
    return _hp_int(hp_text) or 100, 100, False


def _species_name(details: str) -> str:
    m = SPECIES_FROM_DETAILS.match(details.strip())
    return to_id_str(m.group(1)) if m else to_id_str(details.split(",")[0])


def _slot_key(ident: str) -> str:
    return ident.split(":")[0]


class LogStateTracker:
    """Reconstruct battle state from pipe-delimited protocol lines."""

    def __init__(self) -> None:
        self.state = BattleLogState()
        self._team_order: dict[str, list[str]] = {"p1": [], "p2": []}
        self._pending_mega: dict[str, bool] = {}

    def process_line(self, line: str) -> None:
        if not line or line.startswith(">"):
            return
        parts = line.split("|")
        if len(parts) < 2:
            return
        cmd = parts[1]

        if cmd == "player" and len(parts) >= 6:
            side = parts[2]
            try:
                self.state.ratings[side] = int(parts[5])
            except ValueError:
                self.state.ratings[side] = 0
        elif cmd == "poke" and len(parts) >= 4:
            side, details = parts[2], parts[3]
            species = _species_name(details)
            self._team_order.setdefault(side[:2], []).append(species)
            self.state.team_roster[side[:2]] = list(self._team_order[side[:2]])
        elif cmd in ("switch", "drag") and len(parts) >= 5:
            self._set_active(parts[2], parts[3], parts[4])
        elif cmd == "turn" and len(parts) >= 3:
            new_turn = int(parts[2])
            if self.state.turn > 0 and new_turn > self.state.turn:
                self._end_turn()
            self.state.turn = new_turn
            self.state.max_turn = max(self.state.max_turn, self.state.turn)
        elif cmd == "win":
            self.state.finished = True
        elif cmd == "faint" and len(parts) >= 3:
            slot = _slot_key(parts[2])
            if slot in self.state.mons:
                self.state.mons[slot].fainted = True
                self.state.mons[slot].hp = 0
        elif cmd in ("-damage", "-heal") and len(parts) >= 4:
            self._update_hp(parts[2], parts[3])
            if len(parts) >= 5 and "[from] item:" in parts[-1]:
                self._reveal_item_from_suffix(parts[2], parts[-1])
        elif cmd == "move" and len(parts) >= 5:
            actor, move = parts[2], parts[3]
            slot = _slot_key(actor)
            if slot in self.state.mons:
                mon = self.state.mons[slot]
                apply_reveal_move(mon, move)
                self._record_move(mon, move)
        elif cmd == "-singleturn" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            effect = to_id_str(parts[3])
            if slot in self.state.mons and (
                "protect" in effect or effect == "detect"
            ):
                self.state.mons[slot]._turn_protect_success = True
        elif cmd == "-ability" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            ability = parts[3]
            if slot in self.state.mons:
                self.state.mons[slot].ability = to_id_str(ability)
                self.state.mons[slot].ability_revealed = True
        elif cmd == "-item" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            item = parts[3]
            if slot in self.state.mons:
                self.state.mons[slot].item = to_id_str(item)
                self.state.mons[slot].item_revealed = True
        elif cmd == "-enditem" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            item = parts[3]
            if slot in self.state.mons:
                self.state.mons[slot].item = to_id_str(item)
                self.state.mons[slot].item_revealed = True
        elif cmd == "-mega" and len(parts) >= 3:
            slot = _slot_key(parts[2])
            side = slot[:2]
            self.state.team_mega_used[side] = True
            if slot in self.state.mons:
                self.state.mons[slot].mega = True
                self.state.mons[slot].mega_capable = True
                self.state.mons[slot].seen = True
                # Species form comes from |detailschange|, not the stone name in |-mega|.
        elif cmd == "detailschange" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            species = _species_name(parts[3])
            if slot in self.state.mons:
                self.state.mons[slot].species = species
                if "mega" in species:
                    self.state.mons[slot].mega = True
        elif cmd == "replace" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            true_species = parts[3]
            mon = self.state.mons.setdefault(slot, MonPerspective(slot=slot))
            from src.doubles.data.illusion_guiderail import apply_illusion_replace

            apply_illusion_replace(mon, true_species=true_species)
        elif cmd == "-end" and len(parts) >= 4 and to_id_str(parts[3]) == "illusion":
            slot = _slot_key(parts[2])
            if slot in self.state.mons:
                self.state.mons[slot].illusion_broken = True
        elif cmd == "-terastallize" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            tera = parts[3]
            if slot in self.state.mons:
                self.state.mons[slot].terastallized = True
                self.state.mons[slot].tera_type = to_id_str(tera)
        elif cmd in ("-boost", "-unboost") and len(parts) >= 5:
            slot = _slot_key(parts[2])
            stat, amount = parts[3], int(parts[4]) * (1 if cmd == "-boost" else -1)
            if slot in self.state.mons:
                mon = self.state.mons[slot]
                mon.boosts[stat] = mon.boosts.get(stat, 0) + amount
        elif cmd == "-status" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            status = parts[3]
            if slot in self.state.mons:
                self.state.mons[slot].status = to_id_str(status)
        elif cmd == "-weather" and len(parts) >= 3:
            self.state.field.weather = to_id_str(parts[2])
        elif cmd == "-fieldstart" and len(parts) >= 3:
            field_name = to_id_str(parts[2])
            if "trickroom" in field_name:
                self.state.field.trick_room = True
            elif "terrain" in field_name:
                self.state.field.terrain = field_name
        elif cmd == "-fieldend" and len(parts) >= 3:
            field_name = to_id_str(parts[2])
            if "trickroom" in field_name:
                self.state.field.trick_room = False
            elif "terrain" in field_name:
                self.state.field.terrain = ""
        elif cmd == "-sidestart" and len(parts) >= 4:
            side, cond = parts[2], to_id_str(parts[3])
            key = side[:2]
            if "tailwind" in cond:
                if key == "p1":
                    self.state.field.tailwind_p1 = 4
                else:
                    self.state.field.tailwind_p2 = 4
            elif "reflect" in cond and "auroraveil" not in cond:
                if key == "p1":
                    self.state.field.reflect_p1 = 5
                else:
                    self.state.field.reflect_p2 = 5
            elif "lightscreen" in cond.replace(" ", ""):
                if key == "p1":
                    self.state.field.light_screen_p1 = 5
                else:
                    self.state.field.light_screen_p2 = 5
            elif "auroraveil" in cond:
                if key == "p1":
                    self.state.field.aurora_veil_p1 = 5
                    self.state.field.reflect_p1 = max(self.state.field.reflect_p1, 5)
                    self.state.field.light_screen_p1 = max(self.state.field.light_screen_p1, 5)
                else:
                    self.state.field.aurora_veil_p2 = 5
                    self.state.field.reflect_p2 = max(self.state.field.reflect_p2, 5)
                    self.state.field.light_screen_p2 = max(self.state.field.light_screen_p2, 5)
        elif cmd == "-sideend" and len(parts) >= 4:
            side, cond = parts[2], to_id_str(parts[3])
            key = side[:2]
            if "tailwind" in cond:
                if key == "p1":
                    self.state.field.tailwind_p1 = 0
                else:
                    self.state.field.tailwind_p2 = 0
            elif "reflect" in cond and "auroraveil" not in cond:
                if key == "p1":
                    self.state.field.reflect_p1 = 0
                else:
                    self.state.field.reflect_p2 = 0
            elif "lightscreen" in cond:
                if key == "p1":
                    self.state.field.light_screen_p1 = 0
                else:
                    self.state.field.light_screen_p2 = 0
            elif "auroraveil" in cond:
                if key == "p1":
                    self.state.field.aurora_veil_p1 = 0
                    self.state.field.reflect_p1 = 0
                    self.state.field.light_screen_p1 = 0
                else:
                    self.state.field.aurora_veil_p2 = 0
                    self.state.field.reflect_p2 = 0
                    self.state.field.light_screen_p2 = 0

    def _reveal_item_from_suffix(self, ident: str, suffix: str) -> None:
        if "[from] item:" not in suffix:
            return
        item = suffix.split("[from] item:")[-1].strip()
        slot = _slot_key(ident)
        if slot in self.state.mons and item:
            self.state.mons[slot].item = to_id_str(item)
            self.state.mons[slot].item_revealed = True

    def _record_move(self, mon: MonPerspective, move: str) -> None:
        move_id = to_id_str(move)
        mon._turn_move_id = move_vocab_id(move_id)
        if move_id not in _PROTECT_MOVE_IDS:
            mon.protect_counter = 0

    def _decay_side_conditions(self) -> None:
        f = self.state.field
        for attr in (
            "tailwind_p1",
            "tailwind_p2",
            "reflect_p1",
            "reflect_p2",
            "light_screen_p1",
            "light_screen_p2",
            "aurora_veil_p1",
            "aurora_veil_p2",
        ):
            val = getattr(f, attr)
            if val > 0:
                setattr(f, attr, val - 1)

    def _end_turn(self) -> None:
        """Commit per-turn scratch into temporal fields for active Pokémon."""
        self._decay_side_conditions()
        for mon in self.state.mons.values():
            if not mon.active or mon.fainted:
                continue
            mon.turns_active += 1
            if mon._turn_protect_success:
                mon.protect_counter += 1
            if mon._turn_move_id:
                mon.last_move_id = mon._turn_move_id
            mon._turn_move_id = 0
            mon._turn_protect_success = False

    def _reset_temporal_on_switch_in(self, mon: MonPerspective) -> None:
        mon.turns_active = 0
        mon.protect_counter = 0
        mon.last_move_id = 0
        mon._turn_move_id = 0
        mon._turn_protect_success = False

    def _bench_keys(self, side: str) -> list[str]:
        return [f"{side}_b{i}" for i in range(4)]

    def _find_bench_slot(self, side: str, species_key: str) -> str | None:
        from src.core.data.roster_profile import roster_species_key

        for key in self._bench_keys(side):
            mon = self.state.mons.get(key)
            if mon is None or not mon.species:
                continue
            if roster_species_key(mon.species) == species_key:
                return key
        return None

    def _allocate_bench_key(self, side: str, *, exclude: set[str] | None = None) -> str:
        skip = exclude or set()
        for key in self._bench_keys(side):
            if key in skip:
                continue
            mon = self.state.mons.get(key)
            if mon is None or not mon.species:
                return key
        return f"{side}_b0"

    def _move_active_to_bench(self, active_slot: str) -> None:
        import copy

        from src.core.data.roster_profile import roster_species_key

        mon = self.state.mons.get(active_slot)
        if mon is None or not mon.active or not mon.species:
            return
        side = active_slot[:2]
        species_key = roster_species_key(mon.species)
        if self._find_bench_slot(side, species_key) is not None:
            mon.active = False
            return
        bench_key = self._allocate_bench_key(side, exclude={active_slot})
        bench_mon = copy.deepcopy(mon)
        bench_mon.active = False
        bench_mon.slot = bench_key
        self.state.mons[bench_key] = bench_mon
        mon.active = False

    def _set_active(self, ident: str, species_label: str, hp_text: str) -> None:
        from src.core.data.roster_profile import roster_species_key

        slot = _slot_key(ident)
        side = slot[:2]
        species = _species_name(species_label)
        species_key = roster_species_key(species)
        hp, max_hp, fainted = _parse_hp(hp_text)

        bench_src = self._find_bench_slot(side, species_key)
        if slot in self.state.mons:
            outgoing = self.state.mons[slot]
            if outgoing.active and roster_species_key(outgoing.species) != species_key:
                self._move_active_to_bench(slot)

        if bench_src is not None:
            mon = self.state.mons.pop(bench_src)
            mon.slot = slot
        else:
            existing = self.state.mons.get(slot)
            if existing is None or (
                existing.active and roster_species_key(existing.species) != species_key
            ):
                mon = MonPerspective(slot=slot)
            else:
                mon = existing

        mon.species = species
        mon.hp = hp
        mon.max_hp = max_hp
        mon.fainted = fainted
        mon.active = True
        mon.seen = True
        if bench_src is None:
            mon.moves = []
            mon.item = ""
            mon.item_revealed = False
            mon.ability = ""
            mon.ability_revealed = False
            self._reset_temporal_on_switch_in(mon)
        else:
            self._reset_temporal_on_switch_in(mon)
        self.state.mons[slot] = mon

    def _update_hp(self, ident: str, hp_text: str) -> None:
        slot = _slot_key(ident)
        hp, max_hp, fainted = _parse_hp(hp_text)
        if slot not in self.state.mons:
            self.state.mons[slot] = MonPerspective(slot=slot, seen=True)
        mon = self.state.mons[slot]
        mon.hp = hp
        if max_hp > 0:
            mon.max_hp = max_hp
        mon.fainted = fainted


def project_first_person(
    state: BattleLogState,
    side: PlayerSide,
    *,
    rosters=None,
    meta_db=None,
    rng=None,
    format: str = "doubles",
    deterministic_moves: bool = False,
) -> BattleLogState:
    """Hide unrevealed opponent information (anti-leak projection)."""
    import copy

    from src.core.data.perspective import apply_first_person_view
    from src.core.data.roster_profile import (
        apply_own_roster,
        brought_species_set,
        materialize_our_bench_from_roster,
        materialize_opp_preview_bench,
    )
    from src.doubles.data.illusion_guiderail import reconcile_illusion_roster
    from src.doubles.data.mega_state import apply_can_mega_flags, apply_own_mega_capable

    view = copy.deepcopy(state)
    opp = "p2" if side == "p1" else "p1"

    if rosters is not None:
        view.brought_species[side] = brought_species_set(rosters, side)
        view.brought_species[opp] = brought_species_set(rosters, opp)
        apply_own_roster(view, side, rosters)
        materialize_our_bench_from_roster(view, side, rosters, format=format)
        materialize_opp_preview_bench(view, opp, rosters, format=format)
        apply_own_roster(view, side, rosters)
        apply_own_mega_capable(view, side, rosters)
        if format == "doubles":
            reconcile_illusion_roster(view, side, rosters)
        apply_can_mega_flags(view, side)

    if side == "p1" and meta_db is not None:
        if deterministic_moves:
            from src.doubles.data.meta_move_imputation import impute_p1_mon_moves_deterministic

            for slot, mon in view.mons.items():
                if slot.startswith("p1"):
                    impute_p1_mon_moves_deterministic(mon, meta_db)
        elif rng is not None:
            from src.doubles.data.meta_move_imputation import impute_p1_mon_moves

            for slot, mon in view.mons.items():
                if slot.startswith("p1"):
                    impute_p1_mon_moves(mon, meta_db, rng)

    from src.core.data.move_utils import canonical_move_list

    for slot, mon in view.mons.items():
        if slot.startswith(side) and mon.moves:
            mon.moves = canonical_move_list(mon.moves)

    for slot, mon in view.mons.items():
        is_ours = slot.startswith(side)
        projected = apply_first_person_view(mon, is_ours=is_ours)
        view.mons[slot] = projected
        if slot.startswith(opp) and not projected.seen:
            projected.moves = []
            projected.ability = ""
            projected.item = ""
    return view
