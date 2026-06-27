"""Track battle observations and update BeliefState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.data import to_id_str

from src.doubles.planning.damage_eval import infer_defense_stat_range
from src.core.planning.species_normalize import (
    _parse_mega_form_from_details,
    clean_species_name,
    infer_mega_stone,
    is_mega_evolved,
    mega_form_for_ability,
    opponent_belief_key,
)

if TYPE_CHECKING:
    from src.doubles.planning.belief_state import BeliefState
    from src.doubles.planning.meta_database import MetaDatabase


def _find_opponent_battle_mon(battle: DoubleBattle, species: str):
    target = to_id_str(species)
    for mon in battle._opponent_team.values():
        if to_id_str(opponent_belief_key(mon)) == target:
            return mon
    for mon in battle.opponent_active_pokemon:
        if mon and to_id_str(opponent_belief_key(mon)) == target:
            return mon
    return None


@dataclass
class MonSnapshot:
    species: str
    hp: float = 0.0
    max_hp: float = 0.0
    ability: str = ""
    item: str = ""
    moves: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    fainted: bool = False
    active: bool = False
    revealed: bool = False
    mega_confirmed: bool = False
    mega_form: str = ""

    @classmethod
    def from_pokemon(cls, mon, *, active: bool = False, revealed: bool | None = None) -> MonSnapshot:
        if mon is None:
            return cls(species="")
        moves = []
        for move in mon.moves.values():
            if move is not None:
                moves.append(move.id)
        mega = is_mega_evolved(mon)
        mega_form = ""
        if mega:
            base = clean_species_name(
                getattr(mon, "base_species", None) or mon.species or ""
            )
            details = getattr(mon, "_last_details", "") or ""
            mega_form = (
                _parse_mega_form_from_details(details)
                or mega_form_for_ability(base, str(mon.ability or ""))
                or ""
            )
        if revealed is None:
            revealed = bool(getattr(mon, "revealed", False))
        return cls(
            species=opponent_belief_key(mon),
            hp=float(mon.current_hp or 0),
            max_hp=float(mon.max_hp or 0),
            ability=str(mon.ability or ""),
            item=str(mon.item or ""),
            moves=moves,
            stats=dict(mon.stats or {}),
            fainted=bool(mon.fainted),
            active=active,
            revealed=revealed,
            mega_confirmed=mega,
            mega_form=mega_form,
        )


@dataclass
class BattleSnapshot:
    turn: int = 0
    opponent: dict[str, MonSnapshot] = field(default_factory=dict)
    our: dict[str, MonSnapshot] = field(default_factory=dict)

    @classmethod
    def from_battle(cls, battle: DoubleBattle) -> BattleSnapshot:
        snap = cls(turn=battle.turn)
        active_opp = {opponent_belief_key(p) for p in battle.opponent_active_pokemon if p}

        preview_mons = list(battle.teampreview_opponent_team)
        if preview_mons:
            for mon in preview_mons:
                key = opponent_belief_key(mon)
                battle_mon = _find_opponent_battle_mon(battle, key)
                source = battle_mon if battle_mon is not None else mon
                snap.opponent[key] = MonSnapshot.from_pokemon(
                    source,
                    active=bool(battle_mon and key in active_opp),
                    revealed=bool(battle_mon and battle_mon.revealed),
                )
        else:
            for mon in battle.opponent_team.values():
                key = opponent_belief_key(mon)
                snap.opponent[key] = MonSnapshot.from_pokemon(
                    mon, active=key in active_opp
                )

        active_our = {opponent_belief_key(p) for p in battle.active_pokemon if p}
        for mon in battle.team.values():
            key = opponent_belief_key(mon)
            snap.our[key] = MonSnapshot.from_pokemon(mon, active=key in active_our)
        return snap


class ObservationTracker:
    def process(
        self,
        battle: DoubleBattle,
        prev: BattleSnapshot | None,
        belief: BeliefState | None,
        meta_db: MetaDatabase | None = None,
    ) -> None:
        if belief is None or prev is None:
            if belief is None:
                return
            curr = BattleSnapshot.from_battle(battle)
            opp_by_key = {opponent_belief_key(mon): mon for mon in battle.opponent_team.values()}
            for mon in battle.opponent_active_pokemon:
                if mon:
                    opp_by_key[opponent_belief_key(mon)] = mon
            for species, mon_snap in curr.opponent.items():
                if mon_snap.revealed and meta_db is not None:
                    battle_mon = opp_by_key.get(species) or _find_opponent_battle_mon(
                        battle, species
                    )
                    belief.confirm_brought(species, meta_db, battle_mon=battle_mon)
            return

        curr = BattleSnapshot.from_battle(battle)
        opp_by_key = {opponent_belief_key(mon): mon for mon in battle.opponent_team.values()}
        for mon in battle.opponent_active_pokemon:
            if mon:
                opp_by_key[opponent_belief_key(mon)] = mon

        for species, mon_snap in curr.opponent.items():
            prev_snap = prev.opponent.get(species)
            if prev_snap is None:
                continue

            if not prev_snap.revealed and mon_snap.revealed and meta_db is not None:
                battle_mon = opp_by_key.get(species) or _find_opponent_battle_mon(
                    battle, species
                )
                belief.confirm_brought(species, meta_db, battle_mon=battle_mon)

            if mon_snap.fainted and not prev_snap.fainted:
                belief.mark_fainted(species)

            if mon_snap.ability and mon_snap.ability != prev_snap.ability:
                belief.update_ability(species, mon_snap.ability)
                battle_mon = opp_by_key.get(species) or _find_opponent_battle_mon(
                    battle, species
                )
                if meta_db is not None and battle_mon is not None and not mon_snap.mega_confirmed:
                    base = clean_species_name(
                        getattr(battle_mon, "base_species", None)
                        or opponent_belief_key(battle_mon)
                    )
                    inferred_form = mega_form_for_ability(base, mon_snap.ability)
                    if inferred_form:
                        stone = infer_mega_stone(battle_mon)
                        belief.collapse_mega_form(
                            species,
                            meta_db,
                            item=stone,
                            mega_form=inferred_form,
                            ability=mon_snap.ability,
                        )

            if mon_snap.item and mon_snap.item != prev_snap.item:
                belief.update_item(species, mon_snap.item)

            battle_mon = opp_by_key.get(species) or _find_opponent_battle_mon(
                battle, species
            )
            if (
                meta_db is not None
                and battle_mon is not None
                and not prev_snap.mega_confirmed
                and mon_snap.mega_confirmed
            ):
                stone = infer_mega_stone(battle_mon)
                belief.collapse_mega_form(
                    species,
                    meta_db,
                    item=stone,
                    mega_form=mon_snap.mega_form,
                    ability=mon_snap.ability or str(getattr(battle_mon, "ability", "") or ""),
                )

            new_moves = set(mon_snap.moves) - set(prev_snap.moves)
            for move_id in new_moves:
                belief.collapse_move(species, move_id)

            if mon_snap.hp < prev_snap.hp and prev_snap.hp > 0:
                damage = prev_snap.hp - mon_snap.hp
                if damage > 0 and mon_snap.active:
                    for our_species, our_snap in curr.our.items():
                        if not our_snap.active:
                            continue
                        for move_id in our_snap.moves:
                            stat, lo, hi = infer_defense_stat_range(
                                damage,
                                Move(move_id, gen=9),
                                our_snap.stats,
                                defender_hp=int(mon_snap.max_hp or 100),
                            )
                            belief.reweight_spreads_for_defense(species, stat, lo, hi)

        self._update_speed_observations(battle, prev, curr, belief)

    def _update_speed_observations(
        self,
        battle: DoubleBattle,
        prev: BattleSnapshot,
        curr: BattleSnapshot,
        belief: BeliefState,
    ) -> None:
        if curr.turn <= prev.turn:
            return
        our_actives = [p for p in battle.active_pokemon if p and not p.fainted]
        opp_actives = [
            p for p in battle.opponent_active_pokemon if p and not p.fainted
        ]
        if not our_actives or not opp_actives:
            return

        our_min_spe = min(
            (p.stats.get("spe") or p.base_stats.get("spe") or 0) for p in our_actives
        )
        for opp in opp_actives:
            opp_spe = opp.stats.get("spe") or opp.base_stats.get("spe") or 0
            if opp_spe > our_min_spe:
                belief.update_speed_floor(opponent_belief_key(opp), int(opp_spe))
