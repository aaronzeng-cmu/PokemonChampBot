"""Probabilistic belief state over hidden opponent information."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from poke_env.data import to_id_str

from src.core.planning.species_normalize import clean_species_name, opponent_belief_key

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle

    from src.doubles.planning.meta_database import MetaDatabase

_CHOICE_ITEMS = frozenset(
    {
        "choiceband",
        "choicespecs",
        "choicescarf",
    }
)


@dataclass
class Distribution:
    options: dict[str, float]

    def normalized(self) -> Distribution:
        total = sum(self.options.values())
        if total <= 0:
            return Distribution({"": 1.0})
        return Distribution({k: v / total for k, v in self.options.items()})

    def sample(self, rng: random.Random) -> str:
        dist = self.normalized().options
        keys = list(dist.keys())
        weights = [dist[k] for k in keys]
        return rng.choices(keys, weights=weights, k=1)[0]

    def collapse(self, value: str, certainty: float = 1.0) -> None:
        if certainty >= 1.0:
            self.options = {value: 1.0}
        else:
            remaining = max(0.0, 1.0 - certainty)
            others = {k: v for k, v in self.options.items() if k != value}
            other_total = sum(others.values()) or 1.0
            self.options = {value: certainty}
            for k, v in others.items():
                self.options[k] = remaining * (v / other_total)


@dataclass
class BeliefPokemon:
    species: str
    slot: int | None = None
    moves: list[Distribution] = field(default_factory=list)
    item: Distribution = field(default_factory=lambda: Distribution({"": 1.0}))
    ability: Distribution = field(default_factory=lambda: Distribution({"": 1.0}))
    ev_spread: Distribution = field(default_factory=lambda: Distribution({"": 1.0}))
    tera_type: Distribution = field(default_factory=lambda: Distribution({"Normal": 1.0}))
    speed_floor: int | None = None
    revealed_moves: set[str] = field(default_factory=set)
    locked: bool = False
    mega_confirmed: bool = False
    mega_form: str | None = None
    pikalytics_key: str | None = None
    # Reg M-A bring-4: seen 6 at preview, confirm each mon when it appears in battle.
    brought_prob: float = 0.0
    confirmed_brought: bool = False
    confirmed_absent: bool = False
    preview_only: bool = False


@dataclass
class ConcreteSet:
    species: str
    moves: list[str]
    item: str
    ability: str
    ev_spread: str
    tera_type: str
    mega: bool = False


class BeliefState:
    """Belief over opponent roster (6 preview → 4 brought) and hidden sets."""

    PREVIEW_SIZE = 6
    BRING_COUNT = 4

    def __init__(self) -> None:
        self._mons: dict[str, BeliefPokemon] = {}

    @property
    def pokemon(self) -> list[BeliefPokemon]:
        return list(self._mons.values())

    @property
    def confirmed_brought_count(self) -> int:
        return sum(1 for m in self._mons.values() if m.confirmed_brought)

    def _initial_brought_probs(self, species_list: list[str], meta_db: MetaDatabase) -> dict[str, float]:
        """Marginal P(in bring-4) per preview mon; weights sum to BRING_COUNT."""
        weights: dict[str, float] = {}
        for species in species_list:
            prior = meta_db.get_species_prior(species)
            weights[species] = prior.usage_pct if prior.usage_pct and prior.usage_pct > 0 else 1.0
        total = sum(weights.values()) or float(len(species_list))
        return {sp: self.BRING_COUNT * weights[sp] / total for sp in species_list}

    def _apply_set_priors(
        self,
        belief: BeliefPokemon,
        species: str,
        meta_db: MetaDatabase,
        *,
        item: str = "",
        battle_mon=None,
    ) -> None:
        prior = meta_db.get_species_prior(species, item=item)
        move_dist = Distribution(meta_db.top_moves_raw(species, item=item))
        belief.moves = [copy.deepcopy(move_dist) for _ in range(4)]
        belief.item = Distribution(dict(prior.items))
        belief.ability = Distribution(dict(prior.abilities))
        belief.ev_spread = Distribution(dict(prior.ev_spreads)).normalized()
        belief.tera_type = Distribution(dict(prior.tera_types)).normalized()
        belief.pikalytics_key = prior.pikalytics_key
        belief.preview_only = False
        if battle_mon is not None:
            if battle_mon.ability:
                belief.ability.collapse(battle_mon.ability)
            if battle_mon.item:
                belief.item.collapse(str(battle_mon.item))
            for move in battle_mon.moves.values():
                if move and not getattr(move, "disabled", False):
                    self._register_revealed_move(belief, move.id)

    def _renormalize_brought_probs(self) -> None:
        remaining_slots = self.BRING_COUNT - self.confirmed_brought_count
        uncertain = [
            m for m in self._mons.values() if not m.confirmed_brought and not m.confirmed_absent
        ]
        if not uncertain:
            return
        if remaining_slots <= 0:
            for mon in uncertain:
                mon.confirmed_absent = True
                mon.brought_prob = 0.0
            return
        share = remaining_slots / len(uncertain)
        for mon in uncertain:
            mon.brought_prob = share

    def _finalize_absent_if_complete(self) -> None:
        if self.confirmed_brought_count < self.BRING_COUNT:
            return
        for mon in self._mons.values():
            if not mon.confirmed_brought:
                mon.confirmed_absent = True
                mon.brought_prob = 0.0

    def initialize_from_preview(self, battle: DoubleBattle, meta_db: MetaDatabase) -> None:
        self._mons.clear()
        preview_mons = list(battle.teampreview_opponent_team)
        if not preview_mons:
            preview_mons = list(battle.opponent_team.values())

        species_keys: list[str] = []
        for mon in preview_mons:
            key = opponent_belief_key(mon)
            if key and key not in species_keys:
                species_keys.append(key)

        brought_probs = self._initial_brought_probs(
            [clean_species_name(k) for k in species_keys],
            meta_db,
        )

        for slot, mon in enumerate(preview_mons, start=1):
            key = opponent_belief_key(mon)
            norm = clean_species_name(key)
            if not key or key in self._mons:
                continue

            battle_mon = self._find_opponent_battle_mon(battle, key)
            revealed = bool(battle_mon and getattr(battle_mon, "revealed", False))

            belief = BeliefPokemon(
                species=norm,
                slot=slot,
                brought_prob=brought_probs.get(
                    norm,
                    self.BRING_COUNT / max(len(species_keys), 1),
                ),
                preview_only=not revealed,
                confirmed_brought=revealed,
            )
            if revealed:
                self._apply_set_priors(
                    belief,
                    norm,
                    meta_db,
                    item=str(getattr(battle_mon, "item", "") or ""),
                    battle_mon=battle_mon,
                )
                belief.brought_prob = 1.0
            self._mons[key] = belief

        self._renormalize_brought_probs()
        self._finalize_absent_if_complete()

    @staticmethod
    def _find_opponent_battle_mon(battle: DoubleBattle, species: str):
        target = to_id_str(species)
        for mon in battle._opponent_team.values():
            if to_id_str(opponent_belief_key(mon)) == target:
                return mon
        for mon in battle.opponent_active_pokemon:
            if mon and to_id_str(opponent_belief_key(mon)) == target:
                return mon
        return None

    def confirm_brought(
        self,
        species: str,
        meta_db: MetaDatabase,
        *,
        battle_mon=None,
    ) -> None:
        """Collapse roster belief when a preview mon is confirmed in battle."""
        mon = self._get_mon(species)
        if mon is None or mon.confirmed_brought or mon.confirmed_absent:
            return
        mon.confirmed_brought = True
        mon.brought_prob = 1.0
        self._apply_set_priors(
            mon,
            mon.species,
            meta_db,
            item=str(getattr(battle_mon, "item", "") or "") if battle_mon else "",
            battle_mon=battle_mon,
        )
        self._renormalize_brought_probs()
        self._finalize_absent_if_complete()

    def _resolve_key(self, species: str) -> str | None:
        if species in self._mons:
            return species
        target = to_id_str(species)
        for key in self._mons:
            if to_id_str(key) == target:
                return key
        return None

    def get(self, species_or_slot: str | int) -> BeliefPokemon | None:
        if isinstance(species_or_slot, int):
            for mon in self._mons.values():
                if mon.slot == species_or_slot:
                    return mon
            return None
        key = self._resolve_key(species_or_slot)
        return self._mons.get(key) if key else None

    def is_mega_confirmed(self, species: str) -> bool:
        mon = self.get(species)
        return bool(mon and mon.mega_confirmed)

    def collapse_mega_form(
        self,
        species: str,
        meta_db: MetaDatabase,
        *,
        item: str = "",
        mega_form: str = "",
        ability: str = "",
    ) -> None:
        """Collapse blended priors to a confirmed mega form after observation."""
        mon = self._get_mon(species)
        if mon is None or mon.locked or mon.mega_confirmed:
            return

        prior = meta_db.get_species_prior(mon.species, item=item)
        move_dist = Distribution(meta_db.top_moves_raw(mon.species, item=item))
        new_slots: list[Distribution] = []
        for _ in range(4):
            slot = copy.deepcopy(move_dist)
            for revealed in mon.revealed_moves:
                if revealed in slot.options:
                    slot.collapse(revealed)
                else:
                    revealed_id = to_id_str(revealed)
                    for option in slot.options:
                        if to_id_str(option) == revealed_id:
                            slot.collapse(option)
                            break
            new_slots.append(slot)

        mon.moves = new_slots
        mon.item = Distribution(dict(prior.items))
        if item:
            mon.item.collapse(item)
        mon.ability = Distribution(dict(prior.abilities))
        if ability:
            mon.ability.collapse(ability)
        elif prior.abilities:
            top_ability = max(prior.abilities.items(), key=lambda x: x[1])[0]
            if top_ability:
                mon.ability.collapse(top_ability)
        mon.mega_confirmed = True
        mon.mega_form = mega_form or prior.pikalytics_key
        mon.pikalytics_key = prior.pikalytics_key
        mon.preview_only = False

    def _get_mon(self, species: str) -> BeliefPokemon | None:
        key = self._resolve_key(species)
        return self._mons.get(key) if key else None

    def collapse_move(self, species: str, move_name: str, slot_idx: int | None = None) -> None:
        mon = self._get_mon(species)
        if mon is None or mon.locked:
            return
        move_name = move_name.replace("_", " ").title() if move_name.islower() else move_name
        mon.revealed_moves.add(move_name)
        if slot_idx is not None and 0 <= slot_idx < len(mon.moves):
            mon.moves[slot_idx].collapse(move_name)
        else:
            for slot in mon.moves:
                if move_name in slot.options or not mon.revealed_moves:
                    slot.collapse(move_name)
                    break
        item_id = to_id_str(mon.item.sample(random.Random(0)))
        if item_id in _CHOICE_ITEMS:
            mon.item.collapse(mon.item.sample(random.Random(0)))

    def update_speed_floor(self, species: str, min_speed: int) -> None:
        mon = self._get_mon(species)
        if mon is None or mon.locked:
            return
        if mon.speed_floor is None or min_speed > mon.speed_floor:
            mon.speed_floor = min_speed
        scarf_key = next((k for k in mon.item.options if "scarf" in k.lower()), None)
        if scarf_key and min_speed > 100:
            mon.item.collapse(scarf_key, certainty=0.85)

    def update_item(self, species: str, item: str, certainty: float = 1.0) -> None:
        mon = self._get_mon(species)
        if mon is None or mon.locked:
            return
        mon.item.collapse(item, certainty=certainty)

    def update_ability(self, species: str, ability: str, certainty: float = 1.0) -> None:
        mon = self._get_mon(species)
        if mon is None or mon.locked:
            return
        mon.ability.collapse(ability, certainty=certainty)

    def mark_fainted(self, species: str) -> None:
        mon = self._get_mon(species)
        if mon is not None:
            mon.locked = True

    def reweight_spreads_for_defense(
        self, species: str, stat: str, min_value: int, max_value: int
    ) -> None:
        mon = self._get_mon(species)
        if mon is None or mon.locked:
            return
        stat_idx = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}.get(stat)
        if stat_idx is None:
            return
        weights: dict[str, float] = {}
        for key, prob in mon.ev_spread.options.items():
            parts = key.split("|")
            if len(parts) != 2:
                weights[key] = prob * 0.5
                continue
            evs = parts[1].split("/")
            if len(evs) != 6:
                weights[key] = prob * 0.5
                continue
            try:
                val = int(evs[stat_idx])
            except ValueError:
                weights[key] = prob * 0.5
                continue
            if min_value <= val <= max_value:
                weights[key] = prob
            else:
                weights[key] = prob * 0.1
        mon.ev_spread = Distribution(weights).normalized()

    def sample_determinization(self, rng: random.Random) -> dict[str, ConcreteSet]:
        concrete: dict[str, ConcreteSet] = {}
        for species, mon in self._mons.items():
            if mon.locked or mon.confirmed_absent:
                continue
            if mon.preview_only and not mon.confirmed_brought:
                if rng.random() > mon.brought_prob:
                    continue
            moves: list[str] = []
            seen: set[str] = set()
            for slot in mon.moves:
                choice = slot.sample(rng)
                while choice in seen and len(seen) < len(slot.options):
                    choice = slot.sample(rng)
                moves.append(choice)
                seen.add(choice)
            concrete[species] = ConcreteSet(
                species=species,
                moves=moves,
                item=mon.item.sample(rng),
                ability=mon.ability.sample(rng),
                ev_spread=mon.ev_spread.sample(rng),
                tera_type=mon.tera_type.sample(rng),
                mega=mon.mega_confirmed,
            )
        return concrete

    def _register_revealed_move(self, belief: BeliefPokemon, move_id: str) -> None:
        name = move_id.replace("-", " ").title()
        belief.revealed_moves.add(name)
        for i, slot in enumerate(belief.moves):
            if name in slot.options or to_id_str(name) in {to_id_str(k) for k in slot.options}:
                slot.collapse(name)
                return
        if belief.moves:
            belief.moves[len(belief.revealed_moves) % 4].collapse(name)
