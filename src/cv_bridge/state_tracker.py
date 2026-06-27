"""Persistent battle memory updated from CV perception."""

from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np
from poke_env.data import to_id_str

from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import (
    MonPerspective,
    apply_reveal_move,
    move_vocab_id,
)
from src.core.data.roster_profile import roster_species_key
from src.cv_bridge.action_executor import BattleFormat

PlayerSide = Literal["p1", "p2"]

_SLOT_MAP_DOUBLES: dict[str, tuple[str, str]] = {
    "player_slot_a": ("p1", "a"),
    "player_slot_b": ("p1", "b"),
    "opp_slot_a": ("p2", "a"),
    "opp_slot_b": ("p2", "b"),
}

_SLOT_MAP_SINGLES: dict[str, tuple[str, str]] = {
    "player_slot_a": ("p1", "a"),
    "opp_slot_a": ("p2", "a"),
}


def _slot_id(side: str, pos: str) -> str:
    return f"{side}{pos}"


def _hp_from_slot_data(data: dict[str, Any]) -> tuple[int, int]:
    hp = int(data.get("hp") or 0)
    max_hp = int(data.get("max_hp") or 0)
    pct = data.get("hp_percent")
    if max_hp <= 0 and pct is not None:
        max_hp = 100
        hp = int(round(float(pct)))
    if max_hp <= 0 and hp > 0:
        max_hp = hp
    return hp, max_hp


class LiveBattleTracker:
    """
    Match-scoped memory that accumulates partial CV observations into BattleLogState.

    A single frame cannot reveal bench species, boosts, or full move lists; this tracker
    merges repeated observations across turns.
    """

    def __init__(
        self,
        *,
        battle_format: BattleFormat = "doubles",
        player_side: PlayerSide = "p1",
    ):
        self.battle_format = battle_format
        self.player_side: PlayerSide = player_side
        self.opponent_side: PlayerSide = "p2" if player_side == "p1" else "p1"
        self.state = BattleLogState()
        self.last_ui_state: str = "UNKNOWN"
        self._team_index: dict[str, int] = {}
        self._history: list[np.ndarray] = []
        self.ally_species: list[str] = []
        self.enemy_species: list[str] = []
        # Our Bring-3 (set at preview) so unseen bench mons are switch-legal.
        self.brought_ally: list[str] = []
        # Per-(side, species) last-known mon, so a mon that switches out keeps its
        # HP / status / revealed moves for the bench tokens and switch mask.
        self._mon_memory: dict[str, MonPerspective] = {}

    def reset(self) -> None:
        self.state = BattleLogState()
        self.last_ui_state = "UNKNOWN"
        self._team_index.clear()
        self._history.clear()
        self.ally_species.clear()
        self.enemy_species.clear()
        self.brought_ally.clear()
        self._mon_memory.clear()

    def record_team_preview(self, ally_team: list[str], enemy_team: list[str]) -> None:
        """Store the 6 ally + 6 enemy species seen at team preview (Turn 0).

        Keeps full 6-length lists (placeholders included) for the preview model,
        and seeds the per-side rosters with the resolvable species.
        """
        self.ally_species = [to_id_str(s) for s in ally_team]
        self.enemy_species = [to_id_str(s) for s in enemy_team]
        for side, species_list in (
            (self.player_side, self.ally_species),
            (self.opponent_side, self.enemy_species),
        ):
            roster = self.state.team_roster.setdefault(side, [])
            for species in species_list:
                if species and species != "unknown" and species not in roster:
                    roster.append(species)

    def record_brought_ally(self, brought_species: list[str]) -> None:
        """Record our Bring-3 (or Bring-4) from team preview.

        Seeds bench memory at full HP so a not-yet-seen bench mon is encoded as a
        healthy, switch-legal candidate -- matching the replay/eval view, which
        knows the full brought team from turn 1.
        """
        self.brought_ally = [
            to_id_str(s) for s in brought_species if s and to_id_str(s) != "unknown"
        ]
        roster = self.state.team_roster.setdefault(self.player_side, [])
        for species in self.brought_ally:
            if species not in roster:
                roster.append(species)
            key = self._mem_key(self.player_side, species)
            if key not in self._mon_memory:
                self._mon_memory[key] = MonPerspective(
                    species=species, hp=100, max_hp=100, seen=False, active=False
                )

    def _mem_key(self, side: str, species: str) -> str:
        return f"{side}:{roster_species_key(species)}"

    def _remember_actives(self) -> None:
        """Persist each currently-active mon by species so bench memory survives switches."""
        for slot, mon in self.state.mons.items():
            if mon.species and mon.active:
                self._mon_memory[self._mem_key(slot[:2], mon.species)] = copy.deepcopy(mon)

    def _slot_map(self) -> dict[str, tuple[str, str]]:
        if self.battle_format == "singles":
            return _SLOT_MAP_SINGLES
        return _SLOT_MAP_DOUBLES

    def _remap_side(self, logical_side: str) -> str:
        if logical_side == "p1":
            return self.player_side
        return self.opponent_side

    def _ensure_mon(self, slot: str) -> MonPerspective:
        if slot not in self.state.mons:
            self.state.mons[slot] = MonPerspective(slot=slot)
        return self.state.mons[slot]

    def _assign_team_index(self, slot: str, species: str) -> None:
        if species and species not in self._team_index:
            self._team_index[species] = len(self._team_index)
        if species in self._team_index:
            self.state.mons[slot].team_index = self._team_index[species]

    def update_from_perception(self, perception: dict[str, Any]) -> BattleLogState:
        """
        Merge a perception payload into internal state.

        Accepts either a full PerceptionResult-like dict or the nested ``ocr`` field.
        """
        ui_state = str(perception.get("state", self.last_ui_state))
        self.last_ui_state = ui_state

        if ui_state == "RESULTS":
            self.state.finished = True

        ocr = perception.get("ocr")
        if not isinstance(ocr, dict):
            ocr = perception

        fmt_hint = perception.get("battle_format")
        if fmt_hint in ("singles", "doubles"):
            self.battle_format = fmt_hint

        for ocr_key, (logical_side, pos) in self._slot_map().items():
            slot_data = ocr.get(ocr_key)
            if not isinstance(slot_data, dict):
                continue

            side = self._remap_side(logical_side)
            slot = _slot_id(side, pos)
            mon = self._ensure_mon(slot)

            name = str(slot_data.get("species_id") or slot_data.get("name") or "").strip()
            if name:
                species = to_id_str(name)
                mon.species = species
                mon.seen = True
                self._assign_team_index(slot, species)
                self.state.active[slot] = species
                mon.active = ui_state in {
                    "TURN_DECISION",
                    "MOVE_SELECTION",
                    "TARGET_SELECTION",
                    "ANIMATION",
                }
                brought = self.state.brought_species.setdefault(side, set())
                brought.add(species)
                roster = self.state.team_roster.setdefault(side, [])
                if species not in roster:
                    roster.append(species)

            hp, max_hp = _hp_from_slot_data(slot_data)
            if max_hp > 0:
                mon.max_hp = max_hp
                mon.hp = max(0, min(hp, max_hp))
                mon.fainted = mon.hp <= 0
                if mon.species:
                    mon.seen = True

        preview = ocr.get("teampreview")
        if isinstance(preview, dict):
            required = int(preview.get("required") or 0)
            if required == 3:
                self.battle_format = "singles"
            elif required == 4:
                self.battle_format = "doubles"

        self._remember_actives()
        return self.state

    def _active_slots_for_side(self, side: str) -> list[str]:
        return [
            slot
            for slot, mon in self.state.mons.items()
            if slot.startswith(side) and mon.active
        ]

    def _resolve_active_slot(self, target_id: str, is_opponent: bool) -> str | None:
        """Map a parsed species name to a currently-active slot on the right side."""
        side = self.opponent_side if is_opponent else self.player_side
        candidates = self._active_slots_for_side(side)
        if not candidates:
            return None

        target_key = roster_species_key(target_id)
        # 1. Exact roster-species match (handles base forms / mega suffixes).
        for slot in candidates:
            if roster_species_key(self.state.mons[slot].species) == target_key:
                return slot
        # 2. Substring match (OCR truncation / form differences).
        for slot in candidates:
            species = self.state.mons[slot].species
            if species and (species in target_id or target_id in species):
                return slot
        # 3. Unambiguous fallback: a single active mon on that side.
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _resolve_active_by_species(self, target_id: str) -> str | None:
        """Find an active slot on either side whose species matches (no side hint).

        Used for the ability/item banner, which names the holder but not its side.
        Unlike ``_resolve_active_slot`` there is no single-active fallback, so we
        never attribute a reveal to the wrong mon.
        """
        target_key = roster_species_key(target_id)
        actives = [
            (slot, mon)
            for slot, mon in self.state.mons.items()
            if mon.active and mon.species
        ]
        for slot, mon in actives:
            if roster_species_key(mon.species) == target_key:
                return slot
        for slot, mon in actives:
            if mon.species in target_id or target_id in mon.species:
                return slot
        return None

    def apply_log_event(self, event: dict[str, Any] | None) -> bool:
        """Mutate internal state from a battle_log_parser event dict.

        Returns ``True`` when the event changed state. Field-wide events (weather,
        terrain) always apply; subject events require resolving an active slot.
        """
        if not isinstance(event, dict):
            return False
        kind = event.get("type")

        if kind == "weather":
            self.state.field.weather = str(event.get("weather", ""))
            return True
        if kind == "terrain":
            self.state.field.terrain = str(event.get("terrain", ""))
            return True

        if kind == "ability_item":
            holder = str(event.get("holder") or "")
            name_id = str(event.get("name_id") or "")
            if not holder or not name_id:
                return False
            slot = self._resolve_active_by_species(holder)
            if slot is None:
                return False
            mon = self.state.mons[slot]
            if event.get("subtype") == "ability":
                if mon.ability == name_id and mon.ability_revealed:
                    return False
                mon.ability = name_id
                mon.ability_revealed = True
            else:
                if mon.item == name_id and mon.item_revealed:
                    return False
                mon.item = name_id
                mon.item_revealed = True
            return True

        is_opponent = bool(event.get("is_opponent", False))
        subject = str(event.get("target") or event.get("user") or "")
        if not subject:
            return False
        slot = self._resolve_active_slot(subject, is_opponent)
        if slot is None:
            return False
        mon = self.state.mons[slot]

        if kind == "stat_boost":
            amount = int(event.get("amount", 0))
            changed = False
            for stat in event.get("stats", []):
                new_val = max(-6, min(6, mon.boosts.get(stat, 0) + amount))
                if new_val != mon.boosts.get(stat, 0):
                    changed = True
                mon.boosts[stat] = new_val
            return changed
        if kind == "faint":
            mon.hp = 0
            mon.fainted = True
            mon.active = False
            return True
        if kind == "status":
            mon.status = str(event.get("status", ""))
            return True
        if kind == "mega_evolve":
            # Resolved to an active slot; no boost/forme field to mutate yet, but
            # acknowledge so the shadow loop logs it as applied rather than dropped.
            return True
        if kind == "move":
            move_id = str(event.get("move", ""))
            if not move_id:
                return False
            apply_reveal_move(mon, move_id)
            mon._turn_move_id = move_vocab_id(move_id)
            mon.last_move_id = mon._turn_move_id
            return True
        return False

    def to_battle_log_state(self) -> BattleLogState:
        return self.state.clone()

    def update_turn(self, battle_data: dict[str, Any], *, ui_state: str = "TURN_DECISION") -> BattleLogState:
        """Merge a turn-decision OCR payload (from extract_battle_data) into state."""
        return self.update_from_perception({"state": ui_state, "ocr": battle_data})

    def _bench_candidates(self, side: str) -> list[str]:
        """Species for a side's bench, in preference order (no duplicates/actives).

        Our side draws on the known Bring-3 (so unseen mons are still switch-legal);
        the opponent's bench is only the mons we've actually seen (first-person).
        """
        ordered: list[str] = []
        seen_keys: set[str] = set()
        sources: list[str] = list(self.brought_ally) if side == self.player_side else []
        prefix = f"{side}:"
        for key, mon in self._mon_memory.items():
            if key.startswith(prefix) and mon.species:
                sources.append(mon.species)
        for species in sources:
            key = roster_species_key(species)
            if key and key not in seen_keys:
                seen_keys.add(key)
                ordered.append(species)
        return ordered

    def _add_bench_mons(self, state: BattleLogState, side: str, species_list: list[str]) -> None:
        active_keys = {
            roster_species_key(mon.species)
            for slot, mon in state.mons.items()
            if mon.active and slot.startswith(side) and mon.species
        }
        used_slots = {slot for slot in state.mons if slot.startswith(side)}
        bench_slots = [f"{side}{c}" for c in "bcdef"]
        placed = set(active_keys)
        next_idx = 0
        for species in species_list:
            key = roster_species_key(species)
            if not key or key in placed:
                continue
            while next_idx < len(bench_slots) and bench_slots[next_idx] in used_slots:
                next_idx += 1
            if next_idx >= len(bench_slots):
                break
            slot = bench_slots[next_idx]
            next_idx += 1
            used_slots.add(slot)
            placed.add(key)
            mem = self._mon_memory.get(self._mem_key(side, species))
            mon = copy.deepcopy(mem) if mem is not None else MonPerspective(species=species)
            mon.slot = slot
            mon.active = False
            if not mon.species:
                mon.species = species
            if mon.max_hp <= 0 and not mon.fainted:
                mon.max_hp = 100
                mon.hp = 100
            state.mons[slot] = mon

    def _state_with_bench(self) -> BattleLogState:
        """Clone of the live state augmented with bench mons for encode + masking.

        Bench reconstruction lives here (not in ``self.state``) so slot-resolution
        of log events keeps operating on the real, observed actives only.
        """
        self._remember_actives()
        clone = self.state.clone()
        self._add_bench_mons(clone, self.player_side, self._bench_candidates(self.player_side))
        self._add_bench_mons(clone, self.opponent_side, self._bench_candidates(self.opponent_side))
        return clone

    def encode_snapshot(self, *, force_switch: bool = False) -> np.ndarray:
        """Single-turn (13, N_FIELDS) tensor from the current state (BC encoder).

        Encodes the bench-augmented view so bench tokens match the replay/eval
        encoder, and sets the same ``force_switch`` decision flag on token 0.
        """
        from src.core.data.state_tokenizer import encode_log_state

        return encode_log_state(
            self._state_with_bench(),
            self.player_side,
            format=self.battle_format,
            force_switch=force_switch,
        )

    def get_model_inputs(self) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Return (stacked_observation, per_slot_masks) for model inference.

        The observation is the rolling ``(TRAJECTORY_DEPTH * 13, N_FIELDS)`` tensor
        produced by the same ``encode_log_state`` / ``push_trajectory`` path used by
        BC training. Calling this advances the trajectory history by one turn, so
        invoke it once per decision.

        Trajectory parity: like training, the frame stored in history is the
        force-switch-free turn-start snapshot, while the stacked t0 frame carries
        the live ``force_switch`` flag. The legal-action mask is rebuilt from the
        same bench-augmented view so masked argmax matches offline eval / live.
        """
        from src.core.data.state_tokenizer import (
            TRAJECTORY_DEPTH,
            encode_log_state,
            push_trajectory,
        )

        force_switch = self.last_ui_state == "FORCE_SWITCH"
        view = self._state_with_bench()
        snapshot = encode_log_state(
            view, self.player_side, format=self.battle_format, force_switch=force_switch
        )
        history_snapshot = (
            encode_log_state(view, self.player_side, format=self.battle_format, force_switch=False)
            if force_switch
            else snapshot
        )
        obs = push_trajectory(
            self._history,
            snapshot,
            depth=TRAJECTORY_DEPTH,
            maxlen=TRAJECTORY_DEPTH,
            history_snapshot=history_snapshot,
        )
        masks = self._legal_action_masks(state=view, force_switch=force_switch)
        return obs, masks

    def _legal_action_masks(
        self,
        *,
        state: BattleLogState | None = None,
        force_switch: bool = False,
    ) -> dict[str, np.ndarray] | None:
        """Best-effort legality masks from the (bench-augmented) log view.

        Singles and doubles both reconstruct the same masks the BC eval path uses,
        so the CV decoder applies the identical legal-action filter before argmax.
        """
        view = state if state is not None else self._state_with_bench()
        if self.battle_format == "singles":
            try:
                from src.singles.log_action_mask import singles_mask_for_eval

                sample_kind = "force_switch" if force_switch else "turn"
                mask = singles_mask_for_eval(
                    view, side=self.player_side, sample_kind=sample_kind
                )
                return {"slot_a": mask} if mask is not None else None
            except Exception:
                return None
        try:
            from src.doubles.data.log_action_mask import log_turn_slot_mask

            return {
                "slot_a": log_turn_slot_mask(view, self.player_side, "a"),
                "slot_b": log_turn_slot_mask(view, self.player_side, "b"),
            }
        except Exception:
            return None

    @property
    def ready_for_inference(self) -> bool:
        return self.last_ui_state in {"TURN_DECISION", "MOVE_SELECTION", "TARGET_SELECTION"}
