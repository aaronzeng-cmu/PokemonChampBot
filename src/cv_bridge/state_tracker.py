"""Persistent battle memory updated from CV perception."""

from __future__ import annotations

import copy
import json
from pathlib import Path
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


# Items that lock a mon into the first move it uses until it switches out.
_CHOICE_ITEMS = frozenset({"choiceband", "choicespecs", "choicescarf"})

# Single-target protect-family moves whose consecutive use raises the in-game
# fail chance (encoded as protect_counter). Used so the model sees its own
# Protect streak live and stops spamming it.
_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "kingsshield",
        "spikyshield",
        "banefulbunker",
        "obstruct",
        "silktrap",
        "burningbulwark",
        "maxguard",
    }
)


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
        # Our brought lineup (set at preview) so unseen bench mons are switch-legal.
        self.brought_ally: list[str] = []
        # Per-(side, species) last-known mon, so a mon that switches out keeps its
        # HP / status / revealed moves for the bench tokens and switch mask.
        self._mon_memory: dict[str, MonPerspective] = {}
        # Our full known team (species -> {moves, ability, item, max_hp, mega}) read
        # from the in-game team view. Lets us fill our actives' move lists (so the
        # legal-move mask isn't empty) and model the full bench before mons are seen.
        self._known_team: dict[str, dict[str, Any]] = {}
        self.known_team_species: list[str] = []
        # Choice-item lock per active slot suffix ("a"/"b"): the move the mon used
        # while staying in, plus the species it belongs to (so the lock drops on a
        # switch). Move legality is restricted to this move until the mon leaves.
        self._choice_lock_move: dict[str, str] = {}
        self._choice_lock_species: dict[str, str] = {}
        # Per active slot suffix: (species_key, consecutive Protect-family uses).
        self._protect_streak: dict[str, tuple[str | None, int]] = {}

    def reset(self) -> None:
        self.state = BattleLogState()
        self.last_ui_state = "UNKNOWN"
        self._team_index.clear()
        self._history.clear()
        self.ally_species.clear()
        self.enemy_species.clear()
        self.brought_ally.clear()
        self._mon_memory.clear()
        self._choice_lock_move.clear()
        self._choice_lock_species.clear()
        self._protect_streak.clear()
        if self._known_team:
            self._seed_known_team()

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
        """Record our brought lineup (Bring-3 singles / Bring-4 doubles) from preview.

        Seeds bench memory so a not-yet-seen bench mon is encoded as a healthy,
        switch-legal candidate -- matching the replay/eval view, which knows the
        full brought team from turn 1 -- and restricts ``brought_species`` (used by
        the switch mask) to exactly the mons we actually brought.
        """
        self.brought_ally = [
            to_id_str(s) for s in brought_species if s and to_id_str(s) != "unknown"
        ]
        roster = self.state.team_roster.setdefault(self.player_side, [])
        brought_set = self.state.brought_species.setdefault(self.player_side, set())
        for species in self.brought_ally:
            if species not in roster:
                roster.append(species)
            brought_set.add(roster_species_key(species))
            key = self._mem_key(self.player_side, species)
            if key not in self._mon_memory:
                profile = self._known_team.get(roster_species_key(species))
                self._mon_memory[key] = self._mon_from_profile(species, profile)
        self._seed_lead_actives()

    def _seed_lead_actives(self) -> None:
        """Place our chosen leads on the field before any (noisy) sprite read.

        The first picks in team-preview selection order are our leads (2 for
        doubles, 1 for singles). Seeding them as active from turn 1 means the
        switch mask already excludes them (so we never offer to switch in an
        already-active mon) and their movesets are available immediately.
        """
        n_leads = 2 if self.battle_format == "doubles" else 1
        for pos, species in zip(("a", "b"), self.brought_ally[:n_leads]):
            slot = _slot_id(self.player_side, pos)
            profile = self._known_team.get(roster_species_key(species))
            mon = self._mon_from_profile(species, profile)
            mon.slot = slot
            mon.active = True
            self.state.active[slot] = species
            self.state.mons[slot] = mon

    def _species_by_max_hp(self, max_hp: int) -> str | None:
        """Identify one of our known mons by its (unique) max HP."""
        if max_hp <= 0:
            return None
        matches = [
            sp
            for sp in self.known_team_species
            if int((self._known_team.get(roster_species_key(sp)) or {}).get("max_hp") or 0) == max_hp
        ]
        return matches[0] if len(matches) == 1 else None

    def record_party_readout(self, rows: list[dict[str, Any]]) -> None:
        """Sync our bench HP + faint state from the party (force-switch) screen.

        Each row's max HP uniquely identifies one of our known mons, so we can
        persist which mons have fainted -- the switch mask must exclude them, and
        a fainted active that's been replaced otherwise lingers in bench memory as
        a (wrongly) switch-legal candidate.
        """
        if not rows:
            return
        for row in rows:
            try:
                mx = int(row.get("max_hp") or 0)
                hp = int(row.get("hp") or 0)
            except (TypeError, ValueError):
                continue
            species = self._species_by_max_hp(mx)
            if not species:
                continue
            fainted = hp <= 0
            key = self._mem_key(self.player_side, species)
            mem = self._mon_memory.get(key)
            if mem is None:
                profile = self._known_team.get(roster_species_key(species))
                mem = self._mon_from_profile(species, profile)
            mem.hp = hp
            mem.max_hp = mx
            mem.fainted = fainted
            mem.seen = True
            self._mon_memory[key] = mem
            for slot, mon in self.state.mons.items():
                if slot.startswith(self.player_side) and roster_species_key(
                    mon.species
                ) == roster_species_key(species):
                    mon.hp = hp
                    mon.max_hp = mx
                    mon.fainted = fainted

    def load_player_team_file(self, path: str | Path) -> None:
        """Load our known team profile (JSON) and seed the tracker from it."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_player_team(data)

    def load_player_team(self, profile: dict[str, Any]) -> None:
        """Register our full team so our actives carry moves and the bench is modeled.

        ``profile`` is the team-view JSON: ``{"pokemon": [{species, ability, item,
        moves, max_hp, mega}, ...]}``. All names are normalized to poke-env ids so
        they line up with the CV sprite matcher and the BC encoder / mask.
        """
        self._known_team = {}
        self.known_team_species = []
        for entry in profile.get("pokemon", []):
            species = to_id_str(str(entry.get("species") or ""))
            if not species or species == "unknown":
                continue
            key = roster_species_key(species)
            self._known_team[key] = {
                "species": species,
                "ability": to_id_str(str(entry.get("ability") or "")),
                "item": to_id_str(str(entry.get("item") or "")),
                "moves": [to_id_str(str(m)) for m in entry.get("moves", []) if str(m).strip()],
                "max_hp": int(entry.get("max_hp") or 0),
                "mega": bool(entry.get("mega", False)),
            }
            self.known_team_species.append(species)
        self._seed_known_team()

    def _seed_known_team(self) -> None:
        """Populate the player roster + bench memory from the known team."""
        if not self.known_team_species:
            return
        roster = self.state.team_roster.setdefault(self.player_side, [])
        for species in self.known_team_species:
            if species not in roster:
                roster.append(species)
            key = self._mem_key(self.player_side, species)
            profile = self._known_team.get(roster_species_key(species))
            # Don't clobber a richer in-battle memory (HP/boosts seen live).
            if key not in self._mon_memory or not self._mon_memory[key].seen:
                self._mon_memory[key] = self._mon_from_profile(species, profile)

    def _mon_from_profile(
        self, species: str, profile: dict[str, Any] | None
    ) -> MonPerspective:
        max_hp = int(profile.get("max_hp") or 0) if profile else 0
        if max_hp <= 0:
            max_hp = 100
        mon = MonPerspective(
            species=species, hp=max_hp, max_hp=max_hp, seen=False, active=False
        )
        if profile:
            mon.moves = list(profile.get("moves", []))
            if profile.get("ability"):
                mon.ability = profile["ability"]
                mon.ability_revealed = True
            if profile.get("item"):
                mon.item = profile["item"]
                mon.item_revealed = True
            mon.can_mega = bool(profile.get("mega", False))
        return mon

    def _apply_known_profile(self, mon: MonPerspective) -> None:
        """Fill an observed active with our known moves / ability / item / max HP.

        Perception can't read our move list off the battle HUD, so without this the
        legal-move mask is empty and the policy is forced to ``pass``.
        """
        profile = self._known_team.get(roster_species_key(mon.species))
        if not profile:
            return
        if profile.get("moves"):
            mon.moves = list(profile["moves"])
        if profile.get("ability"):
            mon.ability = profile["ability"]
            mon.ability_revealed = True
        if profile.get("item") and not mon.item_revealed:
            mon.item = profile["item"]
            mon.item_revealed = True
        known_max = int(profile.get("max_hp") or 0)
        if known_max > 0:
            mon.max_hp = known_max
            if mon.hp <= 0 and not mon.fainted:
                mon.hp = known_max
            mon.hp = min(mon.hp, known_max)
        if profile.get("mega") and not mon.mega and not mon.terastallized:
            mon.can_mega = True

    def _reset_volatiles_on_switch_in(
        self, mon: MonPerspective, slot: str, side: str, species: str
    ) -> None:
        """Reset slot-volatile state when a new species switches into ``slot``.

        The live tracker reuses one ``MonPerspective`` per slot, so without this a
        new mon inherits the previous occupant's boosts / Protect streak / last move
        / gimmick flags -- diverging from the training tracker, which resets these on
        switch-in (``_reset_temporal_on_switch_in`` + a fresh mon's empty boosts).
        Persistent identity (status, revealed moves/ability/item) is restored from
        per-species memory when we've seen this mon before; HP and our known move
        profile are (re)applied by the caller right after.
        """
        mon.boosts = {}
        mon.turns_active = 0
        mon.protect_counter = 0
        mon.last_move_id = 0
        mon._turn_move_id = 0
        mon._turn_protect_success = False
        mon.mega = False
        mon.terastallized = False
        mon.can_mega = False

        mem = self._mon_memory.get(self._mem_key(side, species))
        mon.status = mem.status if mem is not None else ""
        if mem is not None:
            if mem.moves and not mon.moves:
                mon.moves = list(mem.moves)
            if mem.ability_revealed and not mon.ability_revealed:
                mon.ability = mem.ability
                mon.ability_revealed = True
            if mem.item_revealed and not mon.item_revealed:
                mon.item = mem.item
                mon.item_revealed = True

        # Drop our own per-slot Protect-streak / Choice-lock bookkeeping for the
        # departed mon so the incomer starts clean (mirrors training's reset).
        if side == self.player_side:
            suffix = slot[len(side):]
            self._protect_streak.pop(suffix, None)
            self._choice_lock_move.pop(suffix, None)
            self._choice_lock_species.pop(suffix, None)

    def _reconcile_active_species(self, species: str) -> str | None:
        """Snap a player-side perception read to one of our known 6 (closed set).

        Our own actives are always a member of our known team, so a noisy sprite /
        OCR read is mapped to the nearest known species. Reads that match nothing
        (e.g. ``"unknown"`` or a foreign misread) return ``None`` so they never
        clobber a good active or pollute the switch mask's on-field set.
        """
        if not species or species == "unknown":
            return None
        if not self.known_team_species:
            return species
        key = roster_species_key(species)
        known_keys = {roster_species_key(s): s for s in self.known_team_species}
        if key in known_keys:
            return known_keys[key]
        import difflib

        match = difflib.get_close_matches(key, list(known_keys), n=1, cutoff=0.6)
        return known_keys[match[0]] if match else None

    def reconcile_preview_species(self, parsed: list[str]) -> list[str]:
        """Repair a CV-parsed preview lineup using our known 6 (by elimination).

        Sprite/OCR misreads at preview often yield ``"unknown"`` slots. Since the
        preview shows exactly our known team, any unresolved slot must be one of the
        missing known species, so we fill it in. Slot *order* is preserved so the
        returned list still indexes the on-screen roster_slot tap positions.
        """
        if not self.known_team_species:
            return list(parsed)
        known = list(self.known_team_species)
        known_keys = [roster_species_key(s) for s in known]
        resolved: list[str | None] = []
        used: set[str] = set()
        for sp in parsed:
            key = roster_species_key(sp) if sp and to_id_str(sp) != "unknown" else ""
            if key and key in known_keys and key not in used:
                resolved.append(known[known_keys.index(key)])
                used.add(key)
            else:
                resolved.append(None)
        missing = [known[i] for i, key in enumerate(known_keys) if key not in used]
        out: list[str] = []
        mi = 0
        for i, value in enumerate(resolved):
            if value is not None:
                out.append(value)
            elif mi < len(missing):
                out.append(missing[mi])
                mi += 1
            else:
                out.append(parsed[i] if i < len(parsed) else "unknown")
        return out

    def _mem_key(self, side: str, species: str) -> str:
        return f"{side}:{roster_species_key(species)}"

    def _remember_actives(self) -> None:
        """Persist each active (or just-fainted) mon by species so bench memory
        survives switches. Fainted mons must be remembered too, otherwise a mon that
        faints and is immediately replaced reverts to its stale last-alive snapshot
        and shows up as a (wrongly) switch-legal bench candidate."""
        for slot, mon in self.state.mons.items():
            if mon.species and (mon.active or mon.fainted):
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
            prev_species = mon.species  # occupant before this read (switch detection)

            # Identify the on-field mon by its sprite/icon (CNN match -- nickname-proof)
            # first; the nameplate is only a fallback because trainers can nickname
            # their Pokemon, so the plate text may not be the species at all.
            sprite_id = to_id_str(str(slot_data.get("species_id") or "").strip())
            if sprite_id == "unknown":
                sprite_id = ""
            name_id = to_id_str(str(slot_data.get("name") or "").strip())
            if name_id == "unknown":
                name_id = ""
            species = sprite_id or name_id
            if species and side == self.player_side:
                # Our actives are always one of our known 6; snap noisy reads and
                # drop reads matching nothing so an "unknown"/foreign misread never
                # overwrites a good active or breaks the on-field switch mask.
                species = self._reconcile_active_species(species) or ""
            if species:
                # A different species now occupies this slot -> a switch happened.
                # Match the training tracker: wipe the volatile state that belonged
                # to the mon that left (boosts/turns/protect/last-move/gimmick) and
                # restore the incomer's persistent memory. Compare by roster key so a
                # Mega evolution (same base species) is NOT treated as a switch.
                if (
                    prev_species
                    and roster_species_key(prev_species) != roster_species_key(species)
                ):
                    self._reset_volatiles_on_switch_in(mon, slot, side, species)
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

            # Overlay our known team data (move list, ability, item, exact max HP)
            # onto our own actives so the legal-move mask isn't empty.
            if species and side == self.player_side:
                self._apply_known_profile(mon)

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

    def active_species_keys(self, side: str | None = None) -> set[str]:
        """Roster keys of our on-field (active, non-fainted) mons.

        Used to veto a voluntary switch whose target is already on the field -- the
        game silently rejects selecting an active mon, which otherwise stalls the
        switch sequence.
        """
        side = side or self.player_side
        out: set[str] = set()
        for slot, mon in self.state.mons.items():
            if not slot.startswith(side):
                continue
            if mon.active and not mon.fainted and mon.species:
                out.add(roster_species_key(mon.species))
        return out

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

    def _register_protect_success(self, slot: str, *, is_opponent: bool) -> None:
        """Bump a mon's consecutive-Protect counter on an *observed* success.

        Mirrors BC training (``log_tracker._end_of_turn``), which increments
        ``protect_counter`` only when a Protect actually activated, not when the
        move was merely selected. Driven by a "<mon> protected itself" log line.
        """
        mon = self.state.mons.get(slot)
        if mon is None:
            return
        if is_opponent:
            mon.protect_counter = min(int(getattr(mon, "protect_counter", 0)) + 1, 4095)
            return
        suffix = slot[len(self.player_side):]
        species = roster_species_key(mon.species) if mon.species else None
        streak_species, streak_n = self._protect_streak.get(suffix, (None, 0))
        streak_n = streak_n + 1 if streak_species == species else 1
        self._protect_streak[suffix] = (species, streak_n)
        mon.protect_counter = min(streak_n, 4095)

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
            # Persist the faint into per-species memory *now*: the mon may be
            # replaced before the next perception read, after which it only exists
            # as a bench token reconstructed from memory. Without this it reverts to
            # its last-alive snapshot and the switch mask offers a dead mon.
            if mon.species:
                self._mon_memory[self._mem_key(slot[:2], mon.species)] = copy.deepcopy(mon)
            return True
        if kind == "status":
            mon.status = str(event.get("status", ""))
            return True
        if kind == "protect":
            self._register_protect_success(slot, is_opponent=is_opponent)
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
            # Mirror training's ``_record_move``: a non-Protect move breaks the
            # streak. (Protect success is counted separately on its own log line.)
            # This is the reset path for the *opponent*, whose moves we only see
            # via the log; our own reset also runs in ``record_committed_move``.
            if move_id not in _PROTECT_MOVES:
                mon.protect_counter = 0
                if not is_opponent:
                    sp = roster_species_key(mon.species) if mon.species else None
                    self._protect_streak[slot[len(self.player_side):]] = (sp, 0)
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
        if side == self.player_side:
            # Our brought lineup is authoritative once preview is known; before that,
            # fall back to the full known team so switches aren't spuriously illegal.
            # (Per-mon HP/state is still pulled from memory in ``_add_bench_mons``.)
            sources = list(self.brought_ally) or list(self.known_team_species)
        else:
            # The opponent bench is only what we've actually seen (first-person).
            sources = []
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
        # Don't re-add a species that already occupies a slot on this side -- whether
        # it's active OR a fainted active that hasn't been replaced yet. Otherwise a
        # just-fainted mon appears twice (dead in its slot + alive on the bench).
        existing_keys = {
            roster_species_key(mon.species)
            for slot, mon in state.mons.items()
            if slot.startswith(side) and mon.species
        }
        used_slots = {slot for slot in state.mons if slot.startswith(side)}
        bench_slots = [f"{side}{c}" for c in "bcdef"]
        placed = set(existing_keys)
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

    def get_model_inputs(
        self, *, advance: bool = True
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Return (stacked_observation, per_slot_masks) for model inference.

        The observation is the rolling ``(TRAJECTORY_DEPTH * 13, N_FIELDS)`` tensor
        produced by the same ``encode_log_state`` / ``push_trajectory`` path used by
        BC training. Training appends exactly one frame to history per game turn, so
        ``advance`` must be ``True`` only once per turn. When re-deciding the *same*
        turn (e.g. a dropped action triggered recovery), pass ``advance=False`` to
        re-stack against the existing history without pushing a duplicate frame --
        otherwise the same turn would occupy two trajectory slots and diverge from
        the training stack.

        Trajectory parity: like training, the frame stored in history is the
        force-switch-free turn-start snapshot, while the stacked t0 frame carries
        the live ``force_switch`` flag. The legal-action mask is rebuilt from the
        same bench-augmented view so masked argmax matches offline eval / live.
        """
        from src.core.data.state_tokenizer import (
            TRAJECTORY_DEPTH,
            encode_log_state,
            push_trajectory,
            stack_trajectory,
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
        if not advance:
            # Re-decision of the same turn. The first (advance=True) call this turn
            # already appended this turn's frame to history, so stack against history
            # WITHOUT that trailing frame to reproduce the original observation
            # (rather than showing the same turn in two trajectory slots).
            prior = self._history[:-1] if self._history else []
            masks = self._legal_action_masks(state=view, force_switch=force_switch)
            return stack_trajectory(prior, snapshot, depth=TRAJECTORY_DEPTH), masks
        obs = push_trajectory(
            self._history,
            snapshot,
            depth=TRAJECTORY_DEPTH,
            maxlen=TRAJECTORY_DEPTH,
            history_snapshot=history_snapshot,
        )
        masks = self._legal_action_masks(state=view, force_switch=force_switch)
        return obs, masks

    def record_committed_move(self, suffix: str, move_name: str | None) -> None:
        """Latch what *we* just told an active to do, so the next decision's input
        and masks reflect it. Keyed on our own choice (100% reliable) rather than
        flaky battle-log OCR.

        Updates three things on the active mon:
          * ``last_move_id`` -- the encoder feeds this to the model.
          * ``protect_counter`` -- consecutive *successful* Protect-family uses.
            BC training (``log_tracker``) only bumps this when a Protect actually
            activates (the ``-singleturn ...Protect`` hook), so we do NOT increment
            on commit here; the bump happens when a "<mon> protected itself" log
            line is observed (see ``_register_protect_success``). Committing a
            non-Protect move *resets* the streak, mirroring training's ``_record_move``.
          * Choice lock (item-gated) -- restricts the legal-move mask to this move.

        All reset implicitly when the slot's species changes (a switch).
        """
        if not move_name:
            return
        mon = self.state.mons.get(f"{self.player_side}{suffix}")
        if mon is None or not mon.active or mon.fainted:
            return
        mid = to_id_str(move_name)
        species = roster_species_key(mon.species)

        mon.last_move_id = move_vocab_id(mid)

        if mid not in _PROTECT_MOVES:
            self._protect_streak[suffix] = (species, 0)
            mon.protect_counter = 0

        if to_id_str(mon.item or "") in _CHOICE_ITEMS:
            self._choice_lock_species[suffix] = species
            self._choice_lock_move[suffix] = mid

    def choice_locked_move(self, suffix: str) -> str | None:
        """The move id this slot's Choice-item active is locked into, or None.

        Self-clears the lock when the slot is empty or its species changed (the
        mon switched out, so a Choice item no longer constrains it).
        """
        mon = self.state.mons.get(f"{self.player_side}{suffix}")
        if mon is None or not mon.active:
            self._choice_lock_move.pop(suffix, None)
            self._choice_lock_species.pop(suffix, None)
            return None
        locked = self._choice_lock_move.get(suffix)
        if not locked or mon.fainted:
            return None
        if to_id_str(mon.item or "") not in _CHOICE_ITEMS:
            return None
        if self._choice_lock_species.get(suffix) != roster_species_key(mon.species):
            self._choice_lock_move.pop(suffix, None)
            self._choice_lock_species.pop(suffix, None)
            return None
        return locked

    def active_party_rows(self) -> set[int]:
        """1-based party-screen rows occupied by our on-field (active) mons.

        The in-battle party / force-switch screen lists our brought team in
        preview-selection order, so a mon's row == its index in that order.
        Identifying actives by *species* (sprite-tracked, closed-set reconciled)
        is reliable even when the party-screen HP OCR disagrees with stale tracker
        HP -- which is what made the old HP-matching exclusion pick the still-active
        partner on a force switch.
        """
        brought = list(self.brought_ally) or list(
            self.state.team_roster.get(self.player_side, [])
        )
        if not brought:
            return set()
        brought_keys = [roster_species_key(s) for s in brought]
        rows: set[int] = set()
        for slot, mon in self.state.mons.items():
            if not slot.startswith(self.player_side):
                continue
            if not mon.active or mon.fainted or not mon.species:
                continue
            key = roster_species_key(mon.species)
            if key in brought_keys:
                rows.add(brought_keys.index(key) + 1)
        return rows

    def _apply_choice_lock(
        self, view: BattleLogState, masks: dict[str, np.ndarray]
    ) -> None:
        """Zero out non-locked move actions for Choice-locked actives.

        Switch (1-6) and pass (0) stay legal -- Choice only constrains moves. Move
        *slot* numbering uses the full move list so it still lines up with the
        executor's button mapping.
        """
        from src.doubles.battle.move_order import (
            canonical_move_list,
            decode_move_action_index,
        )

        for suffix, key in (("a", "slot_a"), ("b", "slot_b")):
            locked = self.choice_locked_move(suffix)
            if not locked:
                continue
            mon = view.mons.get(f"{self.player_side}{suffix}")
            if mon is None or not mon.moves:
                continue
            canon = [to_id_str(m) for m in canonical_move_list(list(mon.moves))]
            if locked not in canon:
                continue
            locked_slot = canon.index(locked) + 1
            mask = masks.get(key)
            if mask is None:
                continue
            for idx in range(7, len(mask)):
                if not mask[idx]:
                    continue
                move_slot, _toff, _mega, _tera = decode_move_action_index(idx)
                if move_slot != locked_slot:
                    mask[idx] = False
            if not mask.any():
                mask[0] = True

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

            masks = {
                "slot_a": log_turn_slot_mask(view, self.player_side, "a"),
                "slot_b": log_turn_slot_mask(view, self.player_side, "b"),
            }
            if not force_switch:
                self._apply_choice_lock(view, masks)
            return masks
        except Exception:
            return None

    @property
    def ready_for_inference(self) -> bool:
        return self.last_ui_state in {"TURN_DECISION", "MOVE_SELECTION", "TARGET_SELECTION"}
