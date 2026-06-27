"""Unified meta loader: Pikalytics priors + dex descriptions + pool fallbacks."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.settings import DEX_CACHE_PATH, OPPONENT_POOL_DIR, PIKALYTICS_META_PATH
from src.core.planning.dex_cache import ability_desc, load_dex_cache, move_desc
from src.core.planning.species_normalize import (
    _MEGA_FORM_ITEMS,
    clean_species_name,
    mega_family_keys,
    resolve_pikalytics_candidates,
)
from src.doubles.planning.spread_priors import aggregate_spread_priors, spread_key
from src.doubles.teams.pikalytics_meta import fetch_species_meta

logger = logging.getLogger(__name__)

_OTHER_LABEL = "Other"
_DEFAULT_SPREAD = spread_key("Serious", [0, 0, 0, 0, 0, 0])


@dataclass
class SpeciesPrior:
    species: str
    pikalytics_key: str | None = None
    moves: dict[str, float] = field(default_factory=dict)
    items: dict[str, float] = field(default_factory=dict)
    abilities: dict[str, float] = field(default_factory=dict)
    ev_spreads: dict[str, float] = field(default_factory=dict)
    tera_types: dict[str, float] = field(default_factory=dict)
    teammates: dict[str, float] = field(default_factory=dict)
    usage_pct: float | None = None
    form_variants: dict[str, float] = field(default_factory=dict)
    featured_sets: list[dict] = field(default_factory=list)
    source: str = "unknown"  # pikalytics | pool | default


def _filter_other(raw: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in raw.items() if k and k != _OTHER_LABEL and v > 0}


def _usage_list_raw(entries: list[dict]) -> dict[str, float]:
    """Keep Pikalytics usage % as-is (% of sets with this move/item/ability)."""
    return _filter_other(
        {e["name"]: float(e["usage_pct"]) for e in entries if e.get("name")}
    )


def _blend_weighted_usage(
    parts: list[tuple[dict[str, float], float]],
) -> dict[str, float]:
    """Usage-weighted blend of raw Pikalytics % distributions."""
    blended: dict[str, float] = {}
    for dist, weight in parts:
        if weight <= 0:
            continue
        for name, pct in dist.items():
            blended[name] = blended.get(name, 0.0) + weight * pct
    return _filter_other(blended)


def _normalize_dist(raw: dict[str, float]) -> dict[str, float]:
    filtered = _filter_other(raw)
    if not filtered:
        return {}
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in filtered.items()}


def _parse_evs_string(evs: str, nature: str) -> str:
    parts = evs.strip().split("/")
    if len(parts) != 6:
        return _DEFAULT_SPREAD
    try:
        ints = [int(p) for p in parts]
    except ValueError:
        return _DEFAULT_SPREAD
    return spread_key(nature, ints)


class MetaDatabase:
    def __init__(
        self,
        *,
        pikalytics_path: Path | None = None,
        dex_path: Path | None = None,
        pool_dir: Path | None = None,
        live_fetch: bool = True,
    ):
        self.pikalytics_path = pikalytics_path or PIKALYTICS_META_PATH
        self.dex_path = dex_path or DEX_CACHE_PATH
        self.pool_dir = pool_dir or OPPONENT_POOL_DIR
        self.live_fetch = live_fetch
        self._pikalytics: dict = {}
        self._dex: dict = {}
        self._spread_priors: dict[str, dict[str, float]] = {}
        self._pool_move_priors: dict[str, dict[str, float]] = {}
        self._pool_item_priors: dict[str, dict[str, float]] = {}
        self._pool_ability_priors: dict[str, dict[str, float]] = {}
        self._pool_tera_priors: dict[str, dict[str, float]] = {}
        self._battle_usage: dict[str, float] = {}
        self.reload()

    def reload(self) -> None:
        if self.pikalytics_path.is_file():
            self._pikalytics = json.loads(self.pikalytics_path.read_text(encoding="utf-8"))
        else:
            self._pikalytics = {"format": {}, "pokemon": {}}
        self._battle_usage = {
            k: float(v)
            for k, v in self._pikalytics.get("battle_usage", {}).items()
            if v is not None
        }
        self._dex = load_dex_cache(self.dex_path)
        self._spread_priors = aggregate_spread_priors(self.pool_dir)
        self._aggregate_pool_set_priors()

    def _aggregate_pool_set_priors(self) -> None:
        from poke_env.teambuilder import Teambuilder

        move_counts: dict[str, dict[str, int]] = {}
        item_counts: dict[str, dict[str, int]] = {}
        ability_counts: dict[str, dict[str, int]] = {}
        tera_counts: dict[str, dict[str, int]] = {}

        for path in sorted(self.pool_dir.glob("*.txt")):
            try:
                mons = Teambuilder.parse_showdown_team(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for mon in mons:
                raw_name = mon.nickname or mon.species or ""
                species = clean_species_name(raw_name)
                if not species:
                    continue
                for move in mon.moves or []:
                    move_counts.setdefault(species, {}).setdefault(move, 0)
                    move_counts[species][move] += 1
                if mon.item:
                    item_counts.setdefault(species, {}).setdefault(mon.item, 0)
                    item_counts[species][mon.item] += 1
                if mon.ability:
                    ability_counts.setdefault(species, {}).setdefault(mon.ability, 0)
                    ability_counts[species][mon.ability] += 1
                if mon.tera_type:
                    tera_counts.setdefault(species, {}).setdefault(mon.tera_type, 0)
                    tera_counts[species][mon.tera_type] += 1

        def _to_prob(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
            out: dict[str, dict[str, float]] = {}
            for species, counter in counts.items():
                total = sum(counter.values())
                if total > 0:
                    out[species] = {k: v / total for k, v in counter.items()}
            return out

        self._pool_move_priors = _to_prob(move_counts)
        self._pool_item_priors = _to_prob(item_counts)
        self._pool_ability_priors = _to_prob(ability_counts)
        self._pool_tera_priors = _to_prob(tera_counts)

    @property
    def format_meta(self) -> dict:
        return self._pikalytics.get("format", {})

    @property
    def cached_species_count(self) -> int:
        return len(self._pikalytics.get("pokemon", {}))

    def _pool_key(self, species: str) -> str:
        return clean_species_name(species)

    def _lookup_cached(self, key: str) -> dict | None:
        pokemon = self._pikalytics.get("pokemon", {})
        if key in pokemon:
            return pokemon[key]
        # case-insensitive fallback
        lower = key.lower()
        for name, entry in pokemon.items():
            if name.lower() == lower:
                return entry
        return None

    def _fetch_live(self, key: str) -> dict | None:
        if not self.live_fetch:
            return None
        try:
            meta = fetch_species_meta(key, delay_s=0.0)
            entry = asdict(meta)
            self._pikalytics.setdefault("pokemon", {})[key] = entry
            return entry
        except Exception as exc:
            logger.debug("Live Pikalytics fetch failed for %s: %s", key, exc)
            return None

    def resolve_pikalytics_entry(
        self, species: str, *, item: str = ""
    ) -> tuple[str | None, dict | None]:
        for key in resolve_pikalytics_candidates(species, item=item):
            cached = self._lookup_cached(key)
            if cached and cached.get("moves"):
                return key, cached
        for key in resolve_pikalytics_candidates(species, item=item):
            live = self._fetch_live(key)
            if live and live.get("moves"):
                return key, live
        return None, None

    def _ensure_battle_usage(self) -> None:
        if self._battle_usage or not self.live_fetch:
            return
        try:
            from src.doubles.teams.pikalytics_meta import fetch_battle_usage_list

            for row in fetch_battle_usage_list(delay_s=0.0):
                name = str(row.get("name") or "").strip()
                pct = row.get("percent")
                if name and pct is not None:
                    self._battle_usage[name] = float(pct)
        except Exception as exc:
            logger.debug("Battle usage fetch failed: %s", exc)

    def _form_usage_weights(self, family: list[str]) -> dict[str, float]:
        """Normalized ladder usage weights across a mega family."""
        self._ensure_battle_usage()
        raw: dict[str, float] = {}
        for key in family:
            if key in self._battle_usage:
                raw[key] = self._battle_usage[key]
            else:
                entry = self._lookup_cached(key)
                if entry and entry.get("usage_pct") is not None:
                    raw[key] = float(entry["usage_pct"])
        if not raw:
            return {family[0]: 1.0}
        total = sum(raw.values())
        if total <= 0:
            return {family[0]: 1.0}
        return {k: v / total for k, v in raw.items()}

    def _resolve_family_entries(
        self, family: list[str]
    ) -> list[tuple[str, dict, float]]:
        weights = self._form_usage_weights(family)
        resolved: list[tuple[str, dict, float]] = []
        for key in family:
            weight = weights.get(key, 0.0)
            if weight <= 0:
                continue
            cached = self._lookup_cached(key)
            if cached and cached.get("moves"):
                resolved.append((key, cached, weight))
                continue
            live = self._fetch_live(key)
            if live and live.get("moves"):
                resolved.append((key, live, weight))
        return resolved

    def _prior_from_entry(
        self, norm: str, pika_key: str | None, raw: dict | None
    ) -> SpeciesPrior:
        prior = SpeciesPrior(
            species=norm,
            pikalytics_key=pika_key,
            usage_pct=raw.get("usage_pct") if raw else None,
            featured_sets=raw.get("featured_sets", []) if raw else [],
        )
        if raw:
            prior.source = "pikalytics"
            prior.moves = _usage_list_raw(raw.get("moves", []))
            prior.items = _usage_list_raw(raw.get("items", []))
            prior.abilities = _usage_list_raw(raw.get("abilities", []))
            prior.teammates = _usage_list_raw(raw.get("teammates", []))
        else:
            prior.source = "pool"
            prior.moves = dict(self._pool_move_priors.get(norm, {}))
            prior.items = dict(self._pool_item_priors.get(norm, {}))
            prior.abilities = dict(self._pool_ability_priors.get(norm, {}))
            prior.teammates = {}
        return prior

    def _apply_prior_defaults(self, prior: SpeciesPrior, norm: str) -> SpeciesPrior:
        if not prior.moves:
            prior.moves = {"Protect": 50.0}
            prior.source = "default"
        if not prior.items:
            prior.items = {"Sitrus Berry": 50.0}
        if not prior.abilities:
            prior.abilities = {"": 100.0}

        ev_spreads: dict[str, float] = {}
        pika_key = prior.pikalytics_key
        raw = self._lookup_cached(pika_key) if pika_key else None
        if raw and raw.get("top_nature") and raw.get("top_evs"):
            key = _parse_evs_string(raw["top_evs"], raw["top_nature"])
            pct = raw.get("top_spread_pct") or 100.0
            ev_spreads[key] = float(pct)
        for k, v in self._spread_priors.get(norm, {}).items():
            ev_spreads[k] = ev_spreads.get(k, 0.0) + v
        prior.ev_spreads = _normalize_dist(ev_spreads) or {_DEFAULT_SPREAD: 1.0}

        tera = dict(self._pool_tera_priors.get(norm, {}))
        prior.tera_types = _normalize_dist(tera) or {"Normal": 1.0}
        return prior

    def _get_blended_family_prior(self, species: str) -> SpeciesPrior:
        norm = self._pool_key(species)
        family = mega_family_keys(species) or [norm]
        entries = self._resolve_family_entries(family)
        if not entries:
            pika_key, raw = self.resolve_pikalytics_entry(species)
            prior = self._prior_from_entry(norm, pika_key, raw)
            return self._apply_prior_defaults(prior, norm)

        form_usage = {
            key: self._battle_usage[key]
            for key in family
            if key in self._battle_usage
        }
        for key, raw, _ in entries:
            if key not in form_usage and raw.get("usage_pct") is not None:
                form_usage[key] = float(raw["usage_pct"])

        move_parts: list[tuple[dict[str, float], float]] = []
        item_parts: list[tuple[dict[str, float], float]] = []
        ability_parts: list[tuple[dict[str, float], float]] = []
        teammate_parts: list[tuple[dict[str, float], float]] = []
        featured: list[dict] = []
        dominant_key, dominant_weight = entries[0][0], entries[0][2]

        for key, raw, weight in entries:
            if weight > dominant_weight:
                dominant_key = key
                dominant_weight = weight
            move_parts.append((_usage_list_raw(raw.get("moves", [])), weight))
            item_parts.append((_usage_list_raw(raw.get("items", [])), weight))
            ability_parts.append((_usage_list_raw(raw.get("abilities", [])), weight))
            teammate_parts.append((_usage_list_raw(raw.get("teammates", [])), weight))
            featured.extend(raw.get("featured_sets", []))
            stone = _MEGA_FORM_ITEMS.get(key)
            if stone:
                item_parts.append(({stone: 100.0}, weight))

        prior = SpeciesPrior(
            species=norm,
            pikalytics_key=dominant_key,
            usage_pct=sum(form_usage.values()) if form_usage else None,
            form_variants=form_usage,
            featured_sets=featured,
            source="pikalytics",
            moves=_blend_weighted_usage(move_parts),
            items=_blend_weighted_usage(item_parts),
            abilities=_blend_weighted_usage(ability_parts),
            teammates=_blend_weighted_usage(teammate_parts),
        )
        return self._apply_prior_defaults(prior, norm)

    def top_moves_raw(self, species: str, *, item: str = "", n: int = 8) -> dict[str, float]:
        """Top-N move usage % from Pikalytics (raw, not renormalized)."""
        prior = self.get_species_prior(species, item=item)
        top = sorted(prior.moves.items(), key=lambda x: -x[1])[:n]
        return dict(top)

    def get_species_prior(self, species: str, *, item: str = "") -> SpeciesPrior:
        norm = self._pool_key(species)
        if not item and mega_family_keys(species):
            return self._get_blended_family_prior(species)

        pika_key, raw = self.resolve_pikalytics_entry(species, item=item)
        prior = self._prior_from_entry(norm, pika_key, raw)
        if pika_key:
            prior.form_variants = {pika_key: prior.usage_pct or 0.0}
        return self._apply_prior_defaults(prior, norm)

    def persist_cache(self) -> None:
        """Write in-memory Pikalytics cache (including live fetches) to disk."""
        self.pikalytics_path.parent.mkdir(parents=True, exist_ok=True)
        self._pikalytics["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.pikalytics_path.write_text(
            json.dumps(self._pikalytics, indent=2), encoding="utf-8"
        )

    def move_description(self, move_name: str) -> str:
        return move_desc(self._dex, move_name)

    def ability_description(self, ability_name: str) -> str:
        return ability_desc(self._dex, ability_name)

    def get_matchup_context(self, our_team: list[str], opp_team: list[str]) -> str:
        """Cores and teammate synergy only (move % live in belief section)."""
        lines: list[str] = []
        cores_2 = self.format_meta.get("cores_2", [])
        cores_3 = self.format_meta.get("cores_3", [])
        opp_resolved = {self._pool_key(s) for s in opp_team}
        our_resolved = {self._pool_key(s) for s in our_team}

        for core in cores_2[:15]:
            mons = core.get("pokemon", [])
            if any(clean_species_name(m) in opp_resolved for m in mons):
                lines.append(
                    f"2-mon core: {', '.join(mons)} "
                    f"({core.get('usage_pct', '?')}% of teams)"
                )
        for core in cores_3[:10]:
            mons = core.get("pokemon", [])
            if sum(1 for m in mons if clean_species_name(m) in opp_resolved) >= 2:
                lines.append(
                    f"3-mon core: {', '.join(mons)} "
                    f"({core.get('usage_pct', '?')}% of teams)"
                )

        for species in opp_team:
            prior = self.get_species_prior(species)
            for teammate, pct in sorted(prior.teammates.items(), key=lambda x: -x[1])[:3]:
                if clean_species_name(teammate) in our_resolved:
                    lines.append(
                        f"{prior.species} often pairs with our {teammate} ({pct:.1f}%)"
                    )

        return "\n".join(lines)
