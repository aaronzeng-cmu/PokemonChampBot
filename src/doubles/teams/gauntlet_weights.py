"""Pikalytics-based weights for gauntlet opponent teams."""

from __future__ import annotations

import re

from poke_env.teambuilder import Teambuilder

from src.doubles.planning.meta_database import MetaDatabase
from src.core.planning.species_normalize import clean_species_name

_SPECIES_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9'\-\. ]*?)(?:\s*@|\s*\()")


def parse_team_species_names(export: str) -> list[str]:
    """Extract species names from a Showdown paste."""
    try:
        mons = Teambuilder.parse_showdown_team(export)
        names = []
        for mon in mons:
            raw = getattr(mon, "species", None) or getattr(mon, "nickname", None) or ""
            names.append(clean_species_name(str(raw)))
        if len(names) == 6:
            return names
    except Exception:
        pass

    names: list[str] = []
    for line in export.splitlines():
        line = line.strip()
        if not line or line.startswith(("===", "-", "Ability:", "EVs:", "IVs:", "Level:")):
            continue
        m = _SPECIES_LINE.match(line)
        if m:
            name = clean_species_name(m.group(1))
            if name and name not in names:
                names.append(name)
        if len(names) >= 6:
            break
    return names


def team_meta_score(export: str, meta_db: MetaDatabase | None = None) -> float:
    """
    Proxy for team popularity: mean ladder usage % of the six species.
    Teams with meta staples (e.g. Flutter Mane) score higher.
    """
    db = meta_db or MetaDatabase(live_fetch=False)
    species = parse_team_species_names(export)
    if not species:
        return 1.0
    scores: list[float] = []
    for sp in species:
        prior = db.get_species_prior(sp)
        scores.append(prior.usage_pct if prior.usage_pct and prior.usage_pct > 0 else 1.0)
    return float(sum(scores) / len(scores))


def normalize_team_weights(scores: list[float]) -> list[float]:
    total = sum(scores)
    if total <= 0:
        n = len(scores)
        return [1.0 / n] * n if n else []
    return [s / total for s in scores]
