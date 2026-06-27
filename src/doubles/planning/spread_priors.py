"""Aggregate EV spread priors from opponent team pastes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from poke_env.teambuilder import Teambuilder

from config.settings import OPPONENT_POOL_DIR
from src.core.planning.species_normalize import clean_species_name

_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
_SPECIES_LINE_RE = re.compile(r"^([^@\n]+?)(?:\s+@\s+.+)?$", re.MULTILINE)


def _species_from_block(block: str) -> str:
    first = block.strip().splitlines()[0].strip()
    return first.split("@")[0].strip()


def _evs_to_champions_string(evs: list[int]) -> str:
    parts = [str(evs[i]) for i in range(6)]
    return "/".join(parts)


def spread_key(nature: str, evs: list[int]) -> str:
    return f"{nature}|{_evs_to_champions_string(evs)}"


def aggregate_spread_priors(pool_dir: Path | None = None) -> dict[str, dict[str, float]]:
    """Return species -> {spread_key: probability} from opponent pool pastes."""
    pool_dir = pool_dir or OPPONENT_POOL_DIR
    counts: dict[str, Counter[str]] = defaultdict(Counter)

    for path in sorted(pool_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
        for block in blocks:
            try:
                mons = Teambuilder.parse_showdown_team(block + "\n\n")
            except Exception:
                continue
            if not mons:
                continue
            mon = mons[0]
            species = clean_species_name(
                mon.species or mon.nickname or _species_from_block(block)
            )
            if not species:
                continue
            evs = mon.evs or [0, 0, 0, 0, 0, 0]
            nature = mon.nature or "Serious"
            key = spread_key(nature, evs)
            counts[species][key] += 1

    priors: dict[str, dict[str, float]] = {}
    for species, counter in counts.items():
        total = sum(counter.values())
        if total <= 0:
            continue
        priors[species] = {k: v / total for k, v in counter.items()}
    return priors


def _default_spread_key() -> str:
    return spread_key("Serious", [0, 0, 0, 0, 0, 0])
