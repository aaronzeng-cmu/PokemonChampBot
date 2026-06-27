"""Reg M-A legal species list from Pokémon Showdown Champions learnsets."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from poke_env.data import GenData

from config.settings import META_DIR, SHOWDOWN_DATA_DIR

LEGAL_SPECIES_CACHE = META_DIR / "champions_legal_species.json"
_LEARNSETS_PATH = SHOWDOWN_DATA_DIR / "mods" / "champions" / "learnsets.ts"
_LEARNSET_ID_RE = re.compile(r"^\t([a-z0-9]+):\s*\{", re.MULTILINE)


def _id_to_display_name(species_id: str) -> str | None:
    gd = GenData.from_gen(9)
    entry = gd.pokedex.get(species_id)
    if entry and entry.get("name"):
        return str(entry["name"])
    return None


def parse_learnset_species_ids(path: Path | None = None) -> list[str]:
    path = path or _LEARNSETS_PATH
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _LEARNSET_ID_RE.findall(text)


def build_legal_species_list(*, learnsets_path: Path | None = None) -> list[dict[str, str]]:
    """Return sorted legal entries: {id, name, pikalytics_slug}."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for species_id in parse_learnset_species_ids(learnsets_path):
        name = _id_to_display_name(species_id)
        if not name or name in seen:
            continue
        seen.add(name)
        entries.append(
            {
                "id": species_id,
                "name": name,
                "pikalytics_slug": name,  # Pikalytics uses Showdown display names in URLs
            }
        )
    entries.sort(key=lambda row: row["name"])
    return entries


@lru_cache(maxsize=1)
def load_legal_species_names(*, refresh: bool = False) -> list[str]:
    """Load cached legal species display names (builds cache if missing)."""
    if not refresh and LEGAL_SPECIES_CACHE.is_file():
        data = json.loads(LEGAL_SPECIES_CACHE.read_text(encoding="utf-8"))
        return [row["name"] for row in data.get("species", [])]

    entries = build_legal_species_list()
    payload = {
        "format": "gen9championsvgc2026regmb",
        "source": str(_LEARNSETS_PATH.resolve()),
        "count": len(entries),
        "species": entries,
    }
    LEGAL_SPECIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LEGAL_SPECIES_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return [row["name"] for row in entries]


def pikalytics_pokedex_url(
    species: str,
    *,
    format_code: str = "gen9championsvgc2026regmb",
    lang: str = "en",
) -> str:
    """Public Pikalytics pokedex page (HTML). AI markdown is linked from this page."""
    from src.doubles.teams.pikalytics_meta import pokedex_url

    return pokedex_url(species, format_code, lang=lang)
