#!/usr/bin/env python3
"""Export Reg M-A legal species from Showdown Champions learnsets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import META_DIR
from src.doubles.planning.champions_legal import (
    LEGAL_SPECIES_CACHE,
    build_legal_species_list,
    pikalytics_pokedex_url,
)


def main() -> None:
    entries = build_legal_species_list()
    payload = {
        "format": "gen9championsvgc2026regma",
        "count": len(entries),
        "species": entries,
        "sample_urls": [
            pikalytics_pokedex_url(entries[0]["name"]),
            pikalytics_pokedex_url("Kingambit"),
            pikalytics_pokedex_url("Mimikyu"),
        ],
    }
    LEGAL_SPECIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LEGAL_SPECIES_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} legal species to {LEGAL_SPECIES_CACHE}")
    print(f"Example URL: {pikalytics_pokedex_url('Kingambit')}")


if __name__ == "__main__":
    main()
