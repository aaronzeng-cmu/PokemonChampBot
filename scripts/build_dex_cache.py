#!/usr/bin/env python3
"""Build Champions dex cache from local pokemon-showdown data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DEX_CACHE_PATH, SHOWDOWN_DATA_DIR
from src.core.planning.dex_cache import build_dex_cache, save_dex_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teams/meta/dex_reg_ma.json")
    parser.add_argument(
        "--showdown-data",
        type=Path,
        default=SHOWDOWN_DATA_DIR,
        help=f"Showdown data directory (default: {SHOWDOWN_DATA_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEX_CACHE_PATH,
        help=f"Output JSON path (default: {DEX_CACHE_PATH})",
    )
    args = parser.parse_args()

    if not args.showdown_data.is_dir():
        raise FileNotFoundError(
            f"Showdown data not found at {args.showdown_data}. "
            "Clone pokemon-showdown or set SHOWDOWN_DATA_DIR."
        )

    cache = build_dex_cache(args.showdown_data)
    save_dex_cache(cache, args.out)
    n_moves = len(cache.get("moves", {}))
    n_abilities = len(cache.get("abilities", {}))
    print(f"Wrote {n_moves} moves and {n_abilities} abilities to {args.out}")


if __name__ == "__main__":
    main()
