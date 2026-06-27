#!/usr/bin/env python3
"""Download BSS opponent teams into teams/singles_opponents/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ROOT_DIR, SINGLES_OPPONENT_POOL_DIR
from src.singles.teams.fetch_sources import fetch_singles_opponent_teams

DEFAULT_URLS_FILE = ROOT_DIR / "teams" / "sources" / "tox_bss_top20_urls.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch BSS singles teams from Tox Season 1 Top-20 pokepastes.",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=DEFAULT_URLS_FILE,
        help=f"Ranked pokepast URL list (default: {DEFAULT_URLS_FILE.name})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SINGLES_OPPONENT_POOL_DIR,
        help=f"Output directory (default: {SINGLES_OPPONENT_POOL_DIR})",
    )
    args = parser.parse_args()

    manifest = fetch_singles_opponent_teams(
        urls_file=args.urls_file,
        out_dir=args.out_dir,
    )

    print(f"Saved {manifest['saved']}/{manifest['requested']} teams to {args.out_dir.resolve()}")
    if manifest["errors"]:
        print(f"Skipped {len(manifest['errors'])} pastes:")
        for err in manifest["errors"]:
            print(f"  - {err}")
    print(f"Manifest: {(args.out_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
