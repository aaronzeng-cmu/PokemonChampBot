"""Download BSS / Champions Singles teams from public paste sources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.doubles.teams.fetch_sources import (
    FetchedTeam,
    _save_teams,
    dedupe_by_packed,
    fetch_from_paste_ids,
    paste_id_from_url,
)

RANK_LINE_RE = re.compile(r"^(\d+)\|([^|]+)\|(.+)$")


@dataclass
class SinglesUrlEntry:
    rank: int
    description: str
    paste_id: str


def parse_ranked_url_list(path: Path) -> list[SinglesUrlEntry]:
    """Parse lines like ``01|Lucario & Floette-Mega|https://pokepast.es/...``."""
    entries: list[SinglesUrlEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = RANK_LINE_RE.match(line)
        if not match:
            pid = paste_id_from_url(line)
            if pid:
                entries.append(SinglesUrlEntry(rank=len(entries) + 1, description="", paste_id=pid))
            continue
        rank = int(match.group(1))
        description = match.group(2).strip()
        pid = paste_id_from_url(match.group(3).strip())
        if not pid:
            continue
        entries.append(SinglesUrlEntry(rank=rank, description=description, paste_id=pid))
    return entries


def fetch_singles_opponent_teams(
    *,
    urls_file: Path,
    out_dir: Path,
    source: str = "tox_bss_top20",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_ranked_url_list(urls_file)
    paste_ids = [e.paste_id for e in entries]
    meta = [(f"rank{e.rank:02d}", e.description) for e in entries]

    all_teams, all_errors = fetch_from_paste_ids(
        paste_ids,
        source=source,
        meta=meta,
    )
    all_teams = dedupe_by_packed(all_teams)
    saved = _save_teams(out_dir, all_teams)

    manifest = {
        "source_page": (
            "https://tox.hatenablog.com/entry/2026/05/17/"
            "Battle_Stadium_Singles_blog_%E2%80%94_Season_1_Top-20_Team_Listing_"
            "%28Regulation_M-A%3B_Season_M-1%29"
        ),
        "urls_file": str(urls_file),
        "requested": len(entries),
        "saved": len(saved),
        "errors": all_errors,
        "teams": saved,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
