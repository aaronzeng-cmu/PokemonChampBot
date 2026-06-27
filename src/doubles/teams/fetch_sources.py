"""Download Reg M-A Showdown teams from public paste sources."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from poke_env.teambuilder import Teambuilder

from src.doubles.teams.champions_format import convert_export_to_champions

POKEPAST_ID_RE = re.compile(r"pokepast\.es/([A-Za-z0-9]+)")
TEAM_ID_RE = re.compile(r"^PC(\d+)$", re.IGNORECASE)
USER_AGENT = "PokemonChampBot-TeamFetcher/1.0 (+local research)"

SMOGON_SAMPLE_THREAD = (
    "https://www.smogon.com/forums/threads/"
    "champions-vgc-regulation-m-a-sample-teams.3782777/"
)
SMOGON_METAGAME_THREAD = (
    "https://www.smogon.com/forums/threads/"
    "vgc-reg-m-a-metagame-discussion-thread.3780373/"
)

VGC_PASTES_SHEET_ID = "1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw"
VGC_PASTES_CHAMPIONS_GID = "791705272"
VGC_PASTES_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{VGC_PASTES_SHEET_ID}/export"
    f"?format=csv&gid={VGC_PASTES_CHAMPIONS_GID}"
)

REPLICA_STATUS_YES = frozenset({"yes", "y", "\u2714"})


@dataclass
class FetchedTeam:
    paste_id: str
    source: str
    export: str
    team_id: str = ""
    description: str = ""


def _http_get(url: str, *, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def pokepast_raw_url(paste_id: str) -> str:
    return f"https://pokepast.es/{paste_id}/raw"


def fetch_pokepast(paste_id: str) -> str:
    return normalize_showdown_export(_http_get(pokepast_raw_url(paste_id)))


def normalize_showdown_export(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_showdown_export(text: str) -> str:
    text = normalize_showdown_export(text)
    text = convert_export_to_champions(text)
    mons = Teambuilder.parse_showdown_team(text)
    if len(mons) != 6:
        raise ValueError(f"expected 6 Pokémon, found {len(mons)}")
    return text.strip() + "\n"


def discover_pokepast_ids_from_html(html: str) -> list[str]:
    return sorted(set(POKEPAST_ID_RE.findall(html)))


def discover_smogon_sample_thread() -> list[str]:
    html = _http_get(SMOGON_SAMPLE_THREAD)
    return discover_pokepast_ids_from_html(html)


def discover_smogon_metagame_thread(*, max_pages: int = 150) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        url = SMOGON_METAGAME_THREAD if page == 1 else f"{SMOGON_METAGAME_THREAD}page-{page}"
        try:
            html = _http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break
            raise
        found = discover_pokepast_ids_from_html(html)
        new = [pid for pid in found if pid not in seen]
        if not new and page > 3:
            break
        for pid in new:
            seen.add(pid)
            ids.append(pid)
        time.sleep(0.25)
    return ids


def _replica_status_is_yes(value: str) -> bool:
    return value.strip().lower() in REPLICA_STATUS_YES


def _team_id_sort_key(team_id: str) -> int:
    match = TEAM_ID_RE.match(team_id.strip())
    return int(match.group(1)) if match else -1


@dataclass
class VGCPastesRow:
    team_id: str
    paste_id: str
    description: str
    evs: str
    replica_status: str


def parse_vgcpastes_csv(csv_text: str) -> list[VGCPastesRow]:
    rows = list(csv.reader(io.StringIO(csv_text)))
    header_idx = next(
        (i for i, row in enumerate(rows) if row and row[0].strip() == "Team ID"),
        None,
    )
    if header_idx is None:
        raise ValueError("VGCPastes CSV: could not find header row (Team ID)")

    header = rows[header_idx]
    col = {name.strip(): idx for idx, name in enumerate(header) if name.strip()}

    required = ("Pokepaste", "EVs", "Replica Status")
    for name in required:
        if name not in col:
            raise ValueError(f"VGCPastes CSV: missing column {name!r}")

    paste_col = col["Pokepaste"]
    evs_col = col["EVs"]
    replica_col = col["Replica Status"]
    desc_col = col.get("Team Description", 1)

    entries: list[VGCPastesRow] = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip().upper().startswith("PC"):
            continue
        team_id = row[0].strip()
        if paste_col >= len(row):
            continue
        match = POKEPAST_ID_RE.search(row[paste_col])
        if not match:
            continue
        evs = row[evs_col].strip() if evs_col < len(row) else ""
        replica = row[replica_col].strip() if replica_col < len(row) else ""
        if evs.lower() != "yes" or not _replica_status_is_yes(replica):
            continue
        description = row[desc_col].strip() if desc_col < len(row) else ""
        entries.append(
            VGCPastesRow(
                team_id=team_id,
                paste_id=match.group(1),
                description=description,
                evs=evs,
                replica_status=replica,
            )
        )

    entries.sort(key=lambda e: _team_id_sort_key(e.team_id), reverse=True)
    return entries


def discover_vgcpastes_newest(
    *,
    limit: int = 50,
    csv_cache: Path | None = None,
) -> list[VGCPastesRow]:
    entries = parse_vgcpastes_csv(_load_vgcpastes_csv(csv_cache))
    seen: set[str] = set()
    unique: list[VGCPastesRow] = []
    for entry in entries:
        if entry.paste_id in seen:
            continue
        seen.add(entry.paste_id)
        unique.append(entry)
        if len(unique) >= limit:
            break
    return unique


def load_url_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def paste_id_from_url(url: str) -> str | None:
    match = POKEPAST_ID_RE.search(url)
    return match.group(1) if match else None


def fetch_from_paste_ids(
    paste_ids: list[str],
    *,
    source: str,
    delay_s: float = 0.35,
    meta: list[tuple[str, str]] | None = None,
) -> tuple[list[FetchedTeam], list[str]]:
    teams: list[FetchedTeam] = []
    errors: list[str] = []
    meta = meta or [( "", "")] * len(paste_ids)
    for paste_id, (team_id, description) in zip(paste_ids, meta):
        try:
            raw = fetch_pokepast(paste_id)
            export = validate_showdown_export(raw)
            teams.append(
                FetchedTeam(
                    paste_id=paste_id,
                    source=source,
                    export=export,
                    team_id=team_id,
                    description=description,
                )
            )
        except Exception as exc:  # noqa: BLE001 — collect per-URL failures
            label = team_id or paste_id
            errors.append(f"{label}: {exc}")
        time.sleep(delay_s)
    return teams, errors


def dedupe_by_packed(teams: list[FetchedTeam]) -> list[FetchedTeam]:
    seen: set[str] = set()
    unique: list[FetchedTeam] = []
    for team in teams:
        packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(team.export))
        if packed in seen:
            continue
        seen.add(packed)
        unique.append(team)
    return unique


def collect_paste_ids(
    *,
    extra_urls_file: Path | None,
    max_pages: int,
) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {
        "smogon_sample": discover_smogon_sample_thread(),
        "smogon_metagame": discover_smogon_metagame_thread(max_pages=max_pages),
    }
    manual: list[str] = []
    if extra_urls_file is not None:
        for url in load_url_list(extra_urls_file):
            pid = paste_id_from_url(url)
            if pid:
                manual.append(pid)
    ids["manual_urls"] = manual
    return ids


def _save_teams(out_dir: Path, all_teams: list[FetchedTeam]) -> list[dict]:
    for old in out_dir.glob("*.txt"):
        old.unlink()

    saved = []
    for idx, team in enumerate(all_teams, start=1):
        label = team.team_id or team.paste_id[:8]
        safe_label = re.sub(r"[^\w\-]+", "_", label)
        name = f"{idx:03d}_{team.source}_{safe_label}.txt"
        path = out_dir / name
        path.write_text(team.export, encoding="utf-8")
        saved.append(
            {
                "file": name,
                "paste_id": team.paste_id,
                "source": team.source,
                "team_id": team.team_id,
                "description": team.description,
            }
        )
    return saved


def _load_vgcpastes_csv(csv_cache: Path | None) -> str:
    if csv_cache is not None and csv_cache.is_file():
        return csv_cache.read_text(encoding="utf-8")
    csv_text = _http_get(VGC_PASTES_CSV_URL)
    if csv_cache is not None:
        csv_cache.parent.mkdir(parents=True, exist_ok=True)
        csv_cache.write_text(csv_text, encoding="utf-8")
    return csv_text


def fetch_opponent_teams_from_vgcpastes(
    *,
    out_dir: Path,
    target: int = 50,
    csv_cache: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_text = _load_vgcpastes_csv(csv_cache)
    all_matching = parse_vgcpastes_csv(csv_text)

    entries: list[VGCPastesRow] = []
    seen: set[str] = set()
    for entry in all_matching:
        if entry.paste_id in seen:
            continue
        seen.add(entry.paste_id)
        entries.append(entry)
        if len(entries) >= target:
            break

    paste_ids = [e.paste_id for e in entries]
    meta = [(e.team_id, e.description) for e in entries]
    all_teams, all_errors = fetch_from_paste_ids(
        paste_ids,
        source="vgcpastes",
        meta=meta,
    )
    all_teams = dedupe_by_packed(all_teams)

    if len(all_teams) < target:
        have = {t.paste_id for t in all_teams}
        for entry in all_matching:
            if entry.paste_id in have:
                continue
            batch, errors = fetch_from_paste_ids(
                [entry.paste_id],
                source="vgcpastes",
                meta=[(entry.team_id, entry.description)],
            )
            all_teams.extend(batch)
            all_errors.extend(errors)
            all_teams = dedupe_by_packed(all_teams)
            have = {t.paste_id for t in all_teams}
            if len(all_teams) >= target:
                break

    all_teams = all_teams[:target]
    saved = _save_teams(out_dir, all_teams)

    manifest = {
        "target": target,
        "saved": len(saved),
        "filter": {
            "evs": "Yes",
            "replica_status": "Yes (checkmark)",
            "selection": "newest by Team ID (PC####)",
        },
        "discovered_rows_matching_filter": len(all_matching),
        "errors": all_errors,
        "sources": {
            "vgcpastes_sheet": VGC_PASTES_CSV_URL,
            "sheet_page": (
                "https://docs.google.com/spreadsheets/d/"
                f"{VGC_PASTES_SHEET_ID}/edit#gid={VGC_PASTES_CHAMPIONS_GID}"
            ),
        },
        "teams": saved,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def fetch_opponent_teams(
    *,
    out_dir: Path,
    target: int = 50,
    max_pages: int = 150,
    extra_urls_file: Path | None = None,
    source: str = "vgcpastes",
    csv_cache: Path | None = None,
) -> dict:
    if source == "vgcpastes":
        return fetch_opponent_teams_from_vgcpastes(
            out_dir=out_dir,
            target=target,
            csv_cache=csv_cache,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    id_buckets = collect_paste_ids(
        extra_urls_file=extra_urls_file,
        max_pages=max_pages,
    )

    ordered_ids: list[tuple[str, str]] = []
    seen: set[str] = set()
    for src, paste_ids in id_buckets.items():
        for paste_id in paste_ids:
            if paste_id in seen:
                continue
            seen.add(paste_id)
            ordered_ids.append((src, paste_id))

    all_teams: list[FetchedTeam] = []
    all_errors: list[str] = []
    for src, paste_id in ordered_ids:
        batch, errors = fetch_from_paste_ids([paste_id], source=src)
        all_teams.extend(batch)
        all_errors.extend(errors)
        all_teams = dedupe_by_packed(all_teams)
        if len(all_teams) >= target:
            break

    all_teams = all_teams[:target]
    saved = _save_teams(out_dir, all_teams)

    manifest = {
        "target": target,
        "saved": len(saved),
        "discovered_ids": {k: len(v) for k, v in id_buckets.items()},
        "errors": all_errors,
        "sources": {
            "smogon_sample_thread": SMOGON_SAMPLE_THREAD,
            "smogon_metagame_thread": SMOGON_METAGAME_THREAD,
        },
        "teams": saved,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
