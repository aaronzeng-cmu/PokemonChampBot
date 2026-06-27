#!/usr/bin/env python3
"""Async scraper for gen9championsvgc2026regma Showdown replays."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp
from tqdm import tqdm

from config.settings import BATTLE_FORMAT, RAW_LOGS_DIR, SINGLES_BATTLE_FORMAT, SINGLES_RAW_LOGS_DIR
from src.doubles.data.replay_parser import MIN_TURN, parse_replay_log

SEARCH_BASE = "https://replay.pokemonshowdown.com/search.json"
LOG_URL = "https://replay.pokemonshowdown.com/{replay_id}.log"
MIN_RATING = 1350

FORMAT_TAGS = {
    "doubles": BATTLE_FORMAT,
    "singles": SINGLES_BATTLE_FORMAT,
}


async def fetch_json(session: aiohttp.ClientSession, url: str):
    for attempt in range(5):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt)
    return []


async def fetch_text(session: aiohttp.ClientSession, url: str) -> str:
    for attempt in range(5):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 404:
                    return ""
                resp.raise_for_status()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt)
    return ""


def _parse_ratings_and_turns(log_text: str) -> tuple[int, int, int]:
    r1 = r2 = 0
    max_turn = 0
    for line in log_text.splitlines():
        if line.startswith("|player|"):
            parts = line.split("|")
            if len(parts) >= 6:
                try:
                    rating = int(parts[5])
                except ValueError:
                    rating = 0
                if parts[2] == "p1":
                    r1 = rating
                elif parts[2] == "p2":
                    r2 = rating
        elif line.startswith("|turn|"):
            try:
                max_turn = max(max_turn, int(line.split("|")[2]))
            except (IndexError, ValueError):
                pass
    return r1, r2, max_turn


def _search_url(*, before: int | None = None, battle_format: str = BATTLE_FORMAT) -> str:
    url = f"{SEARCH_BASE}?format={battle_format}"
    if before is not None:
        url += f"&before={before}"
    return url


def _extract_rows(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("replays", data.get("data", []))
    return []


def _row_id(row) -> str | None:
    if isinstance(row, str):
        return row
    return row.get("id") or row.get("replayid")


def _row_uploadtime(row) -> int | None:
    if isinstance(row, dict):
        ut = row.get("uploadtime")
        if ut is not None:
            return int(ut)
    return None


def _uploadtime_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _saved_uploadtime_bounds(manifest: dict) -> tuple[int | None, int | None]:
    """Min/max uploadtime among saved manifest entries (when recorded)."""
    saved_times = [
        int(e["uploadtime"])
        for e in manifest.get("entries", [])
        if e.get("saved") and e.get("uploadtime") is not None
    ]
    if not saved_times:
        return None, None
    return min(saved_times), max(saved_times)


def _oldest_battle_report(manifest: dict) -> dict:
    """Human-readable oldest/newest saved battle bounds for logging."""
    oldest_ut, newest_ut = _saved_uploadtime_bounds(manifest)
    stats = manifest.get("stats", {})
    before_cursor = stats.get("before_cursor")
    report = {
        "saved_count": sum(1 for e in manifest.get("entries", []) if e.get("saved")),
        "oldest_saved_uploadtime": oldest_ut,
        "oldest_saved_utc": _uploadtime_iso(oldest_ut),
        "newest_saved_uploadtime": newest_ut,
        "newest_saved_utc": _uploadtime_iso(newest_ut),
        "search_before_cursor": before_cursor,
        "search_before_cursor_utc": _uploadtime_iso(before_cursor),
    }
    if oldest_ut is None and before_cursor is not None:
        report["oldest_note"] = (
            "No per-battle uploadtime in manifest yet; "
            "search_before_cursor is the furthest-back search page scanned."
        )
    return report


async def _process_ids(
    session: aiohttp.ClientSession,
    *,
    ids: list[str],
    uploadtimes: dict[str, int | None],
    sem: asyncio.Semaphore,
    out_dir: Path,
    manifest: dict,
    saved_ids: set[str],
    processed_ids: set[str],
    pbar: tqdm,
    battle_format: str,
    validate_parse: bool,
) -> int:
    """Download and validate replay IDs. Returns count of newly saved logs."""
    new_saved = 0

    async def handle(rid: str) -> None:
        nonlocal new_saved
        async with sem:
            processed_ids.add(rid)
            dest = out_dir / f"{rid}.log"
            if dest.is_file():
                saved_ids.add(rid)
                pbar.update(1)
                manifest["entries"].append(
                    {
                        "id": rid,
                        "saved": True,
                        "reason": "already_on_disk",
                        "uploadtime": uploadtimes.get(rid),
                    }
                )
                return
            log_text = await fetch_text(session, LOG_URL.format(replay_id=rid))
            if not log_text:
                manifest["entries"].append(
                    {
                        "id": rid,
                        "saved": False,
                        "reason": "download_failed",
                        "uploadtime": uploadtimes.get(rid),
                    }
                )
                return
            r1, r2, max_turn = _parse_ratings_and_turns(log_text)
            reason = "ok"
            ok = True
            if r1 < MIN_RATING or r2 < MIN_RATING:
                ok = False
                reason = f"rating_below_{MIN_RATING}"
            elif max_turn < MIN_TURN:
                ok = False
                reason = f"ended_before_turn_{MIN_TURN}"
            elif validate_parse and not parse_replay_log(log_text, rid, skip_rating=True):
                ok = False
                reason = "parse_failed"
            if ok:
                dest.write_text(log_text, encoding="utf-8")
                saved_ids.add(rid)
                pbar.update(1)
                new_saved += 1
            manifest["entries"].append(
                {
                    "id": rid,
                    "saved": ok,
                    "reason": reason,
                    "p1_rating": r1,
                    "p2_rating": r2,
                    "max_turn": max_turn,
                    "uploadtime": uploadtimes.get(rid),
                }
            )

    if ids:
        await asyncio.gather(*(handle(rid) for rid in ids))
    return new_saved


async def scrape(
    *,
    target_ids: int,
    out_dir: Path,
    concurrency: int,
    prefer_new: bool = False,
    battle_format: str = BATTLE_FORMAT,
    validate_parse: bool = True,
) -> dict:
    """
    Scrape replays using keyset pagination (`before=uploadtime`).

    The legacy `page=N` API is hard-capped at page 100 (~5000 IDs). Keyset
    pagination can reach much older replays for the same format.

    When ``prefer_new`` is True, scan from the newest search pages first to
    pick up battles uploaded since the last run, then continue paginating
    backward from the saved ``before_cursor``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: dict = {"entries": [], "stats": {}}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    processed_ids = {e["id"] for e in manifest.get("entries", [])}
    saved_ids = {e["id"] for e in manifest.get("entries", []) if e.get("saved")}
    before_cursor = manifest.get("stats", {}).get("before_cursor")
    batches_scanned = manifest.get("stats", {}).get("batches_scanned", 0)
    oldest_ut, newest_ut = _saved_uploadtime_bounds(manifest)

    sem = asyncio.Semaphore(concurrency)

    def _write_stats(*, phase: str | None = None) -> None:
        nonlocal oldest_ut, newest_ut
        oldest_ut, newest_ut = _saved_uploadtime_bounds(manifest)
        stats = {
            "saved": len(saved_ids),
            "target": target_ids,
            "batches_scanned": batches_scanned,
            "before_cursor": before_cursor,
            "discovered": len(processed_ids),
            "oldest_saved_uploadtime": oldest_ut,
            "oldest_saved_utc": _uploadtime_iso(oldest_ut),
            "newest_saved_uploadtime": newest_ut,
            "newest_saved_utc": _uploadtime_iso(newest_ut),
        }
        if phase is not None:
            stats["phase"] = phase
        manifest["stats"] = stats
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=target_ids, desc="saved")
        pbar.update(len(saved_ids))

        if prefer_new and len(saved_ids) < target_ids:
            new_before = None
            new_batches = 0
            while len(saved_ids) < target_ids:
                rows = _extract_rows(await fetch_json(session, _search_url(before=new_before, battle_format=battle_format)))
                if not rows:
                    break

                ids: list[str] = []
                uploadtimes: dict[str, int | None] = {}
                all_known = True
                for row in rows:
                    rid = _row_id(row)
                    if not rid:
                        continue
                    uploadtimes[rid] = _row_uploadtime(row)
                    if rid not in processed_ids:
                        ids.append(rid)
                        all_known = False

                await _process_ids(
                    session,
                    ids=ids,
                    uploadtimes=uploadtimes,
                    sem=sem,
                    out_dir=out_dir,
                    manifest=manifest,
                    saved_ids=saved_ids,
                    processed_ids=processed_ids,
                    pbar=pbar,
                    battle_format=battle_format,
                    validate_parse=validate_parse,
                )

                new_batches += 1
                last_ut = _row_uploadtime(rows[-1])
                _write_stats(phase="prefer_new")

                if all_known or len(rows) < 51 or last_ut is None:
                    break
                if new_before is not None and last_ut >= new_before:
                    break
                new_before = last_ut

            batches_scanned += new_batches

        while len(saved_ids) < target_ids:
            url = _search_url(before=before_cursor, battle_format=battle_format)
            rows = _extract_rows(await fetch_json(session, url))
            if not rows:
                break

            ids: list[str] = []
            uploadtimes: dict[str, int | None] = {}
            for row in rows:
                rid = _row_id(row)
                if rid and rid not in processed_ids:
                    ids.append(rid)
                    uploadtimes[rid] = _row_uploadtime(row)

            await _process_ids(
                session,
                ids=ids,
                uploadtimes=uploadtimes,
                sem=sem,
                out_dir=out_dir,
                manifest=manifest,
                saved_ids=saved_ids,
                processed_ids=processed_ids,
                pbar=pbar,
                battle_format=battle_format,
                validate_parse=validate_parse,
            )

            batches_scanned += 1
            last_ut = _row_uploadtime(rows[-1])
            _write_stats(phase="backfill_old")

            if len(rows) < 51 or last_ut is None:
                break
            if before_cursor is not None and last_ut >= before_cursor:
                # uploadtime should decrease; equal means no further pages
                break
            before_cursor = last_ut

        pbar.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Showdown replay logs")
    parser.add_argument("--target", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--prefer-new",
        action="store_true",
        help="Scan newest search pages first before continuing backward pagination",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_TAGS),
        default="doubles",
        help="Ladder format tag for replay search (doubles=VGC, singles=BSS)",
    )
    parser.add_argument(
        "--battle-format",
        default=None,
        help="Explicit Showdown format code override (e.g. gen9championsvgc2026regmb)",
    )
    args = parser.parse_args()

    battle_format = args.battle_format or FORMAT_TAGS[args.format]
    validate_parse = args.format == "doubles"
    out_dir = args.out or (SINGLES_RAW_LOGS_DIR if args.format == "singles" else RAW_LOGS_DIR)

    manifest_path = out_dir / "manifest.json"
    before_manifest: dict = {"entries": [], "stats": {}}
    if manifest_path.is_file():
        before_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before_report = _oldest_battle_report(before_manifest)
    print("=== Before fetch ===")
    print(json.dumps(before_report, indent=2))

    manifest = asyncio.run(
        scrape(
            target_ids=args.target,
            out_dir=out_dir,
            concurrency=args.concurrency,
            prefer_new=args.prefer_new,
            battle_format=battle_format,
            validate_parse=validate_parse,
        )
    )

    after_report = _oldest_battle_report(manifest)
    print("=== After fetch ===")
    print(json.dumps(after_report, indent=2))
    print("=== Stats ===")
    print(json.dumps(manifest.get("stats", {}), indent=2))


if __name__ == "__main__":
    main()
