"""Download Pokémon Champions menu sprites from Bulbagarden Archives."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "assets" / "pokemon_icons"
_API_URL = "https://archives.bulbagarden.net/w/api.php"
_USER_AGENT = "PokemonChampBot/1.0 (offline asset downloader; github.com/AaronChampBot)"
_DEFAULT_DELAY_S = 0.25

_CATEGORY_NORMAL = "Category:Champions_menu_sprites"
_CATEGORY_SHINY = "Category:Champions_Shiny_menu_sprites"

_FILE_TITLE_RE = re.compile(
    r"^File:Menu CP (\d{4})(?:-(.+?))?(?: shiny)?\.png$",
    re.IGNORECASE,
)


def _api_get(params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(
        f"{_API_URL}?{query}",
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _normalize_suffix(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[\s_\-]+", "", value.lower())


def _form_suffix_from_showdown_name(name: str) -> str:
    if "-" not in name:
        return ""
    return name.split("-", 1)[1]


def _build_species_lookup() -> dict[tuple[int, str], str]:
    """Map (national dex number, normalized form suffix) -> bot species id."""
    try:
        from poke_env.data import GenData

        pokedex = GenData.from_gen(9).pokedex
    except Exception as exc:
        raise RuntimeError(
            "Could not load poke-env Gen 9 pokedex for species mapping."
        ) from exc

    lookup: dict[tuple[int, str], str] = {}
    for species_id, entry in pokedex.items():
        num = entry.get("num")
        if not isinstance(num, int):
            continue
        name = str(entry.get("name", ""))
        suffix = _normalize_suffix(_form_suffix_from_showdown_name(name))
        key = (num, suffix)
        if key in lookup:
            logger.debug("Duplicate lookup key %s: %s vs %s", key, lookup[key], species_id)
        lookup[key] = str(species_id)
    return lookup


def _parse_file_title(title: str) -> tuple[int, str, bool] | None:
    match = _FILE_TITLE_RE.match(title)
    if not match:
        return None
    num = int(match.group(1))
    raw_suffix = match.group(2) or ""
    is_shiny = " shiny" in title.lower()
    suffix = _normalize_suffix(raw_suffix)
    return num, suffix, is_shiny


def list_category_files(category: str) -> list[str]:
    """Return all file titles in a Bulbagarden category (handles pagination)."""
    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": 500,
    }
    while True:
        payload = _api_get(params)
        members = payload.get("query", {}).get("categorymembers", [])
        for member in members:
            title = str(member.get("title", ""))
            if title.startswith("File:Menu CP "):
                titles.append(title)
        continue_token = payload.get("continue", {}).get("cmcontinue")
        if not continue_token:
            break
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
            "cmcontinue": continue_token,
        }
    return titles


def resolve_species_id(title: str, lookup: dict[tuple[int, str], str]) -> str | None:
    parsed = _parse_file_title(title)
    if parsed is None:
        return None
    num, suffix, _is_shiny = parsed
    return lookup.get((num, suffix))


def fetch_image_urls(titles: list[str]) -> dict[str, str]:
    """Batch-resolve direct PNG URLs for file titles."""
    urls: dict[str, str] = {}
    batch_size = 50
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        payload = _api_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url",
            }
        )
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            title = str(page.get("title", ""))
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue
            url = imageinfo[0].get("url")
            if url:
                urls[title] = str(url)
        time.sleep(0.1)
    return urls


def discover_downloads(
    *,
    include_shiny: bool = False,
) -> list[tuple[str, str, bool]]:
    """Return [(file_title, species_id, is_shiny), ...] with normal sprites first."""
    lookup = _build_species_lookup()
    planned: list[tuple[str, str, bool]] = []
    seen: set[str] = set()

    unmapped: list[str] = []
    for category, is_shiny in ((_CATEGORY_NORMAL, False), (_CATEGORY_SHINY, True)):
        if is_shiny and not include_shiny:
            continue
        for title in list_category_files(category):
            species_id = resolve_species_id(title, lookup)
            if not species_id:
                unmapped.append(title)
                continue
            if species_id in seen:
                continue
            seen.add(species_id)
            planned.append((title, species_id, is_shiny))

    if unmapped:
        logger.info(
            "Skipped %d Bulbagarden files with no Gen 9 vocabulary match (e.g. Vivillon patterns).",
            len(unmapped),
        )
        for title in unmapped[:5]:
            logger.debug("Unmapped: %s", title)

    return planned


def download_icon(
    title: str,
    species_id: str,
    dest_dir: Path,
    *,
    image_url: str | None = None,
    overwrite: bool = False,
) -> bool:
    """Download one mapped icon. Returns True when saved."""
    dest = dest_dir / f"{species_id}.png"
    if dest.exists() and not overwrite:
        return False

    url = image_url
    if not url:
        resolved = fetch_image_urls([title])
        url = resolved.get(title)
    if not url:
        logger.warning("Skipping %s (%s): no image URL", species_id, title)
        return False

    try:
        data = _fetch_bytes(url)
    except Exception as exc:
        logger.warning("Skipping %s (%s): download failed (%s)", species_id, title, exc)
        return False

    if len(data) < 32:
        logger.warning("Skipping %s (%s): payload too small", species_id, title)
        return False

    dest.write_bytes(data)
    return True


def download_icons(
    dest_dir: Path | str | None = None,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    delay_s: float = _DEFAULT_DELAY_S,
    include_shiny: bool = False,
    clean: bool = False,
) -> dict[str, int | str]:
    """Download Champions menu sprites mapped to bot vocabulary ids."""
    out = Path(dest_dir or _DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)

    planned = discover_downloads(include_shiny=include_shiny)
    if limit is not None:
        planned = planned[:limit]

    if clean:
        keep = {f"{species_id}.png" for _, species_id, _ in planned}
        for path in out.glob("*.png"):
            if path.name not in keep:
                path.unlink()
                logger.info("Removed stale icon %s", path.name)

    titles = [title for title, _, _ in planned]
    url_map = fetch_image_urls(titles)

    downloaded = 0
    skipped = 0
    failed = 0
    for index, (title, species_id, is_shiny) in enumerate(planned, start=1):
        label = f"{species_id}{' (shiny)' if is_shiny else ''}"
        try:
            if download_icon(
                title,
                species_id,
                out,
                image_url=url_map.get(title),
                overwrite=overwrite,
            ):
                downloaded += 1
                logger.info("Downloaded %s from %s", label, title)
            elif (out / f"{species_id}.png").exists():
                skipped += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            logger.warning("Failed %s: %s", label, exc)

        if delay_s > 0 and index < len(planned):
            time.sleep(delay_s)

    return {
        "source": "bulbagarden-champions-menu",
        "total": len(planned),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "dest": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: src/cv_bridge/assets/pokemon_icons/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete PNGs in dest that are not in the planned vocabulary download set.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N mapped icons.")
    parser.add_argument(
        "--include-shiny",
        action="store_true",
        help="Also scan shiny category (normal sprites are always preferred).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_DEFAULT_DELAY_S,
        help="Delay between PNG downloads in seconds (default: 0.25).",
    )
    args = parser.parse_args(argv)

    stats = download_icons(
        args.dest,
        overwrite=args.overwrite,
        limit=args.limit,
        delay_s=args.delay,
        include_shiny=args.include_shiny,
        clean=args.clean,
    )
    print(
        f"Source={stats['source']} total={stats['total']} "
        f"downloaded={stats['downloaded']} skipped={stats['skipped']} failed={stats['failed']}"
    )
    print(f"Saved to {stats['dest']}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
