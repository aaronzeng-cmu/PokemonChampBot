"""Fetch and parse Reg M-A meta stats from Pikalytics AI markdown endpoints."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PIKALYTICS_BASE = "https://www.pikalytics.com"
DEFAULT_FORMAT = "gen9championsvgc2026regmb"
DEFAULT_BATTLE_RATING = "1760"
USER_AGENT = "PokemonChampBot-PikalyticsFetcher/1.0 (+local research)"

SECTION_HEADERS = {
    "moves": "Common Moves",
    "abilities": "Common Abilities",
    "items": "Common Items",
    "teammates": "Common Teammates",
}

USAGE_LINE_RE = re.compile(
    r"^\s*-\s+\*\*(?P<name>[^*]+)\*\*:\s*(?P<pct>[\d.]+)%\s*$",
    re.MULTILINE,
)
QUICK_INFO_RE = re.compile(
    r"^\|\s*\*\*(?P<key>[^|*]+)\*\*\s*\|\s*(?P<value>[^|]+)\|\s*$",
    re.MULTILINE,
)
SPECIES_LINK_RE = re.compile(
    rf"/ai/pokedex/{DEFAULT_FORMAT}/([A-Za-z0-9\-]+)"
)
EV_SPREAD_RE = re.compile(
    r"(?:features a\s+)?(?:\*\*)?(?P<nature>[A-Za-z]+)(?:\*\*)?\s+nature with an EV spread of `(?P<evs>[\d/]+)`\. "
    r"This configuration accounts for (?P<pct>[\d.]+)%",
)
TEAM_SET_RE = re.compile(
    r"### Team \d+ by (?P<author>.+?)\n"
    r"\*Record: (?P<record>[\d\-]+)\*\n"
    r"\*Event: (?P<event>[^*]+)\*\n\n"
    r"\*\*Pokemon\*\*: (?P<pokemon>.+?)\n\n"
    r"\*\*(?P<species>[^*]+) Set\*\*:\n"
    r"- \*\*Ability\*\*: (?P<ability>.+?)\n"
    r"- \*\*Item\*\*: (?P<item>.+?)\n"
    r"- \*\*Moves\*\*: (?P<moves>.+?)(?:\n\n|\Z)",
    re.DOTALL,
)
CORE_ROW_RE = re.compile(
    r"^\|\s*(?P<rank>\d+)\s*\|\s*(?P<core>.+?)\s*\|\s*(?P<teams>\d+)\s*\|\s*(?P<pct>[\d.]+)%\s*\|\s*$",
    re.MULTILINE,
)


@dataclass
class UsageEntry:
    name: str
    usage_pct: float


@dataclass
class FeaturedSet:
    author: str
    record: str
    event: str
    team_pokemon: list[str]
    ability: str
    item: str
    moves: list[str]


@dataclass
class PokemonMeta:
    species: str
    format_code: str = DEFAULT_FORMAT
    usage_pct: float | None = None
    win_rate_pct: float | None = None
    record: str | None = None
    data_date: str | None = None
    moves: list[UsageEntry] = field(default_factory=list)
    abilities: list[UsageEntry] = field(default_factory=list)
    items: list[UsageEntry] = field(default_factory=list)
    teammates: list[UsageEntry] = field(default_factory=list)
    top_nature: str | None = None
    top_evs: str | None = None
    top_spread_pct: float | None = None
    featured_sets: list[FeaturedSet] = field(default_factory=list)
    source_url: str = ""


@dataclass
class FormatMeta:
    format_code: str
    data_date: str | None
    species: list[str]
    top_usage: list[dict[str, Any]]
    cores_2: list[dict[str, Any]]
    cores_3: list[dict[str, Any]]
    source_url: str


def _http_get(url: str, *, timeout: float = 60.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/markdown,text/plain,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _http_get_json(url: str, *, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def battle_format_key(
    format_code: str = DEFAULT_FORMAT,
    *,
    rating: str = DEFAULT_BATTLE_RATING,
) -> str:
    """Showdown ladder key used by Pikalytics Battle Usage (e.g. format-1760)."""
    return f"{format_code}-{rating}"


def battle_usage_list_url(
    format_code: str = DEFAULT_FORMAT,
    *,
    data_date: str,
    rating: str = DEFAULT_BATTLE_RATING,
) -> str:
    """JSON list behind the Battle Usage tab on the format pokedex page."""
    fmt = battle_format_key(format_code, rating=rating)
    return f"{PIKALYTICS_BASE}/api/l/{data_date}/{fmt}"


def resolve_data_date(
    format_code: str = DEFAULT_FORMAT,
    *,
    markdown: str | None = None,
) -> str | None:
    """Read Pikalytics data month (YYYY-MM) from format markdown or live fetch."""
    text = markdown
    if text is None:
        try:
            text = _http_get(format_index_url(format_code))
        except Exception:
            return None
    match = re.search(r"\*\*Data Date\*\*:\s*(\d{4}-\d{2})", text)
    return match.group(1) if match else None


def fetch_battle_usage_list(
    format_code: str = DEFAULT_FORMAT,
    *,
    data_date: str | None = None,
    rating: str = DEFAULT_BATTLE_RATING,
    delay_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Fetch the full Battle Usage species list (same data as the website tab).

    One request returns every Pokemon with >0% ladder usage for the format,
    including inline move/item/ability percentages.
    """
    date = data_date or resolve_data_date(format_code)
    if not date:
        raise ValueError(f"Could not resolve Pikalytics data date for {format_code}")
    url = battle_usage_list_url(format_code, data_date=date, rating=rating)
    data = _http_get_json(url)
    if delay_s > 0:
        time.sleep(delay_s)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected battle usage payload from {url}")
    return data


def discover_species_from_battle_usage(
    format_code: str = DEFAULT_FORMAT,
    *,
    data_date: str | None = None,
    rating: str = DEFAULT_BATTLE_RATING,
) -> list[str]:
    """Species names from Pikalytics Battle Usage (typically ~250+ for Reg M-A)."""
    rows = fetch_battle_usage_list(
        format_code,
        data_date=data_date,
        rating=rating,
        delay_s=0.0,
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        name = str(row.get("name") or row.get("name_trans") or "").strip()
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def ai_url(path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"{PIKALYTICS_BASE}{path}"


def format_index_url(format_code: str = DEFAULT_FORMAT) -> str:
    return ai_url(f"/ai/pokedex/{format_code}")


def pikalytics_slug(species: str) -> str:
    """URL path segment for Pikalytics (spaces -> hyphens, strip dots)."""
    slug = species.strip().replace(" ", "-")
    slug = slug.replace(".", "")
    return slug


def pokedex_url(
    species: str,
    format_code: str = DEFAULT_FORMAT,
    *,
    lang: str = "en",
) -> str:
    """HTML pokedex page (human view); machine-readable markdown at species_url()."""
    slug = urllib.parse.quote(pikalytics_slug(species), safe="-")
    return f"{PIKALYTICS_BASE}/pokedex/{format_code}/{slug}?l={lang}"


def species_url(species: str, format_code: str = DEFAULT_FORMAT) -> str:
    slug = urllib.parse.quote(pikalytics_slug(species), safe="-")
    return ai_url(f"/ai/pokedex/{format_code}/{slug}")


def _strip_md_bold(text: str) -> str:
    return text.replace("**", "").strip()


def discover_species(markdown: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in SPECIES_LINK_RE.finditer(markdown):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _parse_usage_section(markdown: str, header: str) -> list[UsageEntry]:
    start = markdown.find(f"## {header}")
    if start < 0:
        return []
    rest = markdown[start + len(header) + 3 :]
    end = rest.find("\n## ")
    block = rest[:end] if end >= 0 else rest
    entries: list[UsageEntry] = []
    for match in USAGE_LINE_RE.finditer(block):
        entries.append(
            UsageEntry(name=match.group("name").strip(), usage_pct=float(match.group("pct")))
        )
    return entries


def _parse_quick_info(markdown: str) -> dict[str, str]:
    start = markdown.find("## Best ")
    if start < 0:
        return {}
    rest = markdown[start:]
    end = rest.find("\n## Page Links")
    block = rest[:end] if end >= 0 else rest
    info: dict[str, str] = {}
    for match in QUICK_INFO_RE.finditer(block):
        info[match.group("key").strip()] = match.group("value").strip()
    return info


def _parse_ev_spread(markdown: str) -> tuple[str | None, str | None, float | None]:
    match = EV_SPREAD_RE.search(markdown)
    if not match:
        return None, None, None
    return match.group("nature"), match.group("evs"), float(match.group("pct"))


def _parse_featured_sets(markdown: str) -> list[FeaturedSet]:
    start = markdown.find("## Featured Teams with ")
    if start < 0:
        return []
    block = markdown[start:]
    sets: list[FeaturedSet] = []
    for match in TEAM_SET_RE.finditer(block):
        sets.append(
            FeaturedSet(
                author=match.group("author").strip(),
                record=match.group("record").strip(),
                event=match.group("event").strip(),
                team_pokemon=[p.strip() for p in match.group("pokemon").split(",")],
                ability=match.group("ability").strip(),
                item=match.group("item").strip(),
                moves=[m.strip() for m in match.group("moves").split(",")],
            )
        )
    return sets


def parse_species_markdown(species: str, markdown: str) -> PokemonMeta:
    info = _parse_quick_info(markdown)
    nature, evs, spread_pct = _parse_ev_spread(markdown)

    def _pct_or_none(raw: str) -> float | None:
        text = (raw or "").replace("%", "").strip()
        if not text or text.upper() == "N/A":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    meta = PokemonMeta(
        species=species,
        usage_pct=_pct_or_none(info.get("Usage", "")),
        win_rate_pct=_pct_or_none(info.get("Win Rate", "")),
        record=info.get("Record"),
        data_date=info.get("Data Date"),
        moves=_parse_usage_section(markdown, SECTION_HEADERS["moves"]),
        abilities=_parse_usage_section(markdown, SECTION_HEADERS["abilities"]),
        items=_parse_usage_section(markdown, SECTION_HEADERS["items"]),
        teammates=_parse_usage_section(markdown, SECTION_HEADERS["teammates"]),
        top_nature=nature,
        top_evs=evs,
        top_spread_pct=spread_pct,
        featured_sets=_parse_featured_sets(markdown),
        source_url=species_url(species),
    )
    return meta


def _parse_core_section(markdown: str, header: str) -> list[dict[str, Any]]:
    start = markdown.find(header)
    if start < 0:
        return []
    rest = markdown[start + len(header) :]
    end = rest.find("\n### ")
    if end < 0:
        end = rest.find("\n## ")
    block = rest[:end] if end >= 0 else rest
    cores: list[dict[str, Any]] = []
    for match in CORE_ROW_RE.finditer(block):
        pokemon = [p.strip() for p in match.group("core").split(",")]
        cores.append(
            {
                "rank": int(match.group("rank")),
                "pokemon": pokemon,
                "teams": int(match.group("teams")),
                "usage_pct": float(match.group("pct")),
            }
        )
    return cores


def _parse_top_usage_table(markdown: str) -> list[dict[str, Any]]:
    header = "| Rank | Pokemon | Usage % |"
    start = markdown.find(header)
    if start < 0:
        return []
    rest = markdown[start:]
    end = rest.find("\n## ")
    block = rest[:end] if end >= 0 else rest
    rows: list[dict[str, Any]] = []
    for line in block.splitlines():
        if not line.startswith("|") or "Pokemon" in line or line.startswith("| ---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        rank_text = parts[0].rstrip(".")
        if not rank_text.isdigit():
            continue
        usage_text = parts[2].replace("%", "").strip()
        win_text = parts[3].replace("%", "").strip() if len(parts) > 3 else ""

        def _pct_or_none(text: str) -> float | None:
            if not text or text.upper() == "N/A":
                return None
            try:
                return float(text)
            except ValueError:
                return None

        rows.append(
            {
                "rank": int(rank_text),
                "species": _strip_md_bold(parts[1]),
                "usage_pct": _pct_or_none(usage_text),
                "win_rate_pct": _pct_or_none(win_text),
                "record": parts[4] if len(parts) > 4 else None,
            }
        )
    return rows


def parse_format_markdown(markdown: str, format_code: str = DEFAULT_FORMAT) -> FormatMeta:
    data_date_match = re.search(r"\*\*Data Date\*\*:\s*(\S+)", markdown)
    return FormatMeta(
        format_code=format_code,
        data_date=data_date_match.group(1) if data_date_match else None,
        species=discover_species(markdown),
        top_usage=_parse_top_usage_table(markdown),
        cores_2=_parse_core_section(markdown, "### 2-Pokemon Cores"),
        cores_3=_parse_core_section(markdown, "### 3-Pokemon Cores"),
        source_url=format_index_url(format_code),
    )


def fetch_format_meta(
    format_code: str = DEFAULT_FORMAT,
    *,
    delay_s: float = 0.5,
) -> FormatMeta:
    markdown = _http_get(format_index_url(format_code))
    if delay_s > 0:
        time.sleep(delay_s)
    return parse_format_markdown(markdown, format_code)


def fetch_species_meta(
    species: str,
    format_code: str = DEFAULT_FORMAT,
    *,
    delay_s: float = 0.5,
) -> PokemonMeta:
    markdown = _http_get(species_url(species, format_code))
    if delay_s > 0:
        time.sleep(delay_s)
    return parse_species_markdown(species, markdown)


def discover_species_targets(
    *,
    format_code: str = DEFAULT_FORMAT,
    pool_dir: Path | None = None,
    crawl_teammates: bool = False,
    use_battle_usage: bool = True,
    use_legal_list: bool = False,
) -> list[str]:
    """Build species fetch list.

    Default: Pikalytics Battle Usage JSON (`/api/l/{date}/{format}-1760`, ~270 species).
    Optionally add Showdown Champions learnsets, format index links, and pool names.
    """
    from src.doubles.planning.champions_legal import load_legal_species_names
    from src.core.planning.species_normalize import clean_species_name

    pool_dir = pool_dir or Path(__file__).resolve().parents[2] / "teams" / "opponents"
    seen: set[str] = set()
    ordered: list[str] = []

    def add(name: str) -> None:
        clean = clean_species_name(name)
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)

    if use_battle_usage:
        try:
            for name in discover_species_from_battle_usage(format_code):
                add(name)
        except Exception:
            pass

    if use_legal_list:
        for name in load_legal_species_names():
            add(name)

    try:
        format_meta = fetch_format_meta(format_code, delay_s=0.0)
        for name in format_meta.species:
            add(name)
        for row in format_meta.top_usage:
            add(row.get("species", ""))
    except Exception:
        pass

    if pool_dir.is_dir():
        from poke_env.teambuilder import Teambuilder

        for path in sorted(pool_dir.glob("*.txt")):
            try:
                mons = Teambuilder.parse_showdown_team(path.read_text(encoding="utf-8"))
            except Exception:
                for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
                    if block.strip():
                        add(block.strip().splitlines()[0].split("@")[0])
                continue
            for mon in mons:
                add(mon.nickname or mon.species or "")

    if crawl_teammates:
        from config.settings import PIKALYTICS_META_PATH

        pikalytics_path = PIKALYTICS_META_PATH
        if not pikalytics_path.is_file():
            pikalytics_path = pool_dir.parent / "meta" / "pikalytics_reg_ma.json"
        if pikalytics_path.is_file():
            cached = json.loads(pikalytics_path.read_text(encoding="utf-8"))
            for entry in cached.get("pokemon", {}).values():
                for tm in entry.get("teammates", []):
                    add(tm.get("name", ""))

    return ordered


def fetch_all_meta(
    *,
    format_code: str = DEFAULT_FORMAT,
    species: list[str] | None = None,
    top_n: int | None = None,
    delay_s: float = 0.5,
) -> dict[str, Any]:
    format_md = _http_get(format_index_url(format_code))
    format_meta = parse_format_markdown(format_md, format_code)

    target_species = species or format_meta.species
    if top_n is not None:
        if format_meta.top_usage:
            ranked = [row["species"] for row in format_meta.top_usage[:top_n]]
            target_species = ranked
        else:
            target_species = target_species[:top_n]

    from src.core.planning.species_normalize import expand_mega_form_targets

    target_species = expand_mega_form_targets(target_species)

    battle_usage: dict[str, float] = {}
    try:
        for row in fetch_battle_usage_list(format_code, delay_s=0.0):
            name = str(row.get("name") or "").strip()
            pct = row.get("percent")
            if name and pct is not None:
                battle_usage[name] = float(pct)
    except Exception:
        pass

    pokemon: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for i, name in enumerate(target_species):
        try:
            markdown = _http_get(species_url(name, format_code))
            pokemon[name] = asdict(parse_species_markdown(name, markdown))
        except urllib.error.HTTPError as exc:
            errors[name] = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - collect per-species failures
            errors[name] = str(exc)
        if delay_s > 0 and i + 1 < len(target_species):
            time.sleep(delay_s)

    return {
        "format": asdict(format_meta),
        "pokemon": pokemon,
        "battle_usage": battle_usage,
        "errors": errors,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fetch_missing_species(
    existing_path: Path,
    *,
    format_code: str = DEFAULT_FORMAT,
    delay_s: float = 0.5,
) -> dict[str, Any]:
    """Fetch only species absent from an existing meta cache (merge-friendly)."""
    cached: set[str] = set()
    if existing_path.is_file():
        cached = set(json.loads(existing_path.read_text(encoding="utf-8")).get("pokemon", {}))

    from src.core.planning.species_normalize import expand_mega_form_targets

    targets = expand_mega_form_targets(discover_species_from_battle_usage(format_code))
    missing = [name for name in targets if name not in cached]
    if not missing:
        return {"pokemon": {}, "errors": {}, "missing": [], "skipped": len(cached)}

    payload = fetch_all_meta(format_code=format_code, species=missing, delay_s=delay_s)
    payload["missing_fetched"] = missing
    return payload


def merge_meta_json(existing_path: Path, new_data: dict[str, Any]) -> dict[str, Any]:
    if existing_path.is_file():
        merged = json.loads(existing_path.read_text(encoding="utf-8"))
    else:
        merged = {"format": {}, "pokemon": {}, "errors": {}, "battle_usage": {}}

    merged.setdefault("pokemon", {}).update(new_data.get("pokemon", {}))
    if new_data.get("battle_usage"):
        merged["battle_usage"] = new_data["battle_usage"]
    if new_data.get("format"):
        merged["format"] = new_data["format"]
    merged.setdefault("errors", {}).update(new_data.get("errors", {}))
    merged["fetched_at"] = new_data.get("fetched_at", merged.get("fetched_at"))
    return merged


def save_meta_json(data: dict[str, Any], path: Path, *, merge: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.is_file():
        data = merge_meta_json(path, data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
