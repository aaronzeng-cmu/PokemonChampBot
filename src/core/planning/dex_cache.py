"""Parse Showdown dex data (base + Champions mod) into a JSON cache."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from poke_env.data import to_id_str

_ENTRY_START_RE = re.compile(r"^\t([a-z0-9]+):\s*\{", re.MULTILINE)
_STRING_FIELD_RE = re.compile(r'^\s*(?:name|type|category|shortDesc):\s*"([^"]*)"', re.MULTILINE)
_BASE_POWER_RE = re.compile(r"^\s*basePower:\s*(\d+)", re.MULTILINE)
_INHERIT_RE = re.compile(r"^\s*inherit:\s*true", re.MULTILINE)


def _split_ts_entries(text: str) -> dict[str, str]:
    """Split a TS object table into id -> block text."""
    entries: dict[str, str] = {}
    matches = list(_ENTRY_START_RE.finditer(text))
    for i, match in enumerate(matches):
        entry_id = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[entry_id] = text[start:end]
    return entries


def _parse_move_block(block: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("name", "type", "category", "shortDesc"):
        m = re.search(rf'^\s*{key}:\s*"([^"]*)"', block, re.MULTILINE)
        if m:
            fields[key] = m.group(1)
    bp = _BASE_POWER_RE.search(block)
    if bp:
        fields["basePower"] = int(bp.group(1))
    return fields


def _parse_ability_block(block: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("name", "shortDesc"):
        m = re.search(rf'^\s*{key}:\s*"([^"]*)"', block, re.MULTILINE)
        if m:
            fields[key] = m.group(1)
    return fields


def _merge_entry(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    if not base and override.get("inherit"):
        return {}
    merged = dict(base)
    for key, value in override.items():
        if key == "inherit":
            continue
        if value is not None:
            merged[key] = value
    return merged


def build_dex_cache(showdown_data_dir: Path) -> dict[str, Any]:
    """Build move/ability cache from Showdown data + Champions mod overrides."""
    data_dir = Path(showdown_data_dir)
    moves_base = (data_dir / "moves.ts").read_text(encoding="utf-8", errors="ignore")
    moves_text = (data_dir / "text" / "moves.ts").read_text(encoding="utf-8", errors="ignore")
    abilities_base = (data_dir / "abilities.ts").read_text(encoding="utf-8", errors="ignore")
    abilities_text = (data_dir / "text" / "abilities.ts").read_text(encoding="utf-8", errors="ignore")

    champions_dir = data_dir / "mods" / "champions"
    champions_moves = ""
    champions_abilities = ""
    if (champions_dir / "moves.ts").is_file():
        champions_moves = (champions_dir / "moves.ts").read_text(encoding="utf-8", errors="ignore")
    if (champions_dir / "abilities.ts").is_file():
        champions_abilities = (champions_dir / "abilities.ts").read_text(
            encoding="utf-8", errors="ignore"
        )

    base_moves = _split_ts_entries(moves_base)
    text_moves = _split_ts_entries(moves_text)
    champ_moves = _split_ts_entries(champions_moves) if champions_moves else {}

    moves_out: dict[str, Any] = {}
    all_move_ids = set(base_moves) | set(text_moves) | set(champ_moves)
    for move_id in sorted(all_move_ids):
        parsed = _merge_entry(
            _parse_move_block(base_moves.get(move_id, "")),
            _parse_move_block(champ_moves.get(move_id, "")),
        )
        text_parsed = _parse_move_block(text_moves.get(move_id, ""))
        for key in ("name", "shortDesc"):
            if key in text_parsed:
                parsed[key] = text_parsed[key]
        if parsed.get("name"):
            moves_out[move_id] = parsed

    base_abilities = _split_ts_entries(abilities_base)
    text_abilities = _split_ts_entries(abilities_text)
    champ_abilities = _split_ts_entries(champions_abilities) if champions_abilities else {}

    abilities_out: dict[str, Any] = {}
    all_ability_ids = set(base_abilities) | set(text_abilities) | set(champ_abilities)
    for ability_id in sorted(all_ability_ids):
        parsed = _merge_entry(
            _parse_ability_block(base_abilities.get(ability_id, "")),
            _parse_ability_block(champ_abilities.get(ability_id, "")),
        )
        text_parsed = _parse_ability_block(text_abilities.get(ability_id, ""))
        for key in ("name", "shortDesc"):
            if key in text_parsed:
                parsed[key] = text_parsed[key]
        if parsed.get("name"):
            abilities_out[ability_id] = parsed

    return {
        "format": "gen9championsvgc2026regma",
        "source": str(data_dir.resolve()),
        "moves": moves_out,
        "abilities": abilities_out,
    }


def save_dex_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def load_dex_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"moves": {}, "abilities": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def move_desc(cache: dict[str, Any], move_name: str) -> str:
    entry = cache.get("moves", {}).get(to_id_str(move_name), {})
    return entry.get("shortDesc") or entry.get("name") or move_name


def ability_desc(cache: dict[str, Any], ability_name: str) -> str:
    entry = cache.get("abilities", {}).get(to_id_str(ability_name), {})
    return entry.get("shortDesc") or entry.get("name") or ability_name
