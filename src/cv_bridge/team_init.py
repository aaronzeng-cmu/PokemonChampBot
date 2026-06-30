"""Parse our team from the in-game team view into a profile JSON.

The Champions "Team" screen has two tabs, both a 2x3 grid of our six Pokemon:

* **Moves & More** -- per mon: name, ability, held item, and four moves.
* **Stats**        -- per mon: name and the six stats (we keep HP as ``max_hp``).

This reads both screenshots with EasyOCR, buckets the recognized tokens into the
six cells by position, repairs OCR noise by fuzzy-matching moves/species against
the poke-env dex, and writes ``teams/<name>.json`` -- the profile consumed by
``LiveBattleTracker.load_player_team`` so our actives carry move lists (otherwise
the legal-action mask is empty and the bot is forced to pass).

Species whose in-game name is localized (e.g. Grimmsnarl/Staraptor render in
Japanese) won't OCR; we recover them from a mega-stone item when possible and
otherwise emit an empty ``species`` with a warning to fill in by hand.

Usage::

    python -m src.cv_bridge.team_init \
        --moves logs/cv_bridge/screenshots/<moves>.png \
        --stats logs/cv_bridge/screenshots/<stats>.png \
        --name Clams --out teams/champions_live_team.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from poke_env.data import GenData, to_id_str

# --- Grid layout (fractions of the 2x3 team view; resolution-independent) ---
_COL_SPLIT = 0.5  # cx < split -> left mon, else right mon
# (cy_lo, cy_hi) name/ability/item/move band for each of the three rows.
_ROW_BANDS = ((0.26, 0.45), (0.46, 0.66), (0.67, 0.90))
# Per-column x ranges for the info (name/ability/item) vs the four-move sub-column.
_INFO_X = {"L": (0.06, 0.30), "R": (0.50, 0.71)}
_MOVES_X = {"L": (0.30, 0.52), "R": (0.71, 0.99)}
_LINE_GAP = 0.025  # cy gap (fraction of height) that starts a new text line

# Hand-drawn icon boxes (annotate_regions recipe 'teamview') -> species via the
# same sprite matcher used at team preview. Megas are excluded since the team
# view shows base forms only.
_DEFAULT_ICON_ANN = Path("logs/cv_bridge/analysis/region_annotations/teamview.json")


@dataclass
class _Token:
    text: str
    cx: float
    cy: float


@dataclass
class _MonSlot:
    name_tokens: list[_Token] = field(default_factory=list)
    move_tokens: list[_Token] = field(default_factory=list)
    stat_tokens: list[_Token] = field(default_factory=list)


def _ocr(reader: Any, image_path: Path) -> list[_Token]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    tokens: list[_Token] = []
    for box, text, _conf in reader.readtext(image, detail=1):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = (sum(xs) / 4.0) / w
        cy = (sum(ys) / 4.0) / h
        text = str(text).strip()
        if text:
            tokens.append(_Token(text=text, cx=cx, cy=cy))
    return tokens


def _row_for(cy: float) -> int | None:
    for i, (lo, hi) in enumerate(_ROW_BANDS):
        if lo <= cy < hi:
            return i
    return None


def _bucket(tokens: list[_Token], *, into: str, slots: list[_MonSlot]) -> None:
    """Assign tokens to the six mon slots' name/move (or stat) buckets."""
    for tok in tokens:
        row = _row_for(tok.cy)
        if row is None:
            continue
        col = "L" if tok.cx < _COL_SPLIT else "R"
        slot = slots[row * 2 + (0 if col == "L" else 1)]
        if into == "stats":
            slot.stat_tokens.append(tok)
            continue
        if _INFO_X[col][0] <= tok.cx < _INFO_X[col][1]:
            slot.name_tokens.append(tok)
        elif _MOVES_X[col][0] <= tok.cx < _MOVES_X[col][1]:
            slot.move_tokens.append(tok)


def _group_lines(tokens: list[_Token]) -> list[str]:
    """Merge tokens into text lines (same cy) ordered top-to-bottom, left-to-right."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: (t.cy, t.cx))
    lines: list[list[_Token]] = [[ordered[0]]]
    for tok in ordered[1:]:
        if abs(tok.cy - lines[-1][-1].cy) <= _LINE_GAP:
            lines[-1].append(tok)
        else:
            lines.append([tok])
    return [" ".join(t.text for t in sorted(line, key=lambda t: t.cx)) for line in lines]


def _fuzzy(text: str, candidates: set[str], *, cutoff: float = 0.72) -> str | None:
    key = to_id_str(text)
    if not key:
        return None
    if key in candidates:
        return key
    match = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
    return match[0] if match else None


def _load_icon_boxes(path: Path) -> dict[int, list[int]]:
    """Read slot->[x,y,w,h] icon boxes from a teamview annotation or ui_coordinates file."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: dict[int, list[int]] = {}
    sources: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("regions"), dict):
            sources.append(data["regions"])  # annotation-file format
        nested = data.get("teamview")
        if isinstance(nested, dict) and isinstance(nested.get("icon_slots"), dict):
            sources.append(nested["icon_slots"])  # ui_coordinates format
    for src in sources:
        for key, box in src.items():
            m = re.search(r"(\d+)\s*$", str(key))
            if m and isinstance(box, list) and len(box) == 4:
                boxes[int(m.group(1))] = [int(v) for v in box]
    return boxes


def _default_icon_recognizer() -> Any:
    """CNN species recognizer (pHash fallback) when trained weights exist."""
    weights = Path(__file__).resolve().parent / "assets" / "species_cls.pt"
    if weights.is_file():
        from src.cv_bridge.species_classifier import SpeciesRecognizer

        return SpeciesRecognizer(weights=weights)
    from src.cv_bridge.sprite_matcher import SpriteMatcher

    return SpriteMatcher()


def _species_from_icon(
    image: Any, box: list[int] | None, matcher: Any, pokedex: set[str]
) -> str | None:
    """Identify a species from its team-view icon crop (base forms only)."""
    if image is None or not box:
        return None
    x, y, w, h = box
    crop = image[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0:
        return None
    species = matcher.identify_sprite(crop, exclude_forms=True)
    if species and species != "unknown" and species in pokedex:
        return species
    return None


def _species_from_mega_stone(item_id: str, pokedex: set[str]) -> str | None:
    """Derive a species from a held mega stone (e.g. ``staraptite`` -> ``staraptor``)."""
    if not item_id.endswith("ite") and "ite" not in item_id:
        return None
    stem = re.sub(r"ite[xy]?$", "", item_id)  # raichunitey -> raichun, staraptite -> starapt
    stem = stem.rstrip("n")  # raichun -> raichu
    return _fuzzy(stem, pokedex, cutoff=0.6)


def parse_team(
    moves_image: Path,
    stats_image: Path | None,
    *,
    name: str = "Team",
    battle_format: str = "doubles",
    reader: Any | None = None,
    icon_boxes: dict[int, list[int]] | None = None,
    sprite_matcher: Any | None = None,
) -> dict[str, Any]:
    """Parse the two team-view screenshots into a team profile dict.

    Species are resolved by template-matching the per-slot icon crops (when
    ``icon_boxes`` are provided, e.g. from the ``teamview`` annotate recipe),
    falling back to OCR of the name and then to a held mega stone.
    """
    if reader is None:
        import easyocr

        gpu = False
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    gen = GenData.from_gen(9)
    pokedex = set(gen.pokedex)
    move_ids = set(gen.moves)

    icon_boxes = icon_boxes or {}
    moves_bgr = None
    if icon_boxes:
        if sprite_matcher is None:
            sprite_matcher = _default_icon_recognizer()
            sprite_matcher.build_index()
        moves_bgr = cv2.imread(str(moves_image))

    slots = [_MonSlot() for _ in range(6)]
    _bucket(_ocr(reader, moves_image), into="moves", slots=slots)
    if stats_image is not None:
        _bucket(_ocr(reader, stats_image), into="stats", slots=slots)

    pokemon: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, slot in enumerate(slots, start=1):
        info_lines = _group_lines(slot.name_tokens)
        move_lines = _group_lines(slot.move_tokens)[:4]
        raw_name = info_lines[0] if info_lines else ""
        ability = info_lines[1] if len(info_lines) > 1 else ""
        item = info_lines[2] if len(info_lines) > 2 else ""

        item_id = to_id_str(item)
        # OCR name first (reliable for English names); the icon only fills slots
        # OCR can't read (localized/Japanese names) and only when confident, so a
        # wrong-but-confident icon guess can't override a good OCR name.
        species = _fuzzy(raw_name, pokedex, cutoff=0.8)
        if species is None:
            species = _species_from_icon(moves_bgr, icon_boxes.get(i), sprite_matcher, pokedex)
        if species is None:
            species = _species_from_mega_stone(item_id, pokedex)
        if species is None:
            warnings.append(
                f"slot {i}: could not resolve species (OCR name={raw_name!r}); "
                "fill 'species' in by hand."
            )

        moves: list[str] = []
        for line in move_lines:
            mid = _fuzzy(line, move_ids, cutoff=0.6)
            if mid:
                moves.append(mid)
            else:
                warnings.append(f"slot {i}: unresolved move OCR {line!r} (skipped).")

        entry: dict[str, Any] = {
            "species": species or "",
            "ability": to_id_str(ability),
            "item": item_id,
            "moves": moves,
            "max_hp": _parse_max_hp(slot.stat_tokens),
        }
        if item_id.endswith("ite") or re.search(r"ite[xy]$", item_id):
            entry["mega"] = True
        pokemon.append(entry)

    for msg in warnings:
        print(f"[team_init][warn] {msg}")

    return {"name": name, "format": battle_format, "pokemon": pokemon}


def _parse_max_hp(stat_tokens: list[_Token]) -> int:
    """Find the HP stat (number right of the 'HP' label on the same line)."""
    if not stat_tokens:
        return 0
    hp_label = next((t for t in stat_tokens if to_id_str(t.text) in ("hp", "h")), None)
    if hp_label is None:
        return 0
    # Numbers immediately right of "HP" only -- bound cx so we don't grab the
    # right-hand stat column (Sp. Atk shares the HP row).
    near = [
        t
        for t in stat_tokens
        if abs(t.cy - hp_label.cy) <= _LINE_GAP
        and hp_label.cx < t.cx < hp_label.cx + 0.13
    ]
    numbers = [int(m) for t in near for m in re.findall(r"\d+", t.text)]
    # Within the HP cluster the stat is the larger number; the EV is the small one.
    return max(numbers) if numbers else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse our team from team-view screenshots.")
    parser.add_argument("--moves", type=Path, required=True, help="'Moves & More' screenshot.")
    parser.add_argument("--stats", type=Path, default=None, help="'Stats' screenshot (optional).")
    parser.add_argument("--name", default="Team", help="Team name for the profile.")
    parser.add_argument("--format", default="doubles", choices=["singles", "doubles"])
    parser.add_argument("--out", type=Path, required=True, help="Output team profile JSON.")
    parser.add_argument(
        "--icons",
        type=Path,
        default=_DEFAULT_ICON_ANN,
        help=(
            "teamview icon boxes (annotate_regions 'teamview' output or ui_coordinates.json). "
            "When present, species come from icon template matching instead of OCR."
        ),
    )
    args = parser.parse_args()

    icon_boxes = _load_icon_boxes(args.icons)
    if icon_boxes:
        print(f"[team_init] using {len(icon_boxes)} icon box(es) from {args.icons} for species.")
    else:
        print(f"[team_init] no icon boxes at {args.icons}; resolving species via OCR.")

    profile = parse_team(
        args.moves,
        args.stats,
        name=args.name,
        battle_format=args.format,
        icon_boxes=icon_boxes,
    )

    # Don't silently clobber a hand-corrected profile: back it up first.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        backup.write_text(args.out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[team_init] backed up existing {args.out} -> {backup}")
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}")
    for mon in profile["pokemon"]:
        print(
            f"  {mon['species'] or '???':<12} {mon['ability']:<14} {mon['item']:<14} "
            f"hp={mon['max_hp']:<4} moves={mon['moves']}"
        )
    unresolved = [i for i, m in enumerate(profile["pokemon"], 1) if not m["species"]]
    if unresolved:
        print(
            f"\n[team_init][warn] slot(s) {unresolved} have no species -- fill the "
            f"'species' field in {args.out} by hand before using this profile."
        )


if __name__ == "__main__":
    main()
