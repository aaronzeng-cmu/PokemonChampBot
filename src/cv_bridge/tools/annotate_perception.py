"""Generate labeled annotation images for perception / sprite-matcher sanity checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.cv_bridge.action_executor import load_ui_coordinates
from src.cv_bridge.perception import PerceptionModule, _TEMPLATE_SEARCH_REGIONS
from src.cv_bridge.sprite_matcher import SpriteMatcher

_DEFAULT_SCREENSHOTS = Path("logs/cv_bridge/screenshots")
_DEFAULT_OUTPUT = Path("logs/cv_bridge/analysis/annotated")

_COLORS = {
    "ally": (80, 220, 80),
    "enemy": (0, 200, 255),
    "hp": (0, 220, 255),
    "name": (255, 80, 255),
    "template_roi": (255, 160, 40),
    "header_bg": (30, 30, 30),
    "text": (255, 255, 255),
    "accent": (100, 220, 255),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read screenshot: {path}")
    return frame


def _label_bg(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = _COLORS["text"],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(canvas, (x - 2, y - th - 6), (x + tw + 4, y + baseline + 2), _COLORS["header_bg"], -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_box(
    canvas: np.ndarray,
    spec: list[int] | tuple[int, ...],
    *,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x, y, w, h = (int(v) for v in spec)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    _label_bg(canvas, label, (x + 4, max(y + 18, 20)), color=color)


def _banner(canvas: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = canvas.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), _COLORS["header_bg"], -1)
    cv2.putText(out, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COLORS["accent"], 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLORS["text"], 1, cv2.LINE_AA)
    return out


def _save_pair(output_dir: Path, stem: str, image: np.ndarray, meta: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    cv2.imwrite(str(png_path), image)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png_path


def annotate_parse_team_preview(
    module: PerceptionModule,
    screenshot: Path,
    output_dir: Path,
    *,
    stem: str = "parse_team_preview",
    mirror_path: Path | None = None,
) -> dict[str, Any]:
    frame = _load_frame(screenshot)
    result = module.parse_team_preview(frame)
    canvas = _banner(
        frame,
        "parse_team_preview()",
        f"{screenshot.name}  ally + enemy = sprite_matcher.identify_sprite()",
    )

    tp = module.teampreview_regions
    ally_slots = tp.get("ally_sprite_slots", {})
    if not ally_slots:
        ally_slots = tp.get("ally_name_slots", {})
    enemy_slots = tp.get("enemy_sprite_slots", {})

    for i in range(1, 7):
        key = f"slot_{i}"
        ally_spec = ally_slots.get(key)
        enemy_spec = enemy_slots.get(key)
        ally_label = result["ally_team"][i - 1]
        enemy_label = result["enemy_team"][i - 1]
        if ally_spec:
            _draw_box(
                canvas,
                ally_spec,
                color=_COLORS["ally"],
                label=f"A{i}: {ally_label}",
            )
        if enemy_spec:
            _draw_box(
                canvas,
                enemy_spec,
                color=_COLORS["enemy"],
                label=f"E{i}: {enemy_label}",
            )

    meta = {
        "function": "parse_team_preview",
        "screenshot": screenshot.name,
        "generated_at_utc": _utc_now(),
        "result": result,
    }
    _save_pair(output_dir, stem, canvas, meta)
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mirror_path), canvas)
        mirror_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def annotate_extract_battle_data(
    module: PerceptionModule,
    screenshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frame = _load_frame(screenshot)
    data = module.extract_battle_data(frame)
    canvas = _banner(
        frame,
        "extract_battle_data()",
        f"{screenshot.name}  species=sprite match  HP=OCR",
    )

    slot_defs = [
        ("player_slot_a", "player_active_sprite_slot_a", "player_active_hp_slot_a", "P-A"),
        ("player_slot_b", "player_active_sprite_slot_b", "player_active_hp_slot_b", "P-B"),
        ("opp_slot_a", "opp_active_sprite_slot_a", "opp_active_hp_slot_a", "O-A"),
        ("opp_slot_b", "opp_active_sprite_slot_b", "opp_active_hp_slot_b", "O-B"),
    ]
    for slot_key, sprite_key, hp_key, short in slot_defs:
        slot = data.get(slot_key, {})
        species = slot.get("species_id") or "?"
        hp_text = slot.get("hp_text") or "?"
        sprite_spec = module.regions.get(sprite_key)
        hp_spec = module.regions.get(hp_key)
        if sprite_spec:
            _draw_box(
                canvas,
                sprite_spec,
                color=_COLORS["ally"],
                label=f"{short} sprite: {species}",
            )
        if hp_spec:
            _draw_box(
                canvas,
                hp_spec,
                color=_COLORS["hp"],
                label=f"{short} HP: {hp_text}",
            )

    preview_spec = module.regions.get("teampreview_selection_counter")
    if preview_spec and data.get("teampreview"):
        tp = data["teampreview"]
        _draw_box(
            canvas,
            preview_spec,
            color=_COLORS["accent"],
            label=f"preview: {tp.get('text', '')}",
        )

    timer_spec = module.regions.get("decision_state_move_timer")
    if timer_spec and data.get("move_timer") is not None:
        _draw_box(
            canvas,
            timer_spec,
            color=_COLORS["template_roi"],
            label=f"timer: {data['move_timer']}",
        )

    meta = {
        "function": "extract_battle_data",
        "screenshot": screenshot.name,
        "generated_at_utc": _utc_now(),
        "result": data,
    }
    _save_pair(output_dir, "extract_battle_data", canvas, meta)
    return meta


def annotate_match_template(
    module: PerceptionModule,
    screenshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frame = _load_frame(screenshot)
    canvas = _banner(
        frame,
        "match_template()",
        f"{screenshot.name}  template search ROIs + confidence scores",
    )

    scores: dict[str, float] = {}
    for name, region_key in _TEMPLATE_SEARCH_REGIONS.items():
        roi_spec = module.regions.get(region_key) if region_key else None
        if roi_spec:
            _draw_box(
                canvas,
                roi_spec,
                color=_COLORS["template_roi"],
                label=name,
            )
        scores[name] = module.match_template(frame, name)

    y = 70
    for name, conf in sorted(scores.items(), key=lambda item: -item[1]):
        line = f"{name}: {conf:.3f}"
        _label_bg(canvas, line, (frame.shape[1] - 340, y), color=_COLORS["text"], scale=0.48)
        y += 22

    meta = {
        "function": "match_template",
        "screenshot": screenshot.name,
        "generated_at_utc": _utc_now(),
        "confidences": scores,
    }
    _save_pair(output_dir, "match_template", canvas, meta)
    return meta


def annotate_perceive_states(
    module: PerceptionModule,
    shots: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Montage of perceive() results across representative screenshots."""
    entries: list[dict[str, Any]] = []
    thumbs: list[np.ndarray] = []

    for label, path in shots.items():
        frame = _load_frame(path)
        result = module.perceive(frame)
        thumb = cv2.resize(frame, (480, 270))
        cv2.rectangle(thumb, (0, 0), (479, 269), _COLORS["accent"], 2)
        _label_bg(thumb, label, (8, 22), color=_COLORS["accent"], scale=0.45)
        state_line = f"{result.state} ({result.state_confidence:.2f})"
        _label_bg(thumb, state_line, (8, 44), scale=0.42)
        match_line = result.template_match or "none"
        _label_bg(thumb, match_line, (8, 64), scale=0.38)
        thumbs.append(thumb)
        entries.append(
            {
                "label": label,
                "screenshot": path.name,
                "state": result.state,
                "state_confidence": result.state_confidence,
                "template_match": result.template_match,
                "battle_format": result.battle_format,
            }
        )

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    cell_h, cell_w = 270, 480
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * cell_w
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = thumb

    header = np.zeros((48, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        "perceive() / get_current_state()  —  detected game state per screenshot",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _COLORS["accent"],
        2,
        cv2.LINE_AA,
    )
    canvas = np.vstack([header, grid])

    meta = {
        "function": "perceive",
        "generated_at_utc": _utc_now(),
        "entries": entries,
    }
    _save_pair(output_dir, "perceive_states", canvas, meta)
    return meta


def annotate_sprite_matcher(
    matcher: SpriteMatcher,
    module: PerceptionModule,
    screenshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frame = _load_frame(screenshot)
    canvas = _banner(
        frame,
        "sprite_matcher.match_sprite() / identify_sprite()",
        f"{screenshot.name}  enemy teampreview crops with distance + margin",
    )

    enemy_slots = module.teampreview_regions.get("enemy_sprite_slots", {})
    matches: list[dict[str, Any]] = []
    if not matcher.ready:
        matcher.build_index()

    for i in range(1, 7):
        key = f"slot_{i}"
        spec = enemy_slots.get(key)
        if not spec:
            continue
        x, y, w, h = (int(v) for v in spec)
        crop = frame[y : y + h, x : x + w]
        match = matcher.match_sprite(crop)
        species = match.species_id if match else "unknown"
        dist = match.distance if match else None
        margin = match.margin if match else None
        label = species
        if dist is not None and margin is not None:
            label = f"E{i}: {species} d={dist:.2f} m={margin:.3f}"
        _draw_box(canvas, spec, color=_COLORS["enemy"], label=label)
        matches.append(
            {
                "slot": i,
                "species_id": species,
                "distance": dist,
                "margin": margin,
                "crop_box": spec,
            }
        )

    meta = {
        "function": "sprite_matcher.match_sprite",
        "screenshot": screenshot.name,
        "generated_at_utc": _utc_now(),
        "matches": matches,
    }
    _save_pair(output_dir, "sprite_matcher", canvas, meta)
    return meta


def annotate_preprocess_for_ocr(
    module: PerceptionModule,
    screenshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Side-by-side raw vs preprocessed OCR crops for ally teampreview slots."""
    frame = _load_frame(screenshot)
    ally_slots = module.teampreview_regions.get("ally_name_slots", {})
    panels: list[np.ndarray] = []

    samples: list[dict[str, Any]] = []
    for i in range(1, 4):
        spec = ally_slots.get(f"slot_{i}")
        if not spec:
            continue
        x, y, w, h = (int(v) for v in spec)
        raw = frame[y : y + h, x : x + w]
        if raw.size == 0:
            continue
        processed = module.preprocess_for_ocr(raw)
        proc_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        text = module._ocr_crop(raw) if module.ocr_enabled else ""
        species = module._map_ocr_to_species_id(text)

        raw_panel = cv2.resize(raw, (320, 80))
        proc_panel = cv2.resize(proc_bgr, (320, 80))
        stack = np.vstack([raw_panel, proc_panel])
        cv2.rectangle(stack, (0, 0), (319, 159), _COLORS["ally"], 2)
        _label_bg(stack, f"slot {i}: {species}", (8, 18), color=_COLORS["ally"], scale=0.45)
        _label_bg(stack, "raw", (8, 48), scale=0.38)
        _label_bg(stack, "preprocess_for_ocr()", (8, 128), scale=0.38)
        panels.append(stack)
        samples.append({"slot": i, "ocr_text": text, "species_id": species})

    if not panels:
        raise RuntimeError("No ally slots found for OCR preprocess annotation")

    canvas = np.hstack(panels)
    header = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"preprocess_for_ocr() + _ocr_crop()  —  {screenshot.name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        _COLORS["accent"],
        2,
        cv2.LINE_AA,
    )
    canvas = np.vstack([header, canvas])

    meta = {
        "function": "preprocess_for_ocr",
        "screenshot": screenshot.name,
        "generated_at_utc": _utc_now(),
        "samples": samples,
    }
    _save_pair(output_dir, "preprocess_for_ocr", canvas, meta)
    return meta


def _default_screenshot_map(screenshot_dir: Path) -> dict[str, str]:
    coords = load_ui_coordinates()
    index = coords.get("_screenshot_index", {})
    return {label: screenshot_dir / name for label, name in index.items()}


def run_annotations(
    *,
    screenshot_dir: Path = _DEFAULT_SCREENSHOTS,
    output_dir: Path = _DEFAULT_OUTPUT,
    ocr_enabled: bool = True,
) -> dict[str, Any]:
    screenshot_dir = Path(screenshot_dir)
    output_dir = Path(output_dir)
    index = _default_screenshot_map(screenshot_dir)

    def pick(key: str, fallback: str) -> Path:
        path = index.get(key)
        if path is None or not path.exists():
            path = screenshot_dir / fallback
        if not path.exists():
            raise FileNotFoundError(f"Missing screenshot for {key}: {path}")
        return path

    module = PerceptionModule(ocr_enabled=ocr_enabled)
    matcher = module.sprite_matcher

    perceive_shots = {
        "TEAM_PREVIEW": pick("teampreview.singles.empty", "20260619_041802.png"),
        "TURN_DECISION": pick("battle.doubles.command_menu", "20260624_232432.png"),
        "MOVE_SELECTION": pick("battle.doubles.move_menu_mega", "20260624_232439.png"),
        "TARGET_SELECTION": pick("battle.doubles.target_overlay_focus_blast", "20260624_232712.png"),
        "COMMUNICATING": pick("system.communicating", "20260624_232816.png"),
        "LOADING": pick("system.loading", "20260619_041758.png"),
        "RESULTS": pick("post_battle.results_doubles", "20260624_232854.png"),
        "IDLE": pick("navigation.lobby", "20260619_041729.png"),
    }

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "output_dir": str(output_dir.resolve()),
        "ocr_enabled": ocr_enabled,
        "annotations": {},
    }

    teampreview_shot = pick("teampreview.singles.empty", "20260619_041802.png")
    teampreview_mirror = output_dir.parent / "annotated_teampreview_empty.png"

    summary["annotations"]["teampreview_empty"] = annotate_parse_team_preview(
        module,
        teampreview_shot,
        output_dir,
        stem="teampreview_empty",
        mirror_path=teampreview_mirror,
    )
    summary["annotations"]["extract_battle_data"] = annotate_extract_battle_data(
        module,
        pick("battle.doubles.command_menu", "20260624_232432.png"),
        output_dir,
    )
    summary["annotations"]["match_template"] = annotate_match_template(
        module,
        pick("battle.doubles.command_menu", "20260624_232432.png"),
        output_dir,
    )
    summary["annotations"]["perceive_states"] = annotate_perceive_states(
        module,
        perceive_shots,
        output_dir,
    )
    summary["annotations"]["sprite_matcher"] = annotate_sprite_matcher(
        matcher,
        module,
        teampreview_shot,
        output_dir,
    )
    if ocr_enabled:
        summary["annotations"]["preprocess_for_ocr"] = annotate_preprocess_for_ocr(
            module,
            teampreview_shot,
            output_dir,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled perception sanity-check images.")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=_DEFAULT_SCREENSHOTS,
        help="Directory containing emulator PNG screenshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Where to write annotated PNG + JSON pairs.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip EasyOCR (regions only; ally names show as unknown).",
    )
    args = parser.parse_args()

    summary = run_annotations(
        screenshot_dir=args.screenshot_dir,
        output_dir=args.output_dir,
        ocr_enabled=not args.no_ocr,
    )
    print(f"Wrote annotations to {args.output_dir.resolve()}")
    for name in summary["annotations"]:
        print(f"  - {name}.png")


if __name__ == "__main__":
    main()
