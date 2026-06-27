"""Debug figures for misclassified sprite matches (crop vs ground truth vs prediction)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.cv_bridge.action_executor import load_ui_coordinates
from src.cv_bridge.sprite_matcher import SpriteMatcher

_DEFAULT_SCREENSHOTS = Path("logs/cv_bridge/screenshots")
_DEFAULT_OUTPUT = Path("logs/cv_bridge/analysis/sprite_debug")

_PANEL_H = 160
_PANEL_W = 160

_EVAL_SUITES: list[dict[str, Any]] = [
    {
        "id": "teampreview",
        "screenshot": "20260619_041802.png",
        "slots": [
            {
                "label": "ally_1",
                "region": ("teampreview", "ally_sprite_slots", "slot_1"),
                "expected": "pikachu",
            },
            {
                "label": "ally_2",
                "region": ("teampreview", "ally_sprite_slots", "slot_2"),
                "expected": "kingambit",
            },
            {
                "label": "ally_3",
                "region": ("teampreview", "ally_sprite_slots", "slot_3"),
                "expected": "garchomp",
            },
            {
                "label": "ally_4",
                "region": ("teampreview", "ally_sprite_slots", "slot_4"),
                "expected": "azumarill",
            },
            {
                "label": "ally_5",
                "region": ("teampreview", "ally_sprite_slots", "slot_5"),
                "expected": "gyarados",
            },
            {
                "label": "ally_6",
                "region": ("teampreview", "ally_sprite_slots", "slot_6"),
                "expected": "gengar",
            },
            {
                "label": "enemy_1",
                "region": ("teampreview", "enemy_sprite_slots", "slot_1"),
                "expected": "gardevoir",
            },
            {
                "label": "enemy_2",
                "region": ("teampreview", "enemy_sprite_slots", "slot_2"),
                "expected": "heracross",
            },
            {
                "label": "enemy_3",
                "region": ("teampreview", "enemy_sprite_slots", "slot_3"),
                "expected": "drampa",
            },
            {
                "label": "enemy_4",
                "region": ("teampreview", "enemy_sprite_slots", "slot_4"),
                "expected": "azumarill",
            },
            {
                "label": "enemy_5",
                "region": ("teampreview", "enemy_sprite_slots", "slot_5"),
                "expected": "corviknight",
            },
            {
                "label": "enemy_6",
                "region": ("teampreview", "enemy_sprite_slots", "slot_6"),
                "expected": "abomasnow",
            },
        ],
    },
    {
        "id": "battle_active_doubles",
        "screenshot": "20260624_232432.png",
        "slots": [
            {
                "label": "player_a",
                "region": ("shared", "perception_regions", "player_active_sprite_slot_a"),
                "expected": "raichu",
            },
            {
                "label": "player_b",
                "region": ("shared", "perception_regions", "player_active_sprite_slot_b"),
                "expected": "gyarados",
            },
            {
                "label": "opp_a",
                "region": ("shared", "perception_regions", "opp_active_sprite_slot_a"),
                "expected": "azumarill",
            },
            {
                "label": "opp_b",
                "region": ("shared", "perception_regions", "opp_active_sprite_slot_b"),
                "expected": "heracross",
            },
        ],
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_region(coords: dict[str, Any], path: tuple[str, ...]) -> list[int] | None:
    node: Any = coords
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if not isinstance(node, list) or len(node) != 4:
        return None
    return [int(v) for v in node]


def _crop_box(frame: np.ndarray, box: list[int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w].copy()


def _load_icon_bgr(icons_dir: Path, species_id: str) -> np.ndarray | None:
    path = icons_dir / f"{species_id}.png"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3:4] / 255.0
        white = np.full_like(bgr, 255)
        return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
    return image


def _fit_panel(image: np.ndarray, *, height: int = _PANEL_H, width: int = _PANEL_W) -> np.ndarray:
    if image.size == 0:
        return np.full((height, width, 3), 240, dtype=np.uint8)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _distance_for_species(
    species_id: str,
    ranked: list[tuple[str, float]],
) -> float | None:
    for sid, dist in ranked:
        if sid == species_id:
            return dist
    return None


def _analyze_slot(
    matcher: SpriteMatcher,
    frame: np.ndarray,
    box: list[int],
    *,
    expected: str,
    label: str,
) -> dict[str, Any]:
    crop = _crop_box(frame, box)
    analysis = matcher.rank_sprite(crop, top_n=15)
    ranked = analysis["ranked"]
    decision = analysis["decision"]
    predicted = decision.species_id if decision else "unknown"
    pred_dist = decision.distance if decision else analysis["best_distance"]
    pred_margin = decision.margin if decision else analysis["margin"]

    expected_dist = _distance_for_species(expected, ranked)
    expected_rank = next((i + 1 for i, (sid, _) in enumerate(ranked) if sid == expected), None)

    return {
        "label": label,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "crop_box": box,
        "predicted_distance": pred_dist,
        "predicted_margin": pred_margin,
        "best_raw_species": analysis["best_species_id"],
        "best_raw_distance": analysis["best_distance"],
        "best_raw_margin": analysis["margin"],
        "expected_distance": expected_dist,
        "expected_rank": expected_rank,
        "top_ranked": ranked[:5],
        "thresholds": {
            "max_distance": matcher.max_distance,
            "min_margin": matcher.min_margin,
        },
    }


def _save_mismatch_figure(
    out_path: Path,
    *,
    suite_id: str,
    slot: dict[str, Any],
    crop: np.ndarray,
    icons_dir: Path,
) -> None:
    expected = slot["expected"]
    predicted = slot["predicted"]
    expected_icon = _load_icon_bgr(icons_dir, expected)
    predicted_id = predicted if predicted != "unknown" else slot["best_raw_species"]
    predicted_icon = _load_icon_bgr(icons_dir, predicted_id) if predicted_id else None

    panels = [
        (_fit_panel(crop), f"Screenshot crop\n({slot['label']})"),
        (
            _fit_panel(expected_icon) if expected_icon is not None else _fit_panel(np.zeros((1, 1, 3), np.uint8)),
            f"Ground truth\n{expected}",
        ),
        (
            _fit_panel(predicted_icon) if predicted_icon is not None else _fit_panel(np.zeros((1, 1, 3), np.uint8)),
            f"Predicted\n{predicted}",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    exp_dist = slot["expected_distance"]
    exp_rank = slot["expected_rank"]
    lines = [
        f"{suite_id} / {slot['label']}",
        f"predicted: {predicted}  distance={slot['predicted_distance']:.4f}  margin={slot['predicted_margin']:.4f}",
        f"ground truth rank: {exp_rank if exp_rank is not None else 'not in top-15'}"
        + (f"  distance={exp_dist:.4f}" if exp_dist is not None else ""),
        f"raw best: {slot['best_raw_species']} ({slot['best_raw_distance']:.4f}, margin {slot['best_raw_margin']:.4f})",
        "top-5: " + ", ".join(f"{sid}={dist:.3f}" for sid, dist in slot["top_ranked"]),
        f"thresholds: max_dist={slot['thresholds']['max_distance']}, min_margin={slot['thresholds']['min_margin']}",
    ]
    fig.suptitle("\n".join(lines[:3]), fontsize=9, y=1.02)
    fig.text(0.5, -0.02, "\n".join(lines[3:]), ha="center", va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_debug(
    *,
    screenshot_dir: Path = _DEFAULT_SCREENSHOTS,
    output_dir: Path = _DEFAULT_OUTPUT,
    suite_ids: list[str] | None = None,
    include_correct: bool = False,
) -> dict[str, Any]:
    coords = load_ui_coordinates()
    matcher = SpriteMatcher()
    matcher.build_index()

    suites = _EVAL_SUITES
    if suite_ids:
        suites = [s for s in _EVAL_SUITES if s["id"] in suite_ids]

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "output_dir": str(output_dir.resolve()),
        "suites": [],
    }

    for suite in suites:
        image_path = screenshot_dir / suite["screenshot"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Missing screenshot: {image_path}")

        suite_dir = output_dir / suite["id"]
        suite_dir.mkdir(parents=True, exist_ok=True)
        suite_result: dict[str, Any] = {
            "id": suite["id"],
            "screenshot": suite["screenshot"],
            "slots": [],
            "mismatches": [],
        }

        for slot_def in suite["slots"]:
            box = _resolve_region(coords, tuple(slot_def["region"]))
            if box is None:
                continue
            slot = _analyze_slot(
                matcher,
                frame,
                box,
                expected=slot_def["expected"],
                label=slot_def["label"],
            )
            suite_result["slots"].append(slot)
            if slot["correct"] and not include_correct:
                continue
            if not slot["correct"]:
                suite_result["mismatches"].append(slot["label"])
                crop = _crop_box(frame, box)
                fig_name = f"{slot['label']}_exp-{slot['expected']}_got-{slot['predicted']}.png"
                _save_mismatch_figure(
                    suite_dir / fig_name,
                    suite_id=suite["id"],
                    slot=slot,
                    crop=crop,
                    icons_dir=matcher.icons_dir,
                )

        suite_json = suite_dir / "summary.json"
        suite_json.write_text(json.dumps(suite_result, indent=2), encoding="utf-8")
        summary["suites"].append(suite_result)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sprite mismatch debug figures.")
    parser.add_argument("--screenshot-dir", type=Path, default=_DEFAULT_SCREENSHOTS)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--suite", action="append", dest="suites", help="Suite id (default: all)")
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Also write figures for correct matches.",
    )
    args = parser.parse_args()

    summary = run_debug(
        screenshot_dir=args.screenshot_dir,
        output_dir=args.output_dir,
        suite_ids=args.suites,
        include_correct=args.include_correct,
    )

    print(f"Wrote debug output to {args.output_dir.resolve()}")
    for suite in summary["suites"]:
        total = len(suite["slots"])
        wrong = len(suite["mismatches"])
        print(f"  {suite['id']}: {wrong} mismatches / {total} slots")
        for label in suite["mismatches"]:
            print(f"    - {label}")


if __name__ == "__main__":
    main()
