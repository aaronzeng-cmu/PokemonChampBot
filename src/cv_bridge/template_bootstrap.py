"""Bootstrap UI anchor templates from reference screenshots."""

from __future__ import annotations

from pathlib import Path

import cv2

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SCREENSHOTS_DIR = Path(__file__).resolve().parents[2] / "logs" / "cv_bridge" / "screenshots"

# (screenshot_filename, x, y, width, height)
TEMPLATE_SPECS: dict[str, tuple[str, int, int, int, int]] = {
    "fight_button": ("20260624_232432.png", 1720, 595, 100, 100),
    "pokemon_button": ("20260624_232432.png", 1676, 812, 100, 100),
    "move_panel_anchor": ("20260619_041913.png", 1650, 300, 140, 70),
    "target_overlay_close": ("20260624_232712.png", 1330, 888, 90, 48),
    "teampreview_header": ("20260624_232331.png", 700, 80, 520, 120),
    "results_continue": ("20260624_232854.png", 1413, 940, 200, 80),
    "lobby_battle": ("20260619_041729.png", 1186, 346, 120, 100),
    "communicating_banner": ("20260624_232816.png", 760, 280, 400, 80),
}


def ensure_templates(
    templates_dir: Path | None = None,
    screenshots_dir: Path | None = None,
) -> Path:
    """Crop anchor PNGs from reference screenshots if missing."""
    out_dir = templates_dir or _TEMPLATES_DIR
    shots = screenshots_dir or _SCREENSHOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (filename, x, y, w, h) in TEMPLATE_SPECS.items():
        dest = out_dir / f"{name}.png"
        if dest.is_file():
            continue
        source = shots / filename
        if not source.is_file():
            continue
        image = cv2.imread(str(source))
        if image is None:
            continue
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        cv2.imwrite(str(dest), crop)

    return out_dir


if __name__ == "__main__":
    path = ensure_templates()
    created = sorted(p.name for p in path.glob("*.png"))
    print(f"Templates in {path}: {len(created)} files")
    for name in created:
        print(f"  {name}")
