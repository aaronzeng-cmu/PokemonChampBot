"""Interactive tap-point annotator for battle action coordinates.

Click targets on screenshots, then merge into ``ui_coordinates.json``.

Usage::

    conda activate PokemonChampBot
    python -m src.cv_bridge.tools.annotate_actions list
    python -m src.cv_bridge.tools.annotate_actions draw battle_move_menu
    python -m src.cv_bridge.tools.annotate_actions draw battle_target_overlay
    python -m src.cv_bridge.tools.annotate_actions merge
    python -m src.cv_bridge.tools.annotate_actions preview battle_move_menu

Uses Tkinter when OpenCV GUI is unavailable (e.g. opencv-python-headless).
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

_RECIPE_FILE = Path(__file__).with_name("action_recipes.json")
_DEFAULT_COORDS = Path(__file__).resolve().parents[1] / "ui_coordinates.json"
_DEFAULT_SCREENSHOTS = Path("logs/cv_bridge/screenshots")

_COLORS = {
    "pending": (80, 80, 80),
    "current": (0, 220, 255),
    "done": (80, 220, 80),
    "preview": (255, 180, 0),
    "text": (255, 255, 255),
    "bg": (24, 24, 24),
}

_MARKER_RADIUS = 8
_CROSSHAIR_LEN = 14


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_recipes() -> dict[str, Any]:
    data = json.loads(_RECIPE_FILE.read_text(encoding="utf-8"))
    by_id = {recipe["id"]: recipe for recipe in data["recipes"]}
    return {"meta": data.get("_meta", {}), "by_id": by_id, "recipes": data["recipes"]}


def _annotation_path(recipe_id: str, *, output_dir: Path) -> Path:
    return output_dir / f"{recipe_id}.json"


def _load_annotation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"points": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_annotation(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _set_nested(data: dict[str, Any], dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    node = data
    for key in parts[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dot_path}: {key} is not a mapping")
        node = child
    node[parts[-1]] = value


def _get_nested(data: dict[str, Any], dot_path: str) -> Any:
    node: Any = data
    for key in dot_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _norm_point(x: int, y: int) -> list[int]:
    return [int(x), int(y)]


def _draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], *, color: tuple[int, int, int]) -> None:
    x, y = origin
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(canvas, (x - 2, y - th - 6), (x + tw + 4, y + baseline + 2), _COLORS["bg"], -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_point(
    canvas: np.ndarray,
    point: list[int],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = point
    cv2.circle(canvas, (x, y), _MARKER_RADIUS, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x - _CROSSHAIR_LEN, y), (x + _CROSSHAIR_LEN, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - _CROSSHAIR_LEN), (x, y + _CROSSHAIR_LEN), color, 1, cv2.LINE_AA)
    _draw_label(canvas, label, (x + 12, max(y - 8, 18)), color=color)


def _resolve_recipe_id(recipe_id: str, by_id: dict[str, Any]) -> str:
    if recipe_id not in by_id:
        known = ", ".join(by_id)
        raise SystemExit(f"Unknown recipe '{recipe_id}'. Known: {known}")
    return recipe_id


def list_recipes() -> None:
    data = _load_recipes()
    output_dir = Path(data["meta"].get("output_dir", "logs/cv_bridge/analysis/action_annotations"))
    print("Battle action tap recipes:\n")
    for recipe in data["recipes"]:
        ann_path = _annotation_path(recipe["id"], output_dir=output_dir)
        saved = _load_annotation(ann_path)
        done = len(saved.get("points", {}))
        total = len(recipe["points"])
        status = f"{done}/{total} saved" if done else "not started"
        print(f"  {recipe['id']}")
        print(f"    screenshot: {recipe['screenshot']}")
        print(f"    {recipe['summary']}")
        print(f"    taps: {total}  ({status})")
        print()


def _opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("__annotate_actions_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__annotate_actions_test__")
        return True
    except cv2.error:
        return False


def draw_recipe(
    recipe_id: str,
    *,
    screenshot_dir: Path,
    output_dir: Path,
    start_index: int = 0,
) -> None:
    if _opencv_gui_available():
        _draw_recipe_opencv(
            recipe_id,
            screenshot_dir=screenshot_dir,
            output_dir=output_dir,
            start_index=start_index,
        )
    else:
        print("OpenCV GUI unavailable (headless build); using Tkinter window.")
        _draw_recipe_tk(
            recipe_id,
            screenshot_dir=screenshot_dir,
            output_dir=output_dir,
            start_index=start_index,
        )


def _draw_recipe_opencv(
    recipe_id: str,
    *,
    screenshot_dir: Path,
    output_dir: Path,
    start_index: int = 0,
) -> None:
    data = _load_recipes()
    recipe_id = _resolve_recipe_id(recipe_id, data["by_id"])
    recipe = data["by_id"][recipe_id]

    image_path = screenshot_dir / recipe["screenshot"]
    if not image_path.exists():
        raise SystemExit(f"Screenshot not found: {image_path.resolve()}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    ann_path = _annotation_path(recipe_id, output_dir=output_dir)
    annotation = _load_annotation(ann_path)
    annotation.setdefault("recipe_id", recipe_id)
    annotation.setdefault("screenshot", recipe["screenshot"])
    annotation.setdefault("points", {})

    points: list[dict[str, str]] = recipe["points"]
    index = max(0, min(start_index, len(points) - 1))
    preview_point: list[int] | None = None

    window = "annotate_actions — click point | Enter=confirm | U=undo | S=save | Q=quit"

    def redraw() -> np.ndarray:
        canvas = frame.copy()
        for idx, spec in enumerate(points):
            path = spec["path"]
            pt = annotation["points"].get(path)
            if pt is None:
                continue
            color = _COLORS["current"] if idx == index else _COLORS["done"]
            _draw_point(canvas, pt, color=color, label=f"{idx + 1}")

        if preview_point is not None:
            _draw_point(canvas, preview_point, color=_COLORS["preview"], label="preview")

        spec = points[index]
        lines = [
            f"[{index + 1}/{len(points)}] {spec['title']}",
            spec["hint"],
            f"key -> {spec['path']}",
            "Click LMB: place | Enter: next | U: undo | S: save | Q: quit",
        ]
        y = 24
        for line in lines:
            _draw_label(canvas, line, (16, y), color=_COLORS["text"])
            y += 22
        return canvas

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal preview_point
        if event == cv2.EVENT_LBUTTONDOWN:
            preview_point = _norm_point(x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse)

    print(f"Recipe: {recipe_id}")
    print(f"Image:  {image_path.resolve()}")
    print(f"Save:   {ann_path.resolve()}")
    print("Click each target, then press Enter to confirm and advance.\n")

    while True:
        cv2.imshow(window, redraw())
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            annotation["updated_at_utc"] = _utc_now()
            _save_annotation(ann_path, annotation)
            print(f"Saved {ann_path}")
        elif key in (13, 10):
            if preview_point is None:
                print("Click a point first (left mouse button).")
                continue
            path = points[index]["path"]
            annotation["points"][path] = preview_point
            print(f"  {path} = {preview_point}")
            preview_point = None
            if index < len(points) - 1:
                index += 1
                print(f"Next: {points[index]['title']}")
            else:
                annotation["updated_at_utc"] = _utc_now()
                _save_annotation(ann_path, annotation)
                print(f"Recipe complete. Saved {ann_path}")
        elif key in (ord("u"), ord("U")):
            if preview_point is not None:
                preview_point = None
                print("Cleared preview point.")
                continue
            path = points[index]["path"]
            if path in annotation["points"]:
                del annotation["points"][path]
                print(f"Removed point for {path}")
            elif index > 0:
                index -= 1
                path = points[index]["path"]
                if path in annotation["points"]:
                    del annotation["points"][path]
                print(f"Back to: {points[index]['title']}")
            else:
                print("Nothing to undo.")

    annotation["updated_at_utc"] = _utc_now()
    _save_annotation(ann_path, annotation)
    cv2.destroyAllWindows()
    print(f"Wrote {ann_path}")


def _draw_recipe_tk(
    recipe_id: str,
    *,
    screenshot_dir: Path,
    output_dir: Path,
    start_index: int = 0,
) -> None:
    data = _load_recipes()
    recipe_id = _resolve_recipe_id(recipe_id, data["by_id"])
    recipe = data["by_id"][recipe_id]

    image_path = screenshot_dir / recipe["screenshot"]
    if not image_path.exists():
        raise SystemExit(f"Screenshot not found: {image_path.resolve()}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    img_h, img_w = frame.shape[:2]
    ann_path = _annotation_path(recipe_id, output_dir=output_dir)
    annotation = _load_annotation(ann_path)
    annotation.setdefault("recipe_id", recipe_id)
    annotation.setdefault("screenshot", recipe["screenshot"])
    annotation.setdefault("points", {})

    points: list[dict[str, str]] = recipe["points"]
    state: dict[str, Any] = {
        "index": max(0, min(start_index, len(points) - 1)),
        "preview_point": None,
        "marker_ids": [],
    }

    print(f"Recipe: {recipe_id}")
    print(f"Image:  {image_path.resolve()}")
    print(f"Save:   {ann_path.resolve()}")
    print("Click each target, then press Enter to confirm and advance.\n")

    root = tk.Tk()
    root.title(f"annotate_actions — {recipe_id}")
    root.geometry("1320x860")

    header = ttk.Frame(root, padding=8)
    header.pack(fill=tk.X)
    title_var = tk.StringVar()
    hint_var = tk.StringVar()
    path_var = tk.StringVar()
    help_var = tk.StringVar(
        value="Click LMB: place point | Enter: confirm | U: undo | S: save | Q: quit"
    )
    ttk.Label(header, textvariable=title_var, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
    ttk.Label(header, textvariable=hint_var, wraplength=1280).pack(anchor=tk.W)
    ttk.Label(header, textvariable=path_var, foreground="#666666").pack(anchor=tk.W)
    ttk.Label(header, textvariable=help_var).pack(anchor=tk.W, pady=(6, 0))

    canvas_frame = ttk.Frame(root, padding=8)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    max_w, max_h = 1280, 720
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)

    canvas = tk.Canvas(canvas_frame, width=disp_w, height=disp_h, highlightthickness=1, highlightbackground="#444")
    canvas.pack()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="bg")
    canvas.image = photo

    def to_image_coords(cx: int, cy: int) -> tuple[int, int]:
        return int(round(cx / scale)), int(round(cy / scale))

    def clear_markers() -> None:
        for marker_id in state["marker_ids"]:
            canvas.delete(marker_id)
        state["marker_ids"].clear()

    def add_marker(point: list[int], *, outline: str, label: str | None = None) -> None:
        x, y = point
        cx, cy = int(round(x * scale)), int(round(y * scale))
        r = max(4, int(round(_MARKER_RADIUS * scale)))
        h = max(6, int(round(_CROSSHAIR_LEN * scale)))
        state["marker_ids"].append(canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=outline, width=2))
        state["marker_ids"].append(canvas.create_line(cx - h, cy, cx + h, cy, fill=outline, width=1))
        state["marker_ids"].append(canvas.create_line(cx, cy - h, cx, cy + h, fill=outline, width=1))
        if label:
            state["marker_ids"].append(
                canvas.create_text(cx + 12, cy - 8, text=label, fill=outline, anchor=tk.NW, font=("Segoe UI", 9))
            )

    def refresh_header() -> None:
        spec = points[state["index"]]
        title_var.set(f"[{state['index'] + 1}/{len(points)}] {spec['title']}")
        hint_var.set(spec["hint"])
        path_var.set(spec["path"])

    def redraw_canvas() -> None:
        clear_markers()
        for idx, spec in enumerate(points):
            pt = annotation["points"].get(spec["path"])
            if not pt:
                continue
            color = "#00dcff" if idx == state["index"] else "#50dc50"
            add_marker(pt, outline=color, label=str(idx + 1))

        if state["preview_point"] is not None:
            add_marker(state["preview_point"], outline="#ffb400", label="preview")

    def save_annotation() -> None:
        annotation["updated_at_utc"] = _utc_now()
        _save_annotation(ann_path, annotation)
        print(f"Saved {ann_path}")

    def confirm_point() -> None:
        if state["preview_point"] is None:
            print("Click a point first (left mouse button).")
            return
        path = points[state["index"]]["path"]
        annotation["points"][path] = state["preview_point"]
        print(f"  {path} = {state['preview_point']}")
        state["preview_point"] = None
        if state["index"] < len(points) - 1:
            state["index"] += 1
            print(f"Next: {points[state['index']]['title']}")
        else:
            save_annotation()
            print(f"Recipe complete. Saved {ann_path}")
        refresh_header()
        redraw_canvas()

    def undo_point() -> None:
        if state["preview_point"] is not None:
            state["preview_point"] = None
            print("Cleared preview point.")
            redraw_canvas()
            return
        path = points[state["index"]]["path"]
        if path in annotation["points"]:
            del annotation["points"][path]
            print(f"Removed point for {path}")
        elif state["index"] > 0:
            state["index"] -= 1
            path = points[state["index"]]["path"]
            if path in annotation["points"]:
                del annotation["points"][path]
            print(f"Back to: {points[state['index']]['title']}")
        else:
            print("Nothing to undo.")
        refresh_header()
        redraw_canvas()

    def on_click(event: tk.Event) -> None:
        x, y = to_image_coords(event.x, event.y)
        state["preview_point"] = _norm_point(x, y)
        redraw_canvas()

    def on_key(event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ("q", "escape"):
            save_annotation()
            root.destroy()
        elif key == "return":
            confirm_point()
        elif key == "u":
            undo_point()
        elif key == "s":
            save_annotation()

    canvas.bind("<ButtonPress-1>", on_click)
    root.bind("<Key>", on_key)

    refresh_header()
    redraw_canvas()
    root.mainloop()
    print(f"Wrote {ann_path}")


_LEGACY_SWITCH_POPUP_PATHS = {
    "shared.battle.switch.switch_in": "shared.battle.switch.popup_offset.switch_in",
    "shared.battle.switch.cancel": "shared.battle.switch.popup_offset.cancel",
}


def _looks_absolute_popup_click(point: list[int]) -> bool:
    """Distinguish annotated screen clicks from already-merged offset pairs."""
    return int(point[0]) > 280 or int(point[1]) > 400


def _normalize_merge_point(
    recipe: dict[str, Any],
    path: str,
    point: list[int],
    coords: dict[str, Any],
) -> tuple[str, list[int]]:
    """Convert absolute popup clicks to offsets when recipe uses reference_point."""
    path = _LEGACY_SWITCH_POPUP_PATHS.get(path, path)
    ref_path = recipe.get("reference_point")
    if ref_path and ".popup_offset." in path:
        if not _looks_absolute_popup_click(point):
            return path, [int(point[0]), int(point[1])]
        ref = _get_nested(coords, ref_path)
        if ref is None:
            raise SystemExit(
                f"Recipe '{recipe['id']}' needs reference point {ref_path} in ui_coordinates.json"
            )
        ref_x, ref_y = int(ref[0]), int(ref[1])
        return path, [int(point[0]) - ref_x, int(point[1]) - ref_y]
    return path, point


def merge_annotations(  # noqa: C901
    *,
    coords_path: Path,
    output_dir: Path,
    recipe_ids: list[str] | None = None,
) -> None:
    data = _load_recipes()
    by_id = data["by_id"]

    if recipe_ids:
        unknown = [rid for rid in recipe_ids if rid not in by_id]
        if unknown:
            known = ", ".join(by_id)
            raise SystemExit(f"Unknown recipe(s): {', '.join(unknown)}. Known: {known}")
        recipes = [by_id[rid] for rid in recipe_ids]
        require_files = True
    else:
        recipes = [
            recipe
            for recipe in data["recipes"]
            if _annotation_path(recipe["id"], output_dir=output_dir).exists()
            and _load_annotation(_annotation_path(recipe["id"], output_dir=output_dir)).get("points")
        ]
        require_files = False
        if not recipes:
            print(f"No saved annotations in {output_dir.resolve()}")
            print("Run: python -m src.cv_bridge.tools.annotate_actions draw <recipe_id>")
            return
        ids = ", ".join(r["id"] for r in recipes)
        print(f"Merging {len(recipes)} recipe(s) with saved annotations: {ids}")

    coords = json.loads(coords_path.read_text(encoding="utf-8"))
    merged: list[dict[str, Any]] = []
    skipped: list[str] = []

    for recipe in recipes:
        ann_path = _annotation_path(recipe["id"], output_dir=output_dir)
        if not ann_path.exists():
            if require_files:
                raise SystemExit(
                    f"No annotation file for '{recipe['id']}'. "
                    f"Run: python -m src.cv_bridge.tools.annotate_actions draw {recipe['id']}"
                )
            skipped.append(recipe["id"])
            continue
        annotation = _load_annotation(ann_path)
        points = annotation.get("points", {})
        if not points:
            if require_files:
                raise SystemExit(f"Annotation file for '{recipe['id']}' has no points: {ann_path}")
            skipped.append(recipe["id"])
            continue
        print(f"\n{recipe['id']}:")
        for path, point in points.items():
            path, point = _normalize_merge_point(recipe, path, point, coords)
            _set_nested(coords, path, point)
            merged.append({"recipe": recipe["id"], "path": path, "point": point})
            print(f"  {path} = {point}")

    if skipped and not require_files:
        print(f"\nIgnored {len(skipped)} empty/missing file(s): {', '.join(skipped)}")

    coords_path.write_text(json.dumps(coords, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(
        json.dumps({"merged_at_utc": _utc_now(), "entries": merged}, indent=2),
        encoding="utf-8",
    )
    print(f"\nUpdated {coords_path.resolve()} ({len(merged)} tap points)")


def preview_recipe(
    recipe_id: str,
    *,
    screenshot_dir: Path,
    output_dir: Path,
    coords_path: Path,
) -> Path:
    """Render saved + ui_coordinates tap points for a recipe (no GUI)."""
    data = _load_recipes()
    recipe_id = _resolve_recipe_id(recipe_id, data["by_id"])
    recipe = data["by_id"][recipe_id]
    image_path = screenshot_dir / recipe["screenshot"]
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Cannot read {image_path}")

    coords = json.loads(coords_path.read_text(encoding="utf-8"))
    ann = _load_annotation(_annotation_path(recipe_id, output_dir=output_dir))

    canvas = frame.copy()
    for idx, spec in enumerate(recipe["points"], start=1):
        path = spec["path"]
        point = ann.get("points", {}).get(path) or _get_nested(coords, path)
        if not point or not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        _draw_point(canvas, [int(point[0]), int(point[1])], color=_COLORS["done"], label=f"{idx}: {spec['title']}")

    out = output_dir / f"preview_{recipe_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Click battle action tap coordinates on screenshots.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List action recipes and progress.")

    draw = sub.add_parser("draw", help="Interactive point placement for one recipe.")
    draw.add_argument("recipe_id", help="Recipe id from action_recipes.json")
    draw.add_argument("--screenshot-dir", type=Path, default=_DEFAULT_SCREENSHOTS)
    draw.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/action_annotations"),
    )
    draw.add_argument("--start", type=int, default=0, help="Point index to start at (0-based).")

    merge = sub.add_parser("merge", help="Merge saved annotations into ui_coordinates.json.")
    merge.add_argument("--coords", type=Path, default=_DEFAULT_COORDS)
    merge.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/action_annotations"),
    )
    merge.add_argument(
        "--recipe",
        action="append",
        dest="recipes",
        help="Merge only these recipe ids (errors if not annotated yet). Default: all saved files.",
    )

    preview = sub.add_parser("preview", help="PNG preview of saved/coords tap points.")
    preview.add_argument("recipe_id")
    preview.add_argument("--screenshot-dir", type=Path, default=_DEFAULT_SCREENSHOTS)
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/action_annotations"),
    )
    preview.add_argument("--coords", type=Path, default=_DEFAULT_COORDS)

    args = parser.parse_args()

    if args.command == "list":
        list_recipes()
    elif args.command == "draw":
        draw_recipe(
            args.recipe_id,
            screenshot_dir=args.screenshot_dir,
            output_dir=args.output_dir,
            start_index=args.start,
        )
    elif args.command == "merge":
        merge_annotations(
            coords_path=args.coords,
            output_dir=args.output_dir,
            recipe_ids=args.recipes,
        )
    elif args.command == "preview":
        out = preview_recipe(
            args.recipe_id,
            screenshot_dir=args.screenshot_dir,
            output_dir=args.output_dir,
            coords_path=args.coords,
        )
        print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
