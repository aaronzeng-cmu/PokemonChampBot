"""Interactive rectangle annotator for perception / sprite crop regions.

Draw boxes on screenshots, then merge into ``ui_coordinates.json``.

Usage::

    conda activate PokemonChampBot
    python -m src.cv_bridge.tools.annotate_regions list
    python -m src.cv_bridge.tools.annotate_regions draw battle_active_doubles
    python -m src.cv_bridge.tools.annotate_regions draw teampreview
    python -m src.cv_bridge.tools.annotate_regions merge
    python -m src.cv_bridge.tools.annotate_regions merge --recipe battle_active_doubles

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

_RECIPE_FILE = Path(__file__).with_name("region_recipes.json")
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
        return {"regions": {}}
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


def _norm_box(x0: int, y0: int, x1: int, y1: int) -> list[int]:
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return [left, top, max(1, right - left), max(1, bottom - top)]


def _draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], *, color: tuple[int, int, int]) -> None:
    x, y = origin
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(canvas, (x - 2, y - th - 6), (x + tw + 4, y + baseline + 2), _COLORS["bg"], -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_box(canvas: np.ndarray, box: list[int], *, color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    _draw_label(canvas, label, (x + 4, max(y + 16, 18)), color=color)


_RECIPE_ALIASES: dict[str, str] = {
    "teampreview_doubles": "teampreview",
}


def _resolve_recipe_id(recipe_id: str, by_id: dict[str, Any]) -> str:
    recipe_id = _RECIPE_ALIASES.get(recipe_id, recipe_id)
    if recipe_id not in by_id:
        known = ", ".join(by_id)
        raise SystemExit(f"Unknown recipe '{recipe_id}'. Known: {known}")
    return recipe_id


def list_recipes() -> None:
    data = _load_recipes()
    output_dir = Path(data["meta"].get("output_dir", "logs/cv_bridge/analysis/region_annotations"))
    print("Region annotation recipes:\n")
    for recipe in data["recipes"]:
        ann_path = _annotation_path(recipe["id"], output_dir=output_dir)
        saved = _load_annotation(ann_path)
        done = len(saved.get("regions", {}))
        total = len(recipe["regions"])
        status = f"{done}/{total} saved" if done else "not started"
        print(f"  {recipe['id']}")
        print(f"    screenshot: {recipe['screenshot']}")
        print(f"    {recipe['summary']}")
        print(f"    boxes: {total}  ({status})")
        print()


def _opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("__annotate_regions_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__annotate_regions_test__")
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
    annotation.setdefault("regions", {})

    regions: list[dict[str, str]] = recipe["regions"]
    index = max(0, min(start_index, len(regions) - 1))

    drag_start: tuple[int, int] | None = None
    preview_box: list[int] | None = None

    window = "annotate_regions — drag box | Enter=confirm | U=undo | S=save | Q=quit"

    def redraw() -> np.ndarray:
        canvas = frame.copy()
        for idx, spec in enumerate(regions):
            path = spec["path"]
            box = annotation["regions"].get(path)
            if box is None:
                continue
            color = _COLORS["current"] if idx == index else _COLORS["done"]
            _draw_box(canvas, box, color=color, label=f"{idx + 1}")

        if preview_box is not None:
            _draw_box(canvas, preview_box, color=_COLORS["preview"], label="preview")

        spec = regions[index]
        lines = [
            f"[{index + 1}/{len(regions)}] {spec['title']}",
            spec["hint"],
            f"key -> {spec['path']}",
            "Drag LMB: draw | Enter: next | U: undo | S: save | Q: quit",
        ]
        y = 24
        for line in lines:
            _draw_label(canvas, line, (16, y), color=_COLORS["text"])
            y += 22
        return canvas

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal drag_start, preview_box
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            preview_box = None
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            preview_box = _norm_box(drag_start[0], drag_start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            preview_box = _norm_box(drag_start[0], drag_start[1], x, y)
            drag_start = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse)

    print(f"Recipe: {recipe_id}")
    print(f"Image:  {image_path.resolve()}")
    print(f"Save:   {ann_path.resolve()}")
    print("Draw a rectangle for each prompt, then press Enter to confirm and advance.\n")

    while True:
        cv2.imshow(window, redraw())
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            annotation["updated_at_utc"] = _utc_now()
            _save_annotation(ann_path, annotation)
            print(f"Saved {ann_path}")
        elif key in (13, 10):  # Enter
            if preview_box is None:
                print("Draw a box first (drag with left mouse button).")
                continue
            path = regions[index]["path"]
            annotation["regions"][path] = preview_box
            print(f"  {path} = {preview_box}")
            preview_box = None
            if index < len(regions) - 1:
                index += 1
                print(f"Next: {regions[index]['title']}")
            else:
                annotation["updated_at_utc"] = _utc_now()
                _save_annotation(ann_path, annotation)
                print(f"Recipe complete. Saved {ann_path}")
        elif key in (ord("u"), ord("U")):
            if preview_box is not None:
                preview_box = None
                print("Cleared preview box.")
                continue
            path = regions[index]["path"]
            if path in annotation["regions"]:
                del annotation["regions"][path]
                print(f"Removed box for {path}")
            elif index > 0:
                index -= 1
                path = regions[index]["path"]
                if path in annotation["regions"]:
                    del annotation["regions"][path]
                print(f"Back to: {regions[index]['title']}")
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
    annotation.setdefault("regions", {})

    regions: list[dict[str, str]] = recipe["regions"]
    state: dict[str, Any] = {
        "index": max(0, min(start_index, len(regions) - 1)),
        "drag_start": None,
        "preview_box": None,
        "rect_ids": [],
    }

    print(f"Recipe: {recipe_id}")
    print(f"Image:  {image_path.resolve()}")
    print(f"Save:   {ann_path.resolve()}")
    print("Draw a rectangle for each prompt, then press Enter to confirm and advance.\n")

    root = tk.Tk()
    root.title(f"annotate_regions — {recipe_id}")
    root.geometry("1320x860")

    header = ttk.Frame(root, padding=8)
    header.pack(fill=tk.X)
    title_var = tk.StringVar()
    hint_var = tk.StringVar()
    path_var = tk.StringVar()
    help_var = tk.StringVar(
        value="Drag LMB: draw box | Enter: confirm | U: undo | S: save | Q: quit"
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
    canvas.image = photo  # keep reference

    def to_image_coords(cx: int, cy: int) -> tuple[int, int]:
        return int(round(cx / scale)), int(round(cy / scale))

    def to_canvas_rect(box: list[int]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        return (
            int(round(x * scale)),
            int(round(y * scale)),
            int(round((x + w) * scale)),
            int(round((y + h) * scale)),
        )

    def clear_rects() -> None:
        for rect_id in state["rect_ids"]:
            canvas.delete(rect_id)
        state["rect_ids"].clear()

    def add_rect(box: list[int], *, outline: str, width: int = 2, dash: tuple[int, ...] | None = None) -> None:
        x0, y0, x1, y1 = to_canvas_rect(box)
        kwargs: dict[str, Any] = {"outline": outline, "width": width}
        if dash:
            kwargs["dash"] = dash
        rect_id = canvas.create_rectangle(x0, y0, x1, y1, **kwargs)
        state["rect_ids"].append(rect_id)

    def refresh_header() -> None:
        spec = regions[state["index"]]
        title_var.set(f"[{state['index'] + 1}/{len(regions)}] {spec['title']}")
        hint_var.set(spec["hint"])
        path_var.set(spec["path"])

    def redraw_canvas() -> None:
        clear_rects()
        for idx, spec in enumerate(regions):
            box = annotation["regions"].get(spec["path"])
            if not box:
                continue
            color = "#00dcff" if idx == state["index"] else "#50dc50"
            add_rect(box, outline=color)

        if state["preview_box"] is not None:
            add_rect(state["preview_box"], outline="#ffb400", width=2, dash=(6, 4))

    def save_annotation() -> None:
        annotation["updated_at_utc"] = _utc_now()
        _save_annotation(ann_path, annotation)
        print(f"Saved {ann_path}")

    def confirm_box() -> None:
        if state["preview_box"] is None:
            print("Draw a box first (drag with left mouse button).")
            return
        path = regions[state["index"]]["path"]
        annotation["regions"][path] = state["preview_box"]
        print(f"  {path} = {state['preview_box']}")
        state["preview_box"] = None
        if state["index"] < len(regions) - 1:
            state["index"] += 1
            print(f"Next: {regions[state['index']]['title']}")
        else:
            save_annotation()
            print(f"Recipe complete. Saved {ann_path}")
        refresh_header()
        redraw_canvas()

    def undo_box() -> None:
        if state["preview_box"] is not None:
            state["preview_box"] = None
            print("Cleared preview box.")
            redraw_canvas()
            return
        path = regions[state["index"]]["path"]
        if path in annotation["regions"]:
            del annotation["regions"][path]
            print(f"Removed box for {path}")
        elif state["index"] > 0:
            state["index"] -= 1
            path = regions[state["index"]]["path"]
            if path in annotation["regions"]:
                del annotation["regions"][path]
            print(f"Back to: {regions[state['index']]['title']}")
        else:
            print("Nothing to undo.")
        refresh_header()
        redraw_canvas()

    def on_press(event: tk.Event) -> None:
        state["drag_start"] = to_image_coords(event.x, event.y)
        state["preview_box"] = None
        redraw_canvas()

    def on_drag(event: tk.Event) -> None:
        if state["drag_start"] is None:
            return
        x1, y1 = state["drag_start"]
        x2, y2 = to_image_coords(event.x, event.y)
        state["preview_box"] = _norm_box(x1, y1, x2, y2)
        redraw_canvas()

    def on_release(event: tk.Event) -> None:
        if state["drag_start"] is None:
            return
        x1, y1 = state["drag_start"]
        x2, y2 = to_image_coords(event.x, event.y)
        state["preview_box"] = _norm_box(x1, y1, x2, y2)
        state["drag_start"] = None
        redraw_canvas()

    def on_key(event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ("q", "escape"):
            save_annotation()
            root.destroy()
        elif key == "return":
            confirm_box()
        elif key == "u":
            undo_box()
        elif key == "s":
            save_annotation()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Key>", on_key)

    refresh_header()
    redraw_canvas()
    root.mainloop()
    print(f"Wrote {ann_path}")


def merge_annotations(
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
            and _load_annotation(_annotation_path(recipe["id"], output_dir=output_dir)).get("regions")
        ]
        require_files = False
        if not recipes:
            print(f"No saved annotations in {output_dir.resolve()}")
            print("Run: python -m src.cv_bridge.tools.annotate_regions draw <recipe_id>")
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
                    f"Run: python -m src.cv_bridge.tools.annotate_regions draw {recipe['id']}"
                )
            skipped.append(recipe["id"])
            continue
        annotation = _load_annotation(ann_path)
        regions = annotation.get("regions", {})
        if not regions:
            if require_files:
                raise SystemExit(f"Annotation file for '{recipe['id']}' has no regions: {ann_path}")
            skipped.append(recipe["id"])
            continue
        print(f"\n{recipe['id']}:")
        for path, box in regions.items():
            _set_nested(coords, path, box)
            merged.append({"recipe": recipe["id"], "path": path, "box": box})
            print(f"  {path} = {box}")

    if skipped and not require_files:
        print(f"\nIgnored {len(skipped)} empty/missing file(s): {', '.join(skipped)}")

    coords_path.write_text(json.dumps(coords, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(
        json.dumps({"merged_at_utc": _utc_now(), "entries": merged}, indent=2),
        encoding="utf-8",
    )
    print(f"\nUpdated {coords_path.resolve()} ({len(merged)} regions)")


def preview_recipe(
    recipe_id: str,
    *,
    screenshot_dir: Path,
    output_dir: Path,
    coords_path: Path,
) -> Path:
    """Render current saved + ui_coordinates boxes for a recipe (no GUI)."""
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
    for idx, spec in enumerate(recipe["regions"], start=1):
        path = spec["path"]
        box = ann.get("regions", {}).get(path) or _get_nested(coords, path)
        if not box:
            continue
        _draw_box(canvas, box, color=_COLORS["done"], label=f"{idx}: {spec['title']}")

    out = output_dir / f"preview_{recipe_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw perception crop regions on screenshots.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List annotation recipes and progress.")

    draw = sub.add_parser("draw", help="Interactive box drawing for one recipe.")
    draw.add_argument("recipe_id", help="Recipe id from region_recipes.json")
    draw.add_argument("--screenshot-dir", type=Path, default=_DEFAULT_SCREENSHOTS)
    draw.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/region_annotations"),
    )
    draw.add_argument("--start", type=int, default=0, help="Region index to start at (0-based).")

    merge = sub.add_parser("merge", help="Merge saved annotations into ui_coordinates.json.")
    merge.add_argument("--coords", type=Path, default=_DEFAULT_COORDS)
    merge.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/region_annotations"),
    )
    merge.add_argument(
        "--recipe",
        action="append",
        dest="recipes",
        help="Merge only these recipe ids (errors if not annotated yet). Default: all saved files.",
    )

    preview = sub.add_parser("preview", help="PNG preview of saved/coords boxes.")
    preview.add_argument("recipe_id")
    preview.add_argument("--screenshot-dir", type=Path, default=_DEFAULT_SCREENSHOTS)
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/cv_bridge/analysis/region_annotations"),
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
