"""EasyOCR diagnostics: per-region text image + result / confidence / latency.

For each OCR region the live pipeline reads, this renders the raw crop, the exact
preprocessed image EasyOCR sees, the recognized tokens with confidences, the
parsed pipeline value, and the call latency -- as a figure per screenshot plus a
machine-readable JSON summary.

Usage::

    python -m scripts.debug_easyocr                 # curated default screenshots
    python -m scripts.debug_easyocr --shots a.png b.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.cv_bridge.ocr_utils import (  # noqa: E402
    _HP_ALLOWLIST,
    _PERCENT_ALLOWLIST,
    get_hp_percentage_from_bar,
    parse_hp_percent,
    parse_hp_text,
    preprocess_hp_crop,
    read_text_lines,
)
from src.cv_bridge.perception import PerceptionModule  # noqa: E402

_SHOT_DIR = Path("logs/cv_bridge/screenshots")
_OUT_DIR = Path("logs/cv_bridge/analysis/easyocr_debug")

# (region_key, kind). kind drives preprocessing + how the pipeline parses it.
_REGIONS: list[tuple[str, str]] = [
    ("player_active_hp_slot_a", "hp_ally"),
    ("player_active_hp_slot_b", "hp_ally"),
    ("opp_active_hp_slot_a", "hp_enemy"),
    ("opp_active_hp_slot_b", "hp_enemy"),
    ("player_active_name_slot_a", "name"),
    ("player_active_name_slot_b", "name"),
    ("battle_action_log", "log"),
    ("ability_item_popup_left", "log"),
    ("ability_item_popup_right", "log"),
    ("teampreview_selection_counter", "name"),
    ("decision_state_move_timer", "name"),
    *[(f"force_switch_hp_{i}", "log") for i in range(1, 7)],
]

# Screenshots that exercise the different OCR consumers.
_DEFAULT_SHOTS = [
    "20260624_232432.png",  # doubles command menu: HP bars + nameplates
    "20260624_232835.png",  # battle log line
    "20260624_232439.png",  # move menu: timer
    "20260629_034106_teaminit_moves.png",  # team view (dense text)
]


def _prep(kind: str, raw, perception: PerceptionModule):
    """Return (display_image, ocr_input_rgb, allowlist) mirroring the live path."""
    if kind in ("hp_ally", "hp_enemy"):
        scale = 3
        proc = preprocess_hp_crop(raw, scale=scale)
        allow = _HP_ALLOWLIST if kind == "hp_ally" else _PERCENT_ALLOWLIST
        return proc, proc, allow
    if kind == "log":
        proc = preprocess_hp_crop(raw, scale=2)
        return proc, proc, None
    # name / counter / timer: grayscale + blur (perception.preprocess_for_ocr)
    gray = perception.preprocess_for_ocr(raw)
    return gray, gray, None


def _pipeline_result(kind: str, raw, reader) -> str:
    if kind == "hp_ally":
        r = parse_hp_text(raw, reader)
        return f"{r[0]}/{r[1]}" if r else "<none>"
    if kind == "hp_enemy":
        pct = parse_hp_percent(raw, reader)
        if pct is not None:
            return f"{pct:.0f}%"
        return f"~{get_hp_percentage_from_bar(raw) * 100:.0f}% (bar)"
    return read_text_lines(raw, reader) or "<empty>"


def _ocr_tokens(reader, ocr_input, allowlist) -> tuple[list[dict[str, Any]], float]:
    kwargs: dict[str, Any] = {"detail": 1}
    if allowlist:
        kwargs["allowlist"] = allowlist
    t0 = time.perf_counter()
    raw = reader.readtext(ocr_input, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    tokens = [{"text": str(t), "conf": float(c)} for (_box, t, c) in raw]
    return tokens, latency_ms


def _process_shot(shot: Path, perception: PerceptionModule, reader) -> dict[str, Any]:
    frame = cv2.imread(str(shot))
    if frame is None:
        return {"shot": shot.name, "error": "unreadable"}

    rows: list[dict[str, Any]] = []
    for key, kind in _REGIONS:
        raw = perception._crop_region(frame, key)
        if raw is None or raw.size == 0:
            continue
        disp, ocr_input, allow = _prep(kind, raw, perception)
        tokens, latency_ms = _ocr_tokens(reader, ocr_input, allow)
        joined = " ".join(t["text"] for t in tokens).strip()
        mean_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        rows.append(
            {
                "region": key,
                "kind": kind,
                "raw": raw,
                "disp": disp,
                "ocr_text": joined,
                "tokens": tokens,
                "mean_conf": mean_conf,
                "latency_ms": latency_ms,
                "pipeline": _pipeline_result(kind, raw, reader),
            }
        )
    return {"shot": shot.name, "rows": rows}


def _render_figure(result: dict[str, Any], out_dir: Path) -> Path | None:
    rows = result.get("rows", [])
    if not rows:
        return None
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.05 * n), squeeze=False)
    fig.suptitle(f"EasyOCR debug — {result['shot']}", fontsize=13, fontweight="bold")
    for i, row in enumerate(rows):
        ax_raw, ax_proc = axes[i][0], axes[i][1]
        ax_raw.imshow(cv2.cvtColor(row["raw"], cv2.COLOR_BGR2RGB))
        ax_raw.set_title(f"{row['region']}  (raw)", fontsize=8)
        ax_raw.axis("off")

        disp = row["disp"]
        if disp is not None and disp.ndim == 2:
            ax_proc.imshow(disp, cmap="gray")
        elif disp is not None:
            ax_proc.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        ax_proc.axis("off")
        conf_str = ", ".join(f"{t['conf']:.2f}" for t in row["tokens"]) or "-"
        ax_proc.set_title(
            f"OCR: {row['ocr_text']!r}  |  parsed: {row['pipeline']}\n"
            f"conf=[{conf_str}]  mean={row['mean_conf']:.2f}  "
            f"latency={row['latency_ms']:.0f} ms",
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"easyocr_{Path(result['shot']).stem}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="EasyOCR per-region debug figures.")
    ap.add_argument("--shots", nargs="*", default=None, help="Screenshot filenames or paths.")
    ap.add_argument("--shot-dir", type=Path, default=_SHOT_DIR)
    ap.add_argument("--out", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    shot_names = args.shots or _DEFAULT_SHOTS
    shots = [Path(s) if Path(s).is_file() else args.shot_dir / s for s in shot_names]
    shots = [s for s in shots if s.is_file()]
    if not shots:
        raise SystemExit("no readable screenshots found")

    perception = PerceptionModule(ocr_enabled=True)
    reader = perception._get_ocr_reader()
    # Warm up so the first timed call isn't skewed by lazy init / CUDA setup.
    reader.readtext(cv2.imread(str(shots[0]))[:64, :64], detail=0)

    summary: list[dict[str, Any]] = []
    for shot in shots:
        result = _process_shot(shot, perception, reader)
        fig_path = _render_figure(result, args.out)
        srows = [
            {k: r[k] for k in ("region", "kind", "ocr_text", "tokens", "mean_conf", "latency_ms", "pipeline")}
            for r in result.get("rows", [])
        ]
        summary.append({"shot": result["shot"], "figure": str(fig_path) if fig_path else None, "regions": srows})
        if fig_path:
            print(f"[easyocr] {shot.name}: {len(srows)} regions -> {fig_path}")
        else:
            print(f"[easyocr] {shot.name}: no readable regions")

    summary_dir = args.out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "easyocr_debug.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[easyocr] summary -> {summary_path}")


if __name__ == "__main__":
    main()
