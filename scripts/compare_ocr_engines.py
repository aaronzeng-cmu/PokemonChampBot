"""EasyOCR vs. Tesseract: per-region text / confidence / latency comparison.

For every OCR region the live pipeline reads, this runs BOTH engines on the same
crop and renders a 3-column figure (raw | EasyOCR | Tesseract) per screenshot,
plus a machine-readable JSON summary with aggregate latency and agreement so the
trade-off (speed vs. accuracy) is easy to inspect.

Usage::

    python -m scripts.compare_ocr_engines                 # curated default shots
    python -m scripts.compare_ocr_engines --shots a.png b.png
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

from scripts.debug_easyocr import (  # noqa: E402
    _DEFAULT_SHOTS,
    _REGIONS,
    _SHOT_DIR,
    _ocr_tokens as _easyocr_tokens,
    _pipeline_result as _easyocr_pipeline,
    _prep as _easyocr_prep,
)
from src.cv_bridge import tesseract_ocr as tess  # noqa: E402
from src.cv_bridge.perception import PerceptionModule  # noqa: E402

_OUT_DIR = Path("logs/cv_bridge/analysis/ocr_compare")

# Per-region Tesseract config (psm + char whitelist) mirroring how each consumer
# would read that region.
_TESS_CFG: dict[str, dict[str, Any]] = {
    "hp_ally": {"psm": tess.PSM_SINGLE_LINE, "whitelist": "0123456789/"},
    "hp_enemy": {"psm": tess.PSM_SINGLE_LINE, "whitelist": "0123456789%"},
    "name": {"psm": tess.PSM_SINGLE_LINE, "whitelist": None},
    "log": {"psm": tess.PSM_BLOCK, "whitelist": None},
}


def _mean_conf(tokens: list[dict[str, Any]]) -> float:
    return sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0


def _tess_pipeline(kind: str, raw) -> str:
    if kind == "hp_ally":
        r = tess.parse_hp_text(raw)
        return f"{r[0]}/{r[1]}" if r else "<none>"
    if kind == "hp_enemy":
        pct = tess.parse_hp_percent(raw)
        return f"{pct:.0f}%" if pct is not None else "<none>"
    cfg = _TESS_CFG.get(kind, _TESS_CFG["log"])
    return tess.ocr_text(raw, **cfg) or "<empty>"


def _process_shot(shot: Path, perception: PerceptionModule, reader) -> dict[str, Any]:
    frame = cv2.imread(str(shot))
    if frame is None:
        return {"shot": shot.name, "error": "unreadable"}

    rows: list[dict[str, Any]] = []
    for key, kind in _REGIONS:
        raw = perception._crop_region(frame, key)
        if raw is None or raw.size == 0:
            continue

        # --- EasyOCR (live pipeline) ---
        e_disp, e_input, e_allow = _easyocr_prep(kind, raw, perception)
        e_tokens, e_latency = _easyocr_tokens(reader, e_input, e_allow)
        e_text = " ".join(t["text"] for t in e_tokens).strip()
        e_pipe = _easyocr_pipeline(kind, raw, reader)

        # --- Tesseract (new pipeline) ---
        cfg = _TESS_CFG.get(kind, _TESS_CFG["log"])
        t_disp = tess.preprocess_for_tesseract(raw)
        t_tokens, t_latency = tess.ocr_tokens(t_disp, **cfg)
        t_text = " ".join(t["text"] for t in t_tokens).strip()
        t_pipe = _tess_pipeline(kind, raw)

        rows.append(
            {
                "region": key,
                "kind": kind,
                "raw": raw,
                "easy_disp": e_disp,
                "easy_text": e_text,
                "easy_tokens": e_tokens,
                "easy_conf": _mean_conf(e_tokens),
                "easy_latency_ms": e_latency,
                "easy_pipeline": e_pipe,
                "tess_disp": t_disp,
                "tess_text": t_text,
                "tess_tokens": t_tokens,
                "tess_conf": _mean_conf(t_tokens),
                "tess_latency_ms": t_latency,
                "tess_pipeline": t_pipe,
                "agree": _easy_norm(e_pipe) == _easy_norm(t_pipe),
            }
        )
    return {"shot": shot.name, "rows": rows}


def _easy_norm(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _render_figure(result: dict[str, Any], out_dir: Path) -> Path | None:
    rows = result.get("rows", [])
    if not rows:
        return None
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.1 * n), squeeze=False)
    fig.suptitle(
        f"OCR engine comparison — {result['shot']}  (raw | EasyOCR | Tesseract)",
        fontsize=13,
        fontweight="bold",
    )
    for i, row in enumerate(rows):
        ax_raw, ax_easy, ax_tess = axes[i]
        ax_raw.imshow(cv2.cvtColor(row["raw"], cv2.COLOR_BGR2RGB))
        ax_raw.set_title(f"{row['region']}  (raw)", fontsize=8)
        ax_raw.axis("off")

        for ax, disp, prefix, text, pipe, conf, latency in (
            (ax_easy, row["easy_disp"], "EasyOCR", row["easy_text"], row["easy_pipeline"], row["easy_conf"], row["easy_latency_ms"]),
            (ax_tess, row["tess_disp"], "Tesseract", row["tess_text"], row["tess_pipeline"], row["tess_conf"], row["tess_latency_ms"]),
        ):
            if disp is not None and disp.ndim == 2:
                ax.imshow(disp, cmap="gray")
            elif disp is not None:
                ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(
                f"{prefix}: {text!r}  |  parsed: {pipe}\n"
                f"mean_conf={conf:.2f}  latency={latency:.0f} ms",
                fontsize=8,
            )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ocr_compare_{Path(result['shot']).stem}.png"
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return path


def _aggregate(summary: list[dict[str, Any]]) -> dict[str, Any]:
    easy_lat: list[float] = []
    tess_lat: list[float] = []
    agree = 0
    total = 0
    for shot in summary:
        for r in shot.get("regions", []):
            easy_lat.append(r["easy_latency_ms"])
            tess_lat.append(r["tess_latency_ms"])
            total += 1
            agree += 1 if r["agree"] else 0

    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean_ms": 0.0, "max_ms": 0.0, "total_ms": 0.0}
        return {
            "mean_ms": round(sum(xs) / len(xs), 1),
            "max_ms": round(max(xs), 1),
            "total_ms": round(sum(xs), 1),
        }

    return {
        "regions_compared": total,
        "pipeline_agreement": round(agree / total, 3) if total else None,
        "easyocr_latency": stats(easy_lat),
        "tesseract_latency": stats(tess_lat),
        "speedup_x": round(sum(easy_lat) / sum(tess_lat), 2) if sum(tess_lat) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="EasyOCR vs Tesseract per-region comparison.")
    ap.add_argument("--shots", nargs="*", default=None, help="Screenshot filenames or paths.")
    ap.add_argument("--shot-dir", type=Path, default=_SHOT_DIR)
    ap.add_argument("--out", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    if not tess.tesseract_available():
        raise SystemExit(
            "Tesseract not available. Install: conda install -c conda-forge tesseract "
            "&& pip install pytesseract"
        )

    shot_names = args.shots or _DEFAULT_SHOTS
    shots = [Path(s) if Path(s).is_file() else args.shot_dir / s for s in shot_names]
    shots = [s for s in shots if s.is_file()]
    if not shots:
        raise SystemExit("no readable screenshots found")

    perception = PerceptionModule(ocr_enabled=True)
    reader = perception._get_ocr_reader()
    # Warm both engines so the first timed call isn't skewed by lazy init.
    warm = cv2.imread(str(shots[0]))[:80, :120]
    reader.readtext(warm, detail=0)
    tess.ocr_text(warm)

    summary: list[dict[str, Any]] = []
    for shot in shots:
        result = _process_shot(shot, perception, reader)
        fig_path = _render_figure(result, args.out)
        srows = [
            {
                k: r[k]
                for k in (
                    "region", "kind",
                    "easy_text", "easy_conf", "easy_latency_ms", "easy_pipeline",
                    "tess_text", "tess_conf", "tess_latency_ms", "tess_pipeline",
                    "agree",
                )
            }
            for r in result.get("rows", [])
        ]
        summary.append({"shot": result["shot"], "figure": str(fig_path) if fig_path else None, "regions": srows})
        print(f"[ocr-compare] {shot.name}: {len(srows)} regions -> {fig_path}")

    agg = _aggregate(summary)
    summary_dir = args.out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {"aggregate": agg, "shots": summary}
    summary_path = summary_dir / "ocr_compare.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2))
    print(f"[ocr-compare] summary -> {summary_path}")


if __name__ == "__main__":
    main()
