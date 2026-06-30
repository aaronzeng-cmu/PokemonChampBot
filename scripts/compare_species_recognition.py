"""Compare the YOLOv8-cls species classifier vs the pHash sprite matcher.

Two evaluations:
  1. Real team-view icon crops (the actual deployment input) with known labels.
  2. A sample of the synthetic held-out val set (top-1 accuracy, both methods).
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import cv2

from src.cv_bridge.species_classifier import SpeciesClassifier
from src.cv_bridge.sprite_matcher import SpriteMatcher
from src.cv_bridge.team_init import _load_icon_boxes


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Ground truth for the 20260629 team-view screenshot (from abilities + moves).
# floette/florges has no icon asset -> intentionally left out of accuracy.
TEAMVIEW_TRUTH = {
    1: "sinistcha",
    2: "ceruledge",
    3: None,  # floette/florges (icon missing)
    4: "ninetalesalola",
    5: "incineroar",
    6: "milotic",
}


def eval_teamview(moves_shot: Path, matcher: SpriteMatcher, cnn: SpeciesClassifier) -> None:
    ann = Path("logs/cv_bridge/analysis/region_annotations/teamview.json")
    boxes = _load_icon_boxes(ann)
    img = cv2.imread(str(moves_shot))
    print(f"\n=== TEAM-VIEW CROPS ({moves_shot.name}) ===")
    print(f"{'slot':<5}{'truth':<16}{'pHash':<22}{'CNN':<22}")
    ph_ok = cnn_ok = total = 0
    for slot in sorted(boxes):
        x, y, w, h = boxes[slot]
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        ph = matcher.identify_sprite(crop, exclude_forms=True)
        cres = cnn.rank_sprite(crop, top_n=1, exclude_forms=True)
        cn = cres["best_species_id"]
        cprob = cres["best_prob"] or 0.0
        truth = TEAMVIEW_TRUTH.get(slot)
        ph_mark = cn_mark = ""
        if truth is not None:
            total += 1
            if _norm(ph) == _norm(truth):
                ph_ok += 1
                ph_mark = " OK"
            if cn and _norm(cn) == _norm(truth):
                cnn_ok += 1
                cn_mark = " OK"
        print(
            f"{slot:<5}{str(truth):<16}{ph + ph_mark:<22}"
            f"{f'{cn} ({cprob:.2f})' + cn_mark:<22}"
        )
    print(f"\nteam-view accuracy (labeled slots={total}): pHash {ph_ok}/{total}  CNN {cnn_ok}/{total}")


def eval_synth(
    val_dir: Path, matcher: SpriteMatcher, cnn: SpeciesClassifier, per_class: int, seed: int
) -> None:
    rng = random.Random(seed)
    classes = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    samples: list[tuple[str, Path]] = []
    for cls in classes:
        files = list((val_dir / cls).glob("*.jpg"))
        rng.shuffle(files)
        for f in files[:per_class]:
            samples.append((cls, f))
    rng.shuffle(samples)

    ph_ok = cnn_ok = 0
    t_ph = t_cnn = 0.0
    for truth, path in samples:
        crop = cv2.imread(str(path))
        t0 = time.perf_counter()
        ph = matcher.identify_sprite(crop, exclude_forms=True)
        t1 = time.perf_counter()
        cn = cnn.identify_sprite(crop, exclude_forms=True)
        t2 = time.perf_counter()
        t_ph += t1 - t0
        t_cnn += t2 - t1
        ph_ok += _norm(ph) == _norm(truth)
        cnn_ok += _norm(cn) == _norm(truth)

    n = len(samples)
    print(f"\n=== SYNTHETIC VAL SAMPLE (n={n}, {len(classes)} classes) ===")
    print(f"pHash  top1 {ph_ok / n:.1%}   ({1000 * t_ph / n:.1f} ms/img)")
    print(f"CNN    top1 {cnn_ok / n:.1%}   ({1000 * t_cnn / n:.1f} ms/img)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--moves",
        type=Path,
        default=Path("logs/cv_bridge/screenshots/20260629_034106_teaminit_moves.png"),
    )
    ap.add_argument("--val", type=Path, default=Path("data/species_cls/val"))
    ap.add_argument("--per-class", type=int, default=4, help="Val samples per class.")
    ap.add_argument("--weights", type=Path, default=Path("src/cv_bridge/assets/species_cls.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    matcher = SpriteMatcher()
    matcher.build_index()
    cnn = SpeciesClassifier(args.weights)

    if args.moves.is_file():
        eval_teamview(args.moves, matcher, cnn)
    if args.val.is_dir():
        eval_synth(args.val, matcher, cnn, args.per_class, args.seed)


if __name__ == "__main__":
    main()
