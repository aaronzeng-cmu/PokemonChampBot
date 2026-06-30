"""Dump team-view icon crops + top sprite-match candidates for each slot."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from src.cv_bridge.sprite_matcher import SpriteMatcher
from src.cv_bridge.team_init import _load_icon_boxes

shot = Path(sys.argv[1])
ann = Path("logs/cv_bridge/analysis/region_annotations/teamview.json")
out = Path("logs/cv_bridge/analysis/teamview_icon_debug")
out.mkdir(parents=True, exist_ok=True)

boxes = _load_icon_boxes(ann)
img = cv2.imread(str(shot))
print(f"shot={shot}  size={img.shape[1]}x{img.shape[0]}  boxes={sorted(boxes)}")

sm = SpriteMatcher()
sm.build_index()

for slot in sorted(boxes):
    x, y, w, h = boxes[slot]
    crop = img[max(0, y) : y + h, max(0, x) : x + w]
    cv2.imwrite(str(out / f"slot_{slot}.png"), crop)
    res = sm.rank_sprite(crop, top_n=5, exclude_forms=True)
    ranked = res.get("ranked", [])
    decision = res.get("decision")
    dec_id = decision.species_id if decision else None
    print(f"\nslot {slot}  box={boxes[slot]}  decision={dec_id}")
    for sid, dist in ranked:
        print(f"    {sid:<18} {dist:.4f}")
