"""Verify the player nameplate boxes: crop, OCR, fuzzy-match, and save crops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from src.cv_bridge.perception import PerceptionModule
from src.cv_bridge.state_tracker import LiveBattleTracker

SHOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "logs/cv_bridge/screenshots/20260624_232432.png"
)
OUT = Path("logs/cv_bridge/analysis/nameplate_debug")
OUT.mkdir(parents=True, exist_ok=True)

# Use the real live path: tracker loads the profile, perception is constrained.
tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
tracker.load_player_team_file("teams/champions_live_team.json")

p = PerceptionModule()
p.set_own_team(tracker.known_team_species)
print(f"screenshot : {SHOT}")
print(f"own team   : {p.own_team_species}")

frame = cv2.imread(str(SHOT))
if frame is None:
    raise SystemExit(f"could not read {SHOT}")
print(f"frame size : {frame.shape[1]}x{frame.shape[0]} (WxH)")

for key in ("player_active_name_slot_a", "player_active_name_slot_b"):
    spec = p.regions.get(key)
    crop = p._crop_region(frame, key)
    if crop is None or crop.size == 0:
        print(f"\n[{key}] spec={spec} -> EMPTY CROP")
        continue
    out_path = OUT / f"{key}.png"
    cv2.imwrite(str(out_path), crop)
    raw = p._ocr_crop(crop)
    snapped = p._fuzzy_own_species(raw)
    print(f"\n[{key}] spec={spec}")
    print(f"  crop saved : {out_path}  ({crop.shape[1]}x{crop.shape[0]})")
    print(f"  OCR raw    : {raw!r}")
    print(f"  fuzzy snap : {snapped!r}")

# Prove fuzzy tolerance against deliberately garbled OCR strings.
print("\n--- fuzzy tolerance check ---")
for garbled in ("Raichu", "Ralchu", "Garchompl", "GARCHOMP", "Azumaril", "Gyarados"):
    print(f"  {garbled!r:14} -> {p._fuzzy_own_species(garbled)!r}")
