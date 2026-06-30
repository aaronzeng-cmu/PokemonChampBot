"""Synthetic training data for the species classifier.

Each sample = one official menu sprite (RGBA, alpha-composited) pasted onto a
randomized background with augmentation that mimics how the icon appears in the
team-view / team-preview / battle HUD:

* background  : random crop of a real game screenshot, or a procedural
                gradient / noise / solid panel.
* geometry    : random scale, small rotation, sub-pixel placement, partial
                off-canvas crop.
* photometric : brightness / contrast, hue / saturation jitter, gaussian blur,
                sensor noise, JPEG-style recompression, occasional occlusion.

Output is an ImageFolder tree (``<root>/{train,val}/<species>/<n>.jpg``) that
YOLOv8-cls consumes directly.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ICONS_DIR = Path(__file__).resolve().parents[1] / "assets" / "pokemon_icons"
_SCREENSHOT_DIR = Path("logs/cv_bridge/screenshots")
# Forms that never appear outside battle (team view / preview show base forms).
_BATTLE_FORM_RE = re.compile(r"(mega[xy]?|primal|gmax|eternamax)$")


def is_battle_only_form(species_id: str) -> bool:
    return bool(_BATTLE_FORM_RE.search(species_id))


@dataclass
class _Sprite:
    species_id: str
    rgb: np.ndarray  # HxWx3 uint8 (BGR)
    alpha: np.ndarray  # HxW uint8


def load_sprites(icons_dir: Path = _ICONS_DIR, *, exclude_forms: bool = True) -> list[_Sprite]:
    sprites: list[_Sprite] = []
    for path in sorted(icons_dir.glob("*.png")):
        species_id = path.stem.lower()
        if exclude_forms and is_battle_only_form(species_id):
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3].copy()
        if int(alpha.max()) == 0:
            continue
        sprites.append(_Sprite(species_id=species_id, rgb=bgr, alpha=alpha))
    return sprites


def load_background_pool(screenshot_dir: Path = _SCREENSHOT_DIR) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    if screenshot_dir.exists():
        for path in sorted(screenshot_dir.glob("*.png")):
            img = cv2.imread(str(path))
            if img is not None and img.size:
                pool.append(img)
    return pool


def _rand_background(size: int, bg_pool: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """A square BGR background of side ``size``."""
    choice = rng.random()
    if bg_pool and choice < 0.7:
        src = bg_pool[rng.randrange(len(bg_pool))]
        h, w = src.shape[:2]
        side = rng.randint(min(h, w) // 6, min(h, w) // 2)
        x = rng.randint(0, w - side)
        y = rng.randint(0, h - side)
        crop = src[y : y + side, x : x + side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    if choice < 0.85:  # vertical gradient between two random colors
        c0 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c1 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
        grad = (c0[None, None, :] * (1 - t) + c1[None, None, :] * t)
        return np.repeat(grad, size, axis=1).astype(np.uint8)
    if choice < 0.95:  # random noise
        return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    solid = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    return np.full((size, size, 3), solid, dtype=np.uint8)


def _augment_sprite(
    rgb: np.ndarray, alpha: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric + photometric jitter on the sprite (keeps RGB/alpha aligned)."""
    h, w = rgb.shape[:2]

    # Hue / saturation / value jitter.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.75, 1.2), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.25), 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Rotation.
    angle = rng.uniform(-15, 15)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rgb = cv2.warpAffine(rgb, mat, (w, h), borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(alpha, mat, (w, h), borderValue=0)

    # Slight blur (icon downscaled on screen).
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)
    return rgb, alpha


def _composite(
    sprite: _Sprite, size: int, bg_pool: list[np.ndarray], rng: random.Random
) -> np.ndarray:
    canvas = _rand_background(size, bg_pool, rng).astype(np.float32)

    rgb, alpha = _augment_sprite(sprite.rgb, sprite.alpha, rng)

    # Scale the sprite to a random fraction of the canvas.
    scale = rng.uniform(0.55, 1.05)
    side = max(8, int(size * scale))
    rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (side, side), interpolation=cv2.INTER_AREA)

    # Random placement; allow the sprite to spill slightly off-canvas.
    max_off = int(side * 0.18)
    x = rng.randint(-max_off, size - side + max_off)
    y = rng.randint(-max_off, size - side + max_off)

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(size, x + side), min(size, y + side)
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    if x1 <= x0 or y1 <= y0:
        return canvas.astype(np.uint8)

    a = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    fg = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    roi = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = a * fg + (1 - a) * roi

    out = canvas.astype(np.uint8)

    # Whole-image photometric tweaks.
    out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.15), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # sensor noise
        noise = np.random.normal(0, rng.uniform(3, 12), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:  # random occlusion patch
        ow, oh = rng.randint(size // 8, size // 3), rng.randint(size // 8, size // 3)
        ox, oy = rng.randint(0, size - ow), rng.randint(0, size - oh)
        patch = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        out[oy : oy + oh, ox : ox + ow] = patch
    if rng.random() < 0.5:  # JPEG recompression artifacts
        q = rng.randint(35, 92)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def generate(
    out_dir: Path,
    *,
    per_class_train: int = 80,
    per_class_val: int = 20,
    imgsz: int = 128,
    exclude_forms: bool = True,
    seed: int = 0,
) -> dict[str, int]:
    """Write an ImageFolder dataset; returns ``{"classes", "train", "val"}``."""
    rng = random.Random(seed)
    np.random.seed(seed)

    sprites = load_sprites(exclude_forms=exclude_forms)
    bg_pool = load_background_pool()
    if not sprites:
        raise RuntimeError("no sprites found")

    out_dir = Path(out_dir)
    for split, n in (("train", per_class_train), ("val", per_class_val)):
        for sprite in sprites:
            cls_dir = out_dir / split / sprite.species_id
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = _composite(sprite, imgsz, bg_pool, rng)
                cv2.imwrite(str(cls_dir / f"{i:04d}.jpg"), img)

    return {
        "classes": len(sprites),
        "train": len(sprites) * per_class_train,
        "val": len(sprites) * per_class_val,
        "backgrounds": len(bg_pool),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic species-classifier data.")
    ap.add_argument("--out", type=Path, default=Path("data/species_cls"))
    ap.add_argument("--train", type=int, default=80, help="Samples per class (train).")
    ap.add_argument("--val", type=int, default=20, help="Samples per class (val).")
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--include-forms", action="store_true", help="Keep mega/primal/gmax forms.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = generate(
        args.out,
        per_class_train=args.train,
        per_class_val=args.val,
        imgsz=args.imgsz,
        exclude_forms=not args.include_forms,
        seed=args.seed,
    )
    print(
        f"[synth] wrote {stats['train']} train + {stats['val']} val images across "
        f"{stats['classes']} classes -> {args.out}  (bg pool={stats['backgrounds']})"
    )


if __name__ == "__main__":
    main()
