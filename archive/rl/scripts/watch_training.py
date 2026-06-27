#!/usr/bin/env python3
"""Poll TensorBoard event files and refresh training progress charts (separate process)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config.settings import LOGS_DIR


PROGRESS_DIR = LOGS_DIR.parent / "training_progress"
SCALARS = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/entropy_loss",
)


def _latest_run_dir(log_root: Path, stage_prefix: str) -> Path | None:
    if not log_root.is_dir():
        return None
    candidates = sorted(
        (p for p in log_root.iterdir() if p.is_dir() and p.name.startswith(stage_prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in SCALARS:
        if tag not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def _read_pool_size() -> dict | None:
    path = LOGS_DIR / "opponent_pool_size.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def render_progress(
    series: dict[str, list[tuple[int, float]]],
    *,
    run_name: str,
    pool_info: dict | None,
) -> tuple[Path, Path]:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Training progress — {run_name}", fontsize=12)

    plot_map = {
        "rollout/ep_rew_mean": axes[0, 0],
        "rollout/ep_len_mean": axes[0, 1],
        "train/loss": axes[1, 0],
        "train/entropy_loss": axes[1, 1],
    }
    titles = {
        "rollout/ep_rew_mean": "Episode reward (mean)",
        "rollout/ep_len_mean": "Episode length (mean)",
        "train/loss": "PPO loss",
        "train/entropy_loss": "Entropy loss",
    }
    for tag, ax in plot_map.items():
        points = series.get(tag, [])
        if points:
            steps, values = zip(*points)
            ax.plot(steps, values, linewidth=1.5)
        ax.set_title(titles[tag])
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.3)

    if pool_info:
        fig.text(
            0.01,
            0.01,
            f"Opponent pool: {pool_info.get('team_limit')}/{pool_info.get('total_available')} teams",
            fontsize=9,
        )

    fig.tight_layout()
    png_path = PROGRESS_DIR / "latest.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    summary = {
        "timestamp_utc": stamp,
        "run_name": run_name,
        "png": str(png_path.resolve()),
        "pool": pool_info,
        "series": {
            tag: {"steps": [s for s, _ in pts], "values": [v for _, v in pts]}
            for tag, pts in series.items()
        },
    }
    for tag in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
        pts = series.get(tag, [])
        if pts:
            summary[f"latest_{tag.replace('/', '_')}"] = pts[-1][1]
            summary[f"latest_{tag.replace('/', '_')}_step"] = pts[-1][0]

    json_path = PROGRESS_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch TensorBoard and refresh progress plots")
    parser.add_argument(
        "--stage-prefix",
        default="stage1_random",
        help="TensorBoard run name prefix (default: stage1_random)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=90,
        help="Seconds between refreshes (default: 90)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one chart and exit",
    )
    args = parser.parse_args()

    print(f"Watching {LOGS_DIR} for runs matching {args.stage_prefix!r}")
    print(f"Charts -> {PROGRESS_DIR.resolve()} (every {args.interval}s)")

    while True:
        run_dir = _latest_run_dir(LOGS_DIR, args.stage_prefix)
        if run_dir is None:
            print("No TensorBoard run found yet...")
        else:
            series = _load_scalars(run_dir)
            if not series:
                print(f"Waiting for scalars in {run_dir.name}...")
            else:
                png, summary = render_progress(
                    series,
                    run_name=run_dir.name,
                    pool_info=_read_pool_size(),
                )
                rew = series.get("rollout/ep_rew_mean", [])
                last_rew = f"{rew[-1][1]:.2f}" if rew else "n/a"
                print(f"Updated {png.name} | {run_dir.name} | ep_rew_mean={last_rew}")
                print(f"Summary: {summary}")
        if args.once:
            break
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    main()
