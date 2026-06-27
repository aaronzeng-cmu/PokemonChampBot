"""ADB-backed emulator bridge for screen capture and touch input."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

Rotation = Literal["auto", "none", "90cw", "90ccw", "180"]

_LDPLAYER_ADB_GLOB = "LDPlayer*/adb.exe"
_DEFAULT_SCREENSHOT_DIR = Path(__file__).resolve().parents[2] / "logs" / "cv_bridge" / "screenshots"


def _detect_ldplayer_adb() -> Optional[str]:
    """Return the newest LDPlayer adb.exe path on Windows, if present."""
    if sys.platform != "win32":
        return None

    search_roots = [
        Path(r"C:\Program Files\LDPlayer"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "LDPlayer",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "LDPlayer",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob(_LDPLAYER_ADB_GLOB))

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.parent.name, reverse=True)
    return str(candidates[0])


def _resolve_adb_path(adb_path: Optional[str]) -> str:
    if adb_path:
        return adb_path
    detected = _detect_ldplayer_adb()
    if detected:
        return detected
    return "adb"


def _rotation_for_frame(rotation: Rotation, raw_width: int, raw_height: int) -> Rotation:
    if rotation != "auto":
        return rotation
    if raw_height > raw_width:
        return "90cw"
    return "none"


def _rotate_image(image: np.ndarray, rotation: Rotation) -> np.ndarray:
    if rotation == "90cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "90ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _display_to_device_coords(
    x: int,
    y: int,
    raw_width: int,
    raw_height: int,
    rotation: Rotation,
) -> tuple[int, int]:
    """Map landscape/display coordinates back to raw device screencap coords."""
    if rotation == "90cw":
        return raw_width - 1 - y, x
    if rotation == "90ccw":
        return y, raw_height - 1 - x
    if rotation == "180":
        return raw_width - 1 - x, raw_height - 1 - y
    return x, y


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.strip())
    return cleaned.strip("_") or "screenshot"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class EmulatorBridge:
    def __init__(
        self,
        adb_path: Optional[str] = None,
        device_serial: Optional[str] = None,
        *,
        connect_host: str = "127.0.0.1",
        connect_port: int = 5555,
        auto_connect: bool = True,
        rotation: Rotation = "auto",
    ):
        self.adb_path = _resolve_adb_path(adb_path)
        self.connect_host = connect_host
        self.connect_port = connect_port
        self.auto_connect = auto_connect
        self.rotation: Rotation = rotation
        self._last_raw_size: Optional[tuple[int, int]] = None
        self._last_applied_rotation: Rotation = "none"

        if device_serial is None and auto_connect:
            device_serial = f"{connect_host}:{connect_port}"
        self.device_serial = device_serial

        self.base_cmd = [self.adb_path]
        if self.device_serial:
            self.base_cmd.extend(["-s", self.device_serial])

        if auto_connect:
            self._connect_device()

    def _connect_device(self) -> bool:
        target = f"{self.connect_host}:{self.connect_port}"
        try:
            result = subprocess.run(
                [self.adb_path, "connect", target],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
            output = (result.stdout + result.stderr).decode("utf-8", errors="replace").lower()
            return result.returncode == 0 and (
                "connected" in output or "already connected" in output
            )
        except (subprocess.SubprocessError, OSError):
            return False

    def _capture_raw_screen(self) -> Optional[np.ndarray]:
        try:
            result = subprocess.run(
                self.base_cmd + ["exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
            if result.returncode != 0 or not result.stdout:
                return None

            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None:
                return None

            raw_height, raw_width = image.shape[:2]
            self._last_raw_size = (raw_width, raw_height)
            return image
        except (subprocess.SubprocessError, OSError, ValueError):
            return None

    def get_screen(self) -> Optional[np.ndarray]:
        image = self._capture_raw_screen()
        if image is None:
            return None
        raw_height, raw_width = image.shape[:2]
        applied = _rotation_for_frame(self.rotation, raw_width, raw_height)
        self._last_applied_rotation = applied
        return _rotate_image(image, applied)

    def save_screenshot(
        self,
        output_dir: Path | str,
        *,
        label: str = "",
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Capture (or reuse) a frame and write PNG + sidecar metadata JSON."""
        image = frame if frame is not None else self.get_screen()
        if image is None:
            return None

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = _utc_timestamp()
        suffix = f"_{_sanitize_label(label)}" if label.strip() else ""
        stem = f"{stamp}{suffix}"
        png_path = out_dir / f"{stem}.png"
        meta_path = out_dir / f"{stem}.meta.json"

        if not cv2.imwrite(str(png_path), image):
            return None

        height, width = image.shape[:2]
        meta = {
            "filename": png_path.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": label.strip() or None,
            "width": width,
            "height": height,
            "rotation_setting": self.rotation,
            "rotation_applied": self._last_applied_rotation,
            "device_serial": self.device_serial,
            "adb_path": self.adb_path,
            "coordinate_system": (
                "Landscape display pixels matching saved PNG. "
                "Use annotate mode on this file for offline (x, y) mapping."
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return png_path

    def _to_device_coords(self, x: int, y: int) -> tuple[int, int]:
        if self._last_raw_size is None or self._last_applied_rotation == "none":
            return x, y
        raw_width, raw_height = self._last_raw_size
        return _display_to_device_coords(
            x, y, raw_width, raw_height, self._last_applied_rotation
        )

    def tap(self, x: int, y: int) -> None:
        device_x, device_y = self._to_device_coords(x, y)
        try:
            subprocess.run(
                self.base_cmd + ["shell", "input", "tap", str(device_x), str(device_y)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            pass

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
    ) -> None:
        device_x1, device_y1 = self._to_device_coords(x1, y1)
        device_x2, device_y2 = self._to_device_coords(x2, y2)
        try:
            subprocess.run(
                self.base_cmd
                + [
                    "shell",
                    "input",
                    "swipe",
                    str(device_x1),
                    str(device_y1),
                    str(device_x2),
                    str(device_y2),
                    str(duration_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            pass


def _print_bridge_info(bridge: EmulatorBridge, frame: np.ndarray) -> None:
    print(
        f"ADB: {bridge.adb_path} | Device: {bridge.device_serial} | "
        f"Rotation: {bridge.rotation} (applied: {bridge._last_applied_rotation}) | "
        f"Frame: {frame.shape[1]}x{frame.shape[0]}"
    )


def _run_capture_session(
    bridge: EmulatorBridge,
    output_dir: Path,
    *,
    initial_label: str = "",
) -> None:
    """Live preview: save screenshots to disk for offline coordinate mapping."""
    window_name = "Capture - S=save  L=label  R=refresh  Q=quit"
    frame = bridge.get_screen()
    if frame is None:
        print("Failed to capture screen. Is the emulator connected via ADB?")
        return

    pending_label = initial_label.strip()
    _print_bridge_info(bridge, frame)
    print(f"Saving to: {output_dir.resolve()}")
    print("Navigate the emulator to each UI state, then press S to save a screenshot.")
    if pending_label:
        print(f"Next save label: {pending_label!r}")

    while True:
        display = frame.copy()
        if pending_label:
            cv2.putText(
                display,
                f"label: {pending_label}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            refreshed = bridge.get_screen()
            if refreshed is not None:
                frame = refreshed
                print(f"Refreshed frame: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("Failed to refresh screen.")
        elif key in (ord("s"), ord("S")):
            saved = bridge.save_screenshot(output_dir, label=pending_label, frame=frame)
            if saved is None:
                print("Failed to save screenshot.")
            else:
                print(f"Saved: {saved}")
                print(f"       {saved.with_suffix('.meta.json')}")
                pending_label = ""
        elif key in (ord("l"), ord("L")):
            print("Enter label for the next save (empty to clear): ", end="", flush=True)
            cv2.destroyWindow(window_name)
            pending_label = input().strip()
            cv2.namedWindow(window_name)

    cv2.destroyAllWindows()


def _capture_once(bridge: EmulatorBridge, output_dir: Path, label: str) -> int:
    saved = bridge.save_screenshot(output_dir, label=label)
    if saved is None:
        print("Failed to capture screenshot.")
        return 1
    print(f"Saved: {saved}")
    print(f"       {saved.with_suffix('.meta.json')}")
    return 0


def _list_screenshot_images(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=lambda path: path.name)


def _load_annotation_sidecar(image_path: Path) -> dict:
    sidecar = image_path.with_suffix(".coords.json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"image": image_path.name, "points": []}


def _save_annotation_sidecar(image_path: Path, data: dict) -> None:
    sidecar = image_path.with_suffix(".coords.json")
    sidecar.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_offline_annotator(
    image_path: Optional[Path],
    *,
    image_dir: Optional[Path] = None,
) -> None:
    """Click saved screenshots to record (x, y) coordinates offline."""
    directory = image_dir or (image_path.parent if image_path else _DEFAULT_SCREENSHOT_DIR)
    images = _list_screenshot_images(directory)
    if not images:
        print(f"No PNG screenshots found in {directory.resolve()}")
        return

    if image_path is not None:
        try:
            index = images.index(image_path.resolve())
        except ValueError:
            images = [image_path.resolve()] + images
            index = 0
    else:
        index = 0

    window_name = "Annotate - click=point  C=clear  W=write  N/P=next/prev  Q=quit"
    points: list[dict] = []

    def redraw(base: np.ndarray) -> np.ndarray:
        canvas = base.copy()
        for idx, point in enumerate(points, start=1):
            x, y = int(point["x"]), int(point["y"])
            cv2.circle(canvas, (x, y), 8, (0, 255, 255), 2)
            cv2.putText(
                canvas,
                str(idx),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        return canvas

    def load_image_at(idx: int) -> tuple[Path, np.ndarray]:
        path = images[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to load {path}")
        sidecar = _load_annotation_sidecar(path)
        nonlocal points
        points = list(sidecar.get("points", []))
        return path, image

    current_path, base_frame = load_image_at(index)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append({"x": x, "y": y})
            print(f"{current_path.name}: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"Annotating: {current_path.name} ({index + 1}/{len(images)})")
    print(f"Directory: {directory.resolve()}")

    while True:
        header = f"{current_path.name}  [{index + 1}/{len(images)}]  points={len(points)}"
        canvas = redraw(base_frame)
        cv2.putText(
            canvas,
            header,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("n"), ord("N")) and len(images) > 1:
            index = (index + 1) % len(images)
            current_path, base_frame = load_image_at(index)
            print(f"Annotating: {current_path.name} ({index + 1}/{len(images)})")
        elif key in (ord("p"), ord("P")) and len(images) > 1:
            index = (index - 1) % len(images)
            current_path, base_frame = load_image_at(index)
            print(f"Annotating: {current_path.name} ({index + 1}/{len(images)})")
        elif key in (ord("c"), ord("C")):
            points.clear()
            print(f"Cleared points for {current_path.name}")
        elif key in (ord("w"), ord("W")):
            sidecar_path = current_path.with_suffix(".coords.json")
            _save_annotation_sidecar(
                current_path,
                {"image": current_path.name, "points": points},
            )
            print(f"Wrote {sidecar_path}")

    cv2.destroyAllWindows()


def _add_bridge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adb-path",
        default=None,
        help="Path to adb executable (auto-detects LDPlayer on Windows when omitted).",
    )
    parser.add_argument(
        "--device-serial",
        "--serial",
        dest="device_serial",
        default=None,
        help="ADB device serial (default: <connect-host>:<connect-port>).",
    )
    parser.add_argument(
        "--connect-host",
        default="127.0.0.1",
        help="Host used for automatic adb connect (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--connect-port",
        type=int,
        default=5555,
        help="Port used for automatic adb connect (default: 5555).",
    )
    parser.add_argument(
        "--no-auto-connect",
        action="store_true",
        help="Skip adb connect during initialization.",
    )
    parser.add_argument(
        "--rotation",
        choices=["auto", "none", "90cw", "90ccw", "180"],
        default="auto",
        help=(
            "Frame rotation for landscape UI (default: auto — 90cw when capture is "
            "portrait-shaped, otherwise none)."
        ),
    )


def _build_bridge(args: argparse.Namespace) -> EmulatorBridge:
    detected = _detect_ldplayer_adb()
    adb_path = _resolve_adb_path(args.adb_path)

    if args.adb_path is None and detected:
        print(f"Auto-detected LDPlayer ADB: {adb_path}")
    elif args.adb_path is None:
        print(f"Using ADB from PATH: {adb_path}")

    return EmulatorBridge(
        adb_path=adb_path,
        device_serial=args.device_serial,
        connect_host=args.connect_host,
        connect_port=args.connect_port,
        auto_connect=not args.no_auto_connect,
        rotation=args.rotation,
    )


def _parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    _add_bridge_args(common)

    parser = argparse.ArgumentParser(
        description="Capture emulator screenshots and annotate UI coordinates offline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    capture = subparsers.add_parser(
        "capture",
        parents=[common],
        help="Live preview: save screenshots while navigating the emulator (default).",
        description="Capture emulator frames to PNG for offline coordinate mapping.",
    )
    capture.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=_DEFAULT_SCREENSHOT_DIR,
        help=f"Directory for PNG + metadata JSON (default: {_DEFAULT_SCREENSHOT_DIR}).",
    )
    capture.add_argument(
        "--label",
        default="",
        help="Optional label suffix for the next saved screenshot (e.g. teampreview).",
    )
    capture.add_argument(
        "--once",
        action="store_true",
        help="Capture a single screenshot and exit (no preview window).",
    )

    annotate = subparsers.add_parser(
        "annotate",
        help="Mark (x, y) on saved screenshots offline.",
        description="Click saved PNGs to record coordinates into sidecar JSON files.",
    )
    annotate.add_argument(
        "--image",
        "-i",
        type=Path,
        default=None,
        help="Specific PNG to annotate.",
    )
    annotate.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=_DEFAULT_SCREENSHOT_DIR,
        help=f"Screenshot directory for N/P navigation (default: {_DEFAULT_SCREENSHOT_DIR}).",
    )

    argv = sys.argv[1:]
    if not argv or argv[0] not in ("capture", "annotate", "-h", "--help"):
        argv = ["capture", *argv]
    return parser.parse_args(argv)


def _main() -> None:
    args = _parse_args()

    if args.command == "annotate":
        image = args.image.resolve() if args.image else None
        image_dir = args.dir.resolve() if args.image is None else args.dir.resolve()
        _run_offline_annotator(image, image_dir=image_dir)
        return

    bridge = _build_bridge(args)
    output_dir = Path(args.output_dir)

    if args.once:
        raise SystemExit(_capture_once(bridge, output_dir, args.label))

    _run_capture_session(bridge, output_dir, initial_label=args.label)


if __name__ == "__main__":
    _main()
