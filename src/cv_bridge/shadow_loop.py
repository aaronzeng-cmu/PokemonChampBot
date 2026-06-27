"""Shadow loop: observe a live Champions battle and track state from the screen.

This runs alongside a real game (LDPlayer via ADB). Every frame it OCRs the
bottom battle-log box to capture stat changes / faints / weather / moves and
feeds them into a :class:`LiveBattleTracker`. On each turn-decision frame it reads
precise HP from the HUD and builds the model observation tensor.

It is a *shadow* loop by default: it prints the state and the action it *would*
take. Pass a ``policy`` callable (and ``execute=True``) to actually tap.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from src.cv_bridge import battle_log_parser
from src.cv_bridge.action_executor import ActionExecutor, BattleFormat
from src.cv_bridge.bc_policy import BCPolicy, PreviewPolicy
from src.cv_bridge.bc_policy import PreviewPolicy as PreviewPolicy_
from src.cv_bridge.emulator_bridge import EmulatorBridge
from src.cv_bridge.perception import PerceptionModule
from src.cv_bridge.state_tracker import LiveBattleTracker

_DEFAULT_SCREENSHOT_DIR = Path("logs/cv_bridge/shadow")

# A policy maps (observation, masks) -> a doubles action tuple (ca0, ca1) or a
# singles action index. Returning None means "no decision" (stay in shadow mode).
Policy = Callable[[np.ndarray, dict[str, np.ndarray] | None], Any]

_DEFAULT_BC_MODELS = {
    "singles": "models/singles_bc_transformer_latest.pt",
    "doubles": "models/bc_transformer_latest.pt",
}
_DEFAULT_PREVIEW_MODELS = {
    "singles": "models/singles_preview_model.pt",
    "doubles": "models/preview_model.pt",
}


class _Screen(Protocol):
    def get_screen(self) -> np.ndarray | None: ...


class ShadowLoop:
    def __init__(
        self,
        *,
        bridge: _Screen | None = None,
        perception: PerceptionModule | None = None,
        tracker: LiveBattleTracker | None = None,
        executor: ActionExecutor | None = None,
        battle_format: BattleFormat = "doubles",
        policy: Policy | None = None,
        preview_policy: PreviewPolicy_ | None = None,
        execute_taps: bool = False,
        post_action_delay: float = 3.0,
        poll_interval: float = 0.4,
        save_screenshots: bool = True,
        screenshot_dir: Path | str | None = None,
        screenshot_interval: float = 1.0,
        screenshot_keep: int = 300,
    ):
        self.bridge = bridge if bridge is not None else EmulatorBridge()
        self.perception = perception or PerceptionModule()
        self.tracker = tracker or LiveBattleTracker(battle_format=battle_format)
        # Always have an executor so dry-runs can show the taps we *would* send.
        self.executor = executor or ActionExecutor(battle_format=battle_format)
        self.policy = policy
        self.preview_policy = preview_policy
        self.execute_taps = execute_taps
        self.post_action_delay = post_action_delay
        self.poll_interval = poll_interval
        # Inter-tap pacing. A submenu-opening tap (Fight / switch menu) needs the
        # list to animate in before the next tap, or that tap is swallowed.
        self.tap_delay = 0.35
        self.submenu_settle = 0.6

        # Recovery: re-tap the move when the move list is still open (the initial
        # move tap was lost to the menu transition).
        self._pending_move_taps: list[Any] = []
        self._move_recover_attempts = 0
        self._max_move_recover_attempts = 4

        # Forced replacement after a faint: re-tap until the party screen clears.
        self._force_switch_attempts = 0
        self._max_force_switch_attempts = 4

        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else _DEFAULT_SCREENSHOT_DIR
        self.screenshot_interval = screenshot_interval
        self.screenshot_keep = screenshot_keep
        self._last_screenshot_time = 0.0

        self.last_log_text = ""
        self._last_popup_texts: set[str] = set()
        self.turn_processed = False
        self.preview_processed = False

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Single iteration of the loop body (separated for testing).

        Returns a small observation summary used for debug capture.
        """
        state = self.perception.get_current_state(frame)
        observation: dict[str, Any] = {
            "state": state,
            "log_text": None,
            "event": None,
            "decided": False,
        }

        # 1. Always check for battle-log text (animations, stat changes, faints).
        log_text = self.perception.read_battle_log(frame)
        if log_text and log_text != self.last_log_text:
            observation["log_text"] = log_text
            event = battle_log_parser.parse_string(log_text)
            if event:
                applied = self.tracker.apply_log_event(event)
                event["_applied"] = applied
                observation["event"] = event
                status = "applied" if applied else "unresolved"
                print(f"[LOG EVENT] ({status}) {event}  <- {log_text!r}")
            self.last_log_text = log_text

        # 1b. Ability / item activation banners (mid-screen, left/right of the
        # actives) reveal opponent abilities & held items separately from the log
        # box. They linger several frames, so only act on newly-appeared text.
        popups = self.perception.read_ability_item_popups(frame)
        new_popups = [t for t in popups if t not in self._last_popup_texts]
        self._last_popup_texts = set(popups)
        popup_events: list[dict[str, Any]] = []
        for text in new_popups:
            event = battle_log_parser.parse_ability_item_popup(text)
            if not event:
                continue
            applied = self.tracker.apply_log_event(event)
            event["_applied"] = applied
            popup_events.append(event)
            status = "applied" if applied else "unresolved"
            print(f"[POPUP] ({status}) {event}  <- {text!r}")
        if popup_events:
            observation["popups"] = popup_events

        # 2. Team preview (Turn 0): pick the bring slots / lead order once.
        if state == "TEAM_PREVIEW" and not self.preview_processed:
            if self.preview_policy is not None:
                observation["preview"] = self._on_preview(frame)
            else:
                print("[PREVIEW] TEAM_PREVIEW detected but no preview policy loaded.")
                observation["preview"] = {"skipped": "no_preview_policy"}
            self.preview_processed = True
        elif state != "TEAM_PREVIEW":
            self.preview_processed = False

        # 3. State-based logic: act once per turn-decision. Reset the gate as soon
        # as we leave TURN_DECISION (animation / idle) so the next turn fires.
        if state == "TURN_DECISION" and not self.turn_processed:
            self.tracker.update_turn(self.perception.extract_battle_data(frame))
            obs, masks = self.tracker.get_model_inputs()
            observation["decision"] = self._on_decision(obs, masks)
            self.turn_processed = True
            observation["decided"] = True
        elif state != "TURN_DECISION":
            self.turn_processed = False

        # 4. Recovery: the move list is still open, so the move tap was lost to the
        # menu transition. Re-issue just the move/target taps (Fight already done).
        if state == "MOVE_SELECTION":
            observation["recovery"] = self._recover_move_selection()
        else:
            self._move_recover_attempts = 0

        # 5. Forced replacement: a mon fainted and the game is waiting for us to
        # pick a healthy bench Pokemon. Otherwise the loop would sit here forever.
        if state == "FORCE_SWITCH":
            observation["force_switch"] = self._on_force_switch(frame)
        else:
            self._force_switch_attempts = 0

        return observation

    def _on_force_switch(self, frame: np.ndarray) -> dict[str, Any]:
        slots = self.perception.read_party_slots(frame)
        alive = [s for s in slots if s.get("alive")]
        if not alive:
            print("[FORCE_SWITCH] party HP unreadable; cannot choose a replacement")
            return {"status": "no_alive_slots", "slots": slots}
        if self._force_switch_attempts >= self._max_force_switch_attempts:
            return {"status": "exhausted", "slots": slots}
        self._force_switch_attempts += 1
        choice = int(alive[0]["slot"])
        plan = self.executor.plan_force_switch(choice)
        print(
            f"[FORCE_SWITCH] fainted; replacing with party slot {choice} "
            f"(alive={[s['slot'] for s in alive]}) "
            f"attempt {self._force_switch_attempts}/{self._max_force_switch_attempts} "
            f"-> {self._format_taps(plan)}"
        )
        self._run_plan(plan)
        return {
            "status": "switched" if self.execute_taps else "dry_run",
            "slot": choice,
            "attempt": self._force_switch_attempts,
            "alive": [s["slot"] for s in alive],
            "slots": slots,
            "taps": [t.label for t in plan.taps],
        }

    def _recover_move_selection(self) -> dict[str, Any] | None:
        if not self.execute_taps or not self._pending_move_taps:
            return None
        if self._move_recover_attempts >= self._max_move_recover_attempts:
            return {"status": "exhausted"}
        self._move_recover_attempts += 1
        print(
            f"[RECOVER] move list still open; re-tapping move "
            f"(attempt {self._move_recover_attempts}/{self._max_move_recover_attempts})"
        )
        self._tap_sequence(self._pending_move_taps)
        time.sleep(self.post_action_delay)
        return {"status": "retapped", "attempt": self._move_recover_attempts}

    def _on_decision(self, obs: np.ndarray, masks: dict[str, np.ndarray] | None) -> dict[str, Any]:
        if self.policy is None:
            print(f"[DECISION] obs={obs.shape} masks={'yes' if masks else 'none'} (shadow: no policy)")
            return {"skipped": "no_policy"}

        action = self.policy(obs, masks)
        if action is None:
            print("[DECISION] policy returned no action; skipping turn")
            return {"skipped": "no_action"}
        if isinstance(action, tuple):
            ca0, ca1 = action
        else:
            ca0, ca1 = action, None

        plan = self.executor.plan_turn(action)
        tap_plan = self._format_taps(plan)
        print(f"[DECISION] Slot 1: {ca0}, Slot 2: {ca1} -> {tap_plan}")
        # Remember the move/target taps (everything after opening the Fight menu)
        # so we can re-issue them if the move list is still open next frame.
        self._pending_move_taps = [t for t in plan.taps if not self._is_submenu_opener(t.label)]
        self._move_recover_attempts = 0
        self._run_plan(plan)
        return {
            "ca0": ca0,
            "ca1": ca1,
            "taps": [t.label for t in plan.taps],
            "executed": self.execute_taps,
        }

    def _on_preview(self, frame: np.ndarray) -> dict[str, Any]:
        teams = self.perception.parse_team_preview(frame)
        ally = teams.get("ally_team", [])
        enemy = teams.get("enemy_team", [])
        self.tracker.record_team_preview(ally, enemy)
        try:
            slots = self.preview_policy(ally, enemy)  # type: ignore[misc]
        except Exception as exc:
            print(f"[PREVIEW] inference failed ({exc!r}); skipping preview taps")
            return {"ally": ally, "enemy": enemy, "error": repr(exc)}
        print(f"[PREVIEW] ally={ally} enemy={enemy} -> bring slots {slots}")
        brought = [ally[s - 1] for s in slots if 1 <= s <= len(ally)]
        self.tracker.record_brought_ally(brought)
        plan = self.executor.plan_teampreview(slots)
        print(f"[PREVIEW] taps -> {self._format_taps(plan)}")
        self._run_plan(plan)
        return {
            "ally": ally,
            "enemy": enemy,
            "slots": slots,
            "taps": [t.label for t in plan.taps],
            "executed": self.execute_taps,
        }

    @staticmethod
    def _format_taps(plan: Any) -> str:
        return ", ".join(f"{t.label}({t.x},{t.y})" for t in plan.taps) or "<none>"

    @staticmethod
    def _is_submenu_opener(label: str) -> bool:
        return "fight" in label or "pokemon_switch_menu" in label or "switch.open" in label

    def _tap_sequence(self, taps: list[Any]) -> None:
        for tap in taps:
            print(f"  tap {tap.label} -> ({tap.x}, {tap.y})")
            self.bridge.tap(tap.x, tap.y)
            time.sleep(self.tap_delay)
            if self._is_submenu_opener(tap.label):
                # Wait for the move/switch list to slide in before the next tap.
                time.sleep(self.submenu_settle)

    def _run_plan(self, plan: Any) -> None:
        """Execute a tap plan on the device, or print it in dry-run mode."""
        if not self.execute_taps:
            for tap in plan.taps:
                print(f"  [dry-run] tap {tap.label} -> ({tap.x}, {tap.y})")
            return
        self._tap_sequence(plan.taps)
        # Let the UI transition (animations / next prompt) before we read again,
        # so we don't spam inputs into a stale screen.
        time.sleep(self.post_action_delay)

    def _maybe_save_screenshot(self, frame: np.ndarray, observation: dict[str, Any]) -> None:
        """Save a debug screenshot (+ JSONL line) at most once per interval."""
        if not self.save_screenshots:
            return
        now = time.monotonic()
        if now - self._last_screenshot_time < self.screenshot_interval:
            return
        self._last_screenshot_time = now

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        label = str(observation.get("state", "")).lower()

        png_path: Path | None = None
        save_fn = getattr(self.bridge, "save_screenshot", None)
        if callable(save_fn):
            png_path = save_fn(self.screenshot_dir, label=label, frame=frame)
        if png_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            png_path = self.screenshot_dir / f"{stamp}_{label}.png"
            cv2.imwrite(str(png_path), frame)

        debug_line = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "screenshot": png_path.name,
            "state": observation.get("state"),
            "log_text": observation.get("log_text"),
            "event": observation.get("event"),
            "popups": observation.get("popups"),
            "decided": observation.get("decided", False),
            "decision": observation.get("decision"),
            "preview": observation.get("preview"),
            "recovery": observation.get("recovery"),
            "force_switch": observation.get("force_switch"),
            "live": self.execute_taps,
            "has_policy": self.policy is not None,
            "has_preview_policy": self.preview_policy is not None,
        }
        with (self.screenshot_dir / "shadow_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(debug_line) + "\n")

        self._prune_screenshots()

    def _prune_screenshots(self) -> None:
        """Keep only the most recent ``screenshot_keep`` PNGs (+ their meta)."""
        if self.screenshot_keep <= 0:
            return
        pngs = sorted(
            self.screenshot_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in pngs[: -self.screenshot_keep]:
            stale.unlink(missing_ok=True)
            stale.with_suffix(".meta.json").unlink(missing_ok=True)

    def run(self, *, max_iterations: int | None = None) -> None:
        print("Shadow loop started. Ctrl+C to stop.")
        if self.save_screenshots:
            print(f"Saving debug screenshots every {self.screenshot_interval:.1f}s to {self.screenshot_dir}")
        iterations = 0
        try:
            while True:
                frame = self.bridge.get_screen()
                if frame is None:
                    time.sleep(self.poll_interval)
                    continue
                observation = self.process_frame(frame)
                self._maybe_save_screenshot(frame, observation)
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nShadow loop stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CV shadow loop against a live battle.")
    parser.add_argument("--format", choices=["singles", "doubles"], default="doubles")
    parser.add_argument("--poll", type=float, default=0.4, help="Seconds between frames.")
    parser.add_argument("--max-iters", type=int, default=None, help="Stop after N frames (debug).")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (state only).")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="BC transformer checkpoint (defaults to the format-appropriate model).",
    )
    parser.add_argument(
        "--preview-model-path",
        type=str,
        default=None,
        help="Team-preview model checkpoint (defaults to the format-appropriate model).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Skip loading models (pure perception/state shadow run).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute taps on the device (Phase 4). Without this, decisions are dry-run only.",
    )
    parser.add_argument(
        "--post-action-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after issuing taps so the UI can transition.",
    )
    parser.add_argument("--no-screenshots", action="store_true", help="Disable debug screenshots.")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=_DEFAULT_SCREENSHOT_DIR,
        help="Where to save debug screenshots + shadow_log.jsonl.",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=1.0,
        help="Seconds between saved debug screenshots.",
    )
    parser.add_argument(
        "--screenshot-keep",
        type=int,
        default=300,
        help="Max debug screenshots to retain (0 = unlimited).",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy: BCPolicy | None = None
    preview_policy: PreviewPolicy | None = None
    if not args.no_policy:
        from src.cv_bridge.action_executor import load_ui_coordinates

        model_path = args.model_path or _DEFAULT_BC_MODELS[args.format]
        preview_path = args.preview_model_path or _DEFAULT_PREVIEW_MODELS[args.format]

        if Path(model_path).is_file():
            policy = BCPolicy(model_path, device=device)
            model_fmt = "singles" if policy.is_singles else "doubles"
            print(f"[MODEL] BC policy loaded: {model_path} (action_space={model_fmt}, device={device})")
            if model_fmt != args.format:
                print(f"[WARN] model action space '{model_fmt}' != --format '{args.format}'.")
        else:
            print(f"[WARN] BC model not found at {model_path}; running without a turn policy.")

        if Path(preview_path).is_file():
            pick_count = int(load_ui_coordinates()[args.format]["teampreview_pick_count"])
            preview_policy = PreviewPolicy(
                preview_path, battle_format=args.format, device=device, pick_count=pick_count
            )
            print(f"[MODEL] Preview policy loaded: {preview_path} (pick={pick_count}, device={device})")
        else:
            print(f"[WARN] Preview model not found at {preview_path}; team preview disabled.")

    if args.live:
        print("[LIVE] Tap execution ENABLED -- the bot will touch the device.")
    perception = PerceptionModule(ocr_enabled=not args.no_ocr)
    loop = ShadowLoop(
        perception=perception,
        battle_format=args.format,
        policy=policy,
        preview_policy=preview_policy,
        execute_taps=args.live,
        post_action_delay=args.post_action_delay,
        poll_interval=args.poll,
        save_screenshots=not args.no_screenshots,
        screenshot_dir=args.screenshot_dir,
        screenshot_interval=args.screenshot_interval,
        screenshot_keep=args.screenshot_keep,
    )
    loop.run(max_iterations=args.max_iters)


if __name__ == "__main__":
    main()
