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

from src.core.data.roster_profile import roster_species_key
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
        tap_delay: float = 1.0,
        submenu_settle: float = 0.6,
        party_settle: float = 1.2,
        save_screenshots: bool = True,
        screenshot_dir: Path | str | None = None,
        screenshot_interval: float = 1.0,
        screenshot_keep: int = 300,
        stability_frames: int = 1,
    ):
        self.bridge = bridge if bridge is not None else EmulatorBridge()
        self.perception = perception or PerceptionModule()
        self.battle_format = battle_format
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
        self.tap_delay = tap_delay
        self.submenu_settle = submenu_settle
        # The party screen slides in slower than a normal submenu; give it extra
        # time before we re-capture to read the (reordered) party sprites, or the
        # capture lands on the still-animating / previous screen.
        self.party_settle = party_settle

        # Recovery: re-tap the move when the move list is still open (the initial
        # move tap was lost to the menu transition).
        self._pending_move_taps: list[Any] = []
        self._move_recover_attempts = 0
        self._max_move_recover_attempts = 4

        # Forced replacement after a faint: re-tap until the party screen clears.
        self._force_switch_attempts = 0
        self._max_force_switch_attempts = 4

        # Recovery: the command/switch menu is still up after we acted, so the
        # action was lost (timing/coords). Re-issue the full turn plan a few
        # times, then re-arm the gate for a fresh decision -- otherwise the loop
        # sits on the menu until the in-game move timer expires (timeout).
        self._pending_turn_plan: Any | None = None
        self._turn_recover_attempts = 0
        self._max_turn_recover_attempts = 3

        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else _DEFAULT_SCREENSHOT_DIR
        self.screenshot_interval = screenshot_interval
        self.screenshot_keep = screenshot_keep
        self._last_screenshot_time = 0.0

        self.last_log_text = ""
        # Window-scoped set of every battle-log line already applied this post-action
        # window. At the higher frame rate the same line is OCR'd across many frames
        # (and can flicker A->B->A), so deduping only against the *immediately*
        # previous line would re-apply events (e.g. double-counting a stat boost).
        self._seen_log_texts: set[str] = set()
        self._last_popup_texts: set[str] = set()
        self.turn_processed = False
        # Trajectory history must advance once per *game* turn (training parity).
        # A recovery re-decision re-enters the decision branch without the state
        # ever leaving TURN_DECISION; this flag keeps that re-decision from pushing
        # a duplicate frame into the rolling history.
        self._turn_history_advanced = False
        self.preview_processed = False

        # Battle-log / popup OCR is only meaningful in the post-action window: from
        # the moment we submit a move (or replacement) until the next decision point
        # (TURN_DECISION / MOVE_SELECTION / FORCE_SWITCH). On the command/move menus
        # the log box overlaps the turn timer, so reading it there yields junk like
        # "06:48". Gating to this window also skips the most expensive OCR calls.
        self._track_log = False

        # Frame debouncing: require the UI state to hold for N consecutive frames
        # before committing to a fresh decision / preview pick / replacement. This
        # eliminates acting on a single transition frame (faded HUD, mid-animation
        # "0/100" reads, transient form names) that the state detector still labels
        # TURN_DECISION/etc. Recovery retries are *not* gated (they must not stall).
        self.stability_frames = max(1, int(stability_frames))
        self._streak_state: str | None = None
        self._streak_count = 0

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Single iteration of the loop body (separated for testing).

        Returns a small observation summary used for debug capture.
        """
        state = self.perception.get_current_state(frame)
        if state == self._streak_state:
            self._streak_count += 1
        else:
            self._streak_state = state
            self._streak_count = 1
        stable = self._streak_count >= self.stability_frames

        # Close the log/popup tracking window at any decision point: the command /
        # move menus show the timer over the log box, and a fresh decision means the
        # previous turn's animations are done.
        if state in {"TURN_DECISION", "MOVE_SELECTION", "FORCE_SWITCH"}:
            self._track_log = False

        observation: dict[str, Any] = {
            "state": state,
            "stable": stable,
            "streak": self._streak_count,
            "track_log": self._track_log,
            "log_text": None,
            "event": None,
            "decided": False,
        }

        # 1. Battle-log text (damage / stat changes / faints / weather) -- only in
        # the post-action window so we don't OCR the turn timer off the menu.
        if self._track_log:
            log_text = self.perception.read_battle_log(frame)
            # Apply each distinct line at most once per window: a lingering line is
            # re-OCR'd every frame, and the line can flicker out and back, so a
            # last-line-only guard would re-apply the same event.
            if log_text and log_text not in self._seen_log_texts:
                self._seen_log_texts.add(log_text)
                self.last_log_text = log_text
                observation["log_text"] = log_text
                event = battle_log_parser.parse_string(log_text)
                if event:
                    applied = self.tracker.apply_log_event(event)
                    event["_applied"] = applied
                    observation["event"] = event
                    status = "applied" if applied else "unresolved"
                    print(f"[LOG EVENT] ({status}) {event}  <- {log_text!r}")

            # 1b. Ability / item activation banners (mid-screen) reveal opponent
            # abilities & held items; they linger, so only act on newly-seen text.
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
        if state == "TEAM_PREVIEW" and stable and not self.preview_processed:
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
        if state == "TURN_DECISION" and stable and not self.turn_processed:
            self.tracker.update_turn(self.perception.extract_battle_data(frame))
            # Only advance the trajectory once per game turn; a same-turn re-decision
            # (after recovery cleared turn_processed) must not push a duplicate frame.
            obs, masks = self.tracker.get_model_inputs(advance=not self._turn_history_advanced)
            self._turn_history_advanced = True
            observation["decision"] = self._on_decision(obs, masks)
            self.turn_processed = True
            observation["decided"] = True
        elif state == "TURN_DECISION" and self.turn_processed:
            # We already acted but we're still on the command menu -> the action
            # didn't land. Retry instead of waiting out the move timer.
            observation["recovery"] = self._recover_turn_decision()
        elif state != "TURN_DECISION":
            self.turn_processed = False
            # Genuinely left the decision screen -> the next TURN_DECISION is a new
            # game turn and should advance the trajectory again.
            self._turn_history_advanced = False
            self._turn_recover_attempts = 0
            self._pending_turn_plan = None

        # 4. Recovery: the move list is still open, so the move tap was lost to the
        # menu transition. Re-issue just the move/target taps (Fight already done).
        if state == "MOVE_SELECTION":
            observation["recovery"] = self._recover_move_selection()
        else:
            self._move_recover_attempts = 0

        # 5. Forced replacement: a mon fainted and the game is waiting for us to
        # pick a healthy bench Pokemon. Otherwise the loop would sit here forever.
        if state == "FORCE_SWITCH":
            if stable:
                observation["force_switch"] = self._on_force_switch(frame)
        else:
            self._force_switch_attempts = 0

        return observation

    def _on_force_switch(self, frame: np.ndarray) -> dict[str, Any]:
        slots = self.perception.read_party_slots(frame)
        # The party screen is the authoritative readout of our whole team's HP;
        # persist faints so the next turn's switch mask excludes downed mons.
        self.tracker.record_party_readout(slots)
        alive = [s for s in slots if s.get("alive")]
        if not alive:
            print("[FORCE_SWITCH] party HP unreadable; cannot choose a replacement")
            return {"status": "no_alive_slots", "slots": slots}
        if self._force_switch_attempts >= self._max_force_switch_attempts:
            return {"status": "exhausted", "slots": slots}
        self._force_switch_attempts += 1
        # The party screen also lists the still-active partner; never pick a mon
        # that's already on the field. The list REORDERS as mons switch in, so the
        # row order is not the static brought order -- we read each row's sprite to
        # get the true row->species map and exclude rows whose species is currently
        # active. Falls back to brought-order, then HP matching.
        tr = self.tracker
        row_species = self.perception.read_party_species(frame)
        active_species = {
            roster_species_key(mon.species)
            for sl, mon in tr.state.mons.items()
            if sl.startswith(tr.player_side)
            and mon.active
            and mon.species
            and not mon.fainted
        }
        benchable: list[dict[str, Any]] = []
        if row_species:
            benchable = [
                s
                for s in alive
                if int(s["slot"]) in row_species
                and roster_species_key(row_species[int(s["slot"])]) not in active_species
            ]
            if benchable:
                print(f"[FORCE_SWITCH] party rows by sprite -> {row_species}")
        if not benchable:
            active_rows = tr.active_party_rows()
            benchable = [s for s in alive if int(s["slot"]) not in active_rows]
        if not benchable:
            active_hp = {
                (mon.hp, mon.max_hp)
                for sl, mon in tr.state.mons.items()
                if sl.startswith(tr.player_side)
                and mon.active
                and mon.species
                and not mon.fainted
            }
            benchable = [s for s in alive if (s.get("hp"), s.get("max_hp")) not in active_hp]
        if benchable:
            alive = benchable
        else:
            print("[FORCE_SWITCH] could not exclude on-field rows; using first alive")
        choice = int(alive[0]["slot"])
        plan = self.executor.plan_force_switch(choice)
        print(
            f"[FORCE_SWITCH] fainted; replacing with party slot {choice} "
            f"(alive={[s['slot'] for s in alive]}) "
            f"attempt {self._force_switch_attempts}/{self._max_force_switch_attempts} "
            f"-> {self._format_taps(plan)}"
        )
        self._run_plan(plan)
        # Replacement sent -> reopen the window for entry hazards / abilities and
        # the opponent's ensuing action.
        self._open_log_window()
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

    def _resolve_switch_rows(self, ca0: int, ca1: int | None) -> dict[int, int]:
        """Map each switching slot's action to an on-screen party row (1-based).

        The model's switch index N refers to ``team_roster[N-1]`` (same order the
        legal-action mask uses). The in-battle party screen lists the full brought
        team in team-preview *selection order* at fixed rows (actives/fainted are
        shown but unselectable), so a target's row = its position in that order.
        The model only picks legal (benched, healthy) targets, so the row always
        lands on a selectable mon.
        """
        from src.core.data.roster_profile import roster_species_key
        from src.doubles.battle.canonical_inference import decode_canonical_tuple

        tr = self.tracker
        roster = list(tr.state.team_roster.get(tr.player_side, []))
        brought = list(getattr(tr, "brought_ally", []) or []) or roster
        if not brought:
            return {}
        brought_keys = [roster_species_key(s) for s in brought]

        def row_for(ca: int | None) -> int | None:
            if ca is None:
                return None
            decoded = decode_canonical_tuple(ca)
            if decoded.get("kind") != "switch":
                return None
            idx = int(decoded["bench_slot"])
            if not (1 <= idx <= len(roster)):
                return None
            tkey = roster_species_key(roster[idx - 1])
            return brought_keys.index(tkey) + 1 if tkey in brought_keys else None

        rows: dict[int, int] = {}
        r0 = row_for(ca0)
        if r0 is not None:
            rows[0] = r0
        r1 = row_for(ca1)
        if r1 is not None:
            rows[1] = r1
        return rows

    def _recover_turn_decision(self) -> dict[str, Any] | None:
        """Still on the command menu after acting -> re-issue taps, then re-decide.

        Without this a single dropped tap (e.g. a switch that didn't register)
        leaves the loop parked on ``TURN_DECISION`` until the move timer runs out.
        """
        if not self.execute_taps:
            return None
        if self._turn_recover_attempts >= self._max_turn_recover_attempts:
            # Re-tapping the same plan isn't working; force a fresh decision.
            self.turn_processed = False
            self._turn_recover_attempts = 0
            self._pending_turn_plan = None
            print("[RECOVER] turn action didn't land; re-deciding next frame")
            return {"status": "redecide"}
        self._turn_recover_attempts += 1
        plan = self._pending_turn_plan
        if plan is None or not plan.taps:
            # Nothing to re-tap (e.g. pass/pass); re-arm to recompute the turn.
            self.turn_processed = False
            print(
                "[RECOVER] command menu still up, no taps to retry "
                f"(attempt {self._turn_recover_attempts}/{self._max_turn_recover_attempts}); re-deciding"
            )
            return {"status": "redecide", "attempt": self._turn_recover_attempts}
        print(
            "[RECOVER] command menu still up; re-issuing turn plan "
            f"(attempt {self._turn_recover_attempts}/{self._max_turn_recover_attempts})"
        )
        self._run_plan(plan)
        return {"status": "retapped", "attempt": self._turn_recover_attempts}

    def _on_decision(self, obs: np.ndarray, masks: dict[str, np.ndarray] | None) -> dict[str, Any]:
        self._pending_turn_plan = None
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

        switch_rows = None
        if self.battle_format == "doubles" and isinstance(action, tuple):
            self._record_committed_moves(ca0, ca1)
            # A switch needs the live party screen: the in-battle party list reorders
            # as mons faint / switch in, so the static brought-order row is wrong.
            # Execute the turn slot-by-slot, re-perceiving the party rows for each
            # switch instead of pre-baking row taps. (Dry-run keeps the old plan
            # path so the intended taps still log.)
            if self.execute_taps and (self._is_switch_action(ca0) or self._is_switch_action(ca1)):
                board = self._board_summary()
                readable = self._describe_decision(ca0, ca1)
                print(f"[BOARD] {board}")
                print(f"[DECISION] {readable}")
                print(f"[DECISION] raw idx -> Slot A: {ca0}, Slot B: {ca1} (phased: switch present)")
                taps = self._run_turn_phased(ca0, ca1)
                self._pending_turn_plan = None
                self._pending_move_taps = []
                self._move_recover_attempts = 0
                self._turn_recover_attempts = 0
                self._open_log_window()
                return {
                    "ca0": ca0,
                    "ca1": ca1,
                    "readable": readable,
                    "board": board,
                    "taps": taps,
                    "executed": True,
                    "phased": True,
                }
            switch_rows = self._resolve_switch_rows(ca0, ca1)
            plan = self.executor.plan_turn(
                action,
                switch_rows=switch_rows,
                move_names_slot0=self._active_move_order("a"),
                move_names_slot1=self._active_move_order("b"),
            )
        else:
            plan = self.executor.plan_turn(action, move_names=self._active_move_order("a"))
        tap_plan = self._format_taps(plan)
        board = self._board_summary()
        readable = self._describe_decision(ca0, ca1)
        print(f"[BOARD] {board}")
        print(f"[DECISION] {readable}")
        if switch_rows:
            print(f"[DECISION] switch rows (slot->party_row): {switch_rows}")
        print(f"[DECISION] raw idx -> Slot A: {ca0}, Slot B: {ca1} -> {tap_plan}")
        # Remember the move/target taps (everything after opening the Fight menu)
        # so we can re-issue them if the move list is still open next frame.
        self._pending_move_taps = [t for t in plan.taps if not self._is_submenu_opener(t.label)]
        self._move_recover_attempts = 0
        # Remember the full plan so we can re-issue it if we're still parked on
        # the command menu next frame (the action never registered).
        self._pending_turn_plan = plan
        self._turn_recover_attempts = 0
        self._run_plan(plan)
        # Move submitted -> open the post-action window so we OCR the resulting
        # log/popups (damage, faints, abilities) during the animations that follow.
        self._open_log_window()
        return {
            "ca0": ca0,
            "ca1": ca1,
            "readable": readable,
            "board": board,
            "taps": [t.label for t in plan.taps],
            "executed": self.execute_taps,
        }

    def _record_committed_moves(self, ca0: int, ca1: int | None) -> None:
        """Latch each active's committed move so next turn's input/masks reflect it
        (last move, Protect streak, and Choice lock)."""
        from src.core.data.move_utils import canonical_move_list
        from src.doubles.battle.canonical_inference import decode_canonical_tuple

        for suffix, ca in (("a", ca0), ("b", ca1)):
            if ca is None:
                continue
            try:
                decoded = decode_canonical_tuple(ca)
            except Exception:
                continue
            if decoded.get("kind") != "move":
                continue
            order = self._active_move_order(suffix)
            canon = canonical_move_list(list(order)) if order else []
            slot = int(decoded["move_slot"])
            if 1 <= slot <= len(canon):
                self.tracker.record_committed_move(suffix, canon[slot - 1])

    @staticmethod
    def _is_switch_action(ca: int | None) -> bool:
        from src.doubles.battle.canonical_inference import decode_canonical_tuple

        if ca is None:
            return False
        try:
            return decode_canonical_tuple(ca).get("kind") == "switch"
        except Exception:
            return False

    def _run_turn_phased(self, ca0: int, ca1: int | None) -> list[str]:
        """Execute a doubles turn slot-by-slot, re-perceiving the party screen for
        each switch so we tap the target's *current* row (handles reordering)."""
        labels: list[str] = []
        labels += self._execute_slot_phased(ca0, slot=0, suffix="a")
        if ca1 is not None:
            time.sleep(self.submenu_settle)
            labels += self._execute_slot_phased(ca1, slot=1, suffix="b")
        time.sleep(self.post_action_delay)
        return labels

    def _execute_slot_phased(self, ca: int, *, slot: int, suffix: str) -> list[str]:
        from src.doubles.battle.canonical_inference import decode_canonical_tuple

        try:
            decoded = decode_canonical_tuple(ca)
        except Exception:
            return []
        kind = decoded.get("kind")
        if kind == "pass":
            return []
        if kind == "switch":
            return self._execute_switch_phase(int(decoded["bench_slot"]), slot)
        # Move: per-slot plan (fight -> move -> target), no pre-baked switch row.
        seq = self.executor._plan_doubles_slot(
            ca, slot=slot, move_names=self._active_move_order(suffix), switch_row=None
        )
        self._tap_sequence(seq.taps)
        return [t.label for t in seq.taps]

    def _execute_switch_phase(self, bench_slot: int, slot: int) -> list[str]:
        """Open the party screen, read its (reordered) rows live, then tap the
        target's current row -> popup -> confirm.

        ``bench_slot`` is the model's switch index (1-based into ``team_roster``,
        the same order the legal-action mask uses). We resolve it to a species and
        then to whatever on-screen row that species currently occupies.
        """
        from src.core.data.roster_profile import roster_species_key

        tr = self.tracker
        roster = list(tr.state.team_roster.get(tr.player_side, []))
        target_species = roster[bench_slot - 1] if 1 <= bench_slot <= len(roster) else None
        target_key = roster_species_key(target_species) if target_species else None

        # Veto switching to a mon already on the field. The legal mask normally
        # excludes actives, but a flipped/stale active-species read can leak one in;
        # tapping an on-field mon's row does nothing and stalls the sequence.
        if target_key and target_key in tr.active_species_keys():
            print(
                f"  [switch] target {target_key} is already on the field; skipping "
                f"illegal switch for slot {slot}"
            )
            return []

        open_tap = self.executor.switch_open_tap()
        print(f"  tap {open_tap.label} -> ({open_tap.x}, {open_tap.y})  [open party for slot {slot}]")
        self.bridge.tap(open_tap.x, open_tap.y)
        # The party screen slides in slower than a normal submenu; wait it out so
        # the capture doesn't read the previous/animating screen.
        time.sleep(self.tap_delay + self.submenu_settle + self.party_settle)

        frame = self.bridge.get_screen()
        row: int | None = None
        row_species: dict[int, str] = {}
        if frame is not None and target_key:
            row_species = self.perception.read_party_species(frame)
            for r, sp in row_species.items():
                if roster_species_key(sp) == target_key:
                    row = r
                    break
        if row is None and target_key:
            # Sprite read failed -> fall back to brought (preview-selection) order.
            brought = list(getattr(tr, "brought_ally", []) or []) or roster
            brought_keys = [roster_species_key(s) for s in brought]
            if target_key in brought_keys:
                row = brought_keys.index(target_key) + 1
                print(f"  [switch] sprite row not found; brought-order row {row} for {target_key}")
        if row is None:
            print(f"  [switch] could not resolve a row for bench_slot {bench_slot}; aborting slot {slot}")
            return [open_tap.label]
        print(f"  [switch] party rows -> {row_species or '<unread>'}; target {target_key} -> row {row}")
        confirm = self.executor.switch_row_confirm_taps(row)
        self._tap_sequence(confirm)
        time.sleep(self.post_action_delay)
        return [open_tap.label] + [t.label for t in confirm]

    def _active_move_order(self, suffix: str) -> list[str]:
        """Active mon's moves in on-screen (physical) order from the team profile.

        The executor maps the model's canonical (alphabetical) move slot to the
        right physical button using this order, and uses the move name to resolve
        the doubles target (self / ally / foe).
        """
        mon = self.tracker.state.mons.get(f"{self.tracker.player_side}{suffix}")
        if not mon or not mon.moves:
            return []
        return list(mon.moves)

    def _board_summary(self) -> str:
        """Compact, in-game-matchable view of who is currently on the field."""
        def fmt(side: str) -> str:
            parts: list[str] = []
            for suffix in ("a", "b"):
                mon = self.tracker.state.mons.get(f"{side}{suffix}")
                if mon and mon.active and mon.species:
                    hp = f"{mon.hp}/{mon.max_hp}" if mon.max_hp else "?"
                    parts.append(f"{suffix.upper()}:{mon.species} ({hp})")
            return " + ".join(parts) if parts else "?"

        return f"us {fmt(self.tracker.player_side)}  vs  opp {fmt(self.tracker.opponent_side)}"

    def _describe_decision(self, ca0: int, ca1: int | None) -> str:
        """Plain-English action(s) decoded against the live (bench-augmented) view."""
        side = self.tracker.player_side
        try:
            view = self.tracker._state_with_bench()
            if self.battle_format == "doubles":
                from src.doubles.data.action_codec import format_log_action_pair

                return format_log_action_pair(view, side, ca0, ca1 if ca1 is not None else 0)
            from src.singles.log_action_codec import format_singles_log_action

            return format_singles_log_action(view, side, ca0)
        except Exception as exc:  # never let logging break the turn
            return f"<decode failed: {exc!r}>"

    def _on_preview(self, frame: np.ndarray) -> dict[str, Any]:
        teams = self.perception.parse_team_preview(frame)
        ally = teams.get("ally_team", [])
        enemy = teams.get("enemy_team", [])
        self.tracker.record_team_preview(ally, enemy)
        # Constrain enemy-active identification to their previewed set (+ Mega/Primal
        # forms) so the 3D model isn't matched against the whole dex on the field.
        enemy_closed = [s for s in enemy if s and s != "unknown"]
        if enemy_closed:
            self.perception.set_enemy_team(enemy_closed)
            print(f"[PREVIEW] enemy closed set -> {sorted(self.perception._enemy_team_keys)}")
        try:
            slots = self.preview_policy(ally, enemy)  # type: ignore[misc]
        except Exception as exc:
            print(f"[PREVIEW] inference failed ({exc!r}); skipping preview taps")
            return {"ally": ally, "enemy": enemy, "error": repr(exc)}
        print(f"[PREVIEW] ally={ally} enemy={enemy} -> bring slots {slots}")
        # Repair garbled/unknown preview species against our known team so the
        # brought lineup (and thus the switch mask) is complete.
        ally_resolved = self.tracker.reconcile_preview_species(ally)
        brought = [ally_resolved[s - 1] for s in slots if 1 <= s <= len(ally_resolved)]
        self.tracker.record_brought_ally(brought)
        if ally_resolved != ally:
            print(f"[PREVIEW] reconciled ally -> {ally_resolved}; brought {brought}")
        plan = self.executor.plan_teampreview(slots)
        print(f"[PREVIEW] taps -> {self._format_taps(plan)}")
        self._run_plan(plan)
        # Open the log/popup window for the lead send-out: turn-0 abilities (e.g.
        # Intimidate / Drizzle) fire during the entry animation, before the first
        # TURN_DECISION closes the window again.
        self._open_log_window()
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

    def _open_log_window(self) -> None:
        """Start the post-action OCR window with fresh per-window dedup state.

        Called right after we submit a move / replacement / preview lead, so the
        log and ability/item popups produced by the ensuing animations are read
        (each distinct line once) until the next decision point closes the window.
        """
        self._track_log = True
        self.last_log_text = ""
        self._seen_log_texts = set()
        self._last_popup_texts = set()

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

    @staticmethod
    def _poll_hotkey() -> str | None:
        """Non-blocking single-key read from the console (Windows). Returns the
        lowercased character pressed since the last poll, or ``None``."""
        try:
            import msvcrt  # type: ignore
        except Exception:
            return None
        if not msvcrt.kbhit():
            return None
        try:
            ch = msvcrt.getch()
            return ch.decode("utf-8", errors="ignore").lower() or None
        except Exception:
            return None

    def _manual_capture(self, frame: np.ndarray) -> None:
        """Save a debug snapshot on demand (hotkey 'c').

        Kept apart from the throttled auto-screenshots: written to a ``manual/``
        subfolder (never auto-pruned) with a ``manual_`` prefix and a sidecar JSON
        of the current perception so a frame can be inspected after the fact.
        """
        out_dir = self.screenshot_dir / "manual"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        try:
            state = self.perception.get_current_state(frame)
        except Exception as exc:
            state = f"err:{exc!r}"
        label = str(state).lower().replace(":", "_").replace(" ", "_")
        png_path = out_dir / f"manual_{stamp}_{label}.png"
        cv2.imwrite(str(png_path), frame)

        info: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "screenshot": png_path.name,
            "state": state,
        }
        for key, fn in (
            ("board", lambda: self._board_summary()),
            ("battle_log", lambda: self.perception.read_battle_log(frame)),
            ("popups", lambda: self.perception.read_ability_item_popups(frame)),
            ("party_species", lambda: self.perception.read_party_species(frame)),
        ):
            try:
                info[key] = fn()
            except Exception as exc:
                info[key] = f"err:{exc!r}"
        sidecar = png_path.with_suffix(".json")
        sidecar.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        print(f"[CAPTURE] manual snapshot -> {png_path}  (state={state})")

    def run(self, *, max_iterations: int | None = None) -> None:
        print("Shadow loop started. Ctrl+C to stop.")
        print("Hotkey: press 'c' to save a manual debug capture.")
        if self.save_screenshots:
            print(f"Saving debug screenshots every {self.screenshot_interval:.1f}s to {self.screenshot_dir}")
        iterations = 0
        try:
            while True:
                frame = self.bridge.get_screen()
                key = self._poll_hotkey()
                if key == "c":
                    cap = frame if frame is not None else self.bridge.get_screen()
                    if cap is not None:
                        self._manual_capture(cap)
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
    parser.add_argument(
        "--poll",
        type=float,
        default=0.15,
        help="Seconds between perception frames. Lower = catch brief log text (faints) better.",
    )
    parser.add_argument("--max-iters", type=int, default=None, help="Stop after N frames (debug).")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (state only).")
    parser.add_argument(
        "--ocr-gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help="EasyOCR device: 'auto' uses CUDA when available (default), 'on'/'off' force it.",
    )
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
    parser.add_argument(
        "--tap-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between individual taps (move/switch selection pacing).",
    )
    parser.add_argument(
        "--party-settle",
        type=float,
        default=1.2,
        help="Extra seconds to wait for the party screen to slide in before re-capturing it.",
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
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help=(
            "Our team profile JSON (teams/*.json from team_init). Seeds our move "
            "lists / bench so the legal-action mask isn't empty (required for play)."
        ),
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=2,
        help=(
            "Consecutive identical-state frames required before committing to a "
            "fresh decision/preview/replacement (debounce mid-animation reads)."
        ),
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
    ocr_gpu = {"auto": None, "on": True, "off": False}[args.ocr_gpu]
    perception = PerceptionModule(ocr_enabled=not args.no_ocr, ocr_gpu=ocr_gpu)
    # Confirm the species recognizer at startup (and force-load the CNN now so any
    # load error surfaces here, not mid-battle). Live preview + actives all route
    # species ID through perception.sprite_matcher.
    _recognizer = perception.sprite_matcher
    if hasattr(_recognizer, "_cnn_model"):
        _recognizer._cnn_model()
    print(f"[PERCEPTION] species recognizer = {type(_recognizer).__name__}")
    print(f"[PERCEPTION] EasyOCR gpu = {perception.ocr_gpu}")
    loop = ShadowLoop(
        perception=perception,
        battle_format=args.format,
        policy=policy,
        preview_policy=preview_policy,
        execute_taps=args.live,
        post_action_delay=args.post_action_delay,
        poll_interval=args.poll,
        tap_delay=args.tap_delay,
        party_settle=args.party_settle,
        save_screenshots=not args.no_screenshots,
        screenshot_dir=args.screenshot_dir,
        screenshot_interval=args.screenshot_interval,
        screenshot_keep=args.screenshot_keep,
        stability_frames=args.stability_frames,
    )
    if args.team:
        if Path(args.team).is_file():
            loop.tracker.load_player_team_file(args.team)
            # Constrain our-side active identification (sprite + nameplate OCR) to
            # this closed set of 6, so noisy reads snap to a known species.
            loop.perception.set_own_team(loop.tracker.known_team_species)
            print(
                f"[TEAM] Loaded our team from {args.team}: "
                f"{loop.tracker.known_team_species}"
            )
            if loop.perception._own_mega_to_base:
                print(
                    f"[TEAM] mega forms in closed set: "
                    f"{loop.perception._own_mega_to_base}"
                )
        else:
            print(f"[WARN] team profile not found at {args.team}; bot will likely pass.")
    elif not args.no_policy:
        print("[WARN] no --team given; our move/bench mask may be empty (bot passes).")
    loop.run(max_iterations=args.max_iters)


if __name__ == "__main__":
    main()
