"""Transformer BC player: TeamPreviewModel at Turn 0 + neural net from Turn 1."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from config.settings import BATTLE_FORMAT, BC_MODEL_PATH, TEAM_PATH
from src.doubles.battle.canonical_inference import (
    canonical_indices_to_battle_order,
    pick_masked_canonical_indices,
    submission_debug,
)
from src.doubles.battle.move_order import (
    classify_action_correction,
    format_live_canonical_action,
    is_true_fallback,
)
from src.doubles.data.live_log_bridge import (
    encode_live_as_log,
    pick_masked_dual_force_actions,
    pick_masked_live_log_actions,
    slot_mask_for_live,
)
from src.core.data.state_tokenizer import (
    N_FIELDS,
    N_TOKENS,
    TRAJECTORY_DEPTH,
    encode_battle,
    push_trajectory,
    stack_trajectory,
    trajectory_frame_fingerprints,
)
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.evaluation.battle_inference_trace import (
    action_record,
    encode_request_protocol_line,
    format_live_battle_brief,
    format_battle_timeline,
    summarize_protocol_lines,
    summarize_server_request,
    topk_live_choices,
)
from src.core.model.transformer_bot import load_model
from src.doubles.players.preview_orchestrator import PreviewOrchestrator


class TransformerPlayer(Player):
    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        battle_format: str = BATTLE_FORMAT,
        team: Optional[str] = None,
        device: str = "cpu",
        preview: PreviewOrchestrator | None = None,
        log_illegal_top1: bool = True,
        trace_inference: bool = False,
        trace_top_k: int = 5,
        capture_battle_log: bool = True,
        **kwargs,
    ):
        if team is None:
            team = TEAM_PATH.read_text(encoding="utf-8")
        super().__init__(battle_format=battle_format, team=team, **kwargs)
        self.device = device
        path = Path(model_path or BC_MODEL_PATH)
        if path.is_file():
            self.model = load_model(path, device=device)
        else:
            from src.core.model.transformer_bot import VGCBehaviorCloner

            self.model = VGCBehaviorCloner().to(device)
            self.model.eval()
        self.preview = preview or PreviewOrchestrator()
        self._trajectory_history: dict[str, deque[np.ndarray]] = {}
        self.log_illegal_top1 = log_illegal_top1
        self.illegal_top1_events: list[dict] = []
        self.trace_inference = trace_inference
        self.trace_top_k = trace_top_k
        self.capture_battle_log = capture_battle_log or trace_inference
        self._inference_traces: dict[str, list[dict]] = {}
        self._protocol_logs: dict[str, list[str]] = {}
        self._protocol_cursor: dict[str, int] = {}
        self._teampreview_cmds: dict[str, str] = {}
        self._decision_counters: dict[str, int] = {}
        self._last_trajectory_push_turn: dict[str, int] = {}
        self._pending_snapshot: dict[str, np.ndarray] = {}
        self._meta_db = MetaDatabase(live_fetch=False)
        self._log_views: dict[str, object] = {}

    def teampreview(self, battle: DoubleBattle) -> str:
        cmd = self.preview.teampreview(battle)
        if self.trace_inference:
            self._teampreview_cmds[battle.battle_tag] = cmd
        return cmd

    def drain_illegal_top1_events(self) -> list[dict]:
        """Return and clear logged illegal raw top-1 decisions."""
        out = list(self.illegal_top1_events)
        self.illegal_top1_events.clear()
        return out

    def drain_inference_trace(self, battle_tag: str) -> dict:
        """Return full inference + protocol trace for one battle."""
        log = self._protocol_logs.get(battle_tag, [])
        cursor = self._protocol_cursor.get(battle_tag, 0)
        if cursor < len(log):
            tail = summarize_protocol_lines(log[cursor:])
            if tail and self._inference_traces.get(battle_tag):
                self._inference_traces[battle_tag][-1]["battle_events_since_last"] = (
                    self._inference_traces[battle_tag][-1].get(
                        "battle_events_since_last", []
                    )
                    + tail
                )
            self._protocol_cursor[battle_tag] = len(log)

        all_events = summarize_protocol_lines(log)
        return {
            "battle_tag": battle_tag,
            "teampreview": self._teampreview_cmds.pop(battle_tag, None),
            "decisions": self._inference_traces.pop(battle_tag, []),
            "protocol_log": self._protocol_logs.pop(battle_tag, []),
            "battle_timeline": format_battle_timeline(all_events),
            "battle_events": all_events,
        }

    def pop_all_inference_traces(self) -> list[dict]:
        """Drain traces for all battles seen this session."""
        tags = set(self._inference_traces) | set(self._protocol_logs) | set(
            self._teampreview_cmds
        )
        out = [self.drain_inference_trace(tag) for tag in sorted(tags)]
        self._decision_counters.clear()
        self._protocol_cursor.clear()
        self._last_trajectory_push_turn.clear()
        self._pending_snapshot.clear()
        return out

    def _events_since_last_decision(self, battle: DoubleBattle) -> list[str]:
        tag = battle.battle_tag
        log = self._protocol_logs.get(tag, [])
        cursor = self._protocol_cursor.get(tag, 0)
        events = summarize_protocol_lines(log[cursor:])
        self._protocol_cursor[tag] = len(log)
        return events

    async def _handle_battle_message(self, split_messages):
        await super()._handle_battle_message(split_messages)
        if not self.capture_battle_log:
            return
        try:
            battle = await self._get_battle(split_messages[0][0])
            tag = battle.battle_tag
            log = self._protocol_logs.setdefault(tag, [])
            for split_message in split_messages[1:]:
                if not split_message:
                    continue
                if len(split_message) >= 2 and split_message[1] in ("", "request"):
                    if split_message[1] == "":
                        log.append("|" + "|".join(split_message[1:]))
                    elif split_message[1] == "request":
                        if len(split_message) >= 3 and split_message[2]:
                            try:
                                request = json.loads(split_message[2])
                                log.append(encode_request_protocol_line(request))
                            except json.JSONDecodeError:
                                log.append("|request|...")
                        else:
                            log.append("|request|...")
                elif len(split_message) >= 2:
                    log.append("|" + "|".join(split_message[1:]))
        except Exception:
            pass

    def _history_for(self, battle: DoubleBattle) -> deque[np.ndarray]:
        tag = battle.battle_tag
        if tag not in self._trajectory_history:
            self._trajectory_history[tag] = deque(maxlen=TRAJECTORY_DEPTH)
        return self._trajectory_history[tag]

    def _commit_trajectory(self, battle: DoubleBattle, snapshot: np.ndarray) -> None:
        if any(battle.force_switch):
            return
        tag = battle.battle_tag
        turn = int(battle.turn)
        if self._last_trajectory_push_turn.get(tag) == turn:
            return
        dq = self._history_for(battle)
        history_list = list(dq)
        push_trajectory(
            history_list,
            snapshot,
            depth=TRAJECTORY_DEPTH,
            maxlen=TRAJECTORY_DEPTH,
        )
        dq.clear()
        dq.extend(history_list)
        self._last_trajectory_push_turn[tag] = turn

    def _stacked_input(
        self, battle: DoubleBattle
    ) -> tuple[torch.Tensor, list[str], object | None, str, np.ndarray]:
        tag = battle.battle_tag
        protocol = self._protocol_logs.get(tag, [])
        encoded = encode_live_as_log(
            battle,
            protocol_lines=protocol,
            side="p1",
            meta_db=self._meta_db,
        )
        if encoded is not None:
            snapshot, view, sample_kind = encoded
            self._log_views[tag] = view
        else:
            snapshot = encode_battle(battle)
            view = None
            sample_kind = "turn"
            self._log_views.pop(tag, None)

        self._pending_snapshot[tag] = snapshot
        dq = self._history_for(battle)
        history_list = list(dq)
        force = any(battle.force_switch)
        turn = int(battle.turn)
        if force:
            # Parser emits mid-turn force_switch before the turn snapshot is pushed.
            if self._last_trajectory_push_turn.get(tag) == turn and history_list:
                history_list = history_list[:-1]
        stacked = stack_trajectory(history_list, snapshot, depth=TRAJECTORY_DEPTH)
        fingerprints = trajectory_frame_fingerprints(stacked)
        self._last_sample_kind = sample_kind
        return (
            torch.as_tensor(stacked, dtype=torch.long, device=self.device).unsqueeze(0),
            fingerprints,
            view,
            sample_kind,
            snapshot,
        )

    def _canonical_mask(
        self,
        battle: DoubleBattle,
        pos: int,
        *,
        view: object | None,
        sample_kind: str,
        slot0_pred: int | None = None,
    ) -> torch.Tensor:
        if view is None:
            from src.doubles.battle.move_order import (
                canonical_force_switch_mask,
                pokeenv_action_mask_to_canonical,
            )
            from poke_env.environment.doubles_env import DoublesEnv

            if any(battle.force_switch):
                mask = canonical_force_switch_mask(battle, pos)
            else:
                pe = DoublesEnv.get_action_mask_individual(battle, pos)
                mask = pokeenv_action_mask_to_canonical(battle, pos, pe)
            return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        suffix = "a" if pos == 0 else "b"
        mask = slot_mask_for_live(
            battle,
            view,
            side="p1",
            sample_kind=sample_kind,
            slot_suffix=suffix,
            slot0_pred=slot0_pred,
        )
        return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

    def _log_illegal_top1(
        self,
        battle: DoubleBattle,
        *,
        slot: int,
        raw_top1: int,
        picked: int,
        mask: torch.Tensor,
    ) -> None:
        if not self.log_illegal_top1:
            return
        if 0 <= raw_top1 < mask.shape[0] and bool(mask[raw_top1].item()):
            return
        force = any(battle.force_switch)
        correction = classify_action_correction(
            battle, slot, raw_top1, picked, force_switch=force
        )
        self.illegal_top1_events.append(
            {
                "battle_tag": battle.battle_tag,
                "turn": int(battle.turn),
                "force_switch": list(battle.force_switch),
                "slot": slot,
                "raw_top1_index": raw_top1,
                "raw_top1_label": format_live_canonical_action(battle, slot, raw_top1),
                "picked_index": picked,
                "picked_label": format_live_canonical_action(battle, slot, picked),
                "correction_kind": correction,
                "fallback_used": correction == "semantic",
            }
        )

    def _record_inference_trace(
        self,
        battle: DoubleBattle,
        *,
        logits0: torch.Tensor,
        logits1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        ca0: int,
        ca1: int,
        raw0: int,
        raw1: int,
        trajectory_frames: list[str] | None = None,
    ) -> None:
        tag = battle.battle_tag
        idx = self._decision_counters.get(tag, 0) + 1
        self._decision_counters[tag] = idx

        force = any(battle.force_switch)

        def _slot(
            pos: int,
            logits: torch.Tensor,
            mask: torch.Tensor,
            raw: int,
            picked: int,
        ) -> dict:
            sub = submission_debug(battle, pos, picked)
            return {
                "raw_top1": action_record(
                    battle, pos, index=raw,
                    legal=0 <= raw < mask.shape[0] and bool(mask[raw].item()),
                ),
                "picked": action_record(battle, pos, index=picked),
                "submission": sub,
                "correction_kind": classify_action_correction(
                    battle, pos, raw, picked, force_switch=force
                ),
                "topk_legal": topk_live_choices(
                    logits, mask, battle, pos, k=self.trace_top_k, legal_only=True
                ),
            }

        true_fb0 = is_true_fallback(battle, 0, raw0, ca0, force_switch=force)
        true_fb1 = is_true_fallback(battle, 1, raw1, ca1, force_switch=force)

        self._inference_traces.setdefault(tag, []).append(
            {
                "decision_index": idx,
                "kind": "inference",
                "battle_tag": tag,
                "turn": int(battle.turn),
                "wait": bool(battle.wait),
                "force_switch": list(battle.force_switch),
                "trajectory_depth": len(self._history_for(battle)),
                "trajectory_frames": trajectory_frames or [],
                "state_text": format_live_battle_brief(battle),
                "server_request": summarize_server_request(battle),
                "battle_events_since_last": self._events_since_last_decision(battle),
                "slot0": _slot(0, logits0, mask0, raw0, ca0),
                "slot1": _slot(1, logits1, mask1, raw1, ca1),
                "any_fallback": true_fb0 or true_fb1,
                "any_correction": raw0 != ca0 or raw1 != ca1,
            }
        )

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait and not any(battle.force_switch):
            if self.trace_inference:
                tag = battle.battle_tag
                idx = self._decision_counters.get(tag, 0) + 1
                self._decision_counters[tag] = idx
                self._inference_traces.setdefault(tag, []).append(
                    {
                        "decision_index": idx,
                        "kind": "wait",
                        "battle_tag": tag,
                        "turn": int(battle.turn),
                        "state_text": format_live_battle_brief(battle),
                        "server_request": summarize_server_request(battle),
                        "battle_events_since_last": self._events_since_last_decision(battle),
                    }
                )
            return DefaultBattleOrder()

        x, trajectory_frames, view, sample_kind, snapshot = self._stacked_input(battle)
        with torch.no_grad():
            logits0, logits1 = self.model(x)

        row0 = logits0[0]
        row1 = logits1[0]
        raw0 = int(row0.argmax().item())
        raw1 = int(row1.argmax().item())

        fs = list(battle.force_switch)
        dual_force = len(fs) >= 2 and fs[0] and fs[1]
        if dual_force and view is not None:
            ca0, ca1 = pick_masked_dual_force_actions(
                self.model,
                battle=battle,
                protocol_lines=self._protocol_logs.get(battle.battle_tag, []),
                side="p1",
                meta_db=self._meta_db,
                history=list(self._history_for(battle)),
                last_push_turn=self._last_trajectory_push_turn.get(battle.battle_tag),
                device=self.device,
            )
        elif view is not None:
            ca0, ca1 = pick_masked_live_log_actions(
                row0,
                row1,
                battle=battle,
                view=view,
                side="p1",
                sample_kind=sample_kind,
            )
        else:
            ca0, ca1 = pick_masked_canonical_indices(battle, row0, row1)

        mask0 = self._canonical_mask(battle, 0, view=view, sample_kind=sample_kind)
        mask1 = self._canonical_mask(
            battle, 1, view=view, sample_kind=sample_kind, slot0_pred=ca0
        )

        self._log_illegal_top1(
            battle, slot=0, raw_top1=raw0, picked=ca0, mask=mask0
        )
        self._log_illegal_top1(
            battle, slot=1, raw_top1=raw1, picked=ca1, mask=mask1
        )

        if self.trace_inference:
            self._record_inference_trace(
                battle,
                logits0=row0,
                logits1=row1,
                mask0=mask0,
                mask1=mask1,
                ca0=ca0,
                ca1=ca1,
                raw0=raw0,
                raw1=raw1,
                trajectory_frames=trajectory_frames,
            )

        order = canonical_indices_to_battle_order(battle, ca0, ca1)
        if not isinstance(order, DefaultBattleOrder) and not any(battle.force_switch):
            self._commit_trajectory(battle, snapshot)
        return order
