"""Singles BC player: preview model at Turn 0 + 22-class Transformer from Turn 1."""



from __future__ import annotations



import hashlib
import json

from collections import deque

from pathlib import Path

from typing import Optional



import numpy as np

import torch

from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from config.settings import SINGLES_BATTLE_FORMAT, SINGLES_BC_MODEL_PATH, SINGLES_TEAM_PATH
from src.core.data.state_tokenizer import (
    TRAJECTORY_DEPTH,
    trajectory_frame_fingerprints,
)
from src.core.model.transformer_bot import load_model
from src.doubles.evaluation.battle_inference_trace import (
    format_battle_timeline,
    summarize_protocol_lines,
)
from src.singles.action_mask import pick_masked_argmax
from src.singles.battle.live_protocol_support import SinglesLiveProtocolSupport
from src.singles.battle.canonical_inference import (
    canonical_index_to_battle_order,
    submission_debug,
)
from src.singles.evaluation.inference_trace import (
    action_record,
    encode_request_protocol_line,
    format_singles_live_battle_brief,
    summarize_server_request_singles,
    topk_singles_live_choices,
)
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator





class SinglesTransformerPlayer(SinglesLiveProtocolSupport, Player):

    def __init__(

        self,

        *,

        model_path: Path | str | None = None,

        battle_format: str = SINGLES_BATTLE_FORMAT,

        team: Optional[str] = None,

        device: str = "cpu",

        preview: SinglesPreviewOrchestrator | None = None,

        trace_inference: bool = False,

        trace_top_k: int = 5,

        capture_battle_log: bool = False,

        log_illegal_top1: bool = False,

        **kwargs,

    ):

        if team is None:

            team = SINGLES_TEAM_PATH.read_text(encoding="utf-8")

        super().__init__(battle_format=battle_format, team=team, **kwargs)

        self.device = device

        path = Path(model_path or SINGLES_BC_MODEL_PATH)

        if path.is_file():

            self.model = load_model(path, device=device)

            if self.model.action_space != "singles":

                raise ValueError(f"Expected singles BC model at {path}, got {self.model.action_space}")

        else:

            from src.core.model.transformer_bot import VGCBehaviorCloner, VGCBehaviorClonerConfig



            self.model = VGCBehaviorCloner(

                VGCBehaviorClonerConfig(action_space="singles")

            ).to(device)

            self.model.eval()

        self.preview = preview or SinglesPreviewOrchestrator(device=device)

        self._init_live_protocol_state()

        self.log_illegal_top1 = log_illegal_top1

        self.illegal_top1_events: list[dict] = []

        self.trace_inference = trace_inference

        self.trace_top_k = trace_top_k

        self.capture_battle_log = capture_battle_log or trace_inference

        self._inference_traces: dict[str, list[dict]] = {}

        self._protocol_cursor: dict[str, int] = {}

        self._decision_counters: dict[str, int] = {}



    def teampreview(self, battle: Battle) -> str:

        cmd = self.preview.teampreview(battle)

        self._record_teampreview(battle, cmd, capture_protocol=self.capture_battle_log)

        return cmd



    def drain_illegal_top1_events(self) -> list[dict]:

        out = list(self.illegal_top1_events)

        self.illegal_top1_events.clear()

        return out



    def drain_inference_trace(self, battle_tag: str) -> dict:

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



        max_turn = 0
        for line in log:
            if line.startswith("|turn|"):
                try:
                    max_turn = max(max_turn, int(line.split("|")[2]))
                except (IndexError, ValueError):
                    pass
        for turn in range(self._last_flushed_turn.get(battle_tag, 0) + 1, max_turn + 1):
            self._try_flush_turn_trajectory(battle_tag, turn)

        all_events = summarize_protocol_lines(log)

        return {

            "battle_tag": battle_tag,

            "teampreview": self._teampreview_cmds.pop(battle_tag, None),

            "decisions": self._inference_traces.pop(battle_tag, []),

            "protocol_log": self._protocol_logs.pop(battle_tag, []),

            "battle_timeline": format_battle_timeline(all_events),

            "battle_events": all_events,

        }



    def _events_since_last_decision(self, battle: Battle) -> list[str]:

        tag = battle.battle_tag

        log = self._protocol_logs.get(tag, [])

        cursor = self._protocol_cursor.get(tag, 0)

        events = summarize_protocol_lines(log[cursor:])

        self._protocol_cursor[tag] = len(log)

        return events



    async def _handle_battle_message(self, split_messages):

        if self.capture_battle_log:
            await self._capture_protocol_message(split_messages)

        await super()._handle_battle_message(split_messages)



    def _history_for(self, battle: Battle) -> deque[np.ndarray]:

        return self._history_for_tag(battle.battle_tag)



    def _stacked_input(self, battle: Battle) -> tuple[torch.Tensor, np.ndarray, list[str]]:

        stacked = self._stacked_obs(battle)
        snapshot = self._current_snapshot(battle)

        fingerprints = trajectory_frame_fingerprints(stacked)

        x = torch.as_tensor(stacked, dtype=torch.long, device=self.device).unsqueeze(0)

        return x, snapshot, fingerprints



    def _log_illegal_top1(

        self,

        battle: Battle,

        *,

        raw_top1: int,

        picked: int,

        mask: np.ndarray,

    ) -> None:

        if not self.log_illegal_top1:

            return

        if 0 <= raw_top1 < mask.shape[0] and bool(mask[raw_top1]):

            return

        from src.singles.evaluation.inference_trace import format_singles_live_action



        self.illegal_top1_events.append(

            {

                "battle_tag": battle.battle_tag,

                "turn": int(battle.turn),

                "force_switch": bool(battle.force_switch),

                "raw_top1_index": raw_top1,

                "raw_top1_label": format_singles_live_action(battle, raw_top1),

                "picked_index": picked,

                "picked_label": format_singles_live_action(battle, picked),

                "fallback_used": raw_top1 != picked,

            }

        )



    def _record_inference_trace(

        self,

        battle: Battle,

        *,

        logits: np.ndarray,

        mask: np.ndarray,

        picked: int,

        raw: int,

        trajectory_frames: list[str] | None = None,

        stacked_tokens: np.ndarray | None = None,

    ) -> None:

        tag = battle.battle_tag

        idx = self._decision_counters.get(tag, 0) + 1

        self._decision_counters[tag] = idx

        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        logits_t = torch.as_tensor(logits, dtype=torch.float32, device=self.device)



        self._inference_traces.setdefault(tag, []).append(

            {

                "decision_index": idx,

                "kind": "inference",

                "battle_tag": tag,

                "turn": int(battle.turn),

                "wait": bool(battle.wait),

                "force_switch": bool(battle.force_switch),

                "trajectory_depth": len(self._history_for(battle)),

                "trajectory_frames": trajectory_frames or [],
                "protocol_len": len(self._protocol_logs.get(tag, [])),
                "token_digest": (
                    hashlib.md5(np.ascontiguousarray(stacked_tokens).tobytes()).hexdigest()[:16]
                    if stacked_tokens is not None
                    else None
                ),
                "stacked_tokens": (
                    stacked_tokens.astype(np.int64).tolist()
                    if stacked_tokens is not None
                    else None
                ),

                "state_text": format_singles_live_battle_brief(battle),

                "server_request": summarize_server_request_singles(battle),

                "battle_events_since_last": self._events_since_last_decision(battle),

                "raw_top1": action_record(

                    battle,

                    index=raw,

                    legal=0 <= raw < mask.shape[0] and bool(mask[raw]),

                ),

                "picked": action_record(battle, index=picked),
                "submission": submission_debug(battle, picked),
                "topk_legal": topk_singles_live_choices(

                    logits_t,

                    mask_t,

                    battle,

                    k=self.trace_top_k,

                    legal_only=True,

                ),

                "fallback": raw != picked,

            }

        )



    def choose_move(self, battle: Battle) -> BattleOrder:

        if battle.wait and not battle.force_switch:

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

                        "state_text": format_singles_live_battle_brief(battle),

                        "server_request": summarize_server_request_singles(battle),

                        "battle_events_since_last": self._events_since_last_decision(battle),

                    }

                )

            return DefaultBattleOrder()



        x, snapshot, trajectory_frames = self._stacked_input(battle)

        mask = self._live_action_mask(battle)

        stacked_np = x.detach().cpu().numpy().squeeze(0)



        with torch.no_grad():

            logits = self.model(x)[0].cpu().numpy()



        raw = int(np.argmax(logits))

        action = pick_masked_argmax(logits, mask)

        self._log_illegal_top1(battle, raw_top1=raw, picked=action, mask=mask)



        if self.trace_inference:

            self._record_inference_trace(

                battle,

                logits=logits,

                mask=mask,

                picked=action,

                raw=raw,

                trajectory_frames=trajectory_frames,

                stacked_tokens=stacked_np,

            )



        order = canonical_index_to_battle_order(battle, action)

        if self.capture_battle_log:
            self._record_champchoice(battle, action)

        return order

