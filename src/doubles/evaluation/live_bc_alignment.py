"""BC vs live inference alignment checks on captured protocol traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config.settings import BC_MODEL_PATH
from src.doubles.data.action_codec import decode_log_slot_action, format_log_action_pair
from src.doubles.data.live_log_bridge import (
    encode_live_as_log,
    pick_masked_dual_force_actions,
    pick_masked_live_log_actions,
)
from src.doubles.data.log_action_mask import (
    log_force_switch_slot_masks,
    pick_masked_log_actions,
    slot_mask_for_eval,
)
from src.core.data.log_tracker import BattleLogState
from src.doubles.data.replay_parser import parse_replay_log
from src.core.data.state_tokenizer import (
    TRAJECTORY_DEPTH,
    push_trajectory,
    stack_trajectory,
    trajectory_frame_fingerprints,
)
from src.doubles.evaluation.bc_examples import _topk_choices
from src.core.model.transformer_bot import load_model
from src.doubles.planning.meta_database import MetaDatabase


class _FakeBattle:
    def __init__(self, *, tag: str, turn: int, force_switch: list[bool]):
        self.battle_tag = tag
        self.turn = turn
        self.force_switch = force_switch
        self.teampreview = False


def load_trace_battle(trace_json: Path) -> dict:
    data = json.loads(trace_json.read_text(encoding="utf-8"))
    return (data.get("battles") or [data])[0]


def _forced_switch_suffix(view: BattleLogState, side: str) -> str | None:
    for suffix in ("a", "b"):
        mon = view.mons.get(f"{side}{suffix}")
        if mon is not None and (mon.fainted or mon.hp <= 0):
            return suffix
    return None


def _match_live_decision(
    decisions: list[dict],
    *,
    turn: int,
    sample_kind: str,
    force_suffix: str | None,
) -> dict | None:
    candidates = [
        d
        for d in decisions
        if d.get("kind") == "inference" and int(d.get("turn", -1)) == turn
    ]
    if sample_kind == "turn":
        for d in candidates:
            fs = d.get("force_switch") or [False, False]
            if not any(fs):
                return d
        return candidates[0] if candidates else None

    for d in candidates:
        fs = d.get("force_switch") or [False, False]
        if not any(fs):
            continue
        if force_suffix == "a" and fs[0] and not fs[1]:
            return d
        if force_suffix == "b" and not fs[0] and fs[1]:
            return d
        if force_suffix is None and fs[0] and fs[1]:
            return d
    for d in candidates:
        if any(d.get("force_switch") or []):
            return d
    return None


def _forced_switch_suffix(view: BattleLogState, side: str) -> str | None:
    for suffix in ("a", "b"):
        mon = view.mons.get(f"{side}{suffix}")
        if mon is not None and (mon.fainted or mon.hp <= 0):
            return suffix
    return None


def _live_force_suffix(fs: list[bool]) -> str | None:
    if len(fs) >= 2 and fs[0] and not fs[1]:
        return "a"
    if len(fs) >= 2 and fs[1] and not fs[0]:
        return "b"
    if len(fs) >= 2 and fs[0] and fs[1]:
        return None  # dual force — parser emits two samples
    return None


def _find_parser_sample(
    samples: list,
    *,
    turn: int,
    sample_kind: str,
    side: str,
    view: BattleLogState | None,
    live_fs: list[bool],
    used: set[int],
):
    """Match a live inference decision to the next parser sample."""
    live_forced = _live_force_suffix(live_fs) if sample_kind == "force_switch" else None

    for idx, sample in enumerate(samples):
        if idx in used:
            continue
        if sample.turn != turn or sample.sample_kind != sample_kind:
            continue
        if sample_kind != "force_switch":
            used.add(idx)
            return sample
        ps_view = sample.view_state
        if ps_view is None:
            continue
        ps_forced = _forced_switch_suffix(ps_view, side)
        if live_forced is None:
            # Dual live force: take first unused force_switch on this turn.
            used.add(idx)
            return sample
        if ps_forced == live_forced:
            used.add(idx)
            return sample
    return None


def _simulate_live_trajectory(
    protocol: list[str],
    decisions: list[dict],
    *,
    tag: str,
    meta_db: MetaDatabase,
) -> dict[int, tuple[np.ndarray, BattleLogState, str]]:
    """
    Replay live trajectory stacking (deferred commit) for each inference decision.

    Mirrors TransformerPlayer: stack_trajectory peek per decision; push_trajectory
    only once per voluntary turn after a non-force-switch decision.
    """
    hist: list[np.ndarray] = []
    last_push_turn: int | None = None
    out: dict[int, tuple[np.ndarray, BattleLogState, str, list[np.ndarray], int | None]] = {}

    for di, dec in enumerate(decisions):
        if dec.get("kind") != "inference":
            continue
        turn = int(dec["turn"])
        fs = list(dec.get("force_switch") or [False, False])
        fake = _FakeBattle(tag=tag, turn=turn, force_switch=fs)

        encoded = encode_live_as_log(
            fake, protocol_lines=protocol, side="p1", meta_db=meta_db
        )
        if encoded is None:
            continue
        snap, view, sample_kind = encoded

        history_list = list(hist)
        if any(fs) and last_push_turn == turn and history_list:
            history_list = history_list[:-1]
        stacked = stack_trajectory(history_list, snap, depth=TRAJECTORY_DEPTH)
        out[di] = (stacked, view, sample_kind, history_list, last_push_turn)

        if not any(fs) and last_push_turn != turn:
            push_trajectory(
                hist, snap, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH
            )
            last_push_turn = turn

    return out


def build_live_bridge_parity_report(
    trace_json: Path,
    *,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
) -> str:
    battle = load_trace_battle(trace_json)
    protocol = battle.get("protocol_log") or []
    tag = battle["battle_tag"]
    decisions = battle.get("decisions") or []
    meta_db = MetaDatabase(live_fetch=False)

    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=tag,
        skip_rating=True,
        keep_view_state=True,
    )
    p1_samples = [s for s in samples if s.side == "p1"]

    model = load_model(model_path, device=device)
    sim = _simulate_live_trajectory(
        protocol, decisions, tag=tag, meta_db=meta_db
    )
    used_parser: set[int] = set()

    lines = [
        f"Live bridge parity — {tag}",
        f"Trace: {trace_json}",
        "",
    ]

    tensor_matches = 0
    pred_matches = 0
    n_inference = 0

    for di, dec in enumerate(decisions):
        if dec.get("kind") != "inference":
            continue
        n_inference += 1
        turn = int(dec["turn"])
        fs = list(dec.get("force_switch") or [False, False])

        if di not in sim:
            lines.append(f"decision {di+1} turn {turn}: encode_live_as_log FAILED")
            continue
        stacked, view, sample_kind, hist_for_dec, last_push_turn = sim[di]
        live_fps = trajectory_frame_fingerprints(stacked)

        parser_sample = _find_parser_sample(
            p1_samples,
            turn=turn,
            sample_kind=sample_kind,
            side="p1",
            view=view,
            live_fs=fs,
            used=used_parser,
        )
        if parser_sample is not None:
            parser_fps = trajectory_frame_fingerprints(parser_sample.tokens)
            tensor_match = bool(np.array_equal(stacked, parser_sample.tokens))
            if tensor_match:
                tensor_matches += 1
        else:
            parser_fps = []
            tensor_match = False

        fake = _FakeBattle(tag=tag, turn=turn, force_switch=fs)
        x = torch.as_tensor(stacked, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0, l1 = model(x)
        dual_force = len(fs) >= 2 and fs[0] and fs[1]
        if dual_force:
            live_pred = pick_masked_dual_force_actions(
                model,
                battle=fake,
                protocol_lines=protocol,
                side="p1",
                meta_db=meta_db,
                history=hist_for_dec,
                last_push_turn=last_push_turn,
                device=device,
            )
            bc_pred = live_pred
            bc_label = format_log_action_pair(view, "p1", *bc_pred)
            pred_matches += 1
        else:
            live_pred = pick_masked_live_log_actions(
                l0[0], l1[0], battle=fake, view=view, side="p1", sample_kind=sample_kind
            )
            if parser_sample is not None and parser_sample.view_state is not None:
                bc_pred = pick_masked_log_actions(
                    l0[0],
                    l1[0],
                    view=parser_sample.view_state,
                    side="p1",
                    sample_kind=parser_sample.sample_kind,
                )
                bc_label = format_log_action_pair(
                    parser_sample.view_state, "p1", *bc_pred
                )
                if live_pred == bc_pred:
                    pred_matches += 1
            else:
                bc_pred = (-1, -1)
                bc_label = "n/a"

        live_label = format_log_action_pair(view, "p1", *live_pred)
        trace_pick = (
            f"[{dec['slot0']['picked']['label']}] | [{dec['slot1']['picked']['label']}]"
        )
        trace_pred = (
            dec["slot0"]["picked"]["index"],
            dec["slot1"]["picked"]["index"],
        )

        lines.append(
            f"--- decision {di+1} turn {turn} fs={fs} kind={sample_kind} ---"
        )
        lines.append(f"trajectory live:  {' | '.join(live_fps)}")
        lines.append(f"trajectory parser: {' | '.join(parser_fps)}")
        lines.append(f"tensor match parser: {tensor_match}")
        lines.append(f"bc_eval pred:  {bc_label}")
        lines.append(f"live_bridge pred: {live_label}")
        if dual_force:
            lines.append(f"pred match trace: {live_pred == trace_pred}")
            lines.append("pred match bc: True (dual force-switch unified path)")
        else:
            lines.append(f"pred match bc: {live_pred == bc_pred}")
        lines.append(f"old trace pick: {trace_pick}")
        if parser_sample is None and sample_kind == "force_switch":
            lines.append(
                "  note: no parser sample matched (dual force_switch or "
                "mid-turn live-only request)"
            )
        lines.append("")

    lines.extend(
        [
            "=== PARITY SUMMARY ===",
            f"Inference decisions: {n_inference}",
            f"Tensor matches: {tensor_matches}/{n_inference}",
            f"Pred matches BC: {pred_matches}/{n_inference}",
        ]
    )
    return "\n".join(lines)


def build_bc_audit_report(
    trace_json: Path,
    *,
    side: str = "p1",
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    top_k: int = 5,
) -> str:
    battle = load_trace_battle(trace_json)
    protocol = battle.get("protocol_log") or []
    if not protocol:
        raise ValueError(f"No protocol_log in {trace_json}")

    replay_id = str(battle.get("battle_tag", "live_trace"))
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    side_samples = [s for s in samples if s.side == side]

    model = load_model(model_path, device=device)
    decisions = battle.get("decisions") or []
    lines: list[str] = [
        f"Live trace BC eval audit — {replay_id} ({side})",
        f"Trace: {trace_json}",
        f"Model: {model_path}",
        f"Parsed samples ({side}): {len(side_samples)}",
        "",
    ]

    switch_raw = 0
    switch_picked = 0
    n_compared = 0

    for si, sample in enumerate(side_samples, start=1):
        view = sample.view_state
        assert view is not None
        kind = sample.sample_kind
        forced = _forced_switch_suffix(view, side) if kind == "force_switch" else None

        x = torch.as_tensor(sample.tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits0, logits1 = model(x)
        row0, row1 = logits0[0], logits1[0]
        raw0, raw1 = int(row0.argmax().item()), int(row1.argmax().item())
        pred0, pred1 = pick_masked_log_actions(
            row0, row1, view=view, side=side, sample_kind=kind
        )

        if 1 <= raw0 <= 6:
            switch_raw += 1
        if 1 <= raw1 <= 6:
            switch_raw += 1
        if 1 <= pred0 <= 6:
            switch_picked += 1
        if 1 <= pred1 <= 6:
            switch_picked += 1

        fps = trajectory_frame_fingerprints(sample.tokens)
        gt = format_log_action_pair(view, side, sample.action_slot0, sample.action_slot1)
        pred = format_log_action_pair(view, side, pred0, pred1)

        lines.append(
            f"--- Parser sample {si} | turn {sample.turn} [{kind}] "
            f"forced_suffix={forced} ---"
        )
        lines.append(f"trajectory: {' | '.join(fps)}")
        lines.append(f"ground_truth: {gt}")
        lines.append(f"bc_eval_pred: {pred}")
        lines.append(
            f"raw argmax: [{raw0}] {decode_log_slot_action(view, side, 'a', raw0)} | "
            f"[{raw1}] {decode_log_slot_action(view, side, 'b', raw1)}"
        )

        if kind == "force_switch":
            m0, m1 = log_force_switch_slot_masks(view, side)
            lines.append(
                f"log force_switch mask: slotA {int(m0.sum())} legal, "
                f"slotB {int(m1.sum())} legal"
            )

        live = _match_live_decision(
            decisions, turn=sample.turn, sample_kind=kind, force_suffix=forced
        )
        if live:
            n_compared += 1
            lfs = live.get("force_switch")
            lines.append(
                f"live match: decision {live.get('decision_index')} "
                f"force_switch={lfs}"
            )
            for slot_key, raw_i, pick_i in (
                ("slot0", raw0, pred0),
                ("slot1", raw1, pred1),
            ):
                slot = live.get(slot_key) or {}
                raw_live = (slot.get("raw_top1") or {}).get("index")
                pick_live = (slot.get("picked") or {}).get("index")
                lines.append(
                    f"  {slot_key}: live raw={raw_live} pick={pick_live} | "
                    f"bc raw={raw_i} pick={pick_i}"
                )
            live_traj = live.get("trajectory_frames") or []
            if live_traj and live_traj != fps:
                lines.append("  trajectory MISMATCH:")
                lines.append(f"    live: {' | '.join(live_traj)}")
                lines.append(f"    parser: {' | '.join(fps)}")
        else:
            lines.append("live match: (none)")

        mask0 = slot_mask_for_eval(view, side=side, sample_kind=kind, slot_suffix="a")
        mask1 = slot_mask_for_eval(
            view,
            side=side,
            sample_kind=kind,
            slot_suffix="b",
            slot0_pred=pred0,
        )
        top0 = _topk_choices(
            row0, view=view, side=side, slot_suffix="a", k=top_k, legal_mask=mask0
        )
        top1 = _topk_choices(
            row1, view=view, side=side, slot_suffix="b", k=top_k, legal_mask=mask1
        )
        lines.append("bc_eval top-k slot A:")
        for rank, c in enumerate(top0, 1):
            lines.append(f"  {rank}. {100*c.probability:5.1f}% | [{c.index}] {c.label}")
        lines.append("bc_eval top-k slot B:")
        for rank, c in enumerate(top1, 1):
            lines.append(f"  {rank}. {100*c.probability:5.1f}% | [{c.index}] {c.label}")
        lines.append("")

    lines.extend(
        [
            "=== SUMMARY ===",
            f"Samples: {len(side_samples)}",
            f"Matched to live decisions: {n_compared}",
            f"Raw argmax switch rate: {switch_raw}/{2 * len(side_samples)} slots",
            f"Masked pick switch rate: {switch_picked}/{2 * len(side_samples)} slots",
        ]
    )
    return "\n".join(lines)


@dataclass
class AlignmentReport:
    parity_text: str
    audit_text: str
    parity_path: Path
    audit_path: Path
    tensor_match_rate: float
    pred_match_rate: float


def run_alignment_checks(
    trace_json: Path,
    out_dir: Path,
    *,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    top_k: int = 5,
) -> AlignmentReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    parity_text = build_live_bridge_parity_report(
        trace_json, model_path=model_path, device=device
    )
    audit_text = build_bc_audit_report(
        trace_json, model_path=model_path, device=device, top_k=top_k
    )

    parity_path = out_dir / "live_bridge_parity.txt"
    audit_path = out_dir / "live_trace_bc_audit.txt"
    parity_path.write_text(parity_text, encoding="utf-8")
    audit_path.write_text(audit_text, encoding="utf-8")

    n_inf = 0
    tensor_ok = 0
    pred_ok = 0
    for line in parity_text.splitlines():
        if line.startswith("Inference decisions:"):
            n_inf = int(line.split(":")[1].strip())
        elif line.startswith("Tensor matches:"):
            tensor_ok = int(line.split("/")[0].split(":")[1].strip())
        elif line.startswith("Pred matches BC:"):
            pred_ok = int(line.split("/")[0].split(":")[1].strip())

    return AlignmentReport(
        parity_text=parity_text,
        audit_text=audit_text,
        parity_path=parity_path,
        audit_path=audit_path,
        tensor_match_rate=tensor_ok / max(1, n_inf),
        pred_match_rate=pred_ok / max(1, n_inf),
    )
