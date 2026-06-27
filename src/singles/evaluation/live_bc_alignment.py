"""BC vs live inference alignment for singles (trace protocol -> parser -> BC eval)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config.settings import SINGLES_BC_DATASET_PATH, SINGLES_BC_MODEL_PATH
from src.core.data.state_tokenizer import (
    N_TOKENS,
    TRAJECTORY_DEPTH,
    push_trajectory,
    stack_trajectory,
    trajectory_frame_fingerprints,
)
from src.singles.battle.live_log_bridge import encode_live_as_log
from src.core.model.transformer_bot import load_model
from src.singles.evaluation.bc_examples import _topk_choices
from src.singles.evaluation.tensor_diff import format_tensor_diff
from src.singles.log_action_codec import format_singles_log_action, training_action_mask
from src.singles.log_action_mask import (
    singles_force_switch_mask,
    singles_mask_for_eval,
    singles_turn_mask,
)
from src.singles.replay_parser import SinglesParsedSample, parse_singles_replay_log


def tensor_digest(tokens: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(tokens).tobytes()).hexdigest()[:16]


class _ReplayBattle:
    """Minimal battle handle for encode_live_as_log during trace audit."""

    def __init__(self, *, tag: str, turn: int, force_switch: bool):
        self.battle_tag = tag
        self.turn = turn
        self.force_switch = force_switch
        self.team = {}


def _parser_t0_frame(stacked: np.ndarray) -> np.ndarray:
    return np.asarray(stacked, dtype=np.int64).reshape(-1, stacked.shape[-1])[-N_TOKENS:]


def _snapshot_from_protocol(
    protocol: list[str],
    *,
    tag: str,
    turn: int,
    force_switch: bool,
    side: str,
) -> np.ndarray | None:
    battle = _ReplayBattle(tag=tag, turn=turn, force_switch=force_switch)
    encoded = encode_live_as_log(battle, protocol_lines=protocol, side=side)
    if encoded is None:
        return None
    return encoded[0]


def _protocol_for_live_encoding(
    protocol: list[str],
    *,
    turn: int,
    teampreview_cmd: str | None = None,
    protocol_len: int | None = None,
) -> list[str]:
    """Mirror SinglesTransformerPlayer._protocol_for_encoding on a protocol slice."""
    from src.singles.battle.live_log_bridge import teampreview_protocol_line

    lines = list(protocol[:protocol_len] if protocol_len is not None else protocol)
    if teampreview_cmd:
        line = teampreview_protocol_line("p1", teampreview_cmd)
        if line not in lines:
            lines.append(line)
    if turn >= 1:
        turn_line = f"|turn|{turn}"
        if not any(ln.startswith(turn_line) for ln in lines):
            lines.append(turn_line)
    return lines


def _protocol_through_flush_turn(
    protocol: list[str],
    flushed_turn: int,
    protocol_len: int,
) -> list[str]:
    """Protocol prefix through the end of ``flushed_turn`` (before ``|turn|N+1|``)."""
    from src.singles.battle.live_log_bridge import protocol_through_flush_turn

    return protocol_through_flush_turn(protocol, flushed_turn, protocol_len)


def _turns_announced_in_prefix(protocol: list[str], protocol_len: int) -> list[int]:
    turns: list[int] = []
    for line in protocol[:protocol_len]:
        if line.startswith("|turn|"):
            try:
                turns.append(int(line.split("|")[2]))
            except (IndexError, ValueError):
                pass
    return turns


def _simulate_live_trajectory(
    protocol: list[str],
    decisions: list[dict],
    *,
    tag: str,
    side: str,
    teampreview_cmd: str | None = None,
) -> dict[int, np.ndarray]:
    """
    Replay live trajectory stacking (deferred commit) for each inference decision.

    Mirrors SinglesTransformerPlayer turn-boundary flush + _stacked_input.
    """
    from src.singles.battle.live_log_bridge import (
        snapshot_for_turn_flush,
        turn_trajectory_should_push,
    )

    hist: list[np.ndarray] = []
    last_flushed_turn = 0
    out: dict[int, np.ndarray] = {}

    for dec in decisions:
        if dec.get("kind") != "inference":
            continue
        di = int(dec.get("decision_index", -1))
        turn = int(dec["turn"])
        fs = bool(dec.get("force_switch"))
        plen = dec.get("protocol_len")
        if plen is None:
            plen = len(protocol)
        announced = _turns_announced_in_prefix(protocol, int(plen))
        latest_turn_mark = max(announced) if announced else 0
        while last_flushed_turn < latest_turn_mark - 1:
            flush_turn = last_flushed_turn + 1
            proto_flush = _protocol_for_live_encoding(
                _protocol_through_flush_turn(protocol, flush_turn, int(plen)),
                turn=flush_turn,
                teampreview_cmd=teampreview_cmd,
                protocol_len=None,
            )
            if turn_trajectory_should_push(
                proto_flush,
                turn=flush_turn,
                side=side,
                replay_id=tag,
            ):
                snap = snapshot_for_turn_flush(
                    proto_flush,
                    turn=flush_turn,
                    side=side,
                    replay_id=tag,
                    meta_db=None,
                )
                if snap is not None:
                    push_trajectory(
                        hist,
                        snap,
                        depth=TRAJECTORY_DEPTH,
                        maxlen=TRAJECTORY_DEPTH,
                    )
            last_flushed_turn = flush_turn

        battle = _ReplayBattle(tag=tag, turn=turn, force_switch=fs)
        proto = _protocol_for_live_encoding(
            protocol,
            turn=turn,
            teampreview_cmd=teampreview_cmd,
            protocol_len=dec.get("protocol_len"),
        )
        encoded = encode_live_as_log(battle, protocol_lines=proto, side=side)
        if encoded is None:
            continue
        snap, _, _ = encoded

        stacked = stack_trajectory(list(hist), snap, depth=TRAJECTORY_DEPTH)
        out[di] = stacked

    return out


def load_trace_battle(trace_json: Path) -> dict:
    data = json.loads(trace_json.read_text(encoding="utf-8"))
    battles = data.get("battles") or [data]
    return battles[0]


def _match_live_decision(
    decisions: list[dict],
    *,
    turn: int,
    sample_kind: str,
    used: set[int],
) -> dict | None:
    for idx, dec in enumerate(decisions):
        if idx in used:
            continue
        if dec.get("kind") != "inference":
            continue
        if int(dec.get("turn", -1)) != turn:
            continue
        fs = bool(dec.get("force_switch"))
        if sample_kind == "force_switch" and fs:
            used.add(idx)
            return dec
        if sample_kind == "turn" and not fs:
            used.add(idx)
            return dec
    return None


def _find_parser_sample(
    samples: list[SinglesParsedSample],
    *,
    turn: int,
    sample_kind: str,
    side: str,
    used: set[int],
) -> SinglesParsedSample | None:
    for idx, sample in enumerate(samples):
        if idx in used:
            continue
        if sample.side != side:
            continue
        if sample.turn != turn or sample.sample_kind != sample_kind:
            continue
        used.add(idx)
        return sample
    return None


def pick_masked_bc_action(
    logits_row: torch.Tensor,
    view,
    *,
    side: str,
    sample_kind: str,
) -> int:
    """Same masked argmax path as ``generate_bc_examples``."""
    mask = singles_mask_for_eval(view, side=side, sample_kind=sample_kind)
    row = logits_row.clone()
    if mask is not None and mask.any():
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=row.device)
        row[~mask_t] = -float("inf")
        return int(row.argmax().item())
    return int(row.argmax().item())


def _mask_agreement(
    view,
    *,
    side: str,
    sample_kind: str,
    ground_truth: int,
    stored_mask: np.ndarray | None,
) -> tuple[bool, bool]:
    """Compare BC-eval mask, base training mask, and stored parse-time mask."""
    eval_mask = singles_mask_for_eval(view, side=side, sample_kind=sample_kind)
    if sample_kind == "force_switch":
        base_mask = singles_force_switch_mask(view, side)
    else:
        base_mask = singles_turn_mask(view, side)
    train_mask = np.array(
        training_action_mask(view, side, ground_truth=ground_truth, sample_kind=sample_kind),
        dtype=bool,
    )
    eval_matches_base = eval_mask is not None and bool(np.array_equal(eval_mask, base_mask))
    stored_matches_train = stored_mask is not None and bool(np.array_equal(stored_mask, train_mask))
    return eval_matches_base, stored_matches_train


def build_bc_audit_report(
    trace_json: Path,
    *,
    side: str = "p1",
    model_path: Path = SINGLES_BC_MODEL_PATH,
    device: str = "cpu",
    top_k: int = 5,
) -> str:
    """Parse trace protocol as training data and compare BC-eval preds to live trace."""
    battle = load_trace_battle(trace_json)
    protocol = battle.get("protocol_log") or []
    if not protocol:
        raise ValueError(f"No protocol_log in {trace_json}")

    replay_id = str(battle.get("battle_tag", "live_trace"))
    samples = parse_singles_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    side_samples = [s for s in samples if s.side == side]
    decisions = [d for d in (battle.get("decisions") or []) if d.get("kind") == "inference"]

    model = load_model(model_path, device=device)
    used_parser: set[int] = set()
    used_live: set[int] = set()

    lines: list[str] = [
        f"Singles live trace BC eval audit — {replay_id} ({side})",
        f"Trace: {trace_json}",
        f"Model: {model_path}",
        f"Parsed samples ({side}): {len(side_samples)}",
        f"Live inference decisions: {len(decisions)}",
        "",
    ]

    pred_matches = 0
    traj_matches = 0
    digest_matches = 0
    recomputed_digest_matches = 0
    snapshot_matches = 0
    tensor_matches = 0
    mask_eval_ok = 0
    n_compared = 0
    protocol = battle.get("protocol_log") or []
    simulated = _simulate_live_trajectory(
        protocol,
        decisions,
        tag=replay_id,
        side=side,
        teampreview_cmd=battle.get("teampreview"),
    )

    for si, sample in enumerate(side_samples, start=1):
        view = sample.view_state
        assert view is not None
        kind = sample.sample_kind

        live = _match_live_decision(
            decisions,
            turn=sample.turn,
            sample_kind=kind,
            used=used_live,
        )

        x = torch.as_tensor(sample.tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        row = logits.squeeze(0)
        raw = int(row.argmax().item())
        pred = pick_masked_bc_action(row, view, side=side, sample_kind=kind)

        fps = trajectory_frame_fingerprints(sample.tokens)
        parser_digest = tensor_digest(sample.tokens)
        parser_t0 = _parser_t0_frame(sample.tokens)
        bridge_snap = _snapshot_from_protocol(
            protocol,
            tag=replay_id,
            turn=sample.turn,
            force_switch=(kind == "force_switch"),
            side=side,
        )
        snap_match = bridge_snap is not None and bool(
            np.array_equal(parser_t0, bridge_snap)
        )
        if snap_match:
            snapshot_matches += 1
        gt = format_singles_log_action(view, side, sample.action)
        pred_label = format_singles_log_action(view, side, pred)

        eval_m_ok, stored_m_ok = _mask_agreement(
            view,
            side=side,
            sample_kind=kind,
            ground_truth=sample.action,
            stored_mask=sample.action_mask,
        )
        if eval_m_ok:
            mask_eval_ok += 1

        lines.append(
            f"--- Parser sample {si} | turn {sample.turn} [{kind}] ---"
        )
        lines.append(f"trajectory parser: {' | '.join(fps)}")
        lines.append(f"tensor digest parser: {parser_digest}")
        lines.append(f"snapshot t-0 match (bridge vs parser): {snap_match}")
        if bridge_snap is not None and not snap_match:
            lines.append("snapshot diff (parser t-0 - bridge):")
            for diff_line in format_tensor_diff(bridge_snap, parser_t0).splitlines():
                lines.append(f"  {diff_line}")
        lines.append(f"ground_truth: {gt}")
        lines.append(f"bc_eval_pred: {pred_label} [{pred}]")
        lines.append(f"raw argmax: [{raw}] {format_singles_log_action(view, side, raw)}")
        lines.append(f"eval_mask matches base training mask: {eval_m_ok}")
        lines.append(f"stored_mask matches training: {stored_m_ok}")

        topk = _topk_choices(
            row,
            view=view,
            side=side,
            k=top_k,
            legal_mask=singles_mask_for_eval(view, side=side, sample_kind=kind),
        )
        lines.append("top-k (BC eval path):")
        for c in topk:
            mark = " *" if c.index == pred else ""
            lines.append(f"  {100 * c.probability:5.1f}% | [{c.index}] {c.label}{mark}")

        if live:
            n_compared += 1
            live_fps = live.get("trajectory_frames") or []
            traj_match = live_fps == fps
            if traj_match:
                traj_matches += 1
            live_digest = live.get("token_digest")
            digest_match = live_digest is not None and live_digest == parser_digest
            if digest_match:
                digest_matches += 1
            di = int(live.get("decision_index", -1))
            sim_stacked = simulated.get(di)
            sim_digest = (
                tensor_digest(sim_stacked) if sim_stacked is not None else None
            )
            recomputed_match = sim_digest is not None and sim_digest == parser_digest
            if recomputed_match:
                recomputed_digest_matches += 1
            tensor_match = recomputed_match
            if tensor_match:
                tensor_matches += 1
            trace_pick = int(live.get("picked", {}).get("index", -1))
            trace_raw = int(live.get("raw_top1", {}).get("index", -1))
            pred_match = pred == trace_pick
            if pred_match:
                pred_matches += 1
            lines.append(
                f"live decision {live.get('decision_index')}: "
                f"force_switch={live.get('force_switch')}"
            )
            lines.append(f"trajectory live:   {' | '.join(live_fps)}")
            lines.append(f"trajectory match:  {traj_match}")
            if live_digest:
                lines.append(f"tensor digest live (trace):   {live_digest}")
                lines.append(f"tensor digest trace match: {digest_match}")
            else:
                lines.append("tensor digest live (trace):   (not recorded — re-run trace)")
            if sim_digest:
                lines.append(f"tensor digest recomputed:     {sim_digest}")
                lines.append(f"tensor digest recomputed match: {recomputed_match}")
            lines.append(
                f"trace raw/picked: [{trace_raw}] / [{trace_pick}] "
                f"{live.get('picked', {}).get('label', '')}"
            )
            lines.append(f"bc_eval == trace picked: {pred_match}")
            live_stacked = live.get("stacked_tokens")
            if sim_stacked is not None and not recomputed_match:
                lines.append("tensor diff (recomputed live - parser):")
                for diff_line in format_tensor_diff(sim_stacked, sample.tokens).splitlines():
                    lines.append(f"  {diff_line}")
            elif live_stacked is not None and not digest_match:
                lines.append("tensor diff (trace recorded - parser):")
                live_arr = np.asarray(live_stacked, dtype=np.int64)
                for diff_line in format_tensor_diff(live_arr, sample.tokens).splitlines():
                    lines.append(f"  {diff_line}")
            if recomputed_match and not pred_match:
                lines.append("  note: same tensor but pred mismatch — check model/device")
            elif not recomputed_match and not pred_match:
                lines.append("  note: tensor mismatch (trajectory/history) explains pred gap")
            elif digest_match is False and recomputed_match and live_digest:
                lines.append(
                    "  note: recomputed stack matches parser; trace digest is stale (re-run trace)"
                )
            if not recomputed_match and trace_raw == pred:
                lines.append("  note: BC matches trace raw top-1; tensors differ from parser")
        else:
            lines.append("live match: none")
        lines.append("")

    unmatched_live = len(decisions) - len(used_live)
    if unmatched_live:
        lines.append(f"--- Unmatched live decisions: {unmatched_live} ---")
        for idx, dec in enumerate(decisions):
            if idx in used_live:
                continue
            lines.append(
                f"  decision {dec.get('decision_index')} turn {dec.get('turn')} "
                f"fs={dec.get('force_switch')} picked={dec.get('picked', {}).get('index')}"
            )
        lines.append("")

    lines.extend(
        [
            "=== AUDIT SUMMARY ===",
            f"Parser samples: {len(side_samples)}",
            f"Live inference decisions: {len(decisions)}",
            f"Unmatched live decisions: {unmatched_live}",
            f"Matched to parser samples: {n_compared}",
            f"Trajectory fingerprint matches: {traj_matches}/{n_compared}",
            f"Snapshot t-0 matches (encode_live_as_log): {snapshot_matches}/{len(side_samples)}",
            f"Tensor digest matches (trace recorded): {digest_matches}/{n_compared}",
            f"Tensor digest matches (recomputed live stack): {recomputed_digest_matches}/{n_compared}",
            f"BC eval pred == trace picked: {pred_matches}/{n_compared}",
            f"Eval mask agrees with base training mask: {mask_eval_ok}/{len(side_samples)}",
        ]
    )
    return "\n".join(lines)


def build_dataset_crosscheck_report(
    trace_json: Path,
    *,
    dataset_path: Path = SINGLES_BC_DATASET_PATH,
    side: str = "p1",
) -> str:
    """When trace battle exists in the BC dataset, verify parser tensors match stored rows."""
    battle = load_trace_battle(trace_json)
    protocol = battle.get("protocol_log") or []
    replay_id = str(battle.get("battle_tag", "live_trace"))
    if not protocol:
        return f"No protocol_log in {trace_json}"

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    meta: list[dict] = data["meta"]
    tokens = np.asarray(data["token_ids"])

    parsed = parse_singles_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    p1 = [s for s in parsed if s.side == side]

    lines = [
        f"Dataset cross-check — {replay_id}",
        f"Dataset rows for replay_id: {sum(1 for m in meta if m.get('replay_id') == replay_id)}",
        "",
    ]
    if not any(m.get("replay_id") == replay_id for m in meta):
        lines.append("Live trace battle not in training dataset (expected for Showdown tags).")
        return "\n".join(lines)

    ds_hits = 0
    tensor_hits = 0
    for sample in p1:
        for idx, m in enumerate(meta):
            if m.get("replay_id") != replay_id:
                continue
            if m.get("turn") != sample.turn or m.get("sample_kind") != sample.sample_kind:
                continue
            if m.get("side") != side:
                continue
            ds_hits += 1
            match = bool(np.array_equal(tokens[idx], sample.tokens))
            if match:
                tensor_hits += 1
            lines.append(
                f"turn {sample.turn} [{sample.sample_kind}]: "
                f"dataset idx {idx} tensor_match={match}"
            )
            break

    lines.extend(
        [
            "",
            f"Matched samples: {ds_hits}/{len(p1)}",
            f"Tensor matches: {tensor_hits}/{ds_hits}",
        ]
    )
    return "\n".join(lines)


@dataclass
class AlignmentReport:
    audit_text: str
    dataset_text: str
    audit_path: Path
    dataset_path: Path
    pred_match_rate: float
    traj_match_rate: float
    digest_match_rate: float
    recomputed_digest_match_rate: float
    n_compared: int


def run_alignment_checks(
    trace_json: Path,
    out_dir: Path,
    *,
    model_path: Path = SINGLES_BC_MODEL_PATH,
    dataset_path: Path = SINGLES_BC_DATASET_PATH,
    device: str = "cpu",
    top_k: int = 5,
    side: str = "p1",
) -> AlignmentReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_text = build_bc_audit_report(
        trace_json,
        side=side,
        model_path=model_path,
        device=device,
        top_k=top_k,
    )
    dataset_text = build_dataset_crosscheck_report(
        trace_json,
        dataset_path=dataset_path,
        side=side,
    )

    audit_path = out_dir / "live_trace_bc_audit.txt"
    ds_path = out_dir / "dataset_crosscheck.txt"
    audit_path.write_text(audit_text, encoding="utf-8")
    ds_path.write_text(dataset_text, encoding="utf-8")

    n_compared = 0
    pred_ok = 0
    traj_ok = 0
    digest_ok = 0
    recomputed_ok = 0
    for line in audit_text.splitlines():
        if line.startswith("Matched to parser samples:"):
            n_compared = int(line.split(":")[1].strip())
        elif line.startswith("Matched to live decisions:"):
            n_compared = int(line.split(":")[1].strip())
        elif line.startswith("Tensor digest matches (trace recorded):"):
            digest_ok = int(line.split("/")[0].split(":")[1].strip())
        elif line.startswith("Tensor digest matches (recomputed live stack):"):
            recomputed_ok = int(line.split("/")[0].split(":")[1].strip())
        elif line.startswith("BC eval pred == trace picked:"):
            parts = line.split(":")[1].strip().split("/")
            pred_ok = int(parts[0].strip())
        elif line.startswith("Trajectory fingerprint matches:"):
            traj_ok = int(line.split("/")[0].split(":")[1].strip())

    return AlignmentReport(
        audit_text=audit_text,
        dataset_text=dataset_text,
        audit_path=audit_path,
        dataset_path=ds_path,
        pred_match_rate=pred_ok / max(1, n_compared),
        traj_match_rate=traj_ok / max(1, n_compared),
        digest_match_rate=digest_ok / max(1, n_compared),
        recomputed_digest_match_rate=recomputed_ok / max(1, n_compared),
        n_compared=n_compared,
    )
