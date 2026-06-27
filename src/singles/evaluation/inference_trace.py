"""Format live singles inference traces for debugging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from poke_env.battle.battle import Battle
from poke_env.data import to_id_str
from src.singles.action_space_spec import decode_singles_action_index
from src.core.data.move_utils import canonical_move_list
from src.singles.bench_slots import (
    bench_switch_index_to_species_live,
    live_our_bench_mons,
)
from src.singles.log_action_codec import MEGA_BASE, MOVE_BASE, SWITCH_BASE
from src.doubles.evaluation.battle_inference_trace import (
    encode_request_protocol_line,
    format_battle_timeline,
    format_events_block,
    format_server_request_block,
    summarize_protocol_lines,
)
from src.singles.action_mask import singles_action_mask

__all__ = [
    "encode_request_protocol_line",
    "format_singles_live_action",
    "format_singles_live_battle_brief",
    "format_singles_trace_text",
    "summarize_server_request_singles",
    "topk_singles_live_choices",
    "write_trace_report",
]


def format_singles_live_action(battle: Battle, index: int) -> str:
    """Human-readable label for canonical singles action index."""
    spec = decode_singles_action_index(index)
    if spec.is_switch:
        bench_idx = index - SWITCH_BASE
        species = bench_switch_index_to_species_live(battle, bench_idx)
        return f"/choose switch {species} (bench {bench_idx + 1})"

    active = battle.active_pokemon
    if spec.move_slot is None or active is None:
        return f"action {index}"

    moves = canonical_move_list([to_id_str(m.id) for m in active.moves.values() if m])
    if not moves:
        moves = canonical_move_list([to_id_str(m.id) for m in battle.available_moves])
    move_name = moves[spec.move_slot] if spec.move_slot < len(moves) else f"move{spec.move_slot}"
    actor = to_id_str(active.species)
    flags: list[str] = []
    if spec.mega:
        flags.append("mega")
    if spec.tera:
        flags.append("tera")
    flag_text = f" {' '.join(flags)}" if flags else ""
    return f"/choose move {move_name}{flag_text}"


def format_singles_live_battle_brief(battle: Battle) -> str:
    lines: list[str] = [f"Turn {battle.turn}"]
    if battle.wait:
        lines.append("wait=True")
    if battle.force_switch:
        lines.append("force_switch=True")

    weather = next(iter(battle.weather), None)
    if weather is not None:
        lines.append(f"weather={to_id_str(str(weather))}")
    fields = [to_id_str(str(f)) for f in battle.fields]
    if fields:
        lines.append(f"field={', '.join(fields)}")

    for label, mon in (("our", battle.active_pokemon), ("opp", battle.opponent_active_pokemon)):
        if mon is None:
            continue
        moves = ", ".join(
            canonical_move_list([to_id_str(m.id) for m in mon.moves.values() if m])
        ) or "?"
        hp = f"{int(mon.current_hp or 0)}/{int(mon.max_hp or 0)}"
        lines.append(f"{label}: {to_id_str(mon.species)} ({hp}) [{moves}]")
    return "\n".join(lines)


def summarize_server_request_singles(battle: Battle) -> dict:
    req = getattr(battle, "_last_request", None) or {}
    side = req.get("side") or {}
    pokemon = side.get("pokemon") or []
    active_slots: list[dict] = []

    for i, act in enumerate(req.get("active") or []):
        ident = pokemon[i].get("ident", "") if i < len(pokemon) else ""
        moves = [
            {
                "id": m.get("id"),
                "move": m.get("move"),
                "pp": m.get("pp"),
                "maxpp": m.get("maxpp"),
                "disabled": bool(m.get("disabled", False)),
                "target": m.get("target"),
            }
            for m in act.get("moves") or []
        ]
        avail_ids = [to_id_str(m.id) for m in battle.available_moves]
        active_slots.append(
            {
                "slot": "active",
                "ident": ident,
                "moves": moves,
                "available_move_ids": avail_ids,
                "trapped": bool(act.get("trapped", False)),
                "can_mega_evo": bool(act.get("canMegaEvo", False)),
                "can_tera": bool(act.get("canTerastallize", False)),
            }
        )

    force_switch = req.get("forceSwitch")
    if isinstance(force_switch, list):
        fs = bool(force_switch[0]) if force_switch else False
    else:
        fs = bool(force_switch)

    return {
        "turn": int(battle.turn),
        "wait": bool(req.get("wait")),
        "force_switch": fs,
        "active": active_slots,
    }


def topk_singles_live_choices(
    logits_row: torch.Tensor,
    mask: torch.Tensor,
    battle: Battle,
    *,
    k: int = 5,
    legal_only: bool = True,
) -> list[dict]:
    row = logits_row.clone()
    if legal_only:
        row[~mask] = -float("inf")
    probs = torch.softmax(row, dim=-1)
    k = min(k, int((probs > 0).sum().item()) or k)
    k = min(k, probs.numel())
    if k <= 0:
        return []
    values, indices = torch.topk(probs, k=k)
    out: list[dict] = []
    for i in range(k):
        idx = int(indices[i].item())
        out.append(
            {
                "rank": i + 1,
                "index": idx,
                "probability": float(values[i].item()),
                "label": format_singles_live_action(battle, idx),
                "legal": bool(mask[idx].item()) if 0 <= idx < mask.shape[0] else False,
            }
        )
    return out


def action_record(battle: Battle, *, index: int, legal: bool | None = None) -> dict:
    return {
        "index": index,
        "label": format_singles_live_action(battle, index),
        "legal": legal,
    }


def format_singles_trace_text(
    battles: list[dict],
    *,
    include_protocol: bool = True,
) -> str:
    blocks: list[str] = []
    for battle in battles:
        header = (
            f"{'=' * 72}\n"
            f"Battle {battle.get('index', '?')} | {battle.get('battle_tag', '?')} | "
            f"{'WIN' if battle.get('won') else 'LOSS'} | turns={battle.get('turn', '?')}\n"
            f"Brought: {battle.get('brought', [])}\n"
            f"Opponent brought: {battle.get('opponent_brought', [])}\n"
            f"{'=' * 72}"
        )
        blocks.append(header)

        preview = battle.get("teampreview")
        if preview:
            blocks.append(f"[teampreview] {preview}")

        for step in battle.get("decisions", []):
            since = step.get("battle_events_since_last")
            if since:
                blocks.append(
                    format_events_block(since, title="What happened since last decision:")
                )

            if step.get("kind") == "wait":
                blocks.append(
                    f"\n--- decision {step.get('decision_index')} | turn {step.get('turn')} "
                    f"[wait] ---\nDefaultBattleOrder()"
                )
                continue

            fb = " FALLBACK" if step.get("fallback") else ""
            blocks.append(
                f"\n--- decision {step.get('decision_index')} | turn {step.get('turn')} "
                f"force_switch={step.get('force_switch')}{fb} ---"
            )
            blocks.append(step.get("state_text", ""))
            server_request = step.get("server_request")
            if server_request is not None:
                blocks.append(format_server_request_block(server_request))
            traj_frames = step.get("trajectory_frames")
            if traj_frames:
                blocks.append(f"trajectory: {' | '.join(traj_frames)}")

            raw = step.get("raw_top1", {})
            picked = step.get("picked", {})
            blocks.append("Action:")
            blocks.append(
                f"  raw top-1: [{raw.get('index')}] {raw.get('label')} "
                f"(legal={raw.get('legal')})"
            )
            blocks.append(
                f"  picked:    [{picked.get('index')}] {picked.get('label')}"
            )
            blocks.append("  top-k (legal):")
            for choice in step.get("topk_legal", []):
                mark = " *" if choice.get("index") == picked.get("index") else ""
                blocks.append(
                    f"    {choice.get('rank')}. {100 * choice.get('probability', 0):5.1f}% | "
                    f"{choice.get('label')}{mark}"
                )

        timeline = battle.get("battle_timeline")
        if timeline:
            blocks.append("\n--- Full battle log (by turn) ---")
            blocks.append(timeline)

        if include_protocol and battle.get("protocol_log"):
            blocks.append("\n--- Showdown protocol (agent perspective) ---")
            blocks.extend(battle["protocol_log"])

        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def write_trace_report(
    battles: list[dict],
    out_dir: Path,
    *,
    model_path: Path | str | None = None,
    opponent: str = "maxdamage",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "format": "singles",
        "model": str(model_path) if model_path else None,
        "opponent": opponent,
        "n_battles": len(battles),
        "battles": battles,
    }
    json_path = out_dir / f"inference_trace_{stamp}.json"
    txt_path = out_dir / f"inference_trace_{stamp}.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(format_singles_trace_text(battles), encoding="utf-8")
    latest_json = out_dir / "inference_trace_latest.json"
    latest_txt = out_dir / "inference_trace_latest.txt"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_txt.write_text(format_singles_trace_text(battles), encoding="utf-8")
    return txt_path, json_path
