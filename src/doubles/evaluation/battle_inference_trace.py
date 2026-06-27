"""Format live-battle inference traces for debugging."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str

from src.core.data.move_utils import canonical_move_list
from src.doubles.battle.move_order import format_live_canonical_action

_SPECIES_FROM_DETAILS = re.compile(r"^([^,]+)")

# Protocol lines we skip in the human battle narrative.
_SKIP_PROTOCOL_CMDS = frozenset(
    {
        "",
        "request",
        "upkeep",
        "t:",
        "init",
        "player",
        "gametype",
        "gen",
        "tier",
        "rule",
        "teamsize",
        "start",
        "poke",
        "title",
        "showteam",
        "clearpoke",
        "c:",
        "inactive",
        "raw",
        "deinit",
    }
)


def _actor_name(ident: str) -> str:
    return ident.split(":")[-1].strip() if ident else "?"


def _species_label(details: str) -> str:
    m = _SPECIES_FROM_DETAILS.match(details.strip())
    return to_id_str(m.group(1)) if m else to_id_str(details.split(",")[0])


def summarize_protocol_line(line: str) -> str | None:
    """Convert one Showdown protocol line to a short human-readable event."""
    if not line.startswith("|"):
        return None
    parts = line.split("|")
    if len(parts) < 2:
        return None
    cmd = parts[1]

    if cmd in _SKIP_PROTOCOL_CMDS:
        return None
    if cmd.startswith("-anim") or cmd.startswith("-hint"):
        return None

    if cmd == "turn" and len(parts) >= 3:
        return f"__TURN__{parts[2]}"
    if cmd == "move" and len(parts) >= 4:
        actor = _actor_name(parts[2])
        move = to_id_str(parts[3])
        target = _actor_name(parts[4]) if len(parts) >= 5 and parts[4] else ""
        if target:
            return f"{actor} used {move} -> {target}"
        return f"{actor} used {move}"
    if cmd == "switch" and len(parts) >= 4:
        actor = _actor_name(parts[2])
        species = _species_label(parts[3])
        if len(parts) >= 5 and "[from]" in parts[-1]:
            return f"{actor} pivoted to {species}"
        return f"{actor} sent out {species}"
    if cmd == "faint" and len(parts) >= 3:
        return f"{_actor_name(parts[2])} fainted"
    if cmd == "win" and len(parts) >= 3:
        return f"{parts[2]} won the battle"
    if cmd == "-damage" and len(parts) >= 4:
        mon = _actor_name(parts[2])
        hp = parts[3].strip()
        if "fnt" in hp:
            return f"{mon} was knocked out"
        return f"{mon} took damage ({hp})"
    if cmd == "-heal" and len(parts) >= 4:
        return f"{_actor_name(parts[2])} healed ({parts[3].strip()})"
    if cmd == "-status" and len(parts) >= 4:
        return f"{_actor_name(parts[2])} is {parts[3]}"
    if cmd == "-weather" and len(parts) >= 3:
        return f"Weather: {parts[2]}"
    if cmd == "-fieldstart" and len(parts) >= 3:
        return f"Field started: {parts[2]}"
    if cmd == "-fieldend" and len(parts) >= 3:
        return f"Field ended: {parts[2]}"
    if cmd == "-sidestart" and len(parts) >= 4:
        return f"{parts[2]} side: {parts[3]}"
    if cmd == "-sideend" and len(parts) >= 4:
        return f"{parts[2]} side ended: {parts[3]}"
    if cmd == "-ability" and len(parts) >= 4:
        return f"{_actor_name(parts[2])}'s {parts[3]}"
    if cmd == "-terastallize" and len(parts) >= 4:
        return f"{_actor_name(parts[2])} Terastallized into {parts[3]}"
    if cmd == "-mega" and len(parts) >= 3:
        return f"{_actor_name(parts[2])} Mega Evolved"
    if cmd == "cant" and len(parts) >= 4:
        return f"{_actor_name(parts[2])} can't move ({parts[3]})"
    return None


def summarize_protocol_lines(lines: list[str]) -> list[str]:
    """Flatten protocol lines into readable battle events."""
    events: list[str] = []
    for line in lines:
        event = summarize_protocol_line(line)
        if event:
            events.append(event)
    return events


def format_battle_timeline(events: list[str]) -> str:
    """Group parsed events by turn for display."""
    if not events:
        return "(no battle events recorded)"
    lines: list[str] = []
    current_turn = "?"
    for event in events:
        if event.startswith("__TURN__"):
            current_turn = event.split("__TURN__", 1)[1]
            lines.append(f"Turn {current_turn}:")
            continue
        lines.append(f"  {_display_event(event)}")
    return "\n".join(lines)


def _display_event(event: str) -> str:
    if event.startswith("__TURN__"):
        return f"--- Turn {event.split('__TURN__', 1)[1]} ---"
    return event


def format_events_block(events: list[str], *, title: str) -> str:
    if not events:
        return ""
    body = "\n".join(f"  {_display_event(e)}" for e in events)
    return f"{title}\n{body}"


def summarize_server_request(battle: DoubleBattle) -> dict:
    """
    Compact snapshot of the Showdown |request| that drove this decision.

    Includes per-move pp/maxpp/disabled from the server plus poke-env
    available_moves after parsing (what the action mask uses).
    """
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
        avail_ids: list[str] = []
        if i < len(battle.available_moves):
            avail_ids = [to_id_str(m.id) for m in battle.available_moves[i]]
        active_slots.append(
            {
                "slot": "A" if i == 0 else "B",
                "ident": ident,
                "moves": moves,
                "available_move_ids": avail_ids,
                "trapped": bool(act.get("trapped", False)),
                "can_mega_evo": bool(act.get("canMegaEvo", False)),
                "can_tera": bool(act.get("canTerastallize", False)),
            }
        )

    return {
        "turn": int(battle.turn),
        "wait": bool(req.get("wait")),
        "force_switch": list(req.get("forceSwitch") or [False, False]),
        "active": active_slots,
    }


def format_server_request_block(request_summary: dict | None) -> str:
    """Human-readable server request lines for inference trace text."""
    if not request_summary:
        return "Server request: (not captured)"
    lines = [
        "Server request:",
        f"  wait={request_summary.get('wait')} "
        f"force_switch={request_summary.get('force_switch')}",
    ]
    for slot in request_summary.get("active") or []:
        label = slot.get("slot", "?")
        ident = slot.get("ident") or "?"
        lines.append(f"  Slot {label} ({ident}):")
        avail = slot.get("available_move_ids") or []
        lines.append(f"    poke-env available: {avail or '(none)'}")
        for move in slot.get("moves") or []:
            mid = move.get("id") or "?"
            pp = move.get("pp")
            maxpp = move.get("maxpp")
            pp_txt = f"{pp}/{maxpp}" if pp is not None and maxpp is not None else "?"
            disabled = move.get("disabled", False)
            target = move.get("target") or "?"
            flag = " DISABLED" if disabled else ""
            lines.append(
                f"    {mid}: pp={pp_txt} target={target}{flag}"
            )
    return "\n".join(lines)


def encode_request_protocol_line(request: dict) -> str:
    """Serialize a parsed Showdown request for protocol_log storage."""
    return "|request|" + json.dumps(request, separators=(",", ":"))


def parse_request_protocol_line(line: str) -> dict | None:
    """Parse a stored |request|{json} protocol line, if present."""
    if not line.startswith("|request|"):
        return None
    body = line[len("|request|") :]
    if not body or body == "...":
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def format_live_battle_brief(battle: DoubleBattle) -> str:
    """Compact field + active snapshot at decision time."""
    lines: list[str] = [f"Turn {battle.turn}"]
    if battle.wait:
        lines.append("wait=True")
    if any(battle.force_switch):
        lines.append(f"force_switch={list(battle.force_switch)}")

    weather = next(iter(battle.weather), None)
    if weather is not None:
        lines.append(f"weather={to_id_str(str(weather))}")
    fields = [to_id_str(str(f)) for f in battle.fields]
    if fields:
        lines.append(f"field={', '.join(fields)}")

    for label, mons in (
        ("our", battle.active_pokemon),
        ("opp", battle.opponent_active_pokemon),
    ):
        parts: list[str] = []
        for mon in mons:
            if mon is None:
                continue
            moves = ", ".join(
                canonical_move_list([to_id_str(m.id) for m in mon.moves.values() if m])
            ) or "?"
            hp = f"{int(mon.current_hp or 0)}/{int(mon.max_hp or 0)}"
            parts.append(f"{to_id_str(mon.species)} ({hp}) [{moves}]")
        if parts:
            lines.append(f"{label}: {' | '.join(parts)}")
    return "\n".join(lines)


def topk_live_choices(
    logits_row: torch.Tensor,
    mask: torch.Tensor,
    battle: DoubleBattle,
    pos: int,
    *,
    k: int = 5,
    legal_only: bool = True,
) -> list[dict]:
    """Top-k actions with softmax probabilities."""
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
                "label": format_live_canonical_action(battle, pos, idx),
                "legal": bool(mask[idx].item()) if 0 <= idx < mask.shape[0] else False,
            }
        )
    return out


def action_record(
    battle: DoubleBattle,
    pos: int,
    *,
    index: int,
    legal: bool | None = None,
) -> dict:
    return {
        "index": index,
        "label": format_live_canonical_action(battle, pos, index),
        "legal": legal,
    }


def format_trace_text(
    battles: list[dict],
    *,
    include_protocol: bool = True,
) -> str:
    """Render inference traces as a readable text log."""
    blocks: list[str] = []
    for battle in battles:
        header = (
            f"{'=' * 72}\n"
            f"Battle {battle.get('index', '?')} | {battle.get('battle_tag', '?')} | "
            f"{'WIN' if battle.get('won') else 'LOSS'} | turns={battle.get('turn', '?')}\n"
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

            fb = " FALLBACK" if step.get("any_fallback") else ""
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
            for slot_key, slot_name in (("slot0", "Slot A"), ("slot1", "Slot B")):
                slot = step.get(slot_key, {})
                raw = slot.get("raw_top1", {})
                picked = slot.get("picked", {})
                blocks.append(f"{slot_name}:")
                blocks.append(
                    f"  raw top-1: [{raw.get('index')}] {raw.get('label')} "
                    f"(legal={raw.get('legal')})"
                )
                blocks.append(
                    f"  picked:    [{picked.get('index')}] {picked.get('label')}"
                )
                blocks.append("  top-k (legal):")
                for choice in slot.get("topk_legal", []):
                    mark = " *" if choice.get("index") == picked.get("index") else ""
                    blocks.append(
                        f"    {choice.get('rank')}. {100 * choice.get('probability', 0):5.1f}% | "
                        f"{choice.get('label')}{mark}"
                    )
            blocks.append(
                f"Joint: [{step.get('slot0', {}).get('picked', {}).get('label')}] | "
                f"[{step.get('slot1', {}).get('picked', {}).get('label')}]"
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
    policy_source: str = "bc",
    checkpoint: Path | str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path) if model_path else None,
        "policy_source": policy_source,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "opponent": opponent,
        "n_battles": len(battles),
        "battles": battles,
    }
    json_path = out_dir / f"inference_trace_{stamp}.json"
    txt_path = out_dir / f"inference_trace_{stamp}.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(format_trace_text(battles), encoding="utf-8")
    latest_json = out_dir / "inference_trace_latest.json"
    latest_txt = out_dir / "inference_trace_latest.txt"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_txt.write_text(format_trace_text(battles), encoding="utf-8")
    return txt_path, json_path
