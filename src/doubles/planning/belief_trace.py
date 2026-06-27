"""Format BeliefState snapshots for human review and JSON logs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from poke_env.data import to_id_str

from config.settings import LOGS_DIR, ROOT_DIR
from src.doubles.planning.observation_tracker import BattleSnapshot

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle

    from src.doubles.planning.belief_state import BeliefPokemon, BeliefState
    from src.doubles.planning.macro_strategist import GamePlan


def _top_n(dist, n: int = 3) -> list[tuple[str, float]]:
    opts = dist.normalized().options if hasattr(dist, "normalized") else dist.options
    ranked = sorted(opts.items(), key=lambda x: -x[1])
    return [(k or "(blank)", round(v, 4)) for k, v in ranked[:n]]


def belief_mon_to_dict(mon: BeliefPokemon) -> dict[str, Any]:
    return {
        "species": mon.species,
        "slot": mon.slot,
        "brought_prob": round(mon.brought_prob, 4),
        "confirmed_brought": mon.confirmed_brought,
        "confirmed_absent": mon.confirmed_absent,
        "preview_only": mon.preview_only,
        "locked": mon.locked,
        "mega_confirmed": mon.mega_confirmed,
        "mega_form": mon.mega_form,
        "pikalytics_key": mon.pikalytics_key,
        "revealed_moves": sorted(mon.revealed_moves),
        "speed_floor": mon.speed_floor,
        "top_moves": [_top_n(slot, 2) for slot in mon.moves] if mon.moves else [],
        "top_items": _top_n(mon.item),
        "top_abilities": _top_n(mon.ability),
        "top_spreads": _top_n(mon.ev_spread, 2),
        "top_tera": _top_n(mon.tera_type, 2),
    }


def belief_to_dict(belief: BeliefState) -> dict[str, Any]:
    mons = sorted(belief.pokemon, key=lambda m: (m.confirmed_absent, -m.brought_prob, m.slot or 99))
    return {
        "confirmed_brought_count": belief.confirmed_brought_count,
        "remaining_slots": max(0, belief.BRING_COUNT - belief.confirmed_brought_count),
        "pokemon": [belief_mon_to_dict(m) for m in mons],
    }


def snapshot_opponent_to_dict(snap: BattleSnapshot | None) -> dict[str, Any]:
    if snap is None:
        return {}
    out: dict[str, Any] = {}
    for key, mon in snap.opponent.items():
        out[key] = {
            "revealed": mon.revealed,
            "active": mon.active,
            "fainted": mon.fainted,
            "ability": mon.ability,
            "item": mon.item,
            "moves": mon.moves,
            "mega_confirmed": mon.mega_confirmed,
            "mega_form": mon.mega_form,
            "hp": mon.hp,
            "max_hp": mon.max_hp,
        }
    return out


def diff_snapshots(prev: BattleSnapshot | None, curr: BattleSnapshot) -> list[str]:
    if prev is None:
        return ["initial_snapshot"]
    events: list[str] = []
    for species, cs in curr.opponent.items():
        ps = prev.opponent.get(species)
        if ps is None:
            events.append(f"new_preview_mon:{species}")
            continue
        if not ps.revealed and cs.revealed:
            events.append(f"confirm_brought:{species}")
        if not ps.active and cs.active:
            events.append(f"became_active:{species}")
        if ps.active and not cs.active:
            events.append(f"left_field:{species}")
        if not ps.fainted and cs.fainted:
            events.append(f"fainted:{species}")
        if ps.ability != cs.ability and cs.ability:
            events.append(f"ability_revealed:{species}={cs.ability}")
        if ps.item != cs.item and cs.item:
            events.append(f"item_revealed:{species}={cs.item}")
        if not ps.mega_confirmed and cs.mega_confirmed:
            events.append(f"mega_confirmed:{species}={cs.mega_form or '?'}")
        new_moves = set(cs.moves) - set(ps.moves)
        for move in sorted(new_moves):
            events.append(f"move_revealed:{species}={move}")
        if ps.hp > cs.hp and cs.max_hp > 0:
            dmg = int(ps.hp - cs.hp)
            if dmg > 0:
                events.append(f"damage_taken:{species}={dmg}")
    if curr.turn != prev.turn:
        events.insert(0, f"turn:{curr.turn}")
    return events


def format_belief_text(
    belief: BeliefState,
    battle: DoubleBattle | None = None,
    *,
    title: str = "",
    events: list[str] | None = None,
    game_plan: GamePlan | None = None,
) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"\n{'=' * 72}")
        lines.append(title)
        lines.append("=" * 72)
    if battle is not None:
        phase = "teampreview" if battle.teampreview else f"turn {battle.turn}"
        lines.append(f"Battle phase: {phase}")
    if events:
        lines.append(f"Events: {', '.join(events)}")
    if game_plan is not None:
        lines.append(
            f"GamePlan: threats={game_plan.primary_threats} "
            f"lead={game_plan.optimal_lead} opp_lead={game_plan.opponent_likely_lead}"
        )
        if game_plan.win_condition:
            lines.append(f"  win_condition: {game_plan.win_condition}")
        if game_plan.priority_kos:
            lines.append(f"  priority_kos: {game_plan.priority_kos}")

    lines.append(
        f"Roster: {belief.confirmed_brought_count}/4 confirmed, "
        f"{belief.BRING_COUNT - belief.confirmed_brought_count} slots remaining"
    )
    for mon in sorted(belief.pokemon, key=lambda m: (-int(m.confirmed_brought), -m.brought_prob, m.slot or 99)):
        if mon.confirmed_absent:
            status = "ABSENT"
        elif mon.confirmed_brought:
            status = "BROUGHT"
        else:
            status = f"P(brought)={mon.brought_prob * 100:.0f}%"
        line = f"  [{mon.slot or '?'}] {mon.species}: {status}"
        if mon.preview_only:
            line += " (preview-only, no set priors)"
        lines.append(line)
        if mon.confirmed_brought or not mon.preview_only:
            if mon.revealed_moves:
                lines.append(f"       moves seen: {', '.join(sorted(mon.revealed_moves))}")
            if mon.mega_confirmed:
                lines.append(f"       mega: {mon.mega_form or mon.pikalytics_key}")
            if mon.item.options:
                items = _top_n(mon.item, 2)
                if items:
                    lines.append(f"       item: {items}")
            if mon.ability.options:
                abils = _top_n(mon.ability, 2)
                if abils:
                    lines.append(f"       ability: {abils}")
            if mon.speed_floor:
                lines.append(f"       speed_floor: {mon.speed_floor}")
    return "\n".join(lines)


@dataclass
class BeliefTraceLog:
    battle_tag: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        label: str,
        belief: BeliefState,
        battle: DoubleBattle | None = None,
        *,
        events: list[str] | None = None,
        game_plan: GamePlan | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "phase": "teampreview" if battle and battle.teampreview else f"turn_{battle.turn if battle else 0}",
            "events": events or [],
            "belief": belief_to_dict(belief),
        }
        if game_plan is not None:
            entry["game_plan"] = asdict(game_plan)
        if extra:
            entry["extra"] = extra
        self.entries.append(entry)

    def save(self, path: Path | None = None, *, write_global_summary: bool = True) -> Path:
        if path is None:
            out_dir = LOGS_DIR.parent / "belief_traces"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"trace_{stamp}.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "battle_tag": self.battle_tag,
            "entries": self.entries,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if write_global_summary and path.parent.name == "belief_traces":
            summary_path = path.parent / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "latest_trace": str(path.relative_to(ROOT_DIR)),
                        "entry_count": len(self.entries),
                        "battle_tag": self.battle_tag,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return path
