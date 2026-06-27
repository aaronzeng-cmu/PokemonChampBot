"""Win-rate evaluation vs SinglesMaxDamagePlayer on the opponent team pool."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    SINGLES_BATTLE_FORMAT,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
)
from src.singles.evaluation.eval_agent import (
    build_opponent_team,
    build_singles_eval_agent,
    describe_agent_team,
    load_eval_agent_team,
)
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.teampreview import battle_team_summary, opponent_team_summary


async def _run_batch(agent, opponent, *, n_battles: int) -> list[dict]:
    await agent.battle_against(opponent, n_battles=n_battles)
    rows: list[dict] = []
    for battle in agent.battles.values():
        summary = battle_team_summary(battle)
        opp_summary = opponent_team_summary(battle)
        rows.append(
            {
                "battle_tag": battle.battle_tag,
                "won": bool(battle.won),
                "turn": int(battle.turn),
                "lead": summary.get("lead"),
                "brought": summary.get("brought"),
                "opponent_brought": opp_summary.get("brought"),
            }
        )
    return rows


@dataclass
class LiveEvalReport:
    battles: int
    wins: int
    win_rate: float
    illegal_top1_count: int
    report_path: Path
    payload: dict


def run_maxdamage_eval(
    *,
    n_battles: int = 100,
    model_path: Path = SINGLES_BC_MODEL_PATH,
    rl_checkpoint: Path | None = None,
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path,
    mirror: bool = False,
    use_meta_pool: bool = False,
    batch_size: int = 10,
) -> LiveEvalReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_team = load_eval_agent_team(use_meta_pool=use_meta_pool)
    opponent_team = build_opponent_team(mirror=mirror)
    agent_pool_info = describe_agent_team(agent_team)
    pool_info = describe_agent_team(opponent_team)

    agent = build_singles_eval_agent(
        rl_checkpoint=rl_checkpoint,
        bc_model_path=model_path,
        preview_model_path=preview_model_path,
        device=device,
        team=agent_team,
        trace_inference=True,
        capture_battle_log=True,
        log_illegal_top1=True,
        account_name="SinglesRL" if rl_checkpoint else "SinglesTransformer",
    )
    opponent = SinglesMaxDamagePlayer(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesMaxDmg", rand=True),
    )

    all_rows: list[dict] = []
    illegal_top1: list[dict] = []
    wins = 0
    done = 0
    batch_size = min(batch_size, n_battles)
    first_battle_trace: dict | None = None

    while done < n_battles:
        k = min(batch_size, n_battles - done)
        agent.reset_battles()
        opponent.reset_battles()
        rows = asyncio.run(_run_batch(agent, opponent, n_battles=k))
        for row in rows:
            row["index"] = len(all_rows) + 1
            all_rows.append(row)
            wins += int(row["won"])
        if first_battle_trace is None and rows:
            tag = rows[0]["battle_tag"]
            first_battle_trace = agent.drain_inference_trace(tag)
        illegal_top1.extend(agent.drain_illegal_top1_events())
        done += k
        print(f"Progress: {done}/{n_battles} battles, wins={wins}", flush=True)

    if first_battle_trace is not None:
        trace_path = out_dir / "trace_battle_1.json"
        trace_path.write_text(
            json.dumps({"battles": [first_battle_trace]}, indent=2),
            encoding="utf-8",
        )

    payload = {
        "format": SINGLES_BATTLE_FORMAT,
        "battles": n_battles,
        "wins": wins,
        "losses": n_battles - wins,
        "win_rate": wins / n_battles if n_battles else 0.0,
        "agent": "maskable_ppo" if rl_checkpoint else "singles_transformer",
        "bc_model": str(model_path),
        "rl_checkpoint": str(rl_checkpoint) if rl_checkpoint else None,
        "preview_model": str(preview_model_path),
        "opponent": "SinglesMaxDamagePlayer",
        "agent_team_mode": agent_pool_info,
        "opponent_pool": pool_info,
        "use_meta_pool": use_meta_pool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "battles_detail": all_rows,
        "illegal_top1_count": len(illegal_top1),
        "illegal_top1_events": illegal_top1,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"live_eval_{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "live_eval_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return LiveEvalReport(
        battles=n_battles,
        wins=wins,
        win_rate=payload["win_rate"],
        illegal_top1_count=len(illegal_top1),
        report_path=report_path,
        payload=payload,
    )
