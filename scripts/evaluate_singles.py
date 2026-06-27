#!/usr/bin/env python3
"""Evaluate SinglesTransformerPlayer vs SinglesMaxDamagePlayer on local Showdown."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    BC_EVAL_LOG_DIR,
    SINGLES_BC_DATASET_PATH,
    SINGLES_BATTLE_FORMAT,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
    SINGLES_TEAM_PATH,
)
from src.doubles.teams.team_pool import opponent_pool_description
from src.singles.teams.team_pool import load_agent_team, load_opponent_team_builder
from src.singles.evaluation.eval_pipeline import EvalPipelineConfig, run_bc_examples_step
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator
from src.singles.teampreview import battle_team_summary, opponent_team_summary
from src.singles.transformer_player import SinglesTransformerPlayer


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Singles BC Transformer bot")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--bc-model", type=Path, default=SINGLES_BC_MODEL_PATH)
    parser.add_argument("--preview-model", type=Path, default=SINGLES_PREVIEW_MODEL_PATH)
    parser.add_argument(
        "--format",
        type=str,
        default=SINGLES_BATTLE_FORMAT,
        help="Showdown battle format (default: gen9championsbssregma)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Force mirror match instead of opponent pool",
    )
    parser.add_argument("--bc-examples", type=int, default=50, help="Offline BC examples to generate")
    parser.add_argument("--skip-bc-examples", action="store_true")
    args = parser.parse_args()

    out_dir = BC_EVAL_LOG_DIR / "singles"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"eval_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bc_step = None
    if not args.skip_bc_examples:
        print(f"\n--- Offline BC examples ({args.bc_examples}) ---")
        eval_cfg = EvalPipelineConfig(
            model_path=args.bc_model,
            device=args.device,
            dataset_path=SINGLES_BC_DATASET_PATH,
            bc_examples_n=args.bc_examples,
        )
        bc_step = run_bc_examples_step(eval_cfg, run_dir / "bc_examples")
        print(f"BC examples: {bc_step.txt_path}")

    agent_team = load_agent_team()
    opponent_team = load_opponent_team_builder(use_pool=not args.mirror)
    pool_info = opponent_pool_description(opponent_team)

    preview = SinglesPreviewOrchestrator(model_path=args.preview_model, device=args.device)
    agent = SinglesTransformerPlayer(
        model_path=args.bc_model,
        battle_format=args.format,
        team=agent_team,
        device=args.device,
        preview=preview,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesTransformer", rand=True),
    )
    opponent = SinglesMaxDamagePlayer(
        battle_format=args.format,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesMaxDmg", rand=True),
    )

    print(f"Format: {args.format}")
    print(f"BC model: {args.bc_model}")
    print(f"Preview model: {args.preview_model}")
    print(f"Agent team: {SINGLES_TEAM_PATH.name}")
    print(
        f"Opponent teams: {pool_info['mode']} "
        f"(active={pool_info['active_teams']}, pool={pool_info['pool_size']})"
    )
    print(f"Battles: {args.battles}")

    all_rows: list[dict] = []
    wins = 0
    batch = min(10, args.battles)
    done = 0
    while done < args.battles:
        k = min(batch, args.battles - done)
        agent.reset_battles()
        opponent.reset_battles()
        rows = asyncio.run(_run_batch(agent, opponent, n_battles=k))
        for row in rows:
            row["index"] = len(all_rows) + 1
            all_rows.append(row)
            wins += int(row["won"])
        done += k
        print(f"Progress: {done}/{args.battles} battles, wins={wins}")

    report = {
        "format": args.format,
        "battles": args.battles,
        "wins": wins,
        "losses": args.battles - wins,
        "win_rate": wins / max(1, args.battles),
        "bc_model": str(args.bc_model),
        "preview_model": str(args.preview_model),
        "opponent": "SinglesMaxDamagePlayer",
        "opponent_pool": pool_info,
        "battles_detail": all_rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }
    if bc_step is not None:
        report["bc_examples"] = {
            "n": bc_step.n_examples,
            "top1": bc_step.top1,
            "top1_rate": bc_step.top1_rate,
            "top3_hit_rate": bc_step.top3_hit_rate,
            "txt": str(bc_step.txt_path),
            "json": str(bc_step.json_path),
        }

    out_path = run_dir / "eval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"eval_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "eval_latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Singles Eval ===")
    print(f"Win rate: {report['win_rate']:.1%} ({wins}/{args.battles})")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
