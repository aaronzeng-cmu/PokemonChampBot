#!/usr/bin/env python3
"""Run live battles and dump full inference + protocol traces (not win-rate eval).

Default: BC Transformer (--model).
With --checkpoint: trained MaskablePPO RL policy (same resolution as record_rl_replays.py).
"""

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
    BATTLE_FORMAT,
    BC_EVAL_LOG_DIR,
    BC_MODEL_PATH,
    REPLAYS_DIR,
    USE_OPPONENT_TEAM_POOL,
)
from src.core.battle.replay_utils import keep_agent_replays
from src.doubles.evaluation.battle_inference_trace import format_trace_text, write_trace_report
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.rl_replay_player import RLReplayPlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.rl.checkpoints import load_rl_checkpoint
from src.doubles.teams.team_pool import (
    load_agent_team,
    load_opponent_team_builder,
    opponent_pool_description,
)
from src.doubles.teams.teampreview import (
    battle_team_summary,
    opponent_full_team_summary,
    opponent_team_summary,
)


async def _run_one_battle(agent: TransformerPlayer, opponent) -> dict:
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    trace = agent.drain_inference_trace(battle.battle_tag)
    team_info = battle_team_summary(battle)
    opp_info = opponent_team_summary(battle)
    full_info = opponent_full_team_summary(battle)
    trace.update(
        {
            "won": bool(battle.won),
            "turn": battle.turn,
            "leads": team_info["leads"],
            "brought": team_info["brought"],
            "opponent_brought": opp_info["brought"],
            "opponent_full_team": full_info["full_team"],
        }
    )
    return trace


def _build_agent(
    *,
    use_rl: bool,
    model_path: Path,
    checkpoint_arg: Path | None,
    device: str,
    top_k: int,
    replay_dir: Path | None,
):
    common = dict(
        battle_format=BATTLE_FORMAT,
        team=load_agent_team(),
        device=device,
        trace_inference=True,
        trace_top_k=top_k,
        capture_battle_log=True,
        log_illegal_top1=True,
        max_concurrent_battles=1,
        save_replays=str(replay_dir) if replay_dir else False,
        account_configuration=AccountConfiguration.generate("TransformerTrace", rand=True),
    )
    if use_rl:
        rl_model, checkpoint_path = load_rl_checkpoint(checkpoint_arg, device=device)
        return (
            RLReplayPlayer(rl_model, deterministic=True, **common),
            checkpoint_path,
            "rl_checkpoint",
        )
    return TransformerPlayer(model_path=model_path, **common), model_path, "bc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace Transformer or RL inference decisions during live battles"
    )
    parser.add_argument("--battles", type=int, default=1, help="Battles to trace")
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument(
        "--checkpoint",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="Trace a trained RL .zip (omit PATH for best_wr* in models/rl_checkpoints)",
    )
    parser.add_argument(
        "--opponent",
        choices=("maxdamage", "random"),
        default="maxdamage",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k per slot in trace")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: logs/eval/inference_trace/<stamp>)",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Also save HTML replays alongside the trace",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror match instead of opponent pool",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (BC_EVAL_LOG_DIR / "inference_trace" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = REPLAYS_DIR / f"trace_{stamp}" if args.save_replays else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    use_pool = USE_OPPONENT_TEAM_POOL and not args.mirror
    opponent_team = load_opponent_team_builder(
        use_pool=use_pool,
        use_curriculum=False,
    )
    pool_info = opponent_pool_description(opponent_team)

    use_rl = args.checkpoint is not None
    ckpt_arg = None if not use_rl or args.checkpoint == "auto" else Path(args.checkpoint)
    agent, model_ref, policy_source = _build_agent(
        use_rl=use_rl,
        model_path=args.model,
        checkpoint_arg=ckpt_arg,
        device=args.device,
        top_k=args.top_k,
        replay_dir=replay_dir,
    )
    checkpoint_path = model_ref if use_rl else None

    if args.opponent == "maxdamage":
        opponent = MaxDamagePlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("MaxDamageTrace", rand=True),
        )
    else:
        from src.doubles.players.vgc_random_player import VGCRandomPlayer

        opponent = VGCRandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("RandomTrace", rand=True),
        )

    print(f"Format: {BATTLE_FORMAT}", flush=True)
    if use_rl:
        print(f"RL checkpoint: {model_ref}", flush=True)
    else:
        print(f"BC model: {args.model}", flush=True)
    print(f"Opponent: {args.opponent}", flush=True)
    print(f"Battles: {args.battles} | top-k={args.top_k}", flush=True)
    print(f"Trace output -> {out_dir.resolve()}", flush=True)

    battle_traces: list[dict] = []
    for i in range(args.battles):
        agent.reset_battles()
        opponent.reset_battles()
        trace = asyncio.run(_run_one_battle(agent, opponent))
        trace["index"] = i + 1
        battle_traces.append(trace)
        if replay_dir is not None:
            keep_agent_replays(replay_dir, agent.username)
        n_decisions = len(trace.get("decisions", []))
        n_fallback = sum(1 for d in trace.get("decisions", []) if d.get("any_fallback"))
        print(
            f"  [{i + 1}/{args.battles}] "
            f"{'WIN' if trace.get('won') else 'LOSS'} "
            f"turns={trace.get('turn')} decisions={n_decisions} "
            f"fallbacks={n_fallback} tag={trace.get('battle_tag')}",
            flush=True,
        )

    report_model = args.model if not use_rl else None
    txt_path, json_path = write_trace_report(
        battle_traces,
        out_dir,
        model_path=report_model,
        opponent=args.opponent,
        policy_source=policy_source,
        checkpoint=checkpoint_path,
    )

    meta = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "policy_source": policy_source,
        "model": str(args.model.resolve()) if not use_rl else None,
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "opponent": args.opponent,
        "opponent_team_mode": pool_info,
        "battles": args.battles,
        "top_k": args.top_k,
        "trace_text": str(txt_path.resolve()),
        "trace_json": str(json_path.resolve()),
        "replay_dir": str(replay_dir.resolve()) if replay_dir else None,
    }
    (out_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nText trace: {txt_path}", flush=True)
    print(f"JSON trace: {json_path}", flush=True)
    if battle_traces:
        preview = format_trace_text([battle_traces[0]])[:2500]
        print("\n--- First battle preview ---")
        print(preview)
        if len(preview) >= 2500:
            print("... (see inference_trace_latest.txt for full log)")


if __name__ == "__main__":
    main()
