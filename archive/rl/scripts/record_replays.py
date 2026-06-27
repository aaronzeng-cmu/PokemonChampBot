#!/usr/bin/env python3
"""Run policy inference and save Showdown HTML replays (does not touch training)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from archive.rl.env.hybrid_agent_wrapper import HybridAgentWrapper
from archive.rl.env.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player
from sb3_contrib import MaskablePPO

from config.settings import (
    BATTLE_FORMAT,
    EVAL_N_BATTLES,
    MODELS_DIR,
    REPLAYS_DIR,
)
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team, resolve_opponent_team
from archive.rl.env.observation import adapt_observation_for_model
from archive.rl.env.replay_utils import keep_agent_replays
from archive.rl.env.gym_wrappers import wrap_for_sb3
from archive.ismcts.players.hybrid_player import HybridPlayer
from src.players.max_damage_player import MaxDamagePlayer
from archive.rl.players.policy_player import PolicyPlayer
from src.players.vgc_random_player import VGCRandomPlayer
from src.teams.teampreview import battle_team_summary
from archive.rl.training.device import resolve_device

OPPONENT_CHOICES = ("random", "maxdamage", "policy", "hybrid")
AGENT_CHOICES = ("policy", "hybrid")


def _default_model_path() -> Path:
    for name in (
        "maskable_ppo_stage2_maxdamage",
        "maskable_ppo_stage1_random",
    ):
        path = MODELS_DIR / name
        zip_path = path.with_suffix(".zip")
        if zip_path.is_file():
            return zip_path
    return MODELS_DIR / "maskable_ppo_stage1_random.zip"


def _make_opponent(
    kind: str,
    *,
    opponent_model: Path | None,
    device: str,
) -> Player:
    common = dict(
        battle_format=BATTLE_FORMAT,
        start_listening=False,
    )
    if kind == "random":
        return VGCRandomPlayer(**common)
    if kind == "maxdamage":
        return MaxDamagePlayer(**common)
    if kind == "policy":
        if opponent_model is None or not opponent_model.is_file():
            raise FileNotFoundError(
                f"--opponent policy requires --opponent-model (file not found: {opponent_model})"
            )
        opp = MaskablePPO.load(str(opponent_model), device=device)
        return PolicyPlayer(model=opp, team=load_team(), **common)
    if kind == "hybrid":
        return HybridPlayer(**common)
    raise ValueError(f"Unknown opponent {kind!r}")


def _run_battle_policy(
    gym_env,
    env: ChampionsVGCRLEnv,
    model: MaskablePPO,
    *,
    max_steps: int = 500,
) -> dict:
    obs, _ = gym_env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    target_obs = int(model.observation_space.shape[0])
    while not (terminated or truncated):
        masks = gym_env.action_masks()
        policy_obs = adapt_observation_for_model(obs, target_obs)
        action, _ = model.predict(policy_obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, _ = gym_env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if steps >= max_steps:
            break

    battle = env.battle1
    won = bool(battle.won) if battle else False
    tag = battle.battle_tag if battle else "unknown"
    team_info = battle_team_summary(battle) if battle else {"leads": [], "brought": []}
    return {
        "battle_tag": tag,
        "won": won,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "leads": team_info["leads"],
        "brought": team_info["brought"],
    }


def _run_battle_hybrid(gym_env, env: ChampionsVGCRLEnv, *, max_steps: int = 500) -> dict:
    obs, _ = gym_env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = gym_env.step(0)
        total_reward += float(reward)
        steps += 1
        if steps >= max_steps:
            break

    battle = env.battle1
    won = bool(battle.won) if battle else False
    tag = battle.battle_tag if battle else "unknown"
    team_info = battle_team_summary(battle) if battle else {"leads": [], "brought": []}
    return {
        "battle_tag": tag,
        "won": won,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "leads": team_info["leads"],
        "brought": team_info["brought"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play checkpointed policy vs an opponent and save HTML replays.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="MaskablePPO .zip (default: newest stage2, else stage1)",
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENT_CHOICES,
        default="random",
        help="Opponent type (default: random)",
    )
    parser.add_argument(
        "--opponent-model",
        type=Path,
        default=None,
        help="Required when --opponent policy",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=min(10, EVAL_N_BATTLES),
        help=f"Number of battles to play (default: {min(10, EVAL_N_BATTLES)})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Replay output folder (default: {REPLAYS_DIR}/<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: cpu — avoids competing with training GPU)",
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="policy",
        help="Agent type (default: policy). Use hybrid for BeliefState+ISMCTS bot.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = None
    model_path = None
    if args.agent == "policy":
        model_path = args.model or _default_model_path()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = MaskablePPO.load(str(model_path), device=device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPLAYS_DIR / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    opponent = _make_opponent(
        args.opponent,
        opponent_model=args.opponent_model,
        device=device,
    )
    env = ChampionsVGCRLEnv(
        team=load_team(),
        opponent_team=resolve_opponent_team(),
        log_level=40,
        open_timeout=None,
        save_replays=str(out_dir),
    )
    if args.agent == "hybrid":
        agent = HybridPlayer(start_listening=False)
        gym_env = HybridAgentWrapper(env, agent, opponent)
    else:
        gym_env = wrap_for_sb3(SingleAgentWrapper(env, opponent))

    print(f"Format: {BATTLE_FORMAT}")
    print(f"Agent: {args.agent}")
    if model_path:
        print(f"Model: {model_path}")
    print(f"Opponent: {args.opponent}")
    print(f"Battles: {args.battles}")
    print(f"Replays -> {out_dir.resolve()}")

    results: list[dict] = []
    wins = 0
    for i in range(args.battles):
        if args.agent == "hybrid":
            row = _run_battle_hybrid(gym_env, env)
        else:
            row = _run_battle_policy(gym_env, env, model)
        row["index"] = i + 1
        results.append(row)
        if row["won"]:
            wins += 1
        print(
            f"  [{i + 1}/{args.battles}] "
            f"{'WIN' if row['won'] else 'LOSS'} "
            f"reward={row['total_reward']:+.2f} steps={row['steps']} "
            f"leads={row.get('leads', [])} tag={row['battle_tag']}"
        )
        keep_agent_replays(out_dir, env.agent1.username)

    gym_env.close()

    replay_files = sorted(out_dir.glob("*.html"))
    summary = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "agent": args.agent,
        "model": str(model_path.resolve()) if model_path else None,
        "opponent": args.opponent,
        "opponent_model": str(args.opponent_model.resolve())
        if args.opponent_model
        else None,
        "battles": args.battles,
        "wins": wins,
        "losses": args.battles - wins,
        "win_rate": wins / args.battles if args.battles else 0.0,
        "device": device,
        "out_dir": str(out_dir.resolve()),
        "replay_files": [str(p.resolve()) for p in replay_files],
        "battles_detail": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWin rate: {wins}/{args.battles} ({100 * summary['win_rate']:.1f}%)")
    print(f"Replays saved: {len(replay_files)} HTML file(s)")
    print(f"Summary: {summary_path.resolve()}")
    print("Open any .html file in a browser to watch the battle.")


if __name__ == "__main__":
    main()
