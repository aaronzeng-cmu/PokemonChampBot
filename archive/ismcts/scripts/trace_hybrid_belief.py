#!/usr/bin/env python3
"""Run HybridPlayer end-to-end with detailed belief-state tracing and HTML replay."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config.settings import (
    BATTLE_FORMAT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING_ENABLED,
    ISMCTS_VALUE_MLP_PATH,
    MACRO_STRATEGIST_ENABLED,
    REPLAYS_DIR,
    USE_OPPONENT_TEAM_POOL,
)
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team, resolve_opponent_team
from archive.rl.env.hybrid_agent_wrapper import HybridAgentWrapper
from archive.rl.env.replay_utils import keep_agent_replays
from src.planning.belief_trace import (
    BeliefTraceLog,
    diff_snapshots,
    format_belief_text,
    snapshot_opponent_to_dict,
)
from archive.ismcts.players.hybrid_player import HybridPlayer
from src.players.max_damage_player import MaxDamagePlayer
from src.teams.teampreview import battle_team_summary


class TracingHybridPlayer(HybridPlayer):
    def __init__(self, trace: BeliefTraceLog, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.trace = trace
        self.verbose = verbose

    def _log(
        self,
        label: str,
        battle,
        *,
        events: list[str] | None = None,
        game_plan=None,
        extra: dict | None = None,
    ) -> None:
        belief = self._ctx(battle).get("belief")
        if belief is None:
            return
        self.trace.battle_tag = battle.battle_tag
        self.trace.add(label, belief, battle, events=events, game_plan=game_plan, extra=extra)
        if self.verbose:
            print(
                format_belief_text(
                    belief,
                    battle,
                    title=label,
                    events=events,
                    game_plan=game_plan,
                )
            )

    def _sync_belief(self, battle):
        prev = self._ctx(battle).get("prev_snapshot")
        super()._sync_belief(battle)
        curr = self._ctx(battle).get("prev_snapshot")
        if curr is not None:
            events = diff_snapshots(prev, curr)
            if events and events != ["initial_snapshot"]:
                self._log(
                    f"Observation update (turn {battle.turn})",
                    battle,
                    events=events,
                    extra={"snapshot": snapshot_opponent_to_dict(curr)},
                )

    def _ensure_belief(self, battle):
        had = self._ctx(battle).get("belief") is not None
        belief = super()._ensure_belief(battle)
        if not had:
            preview = [
                getattr(m, "species", "?") for m in battle.teampreview_opponent_team
            ]
            self._log(
                "Belief initialized from preview",
                battle,
                events=["initialize_from_preview"],
                extra={"preview_species": preview},
            )
        return belief

    def _ensure_game_plan(self, battle):
        had = self._ctx(battle).get("game_plan") is not None
        plan = super()._ensure_game_plan(battle)
        if not had:
            source = "deepseek" if DEEPSEEK_API_KEY and MACRO_STRATEGIST_ENABLED else "heuristic"
            self._log(
                f"Macro strategist ({source})",
                battle,
                events=["macro_analyze"],
                game_plan=plan,
            )
        return plan

    def choose_move(self, battle):
        plan = self._ctx(battle).get("game_plan")
        order = super().choose_move(battle)
        self._log(
            f"ISMCTS decision (turn {battle.turn})",
            battle,
            game_plan=plan,
            extra={"order_type": type(order).__name__},
        )
        return order


def run_trace(
    *,
    battles: int = 1,
    verbose: bool = True,
    max_steps: int = 500,
    out_dir: Path | None = None,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir or (REPLAYS_DIR / f"{stamp}_hybrid_trace")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = ChampionsVGCRLEnv(
        team=load_team(),
        opponent_team=resolve_opponent_team(),
        log_level=40,
        open_timeout=None,
        save_replays=str(run_dir),
    )
    trace = BeliefTraceLog()
    agent = TracingHybridPlayer(trace=trace, verbose=verbose, start_listening=False)
    opponent = MaxDamagePlayer(start_listening=False)
    gym_env = HybridAgentWrapper(env, agent, opponent)

    print(f"Format: {BATTLE_FORMAT}")
    print(f"DeepSeek: {DEEPSEEK_MODEL} (thinking={'on' if DEEPSEEK_THINKING_ENABLED else 'off'})")
    print(f"Value MLP: {ISMCTS_VALUE_MLP_PATH}")
    print(f"Opponent pool: {'enabled' if USE_OPPONENT_TEAM_POOL else 'mirror team'}")
    print(f"Output -> {run_dir.resolve()}\n")

    results: list[dict] = []
    wins = 0
    for i in range(battles):
        print(f"\n{'#' * 72}\nBATTLE {i + 1}/{battles}\n{'#' * 72}")
        obs, _ = gym_env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = gym_env.step(0)
            total_reward += float(reward)
            steps += 1
            if steps > max_steps:
                print(f"Abort: exceeded {max_steps} steps")
                break
        battle = env.battle1
        won = bool(battle.won) if battle else False
        wins += int(won)
        team_info = battle_team_summary(battle) if battle else {"leads": [], "brought": []}
        if battle and agent._ctx(battle).get("belief"):
            agent._log(
                f"Battle finished ({'WIN' if won else 'LOSS'})",
                battle,
                events=[f"result:{'win' if won else 'loss'}", f"steps:{steps}"],
                extra={"total_reward": round(total_reward, 3)},
            )
        row = {
            "index": i + 1,
            "battle_tag": battle.battle_tag if battle else "unknown",
            "won": won,
            "steps": steps,
            "total_reward": round(total_reward, 4),
            "leads": team_info["leads"],
            "brought": team_info["brought"],
        }
        results.append(row)
        keep_agent_replays(run_dir, env.agent1.username)
        print(
            f"\nResult: {'WIN' if won else 'LOSS'} | reward={total_reward:+.2f} | steps={steps}"
        )

    gym_env.close()

    trace_path = run_dir / "belief_trace.json"
    trace.save(trace_path, write_global_summary=False)
    replay_files = sorted(run_dir.glob("*.html"))
    summary = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "agent": "hybrid",
        "agent_policy": "HybridPlayer (BeliefState + DeepSeek macro + ISMCTS tactician)",
        "opponent": "maxdamage",
        "opponent_policy": (
            "MaxDamagePlayer — random bring-4 at preview; each turn picks the "
            "legal combo with highest estimated total damage"
        ),
        "deepseek_model": DEEPSEEK_MODEL,
        "deepseek_thinking": DEEPSEEK_THINKING_ENABLED,
        "value_mlp_path": ISMCTS_VALUE_MLP_PATH,
        "opponent_team_pool": USE_OPPONENT_TEAM_POOL,
        "battles": battles,
        "wins": wins,
        "losses": battles - wins,
        "win_rate": wins / battles if battles else 0.0,
        "out_dir": str(run_dir.resolve()),
        "belief_trace": str(trace_path.resolve()),
        "replay_files": [str(p.resolve()) for p in replay_files],
        "battles_detail": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBelief trace: {trace_path}")
    print(f"Replays saved: {len(replay_files)} HTML file(s)")
    for replay in replay_files:
        print(f"  {replay.name}")
    print(f"Summary: {summary_path}")
    print(f"Trace entries: {len(trace.entries)} | Session: {wins}/{battles} wins")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace hybrid pipeline belief updates and save HTML replay",
    )
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--quiet", action="store_true", help="JSON only, minimal console output")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output folder (default: {REPLAYS_DIR}/<timestamp>_hybrid_trace)",
    )
    parser.add_argument(
        "--ismcts-ms",
        type=int,
        default=1500,
        help="ISMCTS time budget per decision (default 1500 for tracing)",
    )
    args = parser.parse_args()

    import config.settings as settings

    settings.ISMCTS_TIME_BUDGET_MS = args.ismcts_ms
    run_trace(
        battles=args.battles,
        verbose=not args.quiet,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
