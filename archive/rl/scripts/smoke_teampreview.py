#!/usr/bin/env python3
"""Log legal team-preview combos and decoded species (debug / regression)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from archive.rl.env.single_agent_wrapper import SingleAgentWrapper

from config.settings import BATTLE_FORMAT, DEFAULT_TEAM_PREVIEW, LEARN_TEAM_PREVIEW
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from archive.rl.env.combo_action_space import enumerate_legal_combos
from archive.rl.env.gym_wrappers import wrap_for_sb3
from src.players.vgc_random_player import VGCRandomPlayer
from src.teams.teampreview import describe_preview_combos, parse_team_command


def main(n_battles: int = 3) -> None:
    print(f"LEARN_TEAM_PREVIEW={LEARN_TEAM_PREVIEW}")
    print(f"DEFAULT_TEAM_PREVIEW={DEFAULT_TEAM_PREVIEW!r}")

    env = ChampionsVGCRLEnv(team=load_team(), log_level=50, open_timeout=None)
    opponent = VGCRandomPlayer(battle_format=BATTLE_FORMAT, start_listening=False)
    gym = wrap_for_sb3(SingleAgentWrapper(env, opponent))

    all_rows: list[dict] = []
    for bi in range(n_battles):
        gym.reset()
        battle = env.battle1
        preview_logged = 0
        for _ in range(60):
            battle = env.battle1
            if battle is None or battle.finished:
                break
            if not battle.teampreview:
                if preview_logged >= 2:
                    if battle.active_pokemon:
                        leads = [p.species for p in battle.active_pokemon if p]
                        print(f"  Turn-1 leads after preview: {leads}")
                break
            if battle.teampreview and env.agent1_to_move:
                rows = describe_preview_combos(battle)
                print(f"\nBattle {bi + 1} preview step {preview_logged}: K={len(rows)}")
                for row in rows[:15]:
                    print(f"  combo {row['combo_idx']}: slots {row['slots']} -> {row['species']}")
                if len(rows) > 15:
                    print(f"  ... {len(rows) - 15} more")
                all_rows.append(
                    {
                        "battle": bi + 1,
                        "preview_step": preview_logged,
                        "legal_combos": len(rows),
                        "options": rows,
                    }
                )
                preview_logged += 1
                masks = gym.action_masks()
                legal = [i for i, v in enumerate(masks) if v]
                action = random.choice(legal) if legal else 0
            else:
                action = 0
            _, _, term, trunc, _ = gym.step(action)
            if term or trunc:
                break

    fallback_slots = parse_team_command(DEFAULT_TEAM_PREVIEW)
    print(f"\nFallback {DEFAULT_TEAM_PREVIEW!r} -> slots {fallback_slots}")
    gym.close()

    out = Path("logs") / "teampreview_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
