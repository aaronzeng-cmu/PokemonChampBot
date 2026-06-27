# Architecture Map

This document maps scripts and modules to their pipeline so RL/ISMCTS and BC Transformer code can be separated later.

## Active: Transformer Behavior Cloning (Turn 1+)

| Script / Module | Role |
|-----------------|------|
| `scripts/scrape_replays.py` | Download Showdown replay logs |
| `scripts/parse_replays.py` | Parse logs → `data/processed/bc_dataset.pt` |
| `scripts/train_bc.py` | Supervised training → `models/bc_transformer_latest.pt` |
| `scripts/evaluate_model.py` | 100-game eval vs `MaxDamagePlayer` |
| `src/data/log_tracker.py` | First-person state from protocol lines |
| `src/data/state_tokenizer.py` | 13-token encoder (logs + live battles) |
| `src/data/replay_parser.py` | Log → (state, action) dataset builder |
| `src/data/action_codec.py` | Log actions → poke-env indices |
| `src/model/transformer_bot.py` | `VGCBehaviorCloner` network |
| `src/players/transformer_player.py` | poke-env `Player` (macro preview + BC moves) |
| `src/battle/battle_runner.py` | Async local Showdown battle runner |
| `src/battle/action_space.py` | Joint-legal combo enumeration |
| `src/battle/action_codec.py` | Masked action selection for inference |

## Shared: Team Preview (Turn 0) + Knowledge

| Script / Module | Role |
|-----------------|------|
| `src/planning/macro_strategist.py` | DeepSeek / heuristic bring-4 + leads |
| `src/planning/belief_state.py` | Opponent priors for preview |
| `src/planning/observation_tracker.py` | Live battle reveal tracking |
| `src/planning/meta_database.py` | Pikalytics + dex priors |
| `src/teams/teampreview.py` | `/team` command helpers |

## Shared: Evaluation Baselines

| Script / Module | Role |
|-----------------|------|
| `scripts/smoke_connect.py` | One-battle connectivity smoke test |
| `scripts/evaluate_gauntlet.py` | Legacy hybrid gauntlet (uses archive) |
| `src/players/max_damage_player.py` | Greedy damage baseline |
| `src/players/gauntlet_opponent.py` | Gauntlet opponent heuristics |
| `src/players/vgc_random_player.py` | Random legal moves |
| `src/evaluation/gauntlet_runner.py` | Weighted meta gauntlet runner |

## Archive: RL + ISMCTS (reference only)

| Script / Module | Role |
|-----------------|------|
| `archive/rl/training/train_curriculum.py` | MaskablePPO 3-stage curriculum |
| `archive/rl/env/champions_vgc_env.py` | Gymnasium doubles env |
| `archive/rl/players/policy_player.py` | PPO policy player |
| `archive/ismcts/players/hybrid_player.py` | Belief + macro + ISMCTS |
| `archive/ismcts/planning/ismcts.py` | Information-set MCTS |
| `archive/ismcts/scripts/trace_hybrid_belief.py` | Hybrid debug traces |
| `archive/rl/scripts/record_replays.py` | Policy/hybrid replay recording |

See [archive/README.md](../archive/README.md) for archived entry points.

## Data flow

```
scrape_replays → data/raw_logs/*.log
parse_replays  → data/processed/bc_dataset.pt
train_bc       → models/bc_transformer_latest.pt
evaluate_model → logs/eval/bc_eval_*.json
```

Turn 0: `MacroStrategist` → `/team` command. Turn 1+: `VGCBehaviorCloner` → masked poke-env actions.
