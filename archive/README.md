# Archived Reference Code

Legacy **MaskablePPO + Gymnasium** and **ISMCTS HybridPlayer** stacks. Not used by the active Transformer behavior-cloning pipeline.

Run from repo root with `PokemonChampBot` conda env active.

## Layout

| Path | Contents |
|------|----------|
| `archive/rl/training/` | MaskablePPO curriculum (`train_curriculum.py`, `vec_env.py`, `pool_curriculum.py`) |
| `archive/rl/env/` | `ChampionsVGCRLEnv`, gym wrappers, flat observations, rewards |
| `archive/rl/players/` | `PolicyPlayer` (PPO inference) |
| `archive/rl/scripts/` | RL smoke tests, `record_replays.py`, `watch_training.py` |
| `archive/ismcts/planning/` | ISMCTS, value MLP, fast damage, RL value head |
| `archive/ismcts/players/` | `HybridPlayer` (belief + macro + ISMCTS) |
| `archive/ismcts/scripts/` | `trace_hybrid_belief.py`, `collect_value_data.py`, `train_value_mlp.py` |
| `archive/ismcts/evaluation/` | Value-network data collector |

## Entry points

```bash
conda activate PokemonChampBot

# PPO curriculum training
python -m archive.rl.training.train_curriculum

# Hybrid belief trace + replay
python archive/ismcts/scripts/trace_hybrid_belief.py

# Value MLP training
python archive/ismcts/scripts/collect_value_data.py
python archive/ismcts/scripts/train_value_mlp.py

# RL smoke tests
python archive/rl/scripts/smoke_mega.py
```

## Active pipeline

See [docs/architecture_map.md](../docs/architecture_map.md) for the Transformer BC scripts (`scrape_replays`, `parse_replays`, `train_bc`, `evaluate_model`).
