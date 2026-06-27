# Pokémon Champions VGC RL Bot

Deep RL bot for **Gen 9 Champions VGC 2026 Reg M-A** (`gen9championsvgc2026regma`) using poke-env, Gymnasium, and **MaskablePPO** (sb3-contrib).

## Setup

This project uses the **PokemonChampBot** conda environment (Python 3.11).

```bash
# First time
conda env create -f environment.yml
# Or if the env already exists
conda activate PokemonChampBot
pip install -r requirements.txt
```

Cursor agents are configured via `.cursor/rules/python-environment.mdc` to always use this env.

### DeepSeek API key (hybrid macro strategist)

The generative hybrid bot uses DeepSeek v4 Pro at team preview. **Never commit real API keys.**

1. Copy the template: `cp .env.example .env` (Windows: copy `.env.example` to `.env`)
2. Set `DEEPSEEK_API_KEY` in `.env` (gitignored)
3. Before any GitHub push, run `git status` — **`.env` must not appear`**

Defaults in `.env.example`: `DEEPSEEK_REASONING_EFFORT=max`, `DEEPSEEK_THINKING_ENABLED=true`.

### GPU (CUDA) — recommended for training

The default `pip install -r requirements.txt` may install CPU-only PyTorch. For an **RTX GPU**, reinstall torch with CUDA support:

```bash
conda activate PokemonChampBot
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Training uses `--device auto` by default (CUDA when available). Most wall time is still spent simulating battles on the local Showdown server; **parallel envs** (`N_ENV=16` in config) usually help more than GPU alone.

Start a local Showdown server — see [setup_showdown.md](setup_showdown.md).

## Project layout

| Path | Purpose |
|------|---------|
| `src/env/champions_vgc_env.py` | Combinatorial doubles RL environment |
| `src/env/combo_action_space.py` | Joint-legal action enumeration |
| `src/env/observation.py` | Battle state embedding |
| `src/env/rewards.py` | Dense reward shaping |
| `src/training/train_curriculum.py` | 3-stage curriculum training |
| `teams/reg_ma_team.txt` | Your agent's fixed Reg M-A team |
| `teams/opponents/` | Downloaded meta teams (opponents sample per battle) |
| `scripts/fetch_opponent_teams.py` | Pull teams from VGCPastes sheet (default) or Smogon |

## Tests (offline, no Showdown)

```bash
conda activate PokemonChampBot
python scripts/run_tests.py
```

## Smoke tests (require local Showdown — see setup_showdown.md)

```bash
conda activate PokemonChampBot
node path/to/pokemon-showdown start --no-security
python scripts/smoke_connect.py
python scripts/smoke_action_space.py
python scripts/smoke_verbose_rewards.py
```

### Replay recording (inference only — does not slow training)

Run in a **separate terminal** while training continues. Uses CPU by default so it does not compete with CUDA training.

```bash
conda activate PokemonChampBot
# Stage 1 policy vs random (default 10 battles)
python scripts/record_replays.py --model models/maskable_ppo_stage1_random.zip

# vs MaxDamage, more battles
python scripts/record_replays.py --model models/maskable_ppo_stage1_random.zip --opponent maxdamage --battles 20

# Latest checkpoint from a partial stage 2 run
python scripts/record_replays.py --model models/checkpoints/maskable_ppo_stage2_maxdamage_50000_steps.zip --opponent maxdamage
```

HTML replays and `summary.json` (win rate, paths) are written under `logs/replays/<timestamp>/`. Open `.html` files in a browser.

## Opponent team pool (reduce mirror-match overfitting)

Training keeps **your** team in `teams/reg_ma_team.txt`. Opponents sample a **random team per battle** from `teams/opponents/` when the pool exists (`USE_OPPONENT_TEAM_POOL` in config).

Download **50 newest** teams from the [VGCPastes Repository](https://docs.google.com/spreadsheets/d/1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw/edit?gid=791705272) (Champions M-A tab), with **EVs = Yes** and **Replica Status = checkmark** (full replica spreads). Classic 252-style EV lines are converted to Champions stat points when needed.

```bash
conda activate PokemonChampBot
python scripts/fetch_opponent_teams.py --target 50

# Fallback: Smogon forum scrape
python scripts/fetch_opponent_teams.py --source smogon --target 50
```

Set `USE_OPPONENT_TEAM_POOL = False` in `config/settings.py` to revert to mirror teams.

## Training

```bash
conda activate PokemonChampBot
# Defaults: 16 parallel envs, device=auto (CUDA if installed)
python -m src.training.train_curriculum --stage 1
python -m src.training.train_curriculum --stage 2
python -m src.training.train_curriculum --stage 3

# Quick smoke (fewer envs, in-process — safer on Windows for debugging)
python -m src.training.train_curriculum --stage 1 --timesteps 8192 --n-env 2 --no-subproc

# Force CPU or CUDA
python -m src.training.train_curriculum --stage 1 --device cuda
```

TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Checkpoints are saved under `models/`.

### Checkpoints and resume

During training, periodic saves go to `models/checkpoints/` every **50k** timesteps (config: `CHECKPOINT_SAVE_FREQ`). A final zip is written when a run finishes (`models/maskable_ppo_stage{N}_*.zip`).

Stop anytime (Ctrl+C); resume the same stage without losing weights:

```bash
# Fresh stage 2 (loads stage 1, trains to 1M total)
python -m src.training.train_curriculum --stage 2 --device cuda

# Resume toward stage default (1M for stage 2)
python -m src.training.train_curriculum --stage 2 --device cuda --resume

# Resume with an explicit extra budget (e.g. another 250k steps)
python -m src.training.train_curriculum --stage 2 --device cuda --resume --timesteps 250000
```

Disable periodic saves: `--checkpoint-freq 0`.

## Curriculum

1. **Stage 1** — vs `RandomPlayer`
2. **Stage 2** — vs `MaxDamagePlayer`
3. **Stage 3** — self-play vs previous checkpoint

## Configuration

Edit [config/settings.py](config/settings.py) for `MAX_COMBOS`, `N_ENV` (default **16**), `PPO_DEVICE`, and PPO hyperparameters.
