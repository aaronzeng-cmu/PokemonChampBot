# Mega evolution in training

## Champions Reg M-A bring rules

OTS allows **any 4 of 6**; lead order comes from preview/`/team` ordering, not paste slot numbers. Showdown only uses paste slots 1–2 as **temporary preview placeholders** (see `docs/teampreview.md`).

With `SHUFFLE_TEAM_ORDER=True`, roster block order is randomized each battle so every mon (including **Charizard @ Charizardite Y**) can be brought and mega evolve over training.

## Action space

poke-env already exposes mega via `battle.can_mega_evolve` and adds `mega_space` to per-slot masks (`DoublesEnv`). No separate action head is required.

## Training signal

| Component | Role |
|-----------|------|
| `observation._encode_active` | `can_mega` + `is_mega` flags on actives |
| `rewards.MEGA_EVOLUTION_REWARD` | Bonus when our active mega count increases |
| `MaxDamagePlayer` | Prefers mega-augmented damaging lines (`MEGA_DAMAGE_BOOST`) |

## Verify

```bash
python scripts/smoke_mega.py
```

Expect non-zero `can_mega_steps` and `mega_legal_combos` in `logs/mega_smoke.json`.
