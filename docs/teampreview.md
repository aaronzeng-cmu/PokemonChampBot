# Team preview (bring-4 / leads)

Champions Reg M-A uses **4 Pokémon per side** with Open Team Sheets. OTS shares species, items, abilities, and moves before you commit — **not** fixed roster slots 1–2 as leads. Any **4 of 6** is legal; the `/team` digit order sets who leads turn 1.

## Showdown preview mechanics (not OTS rules)

During preview, Showdown places the **first two blocks in your paste** as temporary on-field placeholders (`active: true` in the request). poke-env’s doubles preview then picks legal **switch** combos into the other four roster slots. Those placeholders are **not** locked brings — they are swapped out as part of the 2-step masked combo flow.

**Training implication:** without shuffling paste order, the two mons in paste slots 1–2 are **never brought** (always swapped away). Enable `SHUFFLE_TEAM_ORDER` in `config/settings.py` so all six mons rotate through bringable positions across battles.

## Training (`LEARN_TEAM_PREVIEW=True`)

- `ChampionsVGCRLEnv(choose_on_teampreview=True)` — preview uses the same masked combo action space as battle turns (typically **K=12** then **K=2** on step 0 for a 6-mon paste).
- `TeampreviewActionWrapper` is **disabled** so MaskablePPO chooses preview actions.
- Observations add 2 features: preview flag + selection progress (`src/env/observation.py`).
- Agent and opponent both sample uniformly over legal preview combos (`SingleAgentWrapper._random_preview_combo` for the external opponent; policy combo index for the agent).

## Opponents

- `VGCRandomPlayer` / `MaxDamagePlayer` expose `random_teampreview_command()` for Player-API paths; **training** uses combo indices via `SingleAgentWrapper`, not `/team` digit strings.
- Opponent pool teams are shuffled per battle when `SHUFFLE_TEAM_ORDER=True`.

## Fallback

Set `LEARN_TEAM_PREVIEW=False` to restore fixed preview combo `0` and `DEFAULT_TEAM_PREVIEW` (`/team 3456`).

## Debug

```bash
python scripts/smoke_teampreview.py
```

Logs legal preview combos and writes `logs/teampreview_smoke.json`.

## Checkpoints

Changing observation size **invalidates** old MaskablePPO weights — restart Stage 1 after enabling learned preview.
