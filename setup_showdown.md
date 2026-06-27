# Local Pokémon Showdown Server (Champions mod)

Training requires a **local** Showdown instance with the Champions mod (VGC 2026 Reg M-A).

## Prerequisites

- Node.js 18+ (10+ minimum per upstream docs)
- Git

## Install

This repo clones Showdown to `c:\Projects\Fun\pokemon-showdown` (sibling of PokemonChampBot).

```bash
cd c:\Projects\Fun
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
git pull
npm install
cp config/config-example.js config/config.js
```

Ensure your checkout includes the Champions mod (merged via [Implement Champions #11910](https://github.com/smogon/pokemon-showdown/commit/b555f4d8ce7250e72939fe0c12cc5b303741d203)).

## Run (development / training)

```bash
node pokemon-showdown start --no-security
```

Default WebSocket: `ws://localhost:8000/showdown/websocket`

`--no-security` disables rate limits and auth — **only use on a trusted local machine**.

## Verify format

In the Showdown client or server logs, confirm `gen9championsvgc2026regma` exists.

## Windows notes

- Allow Node through the firewall if connections fail.
- Run the server in a dedicated terminal while training.
- For `SubprocVecEnv`, each worker opens its own connection; the server must handle multiple concurrent battles.

## poke-env default

`ChampionsVGCRLEnv` uses `LocalhostServerConfiguration` (localhost:8000) by default.

## Champions team paste

Reg M-A uses **Stat Points** (still pasted as `EVs:` in Showdown). Values are SP (max 66 total, 32 per stat), not traditional 508 EVs. See [teams/reg_ma_team.txt](teams/reg_ma_team.txt).
