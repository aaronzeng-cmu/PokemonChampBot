"""Verify the live path carries moves from the team profile into legal actions.

Mirrors shadow_loop's doubles flow without an emulator:
  load team -> preview/brought -> perception(TURN_DECISION) -> model_inputs -> BC policy

Run: conda activate PokemonChampBot && python scripts/verify_team_moves_live.py
"""

from __future__ import annotations

import numpy as np

from src.cv_bridge.state_tracker import LiveBattleTracker

TEAM = "teams/champions_live_team.json"


def main() -> None:
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.load_player_team_file(TEAM)
    print(f"known team: {tracker.known_team_species}")

    # Simulate team preview: pretend OCR garbled two ally slots; reconcile repairs.
    parsed_ally = ["raichu", "unknown", "azumarill", "unknown", "dragonite", "staraptor"]
    enemy = ["incineroar", "rillaboom", "amoonguss", "tornadus", "urshifu", "ironhands"]
    tracker.record_team_preview(parsed_ally, enemy)
    ally_resolved = tracker.reconcile_preview_species(parsed_ally)
    print(f"reconciled preview ally: {ally_resolved}")
    brought = ["raichu", "garchomp", "dragonite", "staraptor"]
    tracker.record_brought_ally(brought)
    print(f"brought (bench-legal): {brought}")

    # Simulate a TURN_DECISION frame: two of our actives + two opponent actives.
    perception = {
        "state": "TURN_DECISION",
        "battle_format": "doubles",
        "ocr": {
            "player_slot_a": {"species_id": "raichu", "hp": 137, "max_hp": 137},
            "player_slot_b": {"species_id": "garchomp", "hp": 185, "max_hp": 185},
            "opp_slot_a": {"species_id": "incineroar", "hp_percent": 100},
            "opp_slot_b": {"species_id": "amoonguss", "hp_percent": 100},
        },
    }
    state = tracker.update_from_perception(perception)

    print("\n-- our actives (moves must be populated from profile) --")
    for slot in ("p1a", "p1b"):
        mon = state.mons[slot]
        print(f"  {slot}: {mon.species:<11} moves={mon.moves} can_mega={mon.can_mega}")

    obs, masks = tracker.get_model_inputs()
    print(f"\nobs shape: {obs.shape}")
    for key in ("slot_a", "slot_b"):
        m = masks[key] if masks else None
        legal = int(np.count_nonzero(m)) if m is not None else 0
        non_pass = int(np.count_nonzero(m[1:])) if m is not None else 0
        print(f"  mask[{key}]: legal={legal} (non-pass={non_pass})")

    # Run the real doubles BC model and confirm it does NOT pick pass/pass.
    from src.cv_bridge.bc_policy import BCPolicy

    policy = BCPolicy("models/bc_transformer_latest.pt", device="cpu")
    ca0, ca1 = policy(obs, masks)
    print(f"\nBC action: slot1={ca0}  slot2={ca1}  (0 == pass)")

    ok = (ca0 != 0 or ca1 != 0) and all(
        tracker.state.mons[s].moves for s in ("p1a", "p1b")
    )
    print("\nRESULT:", "PASS - live knows moves and makes a move" if ok
          else "FAIL - moves missing or model passed both slots")


if __name__ == "__main__":
    main()
