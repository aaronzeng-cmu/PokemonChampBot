"""Dense reward shaping for VGCRLEnv."""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle.double_battle import DoubleBattle

WIN_REWARD = 5.0
LOSS_REWARD = -5.0
FAINT_ENEMY_REWARD = 1.0
FAINT_ALLY_REWARD = -1.0


@dataclass
class BattleSnapshot:
    our_hp_sum: float
    opp_hp_sum: float
    our_fainted: int
    opp_fainted: int

    @classmethod
    def from_battle(cls, battle: DoubleBattle) -> BattleSnapshot:
        our_hp = sum(
            (m.current_hp_fraction if not m.fainted else 0.0)
            for m in battle.team.values()
        )
        opp_hp = sum(
            (m.current_hp_fraction if not m.fainted else 0.0)
            for m in battle.opponent_team.values()
        )
        our_fainted = sum(1 for m in battle.team.values() if m.fainted)
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        return cls(
            our_hp_sum=our_hp,
            opp_hp_sum=opp_hp,
            our_fainted=our_fainted,
            opp_fainted=opp_fainted,
        )


def calc_step_reward(
    last: BattleSnapshot | None, battle: DoubleBattle
) -> tuple[float, BattleSnapshot]:
    snap = BattleSnapshot.from_battle(battle)
    if last is None:
        return 0.0, snap

    reward = 0.0
    damage_dealt = max(0.0, last.opp_hp_sum - snap.opp_hp_sum)
    damage_taken = max(0.0, last.our_hp_sum - snap.our_hp_sum)
    reward += damage_dealt
    reward -= damage_taken

    reward += FAINT_ENEMY_REWARD * (snap.opp_fainted - last.opp_fainted)
    reward += FAINT_ALLY_REWARD * (snap.our_fainted - last.our_fainted)

    if battle.won:
        reward += WIN_REWARD
    elif battle.lost:
        reward += LOSS_REWARD

    return reward, snap
