"""Random baseline opponent with legal VGC team preview."""

from poke_env.player.baselines import RandomPlayer

from src.doubles.teams.teampreview import random_teampreview_command


class VGCRandomPlayer(RandomPlayer):
    """RandomPlayer with random legal bring-4 / lead preview per battle."""

    def teampreview(self, battle) -> str:
        return random_teampreview_command(battle)
