from functools import partial

from easy_cfr.utilities.draw_tree import draw_game_tree
from easy_cfr.murder_mystery import MurderMysteryParams, MurderGameModel

if __name__ == '__main__':
    params = MurderMysteryParams(allow_pass=False, allow_suicide=True, n_people=3, max_turns=4)
    model_factory = partial(MurderGameModel, params)
    draw_game_tree(model_factory)


