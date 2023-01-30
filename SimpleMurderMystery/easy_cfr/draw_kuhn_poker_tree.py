from functools import partial

from easy_cfr.kuhn_poker import KuhnPoker
from easy_cfr.utilities.draw_tree import draw_game_tree

if __name__ == '__main__':
    model_factory = KuhnPoker
    draw_game_tree(model_factory)


