from open_spiel.python.algorithms import exploitability, cfr

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryVariationsState, \
    MurderMysteryParams
import itertools as it

import pyspiel
from open_spiel.python import policy as policy_lib

params = MurderMysteryParams(allow_pass=True, allow_suicide=True)

game = MurderMysteryVariationsGame(game_params=params)

state = game.new_initial_state()
print(state.is_chance_node())

print(state.chance_outcomes())
child = state.child(2)
print(state)
print(child)
print(child)

next: MurderMysteryVariationsState = child.clone()

# print(next.is_chance_node())

# next._apply_action(3)
obs = game.make_py_observer()
for s in [state, child, next]:
    print(f"\ns = {s}")
    print(s.current_player())
    print(s.to_play())
    print(s.is_terminal())
    obs.set_from(s, s.current_player())

policy = policy_lib.TabularPolicy(game)
print("States per player: ", policy.states_per_player)
print(len(policy.states_per_player))
for x in policy.states_per_player:
    print(len(x))


