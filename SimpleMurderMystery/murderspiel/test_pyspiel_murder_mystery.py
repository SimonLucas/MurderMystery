import numpy as np

import pyspiel
from open_spiel.python.algorithms import exploitability, cfr
from open_spiel.python import policy as policy_lib

from pyspiel_murder_mystery import MurderMysteryGame, MurderMysteryState
import itertools as it

game = MurderMysteryGame()

state = game.new_initial_state()
print(state.is_chance_node())

print(state.chance_outcomes())
child = state.child(2)
print(state)
print(child)
print(child)

next: MurderMysteryState = child.clone()

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
# print(policy.action_probability_array)

for name in pyspiel.registered_names():
    print(name)


def print_policy(policy):
  for state, probs in zip(it.chain(*policy.states_per_player),
                          policy.action_probability_array):
    print(f'{state:6}   p={probs}')

print_policy(policy)

loss = exploitability.exploitability(game, policy)

print(f"Exploitability = {loss}")

def get_cfr_policy(n: int):
    cfr_solver = cfr.CFRSolver(game)
    average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
        loss = exploitability.exploitability(game, average_policy)
        print(f"Exploitability ({i}) = {loss}")
    return average_policy


cfr_policy = get_cfr_policy(10)

print_policy(cfr_policy)
print(len(cfr_policy.action_probability_array))

