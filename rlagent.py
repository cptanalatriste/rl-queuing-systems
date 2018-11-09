import numpy as np


class RLAgent:

    def __init__(self, actions):
        self.actions = actions

    def select_action(self, system_state):
        free_servers, priority = system_state

        if free_servers == 0:
            return self.actions[0]

        return np.random.choice(self.actions)

    def observe_action_effects(self, previous_state, action_performed, reward, new_state):
        pass
