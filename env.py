import numpy as np
from utils import add_card

class Env(object):

    def __init__(self):
        self.reset()

    def step(self, action):
        """ Takes a step in the environment. Returns a new state and a reward as a tuple.

        Keyword arguments:
        s -- state list
        a -- action integer, either 0 (stick) or 1 (hit)
        """
        if action == 1:  # Hit
            self._state = add_card(self._state)
            if self._state[1] < 1 or self._state[1] > 21:
                self._state = [0, 0]
                return np.copy(self._state), -1, True
            return np.copy(self._state), 0, False
        if action == 0:  # Stick
            while 0 < self._state[0] < 17:
                self._state = add_card(self._state, True)
                # print("Dealer draws! Now has: " + str(self._state[0]))  # Test output
            if self._state[0] < 1:  # Dealer bust
                self._state = [0, 0]
                return np.copy(self._state), 1, True
            if self._state[0] > 21:
                self._state = [0, 0]
                return np.copy(self._state), 1, True
            if self._state[0] == self._state[1]:  # Draw
                self._state = [0, 0]
                return np.copy(self._state), 0, True
            if self._state[0] > self._state[1]:  # Loss
                self._state = [0, 0]
                return np.copy(self._state), -1, True
            if self._state[0] < self._state[1]:  # Win
                self._state = [0, 0]
                return np.copy(self._state), 1, True

    def reset(self):
        """ Sets and returns the initial state."""
        self._state = [np.random.randint(1, 11), np.random.randint(1, 11)]
        return np.copy(self._state)

    @property
    def action_space(self):
        return 0, 1


