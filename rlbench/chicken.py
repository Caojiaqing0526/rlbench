"""
A simple reinforcement learning environment, implementing something like the
game of "chicken".
"""
import numpy as np
from environment import Environment


class Chicken(Environment):
    ACTIONS = {'advance': 0, 'return': 1}
    EPISODIC = False
    def __init__(self, length):
        self.length = length-1
        self.s0 = 0
        self.reset()

    @property
    def states(self):
        return {s for s in range(self.length+1)}

    def get_actions(self, s=None):
        return (0, 1)

    def do(self, action):
        if action == 0:
            if self._state == self.length:
                sp = self.s0
            else:
                sp = self._state + 1
        elif action == 1:
            sp = self.s0
        else:
            raise Exception("Invalid action:", action)
        # compute reward and set next state
        r = self.rfunc(self.state, action, sp)
        self._state = sp
        return r, sp

    def rfunc(self, s, a, sp):
        if s == self.length and a == self.ACTIONS['advance']:
            return 1
        else:
            return 0

    def is_terminal(self, s=None):
        return False