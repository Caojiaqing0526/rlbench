"""
A simple reinforcement learning environment, implementing something like the 
game of "chicken".
"""
import numpy as np 
from environment import Environment 

# memoizing properties in python?
# const in python?

class Chicken(Environment):
    ACTIONS = {0:'advance', 1:'return'}
    def __init__(self, length):
        self.length = length
        self.s0 = 0
        self._terminals = {length,}
        self.reset()

    @property 
    def states(self):
        return {s for s in range(self.length+1)}

    def actions(self, s=None):
        return (0, 1)

    def do(self, action):
        if self.is_terminal():
            return 0, self.state

        if action == 0:
            sp = self.state + 1
            r = self.rfunc(self.state, action, sp)
        elif action == 1:
            sp = self.s0 
            r = self.rfunc(self.state, action, sp)
        else:
            raise Exception("Invalid action:", action)
        self._state = sp 
        return r, sp 

    def rfunc(self, s, a, sp):
        if self.is_terminal(sp) and not self.is_terminal(s):
            return 1
        else:
            return 0

    def is_terminal(self, s=None):
        if s is None:
            s = self._state 
        return s == self.length



if __name__ == "__main__":
    env = Chicken(5)
    env.max_actions