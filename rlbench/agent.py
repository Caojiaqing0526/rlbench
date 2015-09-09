import numpy as np 


class Agent:
    """Abstract base class for agents in the RL framework."""
    def __init__(self, *args, **kwargs):
        pass

    def choose(self, s, actions):
        pass
    
    def update(self, s, a, r, sp, **params):
        pass


class RandomAgent(Agent):
    """An example agent, which takes an action at random from the set of 
    available actions. """
    def choose(self, state, actions):
        return np.random.choice(actions)
