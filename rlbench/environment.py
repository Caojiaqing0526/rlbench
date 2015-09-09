import numpy


class Environment:
    """Abstract base class for a discrete environment in the RL framework."""
    def __init__(self, *args, **kwargs):
        pass
    
    def do(self, action):
        """Execute an action in the environment."""
        raise NotImplementedError 
        
    def rfunc(self, s, a, sp):
        """Reward function."""
        raise NotImplementedError
    
    def reset(self, s0=None):
        """Reset the environment."""
        raise NotImplementedError
        
    def is_terminal(self, s=None):
        """Return `True` if the environment is in a terminal state."""
        raise NotImplementedError
    
    @property
    def actions(self):
        """Actions available (in the current state)."""
        raise NotImplementedError
        
    @property
    def state(self):
        """The current state of the environment."""
        return self._state

    @property
    def states(self):
        """The set of all states in the environment."""
        return self.nonterminals + self.terminals

    @property 
    def nonterminals(self):
        """The set of nonterminal states of the environment."""
        raise NotImplementedError

    @property 
    def terminals(self):
        """The set of terminal states of the environment."""
        raise NotImplementedError

    @property
    def max_actions(self):
        """The maximum number of actions available over all states."""
        pass