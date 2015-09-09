import numpy


class Environment:
    """Abstract base class for a discrete environment in the RL framework."""
    def __init__(self, *args, **kwargs):
        pass
    
    def do(self, action):
        """Execute an action in the environment.

        Args:
            action: The action to take in the current state.

        Returns:
            r (float): The resulting reward from taking `action`.
            sp : The new environment state resulting from performing `action`. 
        """
        raise NotImplementedError 
        
    def rfunc(self, s, a, sp):
        """Reward function.

        Note:
            In most cases, this should be overwritten specific to the task the
            environment implements.

        Args:
            s: State in which action `a` was performed..
            a: Action taken in state `s`.
            sp: New state resulting from taking action `a` in state `s`.

        Returns:
            r (float): The reward from the transition (`s`, `a`, `sp`).
        """
        raise NotImplementedError
    
    def reset(self, s0=None):
        """Reset the environment.
        
        Args:
            s0: State to set the environment to. Defaults to the state the 
            environment was in at initialization.
        """
        raise NotImplementedError
        
    def is_terminal(self, s=None):
        """Return `True` if the environment is in a terminal state.
        
        Args:
            s: State to check for termination. Defaults to current state.

        Returns:
            True if state is terminal, False otherwise.
        """
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
        """Set: The set of all states in the environment."""
        return self.nonterminals + self.terminals

    @property 
    def nonterminals(self):
        """Set: The set of nonterminal states of the environment."""
        raise NotImplementedError

    @property 
    def terminals(self):
        """Set: The set of terminal states of the environment."""
        raise NotImplementedError

    @property
    def max_actions(self):
        """int: The maximum number of actions available over all states."""
        pass