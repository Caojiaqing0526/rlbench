import numpy as np


# Provide a means of registering available environments
environment_registry = {}


class MetaEnvironment(type):
    def __new__(meta, name, bases, attrs):
        """Perform actions/make changes upon class definition."""
        # Create the class
        cls = type.__new__(meta, name, bases, attrs)

        # Don't touch base classes
        if bases != (object,):
            environment_registry['name'] = cls
        return cls 


class Environment(object, metaclass=MetaEnvironment):
    """Abstract base class for a discrete environment in the RL framework."""
    def __init__(self, *args, **kwargs):
        pass
    
    def observe(self, s=None):
        """Get the available sensory information for a state (defaulting to the
        current state if none is specified).

        Args:
            s: State to get sensory information about (default current state).
        Returns:
            The observation associated with the state.
        """
        return self.state 

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
        if s0 is None:
            self._state = self.s0
        else:
            self._state = s0
        
    def is_terminal(self, s=None):
        """Return `True` if the environment is in a terminal state.
        
        Args:
            s: State to check for termination. Defaults to current state.

        Returns:
            True if state is terminal, False otherwise.
        """
        raise NotImplementedError
    
    def actions(self, s=None):
        """Actions available (in the current state)."""
        raise NotImplementedError
        
    @property
    def state(self):
        """The current state of the environment."""
        return self._state

    @property
    def states(self):
        """Set: The set of all states in the environment."""
        raise NotImplementedError

    @property 
    def nonterminals(self):
        """Set: The set of nonterminal states of the environment."""
        return {s for s in self.states if not self.is_terminal(s)}

    @property 
    def terminals(self):
        """Set: The set of terminal states of the environment."""
        return {s for s in self.states if self.is_terminal(s)}

    @property
    def max_actions(self):
        """int: The maximum number of actions available over all states."""
        return max(len(self.actions(s)) for s in self.states)