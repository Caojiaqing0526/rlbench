import numpy as np 


class Agent:
    """Abstract base class for agents in the RL framework."""
    def __init__(self, *args, **kwargs):
        self.policy = kwargs.get('policy', None)
        self.behavior = kwargs.get('behavior', None)


    def choose(self, s, actions):
        """ Select an action from possible actions in response to state `s`.
        
        Args:
            s: State to take action in.
            actions: The actions available to the agent in state `s`.

        Returns:
            The action the agent chose.
        """
        raise NotImplementedError
    
    def update(self, s, a, r, sp, **params):
        """Update the agent from the experience it received.

        Args:
            s: The state at the beginning of the transition.
            a: The action performed in state `s`.
            r: The reward, a result of the transition (`s`, `a`, `sp`).
            sp: The new state, a result of action `a` in state `s`.
            **params: Any additional parameters needed to make the update. 
        """
        pass


class HordeAgent(Agent):
    """
    A way of running multiple off policy agents a la Horde. 
    """
    def __init__(self, *args, **kwargs):
        raise(NotImplementedError)