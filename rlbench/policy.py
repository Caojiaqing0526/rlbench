"""
Policies for selecting actions.

TODO: Composing policies
TODO: Options
TODO: More generally parameterizable policies.
"""



class Policy:
    def __init__(self, *args, **kwargs):
        pass

    def choose(self, s, actions):
        """ Select an action from possible actions in response to state `s`.
        
        Args:
            s: State to take action in.
            actions: The actions available to the agent in state `s`.

        Returns:
            The action the agent chose.
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """ A policy which selects a random action from the available actions every
    time, regardless of the state that it is in.
    """
    def choose(self, s, actions):
        """Randomly select from available actions, with equal probability."""
        return np.random.choice(actions)

class FixedPolicy(Policy):
    """ A policy which selects actions according to a fixed distribution for 
    each state.

    TODO: A better name for this object?
    """

class ConstantPolicy(Policy):
    """ A policy which takes actions according to a set of fixed preferences,
    or if none of its preferred actions are available, selects a random action.
    """
    def __init__(self, preferences):
        """ 
        Initialize the policy. 

        Args: 
            preferences: An list of actions, in order of preference.
        """
        # check that preferences are well formed
        self.preferences = preferences

    def choose(self, s, actions):
        for p in self.preferences:
            if p in actions:
                return p 
        else:
            return np.random.choice(actions)
