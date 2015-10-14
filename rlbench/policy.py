"""
Policies for selecting actions.

TODO: Composing policies
TODO: Options
TODO: More generally parameterizable policies.
"""
import numpy as np 
from collections import defaultdict


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

    def probabilities(self, s, actions):
        """ Return the probability of selecting each action in state `s`."""
        raise NotImplementedError 

    @property
    def info(self):
        """Summary information about the environment."""
        return dict()
    


class RandomPolicy(Policy):
    """ A policy which selects a random action from the available actions every
    time, regardless of the state that it is in.
    """
    def choose(self, s, actions):
        """Randomly select from available actions, with equal probability."""
        return np.random.choice(actions)

    def probabilities(self, s, actions):
        return np.ones(len(actions))/len(actions)


class FixedPolicy(Policy):
    """ A policy which selects actions according to a fixed distribution for 
    each state.
    """
    # TODO: Rename this
    def __init__(self, dct):
        # define the policy from the supplied dictionary
        self.pol = dict()
        for s, actions in dct.items():
            # dictionary of probabilities-- zero when an action is not present
            self.pol[s] = defaultdict(int)
            for a, p in actions.items():
                # TODO: check that probabilities sum to one
                self.pol[s][a] = p 

    def choose(self, s, actions):
        return np.random.choice(actions, p=[self.pol[s][a] for a in actions])

    def probabilities(self, s, actions):
        return [self.pol[s][a] for a in actions]


class ObliviousPolicy(Policy):
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
