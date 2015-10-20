import numpy as np 


class Agent:
    """Abstract base class for agents in the RL framework.

    It's intended to act as a wrapper for hiding things like the feature 
    mapping, or state-dependent parameters, or other miscellaneous computations
    used in reinforcement learning.
    Unfortunately, RL is sufficiently flexible that attempting to do all of 
    this is likely to either miss an important case or else introduce unwieldy
    complexity...
    """
    def __init__(self, algo, phi=None, default_params=dict(), *args, **kwargs):
        self.algo = algo
        if phi is None:
            self.phi = lambda x: x
        else:
            self.phi = phi 
        self.default_params = default_params.copy()
        raise NotImplementedError
        


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
        # TODO: Rename things here, it seems pretty ugly
        update_params = self.default_params.copy()
        update_params.update(params)

        # This is an idea of how parameters would be passed to the algo
        algo_params['x'] = self.phi(s)
        algo_params['a'] = a 
        algo_params['r'] = r 
        algo_params['xp'] = self.phi(sp)
        raise NotImplementedError


        

class OffPolicyAgent:
    def __init__(self, agent, behavior, phi=None, **kwargs):
        self.agent = agent
        self.behavior = behavior 
        if phi is None:
            self.phi = lambda x: x
        else:
            self.phi = phi 

        # get update parameters needed by the algorithm
        self.update_params = agent.update_params 
        raise NotImplementedError


    def choose(self, s, actions):
        # get the action probabilities for target policy and behavior policy
        prob_pi = self.agent.probabilities(s, actions)
        prob_mu = self.behavior.probabilities(s, actions)
        # choose the action according to behavior policy
        action = np.random.choice(actions, p=prob_mu)
        # compute the importance sampling ratio
        rho = prob_pi[action]/prob_mu[action]

        raise NotImplementedError
