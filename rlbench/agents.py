"""
Implementation of "agents" in the RL framework, here as a sort of wrapper for
the learning algorithm.
The `Agent` class acts as the interface that collects the learning algorithm,
parameters, feature function, and potentially the target and behavior policies
into a single object.
"""
import numpy as np
import parametric


class OnPolicyAgent:
    """An agent designed for on-policy experiments."""
    def __init__(self, algo, behavior, phi=None, update_params=dict()):
        self.algo = algo
        self.behavior = behavior

        # set the feature function
        if phi is None:
            self.phi = lambda x: x
        else:
            self.phi = phi

        # default parameters to use for updating
        self.param_funcs = {k: parametric.to_parameter(v)
                            for k, v in update_params.items()}

        # in the on-policy setting, `rho` is always equal to one.
        self.rho = 1

    def choose(self, s, actions):
        """ Select an action from possible actions in response to state `s`, 
        according to the specified policy.

        Args:
            s: State to take action in.
            actions: The actions available to the agent in state `s`.

        Returns:
            The action the agent chose.
        """
        # choose the action according to behavior policy
        action = self.behavior.choose(s, actions)
        return action

    def update(self, s, r, sp, **params):
        """ Update the agent from the experience it received.

        Uses `self.rho` which will be accurate if the agent's `choose` method
        was used to select the action taken in state `s`.

        Args:
            s: The state at the beginning of the transition.
            r: The reward, a result of the transition (`s`, `a`, `sp`).
            sp: The new state, a result of action `a` in state `s`.
            **params: Any additional parameters needed to make the update.

        Note: The parameters in `**params` override the defaults set during the
        agent's initialization, and are assumed to be numeric, not callables.
        """
        # determine the state dependent update params
        update_params = {k: v(s) for k, v in self.param_funcs.items()}
        # specify rho from previous invocation of `choice`
        update_params['rho'] = self.rho
        # override parameter values as necessary
        update_params.update(**params)
        # determine args to pass to algorithm update
        args = [update_params[k] for k in self.algo.update_params]

        # function approximation
        x = self.phi(s)
        xp = self.phi(sp)

        # update
        self.algo.update(x, r, xp, *args)

    @property
    def theta(self):
        """Return the weight vector `theta` that the algorithm is using for
        function approximation.
        """
        return self.algo.theta

    def predict(self, s):
        """Compute/predict the value for state `s`."""
        return np.dot(self.theta, self.phi(s))

    def get_values(self, states):
        """Compute the values for each of the given states."""
        return {s: np.dot(self.theta, self.phi(s)) for s in states}

    def reset(self):
        """Call the learning algorithm's reset method."""
        self.algo.reset()


class OffPolicyAgent:
    """An agent designed for off-policy experiments."""
    def __init__(self, algo, target, behavior, phi=None, update_params=dict()):
        self.algo = algo
        self.target = target
        self.behavior = behavior

        # set the feature function
        if phi is None:
            self.phi = lambda x: x
        else:
            self.phi = phi

        # default parameters to use for updating
        self.param_funcs = {k: parametric.to_parameter(v)
                            for k, v in update_params.items()}

    def choose(self, s, actions):
        """ Select an action from possible actions in response to state `s`,
        while also modifying `self.rho` to reflect the ratio of probabilities
        between the target policy and the behavior policy.

        Args:
            s: State to take action in.
            actions: The actions available to the agent in state `s`.

        Returns:
            The action the agent chose.
        """
        # get the action probabilities for target policy and behavior policy
        prob_pi = self.target.probabilities(s, actions)
        prob_mu = self.behavior.probabilities(s, actions)
        # choose the action according to behavior policy
        action = np.random.choice(actions, p=prob_mu)
        # compute the importance sampling ratio
        self.rho = prob_pi[action]/prob_mu[action]
        return action

    def update(self, s, r, sp, **params):
        """ Update the agent from the experience it received.

        Uses `self.rho` which will be accurate if the agent's `choose` method
        was used to select the action taken in state `s`.

        Args:
            s: The state at the beginning of the transition.
            r: The reward, a result of the transition (`s`, `a`, `sp`).
            sp: The new state, a result of action `a` in state `s`.
            **params: Any additional parameters needed to make the update.

        Note: The parameters in `**params` override the defaults set during the
        agent's initialization, and are assumed to be numeric, not callables.
        """
        # determine the state dependent update params
        update_params = {k: v(s) for k, v in self.param_funcs.items()}
        # specify rho from previous invocation of `choice`
        update_params['rho'] = self.rho
        # override parameter values as necessary
        update_params.update(**params)
        # determine args to pass to algorithm update
        args = [update_params[k] for k in self.algo.update_params]

        # function approximation
        x = self.phi(s)
        xp = self.phi(sp)

        # update
        self.algo.update(x, r, xp, *args)

    @property
    def theta(self):
        """Return the weight vector `theta` that the algorithm is using for
        function approximation.
        """
        return self.algo.theta

    def predict(self, s):
        """Compute/predict the value for state `s`."""
        return np.dot(self.theta, self.phi(s))

    def get_values(self, states):
        """Compute the values for each of the given states."""
        return {s: np.dot(self.theta, self.phi(s)) for s in states}

    def reset(self):
        """Call the learning algorithm's reset method."""
        self.algo.reset()


class HordeOffPolicy:
    """Off-policy evaluation with multiple concurrent agents."""
    def __init__(self, agents, target, behavior):
        pass


class ScrapAgent:
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
        """ Select an action from possible actions in response to state `s`.

        Args:
            s: State to take action in.
            actions: The actions available to the agent in state `s`.

        Returns:
            The action the agent chose.
        """
        # get the action probabilities for target policy and behavior policy
        prob_pi = self.agent.probabilities(s, actions)
        prob_mu = self.behavior.probabilities(s, actions)
        # choose the action according to behavior policy
        action = np.random.choice(actions, p=prob_mu)
        # compute the importance sampling ratio
        rho = prob_pi[action]/prob_mu[action]

        raise NotImplementedError

    def update(self, s, a, r, sp, **params):
        """ Update the agent from the experience it received.

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