import numpy as np 


# Provide a registry of available algorithms 
algo_registry = {}


class MetaAlgo(type):
    def __new__(meta, name, bases, attrs):
        """Perform actions/make changes upon class definition."""
        # Create the class object
        cls = type.__new__(meta, name, bases, attrs)

        # Don't touch base classes
        if bases != (object,):
            algo_registry['name'] = cls            
        return cls 

    def __call__(cls, *args, **kwargs):
        """ Make any changes necessary upon initialization."""
        return type.__call__(cls, *args, **kwargs)


class Algo(object, metaclass=MetaAlgo):
    """Update the agent from the experience it received.

    Args:
        s: The state at the beginning of the transition.
        a: The action performed in state `s`.
        r: The reward, a result of the transition (`s`, `a`, `sp`).
        sp: The new state, a result of action `a` in state `s`.
        **params: Any additional parameters needed to make the update. 
    """
    pass


class TD(Algo):
    def __init__(self, n, **kwargs):
        # TODO: Documentation
        self.n = n 
        self.z = np.zeros(n)
        self.theta = np.zeros(n)

    def update(self, x, r, xp, alpha, gm, lm):
        # TODO: Documentation
        self.z = x + gm*lm*self.z
        delta = r + gm*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.theta += alpha*delta*self.z 
        


class ETD(Algo):
    def update(self, x, r, xp, alpha, gm, gm_p, lmbda, rho, interest):
        self.e = rho*()