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
        self.theta = np.zeros(n)
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.old_rho = 0

    def update(self, x, r, xp, alpha, gm, lm, rho):
        # TODO: Documentation
        # TODO: Compare updates from Geist 2014 vs. Precup, Sutton, & Singh 2000 
        self.z = x + gm*lm*self.old_rho*self.z
        delta = r + gm*rho*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.theta += alpha*delta*self.z 

        # prepare for next iteration
        self.old_rho = rho
        

class LSTD(Algo):
    def __init__(self, n, epsilon=1e-6, **kwargs):
        self.n  = n                         # number of features
        self.z  = np.zeros(n)               # traces 
        self.A  = np.eye(n,n) * epsilon     # A^-1 . b = theta^*
        self.b  = np.zeros(n) 
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.old_rho = 0

    def update(self, x, r, xp, gm, lm, rho):
        self.z = x + gm*lm*self.old_rho*self.z 
        self.A += np.outer(self.z, (x - gm*rho*xp))
        self.b += r*rho*self.z 

        # prepare for next iteration
        self.old_rho = rho 

    @property
    def theta(self):
        return np.dot(np.linalg.pinv(self.A), self.b)
    


class GTD(Algo):
    pass

class ETD(Algo):
    def __init__(self, n, **kwargs):
        self.n = n 
        self.theta = np.zeros(n)
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.F = 0
        self.M = 0
        self.old_rho = 0

    def update(self, x, r, xp, alpha, gm, gm_p, lm, rho, interest):
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.F = gm*self.old_rho*self.F + interest 
        self.M = lm*interest + (1 - lm)*self.F 
        self.z = rho*(x*self.M + gm*lm*self.z) 
        self.theta += alpha*delta*self.z 

        # prepare for next iteration
        self.old_rho = rho 
