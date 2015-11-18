import inspect
import numpy as np


# Provide a registry of available algorithms
algo_registry = {}


class MetaAlgo(type):
    # TODO: Documentation
    def __new__(meta, name, bases, attrs):
        """Perform actions/make changes upon class definition."""
        # Create the class object
        cls = type.__new__(meta, name, bases, attrs)

        # Don't touch base classes
        if bases != (object,):
            algo_registry[name] = cls
            # store update parameters for each agent as a class attribute
            common_params = ('self', 'x', 'a', 'r', 'xp')
            parameters = inspect.signature(cls.update).parameters
            cls.update_params = [i for i in parameters if i not in common_params]
        return cls

    def __call__(cls, *args, **kwargs):
        """ Make any changes necessary upon initialization."""
        return type.__call__(cls, *args, **kwargs)


class Algo(object, metaclass=MetaAlgo):
    # TODO: Documentation
    """Base class for algorithms."""
    def __init__(self, *args, **kwargs):
        pass

    def update(self, x, r, xp, **params):
        """ Update the agent from the experience it received.

        Args:
            x: The state at the beginning of the transition.
            r: The reward, a result of the transition (`s`, `a`, `sp`).
            xp: The new state, a result of action `a` in state `s`.
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

    def update(self, x, r, xp, alpha, gm, gm_p, lm, rho):
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

    def update(self, x, r, xp, gm, gm_p, lm, rho):
        self.z = x + gm*lm*self.old_rho*self.z
        self.A += np.outer(self.z, (x - gm_p*rho*xp))
        self.b += r*rho*self.z

        # prepare for next iteration
        self.old_rho = rho

    @property
    def theta(self):
        return np.dot(np.linalg.pinv(self.A), self.b)



class ETD(Algo):
    """Emphatic Temporal Difference Learning, or ETD(λ)."""
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


class GTD(Algo):
    """GTD -- Gradient Temporal Difference Learning, minimizing NEU.

    TODO: Is there a version of this that admits eligibility traces?
    TODO: Check this for GVF indexing accuracy
    """
    def __init__(self, n, **kwargs):
        self.n = n
        self.theta = np.zeros(n)
        self.w = np.zeros(n)
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.w = np.zeros(self.n)

    def update(self, x, r, xp, alpha, beta, gm, gm_p, rho):
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.theta += alpha*rho*(x - gm_p*xp)*np.dot(x, self.w)
        self.w += beta*rho*(delta*x - self.w)


class GTD2(Algo):
    """GTD2 -- Gradient Temporal Difference Learning, which minimizes MSPBE.

    TODO: Is there a version of this that admits eligibility traces?
    TODO: Check this for GVF indexing accuracy
    """
    def __init__(self, n, **kwargs):
        self.n = n
        self.theta = np.zeros(n)
        self.w = np.zeros(n)
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.w = np.zeros(self.n)

    def update(self, x, r, xp, alpha, beta, gm, gm_p, rho):
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.theta += alpha*rho*(x - gm_p*xp)*np.dot(x, self.w)
        self.w += beta*(rho*delta - np.dot(x, self.w))*x


class TDC(Algo):
    """The Temporal Difference with Gradient Correction, AKA TDC(λ), AKA GTD(λ).

    See page 74 and 91-92 of Maei's thesis for definition of the algorithm.
    TODO: is there an error in the `w` update? Should it include `rho`?
    """
    def __init__(self, n, **kwargs):
        self.n = n
        self.theta = np.zeros(n)
        self.w = np.zeros(n)
        self.reset()

    def reset(self):
        self.z = np.zeros(self.n)
        self.w = np.zeros(self.n)

    def update(self, x, r, xp, alpha, beta, gm, gm_p, lm, lm_p, rho):
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.z = rho*(x + gm*lm*self.z)
        self.theta += alpha*(delta*self.z - gm_p*(1-lm_p)*np.dot(self.z, self.w)*xp)
        self.w += beta*(delta*self.z - np.dot(x, self.w)*x)