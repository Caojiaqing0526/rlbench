import numpy as np
from numpy import diag, dot
from numpy.linalg import pinv

# Utility
def normalize(array, axis=None):
    """Normalize an array along an axis."""
    def _normalize(vec):
        return vec/np.sum(vec)
    if axis:
        return np.apply_along_axis(_normalize, axis, array)
    else:
        return _normalize(array)

def clip(array, tol=1e-6):
    """Return `array` with values close to zero set to zero."""
    ret = np.copy(array)
    ret[np.abs(ret) < tol] = 0
    return ret 

def as_op_matrix(x):
    """Convert scalar, vector, or matrix to operator matrix for state-dependent
    parameters.
    """
    if np.ndim(x) == 0:
        ret = x * np.eye(ns)
    elif np.ndim(x) == 1:
        ret = np.diag(x)
    elif np.ndim(x) == 2:
        ret = np.copy(x)
    else:
        raise ValueError("Invalid dimension for parameter:", np.ndim(x))
    return ret

# MDPs
def stationary(mat):
    """Compute the stationary distribution for transition matrix `mat`, via 
    c omputing the solution to the system of equations (P.T - I)*\pi = 0. 
        
    NB: Assumes `mat` is ergodic (aperiodic and irreducible).
    Could do with LU factorization -- c.f. 54-14 in Handbook of Linear Algebra
    """
    P = (np.copy(mat).T - np.eye(len(mat)))
    P[-1,:] = 1
    b = np.zeros(len(mat))
    b[-1] = 1
    x = np.linalg.solve(P, b)
    return normalize(x)

def rmse(a, b, weight=None):
    if weight is None:
    	weight = 1.0
    diff = (a - b)
    return np.sqrt(np.mean(weight*diff**2))

# Solutions
###############################################################################

def ls_solver(P, r, X, gm, d=None):
    """Compute the least-squares solution for an MDP under LFA.

    Args:
      P : The transition matrix under a given policy.
      r : The expected immediate reward for each state under the policy.
      X : The feature matrix (one row for each state)
      gm : The discount parameter, gamma
      d (optional): The stationary distribution to use.

    Returns:
      theta: the weight vector found by the TD solution.
    """
    ns = len(P) # number of states
    I = np.eye(ns)
    # TODO: Check for validity of P, r, X (size and values)
    # TODO: Provide a way to handle terminal states

    # account for scalar, vector, or matrix parameters
    G = parameter_matrix(gm)

    # compute the stationary distribution if unspecified
    if d is None:
        d = stationary(P)
    # the stationary distribution as a matrix
    D = np.diag(d)

    # Solve the equation
    A = X.T @ D @ X
    b = X.T @ D @ pinv(I - P @ G) @ r
    return pinv(A) @ b

def td_solver(P, r, X, gm, lm, d=None):
    """Compute the TD solution for an MDP under linear function approximation.

    Args:
      P : The transition matrix under a given policy.
      r : The expected immediate reward for each state under the policy.
      X : The feature matrix (one row for each state)
      gm : The discount parameter, gamma
      lm : The bootstrapping parameter, lambda
      d (optional): The stationary distribution to use.

    Returns:
      theta: the weight vector found by the TD solution.
    """
    ns = len(P) # number of states
    I = np.eye(ns)
    # TODO: Check for validity of P, r, X (size and values)
    # TODO: Provide a way to handle terminal states

    # account for scalar, vector, or matrix parameters
    G = parameter_matrix(gm)
    L = parameter_matrix(lm)

    # compute the stationary distribution if unspecified
    if d is None:
        d = stationary(P)
    # the stationary distribution as a matrix
    D = np.diag(d)

    # Solve the equation
    A = X.T @ D @ pinv(I - P @ G @ L) @ (I - P @ G) @ X
    b = X.T @ D @ pinv(I - P @ G @ L) @ r
    return pinv(A) @ b

def etd_solver(P, r, X, gm, lm, interest, d_pi=None, d_mu=None):
    """Compute the ETD solution for an MDP under linear function approximation.

    Args:
      P : The transition matrix under a given policy.
      r : The expected immediate reward for each state under the policy.
      X : The feature matrix (one row for each state)
      gm : The discount parameter, gamma
      lm : The bootstrapping parameter, lambda
      interest: The interest parameter.
      d (optional): The stationary distribution to use.

    Returns:
      theta: the weight vector found by the ETD solution.
    """
    ns = len(P) # number of states
    I = np.eye(ns)
    # TODO: Check for validity of P, r, X (size and values)
    # TODO: Provide a way to handle terminal states

    # account for scalar, vector, or matrix parameters
    G = parameter_matrix(gm)
    L = parameter_matrix(lm)
    N = parameter_matrix(interest)

    # compute the stationary distribution if unspecified
    if d_pi is None:
        d_pi = stationary(P)
    if d_mu is None:
        d_mu = d_pi

    # compute the "warp" matrix (P_{\pi,\gamma,\lambda})
    P_lm = I - pinv(I - P @ G @ L) @ (I - P @ G)

    # compute the interest weighted distribution and emphasis trace
    d_i = np.dot(N, d_mu)
    m = d_i @ pinv(I - P_lm)
    M = np.diag(m)

    # Solve the equation
    A = X.T @ M @ pinv(I - P @ G @ L) @ (I - P @ G) @ X
    b = X.T @ M @ pinv(I - P @ G @ L) @ r
    return pinv(A) @ b






