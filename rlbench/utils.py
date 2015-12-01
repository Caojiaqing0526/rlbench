import numpy as np


def compute_value_dct(theta_lst, features):
    return [{s: np.dot(theta, x) for s, x in features.items()} for theta in theta_lst]

def compute_values(theta_lst, X):
    return [np.dot(X, theta) for theta in theta_lst]

def compute_errors(value_lst, error_func):
    return [error_func(v) for v in value_lst]

def rmse_factory(true_values, d=None):
    true_values = np.ravel(true_values)

    # sensible default for weighting distribution
    if d is None:
        d = np.ones_like(true_values)
    else:
        d = np.ravel(d)
        assert(len(d) == len(true_values))

    # the actual root-mean square error
    def func(v):
        diff = true_values - v
        return np.sqrt(np.mean(d*diff**2))
    return func



