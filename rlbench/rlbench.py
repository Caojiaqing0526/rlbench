# -*- coding: utf-8 -*-
"""
Implementation of code for benchmarking reinforcement learning algorithms.
"""
import numpy as np
from collections import defaultdict
from parametric import to_parameter


def get_rewards(lst):
    # lst = [(s1, a1, r2, s2), (s2, a2, r3, s3), ...]
    return [i[2] for i in lst]

def get_states(lst):
    # lst = [(s1, a1, r2, s2), (s2, a2, r3, s3), ...]
    return [i[0] for i in lst]

def get_next_states(lst):
    # lst = [(s1, a1, r2, s2), (s2, a2, r3, s3), ...]
    return [i[3] for i in lst]

def get_gammas(lst, gamma):
    # convert gamma to state-dependent parameter
    gamma = to_parameter(gamma)
    return [gamma(s) for s in get_states(lst)]

def stepwise_params(lst, param):
    param = to_parameter(param)
    return [param(s) for s in get_states(lst)]

def stepwise_return(lst, gamma):
    # convert gamma to a state-dependent parameter
    gamma = to_parameter(gamma)
    rewards = np.array(get_rewards(lst))
    gmlst = np.array(get_gammas(lst, gamma))
    n = len(lst)
    ret = []
    for i in range(n):
        # TODO: This seems inefficient, but I'm unsure how to improve it
        ret.append(sum(rewards[j]*np.prod(gmlst[i:j]) for j in range(i, n)))
    return ret

def every_visit_mc(lst, gamma):
    # compute the returns at each step
    glst = stepwise_return(lst, gamma)
    # group returns by state
    dct = defaultdict(list)
    for x, g in zip(lst, retlst):
        s, a, r, sp = x
        dct[x].append(g)

    return

def phi_matrix(states, phi):
    return np.array([phi(s) for s in states])