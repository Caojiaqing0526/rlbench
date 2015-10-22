# -*- coding: utf-8 -*-
"""
Implementation of code for benchmarking reinforcement learning algorithms.
"""
import numpy as np
from collections import defaultdict
from parametric import to_parameter


def run_episode(agent, env, max_steps=None):
    """Run an episode in a policy evaluation experiment."""
    ret = []
    t = 0

    # reset the environment and get initial state
    env.reset()
    s = env.state
    while not env.is_terminal() and t < max_steps:
        actions = env.actions
        a = agent.choose(s, actions)
        r, sp = env.do(a)
        agent.update(s, r, sp)

        # log information about the episode
        ret.append((s, a, r, sp))

        # prepare for next iteration
        t += 1
        s = sp
    return ret

################################################################################
# Data Analysis functions
################################################################################

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
    """Compute the return at each step in a trajectory.

    Uses the fact that the return at each step 'backwards' from the end of the
    trajectory is the immediate reward plus the discounted return from the next
    state.
    """
    # convert gamma to a state-dependent parameter
    gamma = to_parameter(gamma)
    rewards = get_rewards(lst)
    gmlst = get_gammas(lst, gamma)
    n = len(lst)
    ret = []
    tmp = 0
    for r, gm in reversed(list(zip(rewards, gmlst))):
        tmp *= gm
        tmp += r
        ret.append(tmp)
    return list(reversed(ret))


def every_visit_mc(lst, gamma):
    # compute the returns at each step
    glst = stepwise_return(lst, gamma)
    # group returns by state
    dct = defaultdict(list)
    for x, g in zip(lst, glst):
        s, a, r, sp = x
        dct[s].append(g)

    # output the averaged returns for each state
    return {k: np.mean(v) for k, v in dct.items()}

def phi_matrix(states, phi):
    return np.array([phi(s) for s in states])

def get_features(states, phi):
    return {s: phi(s) for s in states}

def get_values(states, phi, theta):
    return {s: np.dot(phi(s), theta) for s in states}
