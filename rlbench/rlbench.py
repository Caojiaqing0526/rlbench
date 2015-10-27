# -*- coding: utf-8 -*-
"""
Implementation of code for benchmarking reinforcement learning algorithms.
"""
import numpy as np
from collections import defaultdict
from parametric import to_parameter


def run_episode(agent, env, max_steps):
    """Run an episode in a policy evaluation experiment."""
    ret = []
    t = 0

    # reset the environment and get initial state
    env.reset()
    s = env.state
    while not env.is_terminal() and t < max_steps:
        # choose an action, take it, and observe the result
        actions = env.actions
        a = agent.choose(s, actions)
        r, sp = env.do(a)

        # update the agent
        agent.update(s, a, r, sp)

        # record the transition
        ret.append((s, a, r, sp))

        # prepare for next iteration
        t += 1
        s = sp
    return ret


def run_errors(agent, env, max_steps, val_dct):
    """Run an episode in a policy evaluation experiment, recording the agent's
    RMSE vs. known state values.
    """
    ret = []
    t = 0

    # record/precompute information about the environment
    states = env.states
    features = {s: agent.phi(s) for s in states}
    target_values = np.array([val_dct[s] for s in states])

    # reset the environment and get initial state
    env.reset()
    s = env.state
    while not env.is_terminal() and t < max_steps:
        # choose an action, take it, and observe the result
        actions = env.actions
        a = agent.choose(s, actions)
        r, sp = env.do(a)

        # update the agent
        agent.update(s, a, r, sp)

        # get the agent's state values and compare with the target values
        theta = agent.theta
        values = np.array([np.dot(theta, features[s]) for s in states])
        difference = target_values - values
        error = np.sqrt(np.mean(difference**2))

        # record the transition
        ret.append(error)

        # prepare for next iteration
        t += 1
        s = sp
    return ret


def run_policy(pol, env, max_steps, param_funcs=dict()):
    """Run a policy in an environment for a specified number of steps."""
    ret = []
    t = 0

    # reset the environment and get initial state
    env.reset()
    s = env.state
    while not env.is_terminal() and t < max_steps:
        # choose and take action according to the policy, observe result
        actions = env.actions
        a = pol.choose(s, actions)
        r, sp = env.do(a)

        # record the transition
        ret.append((s, a, r, sp))

        # prepare for next iteration
        s = sp
        t += 1
    return ret

def run_policy_verbose(pol, env, max_steps, param_funcs=dict()):
    """Run a policy in an environment for a specified number of steps.

    Provide enough information to run the online algorithms offline by recording
    each step's entire context, potentially including the values of parameter
    functions at each point in time.
    """
    ret = []
    t = 0

    # convert parameter functions to `Parameter` type, if needed
    param_funcs = {k: to_parameter(v) for k, v in param_funcs.items()}

    # reset the environment and get initial state
    env.reset()
    s = env.state
    while not env.is_terminal() and t < max_steps:
        # record the context of the time step
        actions = env.actions
        a = pol.choose(s, actions)
        r, sp = env.do(a)

        # record the transition information
        ctx = {'s': s, 'a': a, 'r': r, 'sp': sp, 'actions': actions}

        # record values of parameters for the transition
        for name, func in param_funcs.items():
            ctx[name] = func(s, a, sp)

        # log the context of the transition
        ret.append(ctx)

        # prepare for next iteration
        s = sp
        t += 1
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
