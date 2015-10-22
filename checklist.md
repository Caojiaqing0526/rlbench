
# Environments

- [ ] `info` property

## Testbeds 

- [ ] **Random number generation should be seed-able!**
- [ ] Chicken
- [ ] Gridworld
- [ ] Mountain Car

# Policies

- [ ] For off-policy, need to be able to get the probability of each action taken according to behavior and target policy
- [ ] Use this to reimplement `choose` so that it uses the policy's `probabilities` method 

# Features

# Agents

- [ ] Agents should be wrappers for `algos`, computing *parameters* and *features* for them as needed
- [ ] Agent should also contain `policy` for the algorithm-- both the target and behavior policy; it should be responsible for choosing actions (and therefore also computing `rho`) 
- [ ] These could be specified at initialization, or passed to the agent during `update`, overriding the previously specified values.

# Algorithms

- [ ] Need to eventually wrap `Algorithm` objects in `Agent` objects
- [ ] Algorithm parameters should be ordered alphabetically, (perhaps defaulting to `None`?)
- [ ] Do we have an idea of what parameters need to be specified per-timestep for all the algorithms we'll be testing? Having access to all of them will dramatically simplify testing multiple algorithms/agents at a time

# Benchmarking

## Data Generation

- [ ] Basic data generation
- [ ] Data format spec for reproducibility
- [ ] **Where possible, initialize feature repn. at start!**

## Consistency

- [ ] Reducing MDPs to matrix form and solving exactly
- [ ] Implement Monte Carlo solver (do you have this already?)
- [ ] Solving with value iteration
