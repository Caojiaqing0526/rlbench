



def test_states(env):
    assert(isinstance(env.states, set))
    assert(isinstance(env.nonterminals, set))
    assert(isinstance(env.terminals, set))
    assert(env.states == env.terminals + env.nonterminals)