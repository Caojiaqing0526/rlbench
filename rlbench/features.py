import numpy as np


class Int2Unary:
    """
    Convert integer to unary representation (e.g., for tabular case)
    """
    def __init__(self, length, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = length
        self._array = np.eye(length)

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        return self._array[x]

    @property 
    def terminals(self):
        return self._terminals


class Int2Binary:
    """
    Convert integer to its bit vector representation.
    On initialization, it precomputes an array which is used to extract the 
    individual bits of each integer. 
    """
    def __init__(self, length, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        # Precompute the array for converting integers to bit vectors
        self.length = length
        self._array = (1 << np.arange(length))

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        x = np.array(x)
        ret = []
        for i in x.flat:
            ret.append((i & self._array) > 0)
        return np.ravel(ret).astype(np.uint8)

    @property 
    def terminals(self):
        return self._terminals

    @property 
    def nin(self):
        return 1

    @property 
    def nout(self):
        return self.length


class RandomBinary:
    def __init__(self, length, num_active, terminals=None, random_seed=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = length
        self.num_active = num_active
        self.random_state = np.random.RandomState(random_seed)
        self.mapping = {}

    def __call__(self, x):
        return self.mapping.setdefault(x, self.gen())

    def gen(self):
        # TODO: A better name for this
        indices = np.arange(self.length)
        nonzero = self.random_state.choice(indices, self.num_active, replace=False)
        ret = np.zeros(self.length)
        ret[nonzero] = 1
        return ret

    @property 
    def terminals(self):
        return self._terminals


