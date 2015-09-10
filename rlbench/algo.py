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
    pass

class TD(Algo):
    def update(self, x, r, xp, alpha, gamma, lmbda):
        pass