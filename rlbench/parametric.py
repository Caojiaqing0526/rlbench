"""
Implementation of state-dependent parameters.
"""
import numbers
import numpy as np
from collections import UserDict


def to_parameter(x):
    """Helper function for converting values into `Parameter` objects."""
    if isinstance(x, Parameter):
        return x
    elif isinstance(x, numbers.Number):
        return Constant(x)
    elif isinstance(x, dict):
        return Map(x)
    else:
        raise TypeError("Unable to represent as a parameter:", x)


class Parameter:
    """Base class for parameters."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> float:
        pass


class Constant(Parameter):
    """A constant parameter. Returns the value supplied during initialization
    no matter what arguments is called with.
    """
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, *args, **kwargs) -> float:
        return self.value


class Map(UserDict, Parameter):
    """A function that maps keys to values, essentially like a dictionary."""
    def __call__(self, key) -> float:
        return self[key]


class FirstVisit(Parameter):
    """Returns the value supplied during initialization the first time it is
    called, zero thereafter.
    """
    def __init__(self, val_dct):
        self.val_dct = {k:v for k, v in val_dct.items()}
        self.unseen = {k: True for k in val_dct.keys()}

    def __call__(self, key) -> float:
        if (key in self.val_dct and self.unseen[key]):
            self.unseen[key] = False
            return self.val_dct[key]
        else:
            return 0

    def reset(self):
        self.unseen = {k: True for k in self.unseen}
