from functools import singledispatch
from inspect import signature
from numbers import Number 

class MetaAlgo(type):
    def __new__(meta, name, bases, attrs):
        """ Make any changes necessary upon class definition."""
        print(meta)
        print(name)
        print(bases)
        print(attrs)
        # Don't validate base classes
        if bases != (object,):
            """Validate non-base classes."""
            common = ('self', 'x', 'a', 'r', 'xp')
            if 'update' in attrs:
                specific = []
                sig = signature(attrs['update'])
                print(sig)
                for param in sig.parameters:
                    if param not in common:
                        specific.append(param)
                attrs['_update_params'] = tuple(specific)

        return type.__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        """ Make any changes necessary upon initialization."""
        print('__intercepted!')
        print(cls)
        print(args)
        print(kwargs)
        if hasattr(cls, 'update_params'):
            for key, value in kwargs.items():
                if key in cls.update_params and isinstance(value, Number):
                    # Here, we would setup a function that just returns `value`
                    # if this were the `Agent` created from a learning algo...
                    print(key)

        return type.__call__(cls, *args, **kwargs)


class Algo(object, metaclass=MetaAlgo):
    pass


class TD(Algo):
    def update(self, x, a, r, xp, alpha, gamma, lmbda):
        pass