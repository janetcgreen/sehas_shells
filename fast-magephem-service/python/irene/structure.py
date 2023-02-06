"""
structure.py
by Paul O'Brien
Defines class Structure that's like a cook-book Bunch
Set/Get fields via matlab-like syntax: s.foo = 'bar'
Set/Get fields via dict-like syntax: s['foo'] = 'bar'
Overwrites 'in' to be like matlab's 'isfield' (__contains__ method)
"""
from copy import deepcopy

# Structure - a simple class for allowing struct.field syntax on a dict
class Structure(object):
    """Structure class
    struct = Structure(foo='bar',x = 3)
    struct = Structure(some_other_struct) creates a deepcopy, copying all child structures to new structures
        (otherwise, you get references, which causes trouble)
    struct.foo == struct['foo']
    'foo' in struct evaluates to True
    'y' in struct evaluates to False
    'y' not in struct evaluates to True
    struct.z = 'hello'
    struct['a'] = 'bcd'
    d = struct() evaluates to a dictionary (handy for **struct() when dict is needed)
    """
    def __init__(self, *args, **kwds):
        if (len(args)==1):
            self.__dict__.update(deepcopy(args[0].__dict__))
        elif len(args)!=0:
            raise Exception('Unknown syntax to initialize Structure')
        self.__dict__.update(kwds)
    def __str__(self):
        return self.__dict__.__str__()
    def __repr__(self):
        return self.__dict__.__repr__()
    def __getitem__(self,key):
        return self.__dict__[key]
    def __setitem__(self,key,value):
        self.__dict__[key] = value
    def __contains__(self,key):
        """ key in struct (bool) tests for whether key is a named field of struct """
        return key in self.__dict__
    def __call__(self,*arg):
        """ return Structure as dictionary (shorthand for s.__dict__) """
        assert len(arg)==0, "Call Structure() cannot have any arguments"
        return self.__dict__
    def __copy__(self):
        return Structure(**self.__dict__)
    def __deepcopy__(self, memo):
        return Structure(**deepcopy(self.__dict__, memo))

