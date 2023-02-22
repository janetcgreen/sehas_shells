"""
broadcast_ndarray.py
by Paul O'Brien
Implements broadcast function

(a,b,c,...) = broadcast(a,b,c,...)

"""

def broadcast(*args):
    """(a,b,c,...) = broadcast(a,b,c,...)
    the outputs will be broadcast to all have the same
    size encompassing the sizes of the input arguments
    all output arguments will be ndarrays of at least one
    dimension and at least length 1. Any arguments with
    a length of 0 along any dimension will remain length
    0 in that dimension, but all others will be expanded.
    """
    import numpy as np
    out = []
    dims = []
    for arg in args:
        x = np.atleast_1d(arg)
        out.append(x)
        N = x.ndim
        while len(dims)<N:
            dims.append(1)
        for i in range(N):
            if dims[i]<x.shape[i]:
                dims[i] = x.shape[i]
    for j in range(len(out)):
        for i in range(len(dims)):
            if out[j].ndim < i+1:
                s = list(out[j].shape)
                s.append(1)
                out[j].shape = tuple(s)
            if(out[j].shape[i]==1) and (dims[i]>1):
                out[j] = out[j].repeat(dims[i],axis=i)
    return tuple(out)

