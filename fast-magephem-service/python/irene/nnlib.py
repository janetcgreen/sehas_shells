"""
nnlib.py
by Paul O'Brien
Implements minimal set of neural network library routines, pure python

net - NeuralNet(info) -- construct a neural network object from structure
Y = net.eval(X) -- evaulate a neueral network Nt times, X: Nt x Nx numpy array of real

"""

import numpy as np

class NeuralNet(object):
    """ net = NeuralNet(info)
    a neural network object 
    uses info.Nx, .Nh, .Ny, .Ntheta, .xbar, .ybar, .sx, .sy, 
     and info.theta_struct.w0, .v0, .w, .v
    Y = net.eval(X) - evaluate network at X
    """

    def __init__(self,info):
        """ net = NeuralNet(info) - loads network from Structure """
        strip = lambda x : np.asscalar(np.array(x)) # safe way to handle scalars that may be wrapped in arrays
        self._Nx = int(strip(info.Nx))
        self._Nh = int(strip(info.Nh))
        self._Ny = int(strip(info.Ny))
        self._Ntheta = int(strip(info.theta_struct.Ntheta))
        # the next set of items are 1-d arays, so force then to be row type to ensure correct broadcast
        self._xbar = np.array(info.xbar).reshape(1,self._Nx)
        self._ybar = np.array(info.ybar).reshape(1,self._Ny)
        self._sx = np.array(info.sx).reshape(1,self._Nx)
        self._sy = np.array(info.sy).reshape(1,self._Ny)
        self._w0 = np.array(info.theta_struct.w0).reshape(1,self._Nh)
        self._v0 = np.array(info.theta_struct.v0).reshape(1,self._Ny)

        # these last two are transposed from how we wish to use them in python with dot
        self._w = np.array(info.theta_struct.w)
        assert self._w.shape == (self._Nx,self._Nh), "w shape incorrect"
        self._v = np.array(info.theta_struct.v)
        if self._Ny == 1:
            self._v = self._v.reshape(self._Nh,self._Ny)
        assert self._v.shape == (self._Nh,self._Ny), "v shape incorrect"

    def __str__(self):
        """ String describing neural network object """
        return "A neural network with Nx = " + str(self._Nx) + " Nh = " + str(self._Nh) + " Ny = " + str(self._Ny)

    def __repr__(self):
        """ a wrapper for __str__ """
        return self.__str__()

    def eval(self,X):
        """ Y = net.eval(X) - evaluate network at X
        implement equations 1-3 of nnlib.pdf (IRBEM-LIB extras)
        X is Nt x Nx - ndarray of real
        Y is Nt x Ny - ndarray of real
        """

        assert (X.shape)[1] == self._Nx, "X must be N x Nx(=%d" % self._Nx

        # equation 3:
        Z = (X-self._xbar)/self._sx

        u = np.dot(Z,self._w)+self._w0 # Z*w+w0

        # equation 2:
        g = 1.0/(1.0+np.exp(-u))

        # equation 1:
        Y = (np.dot(g,self._v)+self._v0)*self._sy+self._ybar # (g*v+v0)*sy+ybar

        return Y


