"""
coord_manager.py
by Paul O'Brien
Implements CoordManager class for AE9/AP9
A coordinate Manager so that modules that share the same coordinates do not recompute them

cm = CoordManager(date,x1,x2,x3,sysaxes,alpha=None,dirs=None)
cm.get(var,indices=None)
var is one of 'I','Bm','K','Phi','hmin','Blocal','L','alpha','MLT','BminX','BminY','BminZ'
I - (Nt,Ndir) integral invariant RE
Bm - (Nt,Ndir) mirror magnetic field strength, nT
K - (Nt,Ndir) Kaufmann's K modified second invariant, RE*sqrt(G)
Phi - (Nt,Ndir) third invariant, G*RE^2
hmin - (Nt,Ndir) minimum altitude along drift-bounce orbit, km
L - (Nt,Ndir) McIlwain's L, RE
Blocal - (Nt,) local magnetic field strength, nT
MLT - (Nt,) equatorial magnetic local time, nT
Bmin - (Nt,) equatorial (minimum) magnetic field strength, nT
BminX,BminY,BminZ - (Nt,) GEO coordinates of equatorial crossing, RE
"""

""" change notes:
15 Nov 2022 - 
    added kext optional input (defaults to opq)    
    removed beta, isn't needed, changed dirs2alphabeta to dirs2alpha
    fixed use of dirs vs d1,d2,d3 which was broken
"""

import numpy as np
from .mod_jul_date import datetime2mjd
from .mag_field import fastI, ItoLm, Bfield, coordinate_transform
from .fast_invariants import fast_PhiK, fast_hminK

ANGLE_DEP_COORDS = ['I','Bm','K','Phi','hmin','alpha','L']
SUPPORTED_COORDS = ANGLE_DEP_COORDS+['Blocal','MLT','Bmin','BminX','BminY','BminZ']

class CoordManager(object):
    """
    cm = CoordManager(date,x1,x2,x3,sysaxes,alpha=None,dirs=None,kext='opq')
    date can be array of datenums or mjd
    alpha can be: scalar or (Nt,), (1,Ndirs), or (Nt,Ndirs)
        size (Ndirs,) should also be supported
    x1, x2, x3 - (Ntimes,) or (Ntimes,1), coordinates for points of interest
    dirs - (Ntimes,3) or (Ntimes,Ndirs,3), directions (ECI) for points of interest    
    kext = magnetic field model to use 'opq' or 'igrf'
    public properties:
        mjd - array of MJDs
        Nt - number of times
        Ndir - number of directions
        coords - list of supported coordinates
        alpha - compact pitch angle array, degrees
        alphaI - (Nt,Ndir) expanded pitch angle array, degrees
    methods:
    cm.get(var,indices=None)
    """
    def __init__(self,mjd,x1,x2,x3,sysaxes,alpha=None,dirs=None,kext='opq'):
        self.mjd = datetime2mjd(mjd)
        self.Nt = mjd.size # number of times        
        self.x1 = x1.copy()
        self.x2 = x2.copy()
        self.x3 = x3.copy()
        self.sysaxes = sysaxes
        self.dirs = dirs
        
        assert kext in ['opq','igrf'], 'Unknown field model %s' % kext
        
        self.kext = kext
        
        if (alpha is not None) and (dirs is not None):
            raise Exception('Cannot specify both alpha and dirs')
            
        if dirs is not None:
            self.dirs2alpha()
        else:
            self.alpha = alpha
        self.alphaI = np.atleast_1d(alpha)
        if self.alphaI.ndim == 1:
            if self.alpha.shape[0]==self.Nt:
                self.alphaI = self.alphaI.reshape((self.Nt,1)) # (Nt,1)
            else:                
                # self.alphaI = repmat(self.alphaI.reshape(1,len(self.alphaI)),self.Nt,1) # (Nt,Ndir) - repmat deprecated
                self.alphaI = np.repeat(self.alphaI.reshape(1,len(self.alphaI)),self.Nt,axis=0) # (Nt,Ndir)
        if self.alphaI.shape[0] == 1:
            # self.alphaI = repmat(alpha,self.Nt,1) # Nt,Ndir - repmat deprecated
            self.alphaI = np.repeat(alpha,self.Nt,axis=0) # Nt,Ndir
            
        self.Ndir = self.alphaI.shape[1] # number of angles
        self.coords = SUPPORTED_COORDS
        assert len(self.coords) <= 32,"More than 32 coords not supported"
        self._values = {} # dict of computed coordinates by key=coordinate
        self._bits = {} # set bit for each coordinate by key=coordinate
        for (i,c) in enumerate(self.coords):
            if c in ANGLE_DEP_COORDS:
                self._values[c] = np.full((self.Nt,self.Ndir),np.nan)
            else:
                self._values[c] = np.full((self.Nt,),np.nan)                    
            self._bits[c] = np.int32(1)<<i
        self._bitmask = np.zeros((self.Nt,),dtype=np.int32) # bitmask tracking whether coord has been set, applies to all dirs

    def dirs2alpha(self):
        """compute alpha from directions"""
        # dirs - (Ntimes,3) or (Ntimes,Ndirs,3), directions (ECI) for points of interest    

        unitize = lambda x: x/np.sqrt(np.sum(x**2,axis=1,keepdims=True)) # force unit length of Nx3
        # define my own stack when np < version 1.10.0
        stack = lambda x1,x2,x3 : np.concatenate((x1.reshape((x1.size,1)),x2.reshape((x2.size,1)),x3.reshape((x3.size,1))),axis=1)
        assert self.dirs.shape[self.dirs.ndim-1]==3,'last dimension of dirs must be length 3'
        if self.dirs.ndim == 2: # passed in Nt,3. insert new dim=1 for Ndir=1
            self.dirs = np.expand_dims(self.dirs,1) 
        d1 = self.dirs[:,:,0]
        d2 = self.dirs[:,:,1]
        d3 = self.dirs[:,:,2]

        bhat = unitize(Bfield(self.kext,self.sysaxes,self.mjd,self.x1,self.x2,self.x3))
        (Nt,Ndir) = d1.shape
        alpha = np.full((Nt,Ndir),np.nan)
        for i in range(Ndir):
            dhat = stack(d1[:,i],d2[:,i],d3[:,i])
            dhat = unitize(coordinate_transform(dhat,[self.sysaxes,'GEO'],self.mjd))
            # get angle between dhat and bhat, pitch angle
            alpha[:,i] = np.degrees(np.arccos(np.sum(dhat*bhat,axis=1)))

        alpha = np.minimum(alpha,180.-alpha)

        self.alpha = alpha
        
    def _store(self,at,**kwargs):
        """
        _store(at,**kwargs)
        at - index tuples
        kwargs - key:val pairs to store
        """
        bits = None
        for key in kwargs:
            if bits is None:
                bits = self._bits[key]
            else:
                bits = bits | self._bits[key]
            self._values[key][at] = kwargs[key]
        self._bitmask[at] = np.bitwise_or(self._bitmask[at],bits)
        
    def get(self,key,indices = None):
        """ x = mgr.get(key,indices = None)
        key is string, one of known coords
        indices is an array of time indices into a (Nt,) nd-array
        if indices is omitted, the whole time,dir array is returned
        if indices is a scalar, x will have no Nt dimension
        x is coordinate at requested time indieces
        only some coordinates are angle-dependent
        """

        assert key in self.coords, "Unknown coord '%s'" % (key)
        if indices is None:
            indices = tuple(np.indices(self._bitmask.shape))
        scalar = np.isscalar(indices) 
        if scalar:
            indices = [indices]
        i = np.bitwise_and(self._bitmask[indices],self._bits[key])==0

        if np.any(i): # need to compute some new coords
            at = tuple([x[i] for x in indices])
            if key in ['I','Blocal','Bmin','Bminxyz']:
                (I,BGEO,Bmin,Bminxyz) = fastI(self.kext,self.sysaxes,self.mjd[at],self.x1[at],self.x2[at],self.x3[at],self.alphaI[at])
                Blocal = np.sqrt(np.sum(BGEO**2,axis=1))
                self._store(at,I=I,Blocal=Blocal,Bmin=Bmin,BminX=Bminxyz[:,0],BminY=Bminxyz[:,1],BminZ=Bminxyz[:,2])
            elif key == 'Bm':
                Blocal = self.get('Blocal',at)
                Bm = Blocal.reshape((Blocal.size,1))/np.sin(self.alphaI*np.pi/180)**2
                self._store(at,Bm=Bm)
            elif key == 'alpha':
                Bmin = self.get('Bmin',at)
                Bm = self.get('Bm',at)
                alpha = np.arcsin(np.minimum(1.0,np.sqrt(Bmin.reshape((Bmin.size,1))/Bm)))*180/np.pi
                self._store(at,alpha=alpha)
            elif key == 'K':
                I = self.get('I',at)
                Bm = self.get('Bm',at)/1e5 # nT -> G
                K = I*np.sqrt(Bm) 
                self._store(at,K=K)
            elif key == 'L':
                I = self.get('I',at)
                Bm = self.get('Bm',at)
                Blocal = self.get('Blocal',at)
                Lm = ItoLm(I,Blocal,Bmirror=Bm)
                self._store(at,L=Lm)
            elif key == 'MLT':
                x = self.get('BminX',at)
                y = self.get('BminY',at)
                uthrs = np.remainder(self.mjd,1.0)*24
                MLT = np.remainder(np.arctan2(y,x)*12/np.pi + uthrs+24,24.0)
                self._store(at,MLT=MLT)
            elif key == 'Phi':
                (Phi,K,partials) = fast_PhiK(self.kext,self.sysaxes,self.mjd[at],self.x1[at],self.x2[at],self.x3[at],self.alphaI[at],Bunit='G',partials=self) # Bunit only affects Phi, K
                Bminxyz = partials.Bminxyz
                self._store(at,Phi=Phi,K=K,Blocal=partials.Blocal,Bmin=partials.Bmin,Bm=partials.Bmirror,I=partials.I,BminX=Bminxyz[:,0],BminY=Bminxyz[:,1],BminZ=Bminxyz[:,2])
            elif key == 'hmin':
                (hmin,K,partials) = fast_hminK(self.kext,self.sysaxes,self.mjd[at],self.x1[at],self.x2[at],self.x3[at],self.alphaI[at],Bunit='G',partials=self)  # Bunit only affects K
                Bminxyz = partials.Bminxyz
                self._store(at,hmin=hmin,K=K,Blocal=partials.Blocal,Bmin=partials.Bmin,Bm=partials.Bmirror,I=partials.I,BminX=Bminxyz[:,0],BminY=Bminxyz[:,1],BminZ=Bminxyz[:,2])
            else:
                raise Exception('Unknown variable %s' % key)

        x = self._values[key][indices]
        if scalar:
            x = x.reshape(x.shape[1:]) # remove 1st dim
        return(x)
