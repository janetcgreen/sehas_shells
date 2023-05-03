"""
fast_invariants.py
by Paul O'Brien
Implements fast invariant calculations for Phi, hmin, and Lm

Three fast routines are provided
(Phi,K,partials) = fast_PhiK(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None,Bunit='nT')
(hmin,K,partials) = fast_hminK(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None,Bunit='nT')
(Lm,AlphaEq,partials) = fast_LmAlphaEq(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None)

Note:partials can be passed in as a coordinate manager rather than a structure
It will be converted on output

if alpha is (Nalpha,) or (Nalpha,1) for Nalpha>1 rather than (1,Nalpha) then there can
be some ambiguity as to whether alpha varies with time or not. if Nt == Nalpha,
then it will be assumed alpha varies with time. To ensure this never comes up,
always pass in alpha as a (1,Nalpha) shaped ndarray.


common variables
kext - int or string external field model - 0:IGRF, 4:T89, 5:OPQ only at this time
sysaxes - coordinate system for x1, x2, x3. int or string: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
date - Ntimes x 1, datetime or modified julian date
x1, x2, x3 - Ntimes x 1, coordinates for points of interest
alpha - local pitch angles of interest, degrees
    Ntimes x Nangles - angles of interest vary with time
    1 x Nangles - angles of interest are the same at all times
Bunit - string, if it contains 'G' then Phiis in G^2 Re, K in sqrt(G)Re
    otherwise Phi is in nT^2 Re, K in sqrt(nT)Re
Blocal - local B field in nT
Phi - 3rd (drift) invariant nT^2 Re or G^2 Re
K - modified 2nd invariant I*sqrt(Bmirror) sqrt(nT)Re or sqrt(G)Re
Lm - McIlwain's L, Re
AlphaEq - equatorial pitch angle, degrees
partials - a Structure of various partial results to facilitate computing other
    invariants without retracing the local field line
    BGEO (nT), Bmin (nT), Bmirror (nT), I (Re),
    K (Re*sqrt(B)), KBunit ('nT' or 'G'), Bminxyz (GEO, Re)
    *NOTE*: partials input may be changed in-place by adding new fields
"""

import warnings
warnings.filterwarnings('ignore','invalid value encountered in')
import numpy as np
from .structure import Structure
from .mat2struct import mat2struct
from .nnlib import NeuralNet
from .mod_jul_date import datetime2mjd

# default file locations
# irene/data/fastPhi_net.mat
# irene/data/fast_hmin_net.mat

import os
_slash = os.sep
assert _slash == '/', "non-linux systems not supported yet"
_flyin_path = os.path.dirname(os.path.abspath(__file__))
_NN_files = { # file names by [kext][coord] w/o the final .mat, also w/o _Kp# for t89
        'opq':{'Phi':'fastPhi_net','hmin':'fast_him_net'},
        'igrf':{'Phi':'fastPhi_net_igrf','hmin':'fast_hmin_net_igrf'},
        }
_NNs = {key:{} for key in _NN_files} # nets themselves
_NN_path = os.path.join(_flyin_path,'data')


def _make_partials(kext,sysaxes,mjd,x1,x2,x3,alpha,partials,Kp=None):
    """ populate partials, if needed 
    see module helps for list of partials fields
    partials = _make_partials(kext,sysaxes,mjd,x1,x2,x3,alpha,partials,Kp=None)
    """
    from .mag_field import fastI, expand_alpha
    
    if (partials is not None) and not isinstance(partials,Structure): # must be a coordinate manager
        cm = partials
        if (cm.Nt != mjd.size) or np.any(cm.mjd != mjd):
            raise Exception("requested dates not consistent with provided coordinate manager")
            
        partials = Structure()
        partials.KBunit = 'G'
        partials.I = cm.get('I')
        partials.Blocal = cm.get('Blocal')
        partials.Bmin = cm.get('Bmin')
        partials.Bmirror = cm.get('Bm')
        partials.Bminxyz = np.full((cm.Nt,3),np.nan)
        partials.Bminxyz[:,0] = cm.get('BminX')
        partials.Bminxyz[:,1] = cm.get('BminY')
        partials.Bminxyz[:,2] = cm.get('BminZ')
        partials.K = cm.get('K')
        partials.alpha = expand_alpha(cm.alpha,cm.Nt)
        partials.beta = np.zeros(partials.alpha.shape,dtype=float) # beta is deprecated
        if partials.alpha.shape[0] == 1:
            #partials.alpha = np.matlib.repmat(cm.alpha,cm.Nt,1) # repmat deprecated
            partials.alpha = np.broadcast_to(cm.alpha,(cm.Nt,1))

    alpha = expand_alpha(alpha,len(mjd))
    if ('alpha' in partials) and (not np.array_equal(alpha,partials.alpha)):
        raise Exception("requested pitch angles (alpha) not consistent with provided partials")

    if ('I' not in partials) or ('Bmirror' not in partials):
        (partials.I,partials.BGEO,partials.Bmin,partials.Bminxyz) = fastI(kext,sysaxes,mjd,x1,x2,x3,alpha,Kp=Kp)

    (Nt,Nalpha) = partials.I.shape # get expanded size of time and alpha

    if 'alpha' not in partials:
        partials.alpha = alpha.copy()

    if 'beta' not in partials:
        partials.beta = np.zeros(partials.alpha.shape,dtype=float)

    if 'Blocal' not in partials:
        partials.Blocal = np.sqrt(np.sum(partials.BGEO**2.0,axis=1))

    if 'Bmirror' not in partials:
        partials.Bmirror = partials.Blocal.reshape((Nt,1))/np.sin(np.radians(alpha))**2 # sin(alpha)^2/B = 1/Bmirror

    if 'K' not in partials:
        partials.K = partials.I*np.sqrt(partials.Bmirror) # Re*sqrt(B)
        partials.KBunit = 'nT'
    
    return partials

def _run_nn(mjd,partials,info):
    """Y = _run_nn(mjd,partials,info)
    run info.net with mjd,partials
    and apply MJD, I, Bmirror limits from info
    Y is [Nt x Nalpha]
    info is a structure read in by mat2struct, e.g., from fast_hmin_net.mat
     it holds the fast neural network and its metadata
    """
    from datetime import datetime

    (Nt,Nalpha) = partials.I.shape
    
    log10Bm = np.log10(partials.Bmirror) # expects Bmirror in nT
    root4J = partials.I**0.25

    mjd = np.minimum(mjd,info.max_MJD).ravel() # don't go beyond IGRF
    if (mjd.size < Nt):
        mjd = np.repeat(mjd,Nt)
    UT = np.remainder(mjd,1.0) # day fraction
    mjdref = datetime2mjd(np.array([datetime(1950,1,1)]))
    YearPhase = np.remainder(mjd-mjdref,365.25) # like DOY but starts at 0 and close to sidereal year
    # allX rows are time, cols are variables
    allX = np.zeros((root4J.size,9),dtype=float)
    allX[:,0] = root4J.ravel(order='F') # Fortran (Matlab) ordering
    allX[:,1] = log10Bm.ravel(order='F')  # Fortran (Matlab) ordering
    # these repeats imply Fortran (Matlab) ordering for mapping from row of X to row,col of Y
    duplicate = lambda x : np.repeat(x.reshape(1,Nt),Nalpha,axis=0).ravel()
    allX[:,2] = duplicate(mjd)
    allX[:,3] = duplicate(YearPhase)
    allX[:,4] = duplicate(UT)
    allX[:,5] = duplicate(np.cos(YearPhase*2*np.pi/365.25))
    allX[:,6] = duplicate(np.sin(YearPhase*2*np.pi/365.25))
    allX[:,7] = duplicate(np.cos(UT*2*np.pi))
    allX[:,8] = duplicate(np.sin(UT*2*np.pi))
    X = []
    allX_keys = ['root4I','log10Bm','mjd','YearPhase','UT','cosYP','sinYP','cosUT','sinUT']
    for key in info.inputs:
        if key not in allX_keys:
            raise Exception('Neural Network input %s unknown' % key)
        X.append(allX[:,allX_keys.index(key)])
    X = np.column_stack(X)

    maxBmirror = np.polyval(info.Pmax_log10B,root4J)
    # check whether Bm is too high for the specified value of I
    # and whether I is too high
    # this is based on log10Bm and root4J
    is_trapped = np.nonzero((log10Bm <= maxBmirror) & (root4J < info.max_root4J))
    ind_trapped = np.ravel_multi_index(is_trapped,(Nt,Nalpha),order='F')
    Y = np.zeros((Nt,Nalpha),dtype=float)
    Y.fill(np.nan)
    Y[is_trapped] = info.net.eval(X[ind_trapped,:]).ravel()

    return Y

def _fast_invariants_core(coord,kext,sysaxes,dates,x1,x2,x3,alpha,partials=None,Bunit='nT',Kp=None):
    """ core calculations common to fast_Phi and fast_hmin
        (Y,K,partials) = _fast_invariants_core(kext,sysaxes,dates,x1,x2,x3,alpha,partials=None,Bunit='nT',Kp=None)
        coord is one of 'Phi' or 'hmin'
        Y is neural network output (log10PhiG or hmin)
    """
    from .broadcast_ndarray import broadcast
    from .cmagfield import ikext2str
    
    kext = ikext2str(kext) # standardize as opq, igrf, t89

    assert coord in ['Phi','hmin'],'Coordinate %s not supported in fast NN calculations' % coord
    assert kext in ['opq','igrf'],"Cannot compute fast %s for kext = '%s'" % (coord,kext)

    if not kext in _NNs:
        raise Exception('kext "%s" not supported in coord NN' % (kext,coord))
    if coord in _NNs[kext]:
        info = _NNs[kext][coord]
    else:
        basefile = _NN_files[kext][coord]
        if kext == 't89':
            raise Exception('This kind of run needs to be handled by grouping times by Kp')
            assert (Kp>=0) and (Kp<9.5), 'Kp out of range %s' % str(Kp)
            iKp = min(int(Kp),6)
            basefile = '%s_Kp%d' % (kext,iKp)
        file = os.path.join(_NN_path,basefile+'.mat')
        
        info = mat2struct(file,root_uep='/net') # load data into struct
        info.net = NeuralNet(info) # use struct to initialize NeuralNet
        if 'inputs' not in info:
            info.inputs = 'root4I,log10Bm,mjd,YearPhase,UT'
        info.inputs = info.inputs.replace(' ','').split(',')
        _NNs[kext][coord] = info

    if partials is None:
        partials = Structure() # new structure
        
    mjd = datetime2mjd(dates)
    (mjd,x1,x2,x3) = broadcast(mjd,x1,x2,x3)
    
    partials = _make_partials(kext,sysaxes,mjd,x1,x2,x3,alpha,partials,Kp=Kp) # populate partials, if needed

    K = partials.K # Re sqrt(nT)
    # fix units of K

    if (partials.KBunit.find('G') != -1) and (Bunit.find('G')==-1): # K in G, Bunit in nT
        K = K*np.sqrt(1.0e5) # sqrt(G) -> sqrt(nT)
    elif (partials.KBunit.find('G') == -1) and (Bunit.find('G') !=-1): # K in nT, Bunit in G
        K = K/np.sqrt(1.0e5) # sqrt(nT) -> sqrt(G)

    # now get Y

    Y = _run_nn(mjd,partials,info) # run the neural network
    return (Y,K,partials)

def fast_PhiK(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None,Bunit='nT',maginputs=None,Kp=None):
    """
        compute Phi,K fast!
        see module helps
    """
    (Y,K,partials) = _fast_invariants_core('Phi',kext,sysaxes,dates,x1,x2,x3,alpha,partials=partials,Bunit=Bunit,Kp=Kp)
    Phi = np.power(10.0,Y) # G Re^2
    # fix units of Phi
    if Bunit.find('G')==-1:
        Phi = Phi*1e5 # G -> nT

    return (Phi,K,partials)
    
def fast_hminK(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None,Bunit='nT',Kp=None):
    """
        compute hmin,K fast!
        see module helps
    """

    (Y,K,partials) = _fast_invariants_core('hmin',kext,sysaxes,dates,x1,x2,x3,alpha,partials=partials,Bunit=Bunit,Kp=Kp)

    hmin = Y # no conversion required
    return (hmin,K,partials)


def fast_LmAlphaEq(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,partials=None,Kp=None):
    """
        compute Lm, AlphaEq fast!
        see module helps
    """
    
    from .mag_field import ItoLm

    if partials is None:
        partials = Structure() # new structure

    partials = _make_partials(kext,sysaxes,dates,x1,x2,x3,alpha,partials,Kp=Kp) # populate partials, if needed

    Lm = ItoLm(partials.I,partials.Blocal,Bmirror = partials.Bmirror)
    AlphaEq = np.degrees(np.arcsin(np.sqrt(partials.Bmin.reshape((len(partials.Bmirror),1))/partials.Bmirror)))
    
    return (Lm,AlphaEq,partials)

