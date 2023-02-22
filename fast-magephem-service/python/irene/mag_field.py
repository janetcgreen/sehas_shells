"""
mag_field.py
by Paul O'Brien
Magnetic Field Library interface

Functions:
Y = coordinate_transform(X,transform,dates=None) transform X to Y
BGEO = Bfield(kext,sysaxes,dates,x1,x2,x3,Kp=None)
(I,BGEO,Bmin,Bminxyz) = fastI(kext,sysaxes,dates,x1,x2,x3,alpha = 90.0,Kp=None)
Lm = ItoLm(I,Blocal,alpha=90.0,Bunit='nT')
Lm = ItoLm(I,Blocal,Bmirror=Bmirror,Bunit='nT')
Lstar = Phi2Lstar(Phi,Bunit='nT')
Phi = Phi2Lstar(Lstar,Bunit='nT')
alpha = expand_alpha(alpha,Nt) (turn alpha into row vector if Nt>1)

Common variables:
kext - int or string external field model - 0:IGRF and 5:OPQ only at this time
sysaxes - coordinate system for x1, x2, x3. int or string:
    0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    synonyms: GEI=ECI=J2000
dates - Ntimes x 1, double (mjd) or datetime array
x1, x2, x3 - Ntimes x 1, coordinates for points of interest
d1, d2, d3 - Ntimes x Ndirs, directions (ECI) for points of interest
X - Ntimes x 3 coordinates, following IRBEM-LIB convention
transform - string like 'geo2gdz' or 2-element list of coordiante
  systems, such as ['geo','gdz'] or [1,0] or ['geo',0]
alpha - local pitch angles of interest, degrees
  Ntimes x Ndirs - angles of interest vary with time
  1 x Ndirs - angles of interest are the same at all times
Kp - scalar or Ntimes x 1 value of Kp
I : Ntimes x Nangles - McIlwain's integral invariant, Re
BGEO: Ntimes x 3 - Local magnetic field vector, GEO, nT
Bmin : Ntimes x 1 - Minimum field strength on local field line, nT
Bminxyz: Ntimes x 3 - Location of Bmin in GEO, Re
Y - Ntimes x 3 coordinates, following IRBEM-LIB convention
alpha: Ntimes x Ndirs, local pitch angle, degrees
beta: Ntimes x Ndirs, gyrophase/East-West angle: 0 = zenith, 90 = east, 180 = nadir, -90 = west
  (beta is not used but someday will be)
"""

import warnings
import numpy as np
from .cmagfield import sysaxes2int,kext2int
from .broadcast_ndarray import broadcast
from .mod_jul_date import datetime2mjd

_dipole_only = False
_dipole_B0 = 31000.0 # equatorial field strenght at L=1 in nT

def coordinate_transform(X,transform,dates=None):
    """
    Y = coordinate_transform(X,transform,dates=None)
    transform - string like 'geo2gdz' or 2-element list of coordiante
      systems, such as ['geo','gdz'] or [1,0] or ['geo',0]
    X is Nx3
    transform is
    dates is an array of datetime or mjd, Nx1
    Y is Nx3
    """
    from .cmagfield import coord_convert

    if isinstance(transform,str):
        (sysaxes1,sysaxes2) = transform.split('2')
        transform = [sysaxes2int(sysaxes1),sysaxes2int(sysaxes2)]
    else: # already a list, force to [int1, int2]
        transform = [sysaxes2int(transform[0]),sysaxes2int(transform[1])]
        
    if transform[0] == transform[1]: # no transform required, return a copy
        return X.copy()

    # correct sizes so that X.shape[0] = mjd.shape[0] = max(X.shape[0],mjd.shape[0])
    X = np.array(np.atleast_2d(X),dtype=float)
    if X.size == 3:
        X.shape = (1,3) # force row vector
    mjd = datetime2mjd(dates)
    mjd = np.atleast_1d(mjd)
    if (mjd.size==1) and (X.shape[0]>1):
        mjd = mjd.repeat(X.shape[0],axis=0)
    if (mjd.size>1) and (X.shape[0]==1):
        X = X.repeat(mjd.size,axis=0)
    Y = X.copy()

    # copy X columns to get contiguous data arrays
    X1 = X[:,0].copy()
    X2 = X[:,1].copy()
    X3 = X[:,2].copy()

    # call library function
    (Y[:,0],Y[:,1],Y[:,2]) = coord_convert(mjd,transform[0],transform[1],X1,X2,X3)

    return Y

def _dipole_Bfield(isysaxes,mjd,x1,x2,x3,nargout = 1):
    """
    _dipole_Bfield(isysaxes,mjd,x1,x2,x3,nargout = 1)
    BGEO  = _dipole_Bfield(isysaxes,mjd,x1,x2,x3)
    optional final argument (nargout) specifies number of outputs
    (BGEO,B,Beq,L,Bminxyz) = _dipole_Bfield(isysaxes,mjd,x1,x2,x3,5)
    BGEO - Ntimes x 3, X,Y,Z (GEO) components of B, nT
    optional outputs:
    B - local field strength, nT
    Beq - equtorial field strength, nT
    L - L shell
    Bminxyz - Ntimes x 3, GEO coordinates of equatorial crossing, RE
    """

    (mjd,x1,x2,x3) = broadcast(mjd,x1,x2,x3)
    Ntimes = len(mjd)
    mjd.shape = (Ntimes,1)
    x1.shape = (Ntimes,1)
    x2.shape = (Ntimes,1)
    x3.shape = (Ntimes,1)

    X = np.concatenate((x1,x2,x3),axis=1)
    XGEO = coordinate_transform(X,[isysaxes,'geo'],mjd)

    r = np.sqrt(np.sum(XGEO**2,axis=1))

    lonrads = np.arctan2(XGEO[:,1],XGEO[:,0])
    latrads = np.arcsin(XGEO[:,2]/r)

    L = r/np.cos(latrads)**2
    Beq = _dipole_B0/L**3.0
    smlat = np.sin(latrads)
    cmlat = np.cos(latrads)
    Btmp = Beq/cmlat**6.0
    B = Btmp*np.sqrt(1.0+3.0*smlat**2)
    cphi = np.cos(lonrads)
    sphi = np.sin(lonrads)
    BGEO = np.zeros((Ntimes,3),dtype=float)
    BGEO[:,0] = -3.0*cphi*cmlat*smlat*Btmp # Bx
    BGEO[:,1] = -3.0*sphi*cmlat*smlat*Btmp # By
    BGEO[:,2] = -(3.0*smlat**2-1.0)*Btmp # Bz

    if nargout >= 5:
        Bminxyz = np.zeros((Ntimes,3),dtype=float)
        Bminxyz[:,0] = L*cphi # X
        Bminxyz[:,1] = L*sphi # Y
        # Z = 0

    if nargout == 1:
        return BGEO
    elif nargout == 2:
        return (BGEO,B)
    elif nargout == 3:
        return (BGEO,B,Beq)
    elif nargout == 4:
        return (BGEO,B,Beq,L)
    elif nargout == 5:
        return (BGEO,B,Beq,L,Bminxyz)
    else:
        raise Exception("Number of output arguments (nargout) must be 1-5, requested %d" % nargout)

    return BGEO

def _dipole_fastI(isysaxes,mjd,x1,x2,x3,alpha):
    """
    (I,BGEO,Bmin,Bminxyz) = _dipole_fastI(isysaxes,mjd,x1,x2,x3,alpha)
    isysaxes is already converted to int
    dipole stand in for fastI
    """

    (mjd,x1,x2,x3) = broadcast(mjd,x1,x2,x3)
    Ntimes = len(mjd)

    B = np.nan
    Bmin = np.nan
    L = np.nan
    Bminxyz = np.nan
    (BGEO,B,Bmin,L,Bminxyz) = _dipole_Bfield(isysaxes,mjd,x1,x2,x3,nargout=5)

    # y = sin(alpha_eq)
    # sin(alpha)/B = sin^2(alpha_eq)/Beq  = y^2/Beq
    # y = Beq*sin(alpha)/B
    if len(alpha.shape)==1:
        Nangles = len(alpha)
        alpha = alpha.reshape(1,Nangles) # set up broadcast
    else:
        Nangles = (alpha.shape)[1]
        assert (alpha.shape)[0] == Ntimes,"Size of alpha incompatible with other inputs"

    y = np.reshape(Bmin/B,(Ntimes,1))*np.absolute(np.sin(alpha)) # broadcast to Ntimes x Nangles
    I = L.reshape((Ntimes,1))*_SL_Y(y) # Ntimes x Nangles - McIlwain's integral invariant, Re

    return (I,BGEO,Bmin,Bminxyz)

def _SL_Y(y):
    """ Y(y) as defined by Schulz and Lanzerotti, 1974 """
    T0 = 1.0+1.0/(2.0*np.sqrt(3.0))+np.log(2.0+np.sqrt(3.0)) # S&L 1.28a
    T1 = np.pi/6.0*np.sqrt(2.0) # S&L 1.28b
    Y0 = 2*T0 # S&L 1.31 (limiting case in subsequent text)
    Y = 2*(1-y)*T0+(T0-T1)*(y*np.log(y) + 2.0*y-2.0*np.sqrt(y)) # S&L 1.31
    Y[y==0.0] = Y0
    return Y

def _prep_inputs(sysaxes,dates,x1,x2,x3,kext):
    """
    (isysaxes,mjd,x1,x2,x3,ikext) = _prep_inputs(sysaxes,dates,x1,x2,x3,kext)
    prepare inputs to expected types
    """

    # typecast
    mjd = datetime2mjd(dates)
    (mjd,x1,x2,x3) = broadcast(mjd,np.array(x1,dtype='d'),np.array(x2,dtype='d'),np.array(x3,dtype='d'))
    isysaxes = sysaxes2int(sysaxes)
    ikext = kext2int(kext)

    return (isysaxes,mjd,x1,x2,x3,ikext)

def _make_MagInputs(Nt,Kp=None):
    """
    MagInputs = _make_MagInputs(Nt,Kp=None)
    builds (Nt,25) MagInputs array
    returns None if all geophysical inputs are None
    otherwise returns the (Nt,25) array
    """
    if Kp is None:
        return None
    else:
        MagInputs= np.zeros((Nt,25))
        MagInputs[:,0] = Kp*10
        return MagInputs

def Bfield(kext,sysaxes,dates,x1,x2,x3,Kp=None):
    """ BGEO = Bfield(kext,sysaxes,dates,x1,x2,x3,Kp=None)
    compute local field, see module helps """

    from .cmagfield import get_field
    global _dipole_only # we might assign to this, so we have to declare it global or it will be treated as local

    (isysaxes,mjd,x1,x2,x3,ikext) = _prep_inputs(sysaxes,dates,x1,x2,x3,kext)
    MagInputs = _make_MagInputs(len(mjd),Kp=Kp)

    if not _dipole_only:
        try:
            (BGEO,Bmag) = get_field(mjd,isysaxes,x1,x2,x3,ikext=ikext,MagInputs=MagInputs)
            return BGEO
        except Exception as e:
            print(e)
            warnings.warn("Using dipole only (Bfield)\n" + str(e))
            _dipole_only = True # set module's global flag to henceforth use dipole

    BGEO = _dipole_Bfield(isysaxes,mjd,x1,x2,x3)
    return BGEO

def fastI(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,Kp=None):
    """(I,BGEO,Bmin,Bminxyz) = fastI(kext,sysaxes,dates,x1,x2,x3,alpha=90.0,Kp=None):
    see module help
    """
    from .cmagfield import fastI as fastI_c

    global _dipole_only # we might assign to this, so we have to declare it global or it will be treated as local

    (isysaxes,mjd,x1,x2,x3,ikext) = _prep_inputs(sysaxes,dates,x1,x2,x3,kext)
    alpha = np.array(np.atleast_1d(alpha),dtype=float)
    MagInputs = _make_MagInputs(len(mjd),Kp=Kp)


    if not _dipole_only:
        try:
            (I,BGEO,Bmin,Bminxyz) = fastI_c(mjd,isysaxes,x1,x2,x3,alpha,ikext=ikext,MagInputs=MagInputs)
            return (I,BGEO,Bmin,Bminxyz)
        except Exception as e:
            print(e)
            warnings.warn("Using dipole only (fastI)\n" + str(e))
            _dipole_only = True # set module's global flag to henceforth use dipole


    (I,BGEO,Bmin,Bminxyz) = _dipole_fastI(isysaxes,mjd,x1,x2,x3,alpha)
    return (I,BGEO,Bmin,Bminxyz)

def expand_alpha(alpha,Nt):
    """ alpha = expand_alpha(alpha,Nt)
    expand alpha to 1 x Nalpha, if needed
    """
    if alpha is None:
        return None
    alpha = np.array(np.atleast_1d(alpha),dtype=float)

    if (alpha.ndim == 1) and (alpha.size != Nt):
        alpha = np.reshape(alpha,(1,len(alpha))) # prepare to broadcast

    return(alpha)

def ItoLm(I,Blocal,**kwargs):
    """
    Lm = ItoLm(I,Blocal,alpha=90.0,Bunit='nT')
    Lm = ItoLm(I,Blocal,Bmirror=Bmirror,Bunit='nT')
    compute McIlwain's L (Lm)
    uses McIlwain's value of Earth's dipole
    but Hilton's fast conversion algorithm
    I = Invariant integral (in R_E)
    Blocal = local field strength (in Bunit)
    alpha = local pitch angle (in degrees)
    Bmirror = mirror field strength (in Bunit)
    Bunit = 'nT' or 'G' unit of Blocal,Bmirror
    """
    Bunit = 'nT'
    alpha = 90.0
    Bmirror = None

    for key in kwargs:
        if key.lower() == 'bmirror':
            Bmirror = kwargs[key]
        elif key.lower() == 'alpha':
            alpha = kwargs[key]
        elif key.lower() == 'bunit':
            Bunit = kwargs[key]

    (I,Blocal) = broadcast(I,Blocal)
    if Bmirror is None:
        alpha = expand_alpha(alpha,Blocal.size)
        Bmirror = Blocal/np.sin(np.radians(alpha))**2

    k0 = 0.311653 # G*R_E^3 McIlwain's dipole moment
    Bmk0 = Bmirror/k0 # Bmirror/k0
    if Bunit.find('G') == -1: # no G, assume nT
        Bmk0 /= 1.0e5 # G to nT

    # Hilton's algorithm
    a1 = 3/np.pi*np.sqrt(2.0) # 1.35047 #  3*sqrt(2)/pi;
    a2 = 0.465376 # determined by Hilton's least squares fit
    a3 = (2.0+3.0**(-0.5)*np.log(3.0**0.5 + 2.0))**-3.0 # 0.0475455 # (2+3^(-1/2)*log(3^(1/2) + 2))^-3; % repaired missing - sign in Hilton
    XY = I**3*Bmk0
    YY = 1.0 + a1*XY**(1.0/3.0) + a2*XY**(2.0/3.0) + a3*XY
    Lm = (YY/Bmk0)**(1.0/3.0)
    return Lm

def Phi2Lstar(Phi,Bunit='nT'):
    """ compute L* from Phi, or vice versa
        Lstar = Phi2Lstar(Phi,Bunit='nT') (uses epoch 2000.0 IGRF dipole moment)
        Phi = Phi2Lstar(Lstar,Bunit='nT') (uses epoch 2000.0 IGRF dipole moment)
        converts Phi to Lstar or vice versa based on epoch 2000.0 IGRF dipole moment
        for Phi in <Bunit>*Re^2
    """        
    
    k0nT = 1e5*0.301153 # k0 in nT-RE^3 for 2000.0
    Lstar = 2*np.pi*k0nT/Phi
    if 'G' in Bunit.upper():
        Lstar = Lstar/1e5
        
    return Lstar
    
