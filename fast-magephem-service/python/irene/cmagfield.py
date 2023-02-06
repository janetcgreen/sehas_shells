"""
module - cmagfield.py
Python wrapper to AE9/AP9 SWx CMagField utilities
Interface by Paul O'Brien


The following functions are provided:
isysaxes = sysaxes2int(sysaxes) decode sysaxes int or string to int
ikext = kext2int(kext)    decode kext int or string to int
kext = ikext2str(ikext) decode ikext int or string to standardized str
bool = isShabanskyOrbit(UThour,Bminxyz) - determine which points are Shabansky (used by fastI_opq)
(Y1,Y2,Y3) = coord_convert(mjd,iSysAxesX,iSysAxesY,X1,X2,X3,...) - coordinate transform
(BGEO,Bmag) = get_field(mjd,iSysAxes,X1,X2,X3,ikext,...) - compute B vector and magnitude
    (get_field_opq availalbe for legacy compatibility)
(I,BGEO,Bmin,Bminxyz) = fastI(mjd,iSysAxes,X1,X2,X3,dAngles,ikext,...) - compute I, etc
    (fastI_opq availalbe for legacy compatibility)
init_lib('/path/to/library.so') - override default library file
set_DbPath('/path/to/igrfDB.h5') - override default database file
"""

# this .py file will hold the routines that need to call the spwx/cmagfield DLL
# the parent mag_field.py is a more generic and provides user services
# so, the API to this library is somewhat "private"
# there will only be one library libmag_util.so

import numpy as np
import ctypes as C

import warnings
warnings.filterwarnings('ignore','invalid value encountered in')

# default file locations
# flyin/lib/libmag_util.so
# flyin/data/igrfDB.h5

import os
assert os.sep == '/', "non-linux systems not supported yet"
_flyin_path = os.path.dirname(os.path.abspath(__file__))
_lib_file = os.path.join(_flyin_path,'lib','libmag_util.so')
_szDbPath = os.path.join(_flyin_path,'data','igrfDB.h5')

_lib = None # uninitialized
def init_lib(lib_file = None):
    """
    init_lib(lib_file = None)
    initialize library
    lib_file = '/path/to/library.so'
    """
    global _lib
    if lib_file is None:
        lib_file = _lib_file
    _lib = np.ctypeslib.load_library(_lib_file,'.')
    #  int make_invariant_integral_opq( 
    _lib.make_invariant_integral_opq.restype = C.c_int
    _lib.make_invariant_integral_opq.argtypes = [C.c_char_p, # const char*   szDbPath,
                                                  C.POINTER(C.c_int),    # const int*    piNumtimes, 
                                                  C.POINTER(C.c_double), # const double* pdTimes,     // [iNumTimes]
                                                  C.POINTER(C.c_int),    # const int*    piSysAxes,
                                                  C.POINTER(C.c_double), # const double* pX1,         // [iNumTimes]
                                                  C.POINTER(C.c_double), # const double* pX2,         // [iNumTimes]
                                                  C.POINTER(C.c_double), # const double* pX3,         // [iNumTimes]
                                                  C.POINTER(C.c_int),    # const int*    piNumAngles,
                                                  C.POINTER(C.c_int),    # const int*    piFixedAngles,
                                                  C.POINTER(C.c_double), # const double* pdAngles,    // [iNumAngles]
                                                  C.POINTER(C.c_double), # double*       pdBGEO,      // [iNumTimes x 3] row major order
                                                  C.POINTER(C.c_double), # double*       pdBmin,      // [iNumTimes]
                                                  C.POINTER(C.c_double), # double*       pdBminxyz,   // [iNumTimes x 3] row major order
                                                  C.POINTER(C.c_double)] # double*       pdI          // [iNumTimes x iNumAngles] row major order
    #  int get_field(
    _lib.get_field.restype = C.c_int
    _lib.get_field.argtypes = [C.c_char_p, # const char*   szDbPath,
                                   C.POINTER(C.c_int),    # const int*    piNumtimes, 
                                   C.POINTER(C.c_double), # const double* pdTimes,     // [iNumTimes]
                                   C.POINTER(C.c_int),    # const int*    piSysAxes,
                                   C.POINTER(C.c_double), # const double* pX1,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX2,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX3,         // [iNumTimes]
                                   C.POINTER(C.c_int),    # const int*    pikext,
                                   C.POINTER(C.c_double), # const double* pdMagInputs,  // [iNumTimes x 25] row major order
                                   C.POINTER(C.c_double), # double*       pdBGEO,      // [iNumTimes x 3] row major order
                                   C.POINTER(C.c_double)] # double*       pdBmag      // [iNumTimes]

    #  int get_field_opq(
    _lib.get_field_opq.restype = C.c_int
    _lib.get_field_opq.argtypes = [C.c_char_p, # const char*   szDbPath,
                                   C.POINTER(C.c_int),    # const int*    piNumtimes, 
                                   C.POINTER(C.c_double), # const double* pdTimes,     // [iNumTimes]
                                   C.POINTER(C.c_int),    # const int*    piSysAxes,
                                   C.POINTER(C.c_double), # const double* pX1,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX2,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX3,         // [iNumTimes]
                                   C.POINTER(C.c_double), # double*       pdBGEO,      // [iNumTimes x 3] row major order
                                   C.POINTER(C.c_double)] # double*       pdBmag      // [iNumTimes]


    #  int coord_convert(
    _lib.coord_convert.restype = C.c_int
    _lib.coord_convert.argtypes = [C.c_char_p, # const char*   szDbPath,
                                   C.POINTER(C.c_int),    # const int*    piNumtimes, 
                                   C.POINTER(C.c_double), # const double* pdTimes,     // [iNumTimes]
                                   C.POINTER(C.c_int),    # const int*    piSysAxesX,
                                   C.POINTER(C.c_int),    # const int*    piSysAxesY,
                                   C.POINTER(C.c_double), # const double* pX1,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX2,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pX3,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pY1,         // [iNumTimes]
                                   C.POINTER(C.c_double), # const double* pY2,         // [iNumTimes]
                                   C.POINTER(C.c_double)] # const double* pY3         // [iNumTimes]

def sysaxes2int(sysaxes):
    """
    isysaxes = sysaxes2int(sysaxes):
    decode sysaxes int or string to int:
      0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
      synonyms: GEI=ECI=J2000
    """
    if isinstance(sysaxes,str):
        if sysaxes.lower() == 'gdz':
            return 0
        elif sysaxes.lower() == 'geo':
            return 1
        elif sysaxes.lower() == 'gsm':
            return 2
        elif sysaxes.lower() == 'gse':
            return 3
        elif sysaxes.lower() == 'sm':
            return 4
        elif sysaxes.lower() in ['gei','eci','j2000']:
            return 5
        elif sysaxes.lower() == 'mag':
            return 6
        elif sysaxes.lower() == 'sph':
            return 7
        elif sysaxes.lower() == 'rll':
            return 8
        else:
            raise Exception("sysaxes '%s' not recognized" % sysaxes)
    else:
        return(int(sysaxes))
        
def kext2int(kext):
    """
    ikext = kext2int(kext)
    decode kext int or string to int:
        0:IGRF, 4:T89, 5:OPQ
    """
    if isinstance(kext,str):
        if kext.lower() == 'igrf':
            return 0
        elif kext.lower() == 't89':
            raise Exception('kext t89 not supported yet')
            return 4
        elif kext.lower() == 'opq':
            return 5
        else:
            raise Exception("kext '%s' not recognized" % kext)
    else:
        return(int(kext))
        
def ikext2str(ikext):
    """
    kext = ikext2str(ikext)
    convert integer (or string) ikext to standard string form:
        0:igrf, 4:t89, 5:opq
    """
    ikext = kext2int(ikext) # force to int
    kext_strs = {0:'igrf',4:'t89',5:'opq'}
    if ikext in kext_strs:
        return kext_strs[ikext]
    else:
        raise Exception('unknown/unsupported ikext: %d' % ikext)

def set_DbPath(DbPath):
    """
    set_DbPath('/path/to/igrfDB.h5') - set default database path
    """
    global _szDbPath
    _szDbPath = DbPath

def coord_convert(mjd,iSysAxesX,iSysAxesY,X1,X2,X3,DbPath=None):
    """(Y1,Y2,Y3) = coord_convert(mjd,iSysAxesX,iSysAxesY,X1,X2,X3,DbPath=None)
    Convert X1,X2,X3 from iSysaxesX to iSysAxesY
    
    inputs
    mjd - Ntimes x 1, modified julian dates
    iSysAxesX - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    iSysAxesY - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    X1, X2, X3 - Ntimes x 1, coordinates for points of interest in iSysAxesX
    optional inputs:
    DbPath = '/path/to/igrfDB.h5'

    outputs
    Y1, Y2, Y3 - Ntimes x 1, coordinates for points of interest in iSysAxesY
    """

    # check that the ctypes library object is initialized
    global _lib
    if _lib is None:
        init_lib()

    # supply default path to IGRF data if needed
    if DbPath is None:
        DbPath = _szDbPath

    mjd = np.array(mjd)

    iNumTimes = mjd.size

    Y1 = np.full((iNumTimes,),np.NaN)
    Y2 = np.full((iNumTimes,),np.NaN)
    Y3 = np.full((iNumTimes,),np.NaN)

    result = _lib.coord_convert(C.c_char_p(DbPath.encode('utf-8')),C.pointer(C.c_int(iNumTimes)),
                                mjd.ctypes.data_as(C.POINTER(C.c_double)),
                                C.pointer(C.c_int(iSysAxesX)),
                                C.pointer(C.c_int(iSysAxesY)),
                                X1.ctypes.data_as(C.POINTER(C.c_double)),
                                X2.ctypes.data_as(C.POINTER(C.c_double)),
                                X3.ctypes.data_as(C.POINTER(C.c_double)),
                                Y1.ctypes.data_as(C.POINTER(C.c_double)),
                                Y2.ctypes.data_as(C.POINTER(C.c_double)),
                                Y3.ctypes.data_as(C.POINTER(C.c_double)))

    assert result == 0, "Call to coord_convert did not return with result=0, %g instead" % (result)

    return (Y1,Y2,Y3)

def get_field(mjd,iSysAxes,X1,X2,X3,ikext=5,MagInputs=None,DbPath=None):
    """(BGEO,Bmag) = get_field(mjd,iSysAxes,X1,X2,X3,ikext=5,MagInputs=None,DbPath=None)
    Compute vector field and magnitude (nT) at specified times/locations
    
    inputs
    mjd - Ntimes x 1, modified julian dates
    iSysAxes - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    X1, X2, X3 - Ntimes x 1, coordinates for points of interest
    optional inputs:
    ikext - external field model. int: 0: IGRF, 4: T89, 5: OPQ (default is 5)
    DbPath = '/path/to/igrfDB.h5'
    MagInputs = (Ntimes,25) (default: None)
       maginputs[:,0] = Kp*10

    outputs
    BGEO: Ntimes x 3 - Local magnetic field vector, GEO, nT
    Bmag : Ntimes x 1 - Local field strength, nT
    """

    # check that the ctypes library object is initialized
    global _lib
    if _lib is None:
        init_lib()

    # supply default path to IGRF data if needed
    if DbPath is None:
        DbPath = _szDbPath

    mjd = np.array(mjd)

    iNumTimes = mjd.size

    if MagInputs is None:
        MagInputs= np.zeros((iNumTimes,25))
    assert MagInputs.shape[0] == iNumTimes,'MagInputs must be (Ntimes,25)'
    assert MagInputs.shape[1] == 25,'MagInputs must be (Ntimes,25)'

    dBGEO = np.full((iNumTimes,3),np.NaN)
    dBmag = np.full((iNumTimes,),np.NaN)

    result = _lib.get_field(C.c_char_p(DbPath.encode('utf-8')),C.pointer(C.c_int(iNumTimes)),
                            mjd.ctypes.data_as(C.POINTER(C.c_double)),
                            C.pointer(C.c_int(iSysAxes)),
                            X1.ctypes.data_as(C.POINTER(C.c_double)),
                            X2.ctypes.data_as(C.POINTER(C.c_double)),
                            X3.ctypes.data_as(C.POINTER(C.c_double)),
                            C.pointer(C.c_int(kext)),
                            MagInputs.ctypes.data_as(C.POINTER(C.c_double)),
                            dBGEO.ctypes.data_as(C.POINTER(C.c_double)),
                            dBmag.ctypes.data_as(C.POINTER(C.c_double)))

    assert result == 0, "Call to get_field did not return with result=0, %g instead" % (result)
    return (dBGEO,dBmag)

def get_field_opq(mjd,iSysAxes,X1,X2,X3,DbPath=None):
    """(BGEO,Bmag) = get_field_opq(mjd,iSysAxes,X1,X2,X3,DbPath=None)
    Compute vector field and magnitude (nT) at specified times/locations
    
    inputs
    mjd - Ntimes x 1, modified julian dates
    iSysAxes - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    X1, X2, X3 - Ntimes x 1, coordinates for points of interest
    optional inputs:
    DbPath = '/path/to/igrfDB.h5'

    outputs
    BGEO: Ntimes x 3 - Local magnetic field vector, GEO, nT
    Bmag : Ntimes x 1 - Local field strength, nT
    """
    return get_field(mjd,iSysAxes,X1,X2,X3,ikext=5,DbPath=None)


def fastI(mjd,iSysAxes,X1,X2,X3,dAngles,FixedAngles=None,ikext=5,MagInputs=None,DbPath=None):
    """(I,BGEO,Bmin,Bminxyz) = fastI(mjd,iSysAxes,X1,X2,X3,dAngles,ikext=5,MagInputs=None,DbPath=None)
    Compute invariant integral I and other field line quantities
    
    inputs
    mjd - Ntimes x 1, modified julian dates
    iSysAxes - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    X1, X2, X3 - Ntimes x 1, coordinates for points of interest
    dAngles - local pitch angles of interest, degrees
      Ntimes x Nangles - angles of interest vary with time
      1 x Nangles - angles of interest are the same at all times
    optional inputs:
    ikext - external field model. int: 0: IGRF, 4: T89, 5: OPQ (default is 5)
    DbPath = '/path/to/igrfDB.h5'
    MagInputs = (Ntimes,25) (default: None)
       maginputs[:,0] = Kp*10
    FixedAngles = <bool> specify whether pitch angle varies with time (otherwise, tries to guess)

    outputs
    I : Ntimes x Nangles - McIlwain's integral invariant, Re
    BGEO: Ntimes x 3 - Local magnetic field vector, GEO, nT
    Bmin : Ntimes x 1 - Minimum field strength on local field line, nT
    Bminxyz: Ntimes x 3 - Location of Bmin in GEO, Re
    """


    # check that the ctypes library object is initialized
    global _lib
    if _lib is None:
        init_lib()

    # supply default path to IGRF data if needed
    if DbPath is None:
        DbPath = _szDbPath

    mjd = np.array(mjd)

    iNumTimes = mjd.size
    iFixedAngles = -1 # not initialized
    iNumAngles = dAngles.size # assume FixedAngles

    if MagInputs is None:
        MagInputs= np.zeros((iNumTimes,25))
    assert MagInputs.shape[0] == iNumTimes,'MagInputs must be (Ntimes,25)'
    assert MagInputs.shape[1] == 25,'MagInputs must be (Ntimes,25)'
    
    if FixedAngles is None:
        if (dAngles.ndim == 1):
            if (dAngles.size == iNumTimes):
                iFixedAngles = 0
                iNumAngles = 1 # only one angle, different each time
            else:
                iFixedAngles = 1
        else:
            Na,iNumAngles = dAngles.shape
            if (Na == iNumTimes) :
                iFixedAngles = 0
            else :
                iFixedAngles = 1
    else:
        iFixedAngles = int(FixedAngles)

    dBGEO = np.full((iNumTimes,3),np.NaN)
    dBmin = np.full((iNumTimes,),np.NaN)
    dBminXyz = np.full((iNumTimes,3),np.NaN)
    dI = np.full((iNumTimes,iNumAngles),np.NaN)

    dAngles = dAngles.ravel(order='C')
    # make acute angles
    dAngles = np.minimum(dAngles,180-dAngles)
    dAngles[~(dAngles>0)] = 0 # alpha=0 not traced

    result = _lib.make_invariant_integral(C.c_char_p(DbPath.encode('utf-8')),C.pointer(C.c_int(iNumTimes)),
                                      mjd.ctypes.data_as(C.POINTER(C.c_double)),
                                      C.pointer(C.c_int(iSysAxes)),
                                      X1.ctypes.data_as(C.POINTER(C.c_double)),
                                      X2.ctypes.data_as(C.POINTER(C.c_double)),
                                      X3.ctypes.data_as(C.POINTER(C.c_double)),
                                      C.pointer(C.c_int(iNumAngles)),
                                      C.pointer(C.c_int(iFixedAngles)),
                                      dAngles.ctypes.data_as(C.POINTER(C.c_double)),
                                      C.pointer(C.c_int(ikext)),
                                      MagInputs.ctypes.data_as(C.POINTER(C.c_double)),                                      
                                      dBGEO.ctypes.data_as(C.POINTER(C.c_double)),
                                      dBmin.ctypes.data_as(C.POINTER(C.c_double)),
                                      dBminXyz.ctypes.data_as(C.POINTER(C.c_double)),
                                      dI.ctypes.data_as(C.POINTER(C.c_double)))
    assert result == 0, "Call to make_invariant_integral_opq did not return with result=0, %g instead" % (result)
    
    dI[dI<0] = np.NaN # negative I invalid
    # when I is bad, so should be Bmin, but the C code doesn't
    # automatically do this for us
    bad = (dBmin<0) | np.all(~np.isfinite(dI),1) | np.any(~np.isfinite(dBGEO),1)
    UThour = np.fmod(mjd,1.0)*24
    dBminXyz[bad,:] = np.NaN
    bad |= isShabanskyOrbit(UThour,dBminXyz)
    dBmin[bad] = np.NaN
    dBminXyz[bad,:] = np.NaN
    dI[bad,:] = np.NaN

    return (dI,dBGEO,dBmin,dBminXyz)

def fastI_opq(mjd,iSysAxes,X1,X2,X3,dAngles,FixedAngles=None,DbPath=None):
    """(I,BGEO,Bmin,Bminxyz) = fastI_opq(mjd,iSysAxes,X1,X2,X3,dAngles,FixedAngles=None,DbPath=None)
    Compute invariant integral I and other field line quantities for OPQ field
    
    inputs
    mjd - Ntimes x 1, modified julian dates
    iSysAxes - coordinate system for X1, X2, X3. int: 0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
    X1, X2, X3 - Ntimes x 1, coordinates for points of interest
    dAngles - local pitch angles of interest, degrees
      Ntimes x Nangles - angles of interest vary with time
      1 x Nangles - angles of interest are the same at all times
    optional inputs:
    DbPath = '/path/to/igrfDB.h5'
    FixedAngles = <bool> specify whether pitch angle varies with time (otherwise, tries to guess)

    outputs
    I : Ntimes x Nangles - McIlwain's integral invariant, Re
    BGEO: Ntimes x 3 - Local magnetic field vector, GEO, nT
    Bmin : Ntimes x 1 - Minimum field strength on local field line, nT
    Bminxyz: Ntimes x 3 - Location of Bmin in GEO, Re
    """
    return fastI(mjd,iSysAxes,X1,X2,X3,dAngles,FixedAngles=FixedAngles,DbPath=None)    

def isShabanskyOrbit(UThour,Bminxyz):
    """bool = isShabanskyOrbit(UThour,Bminxyz)
    apply Z/MLT limits for Shabansky Orbits in OPQ
    UThour: decimal UT hour (Ntimes x 1)
    Bminxyz: Ntimes x 3 - Location of Bmin in GEO, Re
    bool: boolean Ntimes x 1, true for points that are suspected of
      being on field lines with multiple local minima
    """

    Zlimit = 4; # limit on abs(Z)
    PMLT = np.poly1d([7.84280e-4,0,7.81284e-2,0,9.80100e0]) # polymonial for Rlimit in MLT-12

    #MLT = rem(24+atan2(Bminxyz(:,2),Bminxyz(:,1))*12/pi + rem(matlabd,1)*24,24);
    MLT = np.fmod(24.0+np.arctan2(Bminxyz[:,1],Bminxyz[:,0])*12./np.pi + UThour,24);

    Rlimit = PMLT(MLT-12);
    R = np.sqrt(np.sum(Bminxyz**2,1));
    #result = ((abs(Bminxyz(:,3))>Zlimit) | (R>=Rlimit));
    result = (np.abs(Bminxyz[:,2])>Zlimit) | (R>=Rlimit)

    return result

if __name__ == '__main__':

    X = np.array([[6.6,0,0],[0,6.6,0],[0,0,6.6],[1.0,0,0]],order='F') # hack to allow [:,i] indexing
    mjd = np.full((X.shape[0],),59610.0) # 1/31/2022
    iSysAxes = 1 # GEO
    
    # test coordinate transform
    print('****************')
    print('OPQ get_field test')
    iSysAxesGDZ = 0
    Y = np.zeros(X.shape,order='F') # hack to allow [:,i] indexing
    X2 = np.zeros(X.shape,order='F') # hack to allow [:,i] indexing
    (Y[:,0],Y[:,1],Y[:,2]) = coord_convert(mjd,iSysAxes,iSysAxesGDZ,X[:,0],X[:,1],X[:,2])
    (X2[:,0],X2[:,1],X2[:,2]) = coord_convert(mjd,iSysAxesGDZ,iSysAxes,Y[:,0],Y[:,1],Y[:,2])
    print('X - GEO',X)
    print('Y - GDZ',Y)
    print('X-X2 - GEO',X-X2)
    
    # test IGRF, OPQ, T89 get_field

    print('****************')
    print('OPQ get_field test')
    kext = 5 # OPQ
    (BGEO_opq,Bmag_opq) = get_field_opq(mjd,iSysAxes,X[:,0],X[:,1],X[:,2])
    (BGEO,Bmag) = get_field(mjd,iSysAxes,X[:,0],X[:,1],X[:,2],kext)
    print('BGEO',BGEO_opq,BGEO)
    print('Bmag',Bmag_opq,Bmag)

    print('****************')
    print('IGRF get_field test')
    kext = 0 # IGRF
    (BGEO,Bmag) = get_field(mjd,iSysAxes,X[:,0],X[:,1],X[:,2],kext)
    print('BGEO',BGEO)
    print('BGEO-opq',BGEO_opq)
    print('Bmag',Bmag)
    print('Bmag-opq',Bmag_opq)
    
    #T89 not supported
    #print('****************')
    #print('T89 get_field test')
    #kext = 4 # T89
    #maginputs = np.zeros((X.shape[0],25))
    #maginputs[:,0] = np.array([33.0,23,67,43])
    #(BGEO,Bmag) = get_field(mjd,iSysAxes,X[:,0],X[:,1],X[:,2],kext,MagInputs=maginputs)
    #print('BGEO',BGEO)
    #print('Bmag',Bmag)


    print('***OPQ fastI test')    
    dAngles = np.array([90.0])
    (I,BGEO,Bmin,Bminxyz) = fastI_opq(mjd,iSysAxes,X[:,0],X[:,1],X[:,2],dAngles)    
    print('I',I)
    print('BGEO',BGEO)
    print('Bmin',Bmin)
    print('Bminxyz',Bminxyz)
    
    opq = {'I':I,'BGEO':BGEO,'Bmin':Bmin,'Bminxyz':Bminxyz}
    print('***IGRF fastI test')    
    dAngles = np.array([90.0])
    (I,BGEO,Bmin,Bminxyz) = fastI(mjd,iSysAxes,X[:,0],X[:,1],X[:,2],dAngles,ikext=0)    
    print('I',I)
    print('I-opq',opq['I'])
    print('BGEO',BGEO)
    print('BGEO-opq',opq['BGEO'])
    print('Bmin',Bmin)
    print('Bmin-opq',opq['Bmin'])
    print('Bminxyz',Bminxyz)
