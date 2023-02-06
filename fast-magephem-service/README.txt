SSD HAS Fast MagEphem Service

POC: Paul O'Brien paul.obrien@aero.org

This service provides two basic functions:
1. fast calculation of magnetic ephemeris / coordinates
2. coordinate transforms for various widely-used 3-D systems

The intent is that the service runs as a python/connexion OpenAPI-defined service. The
file python/fast-magephem.yaml provides the OpenAPI spec for this service.

The python code inherits from the AE9/AP9-IRENE python prototype. That code, in turn,
calls the AE9/AP9-IRENE C/C++ code via a stub library libmag_util.so and python's ctypes module.

The C code needed to build libmag_util.so is found in python/c. It expects libmag_util.so
to be found in python/lib. The DLL libmag_util.so links against the following DLLs in the 
AE9/AP9-IRENE spwx library: libcmagfield.so, libspwxcommon.so, libCoordXform.so, libadiabat.so.

The python code and C/C++ code rely on some data files stored in python/data. This includes the
IGRF coefficients for the internal magnetic field of the earth, as well as neural network data
files (Matlab save sets in HDF5 format) used for computing fast L* and hmin coordinates.

Spiral 1 of the service will provide only the coordinate transfrom function (coord_trans).
Spiral 2 will add the magnetic ephemeris function (get_magephem) for the OPQ field model.
Spiral 3 will add support for IGRF field models. (T89 cancelled. not available from IRENE libs)
Spiral 4 adds testing
STATUS: Spiral 4 is in progress.
