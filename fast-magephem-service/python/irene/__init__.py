"""
IRENE module supplies several sub-modules
by Paul O'Brien

This version is truncated to remove modules that are not needed for
the fast-magephem app.

accumulator - accumulator class for time accumulations with IRENE

ae9ap9_kernel - kernel class for transforming IRENE flux to engineering quantities

broadcast_ndarray - a broadcast utility for numpy ndarrays

cmagfield - ctypes interface to AE9/AP9 Space Weather library's magnetic field routines

coord_manager - magnetic coordinate manager (reduces calls to magnetic field routines)

coordinate - defines coordinate classes used by IRENE

fast_invariants - provides routines for computing adiabatic invariants and magnetic coordinates quickly

flyin - provides the fly-in functional interface for IRENE

grid - provide sthe IRENE grid classes

lin_bas_fun - provides linear basis functions used for grid interpolation

mag_field - provides magnetic field functions

marginal_dist - provides marginal distribution classes for IRENE

mat2struct - convert .mat (Matlab HDF5 files) to Structure object (see structure below)

mod_jul_date - functions for converting between Modified Julian Date and Python datetime objects

module_catalog - ModuleCatalog class used by IRENE

nnlib -- neural network class

orbit_prop - orbit propagator

posinfo - position information class used by IRENE

rad_model - radiation model class used by IRENE

rad_model_util - utility functions

rand_num_gen - random number generator class used by IRENE

run_mode - run mode classes used by IRENE

scene - scene class used by IRENE

stitcher - Stitch* classes used by IRENE

structure - structure class used by IRENE

transform - coordinate transform classes, as in log Energy or sqrt(K), used by IRENE

"""

#from . import accumulator
#from . import ae9ap9_kernel
from . import broadcast_ndarray
from . import cmagfield
from . import coord_manager
#from . import coordinate
from . import fast_invariants
#from . import flyin
#from . import grid
#from . import lin_bas_fun
from . import mag_field
#from . import marginal_dist
from . import mat2struct
from . import mod_jul_date
#from . import module_catalog
from . import nnlib
#from . import orbit_prop
#from . import posinfo
#from . import rad_model
from . import rad_model_util
#from . import rand_num_gen
#from . import run_mode
#from . import scene
#from . import stitcher
from . import structure
#from . import transform

