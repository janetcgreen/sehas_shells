"""
mat2struct.py
by Paul O'Brien
Reads .mat HDF5 file into structure

struct = mat2struct(filename)
b = squeeze_array(a)
"""

import warnings
import numpy as np

def mat2struct(filename,root_uep='/',squeeze=True):
    """
    struct = mat2struct(filename,root_uep='/',squeeze=True)
    Read .mat file into Structure preserving hierarchy
    numeric Matlab types are converted to numpy ndarray objects

    use root_uep to load only variable named var via '/var'
      or its field var.field via '/var/field'
      root_uep allows arbitrary depth of /var/field/...

    use squeeze=False to retain leading/trailing singleton dimensions
      otherwise they'll be removed. Note this causes cell array {a} to just be a.

    """
    import h5py
    # filter HDF5 warnings about unrecognized attribute type in MATLAB_fileds
    warnings.filterwarnings("ignore",".*Unsupported type for attribute.*MATLAB_fields.*Offending HDF5 class.*9.*")
    warnings.filterwarnings("ignore",".*UnImplemented.*")

    struct = None
    with h5py.File(filename,'r') as db:
        top = db[root_uep]
        struct = _convert2struct(db,top,squeeze)

    return struct

def _convert2struct(db,top,squeeze=True):
    """
    struct = _convert2struct(db,top,squeeze=True)
    convert PyTables database into structure
    starting from top
    if squeeze == True, trailing singletons
    are removed via squeeze_array
    """
    from .structure import Structure

    out = None

    try:
        top_mclass = top.attrs['MATLAB_class'].decode('utf-8')
    except:
        top_mclass = 'struct' # '/' gets type struct

    if top_mclass == 'struct':
        out = Structure()
        for var in top: # convert each member of group/struct
            if str(var)[0] != '#': # skip "hidden" groups/vars that have illegal Matlab names starting with '#'
                out[var] = _convert2struct(db,top[var]) # recursively convert
    elif top_mclass in ['double', 'single', 'int8', 'int16', 'int32', 'int64', 
                        'uint8', 'uint16', 'uint32', 'uint64','logical','cell']:
        # these classes need transpose and maybe squeeze
        # cell and logical are special cases
        no_scalar = False # normally, allow scalars after squeeze
        if top_mclass == 'cell': # need to de-reference
            # use an ndarray of dtype generic "object"
            # This retains [i,j,k] indexing vs a list which would be [i][j][k]
            # will retype if possible after evaluating entries
            out = np.ndarray(top.shape,dtype='object')
            classes = set() # keeps track of unique classes of member objects
            for (i,v) in np.ndenumerate(top):
                out[i] = _convert2struct(db,db[v]) # convert element, v is an obj ref
                classes.add(out[i].__class__) # track resulting class
            if len(classes) == 1: # if only one type try to unify
                out = out.astype(dtype = classes.pop()) # try to recast the array
                # note: only numpy classes will recast. Others remain dtype='object'
            no_scalar = True # cell arrays should never collapse to scalar
        elif top_mclass == 'logical':  # convert unit8 to native bool
            out = np.array(top,dtype=bool)
        else:  # retain type chosen by h5py, just typecast
            out = np.array(top)
        out = out.T # reverse order of dimensions (works for >2 dims, too)
        if squeeze:  # remove singleton dimensions (everything in matlab is at least 2-D)
            out = squeeze_array(out,no_scalar)
    elif top_mclass == 'char':
        out = str(''.join([chr(i) for i in np.array(top[:]).flatten()])) # convert ASCII/unicode to string
    elif top_mclass == 'canonical empty':
        out = np.float64([]) # literally Matlab's []
    else: # it's a leaf of some unknown kind
        warnings.warn("Don't know what to do with %s MATLAB_class = %s\n%s" % (top.name,top_mclass,repr(top)))
        out = None

    return out

def squeeze_array(a,no_scalar=False):
    """ b = squeeze_array(a,no_scalar=False)
        removes trailing singleton dimensions of a
        converts singletons to scalar (i.e., not an np array at all)
        no_scalar prevents collapse to scalar rather than (1,) array
    """
    b = np.atleast_1d(a).copy()
    sz = list(b.shape)
    while sz and (sz[0] == 1):
        sz = sz[1:]
    while sz and (sz[-1] == 1):
        sz.pop()
    if (len(sz)==0) and no_scalar:
        sz = (1,) # retain as singleton (1,) array
    if sz:
        b.shape = tuple(sz)
    else:
        b = a.item() # singleton to scalar
    return b

