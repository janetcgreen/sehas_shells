"""
mod_jul_date.py
Paul O'Brien
routines for converting between modified julian date and python datetime object

dt = mjd2datetime(mjd)
mjd = datetime2mjd(dt) (has no effect if dt is already a float)

Tries to be smart about preserving scalar/array characteristics
"""

import datetime
import numpy as np

def mjd2datetime(mjd):
    """ dt = mjd2datetime(mjd) mjd is a float, long, or int or numpy array, dt is a datetime object."""
    if isinstance(mjd,datetime.datetime):
        return mjd # already a datetime
    elif isinstance(mjd,(int,float)):
        return datetime.datetime(1950,1,1) + datetime.timedelta(days = float(mjd)-33282.0)
    elif np.isscalar(mjd):
        raise Exception('Unexpected type %s for mjd' % type(mjd))
    else:
        vfunc = np.vectorize(mjd2datetime)
        return vfunc(mjd)


def datetime2mjd(dt):
    """ mjd = datetime2mjd(dt) dt is a datetime, mjd is a numpy array of floats """
    if isinstance(dt,datetime.datetime):
        # return 33282.0 + np.array((dt - datetime.datetime(1950,1,1)).total_seconds(),dtype=np.float)/24.0/60.0/60.0
        return 33282.0 + np.array(dt.toordinal()-datetime.datetime(1950,1,1).toordinal()) + dt.hour/24.0 + dt.minute/24.0/60.0 + dt.second/24.0/60.0/60.0+ dt.microsecond/24.0/60.0/60.0/1.0e6
    elif isinstance(dt,(int,float)):
        return float(dt) # already an mjd
    elif np.isscalar(dt):
        raise Exception('Unexpected type %s for dt' % type(dt))
    else:
        vfunc = np.vectorize(datetime2mjd)
        return vfunc(dt)

