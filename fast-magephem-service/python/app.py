"""
HTTP(S) server application providing Fast Magnetic Ephemeris functions

TODO:
    update to python 3.7+ to get fromisoformat
        replace strptime call
        modify OpenAPI YAML file to note time format S(.mmmuuu) instead of S.mmm(uuu)
    add magephem service for OPQ
    add magephem service for IGRF, T89
"""

import sys
import connexion
import numpy as np
import datetime as dt
from irene.mag_field import coordinate_transform
from irene.coord_manager import CoordManager,ANGLE_DEP_COORDS

BAD_FLOAT = -1e30

def coord_trans(body):
    """
    inputs: fromSys,toSys,dates,xIN
    outputs: xOUT
    """
    fromSys = body['fromSys']
    toSys = body['toSys']
    dates = body['dates']
    xIN = body['xIN']
    X = np.array(xIN)
    dates = np.array([dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ") for s in dates])
    #Y = coordinate_transform(X,transform,dates=None) transform X to Y
    Y = coordinate_transform(X,[fromSys,toSys],dates)
    Y[np.isnan(Y)] = BAD_FLOAT
    return {'dates':body['dates'],'xOUT':Y.tolist()}

def magephem(body):
    """
    inputs: dates,X,sys,alpha,beta,dirs,kext
    outputs: 
    """
    dates = body['dates']
    dates = np.array([dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ") for s in dates])
    X = np.array(body['X'])
    sysaxes = body['sys']
    alphaScalar = False
    if 'alpha' in body:
        alphaScalar = np.isscalar(body['alpha'])
        alpha = np.array(body['alpha'])
    else:
        alpha = None
    if 'dirs' in body:
        dirs = np.array(body['dirs'])
    else:
        dirs = None
    if 'kext' in body:
        kext = body['kext']
    else:
        kext = 'opq'
    if (alpha is not None) and (dirs is not None):
        return 'Not Found', 404, {'x-error': 'not found'}        
        raise Exception("Cannot specify both alpha and dirs inputs")
        
    if (alpha is None) and (dirs is None):
        alphaScalar = True
        alpha = 90.0
    if alpha is not None:
        alpha = np.atleast_1d(alpha)
    cm = CoordManager(dates,X[:,0],X[:,1],X[:,2],sysaxes,alpha=alpha,dirs=dirs,kext=kext)
    out = {'dates':body['dates']}
    for coord in body['outputs']:
        c = cm.get(coord)
        # strip directions dimension if alpha was passed in as a scalar
        if alphaScalar and (coord in ANGLE_DEP_COORDS):
            c = c.ravel()
        c[np.isnan(c)] = BAD_FLOAT
        out[coord] = c.tolist()
    return out

# set up the flask app
app = connexion.FlaskApp(__name__.split('.')[0])
# resolve endpoint names relative to this module
resolver=connexion.resolver.RelativeResolver(sys.modules[__name__])
# Read the openapi yaml file to configure the endpoints
app.add_api('fast-magephem.yaml',resolver=resolver)

# create a URL route in our application for "/"
@app.route("/")
def home():
    """
    This function just responds to the browser URL
    localhost:$port/
    """
    return 'Fast Magnetic Ephemeris service (fast-magephem)<p>See: <a href=/api/ui>API</a>'

if __name__ == "__main__":

    app.run(port=23761)
