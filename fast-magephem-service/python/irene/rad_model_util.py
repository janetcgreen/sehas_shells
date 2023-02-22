"""
rad_model_util.py
by Paul O'Brien
Utilities for working with text files used by C++ command line utility

time format specifiers:
'MJD' = modified julian date
'YrDDDFrac' = Year, DOY
'YrDDDGmt' = Year, DOY, GMTsec
'YrMoDaGmt' = Year, Month, Day, GMTsec
'YrMoDaHrMnSc' = Year, Month, Day, hour, minute ,sec

record separator specifiers
',' or 'comma' = comma delimited
' ', or 'space', or 'tab' = space delimited (tabs used when writing)

path = model_data_path()
    returns path to the model data folder
date = convert_time(inTime,outFormat=None)
    convert input time to requested format
toMeV = toMeVfactor(energy_unit) e.g., toMeVfactor('keV') = 1e-3
x = atleast_Nd(x,N) append singleton dimensions until x.ndim == N

d = h5todict(hdf5file) load contents of an hdf5 file into a dict

yI = smart_log_interp(x,y,xI) interpolate y(x) to xI

(cl,dcl) = compute_confidence_levels(ylist,P) compute confidence levels
(q,dq) = martizjarret_percentiles(x,P) general purpose percentiles w/ error factors

These two readers share some common syntax:
(time,XYZ,data) = read_output_file(filename,delim='comma',returnHeaders=False,NumDataCols=None)
(time,XYZ,data,headers) = read_output_file(...,returnHeaders=True)
    read data from text file <filename>
(time,XYZ) = read_ephem(filename,delim='comma',returnHeaders=False)
(time,XYZ,headers) = read_ephem(...,returnHeaders=True)
    read ephemeris from text file <filename>
Common Syntax:
    time is an array (N,) of datetimes
    XYZ is an array (N,3) of positions, usually in ECI in RE (RE = 6371.2 km)
    headers is a dict of header values present in some machine-generated ephemeris files
    e.g.,:
    # System: GEO -> headers['System'] == 'GEO'
    # Units: Re -> headers['Units'] == 'Re'
    # Data Delimiter: Tab -> headers['Data Delimiter'] == 'Tab'

"""

import numpy as np
import re
import h5py
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings('ignore','divide by zero encountered in log')

def model_data_path():
    """path = model_data_path()
    returns path to the model data folder
    """
    import os
    slash = os.sep
    refpath = os.path.dirname(os.path.abspath(__file__))
    datapath = slash.join([refpath,'data'])
    return datapath

def toMeVfactor(energy_unit):
    """ toMeV = toMeVfactor(energy_unit)
        provides multiplier to convert from unit
        'eV','keV',... to MeV
        e.g., toMeVfactor('keV') = 1e-3
    """

    if energy_unit.lower() == 'ev':
        return 1e-6
    elif energy_unit.lower() == 'kev':
        return 1e-3
    elif energy_unit.lower() == 'mev':
        return 1
    elif energy_unit.lower() == 'gev':
        return 1e3
    else:
        raise Exception('Unknown energy unit "%s"' % energy_unit)


def atleast_Nd(x,N):
    """x = atleast_Nd(x,N)
    append singleton dimensions until x.ndim == N
    (note, this is different from atleast_2d and atleast_3d, which out existing dim in 2nd axis)
    """
    x = np.atleast_1d(x)
    s = list(x.shape)
    while len(s)<N:
        s.append(1)
    return x.reshape(tuple(s))

def read_output_file(infile,delim=',',timeFormat='MJD',returnHeaders=False,NumDataCols=None):
    """(time,XYZ,data) = read_output_file(filename,delim='comma',returnHeaders=False,NumDataCols=None)
    (time,XYZ,data,headers) = read_output_file(...,returnHeaders=True)
    read data from text file <filename>
    data is an (N,?) array of fluxes, fluences, doserates, doses, etc
    supply NumDataCols to specify number of data columns, excluding time and XYZ cols
    see module helps for remaining syntax """

    import os
    if not os.path.exists(infile):
        raise Exception('File %s does not exist' % (infile))

    if not os.access(infile,os.R_OK):
        raise Exception('Unable to access file %s' % (infile))

    headers= {}

    data = None
    lastHeader = None
    with open(infile) as fid:
        while True:
            before = fid.tell()
            line = fid.readline()
            line = line.strip()
            if len(line) == 0:
                break;
            if line.startswith('#'):
                while line.startswith('#'): # eat multiple leading ###
                    line = line[1:]
                if line.find(':') != -1:
                    (key,value) = line.split(':',1)
                    headers[key.strip()] = value.strip()
                elif len(line)>0:
                    lastHeader = line
            else:
                if 'Data Delimiter' in headers:
                    delim = headers['Data Delimiter']
                # convert keyword delimiters to fromfile counterparts
                if delim.lower() == 'comma':
                    sep = ','
                elif delim.lower() in ['tab','space',' ']:
                    sep = None
                else:
                    sep = delim

                fid.seek(before)
                data = np.atleast_2d(np.genfromtxt(fid,delimiter=sep)) # prefers row vector
                break


    if lastHeader is not None:
        headers['colheading'] = lastHeader
        timefields = re.match(r'datetime\(([^\)]+)\)',lastHeader)
        if timefields is not None:
            timefields = timefields.group(1).split(',')
            colheads = timefields + re.sub(r'datetime\([^\)]+\),',r'',lastHeader).split(',')
            if NumDataCols is None:
                NumDataCols = len(colheads)-len(timefields)-3
        else:
            colheads = lastHeader.split(',')
        headers['colheads'] = colheads
    ncols = data.shape[1]
    if (NumDataCols is None) and ('Time format' in headers):
        # guess from Time format header
        ntime = None
        if headers['Time format'].lower() in ['modified julian date','mjd']:
            ntime = 1
        elif headers['Time format'].lower() in ['year, day_of_year.frac','yrdddfrac']:
            ntime = 2
        elif headers['Time format'].lower() in ['year, day_of_year, gmt_seconds_of_day','yrdddgmt']:
            ntime = 3
        elif headers['Time format'].lower() in ['year, month, day, gmt_seconds_of_day','yrmodagmt']:
            ntime = 4
        elif headers['Time format'].lower() in ['year, month, day, hour, minute, seconds','yrmodahrmnsc']:
            ntime = 6

        if ntime is not None:
            NumDataCols = ncols-3-ntime

    if NumDataCols is None:
        raise Exception('Cannot determine number of time/data columns in "%s"' % (infile))

    ntime = ncols-3-NumDataCols
    time = convert_time(data[:,0:ntime]) # convert to datetime
    XYZ = data[:,ntime:(ntime+3)]
    data = data[:,(ntime+3):]

    if returnHeaders:
        return (time,XYZ,data,headers)
    else:
        return (time,XYZ,data)


def read_ephem(infile,delim=',',timeFormat='MJD',returnHeaders=False):
    """(time,XYZ) = read_ephem(filename,delim='comma',returnHeaders=False)
    (time,XYZ,headers) = read_ephem(...,returnHeaders=True)
    read ephemeris from text file <filename>
    see module helps for remaining syntax """

    result = read_output_file(infile,delim=delim,timeFormat=timeFormat,returnHeaders=returnHeaders,NumDataCols=0)
    # time, XYZ, data, [headers]
    if returnHeaders:
        return (result[0],result[1],result[3]) # time, XYZ, headers
    else:
        return (result[0],result[1]) # time, XYZ

def convert_time(inTime,outFormat=None):
    """date = convert_time(inTime,outFormat=None)
    convert input time to requested format
    supported outFormats:
        None = returns a datetime
        string - one of the time format specifiers above
    input time format is inferred from shape/value of inTime
    string inTime are allowed, assumed to be space-delimited lists of numbers
    inTime and date have one row per date, may have zero or multiple columns
       depending on format
    """

    from datetime import datetime, timedelta
    from .mod_jul_date import datetime2mjd,mjd2datetime
    outscalar = np.isscalar(inTime)

    inTime = atleast_Nd(inTime,2)
    dt = None
    if isinstance(inTime[0,0],str):
        it = []
        for t in inTime:
            a = np.array([float(x) for x in t[0].replace(',',' ').split()])
            it.append(a)
        inTime = np.array(it)
    elif isinstance(inTime[0,0],datetime):
        dt = inTime

    # convert to datetime
    if dt is None:
        dt = []
        if inTime.shape[1] == 1: # MJD
            dt = mjd2datetime(inTime)
        elif inTime.shape[1] == 2: # YrDDDFrac = Year, DOY
            dt = np.array([datetime(int(t[0]),1,1) + timedelta(days=t[1]-1) for t in inTime]) # allow for decimal doy
        elif inTime.shape[1] == 3: # YrDDDGmt = Year, DOY, GMTsec
            dt = np.array([datetime(int(t[0]),1,int(t[1])) + timedelta(seconds=t[2]) for t in inTime])
        elif inTime.shape[1] == 4: # YrMoDaGmt = Year, Month, Day, GMTsec
            dt = np.array([datetime(int(t[0]),int(t[1]),int(t[2])) + timedelta(seconds=t[3]) for t in inTime])
        elif inTime.shape[1] == 6: # YrMoDaHrMnSc= Year, Month, Day, hour, minute ,sec
            dt = np.array([datetime(int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4]),0) + timedelta(seconds=t[5]) for t in inTime])

    dt = dt.reshape((dt.size,))

    # convert to outputFormat

    if outFormat is None:
        outTime = dt
    elif outFormat.lower() == 'mjd':
        outTime = datetime2mjd(dt)
    elif outFormat.lower() == 'yrdddfrac':
        doy = dt-np.array([datetime(d.year,1,1) for d in dt])
        doy = np.array([d.total_seconds()/24/60/60+1 for d in doy])
        yr = np.array([d.year for d in dt])
        outTime = np.stack((yr,doy),axis=1)
    elif outFormat.lower() == 'yrdddgmt':
        doy = dt-np.array([datetime(d.year,1,1) for d in dt])
        doy = np.array([d.total_seconds()/24/60/60+1 for d in doy])
        gmtsec = np.remainder(doy,1.0)*24*60*60
        doy = np.floor(doy)
        yr = np.array([d.year for d in dt])
        outTime = np.stack((yr,doy,gmtsec),axis=1)
    elif outFormat.lower() == 'yrmodagmt':
        yr = np.array([d.year for d in dt])
        mo = np.array([d.month for d in dt])
        da = np.array([d.day for d in dt])
        gmtsec = np.array([d.hour*60*60 + d.minute*60 + d.second + d.microsecond/1.0e6 for d in dt])
        outTime = np.stack((yr,mo,da,gmtsec),axis=1)
    elif outFormat.lower() == 'yrmodahrmnsc':
        yr = np.array([d.year for d in dt])
        mo = np.array([d.month for d in dt])
        da = np.array([d.day for d in dt])
        hr = np.array([d.hour for d in dt])
        mn = np.array([d.minute for d in dt])
        sc = np.array([d.second + d.microsecond/1.0e6 for d in dt])
        outTime = np.stack((yr,mo,da,hr,mn,sc),axis=1)
    else:
        raise Exception('Time format "%s" not supported' % (outFormat))

    if outscalar:
        if outTime.ndim == 2:
            outTime = outTime = outTime[0,:]
        else:
            outTime = outTime[0]

    return outTime

def h5todict(hdf5file):
    """
    d = h5todict(hdf5file) load contents of an hdf5 file into a dict
    """
    def get_var(fp,parent,key):
        """
        val = get_var(fp,parent,,key)
        retrieve value from hdf5 file pointer
        fp - file pointer
        parent - string path of parent containing key (no /)
        key - string name of variable
        val - value stored at parent/key
        """
        if parent is None:
            longname = key
        else:
            longname = parent+'/'+key
            
        if isinstance(fp[longname],h5py.Dataset):
            
            if hasattr(fp[longname],'value'): # legacy h5py
                return fp[longname].value # just return the dataset
            else: # newer h5py
                if fp[longname].dtype == 'O':
                    return fp[longname][()].decode('utf-8') # decode bytes to str
                else:
                    return fp[longname][()] # just return the dataset

        else: # it's a group
            val = {} # make a dict, return that
            for key2 in fp[longname]:
                val[key2] = get_var(fp,longname,key2)
            return val
        
    with h5py.File(hdf5file,'r') as fp:
        d = {}
        for key in fp:
            d[key] = get_var(fp,None,key)
    return d        


def smart_log_interp(x,y,xI,*args,**kwargs):
    """ yI = smart_log_interp(x,y,xI,*args,**kwargs)
    interpolate y(x) to xI
    log interp positive y, linear interp y<=0
    pass args and kwargs on to scipy.interpolate.interp1d
    """        
    isscalar = np.isscalar(xI)
    if isscalar:
        xI = np.atleast_1d(xI)
    shape = xI.shape
    logout = interp1d(x,np.log(y),bounds_error=False,*args,**kwargs)(xI)
    out = np.exp(logout)
    i = ~np.isfinite(logout)
    if np.any(i):
        xI = np.array(xI)
        out[i] = interp1d(x,y,bounds_error=False,*args,**kwargs)(xI[i])
    i = ~np.isfinite(out)
    if np.any(i):
        out[i] = 0.0
    out.shape = shape
    if isscalar:
        out = out[0]
    
    return out


def maritzjarret_percentiles(x,P,axis=0,dln=True,nodq=False):
    """
    (q,dq) = martizjarret_percentiles(x,P,axis=0,dln=True,nodq=False)
    return percentiles (q) and standard errors (dq)
    x - data numpy array
    P - list of percentiles or scalar
    axis - along which dimension in x to compute percentiles
    dln - if true dq is error on ln(q)
    nodq - if true output is q rather than (q,dq)
    q and dq have same size as x except, axis is either removed for scalar P or len(P)
    
    based on:
    http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/quantse.htm
    Introduction to Robust Estimation and Hypothesis Testing", Rand Wilcox, Academic Press, 1997.
    Maritz-Jarret algorithm
    Maritz, J.S. and Jarrett, R.G., "A note on estimating the variance of the
    sample median", Journal of the American Statistical Association v73 n361
    (Mar 1978) pp194-196.
    """
    
    from scipy.special import betainc
    
    scalarP = np.isscalar(P)
    if scalarP:
        P = [P]
    p = np.array(P)/100 # convert to [0,1]
    
    # permute to get axis=0
    x = np.moveaxis(x,axis,0)
    if x.ndim>1:
        szq = (len(p),*x.shape[1:]) # size of permuted x,q (N-D, analysis axis first)
        x = x.reshape((x.shape[0],int(np.prod(szq[1:]))),order='F') # make x 2-D, with analysis axis first
        szq2 = (len(p),x.shape[1])
    else:
        szq = (len(p),) # output size
        szq2 = (len(p),1) # working size
        x = x.reshape((len(x),1))
    
    # allocate output
    q = np.full(szq2,np.nan)
    dq = np.full(szq2,np.nan)
    
    # now do each column separately
    for icol in np.arange(x.shape[1]):
        s = x[:,icol]
        s = np.sort(s[np.isfinite(s)])
        n = len(s)
        if n < 2:
            continue # can't really do thi swith only 2 finite points
        
        u = (1+np.arange(n))/(n+1)
        q[:,icol] = np.interp(p,u,s) # linear, left/right endpoint limiting built in
        
        if nodq:
            continue # don't compute dq
        
        if dln:
            if np.any(s<=0):
                dln_method = 0 # special case, can't take log
            else:
                s = np.log(s)
                dln_method = 1
                
        m = np.maximum(1.0,np.round(n*p))
        for i in range(len(p)):
            w = np.diff(betainc(m[i]-1,n-m[i],np.arange(n+1)/n))
            C1 = np.sum(w*s)
            C2 = np.sum(w*s**2)
            dq[i,icol] = np.sqrt(C2-C1**2)
            if dln and (dln_method == 0): # special case 0s
                f = dq[:,icol]>0
                if np.any(f): # leave dq=0 alone
                    dq[f,icol] = dq[f,icol]/np.abs(q[f,icol])
        
    
    # reshape q,dq
    q = q.reshape(szq,order='F')
    dq = dq.reshape(szq,order='F')
    
    if scalarP: # remove P axis
        if q.ndim==1: # squeeze does not remove single dimension
            q = np.asscalar(q)
            dq = np.asscalar(dq)
        else:
            q = np.squeeze(q,axis=0)
            dq = np.squeeze(dq,axis=0)
    else: # put P axis back in its place
        q = np.moveaxis(q,0,axis)
        dq = np.moveaxis(dq,0,axis)
    
    if nodq:
        return q
    else:
        return (q,dq)

def compute_confidence_levels(y,P,dln=True,nodcl=False):
    """
    (cl,dcl) = compute_confidence_levels(y,P,dln=True,nodcl=False) compute confidence levels
    y - list of numpy arrays, all of same size OR np array of at least two dimensions
        confidence levels will be computed over list or over 1st dimension
    P - scalar or list of percentiles to calculate
    cl - np array of confidence levels, P along first dimension or 1st dimension removed if P is scalar
    dcl - np array of standard errors on cl, same shape as cl
    dln - True to have dcl be standard error on ln(cl)
    nodcl - True to skip calculation of dcl
    """
    # just a wrapper for maritzjarret_percentiles
    return  maritzjarret_percentiles(np.array(y),P,axis=0,dln=dln,nodq=nodcl)
