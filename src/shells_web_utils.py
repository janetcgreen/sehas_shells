import datetime as dt
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import stats
from scipy.signal import savgol_filter
import configparser
import logging
import math
import os
import netCDF4 as nc4
import sys
import configparser
import io
import boto3
import glob
import json
import csv
import pandas as pd
from json import encoder
#encoder.FLOAT_REPR = lambda o: format(o, '.5f')
#from sklearn.externals.joblib import load
from joblib import load
import keras
from spacepy import coordinates as coord
from spacepy import time
from spacepy.irbempy import get_Lstar
from scipy.stats import gengamma
import sqlite3 as sl
#sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
import poes_utils as pu
from scipy.interpolate import NearestNDInterpolator
#import data_utils as du
#import time as ti

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)

# To override the default severity of logging
logger.setLevel('INFO')

#def my_handler(type, value, tb):
#    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
#sys.excepthook = my_handler


def read_config(configfile,section):
    '''
    PURPOSE: To read a shells config file and return
    # a dict with values and the section name
    # S3 has the info for connecting to an S3 bucket and creating files
    # Dbase has the info for connecting to a dbase
    # The assumed structure of the config file is
    # [Section name]
    # dbuser = username
    # dbpass = password
    # The config file will need an input_type and an output_type
    # input_type can be hapi,S3,dbase
    # output_type can be csv,nc,json,dbase

    :param configfile (str): location and name of the configfile
    :return: cdict, config.sections()[0]
    '''
    try:
        config = configparser.ConfigParser() # Create config object
        config.read(configfile)
        cdict = dict(config.items(section))
        # Check that the configfile has the right info
        # Raises exception and writes in logfile 'Config file must have '+ key if not
        if section =='S3':
            # This is for the S3 bucket
            keys = ['service_name', 'region_name', 'aws_access_key_id', 'aws_secret_access_key']
            checkKey(cdict, keys)
        if ((cdict['input_type']=='dbase') | (cdict['output_type']=='dbase')):
            # If you want to read or write data to a dbase then you need
            # to have user password info etc
            keys = ['dbuser','dbpass','dbhost','dbase','tblname']
            checkKey(cdict, keys)
        if cdict['input_type']=='sqlite':
            # For sqlite you don't need passwords etc
            keys = [ 'dbase', 'tblname']
            checkKey(cdict, keys)         

    except Exception as e:
        logger.exception(e)
        raise Exception("Trouble reading config info:" + str(e.args))
        #return None
    #logger.info('Read config file')
    return cdict, section

def checkKey(testdict, keys):
    '''
    ---------------------------------------------------------------
    PURPOSE: to check the config file to make sure it has the need keys
            given in testdict
    :param dict: The dict to check
    :param keys: a list of keys to look for
    :return:
    ---------------------------------------------------------------
    '''
    if not isinstance(keys,list):
        keys = list(keys)

    for key in keys:
        if key not in testdict.keys():
            msg = 'Config file must have '+ key
            print(msg)
            raise Exception(msg)
    return

def create_conn_sqlite(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sl.connect(db_file)
    except Exception as e:
        logger.error(e)

    return conn

def create_mysql_conn(cdict):
    try:
        conn = mysql.connector.connect(user=cdict['dbuser'],
                                  password=cdict['dbpass'],
                                  host=cdict['dbhost'],
                                  )
        logger.info("Connection to database successful.")


    except Exception as e:
        logger.error(e)

    return conn

class poes_sat_sem2:
    # PURPOSE to hold the info for all the poes sem2
    # satellites that will be needed
    def __init__(self, shortname):
        self.shortname = shortname.lower()
    all_poes_sats=['n15','n16','n17','n18','n19','m01','m02','m03']
    def sdate(self):
        sdates = {
              'n15': dt.datetime(1998,7,1),'n16':dt.datetime(2001,1,10),'n17':dt.datetime(2002,7,12),
              'n18':dt.datetime(2005,6,7),'n19':dt.datetime(2009,2,23),'m01':dt.datetime(2012,10,3),
              'm02':dt.datetime(2006,12,3),'m03':dt.datetime(2019,1,1)}
        return sdates[self.shortname]
    
    def edate(self):
        edates = {
              'n15': None, 'n16': dt.datetime(2014, 7, 11), 'n17': dt.datetime(2013, 4, 10),
              'n18': None, 'n19': None, 'm01': None,
              'm02': dt.datetime(2021, 11, 16), 'm03': None}
        return edates[self.shortname]
    def longname(self):
        longnames={
        'n15':'noaa15','n16':'noaa16','n17':'noaa17','n18':'noaa18',
            'n19':'noaa19','m01':'metop01','m02':'metop02','m03':'metop03'
        }
        return longames[self.shortname]
    def satid(self):
        satids ={'n15':1,'n16':2,'n17':3,'n18':4,
            'n19':5,'m01':6,'m02':7,'m03':8}
        return satids[self.shortname]


def satname_to_id(satname):
    all_poes_sats=['n15','n16','n17','n18','n19','m01','m02','m03']
    satId= all_poes_sats.index(satname)+1
    return satId

def poeschan_to_id(channel):
    all_poes_channels=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']
    chId= all_poes_channels.index(channel)+1
    return chId


def get_HAPI_data(url,id,sdate,edate,params=None,format='json'):
    '''
    PRUPOSE: To get data from a HAPI server
    :param url:
    :param id (str): name of data id
    :param sdate (datetime):
    :param edate (datetime):
    :param params(list(str)): list of parameters to get
    :param format(str): should be json or text
    :return:
    '''
    datastr = 'data?id='+id
    tminstr = '&time.min='+sdate.strftime('%Y-%m-%dT%H:%M:%S')+'.0Z'
    tmaxstr = '&time.max='+edate.strftime('%Y-%m-%dT%H:%M:%S')+'.0Z'
    Hquery = url+datastr+tminstr+tmaxstr
    if params is not None:
        pstr = '&parameters='+','.join(params)
        Hquery = Hquery+pstr
    if format =='json':
        Hquery = Hquery+'&format=json'
    
    return Hquery

    


#---------------------------------------------------------------------------------------------------------------------
#               SHELLS CODE
#=====================================================================================================================

def make_Lbin_data(data, Lbins=np.arange(1,8.5,.25), Lcol = 'L_IGRF', vars = None):
    '''
    PURPOSE: To take the POES data from poe_utils_V2.get_data_dict and reform it into a time series of
    .25 Lbin data for each pass. The binned data will have 1 record per pass
    :param: data (dict) - a dictionary of poes data returned from poes_utils_V2.get_data_dict
            These values are required. 'time','L*****','lat','lon','MLT'
    :param: Lbins (list) - a list of the Lbin edge values for the binning. The binned data will have len 1 less than Lbins
            default (np.arange(1,8.5,.25))
    :param L Lcol (str) - The L column to use for binning
    :param: vars (list(str)) - a list of the variables to bin. If none are passed then all non-location variables in
            data.keys() will be used
    :return:
    '''
    # These values are required for the output files
    # Todo Check that the required vars are there
    loc_vars = ['time',Lcol,'lat','lon','MLT']

    findat = dict()     # This will hold the returned data

    if vars is not None:

        allvars = loc_vars + list(vars)

        # Get only the subset of location variables and requested vars
        data = {k: data[k] for k in allvars}

    passes, breaks = pu.getLpass(data[Lcol][:])

    # Create the NS direction that we need for SARS
    #---------------------------------------------
    NS = np.zeros((len(data['lat'])), dtype=float)
    NS[0:-1] = np.diff(data['lat'])
    # Repeat the last val
    NS[-1] = NS[-2]

    NS[np.where(NS >= 0)] = 0  # Northbound
    NS[np.where(NS < 0)] = 1  # Southbound

    # Add to data
    data['NS'] = NS

    # Bin the data by L and pass
    #--------------------------------------------

    # The e4 channels is sometimes negative because it is
    # flagged whenever the proton flux is high
    # Need to mask that before binning and pass a masked array to
    # make_Lbin_data
    # To do that, I change the dict to a numpy array
    # and then create a mask for the data channels
    #vtemp = list(data.keys())
    #for k in range(0, len(data.keys())):
    #    if k == 0:
    #        npdata = [np.ma.array(data[vtemp[k]])]
    #    else:
    #        npdata = np.append(npdata, [np.ma.array(data[vtemp[k]])], axis=0)

    #tmask = np.zeros_like(npdata)

    for ch in vars:
        data[ch][data[ch]<0]=np.nan
    # e4 often has negative values in the SAA because the proton flux is high
    # so the data is not a good electron channel anymore
    #    chindex = vtemp.index(ch)
    #    tmask[ chindex, np.where(npdata[chindex,:] < 0)] = 1
        
    #mdata = np.ma.MaskedArray(npdata, tmask)
    
    # Todo: What to do if the last bit of data is not complete?
    # In reprocessing mode it will be overwritten but need to make sure the times are the same
    # In rt mode make sure that sdate starts at the time before the last time

    # JGREEN: 09/2021 Changed this so that pbins include 1 more than the last pass
    # because bindata needs the edge bins. i.e. if the last pass is 56 then
    # pbins should be 0 to 57.
    pbins = np.arange(0, (passes[-1] + 2))

    bindat,bvars = pu.bindata(data, passes, data['L_IGRF'][:], pbins, Lbins)
    
    # If you call bindata with anything that is not a dict, then bvars is not there
    if len(bvars)<1:
        bvars = list(data.keys())
        
    # Some NS vals end up in the middle
    # Sometimes the NS direction does not match with the Lpasses
    # because the L shell may be increasing even as lat has switched
    nind = bvars.index('NS')
    temp = bindat.statistic[nind,:,:]
    temp[temp<.5] = 0
    temp[temp >= .5] = 1

    # This gives the NS value for most of the pass
    test = np.nanmedian(temp,axis=1)
    for nsco in range(0,np.shape(temp)[0]):
        temp[nsco,:]=test[nsco]
    bindat.statistic[nind,:,:] = temp


    # Bin MLT and lon and replace
    #-------------------------------------------
    for col in ['lon', 'MLT']:
        if col == 'lon':
            fac = np.pi / 180.0
        else:
            fac = np.pi / 12.0
        xval = [math.sin(co * fac) for co in data[col][:]]
        yval = [math.cos(co * fac) for co in data[col][:]]
        x = stats.binned_statistic_2d(passes[:], data['L_IGRF'][:], xval, statistic=np.ma.mean,
                                      bins=[pbins, Lbins])
        y = stats.binned_statistic_2d(passes[:], data['L_IGRF'][:], yval, statistic=np.ma.mean,
                                      bins=[pbins, Lbins])
        temp = (1.0 / fac) * np.arctan2(x.statistic, y.statistic)
        temp[temp < 0] = temp[temp < 0] + 2 * np.pi * (1 / fac)  # This is [Lbins,times]
        # replace the value in bindat
        inds = bvars.index(col)
        bindat.statistic[inds,:,:] = temp

    # Define a single time for each pass
    #-------------------------------------
    # I was doing nanmedian but changed it to nanmin
    # Not sure why I changed this from nanmed to nan min
    # I'll try changing it back
    #time_med = np.nanmin(bindat.statistic[0, :, :], axis=1)
    time_med = np.nanmedian(bindat.statistic[0, :, :], axis=1)

    # Make sure the last pass has some valid data
    # It could be all nans if the last pass is only partial and all greater than L=8
    # Or there could be some weird times because the L passes get screwed up
    tinds = np.where(~np.isnan(time_med))[0]

    # Make the return dict with the good tinds
    for vind in range(0,len(bvars)):
        findat[bvars[vind]] = bindat.statistic[vind,tinds,:]

    # JGREEN: 2/2023 Added this to fill in gaps where the e4 flux is negative
    # Try doing a 2-D interpolation
    for ch in vars:
        pass

    findat['time_pass'] = time_med[tinds]

    #print('Here')
    return findat

def write_dict_to_monthly_nc(data, time_col, dims, unlim_dim=None, vars = None, outdir=os.getcwd()+'/temp/', outfile = 'temp_' ):
    '''
    PURPOSE: To write data to a monthly netcdf file. This assumes the data is time series data
    and requires a time column to define the filename. The files are named as outfile_YYYYMM.nc
    where outfile is either temp_ or the passed variable. The files are put in yearly directories
    under outdir. If the file and year directory does not already exist then it is created. If it
     does already exist then the data is added or overwritten.

    :param data (dict): A dictionary of data
    :param time_col(str): The key for the time_col (needed to know what monthy file to put data in )
    :param dims list(str): A list of the variables to use as dimensions i.e. ['time_pass','Lbin']
    :param unlim_dim (str): The variable to use as the unlimited dimension- most likely the time col
    :param outdir (str): Directory to put files
    :param outfile (str): Start of filenames YYYYMM.nc will be added to the end
    :return:
    '''

    # Make sure dims is a list to iterate
    if not isinstance(dims,list):
        dims = [dims]

    # Check if time column is date time, ctime (sec), or ctime (msec)
    if isinstance(data[time_col][0], dt.datetime):
        # It's a datetime
        syear = data[time_col][0].year
        eyear = data[time_col][-1].year
        tdata = data[time_col]
    else:
        # Check if it is msec (like POES) or sec. If there are more than 10 digits it is msec
        # That should work from 1971 -3000. Before 1971 we have no data and in 3000 this is so obsolete.
        # Todo raise an error if time is before 1971 or after 3000
        ttime = str(int(data[time_col][0]))
        if len(ttime)>10:
            # The time is in msec
            syear = (pu.unix_time_ms_to_datetime(data[time_col][0])).year
            eyear = (pu.unix_time_ms_to_datetime(data[time_col][-1])).year
            tdata = pu.unix_time_ms_to_datetime(data[time_col][:])
        else:
            # The time is in sec
            syear = pu.unixtime(data[time_col][0]).year
            eyear = pu.unixtime(data[time_col][-1]).year
            tdata = pu.unixtime(data[time_col][:])

    while syear<=eyear:
        # Get indices of data for the year
        sinds = np.where((tdata>dt.datetime(syear,1,1))& (tdata<dt.datetime(syear,12,31,23,59,59)))[0]
        smo = tdata[sinds[0]].month
        emo = tdata[sinds[-1]].month
        while smo<=emo:
            # Get indices of data for the month and year
            minds = np.where((tdata>=dt.datetime(syear,smo,1))& (tdata<dt.datetime(syear,smo+1,1,0,0,0)) )[0]
            stryear = str(syear)
            strmo = str(smo).zfill(2)

            # Check if directory exists and create it if not
            direc = outdir+'/'+stryear+'/'
            if not os.path.isdir(direc):
                os.makedirs(direc)

            ofile = direc+outfile+stryear+strmo+'.nc'
            if os.path.isfile(ofile):
                ftype = 'r+'
                #odata = nc4.Dataset(ofile,'r+')
            else:
                ftype ='w'
                #odata = nc4.Dataset(ofile, 'w')

            with nc4.Dataset(ofile,ftype) as odata:
                if ftype == 'w':
                    # Need to create  dims
                    for dim in dims:
                        if unlim_dim is not None:
                            if dim==unlim_dim:
                                odata.createDimension(dim,None)
                            else:
                                odata.createDimension(dim, len(data[dim][:]))
                        else:
                            odata.createDimension(dim, None)

                    dimsizes = [len(data[dim]) for dim in dims]
                    # Create Variables
                    if vars is None:
                        vars = data.keys()
                    # Get the dimension
                    for var in vars:
                        # Find the dimensions that match
                        vardat = data[var]
                        vardims = np.shape(vardat)

                        try:
                            dimnames = [dims[dimsizes.index(x)] for x in vardims]
                        except:
                            msg = 'No matching dimensions for '+var
                            print(msg)
                            raise Exception(msg)

                        odata.createVariable(var, np.float64, dimnames)
                        odata.variables[var] = data[var]
                elif ftype =='r+':
                    print('Need to add to existing file')
                    
def map_poes(data, evars, sat, satref, ref_ind, cdf_dir, year, ref_syear, ref_eyear):
    '''
    PURPOSE: To take the pass and L binned data from POES and map it to 
    a consistent longitude for the shells neural network to read
    :param data (dict) Data binned by pass and L
    :param evars (list) the electron flux channels to process
    :param sat (str) the satellite to process
    :param satref (str) the ref satellite to use m02
    :param refind (int) the index of the reference longitude
    :param cdf_dir (str) the directory to look for the cdf files
    :param ref_syear the start year for the ref sat of the cdf file to use
    :param ref_eyear the end year for the ref sat of the cdf files to use

    # Note this is expecting Kp*10
    
    '''
    
    #----------------------------- Set some data for later --------------------------
    
    Lbins = data['Lbins']
    # These are the cdf variables to get
    svars = list()
    for var in evars:
    # These have the flux for each percentile
        svars.append(var+'_sar')
    
    # This will hold the final output data
    map_data = {}
    map_data['time'] = data['time_pass']
    
    
    eyear_all = year
    syear_all = eyear_all-4

    #JGREEN: 02/2023 I took these calculations out of the
    # loop through the energy channels because they will be
    # the independent of that

    # Get the hemisphere
    # JGREEN: data['lat'] was being overwritten for every channel
    # so that hemi was just 0. this didn't happen in previous
    # versions because data['lat'] was a netcdf variable
    hemi = np.copy(data['lat'][:])
    hemi[hemi >= 0] = 0  # Northern hemi
    hemi[hemi < 0] = 1  # Southern hemi
    hemi1 = hemi.astype(int)

    # Get the NS direction
    NSco = data['NS'][:]
    NSco1 = NSco.astype(int)

    # Get the lon bin
    lon = np.floor(data['lon'][:] / 10)
    lon1 = lon.astype(int)
    lon1[lon1 > 35] = 0  # If its exactly 360 deg it gets put in bin 0

    Kp = np.floor(data['Kp*10'][:] / 10)
    Kp1 = Kp.astype(int)
    # Need to make Kp into a vector
    Kpvec = np.tile(Kp1, (len(Lbins), 1)).T

    # Need to make an array of Ls
    Ls = np.zeros((len(data['time_pass'][:]), len(Lbins)), dtype=int)

    for lco in range(0, len(Lbins)):
        Ls[:, lco] = Ls[:, lco] + lco

    # The mapping tables are stored separately for each variable and sat
    # So step through each variable to do mapping
    for eco in range(0, len(evars)):
        # ------------------ Get the cdf files for mapping ------------------------------
        # Get cdfs for current sat and variable
        satsarfile = os.path.join(cdf_dir, sat,'poes_cdf_' + sat + '_' + str(syear_all).zfill(4) \
                    + '_' + str(eyear_all).zfill(4) + evars[eco] + '.nc')

        # Ideally, the process would use the cdf from the current year back 5
        # But in real time it will always be one year behind
        # So if you don't find a file for the current year then look back one year
        if ~os.path.exists(satsarfile):
            newyear = eyear_all
            while (~os.path.exists(satsarfile)) & (newyear>=eyear_all-2):
                satsarfile = os.path.join(cdf_dir, sat, 'poes_cdf_' + sat + '_' + str(newyear-4).zfill(4) + '_' \
                                          + str(newyear).zfill(4) + evars[eco] + '.nc')
                newyear=newyear-1
        # But If you are reprocessing, then it may have to use cdfs from the future when the data
        # starts, so try looking ahead if there is still no file
        if ~os.path.exists(satsarfile):
            newyear = eyear_all
            while (~os.path.exists(satsarfile)) & (newyear<=eyear_all+5):
                satsarfile = os.path.join(cdf_dir, sat, 'poes_cdf_' + sat + '_' + str(newyear-4).zfill(4) + '_' \
                                          + str(newyear).zfill(4) + evars[eco] + '.nc')
                newyear = newyear + 1

        sar_sat = nc4.Dataset(satsarfile, 'r')
        print(satsarfile)
        
        # Get cdfs for reference sat m02
        srefile = os.path.join(cdf_dir, satref,'poes_cdf_' + satref + '_' + str(ref_syear).zfill(4) + '_' \
                  + str(ref_eyear).zfill(4) + evars[eco] + '.nc')
        sar_ref = nc4.Dataset(srefile, 'r')
        print(srefile)

        # This is the percentile for each flux for the sat being processed and the current column
        # It uses evars which are the electron flux cols
        sar = sar_sat[evars[eco]][:]

        # This is the flux at each percentile for the ref sat
        # Notice this uses svars
        sarout = sar_ref[svars[eco]][:]
        
        # Get the flux bins for the data being processed
        # data is stored as data[pass,Lbin] for each variables
        # Todo Sometimes the flux is a nan and then it can't round
        fluxbin = np.round(np.log10(data[evars[eco]][:]) * 10)
        fluxbin1 = fluxbin.astype(int)

        # This is bad data
        # This happens because the sat doesn't go out to large Ls sometime
        # so you can't fill in values with statistics
        nan_inds = np.where((fluxbin1 < -10) | (hemi1 < -10) | (lon1 < -10) | (Kpvec < -10) | (NSco1 < -10))

        # Set these nan_inds to zero for now so that it is a valid index
        # but flag it later
        fluxbin1[nan_inds] = 0
        hemi1[nan_inds] = 0
        lon1[nan_inds] = 0
        NSco1[nan_inds] = 0
        Kpvec[nan_inds] = 0

        # Get the percentile that corresponds to each flux value for the current sat
        per1 = sar[hemi1, NSco1, Ls, lon1, Kpvec, fluxbin1]
        perbin1 = np.round(per1 * 100).astype(int) #

        # In northern lats some sar dat is nan's
        # Set those to bin 0 just so it will work but flag them later
        # per_nan = np.where(perbin1 < -10)[0]
        #JGREEN 6/2/2021 This was a mistake that was setting the whole pass to flags
        per_nan = np.where(perbin1 < -10)
        perbin1[per_nan] = 0

        # Get the flux at the ref satellite for the measured percentile
        # hemi is 1 for southern, direction is southbound, ref_idn is 20
        fluxval = sarout[1, 1, Ls, ref_ind, Kpvec, perbin1]
        #fluxval = np.log10(data[evars[eco]][:]) # FOR TESTING I SET THIS BACK TO THE ORIGINAL DATA
        # Flag the bad values again
        fluxval[nan_inds] =np.nan
        fluxval[per_nan] = np.nan

        # Now fill in the bad values with the nearest values
        # because the NN needs complete data
        # I do this here after mapping because if you do it
        # before mapping then it mixes data from different lons and
        # hemispheres which can be very different
        mask = np.where(~np.isnan(fluxval))
        interp = NearestNDInterpolator(np.transpose(mask), fluxval[mask])
        fluxval = interp(*np.indices(fluxval.shape))

        map_data[evars[eco]] = fluxval

    sar_ref.close()
    sar_sat.close()

    return map_data
def map_poes_fits(data, evars, sat, satref, ref_ind, cdf_dir, year, ref_syear, ref_eyear):
    '''
    PURPOSE: To take the pass and L binned data from POES and map it to
    a consistent longitude for the shells neural network to read

    This version reads cdf files that have generalized gamma fit parameters
    Instead of the whole cdf

    :param data (dict) Data binned by pass and L
    :param evars (list) the electron flux channels to process
    :param sat (str) the satellite to process
    :param satref (str) the ref satellite to use (m02)
    :param refind (int) the index of the reference longitude
    :param cdf_dir (str) the directory to look for the cdf files
    :param year (int) the year to process
    :param ref_syear (int) the start year for the ref sat of the cdf file to use 2014
    :param ref_eyear the end year for the ref sat of the cdf files to use 2018

    # Note this is expecting Kp*10
    '''

    # ----------------------------- Set some data for later --------------------------

    Lbins = data['Lbins']

    # The cdf fit data files have evars['hemi','NS','Lbin','lonbin','Kp','fits']
    # hemi:  0,1 (for norht and south)
    # NS: 0,1 (for north or south direction)
    # Lbin: 1-8
    # lonbin:
    # Kp: 0-5
    # fits: four parameters for the gen gamma a,c,loc and scale

    # This will hold the final output of mapped data
    map_data = {}
    map_data['time'] = data['time_pass']

    eyear_all = year # These are used to find the right cdf file
    syear_all = eyear_all - 4

    # JGREEN: 02/2023 I took these calculations out of the
    # loop through the energy channels because they will be
    # the independent of that

    # Get the hemisphere
    # JGREEN: data['lat'] was being overwritten for every channel
    # so that hemi was just 0. this didn't happen in previous
    # versions because data['lat'] was a netcdf variable
    hemi = np.copy(data['lat'][:])
    hemi[hemi >= 0] = 0  # Northern hemi
    hemi[hemi < 0] = 1  # Southern hemi
    hemi1 = hemi.astype(int)

    # Get the NS direction
    NSco = data['NS'][:]
    NSco1 = NSco.astype(int)

    # Get the lon bin
    lon = np.floor(data['lon'][:] / 10)
    lon1 = lon.astype(int) # This is so we can use it as an index
    lon1[lon1 > 35] = 0  # If its exactly 360 deg it gets put in bin 0

    Kp = np.floor(data['Kp*10'][:] / 10)
    Kp1 = Kp.astype(int)
    Kp1[Kp1>4] = 4 # We combine high kps for the cdf file
    # Make Kp into an array that's duplicated for each Lbin
    Kpvec = np.tile(Kp1, (len(Lbins), 1)).T

    # Make an array of Ls
    Ls = np.zeros((len(data['time_pass'][:]), len(Lbins)), dtype=int)

    for lco in range(0, len(Lbins)):
        Ls[:, lco] = Ls[:, lco] + lco

    # The mapping tables are stored separately for each sat
    # They used to be separate for each variable

    # ------------------ Get the cdf files for mapping ------------------------------
    # Get cdfs for current sat and variable
    satsarfile = os.path.join(cdf_dir, sat, 'poes_cdf_' + sat + '_' + str(syear_all).zfill(4) \
                              + '_' + str(eyear_all).zfill(4) + '.nc')
    # Ideally, the process would use the cdf from the current year back 5
    # But in real time it will always be one year behind because it takes a year
    # to collect data.
    # So if you don't find a file for the current year then look back one year
    if ~os.path.exists(satsarfile):
        newyear = eyear_all # eyear_all is the year of the data being processed
        while (~os.path.exists(satsarfile)) & (newyear >= eyear_all - 2):
            satsarfile = os.path.join(cdf_dir, sat, 'poes_cdf_' + sat + '_' + str(newyear - 4).zfill(4) + '_' \
                                      + str(newyear).zfill(4) + '.nc')
            newyear = newyear - 1
    # But If you are reprocessing, then it may have to use cdfs from the future when the data
    # starts, so try looking ahead if there is still no file
    if ~os.path.exists(satsarfile):
        newyear = eyear_all
        while (~os.path.exists(satsarfile)) & (newyear <= eyear_all + 5):
            satsarfile = os.path.join(cdf_dir, sat, 'poes_cdf_' + sat + '_' + str(newyear - 4).zfill(4) + '_' \
                                      + str(newyear).zfill(4)  + '.nc')
            newyear = newyear + 1
    #satsarfile = '/Users/janet/PycharmProjects/SHELLS/cdfits2/n15/poes_cdf_n15_2015_2020.nc'
    sar_sat = nc4.Dataset(satsarfile, 'r')
    #print(satsarfile)

    # Get cdfs fits for reference sat m02
    srefile = os.path.join(cdf_dir, satref, 'poes_cdf_' + satref + '_' + str(ref_syear).zfill(4) + '_' \
                           + str(ref_eyear).zfill(4) + '.nc')
    #srefile = '/Users/janet/PycharmProjects/SHELLS/cdfits2/n15/poes_cdf_n15_2015_2020.nc'
    sar_ref = nc4.Dataset(srefile, 'r')
    #print(srefile)

    # Step through each variable to do the mapping
    for eco in range(0, len(evars)):

        # This is the fits for the percentile for each flux for the sat being
        # processed and the current column. It uses evars which are the electron flux cols
        sar = sar_sat[evars[eco]][:]

        # This is the fits for the percentile for the ref sat
        sarout = sar_ref[evars[eco]][:]

        # Now step through each flux val
        # and get the percentile
        # then get the ppf or flux for each percentile for the ref

        logflux = np.log10(data[evars[eco]])

        # Set bad values to 0 and then flag it later
        nan_inds = np.where((logflux < -10) | (hemi1 < -10) | (lon1 < -10) | (Kpvec < -10) | (NSco1 < -10))

        logflux[nan_inds] = 0
        hemi1[nan_inds] = 0
        lon1[nan_inds] = 0
        NSco1[nan_inds] = 0
        Kpvec[nan_inds] = 0

        fluxval = 0*logflux
        for Lco in range(0,len(Lbins)):
            params1=sar[hemi1[:,Lco],NSco1[:,Lco],Lco,lon1[:,Lco],Kpvec[:,Lco]]
            for fco in range(0,len(params1[:,0])):
                if ((~np.isnan(logflux[fco,Lco])) & (~np.isinf(logflux[fco,Lco]))):
                    rv1 = gengamma(params1[fco,0], params1[fco,1], params1[fco,2], params1[fco,3])
                    per = rv1.cdf(logflux[fco,Lco]) # This is percentile for that flux
                    if per>=0:
                        params2 = sarout[1, 1, Lco, ref_ind, Kpvec[fco, Lco]]
                        rv2 = gengamma(params2[0], params2[1], params2[2], params2[3])
                        fluxval[fco,Lco] = rv2.ppf(per)
                    else:
                        fluxval[fco, Lco] = np.nan
                else:
                    fluxval[fco,Lco]= np.nan

        fluxval[nan_inds] = np.nan

        # Now fill in the bad values with the nearest values
        # because the NN needs complete data
        # I do this here after mapping because if you do it
        # before mapping then it mixes data from different lons and
        # hemispheres which can be very different
        mask = np.where(~np.isnan(fluxval))
        interp = NearestNDInterpolator(np.transpose(mask), fluxval[mask])
        fluxval = interp(*np.indices(fluxval.shape))

        map_data[evars[eco]] = fluxval

    sar_ref.close()
    sar_sat.close()

    return map_data


def qloss(y_true,y_pred):
    qs = [0.25,0.5,0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true-y_pred
    v = np.maximum(q*e, (q-1)*e)
    return keras.backend.mean(v)

def run_nn(data,evars,Kpdata,Kpmax_data,out_scale,in_scale,m, L = None, Bmirrors = None, Energies = None):
    '''
    PURPOSE: To take the values in data, apply the neural network from Alex and
    then output the near equatorial flux
    :param data (dict): Dict with data[ecol][timeXL] for each of the 4 POES energy channels
    :param evars (list): List of the energy channel names
    :param Kpdata (list): List of Kp*10 for each time
    :param Kpmax_data (list):List of Kp*10_max_3d for each time
    :param out_scale (str): Name of the output transform file for the NN
    :param in_scale (str): Name of the input transform file for the NN
    :param m (str): Name of file used by the NN
    :return:
    '''
    # List of energies we want for the output data
    if Energies is None:
        Energies = np.arange(200.0,3000.0,200.0)  # keV
    # The Bmirror values for the output at each L
    if Bmirrors is None:
        # This is for the value at the equator
        Bmirrors = [2.591e+04*(L**-2.98) for Ls in L ]
    # L values for the ouput corresponding to each bmirror
    if L is None:
        L = np.arange(3.,6.3,1)
    
    # Check for -1 where L>7.5
    # Sometimes this happens because the orbit does not go out that far
    for wco in range(0, len(evars)):
        bad_inds = np.where((data[evars[wco]][:])<0)
        if len(bad_inds[0])>0:
            for bco in range(0,len(bad_inds[0])):
                # Set the flux equal to the neighbor
                data[evars[wco]][bad_inds[0][bco]][bad_inds[1][bco]] = data[evars[wco]][bad_inds[0][bco]][bad_inds[1][bco]-1]

    # Todo need to deal with bad data
    # My data has timeXL for each energy in a dict
    # This expected input is timeXL e1, timeXL e2 timeXL e3, timeXL e4

    # This concatetnates the fluxes at each energy into one array
    new_dat = np.array(data[evars[0]][:])
    for wco in range(1,len(evars)):
        new_dat = np.append(new_dat,data[evars[wco]],axis=1)

    l, w = np.shape(new_dat)
    
    # What do we want as output if we are going to make a netcdf file
    # data[E][time x L] at Beq for that L
    # data[E1_upperq] [timeXL] upper quantile
    # data[E1_lowerq] [timexL] lower quantile

    # Create a dict for the output data
    outdat = {}
    outdat['L'] = L
    outdat['Bmirrors'] = Bmirrors
    outdat['Energies'] = Energies
    # Then create arrays for each E and E quantiles
    for E in Energies:
        col = 'E flux '+str(int(E))
        outdat[col] = np.zeros((0,len(L)),dtype=np.float)
        colh = 'E flux ' + str(int(E))+' upper q'
        outdat[colh] = np.zeros((0,len(L)),dtype=np.float)
        coll = 'E flux ' + str(int(E))+' lower q'
        outdat[coll] = np.zeros((0,len(L)),dtype=np.float)
    outdat['time'] = list()
    outdat['Kp'] = list()
    outdat['Kpmax'] = list()

    # Step through the POES passes one at a time
    for pco in range(0,l):
        # The input needs Kp, Kpmax, E, Bmirror for each L
        # Check that the poes input does not have Nans
        check_dat = np.where((np.isnan(new_dat[pco][:])) | (new_dat[pco][:]<0))[0]

        if len(check_dat)<1:
            outdat['time'].append(data['time'][pco])
            outdat['Kp'].append(Kpdata[pco]/10)
            outdat['Kpmax'].append(Kpmax_data[pco]/10)
            
            # The NN code can calculate flux for all Ls at once
            kp = np.tile(Kpdata[pco], len(L)) # Create a list of Kp for each L calc
            maxkp = np.tile(Kpmax_data[pco], len(L)) #Create a list of maxKp for each L
            # Create a list of POES data to be used for each L calc
            poes = np.tile(new_dat[pco:pco + 1], (len(L), 1))

            # Step through each energy and create outdat[Ecol] that is len L
            for eco in range(0,len(Energies)):
                # Make a list of one energy at all L's
                energy = np.tile(Energies[eco], len(L))
                # The Bmirror is different for each L
                input = np.concatenate((np.array(energy).reshape(-1, 1), np.array(Bmirrors).reshape(-1, 1),
                            np.array(L).reshape(-1, 1), np.array(kp).reshape(-1, 1),
                            np.array(maxkp).reshape(-1, 1),
                            poes), axis=1)
                # This returns the lowerq, log(flux), upperq data for one E and Bmirror(L) at each L
                #start=ti.time()
                fpre = out_scale.inverse_transform(m.predict(in_scale.transform(input)))
                #tend = ti.time()
                #print('time to do nn',tend-start)
                cols = ['E flux ' + str(int(Energies[eco]))+' upper q',
                    'E flux '+str(int(Energies[eco])),
                    'E flux ' + str(int(Energies[eco]))+' lower q',]
                for cco in range(0,len(cols)):
                    temp = outdat[cols[cco]][:]
                    outdat[cols[cco]]= np.vstack((temp,fpre[:,cco]))
    #plt.set_cmap('jet')
    #plt.pcolormesh(pu.unix_time_ms_to_datetime(outdat['time'][:]),L,np.transpose(outdat['E flux 2000'][:,:]))
    #plt.pcolormesh(pu.unix_time_ms_to_datetime(outdat['time'][:]), np.arange(0,29),np.transpose(data[evars[2]][:,:]))
    return outdat

def write_shells_netcdf_S3(outdir, outdat, Ls, Bmirrors, Energies, sat, modelname, cdata):
    '''
    PURPOSE: To write the data to netcdf files under outdir
    :param outdir: Directory to write the data with YYYY/SHELLS_YYYYMMDD.nc beneath
    :param outdat: The NN data dict (timexL) with 'E flux 200.0','E flux 100.0 upper q','E flux 100.0 lower q'
    :param Kp: Kp data for each pass
    :param Kpmax: Kp max in 3 days for each pass
    :param Ldims: The L values for the ouputs
    :param Bmirror: The bmirror values chosen for each L
    :param Energies:
    :param sat:
    :param modelname
    :return:
    '''
    
    # The data will be stored as netcdf with log flux stored as time X L for each energy
    # The data from each satellite is added separately so we have to 
    # 1) read any existing file
    # 2) Add and sort new data
    # 3) rewrite the file
    # File format will be outdir/year/shells_YYYYMMDD_V1.nc

    fstart = outdat['time'][0]
    fdate = pu.unix_time_ms_to_datetime(fstart)

    fedate = pu.unix_time_ms_to_datetime(outdat['time'][-1])
    allvars = outdat.keys()
    fluxvars = [i for i in allvars if 'flux' in i] # Get names of flux cols
    Kpvars = [i for i in allvars if 'Kp' in i]
    
    # Step through each day and write the data
    # fdate is the datetime of the first data point and fedate is the datetime of the last data point
    while fdate<fedate:
        # This is the day file associated with fdate
        file = 'shells_'+fdate.strftime('%Y%m%d')+'.nc'

        # This is the local file and path
        loc_fname = os.path.join(outdir,fdate.strftime('%Y'),file)
        
        # Check if the file exist already

        # If there is a configfile then check S3 for the file
        file_exists = 0
        if cdata is not None:
            # Check if the file exists on S3
            s3dir = '/data/SHELLS/'+ fdate.strftime('%Y')
            s3fname = os.path.join(s3dir, file)

            # If it is an S3 bucket, then you have to first connect
            s3 = boto3.resource(
                service_name=cdata['service_name'],
                region_name=cdata['region_name'],
                aws_access_key_id=cdata['aws_access_key_id'],
                aws_secret_access_key=cdata['aws_secret_access_key']
            )

            # Make an object with all files filtered on the filename
            obj_summary = s3.Bucket(cdata['bucket']).objects.filter(Prefix=s3fname)
            # Turn that object into a list
            files = [x.key for x in obj_summary]

            # If the list is not 0 then the file already exists at S3
            if len(files) > 0:
                # Then get the file
                # s3.Bucket(cdata['bucket']).objects.get(files)
                # obj = s3.Object(cdata['bucket'], files[0])
                # sdata = obj.get()
                # sdata = s3.Bucket(cdata['bucket']).objects.get(files)
                # Now read that object as a netcd file
                # This will read it directly in but then you can't change it at all
                # you can't do mode = r+
                # s3temp_data = nc4.Dataset(loc_fname, memory=io.BytesIO(sdata['Body'].read()).read(), mode='r')
                #s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')
                # download the file and read it in again
                res = s3.Bucket(cdata['bucket']).download_file(files[0], 's3temp.nc')
                sat_data = nc4.Dataset('s3temp.nc',mode='r+')
                old_data = dict()
                # Turn the old data file into a dict
                for val in sat_data.variables.keys():
                    old_data[val] = sat_data[val][:]
                file_exists=1
                # If the file already exists, and we create sat_data, what happens
                # when we close the file? what is it called????

        elif ( os.path.exists(loc_fname) ):
            # Then read in the old data file
            sat_data = nc4.Dataset(loc_fname, 'r+')
            old_data = dict()
            # Turn the old data file into a dict
            for val in sat_data.variables.keys():
                old_data[val] = sat_data[val][:] 
            file_exists = 1
           
        if file_exists==0:
            # If the file does not exist at S3 or locally then start a new file
            # Check that the directory exists and create it if it doesn't

            local_dir = os.path.join(outdir, fdate.strftime('%Y'))
            if not (os.path.exists(local_dir)):
                os.makedirs(local_dir)
                
            # Create the file locally
            if cdata is not None:
                sat_data = nc4.Dataset('s3temp.nc', 'w')
            else:
                sat_data = nc4.Dataset(loc_fname, 'w')
            sat_data.Data_Version = 'Version 1.0'
            #Todo pass this value
            sat_data.Model_name=modelname
            
            # Create the dimensions
            # Flux variables are time X L, Kp and Kpmax are time,
            sat_data.createDimension('time', None)
            sat_data.createDimension('Lshell', len(Ls))
            
            # Create time variable
            time = sat_data.createVariable('time', np.uint64, ('time'))
            time.units = 'msec since 1970'
            time.description = 'UT time in msec'
            
            # Create Lshell variable
            L=sat_data.createVariable('Lshell', np.float64, ('Lshell'))
            L[:] = Ls
            L.units = 'L '
            L.description = 'L shell'
            
            # Create Bmirror variable that is different for each L
            Bmir = sat_data.createVariable('Bmirror', np.float64, ('Lshell'))
            Bmir[:] = Bmirrors
            Bmir.units = 'nT'
            Bmir.description = 'Magnetic field at the electron mirror point'

            # Create the flux variables
            for fval in fluxvars:
                flux = sat_data.createVariable(fval, np.float64, ('time', 'Lshell'))
                sat_data[fval].units = '#/cm2-s-str-keV'
                sat_data[fval].description = 'Near equatorial electron flux'
                
            # Create the Kpvars:
            for Kval in Kpvars:
                sat_data.createVariable(Kval, np.float64, ('time'))
                sat_data[Kval].units = 'Decimal Kp from 0-9 '
            sat_data['Kp'].description = 'Kp value'
            sat_data['Kpmax'].description = 'Maximum Kp value in time interval'
            
            # Create the sat variable so you know which satellite pass it was from
            sat_data.createVariable('sat', str, ('time'))
            sat_data['sat'].description = 'Name of satellite'
            sat_data['sat'].units = 'Name of satellite'
            

        # Now write each variable
        # Need to clobber any old data with new data
        # Combine all data (with a 0 for older and 1 for newer)
        # Then sort and diff and if diff< some amount then
        # throw out the older data

        old_times = sat_data['time'][:]
        oflag = np.array([0]*len(old_times))
        
        # Get the indices for fdate.day only
        startofday = 1000.0*pu.unixtime(fdate.replace(hour=0,minute=0,second=0,microsecond=0))
        endofday = 1000.0*pu.unixtime((fdate+dt.timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0))
        tinds = np.where((np.array(outdat['time'][:])>=startofday) & (np.array(outdat['time'][:])<=endofday))[0]
        
        # Do I want this to be an array or list?
        new_times = np.array([outdat['time'][x] for x in tinds])
        nflag = np.array([1]*len(new_times))
    
        if len(old_times)>0:
            # If there is already some data
            # Combine and sort old and new time
            alltimes = np.append(old_times,new_times)
            order_args = np.argsort(alltimes)

            # This is all the times and flags sorted
            # The old data has a flag of 0 and the new data has a flag of 1
            sort_time = alltimes[order_args]
            allflags = np.append(oflag,nflag) 
            sortflags = allflags[order_args]
            
            time_inds = np.append(np.arange(0,len(old_times)),np.arange(0,len(new_times)))
            sort_inds = time_inds[order_args]
            
            # diff sbtracts next -prior
            # So the dups are test_t and test_t+1
            test_t = np.diff(sort_time)
            
            # These are the duplicate times
            # The times are off by msecs
            dups = np.where(np.abs(test_t)<5000)[0]

            # Now replace any dups with the most recent
            for dval in dups:
                # Find the most recent data
                if sortflags[dval]==0:
                    sort_time[dval] =0
                else:
                    sort_time[dval+1]=0

            # Now sort_time should have 0s for data we don't want to keep
            good_inds = np.where(sort_time>0)[0]

            # Now translate that into the data that we want
            for fval in fluxvars:
                temp_data = outdat[fval][tinds]
                alldat = np.append(old_data[fval][:], temp_data, axis=0)
                sat_data[fval][:] = alldat[order_args[good_inds]]
            for Kval in Kpvars:
                temp_data = [outdat[Kval][t] for t in tinds]
                alldat = np.append(old_data[Kval][:], temp_data, axis=0)
                sat_data[Kval][:] = alldat[order_args[good_inds]]
            
            # Now set the time variable
            temp_data = [outdat['time'][t] for t in tinds]
            alldat = np.append(old_data['time'][:], temp_data, axis=0)
            sat_data['time'][:] = alldat[order_args[good_inds]]

            # Need the sat variable
            fval = 'sat'
            temp_data = np.array([sat] * len(tinds))
            alldat = np.append(old_data[fval][:], temp_data, axis=0)
            sat_data[fval][:] = alldat[order_args[good_inds]]


        else:
            # Just Create the file with the new data
            sat_data['time'][:] = new_times
            # Create the flux vars
            for fval in fluxvars:
                sat_data[fval][:] = np.array([outdat[fval][x] for x in tinds])
            for Kval in Kpvars:
                sat_data[Kval][:] = np.array([outdat[Kval][x] for x in tinds])
            sat_data['sat'][:] = np.array([sat]*len(tinds))
            
        sat_data.close()
        # That makes the file here
        # But if were using S3 then need to move it to S3
        # The S3 bucket has data/SHELLS/YYYY
        if cdata is not None:
            # Create the bucket resource and the file object
            s3dir = 'data/SHELLS/'+ fdate.strftime('%Y')
            #file = 'shells_' + fdate.strftime('%Y%m%d') + '.nc'
            s3fname = os.path.join(s3dir,file)
            
            obj = s3.Object(cdata['bucket'],s3fname)

            result = obj.put(Body=open('s3temp.nc', 'rb'))
            #s3.Bucket(cdata['bucket']).download_file(files[0], loc_fname)
            #result = s3.Bucket(cdata['bucket']).object(s3fname).put(Body=open(loc_fname, 'rb'))
            res = result.get('ResponseMetadata')

        fdate = (fdate+dt.timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)


def write_shells_netcdf(outdir, outdat, fname,modelname=None):
    '''
    PURPOSE: To write the data to netcdf files under outdir with base fname
    :param outdir: Directory to write the data with YYYY/SHELLS_YYYYMMDD.nc beneath
    :param outdat:
        The input data is dict (timeXL) for each energy channel
        The NN data dict (timexL) with 'E flux 200.0','E flux 100.0 upper q','E flux 100.0 lower q'
        Kp*10: Kp data for each pass
        Kpmax: Kp max in 3 days for each pass
        Ldims: The L values for the ouputs
        Bmirror: The bmirror values chosen for each L
        Energies:
        sat:
        modelname
    :return:
    '''

    # The data will be stored as netcdf with log flux stored as time X L for each energy
    # The data from each satellite is added separately so we have to
    # 1) read any existing file
    # 2) Add and sort new data
    # 3) rewrite the file
    # File format will be outdir/year/shells_YYYYMMDD_V1.nc

    fstart = outdat['time'][0]
    fdate = pu.unix_time_ms_to_datetime(fstart)

    fedate = pu.unix_time_ms_to_datetime(outdat['time'][-1])
    allvars = outdat.keys() # this is all the data

    #fluxvars = [i for i in allvars if 'flux' in i]  # Get names of flux cols
    Kpvars = [i for i in allvars if 'Kp' in i] # this is the two Kp values

    # Step through each day and write the data
    # fdate is the datetime of the first data point and fedate is the datetime of the last data point
    while fdate < fedate:
        # This is the day file associated with fdate
        file = fname + fdate.strftime('%Y%m%d') + '.nc'

        # This is the local file and path
        loc_fname = os.path.join(outdir, fdate.strftime('%Y'), file)

        # Check if the file exist already

        file_exists = 0

        if (os.path.exists(loc_fname)):
            # Then read in the old data file
            sat_data = nc4.Dataset(loc_fname, 'r+')
            old_data = dict()
            # Turn the old data file into a dict
            for val in sat_data.variables.keys():
                old_data[val] = sat_data[val][:]
            file_exists = 1

        if file_exists == 0:
            # If the file does not exist then start a new file
            # Check that the directory exists and create it if it doesn't

            local_dir = os.path.join(outdir, fdate.strftime('%Y'))
            if not (os.path.exists(local_dir)):
                os.makedirs(local_dir)

            # Create the file locally

            sat_data = nc4.Dataset(loc_fname, 'w')
            sat_data.Data_Version = 'Version 1.0'

            # Todo pass this value
            if modelname is not None:
                sat_data.Model_name = modelname

            # Create the dimensions
            # Flux variables are time X L, Kp and Kpmax are time,
            # The dimensions here are L and time
            for dim in outdat['dims']:
                if dim=='time':
                    sat_data.createDimension(dim, None)
                else:
                    sat_data.createDimension(dim, len(outdat[dim]))
                dimvar = sat_data.createVariable(dim, np.float64, (dim))
                #dimvar[:] = outdat[dim]
                if dim =='time':
                    time.units = 'msec since 1970'
                    time.description = 'UT time in msec'

            # Now create the rest of the vars

            for key in allvars:
                print(key)
                # Create the variables for all columns but the time,L and dims
                if key not in list(outdat['dims']+['dims']):
                    vdims = np.shape(outdat[key])
                    if len(vdims)>1:
                        # All the flux data will have dimensions timeXL
                        sat_data.createVariable(key, np.float64, tuple(outdat['dims']))
                    elif len(vdims)==1:
                        # This will be  Energies, or Bmirrors,Kpvals
                        # JGREEN 5/6/23 Changed this so that it works for Energies and Bmirrors
                        # Check if it has the len of one of the dims
                        ldims = []
                        for dim in outdat['dims']:
                            ldims= ldims+[len(outdat[dim])]
                        # Energies shouldn't be tied to L so leav it out here
                        if (len(outdat[key]) in ldims) & (key!='Energies') :
                            #If the len of the variable matches a dimension then give it that dim
                            sat_data.createVariable(key, np.float64, (outdat['dims'][ldims.index(len(outdat[key]))]))
                        else:
                            # Otherwise create a new one
                            newdim= sat_data.createDimension(key, len(outdat[key]))
                            sat_data.createVariable(key, np.float64, (key))

            # Create the sat variable so you know which satellite pass it was from
            sat_data.createVariable('sat', str, ('time'))
            sat_data['sat'].description = 'Name of satellite'
            sat_data['sat'].units = 'Name of satellite'

        # Now write each variable
        # Need to clobber any old data with new data
        # Combine all data (with a 0 for older and 1 for newer)
        # Then sort and diff and if diff< some amount then
        # throw out the older data

        old_times = sat_data['time'][:]
        oflag = np.array([0] * len(old_times))

        # Get the indices for fdate.day only
        startofday = 1000.0 * pu.unixtime(fdate.replace(hour=0, minute=0, second=0, microsecond=0))
        endofday = 1000.0 * pu.unixtime(
            (fdate + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
        tinds = np.where((np.array(outdat['time'][:]) >= startofday) & (np.array(outdat['time'][:]) <= endofday))[0]

        # Do I want this to be an array or list?
        new_times = np.array([outdat['time'][x] for x in tinds])
        nflag = np.array([1] * len(new_times))

        if len(old_times) > 0:
            # If there is already some data
            # Combine and sort old and new time
            alltimes = np.append(old_times, new_times)
            order_args = np.argsort(alltimes)

            # This is all the times and flags sorted
            # The old data has a flag of 0 and the new data has a flag of 1
            sort_time = alltimes[order_args]
            allflags = np.append(oflag, nflag)
            sortflags = allflags[order_args]

            time_inds = np.append(np.arange(0, len(old_times)), np.arange(0, len(new_times)))
            sort_inds = time_inds[order_args]

            # diff sbtracts next -prior
            # So the dups are test_t and test_t+1
            test_t = np.diff(sort_time)

            # These are the duplicate times
            # The times are off by msecs
            dups = np.where(np.abs(test_t) < 5000)[0]

            # Now replace any dups with the most recent
            for dval in dups:
                # Find the most recent data
                if sortflags[dval] == 0:
                    sort_time[dval] = 0
                else:
                    sort_time[dval + 1] = 0

            # Now sort_time should have 0s for data we don't want to keep
            good_inds = np.where(sort_time > 0)[0]

            # Now translate that into the data that we want
            for key in allvars:
                # 05/08/2023 had to change this because the neural data has Bmirrors and
                # Energies. Don't write them again or time L
                if key not in list(outdat['dims'] + ['dims']+['Energies','Bmirrors','sat']):
                    print(key)
                    #Check if it is an array or list
                    if type(outdat[key])==list:
                        temp_data = [outdat[key][t] for t in tinds]
                    else:
                        temp_data = outdat[key][tinds]
                
                    alldat = np.append(old_data[key][:], temp_data, axis=0)
                    sat_data[key][:] = alldat[order_args[good_inds]]
            # Now set the time variable
            temp_data = [outdat['time'][t] for t in tinds]
            alldat = np.append(old_data['time'][:], temp_data, axis=0)
            sat_data['time'][:] = alldat[order_args[good_inds]]

            # Need the sat variable
            fval = 'sat'
            temp_data = np.array([outdat[fval]] * len(tinds))
            alldat = np.append(old_data[fval][:], temp_data, axis=0)
            sat_data[fval][:] = alldat[order_args[good_inds]]

        else:
            # Just Create the file with the new data

            sat_data['time'][:] = new_times
            # Create the flux vars
            for key in allvars:
                print(key)
                # Write all the flux variables
                if key not in list(outdat['dims'] + ['dims'] +['Bmirrors','Energies','sat']):
                    sat_data[key][:] = np.array([outdat[key][x] for x in tinds])
                if key in ['Bmirrors','Energies','L']:
                    # These are separate because there is no time component
                    sat_data[key][:] = np.array(outdat[key][:])
                # Create the sat variable
                if key =='sat':
                    sat_data[key][:] = np.array([outdat[key]] * len(tinds))
            

        sat_data.close()

        fdate = (fdate + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def write_shells(outdir, outdat, otype, fname):
    if (otype == 'json') | (otype =='csv'):
        if outdir is None:
            outdir = os.getcwd()
        # This writes files to an existing daily json file called
        # either shells_inputs_YYYYMMDD_HHMMSS
        # or shells_YYYMMDD_HHMMSS
        finish = write_shells_text(outdir, outdat, fname,otype)
    elif cdict['outfile_type'] == 'nc':
        if outdir is None:
            outdir = os.getcwd()
            finish = write_shells_netcdf(outdir, outdat,neural,fname)
    else:
        # This will write the data to a dbase
        # Todo fix this
        finish = write_shells_inputs_dbase(cdict, map_data, channels, sat, logger)
        
def write_shells_dbase(outdir,outdat,cdict):
    '''
    PURPOSE: To write the data in outdir to an sqlite dbase at outdir/dbname
    and store the data in tblname.
    This assumes that the database already exists with name dbname in outdir and has
    a table called tblname that has the column names set up already with names
    time, 'mep_ele_tel90_flux_e1_L_1.0',...'mep_ele_tel90_flux_e1_L_8.0',
    ...'mep_ele_tel90_flux_e4_L_8.0', 'Kp*10','Kp_max','satID'
    :param outdir:
    :param outdat:
    :param dbname:
    :param tblname:
    :return:
    Todo: This could be updated to look for the dbase and table and if it is not there then
    create it with the appropriate cols ets
    '''
    # Get the start and end time of the data
    dbase = cdict['dbase']
    tblname =cdict['tblname']
    dbtype = cdict['output_type']

    fstart = outdat['time'][0]
    fdate = pu.unix_time_ms_to_datetime(fstart)  # Change to datetime
    fedate = pu.unix_time_ms_to_datetime(outdat['time'][-1])
    dformat = '%Y-%m-%dT%H:%M:%S.000Z'

    # Get just the eflux columns
    fcols = [x for x in list(outdat.keys()) if 'flux' in x]

    newtime = pu.unix_time_ms_to_datetime(outdat['time'])

    # Add data to daily files
    while fdate <= fedate:
        # Reformat the outdat dict so that every e flux and L has a key
        # and get only values for fdate
        nextday = (fdate + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        dayinds = np.where((newtime >= fdate) & (newtime < nextday))[0]

        newdat = {}  # This is the reformatted new data that will be written

        # Create a string from time that json/csv can write

        newdat['time'] = [newtime[x].strftime(dformat) for x in dayinds]

        for key in fcols:
            # Split data into lists for each L bin and E
            if len(np.shape(outdat[key])) > 1:
                Lname = [i for i in outdat['dims'] if 'L' in i][0]  # Get the L name and
                for lco, Lval in enumerate(outdat[Lname]):
                    newcol = key + '_L_' + str(Lval)  # rename the cols to have e flux name and L bin
                    newdat[newcol] = list(outdat[key][dayinds, lco])
            else:
                newdat[key] = list(outdat[key][dayinds])

        # Add the Kp value that was used
        Kpcols = [x for x in list(outdat.keys()) if 'Kp' in x]
        for Kpcol in Kpcols:
            newdat[Kpcol] = list(outdat[Kpcol][dayinds])

        # Add a satid col
        satid = poes_sat_sem2(outdat['sat']).satid()
        newdat['satID'] = [satid] * len(dayinds)

        try:
            # connect to the dbase
            if dbtype=='sqlite':
                conn = create_conn_sqlite(os.path.join(outdir,dbase))
                cursor = conn.cursor()

                # Now add the data
                query = 'INSERT or REPLACE INTO '+tblname+ '('
            else:
                conn = create_conn_msql(cdict)
                cursor = conn.cursor(buffered=True)
                query = 'INSERT INTO ' + tblname + '('

            keys = list(newdat.keys())
            for kco in range(0,len(keys)-1):
                query = query+'"'+keys[kco]+'", '
            query = query+keys[-1]
            if dbtype =='sqlite':
                query = query+') VALUES ('
                for kco in range(0,len(keys)-1):
                    query = query+'?,'
                query = query+'?)'
                cursor.executemany(query, np.array(list(newdat.values())).T)
            else:
                query = query+') VALUES ('
                for kco in range(0,len(keys)-1):
                    query = query+'%s,'
                query = query+'%s)'
                #Todo need to test this for mysql
                query =query +' ON DUPLICATE KEY UPDATE VALUES '
                for kco in range(1, len(keys) - 1):
                    query = query + '"' + keys[kco] + '"=%s, '
                cursor.executemany(query, np.array(list(newdat.values())).T,np.array(list(newdat.values())).T)

        except:
            logger.error("Can't add to dbase")

        fdate = nextday
            
    cursor.close()
    conn.commit()
    conn.close()

    return

def write_shells_text(outdir, outdat, fname, otype):
    '''
    PURPOSE: To write the shells input or neural data to a json file
    :param outdir (str): The directory location to write the files to
    :param outdat (str): The data to write out
    :param fname (str): the base name of the file to write
    :param otype (str): either 'csv' or 'json'
    :return:
    
    This writes out the mapped shells input data or the neural network data
    Both are stored as dicts of arrays with dimensions timexL for each E flux
    The dimensions are passed as outdat['dims'] = ['time','L_IGRF']
    The data are reformated here to have a column/key for each E and L bin
    New data is added to existing files for that day and then the filename
    is changed to have the time of the last datapoint.
    
    Writes out time, e1 L1, e1 L2, ...e2 L1, e2 L2, ... Kp, Kpmax, satId
    '''
    # Get the start and end time of the data

    fstart = outdat['time'][0]
    fdate = pu.unix_time_ms_to_datetime(fstart) # Change to datetime
    fedate = pu.unix_time_ms_to_datetime(outdat['time'][-1])
    dformat ='%Y-%m-%dT%H:%M:%S.000Z'

    # Get just the eflux columns
    fcols = [x for x in list(outdat.keys()) if 'flux' in x]

    newtime = pu.unix_time_ms_to_datetime(outdat['time'])

    # Add data to daily files
    while fdate<=fedate:
        # Reformat the outdat dict so that every e flux and L has a key
        # and get only values for fdate
        nextday = (fdate+dt.timedelta(days = 1)).replace(hour = 0,minute=0,second=0,microsecond=0)
        dayinds = np.where((newtime>=fdate) & (newtime<nextday))[0]

        newdat = {} # This is the reformatted new data that will be written
        
        # Create a string from time that json/csv can write

        newdat['time'] = [newtime[x].strftime(dformat) for x in dayinds]

        for key in fcols:
            # Split data into lists for each L bin and E
            if len(np.shape(outdat[key]))>1:
                Lname = [i for i in outdat['dims'] if 'L' in i][0] # Get the L name and
                for lco,Lval in enumerate(outdat[Lname]):
                    newcol= key+'_L_'+str(Lval) # rename the cols to have e flux name and L bin
                    newdat[newcol] = list(outdat[key][dayinds,lco])
            else:
                newdat[key] = list(outdat[key][dayinds])
        

        # Add the Kp value that was used
        # Add the Kp value that was used
        Kpcols = [x for x in list(outdat.keys()) if 'Kp' in x]
        for Kpcol in Kpcols:
            newdat[Kpcol] = list(outdat[Kpcol][dayinds])

        # Add a satid col
        #satid = poes_sat_sem2(outdat['sat']).satid()
        satid = outdat['sat']
        # JGREEN: 08/2023 Changed satid to be a string
        newdat['satID'] = [satid]*len(dayinds)

        # Check if there is a daily file started already
        fout = os.path.join(outdir,fname+'_'+fdate.strftime('%Y%m%d')+'*.'+otype)
        flist = glob.glob(fout)
        
        if len(flist)>0:
            # If there is a file already, then read in the data
            # Opening JSON file
            if otype == 'json':
                with open(flist[0], 'r') as openfile:
                    oldat = json.load(openfile)
            else:
                #
                oldat = pd.read_csv(flist[0]).to_dict(orient='list')
            
            oflag = [0]*len(oldat['time'])
            otime = [dt.datetime.strptime(x, dformat).timestamp() for x in oldat['time']]
            nflag = [1]*len(dayinds)
            ntime = [dt.datetime.timestamp(newtime[x]) for x in dayinds]
            # Now append the new data, get rid of dups and rewrite the file
            alltimes = np.array(otime+ntime)
            order_times = np.argsort(alltimes) # sorted indices
            sortedtimes = alltimes[order_times]
            
            allflags = np.append(oflag, nflag)
            sortflags = allflags[order_times]

            # diff sbtracts next -prior
            test_t = np.diff(sortedtimes)

            # These are the duplicate times
            # The times are sometimes off by msecs
            dups = np.where(np.abs(test_t) < 5)[0]

            # Now replace any dups with the new data
            for dval in dups:
                # Set the duplicate value of old data to 0
                if sortflags[dval] == 0:
                    sortedtimes[dval] = 0
                else:
                    sortedtimes[dval + 1] = 0
            # Now sortedtimes should have 0s for data we don't want to keep
            good_inds = np.where(sortedtimes>0)[0]

            for key in list(newdat.keys()):
                temp = oldat[key]+newdat[key]
                newdat[key] = [temp[order_times[x]] for x in good_inds]
            if otype=='json':
                json_object = json.dumps(round_floats(newdat), indent=4)
                
        else:
            # Serializing json
            if otype=='json':
                json_object = json.dumps(round_floats(newdat), indent=4)

        # Writing file
        tfile = dt.datetime.strptime(newdat['time'][-1],dformat)
        foutnow = os.path.join(outdir,fname+'_'+tfile.strftime('%Y%m%dT%H%M%S')+'.'+otype)

        try:
            if otype == 'json':
                with open(foutnow, "w") as outfile:
                    outfile.write(json_object)
            else:
                with open(foutnow, "w") as outfile:

                    # pass the csv file to csv.writer function.
                    writer = csv.writer(outfile)

                    # pass dict keys to writerow
                    # function to give the columns
                    writer.writerow(newdat.keys())

                    # use writerows function to append values to the corresponding
                    # columns using zip function.
                    writer.writerows(zip(*round_floats(newdat).values()))
        except Exception as e:
            logger.error("Couldn't write file")

        if len(flist)>0:
            # If its updating in real time then the HMS of the filname
            # will change as new data is added so delete the old one
            if foutnow !=flist[0]:
                os.remove(flist[0])

        fdate = nextday

    return

def round_floats(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def get_Lvals(data, inds=None):
    '''
    :param data: a dictionary with POES data
    :param inds: indices to get if not all
    :return: 
    '''
    # -----------------------------Get L, MLT data -------------------
    # The raw data does not have L or MLT so create it here with spacepy


    # Create coords object needed by spacepy to create L
    # Sometimes the poes location data is bad and will give a weird answer
    if inds is None:
        inds = np.arange(0,len(data['L_IGRF']))

    # Spacepy needs a special time object.
    # Divide by 1000 becase poes time is in msec

    ticks = time.Ticktock(data['time'][inds] / 1000.0, 'UNX')
        
    bad_inds = np.where((data['alt'][inds] <= 0) | (data['lat'][inds] < -90) |
                        (data['lon'][inds] < 0))[0]

    orbit = [[(data['alt'][x] + 6371.0) / 6371.0, data['lat'][x], data['lon'][x]]
             for x in inds]
    locinew = coord.Coords(orbit, 'GEO', 'sph')
    
    locinew.ticks = ticks[:]
    opts = [0, 0, 0, 0, 0]

    # Create L assuming locally 90 deg pitch for IGRF and no external field
    # Ldata=sp.irbempy.get_Lm(ticks, locinew, 90, intMag='IGRF', extMag='0',IGRFset=0, omnivals=None)
    # With opts all set to 0 this gets Lm but uses multi-threading so its fast
    # I had to tweak this code to work so that it did not call shell_splitting even for 90 deg

    # This is expecting a dictionary of omnivals even though you don't need it for IGRF
    zerovec = [0] * len(ticks)
    # This is the input dictionary needed even though all it uses is Kp
    # It needs a dict with more than one value even though it is only using the first, It's a bug in spacepy
    om = {'Kp': zerovec, 'Dst': zerovec, 'dens': zerovec, 'velo': zerovec, 'Pdyn': zerovec,
          'ByIMF': zerovec, 'BzIMF': zerovec, 'G1': zerovec, 'G2': zerovec,
          'G3': zerovec, 'W1': zerovec, 'W2': zerovec, 'W3': zerovec, 'W4': zerovec, 'W5': zerovec,
          'W6': zerovec, 'AL': zerovec}
    
    #Ldata = sp.irbempy.get_Lstar(ticks, locinew, 90, extMag='0', options=opts, omnivals=om)
    Ldata = get_Lstar(ticks, locinew, 90, extMag='0', options=opts, omnivals=om)
    Ldata['Lm']= np.abs(Ldata['Lm'][:])
    Ldata['Lm'][bad_inds] = -1
    
    return Ldata


