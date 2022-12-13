import argparse
import glob
import datetime as dt
import numpy as np
import logging
import sys
import os
import fnmatch
import requests
from joblib import load
import keras
import netCDF4 as nc4
from io import BytesIO
import boto3
import mysql.connector
import sqlite3 as sl
import matplotlib.pyplot as plt
import time

#from spacepy import coordinates as coord
#from spacepy.irbempy import get_Lstar
#import spacepy as sp

#sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
import poes_utils as pu
import data_utils as du
import shells_web_utils as swu


# ----------------------- Basic functions ----------------------------
#---------------------------------------------------------------------
def qloss(y_true,y_pred):
    qs = [0.25,0.5,0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true-y_pred
    v = np.maximum(q*e, (q-1)*e)
    return keras.backend.mean(v)

def valid_date(s):
    '''
    ---------------------------------------------------------------
    PURPOSE: To check that a valid date is entered as an input

    :param s (str) a date in the format Y-m-d or Y-m-d H:M:S
    ---------------------------------------------------------------
    '''

    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d")
        return test
    except:
        pass
    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return test
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def longsat(sat):
    '''
    ---------------------------------------------------------------
    PURPOSE: to change from short sat name to long sat name
    :param sat (str):       A POES satellite name like 'n18'
    :return longpoes (str): A long POES satellite name lik 'noaa18'
    USAGE:
    ---------------------------------------------------------------
    '''
    allpoes = ['n12','n14','n15','n16','n17','n18','n19','m01','m02','m03']
    longpoes = ['noaa12','noaa14','noaa15','noaa16','noaa17','noaa18','noaa19','metop01','metop02','metop03']
    num = allpoes.index(sat)
    return(longpoes[num])


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

def setupLogging(logfile):
    '''
    PURPOSE: To set up a logger for tracking issues
    :param logfile:
    :return:
    '''

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) # log at the lowest level

    # Create the output format
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:  %(message)s')

    # JGREEN: To make this work with the async system
    # you can't have the filehandler
    file_handler = logging.FileHandler( logfile )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def find_s3_last_time(cdata,fn_pattern):
    '''
    :param cdata:
    :param fn_pattern:
    :return:
    '''

    # If it is an S3 bucket, then first connect to bucket
    #04/18/2022 Changed shells to SHELLS
    prefix = 'data/SHELLS/'
    s3 = boto3.resource(
        service_name=cdata['service_name'],
        region_name=cdata['region_name'],
        aws_access_key_id=cdata['aws_access_key_id'],
        aws_secret_access_key=cdata['aws_secret_access_key']
    )
    # List objects in the directory
    # for the last 2 years
    allfiles = []
    for year in [dt.datetime.utcnow().year-1,dt.datetime.utcnow().year]:
        prefixyear = os.path.join(prefix,str(year),'shells')
        obj_summary = s3.Bucket(cdata['bucket']).objects.filter(Prefix=prefixyear)
        # Create a lsit of files in the directory
        files = [x.key for x in obj_summary]
        if len(files)>0:
            allfiles.extend(files)
    if len(files)<1:
        sdate = dt.datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
    else:
        #Get the last file and find the last time
        allfiles.sort()
        # This will download a file
        #test=s3.Object('spacehaz.com','shells_data/test.csv').get()
        # JCGREEN This changed in boto3 so I fixed it to get files
        obj = s3.Object(cdata['bucket'], allfiles[-1])
        sdata = obj.get()
        #sdata = s3.Bucket(cdata['bucket']).objects.get(allfiles[-1])
        data = nc4.Dataset('noname', memory=BytesIO(sdata['Body'].read()).read(),mode='r')

        # Or can use urllib to get the file
        #data = requests.get(allfiles[-1]).content
        #test = nc4.Dataset('noname', memory=data,mode='r')
        # Now get the las time in the file
        # JGREEN I don't know why this changed 04/18/2022
        #times = data['time'].set_auto_mask(False)
        times = data['time'][:]
        dtimes = pu.unix_time_ms_to_datetime(times)
        sdate = dtimes[-1]

    return sdate

def find_last_file(outdir,fn_pattern):
    '''
    PURPOSE: To find the latest file to add SHELLS data to.
    :param outdir: The directory to look for files or a config file with .ini as the last bit
    :return fname: The file to add to
    '''

    # The assumed structure is outdir/YYYY/SHELLS_YYYYMMDD.nc
    # So list all the files in this year and last year

    thisyear = dt.datetime.utcnow().year
    # make a list of directories with the current year and next year
    dir_root_list = [os.path.join(outdir,str(thisyear-1)),os.path.join(outdir,str(thisyear))]
    fn_list = list()

    # Make a list of all the files in the directories with fn_pattern
    for dir_root in dir_root_list:
        #print(dir_root)
        for root, dirnames, filenames in os.walk(dir_root):
            #print(root,dirnames,filenames)
            for filename in fnmatch.filter(filenames, fn_pattern):
                fn_list.append(os.path.join(root, filename))

    # Now order the files
    fn_list.sort()
    if len(fn_list)>0:
        last_file = fn_list[-1]
    else:
        last_file = []
    return last_file

def get_last_time(fname):
    '''
    ----------------------------------------------------------
    PURPOSE: To get the last time in the most recent file to start
    processing from

    :param fname:
    :return:
    '''
    try:
        # Load the file
        last_data = nc4.Dataset(fname, 'r')

        # The file should have 'time column
        time = last_data['time'][:]
        ltime = pu.unix_time_ms_to_datetime(time[-1])

        # If it is more than 10 days ago then start with current time

        current_time = dt.datetime.utcnow()
        if ltime< current_time-dt.timedelta(days = 10):
            ltime = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        last_data.close()
    except:
        ltime = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    return ltime


def connectToDb(cdict, logger):
    '''
    PURPOSE: To connect to a mysql or sqlite dbase
    :param cdict: dictionary with needed infor for connecting to dbase
    :param logger: for tracking errors
    :return conn: the dbase connection object

    # For testing we use an sqlite database. The connection process for
    # that is different because you don't need a user, pass or host
    # So I added a second try except
    '''
    conn = None

    try:

        dbuser = cdict['dbuser']
        dbpass = cdict['dbpass']
        dbhost = cdict['dbhost']
        dbase = cdict['dbase']

        # Try connecting
        conn = mysql.connector.connect(user=dbuser,
                                       password=dbpass,
                                       host=dbhost,
                                       database=dbase)

    except Exception as e:
        # try connectin as sqlite
        try:
            conn = sl.connect(cdict['dbase'])
        except Exception as e:
            logger.exception(e)
        if conn is not None:
            conn.close()

    return conn




def get_start_rt(cdict,sat,logger):
    '''
    PURPOSE: to get the start data for real time processing by checking the last data
    # in the dbase
    :param cdict: dictionary with info for connection to dbase
    :param sat: the satellite data to lok or
    :param logger:
    :return:

    A satellite integer id is used for storing data
    That id could be table in the database with the names
    of the satellites (i.e. 'n15') or use the simple function since
    there are only 8 so for now its just a function.
    '''

    # ----------- Connect to dbase --------------
    # This will connect to sql or sqlite
    conn = connectToDb(cdict, logger)
    cursor = conn.cursor()

    # Get the sat Id from dbase
    # query = "SELECT id from "+cdict['sattbl'] +" WHERE satName = %s"
    # cursor.execute(query,(sat,))
    # satId = cursor.fetchone()[0]

    # Get sat Id from fixed function that numbers
    # the sats 1-8
    satId = swu.satname_to_id(sat)

    if satId is not None:
        query = "SELECT max(unixTime_utc) from "+ cdict['inputstbl'] + " WHERE satId=%s"
        cursor.execute(query,(satId,))
        stime = cursor.fetchone()[0]
        if stime is None:
            sdate = dt.datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
        else:
            #Todo check that this is giving the right time
            sdate = dt.datetime.utcfromtimestamp(stime)
    else:
        # Todo This should probably exit and throw an error
        sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    return sdate

def find_model_files(modeldir, model):
    '''
    PURPOSE: To retrieve the three model files to use for processing. The files are
     in_scale*.bin, out_scale *.bin, and shells_model*.hd5.

    Most often this will be called with None as the modeldir and model. In that
    case it finds the model with the latest date. The models will all be saved
    as name_MMDDYYYY.*. If a directory is passed it will look for the files in that
    directory otherwise it assumes the local directory. If a model name is passed
    it will look for models with that string in it so that a model can be
    retrieved with a specific date.

    :param modeldir (str): The directory where the model is kept
    :param model (str): The name of the model to use
    :return: 3 file names to open

    '''
    if modeldir is None:
        modeldir = os.getcwd()
    elif type(modeldir) is str:
        # Ensure the path exists
        if not os.path.isdir(modeldir):
            msg = 'The path to the Keras model directory was not found.'
            print(msg)
            raise Exception(msg)

    if model is None:
        #
        # Use the latest instance of the model
        #
        modelInsts = glob.glob(os.path.join(modeldir,"shells_model_*" + "h5"))
        if len(modelInsts) == 1:
            # If only one version is found then use that one
            model = modelInsts[0].split("/")[-1].split("_")[-1].split(".")[0]
        else:
            #
            # There is more than one model present
            #
            modelDates = [int(_m.split("/")[-1].split("_")[-1].split(".")[0]) for _m in modelInsts]
            model = str(max(modelDates))
    else:
        if not type(model) is str:
            model = str(model)

    # model here is just the date
    out_scale_file = glob.glob(os.path.join(modeldir,"out_scale_*" + model + ".bin"))
    in_scale_file = glob.glob(os.path.join(modeldir, "in_scale_*" + model + ".bin"))
    hdf5File = glob.glob(os.path.join(modeldir,"shells_model_*" + model + ".h5"))

    return(out_scale_file,in_scale_file,hdf5File)

def write_shells_inputs_dbase(cdict,map_data,channels,sat,logger):
    '''
    PURPOSE: to write the shells input data to a dbase
    :param cdict: dictionary with dbase info
    :param map_data:
    :return:
    '''
    # The mapped data is dicts with time and channels
    # Each chanel is an array that is timeXL
    # First connect to dbase
    # Connect to dbase
    conn = connectToDb(cdict, logger)
    cursor = conn.cursor()
    satId = swu.satname_to_id(sat)

    query = "INSERT INTO " + cdict['inputstbl'] + "(unixTime_utc,channelId,LId,satId,eflux)" \
                                                  " VALUES (%s,%s,%s,%s,%s)"
    for ch in channels:
        chId = swu.poeschan_to_id(ch)

        indat = []
        row,col = np.shape(map_data[ch])
        for rco in range(0,row):
            for cco in range(0,col):
                indat.append((int((map_data['time'][rco])/1000),chId,cco+1,satId,map_data[ch][rco,cco]))

        cursor.executemany(query,indat)
        conn.commit()
        cursor.close()



    print('Here')
    return


#------------------- The main process_SHELLS function--------------------------
#===========================================================================

def process_SHELLS(sdate_all=None, edate=None, realtime=False, neural=False, localdir=None, outdir=None, cdfdir= None,
                noaasite=None,sats=None,
                vars=['time','alt','lat','lon','L_IGRF','MLT',
                      'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3','mep_ele_tel90_flux_e4'],
                channels = ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                            'mep_ele_tel90_flux_e4'], model=None, modeldir=None, logfile = None, configfile=None):
    '''
    ---------------------------------------------------------------------------------------------
    PURPOSE: To collect POES/MetOP data from NCEI/NGDC, and process it for the shells model

    :param: sdate_all (datetime) - date to start creating shells data if reprocessing and not rt mode
    :param: edate (datetime) - date to stop creating shells data if reprocessing and not rt mode
    :param: realtime (int) - 0 for not realtime and 1 for real time
            Default-  0 False
    :param: localdir (str) - Local directory of POES data. If this is passed it will look for data
                         locally instead of going to the noaa website (This is not well tested)
    :param: outdir (str) - Output directory for daily netcdf files of processed  data. If a config
                            file is passed then output data goes into dbase or S3
    :param: cdfdir (str) - Directory of the cumulative dist. function. files needed for mapping
                            poes data to a consistent location
    :param: noaasite (str) - noaa website to get data from if the data is not local 'satdat.ngdc.noaa.gov'
    :param: sats list(str) - satellites to get data from
            default from command line= ['n15','n18','n19','m01','m02','m03']
    :param: vars list(str) - all variables to get from the poes data files needed for processing
            default = ['time','alt','lat','lon','L_IGRF','MLT','mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
             'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']
    :param: channels list(str) - particle flux data variables to use from poes data files
            default from commnad line ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                               'mep_ele_tel90_flux_e4']
    :param: model (datetime or str) - The instance of the neural network to use in processing. If the input is a str,
             it must be in the YYYYMMDD format. Datetime strings will be reformatted to conform to this naming convention
             Default from command line is None. If this is the case, the most recent instance  of the neural network will
             be used.
    :param: modeldir (str) - The path (relative or absolute) to the keras model files identified in the model parameter.
             Default from command line is None, which will invoke a search in the current working directory. (i.e. ./.)
    :param  logfile (str) - location and base name of logfile. The year month and .log will be appended to the basename
    :param: configfile (str) - filename with configuration setting. This is used in conjuction with AWS
             so that data can be passed to the S3 bucket that has the SHA web pages or with a mysql database

    USAGE:

    FROM COMMAND LINE
    ----------------
        (REAL TIME example)
        python process_SHELLS.py -rt -ns satdat.ngdc.noaa.gov -sa n15 n18 n19 m01 m02
            -cdf /efs/spamstaging/SHELLS/cdfdata -log /efs/spamstaging/logs/process_SHELLS_rt -c ./configAWS.ini

        (REPROCESSING getting data from noaa example)
        (This automatically creates a logfile in the current directory)
        python process_SHELLS.py -s 2015-01-01 -e 2015-01-02 -ns satdat.ngdc.noaa.gov -sa n15 n18
            -cdf /Users/janet/PycharmProjects/SHELLS/cdfdata

        (REPROCESSING getting data locally)
        python process_SHELLS.py -s 2001-01-01 -e 2002-01-01 -ld /Users/janet/PycharmProjects/data/sem/poes/data/raw/ngdc/
            -sa n15 n16 n18 n19 m01 m02

    AS A FUNCTION
    -------------------
    (REAL TIME on AWS)
    import process_POES as proc
    proc.process_POES(realtime=True, noaasite='satdat.ngdc.noaa.gov', sats= ['n15','n18','n19','m01','m02'],
        logfile = '/efs/spamstaging/logs/process_POES_reprocess'

    (REPROCESSING on AWS)
    import process_POES as proc
    proc.process_POES(realtime=FALSE, noaasite='satdat.ngdc.noaa.gov', sats= ['n15','n18','n19','m01','m02'],
        logfile = '/efs/spamstaging/logs/process_POES_reprocess'

    '''
    try:
        #-------------------- Logging Setup ----------------------------------------------
        # Create a logile if requested.
        # When called from command line theres always a logfile because a default is given
        # './process_shells_inputs' The year and month are appended here.

        if logfile is not None:
            YMO = dt.datetime.utcnow().strftime("%Y_%m_%d")
            logfile = logfile + '_' + YMO
            logger = setupLogging(logfile)

        #---------------- Set up mapping info -------------------
        # cdf_dir: dir where the cummulative distribution function data is located
        # that are needed for mapping
        # These cdf files are lookup tables to go from flux to percentile and vice versa
        # Todo make a default here because its required and add a check for the files
        cdf_dir = cdfdir
        # These should not be changed because it is what the neural network was
        # trained with
        sat_ref = 'm02'  # The reference sat to map to. Will always be m02 for now
        reflon = 20  # The reference lon to map to
        ref_ind = int(np.floor(reflon / 10)) # The index of the ref lon to map to

        # ------------- Input Checks -------------------------------
        # Make sure sats is a list ['n18','n19'] or ['n19] and not one str 'n19'
        # Thats a common error
        if type(sats) != list:
            sats = [sats]

        # ----------------- Set up nn if needed --------------------
        # load in transforms,model for Neural Network that are always a trio of files:
        # in_scale, out_scale binary files and a model_quantile HDF5 file.
        #
        if neural is True:
            out_scale_file, in_scale_file, hdf5File = find_model_files(modeldir,model)
            # Check Models before loading
            #
            if (len(out_scale_file) == 0) | (len(in_scale_file) == 0) | len(hdf5File) == 0:
                msg = 'The NN model files were not found.'
                print(msg)
                raise Exception(msg)
        
            out_scale = load(out_scale_file[0])
            in_scale = load(in_scale_file[0])
            m = keras.models.load_model(hdf5File[0], custom_objects={'loss': qloss}, compile=False)

        # -----------------------Read config file if passed --------------
        # If a config file is passed the shells data will be accessed and stored
        # either at AWS S3 bucket or mysql type dbase (CCMC)

        if configfile is not None:
            cdict, dbase_type = swu.read_config(configfile)
        else:
            cdict = None

        #------------- Get processing end time ------------------------
        # Real time mode: end time will be utc now.
        # Reprocessing mode: end is whatever is passed

        if realtime:
            edate = dt.datetime.utcnow()
        else:
            # If reprocessing mode and no start and end time are passed then
            # print messsage and return. It can't run without those
            if ( (sdate_all is None) | (edate is None) ):
                msg = 'Must give start and end time if not real time mode'
                print(msg)
                raise Exception(msg)

        # --------------- Step through through each sat that is passed -----------------------

        for sat in sats:

            # ----------- Get start times ---------------------
            # Real time mode: start time is from last processed data for that sat
            # Reprocessing mode: start and end time must be passed

            if realtime:
                # sdate should be the last processed data time for sat
                # If no processed data exists yet then start with current day
                # Data can be stored as files (local or S3 bucket) or in sql dbase

                # If no output file directory passed then assume it is the current one
                # Todo Do I need this?
                if outdir is None:
                    outdir =os.getcwd()

                if configfile is not None:
                    # If there's a configfile the data is in S3 bucket or sql dbase
                    # Which one is set by the header in the config file
                    if dbase_type=='S3':
                        # Get start date from files at S3 bucket
                        sdate = find_s3_last_time(cdict, 'shells_*.nc')
                    else:
                        # Or get start date from dbase
                        # Todo make this also work with sqlite testing dbase
                        sdate = get_start_rt(cdict, sat, logger)
                else:
                    # Otherwise check local files in outdir
                    fname = find_last_file(outdir,'shells_*.nc')

                    if len(fname)<1:
                        # If no files are found then start a new file at the beginning of the day
                        sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    else:
                        sdate = get_last_time(fname)

            else:
                # Reprocessing mode: use sdate_all that is passed
                # sdate gets reset to sdate_all for each satellite loop
                sdate = sdate_all

            #---------------   Process 1 day at a time-----------------------
            #
            while sdate <= edate:

                # Todo Check what happens in real time if I try and get data that is not there
                print(sdate.strftime("%Y%m%d")+' '+sat)

                # Real time mode: Check if latest file has been updated before
                # pulling it so as not to annoy NOAA
                # Check if Last-Modified of the file is after sdate
                if realtime == 1:
                    # Make a list of remote files
                    # TODO: make sure that if the code can't connect then it exits
                    # gracefully with a logged error

                    # Check what files are at noaasite for sdate and sat
                    file_test = pu.get_file_list_remote(sat, sdate, sdate, dir_root_list=None, dtype = 'proc', swpc_root_list=None,
                                all=True ,site = noaasite)

                    # Get info on the first file
                    #TODO: check that returned value is ok
                    check_file = requests.head(file_test[0], allow_redirects=True)

                    # This is the last update to the file at NOAA
                    fdate = dt.datetime.strptime(check_file.headers['Last-Modified'],'%a, %d %b %Y %H:%M:%S GMT')

                    # If it has been updated then pull the data

                    # Passes may go across the day file boundary so get data from the prior 90
                    # minutes to ensure a complete first pass. (Data is usually dumped once (or twice) per orbit.
                    # i.e., if sdate is 2000-01-02 00:00:00 the code gets data the day before as well
                    # If sdate is 2000-01-02 01:30:00 then it will just get one day of data
                    if fdate > sdate:
                        data = pu.get_data_dict(sat, sdate - dt.timedelta(minutes=90), sdate, dataloc=localdir,
                                        vars=vars, all=True, dtype='proc', site=noaasite)
                    else:
                        data = None
                else:
                    # Reprocessing mode: just get the full day of data from the remote NOAA site
                    data = pu.get_data_dict(sat, sdate - dt.timedelta(minutes=90), sdate, dataloc=localdir,
                                            vars=vars, all=True, dtype='proc', site=noaasite)

                # If data was returned then process it. Otherwise do nothing but log missing data

                #------------------------- Process returned data ---------------------------------------

                if data is not None:
                    #------------ Make sure data starts with a complete pass -------------
                    # Find index of sdate - 90 minutes to make sure we start with a complete pass
                    i_last = np.where(data['time'][:] > pu.unixtime(sdate -dt.timedelta(minutes=90)) * 1000)[0]

                    # Update the data dict so lists of data start at the 90 minute before index
                    if len(i_last)>0:
                        for key in data.keys():
                            data[key]=data[key][i_last[0]::]

                    # Fill in L_IGRF where it is -1 in the poes data (in SAA)
                    Linds = np.where(data['L_IGRF'][:]<0)[0]
                    Lfills = swu.get_Lvals(data, inds = Linds)
                    data['L_IGRF'][Linds] = Lfills['Lm'][:]

                    # Divide data into passes
                    # passnums is a list of numbered passes and passInds is the index
                    # of the endpoints of each pass
                    passnums, passInds = pu.getLpass(data['L_IGRF'][:], dist=200, prom=.5)

                    # Find the index of sdate
                    i_last = np.where(data['time'][:] >= pu.unixtime(sdate) * 1000)[0]

                    if len(i_last) > 0:
                        # i_pass1 is the index of the pass just before sdate
                        i_pass1 = np.where(passInds <= i_last[0])[0]

                        # JGREEN 04/15/2021 Fixed this
                        # For 1 file on 2017 09/15 the first half of data is flags so there
                        # are no passinds< i_last and i_pass1 is [] empty.
                        if len(i_pass1)>0:
                            # If there is a pass right before sdate then start there
                            sind = passInds[i_pass1[-1]] # This is the index of the pass just before start
                        else:
                            # If not then start with the first pass after sdate
                            sind = passInds[0]

                        # Now adjust the dictionary so that it has data from the last pass onward
                        # because binning is slow with lots of data
                        for key in data.keys():
                            data[key]=data[key][sind::]

                        # ------------------------- Bin data by pass and L -----------------
                        # The input for the NN is a full pass of poes data in .25 L bins
                        Lbins = np.arange(1,8.5,.25) # These are the Lbins

                        # Returns a dictionary with 4 channels of data binned by pass and Linds
                        # When the data is binned, a new time column is created called 'time_pass'
                        # That has the average time of the pass not including nans
                        # That is important because sometimes the pass is short
                        binned_data = swu.make_Lbin_data(data, Lbins, Lcol = 'L_IGRF', vars = channels)

                        # ------------------------- Get the Kp data ---------------------
                        # Todo Update this so it can get Kp from CCMC
                        # The NN needs Kp as an input
                        # Get the start and end values to get Kp data as datetimes
                        Kp_sdate = pu.unix_time_ms_to_datetime(binned_data['time_pass'][0])
                        Kp_edate = pu.unix_time_ms_to_datetime(binned_data['time_pass'][-1])

                        # Kp data comes from two sources.
                        # Recent data: from potsdam text file with last 28 days
                        # Older data: from omnidata website because it has historical data but not real time

                        if sdate > (dt.datetime.now() - dt.timedelta(days=24)):
                            # Get the Potsdam data and return
                            # Kpdate(datetime), Kpdata (Kp*10), and Apdata, Kpmax

                            mdays = 3
                            Kpdate, Kpdata, Kpmax, Apdata = du.get_Kp_rt(sdate, sdate + dt.timedelta(days=1), days = mdays)
                            # Todo add a check here if there is no omni data returned
                            # In that case, I think we should flag the Kp data and set it to -99
                            Kptime1 = pu.unixtime(Kpdate)
                            Kptime = [1000 * co for co in Kptime1] # Change to msec like poes time

                            # Check get_Kp_rt to see what is returned if no data is present
                            # Make len of Kpdata equal to the length of the POES data
                            if type(Kpdata) is list:
                                if not Kpdata:
                                    Kpdata = np.ones(len(binned_data['time_pass'][:])) * -99

                            elif type(Kpdata) is np.ndarray:
                                if not Kpdata.tolist():
                                    Kpdata = np.ones(len(binned_data['time_pass'][:])) * -99

                            if type(Kpmax) is list:
                                if not Kpmax:
                                    Kpmax = np.ones(len(binned_data['time_pass'][:])) * -99

                            elif type(Kpmax) is np.ndarray:
                                if not Kpmax.tolist():
                                    Kpdata = np.ones(len(binned_data['time_pass'][:])) * -99

                            # Interpolate to the POES pass times; make as long as binned_data time variable
                            Kpnew = np.interp(binned_data['time_pass'][:], Kptime, Kpdata)
                            Kpfin = np.floor(Kpnew)
                            binned_data['Kp*10'] = Kpfin

                            # This is the max Kp in 3 days
                            Kpnew = np.interp(binned_data['time_pass'][:], Kptime, Kpmax)
                            Kpfin = np.floor(Kpnew)
                            binned_data['Kp*10_max_'+str(mdays)+'d'] = Kpfin
                        else:
                            # Get Kp from omnidata
                            mdays = 3 # Number of days for finding max Kp
                            omvars = ['Kp*10','Kp*10_max_'+str(mdays)+'d'] # variables to interpolate
                            # Retunrs a dict with ['time','Kp*10','Kp*10_max_3d']
                            Kp = du.get_omni_max(Kp_sdate, Kp_edate, [omvars[0]],days=mdays)

                            # Todo add a check here if there is no omni data returned
                            # In that case, I think we should flag the Kp data and set it to -99
                            Kptime1 = pu.unixtime(Kp['time'])
                            Kptime = [1000 * co for co in Kptime1]
                            if len(Kp.get('Kp*10')) == 0:
                                Kp['Kp*10'] = np.ones(len(binned_data['time_pass'][:])) * -99
                            if len(Kp.get('Kp*10_max_3d')) == 0:
                                Kp['Kp*10_max_3d'] = np.ones(len(binned_data['time_pass'][:])) * -99

                            # Interpolate to the pass times
                            for omco in range(0, len(omvars)):
                                Kpnew = np.interp(binned_data['time_pass'][:], Kptime, Kp[omvars[omco]])
                                Kpfin = np.floor(Kpnew)
                                binned_data[omvars[omco]]=Kpfin

                        # Add the satellite name to the dict;
                        binned_data['sat'] = [sat for x in range(0,len(binned_data['time_pass'][:]))]

                        binned_data['Lbins'] = Lbins[0:-1] # We use one less because the binning needs the edge value
                        bin_dims = ['time_pass','Lbins']

                        # -------------- map binned data------------
                        # m02 at 20 deg is what everything gets mapped to
                        # mapping uses the cdf file with the end year closest to this year

                        if sat=='m02':
                            # If the sat is m02 there is only one cdf file because it doesn't move in LT
                            # so always use a fixed year
                            myear = 2018
                        else:
                            myear = sdate.year

                        # m02 is also the reference satellite
                        ref_syear= 2014
                        ref_eyear = 2018

                        # This bit switches time_pass to time
                        print('Mapping data')

                        map_data = swu.map_poes(binned_data, channels, sat, sat_ref, ref_ind, cdf_dir, myear, ref_syear, ref_eyear)
                        print('Done mapping data')

                        # --------------------- Now do NN if requested ----------------------------
                        if neural:
                            # Energies, Bmirrors, Ls based on Seths analysis
                            Energies=np.arange(200.,3000.,200.)
                            Ls = np.arange(3.,6.3,.25)

                            Bmirrors = [np.floor(2.591e+04*(L**-2.98) )for L in Ls ]

                            print('Doing nn')
                            outdat = swu.run_nn(map_data,channels,binned_data['Kp*10'][:],binned_data['Kp*10_max_'+str(mdays)+'d'][:],
                               out_scale,in_scale,m, L = Ls, Bmirrors = Bmirrors, Energies = Energies)
                            print('Done with nn')

                            # Save the model name with the data
                            hdf5name = os.path.basename(hdf5File[0])

                        # --------------- Write the data ------------------------
                        # Data may go to a dbase, files at S3 bucket or locally
                        # Todo Update this to write to a dbase
                        print('Writing data')
                        if configfile is not None:
                            if dbase_type =='S3':
                                # Right now this writes nn data to S3 bucket I think?
                                # The S3 bucket is not setup for just inputs
                                finish = swu.write_shells_netcdf(outdir, outdat, Ls, Bmirrors, Energies, sat, hdf5name, cdict)
                            else:
                                if neural:
                                    pass
                                else:
                                    # If were writing shells inputs to the dbase
                                    # map_data has time and channels in a dict
                                    finish = write_shells_inputs_dbase(cdict,map_data,channels,sat,logger)
                        print('Done writing data')
                    else:
                        # Don't crash the program if data is missing but log a msg
                        msg = 'No data for ' + sat + ' ' + sdate.strftime("%Y-%m-%d")
                        logging.warning(msg)
                else:
                    # Don't crash the program if data is missing but log a msg
                    # Todo: Just a log a message of it is not real time
                    # I don't think we want to do this for real time.
                    # all the satellites will not update every 5 minuts
                    if ~realtime:
                        msg = 'No data for '+sat+' '+sdate.strftime("%Y-%m-%d")
                        logging.warning(msg)
                sdate = (sdate+dt.timedelta(days=1)).replace(hour = 0, minute=0, second=0, microsecond=0)

    except Exception as e:
        # If there are any exceptions then log the error
        print(e)
        logging.error(e)



if __name__ == "__main__":
    '''
    PURPOSE: To collect POES/MetOP data from NCEI/NGDC and process it to make 
    just shells input format. If requested, it will also apply the neural network
    to make SHELLS electron flux files at the equator
    
    The code can be run in a realtime or reprocessing mode.
    Real time mode (-rt) would be run on a cron and pulls and processes data from
    the last date so that the processing continually updates.
    Reprocessing mode is run with a set start and end time.
    
    :param: -s --startdate  (str, Not required)
            Date to start collecting data, format YYYY-mm-dd or "YYYY-mm-dd H:M:S"
            (Only needs to be passed if not realtime mode (reprocessing))
            
    :param: -e --enddate (str, Not required)
            Date to start collecting data, format YYYY-mm-dd or "YYYY-mm-dd H:M:S"
            (Only needs to be passed if not realtime mode (reprocessing))
            
    :param: -rt --realtime (str, True or False, Not required))
            If -rt is passed, (real time mode) then the code checks the processed
            data (in a dbase, S3 bucket or local files)
            for last time data was processed and uses that as the start time and the current utc
            time as the end time (start and end time do not need to be passed as inputs)
            
    :param: -nn --neural (str, True or False, not required)
            If -nn is uses then the neural network will be applied to the data to create a
            final output of elecgtron flux at the equator. If not, then the poes data are
            processed into needed shells inputs without applying the nn
            Default False
            
    :param: -ld --localdir (str, Not required)
            Local directory of POES data. In most cases, data will be pulled remotely.
            But if localdir is passed the code will look for POES data locally instead of remotely
    
    :param: -od --outdir (str, Not required)
            Local directory to put output files if not stored elsewhere. If a config file
            is passed the output is stored in dbase or S3
            Default ../SHELLS/
            
    :param: -cdf --cdfdir (str, required)
            Directory where the cumulative distribution data files are located that are needed
            to map POES data to a fixed lon and hemisphere
            Default ../SHELLS/cdf/
            
    :param: -ns --noaasite (str, Not required)
            This is the noaa website with data. 
            Required in rt mode and currently should be satdat.ngdc.noaa.gov
            
    :param: -sa --sats (multiple strings)
            List of satellite data to process, i.e. n15 n16 n17
            Default=['n15','n18','n19','m01','m02','m03']
            
    :param: -v --vars (multiple strings)
            List of all variables to get from the POES data files
            Default=['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT', 'mep_ele_tel90_flux_e1',
             'mep_ele_tel90_flux_e2','mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']
                               
    :param: -ch --channels (multiple strings)
            List of particle flux channels to work with
            Default=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2','mep_ele_tel90_flux_e3', 
            'mep_ele_tel90_flux_e4']
            
    :param: -m --model (str, YYYYMMDD) - The instance of the neural network to use in processing. 
            Default is None. If this is the case, the most recent instance of the neural network will
            be used.

    :param: -md --modeldir (str) - The path (relative or absolute) to the keras model files identified
            in the model parameter. Default from command line is None, which will invoke a search
            in the current working directory. (i.e. ./.)
            
    :param: -log --logfile (str)
            location and name of logfile
            default = './process_POES_inputs' (year and month will be added to whatever is passed)       
            
    :param: -c --config (str) 
            Name of the config file with database or S3 bucket info. Default is None. If it is passed
            then it is assumed that the output data are at S3 or in dbase. If it is not passed, 
            then it is assumed that output data are stored as local files.
    
    USAGE:
    It is expected that this will run in rt at AWS or CCMC on a cron, log any exceptions,
     and send an email if any errors. Reprocessing mode would be used to backfill past data
    
    AWS/CCMC Real time SHELLS mode (POES data is always pulled from NOAA)
    -----------------------------------------
    python process_SHELLS_inputs.py -rt -ns satdat.ngdc.noaa.gov 
        -sa n15 n18 n19 m01 m02 
        -cdf /efs/shells/live/cdfdata/
        -log /efs/shells/logs/process_SHELLS 
        -c /efs/shells/live/config_shells.ini
    
    Reprocessing mode (POES data pulled from NOAA)
    -----------------------------------------
    (This automatically creates a logfile in the current directory)
    python process_SHELLS_inputs.py -s 2015-01-01 -e 2015-12-31 -ns satdat.ngdc.noaa.gov
        -sa n15 n18
        -cdf /Users/janet/PycharmProjects/SHELLS/cdfdata
            
    Reprocessing mode (POES data from local files)
    python process_SHELLS_inputs.py -s 2001-01-01 -e 2002-01-01 
    -ld /Users/janet/PycharmProjects/data/sem/poes/data/processed/ngdc/
    -sa n15 n16 n18 n19 m01 m02

    -----------------------------------------
    # Changes
    # 10/19/2020 JGREEN Added outdir so that output could be written to a file.
    # 09/2021 JGREEN Made significant changes so that data could be accessed at S3 bucket
    # 12/2022 JGREEN Mad significant changes so that data could be accessed from sql database

    '''
    #  PARSE COMMAND LINE ARGUMENTS

    #-------------------------------------------------------------------
    #           GET COMMAND LINE ARGUMENTS
    #-------------------------------------------------------------------
    parser = argparse.ArgumentParser('This program gets POES data from NOAA and processes it for SHELLS')
    parser.add_argument('-s', "--startdate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=False,
                        default = None,
                        type=valid_date)
    parser.add_argument('-e', "--enddate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=False,
                        default = None,
                        type=valid_date)
    parser.add_argument('-rt',"--realtime",action='store_true', default =False)
    parser.add_argument('-nn', "--neural", action='store_true', default=False)
    parser.add_argument('-ld', "--localdir",
                        help="The local directory for looking for files",
                        required=False)
    parser.add_argument('-od', "--outdir",
                        help="The local directory to put the output files",
                        required=False, default = os.path.join(os.getcwd(),'SHELLS'))
    parser.add_argument('-cdf', "--cdfdir",
                        help="The local directory with the cdf files",
                        required=False, default = os.path.join(os.getcwd(),'SHELLS','cdf'))
    parser.add_argument('-ns', "--noaasite",
                        help="The remote noaa site to get the noaa files",
                        required=False)
    parser.add_argument('-sa', "--sats",
                        help="The sat names to get",
                        required=False,
                        default=['n15','n16','n17','n18','n19','m01','m02','m03'],
                        nargs='+')
    parser.add_argument('-v', "--vars",
                        help="The variables to get",
                        required=False,
                        default=['time','alt','lat','lon','L_IGRF','MLT','mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                               'mep_ele_tel90_flux_e4'],
                        nargs='+')
    parser.add_argument('-ch', "--channels",
                        help="The variables to process ",
                        required=False,
                        default=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                           'mep_ele_tel90_flux_e4'],
                        nargs='+')
    parser.add_argument("-m", "--model",
                        help="The Keras model instance to use",
                        required=False,
                        type=str,
                        default=[None],
                        nargs=1)
    parser.add_argument("-md", "--modeldir",
                        help="The directory containing the Keras models used in processing",
                        required=False,
                        default=[None],
                        nargs=1)
    parser.add_argument('-log',"--logfile",
                        help="The full directory and logfile name",
                        required=False,
                        default = os.path.join(os.getcwd(),'SHELLS','process_SHELLS_'))
    parser.add_argument('-c', "--config",
                        help="The full directory and name of the config file",
                        default = os.path.join(os.getcwd(),'shells_config.ini'),
                        required=False)
    args = parser.parse_args()


    #----------------------------------------------------------------

    x = process_SHELLS(sdate_all=args.startdate, edate=args.enddate, realtime=args.realtime, doNN=args.neural,localdir=args.localdir, outdir=args.outdir, cdfdir=args.cdfdir, noaasite=args.noaasite, sats=args.sats, vars=args.vars, channels=args.channels, model=args.model[0], modeldir=args.modeldir[0], logfile=args.logfile, configfile=args.config)
