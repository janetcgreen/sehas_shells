import argparse
import datetime as dt
import glob
import logging
import os
import sqlite3 as sl

import keras
import mysql.connector
import numpy as np
import pandas as pd
from hapiclient import hapi
from hapiclient.hapitime import hapitime2datetime
from joblib import load


# ----------------------- Basic functions ----------------------------
# --------------------------------------------------------------------
def qloss(y_true, y_pred):
    qs = [0.25, 0.5, 0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true - y_pred
    v = np.maximum(q * e, (q - 1) * e)
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


def setupLogging(logfile):
    '''
    PURPOSE: To set up the root logger for tracking issues
    all other modules will add to this logger when setup this way
    :param logfile:
    :return:
    '''

    try:
        # Create the output format
        format = '%(asctime)s:%(levelname)s:%(filename)s:%(lineno)s:  %(message)s'
        # Set up the root logger to write to file
        logging.basicConfig(filename=logfile, format=format, level=logging.INFO)
        # Check if a console handler for root already exists
        l = logging.getLogger()
        if len(l.handlers) < 2:
            # Then add the console too
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            # set a format which is simpler for console use
            formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler to the root logger
            logging.getLogger().addHandler(console)

    except Exception as err:
        raise (err)

    return


def connectToDb(cdict, outdir=None):
    '''
    PURPOSE: To connect to a mysql or sqlite dbase
    :param cdict (dict): dictionary with needed infor for connecting to dbase
    :param outdir (str): The directory if it is an sqlite dbase
    :return conn: the dbase connection object

    # For testing we use an sqlite database that has
    # a different connection process because you don't need a user, pass or host
    # So I added a second try except for testing
    '''
    conn = None

    try:
        # If its a normal sql dbase you need all these
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
            conn = sl.connect(os.path.join(outdir, cdict['dbase']))
        except Exception as err:
            logging.exception(err)
            if conn is not None:
                conn.close()

    return conn


def get_start_rt(outdir, cdict, sat):
    '''
    PURPOSE: to get the start date for real time processing by checking the last data
    # in the sqlite dbase
    :param outdir (str): either the local dir or one that is passed with the location of the
            sqlite dbase
    :param cdict (dict): dictionary with info for connection to dbase
    :param sat (str): the satellite data to lok or
    :return sdate (datetime):

    '''
    # Create a satellite info instance
    satinfo = swu.poes_sat_sem2(sat)
    satId = satinfo.satid()  # Get the satID

    # ----------- Connect to dbase --------------
    # This will connect to sql or sqlite
    try:
        conn = connectToDb(cdict, outdir)
        cursor = conn.cursor()

        if satId is not None:
            # This is annoying because sqlite and mysql use different parameter specification
            query = "SELECT max(time) from " + cdict['tblname'] + " WHERE satId=" + str(satId)
            # cursor.execute(query,(satId,))
            # Todo check if this works with data in there
            cursor.execute(query)
            stime = cursor.fetchone()

            if stime[0] is None:
                sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Todo check that this is giving the right time
                sdate = dt.datetime.utcfromtimestamp(stime[0])
        else:
            sdate = None
            logging.error('No satId for sat=' + sat)
            raise Exception("No satId")

    except Exception as err:
        sdate = None
        logging.error('Problems connecting to dbase' + str(err))

    return sdate


def get_start_rt_hapi(cdict, sat):
    '''
    PURPOSE: to get the last processed data from the CCMC iswa HAPI server
    If the current day is outside of the satellite mission then sdate is None
    or if it can't get data for some reason then sdate is None
    :param cdict (dict):
        A config file dict with info for accessing the processed shells
        input data at ISWA that includes the server name and shells_data table name
    :param sat (string):
        name of poes/metop sat to get data for, i.e. 'n15'
    :param logger:
    :return sdate (datetime): T
        The time of the last processed POES data
    '''
    # The CCMC ISWA database uses a HAPI server
    # These are the required inputs
    server = cdict['server']
    dataset = cdict['dataset']
    parameters = sat

    # For real time we need the last time processed for the satellite of interest.
    # First check if the satellite was even operating on this day
    # We want to avoid repeatedly checking for data from a satellite that is not operational

    satinfo = swu.poes_sat_sem2(sat)  # Create a poes sat class instance
    satstart = satinfo.sdate()  # Get the date the sat became operational

    # If the sat is still running edate is None
    satend = satinfo.edate()
    if satend is None:
        satend = dt.datetime.utcnow() + dt.timedelta(days=1)

    thisday = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # If the sat is operational for the date requested then proceed
    # Otherwise return None if outside of operational dates
    if ((thisday >= satstart) & (satend > thisday)):
        # We need the last data processed in real time in the CCMC dbase
        # so check if there is any in the last two days
        start = (thisday - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
        stop = thisday.strftime('%Y-%m-%dT%H:%M:%S')
        # The ISWA dbase will have time, 39 eflux values and sat
        # We just need time and sat to get the last processed data
        parameters = 'Time,sat'
        opts = {'usecache': True, 'method': 'numpy'}
        try:
            # If no data this call returns []
            data, meta = hapi(server, dataset, parameters, start, stop, **opts)
            # If we expect the satellite should have data but no real time data exists yet
            # Then start with the current day
            # If the real time processing is down for more than 7 days it will
            # have to be filled in
            if len(data['Time']) < 1:
                sdate = thisday
            else:
                # This has tzinfo
                inds = np.where(data[sat] == sat)
                if len(inds) > 0:
                    sdate = hapitime2datetime(data['Time'])[-1]
                else:
                    sdate = thisday
        except Exception as err:
            # If there is no table etc the hapi call will raise an error so
            # we set that to None and log the issue
            sdate = None
            logging.error('Problems connecting to hapi:' + str(err))

    else:
        sdate = None

    return sdate


def get_start_rt_text(cdict, sat, outdir):
    '''
    PRUPOSE: to get the date to start processing data in real time mode by
    checking an already processed archive of data with daily json or csv files
    The start date will be sat dependent because data are downlinked at different
    times

    :param cdict (dict): dictionary with information like the output_type
    :param sat (str): Should be like 'n15'
    :param outdir (str): directory to look for data, Default is current dir
    :return: sdate
    '''
    # For real time we need the last time processed for the satellite of interest.
    # First check if the satellite was even operating on this day
    # We want to avoid repeatedly checking for data from a satellite that is not operational

    dformat = '%Y-%m-%dT%H:%M:%S.%fZ'
    satinfo = swu.poes_sat_sem2(sat)  # Create a poes sat class instance
    satstart = satinfo.sdate()  # Get the date the sat became operational

    # Get the date the sat stopped
    # If the sat is still running edate is None
    satend = satinfo.edate()
    if satend is None:
        satend = dt.datetime.utcnow() + dt.timedelta(days=1)

    thisday = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # If the sat is operational for the date requested then proceed
    # Otherwise return None if outside of operational dates
    if ((thisday >= satstart) & (satend > thisday)):
        # We need the last data processed in real time from the archive of
        # daily json or csv files in outdir so make a list of files
        # Files will be called fname+'_YYYYMMDDTHHMMSS.' input_type (either json or csv)
        # Check if there is a daily file started already
        fout = os.path.join(outdir, '**', cdict['fname'] + '*.' + cdict['input_type'])
        flist = glob.glob(fout, recursive=True)
        if len(flist) < 1:
            # If np files yet then start with today
            sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Open the last file and look for any data for that sat
            flist.sort(reverse=True)
            ltime = None
            fco = 0
            if len(flist) > 2:
                fend = 2
            else:
                fend = len(flist)
            while ((fco < fend) & (ltime is None)):
                lastdata = pd.read_csv(flist[fco]).to_dict(orient='list')
                satIDs = np.array(lastdata['satID'])
                inds = np.where(satIDs == satinfo.satid())[0]
                if len(inds) > 0:
                    ltime = dt.datetime.strptime(lastdata['time'][inds[-1]], dformat)
                fco = fco + 1
            if ltime is None:
                # If there are no old files then start with today
                sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                #
                sdate = ltime

    else:
        # IF the sat is not operational then set sdate to None
        # so that it doesn't check for new data
        sdate = None
    return sdate


def get_kp_data_iswa(Kp_sdate, Kp_edate, iswaserver, kpdataset):
    '''
    PURPOSE: To get the Kp data from the ISWA HAPI server
    :param Kp_sdate (datetime):
    :param Kp_edate (datetime):
    :param iswaserver (str):
    :param kpdataset (str):
    :return data,meta:
    '''
    start = Kp_sdate.strftime('%Y-%m-%dT%H:%M:%S')
    stop = Kp_edate.strftime('%Y-%m-%dT%H:%M:%S')
    # The ISWA dbase will have time, 39 eflux values and sat
    # We just need time and sat to get the last processed data
    parameters = 'Time,KP_3H'
    opts = {'usecache': True, 'method': 'numpy'}
    try:
        data, meta = hapi(iswaserver, kpdataset, parameters, start, stop, **opts)
    except:
        data = None
        meta = None
        logging.error('Cant get to iswa Kp data')

    return data, meta


# ------------------- The main process_SHELLS function--------------------------
# ===========================================================================

def process_data(sdate=None, edate=None):
    try:
        # ---------------- Set up mapping info -------------------

        # ----------------- Set up nn if needed --------------------
        # load in transforms,model for Neural Network
        # They are always a trio of files: in_scale, out_scale binary files
        # and a model_quantile HDF5 file.
        #

        print(os.environ.get('OUT_SCALE_FILE'))

        out_scale = load(os.environ.get('OUT_SCALE_FILE'))
        in_scale = load(os.environ.get('IN_SCALE_FILE'))
        m = keras.models.load_model(os.environ.get('HDF5FILE'), custom_objects={'loss': qloss}, compile=False)

    except Exception as e:
        # If there are any exceptions then log the error
        print(e)
        logging.error(e)
