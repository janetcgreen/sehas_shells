import argparse
import csv
import datetime as dt
import json
import os
import re
import requests
import sys
from collections import OrderedDict

import numpy as np

from dotenv import load_dotenv

app_path = os.path.join(os.path.dirname( __file__ ), '..','Docker')
sys.path.insert(0, app_path)  # take precedence over any other in path
import process_inputs as pi
from app import create_app
from flask import current_app

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

def make_times(tsdate, tedate, tstep):
    t = tsdate
    times = []
    while t < tedate:
        times.append(t)
        t = t + dt.timedelta(minutes=int(tstep))

    return times

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

def fixed_files_shells(sdate=None,edate=None,realtime=False,
                    Es=list(np.arange(200,3000,200)), Ls=list(np.arange(3,6.3,.5)), tstep=60,
                       outdir = os.getcwd(),testing=False):
    '''
    PURPOSE: Returns the SHELLS electron fluxes at near equatorial Bmirror points
    for a fixed set of Ls and energies at a user chosen time cadence
    :param sdate (datetime): start date to process data
    :param edate (datetime): end date to process data
    :param realtime (0 or 1): flag for real time processing
    :param Es (list): list of electron energies (keV) to process
    :param Ls (list): list of L values to return the electron flux at
    :param tstep (int): time cadence of the returned data (minutes)
    :param outdir (str): location for the output files
    :return:
    '''
    # Set Bmirrors
    Enew = [float(x) for x in Es] # turn the passed energies string into floats

    # Calculate the near equatorial Bmirror value for the input Ls
    Bm = [np.floor(2.591e+04 * (L ** -2.98)) for L in Ls]

    # If it's running in real time then always start at the current utc day
    # otherwise use the start and end passed as inputs
    if realtime:
        # Get the current days start at hour 00:00:00
        sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        # Get the current time
        edate = dt.datetime.utcnow()
        last_edate=edate
    else:
        last_edate=edate
        edate=(sdate+dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sdate=sdate.replace(minute=0, second=0, microsecond=0) # Force it to start on an hour

    tform = '%Y-%m-%dT%H:%M:%S.%fZ'# Time format to work with

    # Step through in day increments
    print("Running data")
    while sdate<=last_edate:
        # Create and format the time array between sdate and edate at timestep tstep
        times = make_times(sdate, edate, tstep) # Returns a list of datetimes
        times_form = [time.strftime(tform) for time in times] # reformat time string for app

        # Specify the json output file name
        fname = f"shells_fixed_{times[-1].strftime('%Y%m%d')}.json"

        # This uses the code from the app but since we do not need the
        # magephem service we don't actually have to call it from the service.
        # We need to use functionality that requires an application context
        # Set up an application context with app.app_context()
        # to avoid RuntimeError: working outside of application context
        # app = create_app(test_config="test_config")
        print("Starting app")
        if testing==True:
            app = create_app(test_config="test_config")
        else:
            app = create_app()
            print("Created app")
        app.app_context().push()

        #print("Push contect")
        # In the normal shells usage we have a list of Ls and Bms for each time
        # that are len(pitchangles)
        # In this case we have to replicate the L shells for each time

        Lvals =[Ls] * len(times_form)
        Bvals = [Bm] * len(times_form)
        shdata,res_code = pi.process_data(times_form, Lvals, Bvals, Enew)

        if res_code ==200:
            outdata = OrderedDict()
            outdata["time"] = times_form
            for key in shdata:
                outdata[key] = shdata[key]

            with open(os.path.join(outdir,fname), "w") as outfile:
                json.dump(outdata, outfile)
        sdate = sdate+dt.timedelta(days=1)
        edate = sdate + dt.timedelta(days=1)

    print("Done")

if __name__ == "__main__":
    '''
    PURPOSE: To create an updating dataset of daily files with SHELLS electron fluxes
     at fixed L and Bmirror locations:
     
    It does the following tasks:
    1) Gets the POES input data from the CCMC Hapi dbase. 
       (the current day up to the current time.)   
    2) Runs the SHELLS neural network directly by calling process_inputs (not the app)
       that calls run_nn() using 
       Energies = np.arange(200,3000,200),
       Ls = np.arange(3,6.3,..5) and 
       Bmirrors = [np.floor(2.591e+04*(L**2.98)) for L in Ls] and 
       times depend on the input cadence.
    3) Writes the output to a daily file that updates in time shells_fixed_YYYYMMDD.csv         
    '''

    parser = argparse.ArgumentParser('This program outputs daily files of SHELLS electron fluxes'
            'at fixed L and Bmirrors near the equator ')
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

    parser.add_argument('-es', "--energies",
                        help="List of energies",
                        nargs='+',
                        required=False,
                        default=list(np.arange(200,3000,200)))
    parser.add_argument('-ls', "--lshells",
                        help="List of L shells",
                        nargs='+',
                        required=False,
                        default=list(np.arange(3,6.3,.5)))
    parser.add_argument('-c', "--cadence",
                        help="The time cadence in minutes",
                        required=False,
                        default=60)
    parser.add_argument('-od', "--outdir",
                        help="The local directory to put the output files",
                        required=False, default=os.getcwd())
    parser.add_argument('-t', "--testing", action='store_true', default=False)

    args = parser.parse_args()

    #----------------------------------------------------------------

    x = fixed_files_shells(sdate=args.startdate,edate=args.enddate,realtime=args.realtime,
                           Es=args.energies, Ls=args.lshells, tstep=args.cadence,
                           outdir=args.outdir,testing=args.testing)