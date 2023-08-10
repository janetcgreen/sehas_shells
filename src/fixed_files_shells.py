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
                       outdir = os.getcwd()):
    # Set Bmirrors
    Enew = [int(x) for x in Es]
    Bm = [np.floor(2.591e+04 * (L ** 2.98)) for L in Ls]

    #If its running in real time then start at the current day
    # otherwise use the start and end times given
    if realtime:
        # Get the current date
        sdate = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        # Get the current time
        edate = dt.datetime.utcnow()
        last_edate=edate
    else:
        last_edate=edate
        edate=(sdate+dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sdate=sdate.replace(minute=0, second=0, microsecond=0) # Force it to start on an hour

    #edate = sdate+dt.timedelta(hours=24)
    tform = '%Y-%m-%dT%H:%M:%S.%fZ'
    # Step through in day increments
    while sdate<=last_edate:
        # Create and format the times array between sdate and edate at timestep tstep

        times = make_times(sdate, edate, tstep)
        times_form = [time.strftime(tform) for time in times]
        #print(times_form)

        # Specify the CSV file name
        fname = f"shells_fixed_{times[-1].strftime('%Y%m%dT%H%M')}.json"

        # We need to use functionality that requires an application context
        # Set up an application context with app.app_context()
        # to avoid RuntimeError: working outside of application context
        app = create_app(test_config="test_config")
        app.app_context().push()

        # In the normal shells usage we have a list of Ls and Bms for each time that are len(pitchangles)
        # In this case we have to replicate the L shells for each time

        Lvals =[Ls]*len(times_form)
        Bvals = [Bm] * len(times_form)
        shdata = pi.process_data(times_form, Lvals, Bvals, Enew)

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
    PURPOSE: This is a python script that can be run every hour with cron that will:
    1) get the POES input data from the CCMC Hapi dbase from the last hour (use the sqlite dbase for now). 
       Get the current day up to the current time.   
    2) run the neural network directly (not calling the app) with process_inputs that calls run_nn() using 
       Energies = np.arange(200,3000,200),
       Ls = np.arange(3,6.3,..5) and 
       Bmirrors = [np.floor(2.591e+04*(L**2.98)) for L in Ls] and 
       times that are the same as the POES_input data from the dbase.
    3) write the output to a daily file that updates in time shells_fixed_YYYYMMDDTHHMM.csv 
       where the second one is the time it was written        
    '''

    parser = argparse.ArgumentParser('This program runs every hour with cron '
                                     'and writes the output to a daily file')
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

    args = parser.parse_args()

    #----------------------------------------------------------------

    x = fixed_files_shells(sdate=args.startdate,edate=args.enddate,realtime=args.realtime,
                           Es=args.energies, Ls=args.lshells, tstep=args.cadence,
                           outdir=args.outdir)