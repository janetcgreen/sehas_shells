import argparse
import csv
import datetime as dt
import json
import os
import re
import requests
import sys

import numpy as np

from dotenv import load_dotenv

app_path = os.path.join(os.path.dirname( __file__ ), '..','Docker')
sys.path.insert(0, app_path)  # take precedence over any other in path
import process_inputs as pi
from app import create_app

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

def fixed_files_shells(Es=[200,3000,200], Ls=[3,6.3,.5], tstep=60):
    # Set Bmirrors
    Bm = [np.floor(2.591e+04 * (L ** 2.98)) for L in Ls]

    # Get the current date
    sdate = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Get the current time
    edate = dt.datetime.now()

    # Create and format the times array between sdate and edate at timestep tstep
    tform = '%Y-%m-%dT%H:%M:%S.%fZ'
    times = make_times(sdate, edate, tstep)
    times_form = [time.strftime(tform) for time in times]
    print(times_form)

    # Specify the CSV file name
    fname = f"shells_fixed_{edate.strftime('%Y%m%dT%H%M')}.csv"

    # We need to use functionality that requires an application context
    # Set up an application context with app.app_context()
    # to avoid RuntimeError: working outside of application context
    app = create_app(test_config="test_config")
    app.app_context().push()

    shdata = pi.process_data(times_form, Ls, Bm, Es)

    # Open the CSV file in write mode
    with open(fname, mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the dictionary keys as the header
        csv_writer.writerow(shdata.keys())

        # Write the dictionary values as a row
        csv_writer.writerow(shdata.values())

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

    parser.add_argument('-es', "--energies",
                        help="List of energies",
                        nargs='+',
                        required=False,
                        default=[200,3000,200])
    parser.add_argument('-ls', "--lshells",
                        help="List of L shells",
                        nargs='+',
                        required=False,
                        default=[3,6.3,.5])
    parser.add_argument('-c', "--cadence",
                        help="The time cadence in minutes",
                        required=False,
                        default=60)

    args = parser.parse_args()

    #----------------------------------------------------------------

    x = fixed_files_shells(Es=args.energies, Ls=args.lshells, tstep=args.cadence)