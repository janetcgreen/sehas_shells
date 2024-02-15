import datetime as dt
import unittest
import numpy as np
import os
import sys
# This adds ../src to the path so they can be imported when
# running tests from the tests directory
func_path = os.path.join(os.path.dirname( __file__ ), '../src')
sys.path.insert(0, func_path)
import fixed_files_shells as ff
import shells_web_utils as swu
import glob
import json
from datetime import timezone
import pandas as pd
import tempfile
import csv
import netCDF4 as nc4
from bisect import bisect_right


def get_nearest(list_dt, dt):
    """
    Assumes list_dt is sorted. Returns nearest position to dt.

    bisect_left gives the index where the value will be inserted
    so the value of data we want is that index -1
    However, if the times are identical then it gives the index to
    the left

    """

    pos = bisect_right(list_dt, dt)

    if pos == 0:
        return 0
    if pos == len(list_dt):
        return len(list_dt) - 1

    return pos - 1

class test_fixed_shells(unittest.TestCase):
    # PURPOSE: To test make_GPS_shells that creates shells output and plots
    # of electron fluxes along a GPS satellite


    def setUp(self):
        #--------------------------------------------
        # SETUP:
        # None yet
        self.test_dir = tempfile.mkdtemp()
        # ---------------------- Define some tables in configfile---------------------

        pass

    def tearDown(self):
        # ------------------------------------------------
        # Teardown:
        # None yet
        pass



    #===================================================================
    # These are the tests that run
    # TEST 1: Test that files are created between sdate and edate
    # TEST 2: Test that a file is created in real time that starts at 00


    def test_A_reprocess(self):
        #===============================================================
        # TEST 2: Test that files are created between sdate and edate
        # and that they match expected output
        #==============================================================
        odir = self.test_dir
        startt = dt.datetime(2022,1,1)
        endt = dt.datetime(2022, 1, 2)

        # This calls in test mode which creates the app with test_config
        # So it uses the sql dbase as inputs insted of HAPI
        ff.fixed_files_shells(sdate=startt,edate=endt,outdir=odir,testing=True)

        # Look for the files created
        fstart='shells_fixed_'
        fname = fstart+'*.json'
        ofiles = glob.glob(os.path.join(odir, fname))

        # Make a list of the files that should have been created
        tfiles = []
        startf = startt
        while startf<=endt:
            tfiles.append(os.path.join(odir,fstart+startf.strftime('%Y%m%d')+'.json'))
            startf=startf+dt.timedelta(days=1)

        # Check that two files are created with names shells_fixed_20220101
        self.assertEqual(set(tfiles), set(ofiles))

        # Then compare to the previous output
        # This file does not have the exact same times as in the input dbase

        fname = '../Docker/tests/shells_neural20220101.nc'
        # This has 'E flux 200'[57,14] to 'E flux 2800'
        # where Ls are 3-6.25 in .25 incs
        # Did I make shells_neural20220101.nc from the same sqlite dbase
        # The test data is log10(flux)
        # but the shells output is flux now
        test_data = nc4.Dataset(fname, 'r')

        # Create a list of times from the file
        times1 = [dt.datetime.utcfromtimestamp(x / 1000) for x in test_data['time']]

        # Check the time cadence is correct
        ofiles.sort()
        if len(ofiles)>0:
            # If the file is there then check that it matches the nc file
            with open(ofiles[0]) as f:
                sdata = json.load(f)
                times2 = [dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%fZ") for x in sdata['time'][:]]

                # Compare times that are close
                nearest_t = [get_nearest(times1, t) for t in times2]
                tt = [times1[t] for t in nearest_t]
                ndata = np.array(sdata['E flux'][:])

                # Find where times2 is closed to times1 and check that the output is the same
                fsum=0
                for tco in range(0,len(times2)):
                    difft = ((times2[tco]-tt[tco]).total_seconds())/60
                    if (difft<10) & (difft>0):
                        fdif = ((np.log(ndata[tco,0,0])) - test_data['E flux 200'][nearest_t[tco],0])
                        print(fdif,tco)
                        fsum = fsum+fdif
        if fsum<1:
            test=1

        self.assertEqual(test,1)

        print('Here')

    def test_B_real_time(self):
        #===============================================================
        # TEST 2: Test that a file is created in real time that starts at 00
        #==============================================================

        # This calls the file in real time mode
        # and should create a file called shells_fixed for the current date
        # in the temporary test directory

        current_day = dt.datetime.utcnow().strftime('%Y%m%d')
        odir = self.test_dir

        ff.fixed_files_shells(realtime=True,outdir=odir)

        fname = 'shells_fixed_'+current_day+'.json'
        ofile = glob.glob(os.path.join(odir, fname))

        if len(ofile)>0:
            # If the file is there then check that it is the right
            # time cadence and start time
            with open(ofile[0]) as f:
                tdata = json.load(f)
                ltime = dt.datetime.strptime(tdata['time'][-1],"%Y-%m-%dT%H:%M:%S.%fZ")
            scheck = dt.datetime.utcnow()
            if np.abs((ltime-scheck).total_seconds())<60*3600:
                success=1

        else:
            success=0
        self.assertEqual(1, success)




if __name__ == "__main__":
    if len(sys.argv) > 1:
        # This is here so you can pass things like a config file if needed
        unittest.main()
    else:
        unittest.main()