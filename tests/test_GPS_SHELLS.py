import datetime as dt
import unittest
import numpy as np
import os
import sys
# This adds ../src to the path so they can be imported when
# running tests from the tests directory
func_path = os.path.join(os.path.dirname( __file__ ), '../src')
sys.path.insert(0, func_path)
import make_GPS_shells as mgs
import shells_web_utils as swu
import glob
import json
from datetime import timezone
import pandas as pd
import tempfile
import csv

#sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
#sys.path.insert(1, '/efs/spamstaging/live/chargehaz/')
#import shells_web_utils as swu

class test_GPS_shells(unittest.TestCase):
    # PURPOSE: To test make_GPS_shells that creates shells output and plots
    # of electron fluxes along a GPS satellite

    def setUp(self):
        #--------------------------------------------
        # SETUP:
        # None yet

        # ---------------------- Define some tables in configfile---------------------

        pass

    def tearDown(self):
        # ------------------------------------------------
        # Teardown:
        # None yet
        pass

    #===================================================================
    # These are the tests that run
    # TEST 1: Check that get_TLES returns a value
    # TEST 2: Check that get_TLES returns none for bad request
    # TEST 4: Check that data is added to the output file


    def test_B_check_tles(self):
        #===============================================================
        # TEST 1: Check that get_TLES returns a value
        #==============================================================

        print('*** TEST 1: Check that get_TLES returns a value')
        group = "GPS-OPS"
        format = "TLE"
        celes_url = "https://celestrak.org/NORAD/elements/gp.php"
        sat = "PRN 32"
        tle1, tle2 = mgs.get_TLES(celes_url, group, format, sat)
        if tle1 is not None:
            success = 1

        self.assertEqual(1,success)

    def test_B_check_no_tles(self):
        #===============================================================
        # TEST 2: Check that get_TLES returns none for bad request
        #==============================================================

        print('*** TEST 2: Check that get_TLES returns none for bad request')
        group = "GPS-OPS"
        format = "TLE"
        celes_url = "https://celestrak.org/NORAD//gp.php"
        sat = "PRN 325"
        tle1, tle2 = mgs.get_TLES(celes_url, group, format, sat)
        if tle1 is None:
            success = 1

        self.assertEqual(1,success)

    def test_B_check_new_output(self):
        #===============================================================
        # TEST 3: Check that a file with the last 7 days is created when none exists
        #==============================================================

        # First delete any existing test output
        sat = "PRN 32"
        odir = os.getcwd()
        ndays = 2
        tstep=5
        oname = 'GPS_SHELLS_test_'
        fname = oname+str(ndays)+'day.txt'
        file = glob.glob(os.path.join(odir,fname))

        # If a test file exists then delete it
        if len(file)>0:
            os.remove(file[0])

        sh_url = 'http://172.17.0.3:5005/shells_io/'

        # This should create the file s.path.join(odir,fname)
        mgs.make_GPS_shells(None, None, sat, sh_url, realtime=1, tstep=tstep, ndays=ndays,
                        Es=[500, 2000], outdir=os.getcwd(),outname =oname, testing=1)

        file = glob.glob(os.path.join(odir, fname))

        if len(file)>0:
            # If the file is there then check that it is the right len
            # and has the right start time
            scheck = dt.datetime.utcnow()
            df = pd.read_csv(file[0])
            odict = df.to_dict(orient='list')
            last = dt.datetime.strptime(odict['time'][-1],'%Y-%m-%dT%H:%M:%S.%fZ')
            tdiff=np.abs(((last - scheck).total_seconds()) / 60)
            nvals = len(odict['time'])
            if (tdiff<10) & (nvals>(ndays*24*60-60)/tstep):
                success=1
        else:
            success=0
        self.assertEqual(1, success)

    def test_A_check_add_output(self):
        #===============================================================
        # TEST 4: Check that data is added to the output file
        #==============================================================
        # For this test, first create and output file. Then subtract some
        # data from it and run it again
        # First delete any existing test output
        sat = "PRN 32"
        odir = os.getcwd()
        ndays = 2
        tstep=10
        oname = 'GPS_SHELLS_test_'
        fname = oname+str(ndays)+'day.txt'
        file = glob.glob(os.path.join(odir,fname))
        # If a test file exists then delete it
        if len(file)>0:
            os.remove(file[0])

        sh_url = 'http://172.17.0.3:5005/shells_io/'

        # This should create the file s.path.join(odir,fname)
        mgs.make_GPS_shells(None, None, sat, sh_url, realtime=1, tstep=tstep, ndays=ndays,
                        Es=[500, 2000], outdir=odir,outname =oname, testing=1)

        file = glob.glob(os.path.join(odir, fname))

        if len(file)>0:
            # subtract a few lines and write out again
            df = pd.read_csv(file[0])
            odict = df.to_dict(orient='list')
            lind = int((len(odict['time'])-120/tstep)) # take off 2 hours

            lasttime = dt.datetime.strptime(odict['time'][lind],'%Y-%m-%dT%H:%M:%S.%fZ')
            ekeys = [x for x in list(odict.keys()) if ('E flux' in x)]
            Bkeys = [x for x in list(odict.keys()) if ('Bmirrors' in x)]
            skeys = ['time', 'L'] + ekeys + Bkeys
            with open(os.path.join(odir, fname), 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(skeys)
                for ico in range(0, lind):
                    row1 = [odict['time'][ico], odict['L'][ico]]  # Time and L
                    row2 = ["{0:.5f}".format(odict[k][ico]) for k in skeys[2::]]
                    row = row1 + row2
                    writer.writerow(row)
            # Then add to the file
            mgs.make_GPS_shells(None, None, sat, sh_url, realtime=1, tstep=tstep, ndays=ndays,
                                Es=[500, 2000], outdir=os.getcwd(), outname=oname, testing=1)

            df = pd.read_csv(file[0])
            odict = df.to_dict(orient='list')

            if dt.datetime.strptime(odict['time'][-1],'%Y-%m-%dT%H:%M:%S.%fZ')>lasttime:
                success=1

        self.assertEqual(1, success)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        # This is here so you can pass things like a config file if needed
        unittest.main()
    else:
        unittest.main()