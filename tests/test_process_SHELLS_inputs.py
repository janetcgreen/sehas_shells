import datetime as dt
import unittest
import numpy as np
import os
import sys
import process_SHELLS_inputs as pr
import sqlite3 as sl
from collections import OrderedDict
import glob
import json
from datetime import timezone
import pandas as pd
import tempfile
import shutil

sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
#sys.path.insert(1, '/efs/spamstaging/live/chargehaz/')
import shells_web_utils as swu

class test_process_shells_inputs(unittest.TestCase):
    # PURPOSE: To test process_shells_inputs that gets poes data ready for shells model
    # The default is for my local config files
    # but it will accept any config file name as a command line input
    # To set a different configfile call test_process_shells_inputs ./configlocal.ini
    configfile = os.path.join(os.getcwd(), 'config_test_shells_inputs_local.ini')

    def setUp(self):
        #--------------------------------------------
        # SETUP:
        # 1) read the config file
        # 2) create an sqlite dbase with the needed tables

        # ---------------------- Define some tables in configfile---------------------

        self.test_dir = tempfile.mkdtemp()
        self.cdict,self.dbase_type = swu.read_config(self.configfile,'DEFAULT')
        # The DEFAULT test config has the name of the sqlite dbase
        # and the table names, no password
        # First need to delete the test dbase if it exists
        self.conn = sl.connect(os.path.join(self.test_dir,self.cdict['dbase'])) # Create a dbase called sehas_shells
        cursor = self.conn.cursor()
        # ----------------------- Create tables ------------------
        tables = self.get_shells_tables()  # return a dict with the table definitions

        for name, ddl in tables.items():
            try:
                print("TEST SETUP: Creating table {}: ".format(name))
                cursor.execute(ddl)
            except Exception as err:
                print(err)
            else:
                print("OK")

        print("*** TEST SETUP: Done with tables")

    def tearDown(self):
        # ------------------------------------------------
        # Teardown:
        # 1) Delete the sat 'testsat'
        # 2) delete the channel test_channel
        # 3) Remove the log file
        #--------------------------------------------------
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)


    def get_shells_tables(self):
        '''
        PURPOSE: to create a dict called tables with the table definitions to be
        used for the shells inputs
        :return: tables (dict)

        '''

        tables = OrderedDict()

        # This will have the electron chanel names i.e. 'mep_ele_tel90_flux_e1',
        # 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4'
        #tables['ChannelTbl'] = (
        #    "CREATE TABLE ShellsChannelTbl ("
        #    "  id int NOT NULL ,"
        #    "  channelName char(30) NOT NULL,"
        #    "  PRIMARY KEY (`id`)"
        #    ") ")

        # This will have the satellite names
        # i.e. noaa15, noaa16, noaa17, metop01
        #tables['ShellsSatTbl'] = (
        #    "CREATE TABLE ShellsSatTbl ("
        #    "  id int NOT NULL AUTO_INCREMENT,"
        #    "  satName char(10) NOT NULL,"
        #    "  PRIMARY KEY (`id`)"
        #    ") ")

        # The input table will have the same structure as the csv/json files
        # time, e1 L1, e1 L2, ...e2 L1, e2 L2, ... Kp, Kpmax, satId
        tables['ShellsInputsTbl'] = ("CREATE TABLE ShellsInputsTbl ("
            "  time int NOT NULL,")
        channels = ["mep_ele_tel90_flux_e1","mep_ele_tel90_flux_e2","mep_ele_tel90_flux_e3","mep_ele_tel90_flux_e4"]
        Lbins = np.arange(1, 8.25, .25)
        for channel in channels:
            for L in Lbins:
                colname = '"'+channel+'_L_'+str(L)+'"'
                tables['ShellsInputsTbl']= tables['ShellsInputsTbl']+' '+colname+' double DEFAULT NULL,'

        tables['ShellsInputsTbl'] = tables['ShellsInputsTbl']+'  "Kp*10" int DEFAULT NULL, ' \
                       "Kp_max double DEFAULT NULL, satId int NOT NULL,"\
                        "  PRIMARY KEY (time)" \
                         ") "
        # "  FOREIGN KEY (channelId) REFERENCES ShellsChannelTbl(id),"
        # "  FOREIGN KEY (SatId) REFERENCES ShellsSatTbl(id)"
        # ") ENGINE=InnoDB")
            #"  e1_L1 int NOT NULL,"
            #"  e1_L2 int NOT NULL,"
            #"  e1_L3 int NOT NULL,"
            #"  e1_L4 int NOT NULL,"
            #"  e1_L5 int NOT NULL,"
            #"  e1_L6 int NOT NULL,"
            #"  e1_L7 int NOT NULL,"
            #"  e1_L8 int NOT NULL,"
            #"  satId int NOT NULL,"
            #"  eflux double DEFAULT NULL)")

            #"  PRIMARY KEY (unixTime_utc, channelId, LId),"
            #"  FOREIGN KEY (channelId) REFERENCES ShellsChannelTbl(id),"
            #"  FOREIGN KEY (SatId) REFERENCES ShellsSatTbl(id)"
            #") ENGINE=InnoDB")

        return tables


    def test_A_check_test_start_date(self):
        #===============================================================
        # TEST: Check that test sqlite real time mode returns a valid start date
        #==============================================================
        print('*** TEST: Check that test sqlite real time mode returns a valid start date')
        # This testing config file has
        # dbase = sehas_shells
        # inputstbl = ShellsInputsTbl

        # The testing setup creates self.cdict and self dbase_type
        # This routine will return the start of the last processed data in
        # the previous 10 days for all currently operational sats
        # In this case no data is entered yet so it should return todays
        # date at 0:0:0
        sat = 'n15'
        outdir = self.test_dir
        sdate = pr.get_start_rt(outdir, self.cdict, sat)

        self.assertEqual(sdate, dt.datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0))

    def test_B_check_CCMC_start_date(self):
        #===============================================================
        # TEST: Check that CCMC real time mode returns a valid start date
        #==============================================================
        # Under SHELLS_CCMC this config file has
        # server=https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/
        # shells_data_table = shells_inputs
        print('*** TEST: Check that CCMC real time mode returns a valid start date')

        # Return a dictionary with the items in the config file
        # and the dbase_type i.e. S3,CCMC, etc
        cdict,dbtype = swu.read_config('./config_test_shells_inputs_local.ini','SHELLS_CCMC')

        # This routine will return the start of the last processed data in
        # the previous 2 days for all currently operational sats
        sdate = pr.get_start_rt_hapi(cdict, 'n15')
        # If there are no tables at CCMC ISWA this will return None
        # Once there are tables it should return a valid sdate

        self.assertEqual(sdate, None)

    def test_poes_sat_sem2(self):
        #================================================================
        # TEST: Check that the poes_sat_sem2 class returns a mission start
        #================================================================
        print('*** TEST:Check that the poes_sat_sem2 class returns a mission start')
        satclass = swu.poes_sat_sem2('N16')
        sdate = satclass.sdate()
        self.assertEqual(sdate,dt.datetime(2001,1,10,0,0))

    def test_B_run_shells_reprocess_sqlite(self):
        #==================================================================
        # TEST: Check that data is created as nc file with no config
        #==================================================================
        # Arguments here are sdate,edate,
        # realtime,neural,localdir,outdir,cdfdir
        # noaasite,sat,
        # vars,channels
        # model,modeldir,logfile,config, csection
        #

        print('*** TEST: Check that data is created as nc file with no config')

        outdir = self.test_dir
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        # Run the code that should create a dbase data for 2022/1/1
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, outdir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(outdir,'test_process_shells_'),
                          None,None)
        print('Here')

    def test_B_run_shells_reprocess_sqlite(self):
        #==================================================================
        # TEST: Check that data is added to dbase in reprocessing
        #==================================================================
        # Arguments here are sdate,edate,
        # realtime,neural,localdir,outdir,cdfdir
        # noaasite,sat,
        # vars,channels
        # model,modeldir,logfile,config, csection
        #
        # The configfile has
        # input_type = sqlite
        # dbase = sehas_shells
        # inputstbl = ShellsInputsTbl
        # output_type = csv
        # fname = shells_inputs
        # This should create a file called shells_inputs_20220101HHMMSS.csv
        print('*** TEST:Check sqlite data added when running in reprocessing mode')

        cdict,dbtype = swu.read_config(self.configfile,'SHELLS_TESTING_SQLITE')
        outdir = self.test_dir
        cdfdir = os.path.join( os.getcwd(), '..','SHELLS','cdf' )

        # Run the code that should create a dbase data for 2022/1/1
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, outdir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(outdir,'test_process_shells_'),
                          self.configfile,'SHELLS_TESTING_SQLITE')
        # Open the dbase and check that there is data
        conn = swu.create_conn_sqlite(os.path.join(outdir, cdict['dbase']))
        cursor = conn.cursor()
        query = 'SELECT * FROM '+cdict['tblname']
        cursor.execute(query,)
        rows = cursor.fetchall()
        test = 0
        if len(rows)>50:
            test=1
        self.assertEqual(test,1)

    def test_B_run_shells_reprocess_csv(self):
        #==================================================================
        # TEST: Check csv file creation when running in reprocessing mode
        #==================================================================
        # Arguments here are sdate,edate,
        # realtime,neural,localdir,outdir,cdfdir
        # noaasite,sat,
        # vars,channels
        # model,modeldir,logfile,config, csection
        #
        # The configfile has
        # input_type = sqlite
        # dbase = sehas_shells
        # inputstbl = ShellsInputsTbl
        # output_type = csv
        # fname = shells_inputs
        # This should create a file called shells_inputs_20220101HHMMSS.csv
        print('*** TEST:Check csv file creation when running in reprocessing mode')

        cdict,dbtype = swu.read_config(self.configfile,'SHELLS_TESTING_CSV')
        # Check if a testfile exists and delete it
        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.csv'
        flist = glob.glob(os.path.join(self.test_dir,fbase))
        if len(flist)>1:
            os.remove(flist[0])
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        # Run the code that should create a csv file for 2022/1/1
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, self.test_dir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(self.test_dir,'test_process_shells_'),
                          self.configfile,'SHELLS_TESTING_CSV')

        # Check that the file is created and has a certain length

        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.csv'
        flist = glob.glob(os.path.join(self.test_dir,fbase))
        if len(flist)>0:
            with open(flist[0], 'r') as openfile:
                testfile = pd.read_csv(openfile).to_dict(orient='list')
            if len(testfile['time'])>56:
                testit = 1
            else:
                testit=0
            # Now delete the file
            os.remove(flist[0])
        else:
            testit = 0
        self.assertEqual(1, testit)

    def test_B_run_shells_reprocess_json(self):
        #==================================================================
        # TEST: Check json file creation when running in reprocessing mode
        #==================================================================
        # Arguments here are sdate,edate,
        # realtime,neural,localdir,outdir,cdfdir
        # noaasite,sat,
        # vars,channels
        # model,modeldir,logfile,config, csection
        #
        # The configfile has
        # input_type = sqlite
        # dbase = sehas_shells
        # inputstbl = ShellsInputsTbl
        # output_type = json
        # fname = shells_inputs
        # This should create a file called shells_inputs_20220101HHMMSS.json
        print('*** TEST: Check json file creation when running in reprocessing mode')

        # Read the config file to get the filename
        cdict,dbtype = swu.read_config(self.configfile,'SHELLS_TESTING_JSON')
        # Check if a testfile exists and delete it
        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.json'
        flist = glob.glob(os.path.join(os.getcwd(),fbase))
        if len(flist)>1:
            os.remove(flist[0])
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, self.test_dir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(self.test_dir,'test_process_shells_'),
                          self.configfile,'SHELLS_TESTING_JSON')

        # Check that the file is created and has a certain length
        cdict,dbtype = swu.read_config(self.configfile,'SHELLS_TESTING_JSON')
        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.json'
        flist = glob.glob(os.path.join(self.test_dir,fbase))

        if len(flist)>0:
            with open(flist[0], 'r') as openfile:
                testfile = json.load(openfile)
            if len(testfile['time'])>56:
                testit = 1
            else:
                testit=0
            # Now delete the file
            os.remove(flist[0])
        else:
            testit = 0
        self.assertEqual(1, testit)

    def test_B_run_shells_reprocess_overwrite(self):
        #============================================================
        # TEST:Check that old data is overwritten by new data when adding
        # to an existing file
        #============================================================
        # First create an existing json file of data like in the last test
        #
        # The configfile has
        # input_type = sqlite
        # dbase = sehas_shells
        # inputstbl = ShellsInputsTbl
        # output_type = json
        # fname = shells_inputs
        # This should create a file called shells_inputs_test_20220101HHMMSS.json
        print('*** TEST:Check that old data is overwritten**')

        # Delete any old testing files. This won't delete good files because
        # fname has test in it
        cdict,dbtype = swu.read_config(self.configfile,'SHELLS_TESTING_JSON')
        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.json'
        flist = glob.glob(os.path.join(self.test_dir,fbase))

        if len(flist)>0:
            os.remove(flist[0])
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, self.test_dir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(self.test_dir,'./test_process_shells_'),
                          self.configfile,'SHELLS_TESTING_JSON')

        # Now read the file created above
        fbase = cdict['fname']+'_'+dt.datetime(2022,1,1).strftime('%Y%m%dT')+'*.json'
        flist = glob.glob(os.path.join(self.test_dir,fbase))

        if len(flist)>0:
            with open(flist[0], 'r') as openfile:
                testfile = json.load(openfile)

        # Now try to overwrite the file with all -1
        # Create a fake set of data outdat
        outdat={}
        keys = [x for x in list(testfile.keys()) if x not in ['time', 'satId']]
        for key in keys:
            #
            outdat[key]= np.array([-1]*len(testfile[key]),dtype=np.float64)
        outdat['sat'] = 'n15'
        # Turn the time back into timestamp in msec
        dformat = '%Y-%m-%dT%H:%M:%S.%fZ'
        # datetime gives the right time
        # But timestamp assumes it is MT
        outdat['time'] = [dt.datetime.strptime(x,dformat).replace(tzinfo=timezone.utc).timestamp()*1000 for x in testfile['time']]

        swu.write_shells_text(self.test_dir, outdat, cdict['fname'],'json')

        # Then check that it is all -1s
        # Read the file back in
        with open(flist[0], 'r') as openfile:
            testfile2 = json.load(openfile)

        if testfile.keys()==testfile2.keys():
            # Then check that the values are -1 to show that the oldata
            # was replaced
            tsum=0
            for key in [x for x in list(testfile2.keys()) if x not in ['time']]:
                tsum = tsum +np.sum(np.array(testfile2[key]))
            if tsum<0:
                testit=1
            else:
                testit=0
        else:
            testit=0

        # Then delete the ilfe
        os.remove(flist[0])
        self.assertEqual(1, testit)

    def test_B_run_shells_csv_realtime_start(self):
        #==================================================
        # TEST: Check that todays date is returned when no csv files exist yet
        #==================================================
        # If you start running in real time with no data then it should
        # begin processing data for the current day
        print('*** TEST: Check that todays date is returned when no csv files exist yet')
        cdict, dbtype = swu.read_config(self.configfile, 'SHELLS_TESTING_RT')
        outdir = self.test_dir
        sat = 'n15'
        sdate = pr.get_start_rt_text(cdict, sat, outdir)

        testdate = dt.datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
        self.assertEqual(sdate, testdate)

    def test_A1_run_shells_csv_realtime_update(self):
        #==================================================
        # TEST: Check that a csv file updates in real time mode
        #==================================================
        # Start by creating a csv file in reprocessing mode for the
        # previous day
        # Then update it in rt mode
        print('*** TEST: Check that a csv file updates in real time mode')

        cdict, dbtype = swu.read_config(self.configfile, 'SHELLS_TESTING_RT')
        outdir = self.test_dir
        sat = 'n15'
        sdate =(dt.datetime.utcnow()-dt.timedelta(days=2))
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        # This should add a file for the day before
        pr.process_SHELLS(sdate, sdate,
                          False, False, None, outdir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                           'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                           'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                           'mep_ele_tel90_flux_e4'],
                          None, None, os.path.join(self.test_dir,'test_process_shells_'),
                          self.configfile, 'SHELLS_TESTING_RT')

        # Get the last processed time for the file
        dformat = '%Y%m%d'
        fbase = cdict['fname']+'_'+sdate.strftime(dformat)+'*.csv'
        flist = glob.glob(os.path.join(self.test_dir,fbase))
        flist.sort(reverse=True)
        dat = pd.read_csv(flist[0]).to_dict(orient='list')
        dformat = '%Y-%m-%dT%H:%M:%S.%fZ'
        lasttime = dt.datetime.strptime(dat['time'][-1],dformat)
        cdfdir = os.path.join(os.getcwd(), '..', 'SHELLS', 'cdf')
        # Then try to update in rt
        pr.process_SHELLS(None, None,
                          True, False, None, self.test_dir, cdfdir,
                          "www.ncei.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                           'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                           'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                           'mep_ele_tel90_flux_e4'],
                          None, None, os.path.join(self.test_dir,'test_process_shells_'),
                          self.configfile, 'SHELLS_TESTING_RT')

        # Check that the csv file updated as expected
        # Get the last processed time for the file
        # This might fail if you run it at 1:00 UT because there might
        # not be new data
        dformat = '%Y%m%d'
        #sdate = dt.datetime.utcnow()
        #fbase = cdict['fname']+'_'+sdate.strftime(dformat)+'*.csv'
        fbase = cdict['fname']+'_'+'*.csv'
        flist = glob.glob(os.path.join(self.test_dir,fbase))
        flist.sort(reverse=True)
        dat = pd.read_csv(flist[0]).to_dict(orient='list')
        dformat = '%Y-%m-%dT%H:%M:%S.%fZ'
        updatedtime = dt.datetime.strptime(dat['time'][-1],dformat)
        # Check the time
        if updatedtime>lasttime:
            testit = 1
        else:
            testit =0

        # Now cleanup

        fbase = cdict['fname']+'_'+'*.csv'
        flist = glob.glob(os.path.join(self.test_dir, fbase))
        for file in flist:
            os.remove(file)

        self.assertEqual(testit,1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_process_shells_inputs.configfile = sys.argv.pop()
        unittest.main()
    else:
        unittest.main()