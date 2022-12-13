import datetime as dt
import unittest
import numpy as np
import os
import sys
import process_SHELLS_inputs as pr
import sqlite3 as sl
from collections import OrderedDict

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

        self.cdict,dbase_type = swu.read_config(self.configfile)
        # The test config just has the name of the dbase
        # and the tables, no password
        self.conn = sl.connect(self.cdict['dbase']) # Create a dbase called sehas_shells
        cursor = self.conn.cursor()
        # ----------------------- Create tables ------------------
        tables = self.get_shells_tables()  # return a dict with the table definitions

        for name, ddl in tables.items():
            try:
                print("Creating table {}: ".format(name))
                cursor.execute(ddl)
            except Exception as err:
                print(err)
            else:
                print("OK")
        print("Done with tables")
        print('Here')

    def tearDown(self):
        # ------------------------------------------------
        # Teardown:
        # 1) Delete the sat 'testsat'
        # 2) delete the channel test_channel
        # 3) Remove the log file
        #--------------------------------------------------
        pass

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

        # The input table will have the time, channelId, LId, satId, lat, lon, NS, and eflux
        tables['ShellsInputsTbl'] = (
            "CREATE TABLE ShellsInputsTbl ("
            "  unixTime_utc int NOT NULL,"
            "  channelId int NOT NULL,"
            "  LId int NOT NULL,"
            "  satId int NOT NULL,"
            "  eflux double DEFAULT NULL)")

            #"  PRIMARY KEY (unixTime_utc, channelId, LId),"
            #"  FOREIGN KEY (channelId) REFERENCES ShellsChannelTbl(id),"
            #"  FOREIGN KEY (SatId) REFERENCES ShellsSatTbl(id)"
            #") ENGINE=InnoDB")

        return tables

    def test_run_shells_reprocess_dbase(self):
        #==================================================================
        # TEST: Check input data creation in dbase in reprocessing mode
        #==================================================================
        # Inputs here are sdate,edate,
        # realtime,neural,localdir,outdir,cdfdir
        # noaasite,sat,
        # vars,channels
        # model,modeldir,logfile,config

        # This is running in reprocessing mode for one day
        pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,1),
                          False, False, None, None, './SHELLS/cdf/',
                          "satdat.ngdc.noaa.gov", ["n15"],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,'./test_process_shells_',
                          self.configfile)
        # Todo the assert here should check that data is created in the dbase
        # for all e flux channels
        self.assertEqual(1, 1)

    def test_run_shells_dbase_realtime(self):
        #==================================================
        # TEST: Check that data are created in the dbase in real time mode
        #==================================================
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_process_shells_inputs.configfile = sys.argv.pop()
        unittest.main()
    else:
        unittest.main()