import datetime as dt
import numpy as np
import os
import sys
import process_SHELLS_inputs as pr
import sqlite3 as sl
from collections import OrderedDict
import argparse

sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
import shells_web_utils as swu


def get_shells_tables():
    '''
    PURPOSE: to create a dict called tables with the table definitions to be
    used for the shells inputs
    :return: tables (dict)

    '''

    tables = OrderedDict()

    # The input table will have the same structure as the csv/json files
    # time, e1 L1, e1 L2, ...e2 L1, e2 L2, ... Kp, Kpmax, satId
    tables['ShellsInputsTbl'] = ("CREATE TABLE ShellsInputsTbl ("
                                 "  time int NOT NULL,")
    channels = ["mep_ele_tel90_flux_e1", "mep_ele_tel90_flux_e2", "mep_ele_tel90_flux_e3", "mep_ele_tel90_flux_e4"]
    Lbins = np.arange(1, 8.25, .25)
    for channel in channels:
        for L in Lbins:
            colname = '"' + channel + '_L_' + str(L) + '"'
            tables['ShellsInputsTbl'] = tables['ShellsInputsTbl'] + ' ' + colname + ' double DEFAULT NULL,'

    tables['ShellsInputsTbl'] = tables['ShellsInputsTbl'] + '  "Kp*10" int DEFAULT NULL, ' \
                                                            "Kp_max double DEFAULT NULL, satId int NOT NULL," \
                                                            "  PRIMARY KEY (time)" \
                                                            ") "
    # "  FOREIGN KEY (channelId) REFERENCES ShellsChannelTbl(id),"
    # "  FOREIGN KEY (SatId) REFERENCES ShellsSatTbl(id)"
    # ") ENGINE=InnoDB")
    # "  e1_L1 int NOT NULL,"
    # "  e1_L2 int NOT NULL,"
    # "  e1_L3 int NOT NULL,"
    # "  e1_L4 int NOT NULL,"
    # "  e1_L5 int NOT NULL,"
    # "  e1_L6 int NOT NULL,"
    # "  e1_L7 int NOT NULL,"
    # "  e1_L8 int NOT NULL,"
    # "  satId int NOT NULL,"
    # "  eflux double DEFAULT NULL)")

    # "  PRIMARY KEY (unixTime_utc, channelId, LId),"
    # "  FOREIGN KEY (channelId) REFERENCES ShellsChannelTbl(id),"
    # "  FOREIGN KEY (SatId) REFERENCES ShellsSatTbl(id)"
    # ") ENGINE=InnoDB")

    return tables


def make_dB(configfile,db_dir):
    # --------------------------------------------
    # SETUP:
    # 1) read the config file
    # 2) create an sqlite dbase with the needed tables

    # ---------------------- Define some tables in configfile---------------------

    cdict, dbase_type = swu.read_config(configfile, 'DEFAULT')

    # ----------------------- Create tables ------------------
    # The DEFAULT test config has the name of the sqlite dbase test_sehas_shells
    # and the table names, no password

    conn = sl.connect(os.path.join(db_dir, cdict['dbase']))  # Create a dbase called sehas_shells
    cursor = conn.cursor()
    # ----------------------- Create tables ------------------
    tables = get_shells_tables()  # return a dict with the table definitions

    for name, ddl in tables.items():
        try:
            print("TEST SETUP: Creating table {}: ".format(name))
            cursor.execute(ddl)
        except Exception as err:
            print(err)
        else:
            print("OK")

    print("*** TEST SETUP: Done with tables")

def make_sqlite_inputs(configfile,db_dir):
    '''
    PURPOSE: To create a small sqlite dbase with needed shells inputs for a short
    time range for the sole purpose of testing. This sqlite dbase will have the same
    columns and structure as the NASA CCMC hapi dbase

    :param configfile:
    :param db_dir:
    :return:
    '''
    # configfile = 'config_test_shells_inputs_local.ini'

    # Create the dbase with the table and colnames
    make_dB(configfile, db_dir)

    # location of cdf files to pass to process_SHELLS
    outdir = db_dir
    cdfdir = os.path.join( os.getcwd(), '..','SHELLS','cdf' )

    # Run the code that should create a dbase data for 2022/1/1 to 2022/1/10
    # the config_test_shells_input_local.ini has the name as test_sehas_shells
    pr.process_SHELLS(dt.datetime(2022,1,1),dt.datetime(2022,1,10),
                          False, False, None, outdir, cdfdir,
                          "www.ncei.noaa.gov", ['n15','n18'],
                          ['time', 'alt', 'lat', 'lon', 'L_IGRF', 'MLT',
                                'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                'mep_ele_tel90_flux_e4'],
                          ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3',
                                    'mep_ele_tel90_flux_e4'],
                          None, None,os.path.join(outdir,'test_process_shells_'),
                          configfile,'SHELLS_TESTING_SQLITE')

    # Check that the dbase was created
    cdict, dbase_type = swu.read_config(configfile, 'SHELLS_TESTING_SQLITE')
    conn = sl.connect(os.path.join(db_dir, cdict['dbase']))  # Create a dbase called sehas_shells
    cursor = conn.cursor()
    # Get all the data
    cursor.execute("SELECT * from ShellsInputsTbl")
    rows = cursor.fetchall()
    # Get all the names
    names = [description[0] for description in cursor.description]
    time = list()
    for row in rows:
        time.append(row[0])
        print(row[0])
    print('Here')

if __name__ == "__main__":
    #-------------------------------------------------------------------
    #           GET COMMAND LINE ARGUMENTS
    #-------------------------------------------------------------------
    parser = argparse.ArgumentParser('This program gets POES data from NOAA and processes it for SHELLS')
    parser.add_argument('-c', "--config",
                        help="The full directory and name of the config file",
                        default = os.path.join(os.getcwd(),'config_test_shells_inputs_local.ini'),
                        required=False)
    parser.add_argument('-od', "--outdir",
                        help="The local directory to put the output files",
                        required=False, default = os.path.join(os.getcwd(),'SHELLS'))
    args = parser.parse_args()

    # ----------------------------------------------------------------

    x = make_sqlite_inputs(args.config,args.outdir)