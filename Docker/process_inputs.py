import logging
import os
import sqlite3 as sl
from bisect import bisect_right

import requests
from flask import current_app
import keras
import numpy as np
from joblib import load
from flask import current_app
import datetime as dt
import requests
from hapiclient import hapi


def qloss(y_true, y_pred):
    qs = [0.25, 0.5, 0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true - y_pred
    v = np.maximum(q * e, (q - 1) * e)
    return keras.backend.mean(v)


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
        return len(list_dt)-1

    return pos-1

def read_hapi_inputs(req_times,server,dataset):
    '''
    PURPOSE: To get the hapi data from the CCMC iswa server at server for
    the shells_input dataset for the reqested times
    :param req_times: A list of times to return input data from
    :param server: The server name i.e. 'https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/hapi/'
    :param dataset the dataset name i.e. shell_input
    :return:
    '''

    # Create a dictionary with info to build the hapi query
    hinput = {}
    hinput['id'] = dataset

    # reformat the start and stop datetimes into HAPI times
    # Get data from 3 hours before the first requested time so it has a value prior
    hinput['time.min'] = (dt.datetime.strptime(req_times[0],'%Y-%m-%dT%H:%M:%S.%fZ') - dt.timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    hinput['time.max'] = req_times[-1]
    hinput['format']='json'

    # Build the hapi query
    query = server+'data?'
    for key in list(hinput.keys()):
        query= query+key+'='+hinput[key]+'&'

    iswa_data = requests.get(query[0:-1]) # query will have an extra &
    hapi_data = {} # dict for refromatting the returned data

    # Check that ther was a valid response
    if iswa_data.status_code==200:

        # Get a list of the col names
        iswa_json= iswa_data.json()
        cols = [x['name'] for x in iswa_json['parameters']]
        cdata= np.array(iswa_json['data']) # Make an array because it is simpler to work with

        # iswa_json['data'] is a list of lists
        # The next step needs the data as a dict with 'time,
        # 'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4', 'Kp*10', 'Kp_max'
        if len(cdata)>0:
            hapi_data['time'] = cdata[:,cols.index('Time')] # Change the time column capital
            #map_data['time']=[x[cols.index('Time')] for x in iswa_json['data']]
            hapi_data['Kp*10'] = cdata[:, cols.index('kp10')].astype(float) # Change the k
            #map_data['Kp*10'] = [x[cols.index('kp10')] for x in iswa_json['data']]
            hapi_data['Kp_max'] = cdata[:, cols.index('kp_3day_max')].astype(float)
            #map_data['Kp_max'] = [x[cols.index('kp_3day_max')] for x in iswa_json['data']]

            # Todo fix this when we have the changes
            # Right now the hapi server returns the columns with the worng names
            ecolscc = ['map_ele_tel90_flux_e1', 'map_ele_tel90_flux_e2', 'map_ele_tel90_flux_e3', 'map_ele_tel90_flux_e4']
            ecols = ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']
            for ecolcc,ecol in zip(ecolscc,ecols):
                # Need to reformat data to have time X L
                alles = [x for x in cols if ecolcc in x]
                allinds= [cols.index(x) for x in alles]
                hapi_data[ecol]=cdata[:,allinds].astype(iswa_json['parameters'][allinds[0]]['type'])
        else:
            hapi_data= None
    else:
        hapi_data = None

    return hapi_data

def reorg_hapi(in_times,hdata):
    '''

    :param in_time:
    :param hdata:
    :return:
    '''
    nearest_pos = np.array([get_nearest(list(hdata['time']),t) for t in in_times])
    map_data={}
    for key in hdata.keys():
        if len(np.shape(hdata[key]))>1:
            map_data[key] = hdata[key][nearest_pos,:]
        else:
            map_data[key] = hdata[key][nearest_pos]
    return map_data
def read_db_inputs(req_times):
    '''

    :param req_times: list of times requested by user
    :return:
    '''
    conn = None
    names = None
    rows = None

    try:
        # Connect to the dbase
        #print("Connecting to dbase")
        # Todo make this a parameter in the config
        dbname =os.environ.get('DBNAME')
        #print(dbname)
        #dbname = 'test_sehas_shells.sq'
        dbase = os.path.join(os.path.dirname( __file__ ), 'resources',dbname)
        #print(dbase)
        #conn = sl.connect('./resources/test_sehas_shells.sq')
        conn = sl.connect(dbase)
        cursor = conn.cursor()

        # Read the SHELLS input times for the range of times passed by the user
        # mtime = '2022-01-02T22:31:50.327000Z'
        cursor.execute(
            "SELECT * FROM ShellsInputsTbl "
            "WHERE (time >= ? AND time <= ?) "
            "OR time = (SELECT MAX(time) FROM ShellsInputsTbl WHERE time <= ?) "
            "ORDER BY time", (min(req_times), max(req_times), min(req_times)))
        rows1 = cursor.fetchall()
        input_times = [item[0] for item in rows1]
        #print('input times: ', input_times)

        # Find the input poes position nearest to the requested times in the CCMC HAPI database
        nearest_pos = [get_nearest(input_times, t) for t in req_times]
        #print(nearest_pos)

        rows = [rows1[x] for x in nearest_pos]

        # Get the actual times to make sure it works right
        nearest_times = [input_times[x] for x in nearest_pos]


        #print('input times: ', input_times)
        #print('request times: ', req_times)
        #print('nearest times: ', nearest_times)

        # Read in the SHELLS input data for the range of nearest times
        # sql = "SELECT * FROM ShellsInputsTbl WHERE time IN ({seq})".format(seq=','.join(['?'] * len(nearest_times)))
        # cursor.execute(sql, nearest_times)
        # rows = cursor.fetchall()

        # Get the column names
        names = [description[0] for description in cursor.description]
        # print(names)

        # Get the column number
        #cursor.execute("SELECT COUNT(*) FROM pragma_table_info('ShellsInputsTbl')")
        #count = cursor.fetchall()
        # print(count[0][0])

        if conn is not None:
            conn.close()

    except Exception as err:
        logging.error('Problems connecting to dbase' + str(err))
        if conn is not None:
            conn.close()

    return names, rows

def reorg_data(keys,rows,channels):
    # List of the channels

    # Initialize map_data
    map_data = {}
    time_arr = []
    e1_arr, e2_arr, e3_arr, e4_arr = [], [], [], []
    Kp10, Kp_max = [], []

    for i in range(len(rows)):
        time_arr.append(rows[i][keys.index("time")])

        # zip function takes elements of input tuple 1 as keys and input tuple 2 elements as values
        # Then we convert this to a dictionary using dict()
        row = dict(zip(keys, rows[i]))

        e1_arr.append([value for key, value in row.items() if '_e1_' in key])
        e2_arr.append([value for key, value in row.items() if '_e2_' in key])
        e3_arr.append([value for key, value in row.items() if '_e3_' in key])
        e4_arr.append([value for key, value in row.items() if '_e4_' in key])

        Kp10.append(row['Kp*10'])
        Kp_max.append(row['Kp_max'])

    map_data['time'] = np.array(time_arr)
    map_data['mep_ele_tel90_flux_e1'] = np.array(e1_arr)
    map_data['mep_ele_tel90_flux_e2'] = np.array(e2_arr)
    map_data['mep_ele_tel90_flux_e3'] = np.array(e3_arr)
    map_data['mep_ele_tel90_flux_e4'] = np.array(e4_arr)
    map_data['Kp*10'] = Kp10
    map_data['Kp_max'] = Kp_max

    return map_data



def run_nn(data, evars, Kpdata, Kpmax_data, out_scale, in_scale, hdf5, L=None, Bmirrors=None, Energies=None):
    '''
    PURPOSE: To take the values in data, apply the shells neural network and
    then output the electron flux

    :param data (dict): Dict with data[ecol][timeXL] for each of the 4 POES energy channels
    :param evars (list): List of the energy channel names for POES
    :param Kpdata (list): List of Kp*10 for each time
    :param Kpmax_data (list):List of Kp*10_max_3d for each time
    :param out_scale (str): Name of the output transform file for the NN
    :param in_scale (str): Name of the input transform file for the NN
    :param hdf5 (str): Name of file used by the NN
    :param L (list(list)): list of single L values for each xyz or a fixed set for every time
    :param Bmirrors (list(list)): list of Bmirrors for every time step or a fixed set
    :param Energies (list): list of electron energies to return
    :return:
    '''

    # List of energies we want for the output data
    #print(" Doing nn")
    if Energies is None:
        # If no Energies are passed it assumes this
        Energies = np.arange(200.0, 3000.0, 200.0)

    # The Bmirror values for the output at each l
    if Bmirrors is None:
        # This is for the value at the equator
        Bmirrors = [2.591e+04 * (l ** -2.98) for l in L]

    # L values for the ouput corresponding to each bmirror
    if L is None:
        L = np.arange(3., 6.3, 1)

    # Check for -1 where L>7.5
    # Sometimes this happens in the input data because the POES orbit does not
    # go out very far. In that case, set it to neighboring values
    # Step through each energy channel
    for wco in range(0, len(evars)):
        bad_inds = np.where((data[evars[wco]][:]) < 0)
        if len(bad_inds[0]) > 0:
            for bco in range(0, len(bad_inds[0])):
                # Set the flux equal to the neighbor
                data[evars[wco]][bad_inds[0][bco]][bad_inds[1][bco]] = data[evars[wco]][bad_inds[0][bco]][
                    bad_inds[1][bco] - 1]

    # Todo need to deal with bad data
    # The input data has timeXL for each energy in a dict
    # The expected input for the nn is timeXL e1, timeXL e2 timeXL e3, timeXL e4
    # So need to concatentate into one  array
    new_dat = np.array(data[evars[0]][:])
    for wco in range(1, len(evars)):
        new_dat = np.append(new_dat, data[evars[wco]], axis=1)

    l, w = np.shape(new_dat)

    # Output needed if we are going to make a netcdf file
    # data[E][time x L] at Beq for that L
    # data[E1_upperq] [timeXL] upper quantile
    # data[E1_lowerq] [timexL] lower quantile

    # Create a dict for the output data
    outdat = {}
    outdat['L'] = L # This could be 1D or 2D
    outdat['Bmirrors'] = Bmirrors # This could be 1d or 2d
    outdat['Energies'] = Energies # This should be 1d

    # Then create arrays for flux at each Energy and E quantiles

    # Need to check if Bmirrors/Ls is 1D or 2d
    # If its 2D then its an xyz request

    if len(np.shape(Bmirrors))>1:
        Bl, Bw = np.shape(Bmirrors) #(2D)
    else:
        Bw = len(Bmirrors) #1D

    # Define the output columns
    for E in Energies:
        # I changed this so that the output will be timeXBmirrors (or pitch angles)
        col = 'E flux ' + str(int(E))
        outdat[col] = np.zeros((0, Bw), dtype=float)
        colh = 'E flux ' + str(int(E)) + ' upper q'
        outdat[colh] = np.zeros((0, Bw), dtype=float)
        coll = 'E flux ' + str(int(E)) + ' lower q'
        outdat[coll] = np.zeros((0, Bw), dtype=float)

    outdat['time'] = list() # same times will be returned except bad data
    outdat['Kp'] = list() # same Kp will be returned
    outdat['Kpmax'] = list() # same Kpmax will be returned

    # Step through each time step in the input data and do the nn
    for pco in range(0, l):
        # The input needs Kp, Kpmax, E, Bmirror for each L
        # Check that the input data does not have Nans
        check_dat = np.where((np.isnan(new_dat[pco][:])) | (new_dat[pco][:] < 0))[0]

        if len(check_dat) < 1:
            # Append the current value to the outdat list
            outdat['time'].append(data['time'][pco])
            outdat['Kp'].append(Kpdata[pco] / 10)
            outdat['Kpmax'].append(Kpmax_data[pco] / 10)

            # The NN code can calculate flux for all Ls/Bm at once
            kp = np.tile(Kpdata[pco], Bw)  # Create a list of Kp for each Bmirror
            maxkp = np.tile(Kpmax_data[pco], Bw)  # Create a list of maxKp for each Bmirror
            # Create a list of POES data to be used for each Bmirror calc
            poes = np.tile(new_dat[pco:pco + 1], (Bw, 1))

            # Check if Bmirrors is 2D or 1D
            if len(np.shape(Bmirrors))>1:
                Bthis = Bmirrors[pco]
            else:
                Bthis = Bmirrors

            # Check if L is 2D or 1D
            if len(np.shape(L))>1:
                Lthis = L[pco]
            else:
                Lthis = L

            # If there is just one L then need to repeat it for each Bmirror
            if len(Lthis) != len(Bthis):
                Lthis = np.tile(Lthis, Bw)

            # Step through each energy and create outdat[Ecol] that is len Bmirror
            for eco in range(0, len(Energies)):
                # Make a list of one energy at all Bm's
                energy = np.tile(Energies[eco], Bw)
                input = np.concatenate((np.array(energy).reshape(-1, 1), np.array(Bthis).reshape(-1, 1),
                                        np.array(Lthis).reshape(-1, 1), np.array(kp).reshape(-1, 1),
                                        np.array(maxkp).reshape(-1, 1),
                                        poes), axis=1)
                # This returns the lowerq, log(flux), upperq data for one E and Bmirror(L) at each L
                # start=ti.time()
                fpre = out_scale.inverse_transform(hdf5.predict(in_scale.transform(input),verbose=0))
                # tend = ti.time()
                # print('time to do nn',tend-start)
                cols = ['E flux ' + str(int(Energies[eco])) + ' upper q',
                        'E flux ' + str(int(Energies[eco])),
                        'E flux ' + str(int(Energies[eco])) + ' lower q', ]
                for cco in range(0, len(cols)):
                    # Todo check if this works for multiple Bms
                    # If there is multiple Bms then each energy channel will be [timeXBm]
                    temp = outdat[cols[cco]][:] # Get the current data for that energy col
                    #Todo set Ls>6.3 to nan
                    outdat[cols[cco]] = (np.vstack((temp, fpre[:, cco]))).tolist()

    return outdat


def process_data(time, Ls, Bmirrors, Energies):
    '''
    # PURPOSE: Translates input data into shells electron flux
    # First it gets the input POES data for the sleected time from
    # the NASA HAPI server or from an sqlite dbase (testing)
    #
    :param time (list):
    :param Ls (list 1D or 2D):
    :param Bmirrors (list 1D or 2D):
    :param Energies (list):
    :return:
    '''

    # Create a dict for the output data
    outdat = {}

    try:

        # --------------------- Set up nn ------------------------
        # Load in transforms, model for Neural Network.
        # Always a trio of files: in_scale, out_scale binary files
        # and a model_quantile HDF5 file. All are defined in .env

        #print(os.environ.get('OUT_SCALE_FILE'))
        out_scale = load(os.environ.get('OUT_SCALE_FILE'))
        in_scale = load(os.environ.get('IN_SCALE_FILE'))
        hdf5 = keras.models.load_model(os.environ.get('HDF5FILE'), custom_objects={'loss': qloss}, compile=False)

        # These are the POES electron fluxchannel names
        channels = ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']

        # Check if testing or no
        if current_app.testing==True:
            if current_app.config['HAPI_TEST']==False:
                # In testing mode read the poes data from an sqlite dbase
                keys, rows = read_db_inputs(time)
                # This reorganizes the data from the dbase into a dict
                # These are all numpy arrays
                map_data = reorg_data(keys,rows,channels)
                print('Here')
            else:
                server = os.environ.get('HAPI_SERVER')
                dataset = os.environ.get('HAPI_DATASET')
                hapi_data = read_hapi_inputs(time, server, dataset)

                # get the closest hapi data to the times in time
                map_data = reorg_hapi(time,hapi_data)
                #nearest_pos = [get_nearest(input_times, t) for t in req_times]
                #map_data = reorg_data(keys, rows, channels)

        else:
            # In testing mode read the data from an sqlite dbase
            # https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/hapi/data?id=shell_input&time.min=2023-08-17T00:00:00.0Z&time.max=2023-08-18T00:00:00.0Z&format=json

            server = os.environ.get('HAPI_SERVER')
            dataset = os.environ.get('HAPI_DATASET')
            hapi_data = read_hapi_inputs(time, server, dataset)

            # get the closest hapi data to the times in time
            map_data = reorg_hapi(time, hapi_data)

        # Replace map_data['time'] with the requested times
        map_data['time'] = time
        #print('Doing nn')
        # Before we were calling the nn with a fixed set of Ls, and Bmirrors
        # for every time stamp that were passed with L=Ls and Bmirrors = Bmirros.
        # Now we call it with a different L for every time step
        # And different Bmirrors (timeX pitch angles)
        # We might still want the ability to pass a fixed set of Ls and Bmirrors
        # To do that we check if Ls is 1D or 2D in run_nn
        # The way the call to magephem works, this will always return a 2D array
        # for Ls and Bmirrors

        outdat = run_nn(map_data, channels, map_data['Kp*10'], map_data['Kp_max'],
                        out_scale, in_scale, hdf5, L=Ls, Bmirrors=Bmirrors,
                        Energies=Energies)

        print('Done with nn')

    except Exception as e:
        # If there are any exceptions then log the error
        print(e)
        logging.error(e)

    return outdat
