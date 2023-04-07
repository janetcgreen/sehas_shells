import logging
import os
import sqlite3 as sl

import keras
import numpy as np
from joblib import load


def qloss(y_true, y_pred):
    qs = [0.25, 0.5, 0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true - y_pred
    v = np.maximum(q * e, (q - 1) * e)
    return keras.backend.mean(v)


def read_db_inputs(sdate, edate):
    conn = None
    names = None
    rows = None

    try:
        # Connect to the dbase
        conn = sl.connect('./resources/test_sehas_shells.sq')
        cursor = conn.cursor()

        # Read in the SHELLS input data for the range of times passed by the user
        cursor.execute(
            "SELECT * FROM ShellsInputsTbl "
            "WHERE (time BETWEEN ? AND ?) "
            "OR time = (SELECT MAX(time) FROM ShellsInputsTbl WHERE time < ?) "
            "ORDER BY time", (sdate, edate, sdate))
        rows = cursor.fetchall()

        # Get the column names
        names = [description[0] for description in cursor.description]
        # print(names)

        # Get the column number
        cursor.execute("SELECT COUNT(*) FROM pragma_table_info('ShellsInputsTbl')")
        count = cursor.fetchall()
        # print(count[0][0])

        if conn is not None:
            conn.close()

    except Exception as err:
        logging.error('Problems connecting to dbase' + str(err))
        if conn is not None:
            conn.close()

    return names, rows


def run_nn(data, evars, Kpdata, Kpmax_data, out_scale, in_scale, hdf5, L=None, Bmirrors=None, Energies=None):
    '''
    PURPOSE: To take the values in data, apply the neural network from Alex and
    then output the near equatorial flux
    :param data (dict): Dict with data[ecol][timeXL] for each of the 4 POES energy channels
    :param evars (list): List of the energy channel names
    :param Kpdata (list): List of Kp*10 for each time
    :param Kpmax_data (list):List of Kp*10_max_3d for each time
    :param out_scale (str): Name of the output transform file for the NN
    :param in_scale (str): Name of the input transform file for the NN
    :param hdf5 (str): Name of file used by the NN
    :return:
    '''

    # List of energies we want for the output data
    if Energies is None:
        Energies = np.arange(200.0, 3000.0, 200.0)

    # The Bmirror values for the output at each L
    if Bmirrors is None:
        # This is for the value at the equator
        Bmirrors = [2.591e+04 * (L ** -2.98) for L in Ls]

    # L values for the ouput corresponding to each bmirror
    if L is None:
        L = np.arange(3., 6.3, 1)

    # Check for -1 where L>7.5
    # Sometimes this happens because the orbit does not go out that far
    for wco in range(0, len(evars)):
        bad_inds = np.where((data[evars[wco]][:]) < 0)
        if len(bad_inds[0]) > 0:
            for bco in range(0, len(bad_inds[0])):
                # Set the flux equal to the neighbor
                data[evars[wco]][bad_inds[0][bco]][bad_inds[1][bco]] = data[evars[wco]][bad_inds[0][bco]][
                    bad_inds[1][bco] - 1]

    # Todo need to deal with bad data
    # My data has timeXL for each energy in a dict
    # This expected input is timeXL e1, timeXL e2 timeXL e3, timeXL e4

    # This concatetnates the fluxes at each energy into one array
    new_dat = np.array(data[evars[0]][:])
    for wco in range(1, len(evars)):
        new_dat = np.append(new_dat, data[evars[wco]], axis=1)

    l, w = np.shape(new_dat)

    # What do we want as output if we are going to make a netcdf file
    # data[E][time x L] at Beq for that L
    # data[E1_upperq] [timeXL] upper quantile
    # data[E1_lowerq] [timexL] lower quantile

    # Create a dict for the output data
    outdat = {}
    outdat['L'] = L
    outdat['Bmirrors'] = Bmirrors
    outdat['Energies'] = Energies
    # Then create arrays for each E and E quantiles
    for E in Energies:
        col = 'E flux ' + str(int(E))
        outdat[col] = np.zeros((0, len(L)), dtype=np.float)
        colh = 'E flux ' + str(int(E)) + ' upper q'
        outdat[colh] = np.zeros((0, len(L)), dtype=np.float)
        coll = 'E flux ' + str(int(E)) + ' lower q'
        outdat[coll] = np.zeros((0, len(L)), dtype=np.float)
    outdat['time'] = list()
    outdat['Kp'] = list()
    outdat['Kpmax'] = list()

    # Step through the POES passes one at a time
    for pco in range(0, l):
        # The input needs Kp, Kpmax, E, Bmirror for each L
        # Check that the poes input does not have Nans
        check_dat = np.where((np.isnan(new_dat[pco][:])) | (new_dat[pco][:] < 0))[0]

        if len(check_dat) < 1:
            outdat['time'].append(data['time'][pco])
            outdat['Kp'].append(Kpdata[pco] / 10)
            outdat['Kpmax'].append(Kpmax_data[pco] / 10)

            # The NN code can calculate flux for all Ls at once
            kp = np.tile(Kpdata[pco], len(L))  # Create a list of Kp for each L calc
            maxkp = np.tile(Kpmax_data[pco], len(L))  # Create a list of maxKp for each L
            # Create a list of POES data to be used for each L calc
            poes = np.tile(new_dat[pco:pco + 1], (len(L), 1))

            # Step through each energy and create outdat[Ecol] that is len L
            for eco in range(0, len(Energies)):
                # Make a list of one energy at all L's
                energy = np.tile(Energies[eco], len(L))
                # The Bmirror is different for each L
                input = np.concatenate((np.array(energy).reshape(-1, 1), np.array(Bmirrors).reshape(-1, 1),
                                        np.array(L).reshape(-1, 1), np.array(kp).reshape(-1, 1),
                                        np.array(maxkp).reshape(-1, 1),
                                        poes), axis=1)
                # This returns the lowerq, log(flux), upperq data for one E and Bmirror(L) at each L
                # start=ti.time()
                fpre = out_scale.inverse_transform(hdf5.predict(in_scale.transform(input)))
                # tend = ti.time()
                # print('time to do nn',tend-start)
                cols = ['E flux ' + str(int(Energies[eco])) + ' upper q',
                        'E flux ' + str(int(Energies[eco])),
                        'E flux ' + str(int(Energies[eco])) + ' lower q', ]
                for cco in range(0, len(cols)):
                    temp = outdat[cols[cco]][:]
                    outdat[cols[cco]] = np.vstack((temp, fpre[:, cco]))

    return outdat


def process_data(sdate, edate, Ls, Energies):
    # Create a dict for the output data
    outdat = {}

    try:
        # ---------------- Set up mapping info -------------------

        # --------------------- Set up nn ------------------------
        # Load in transforms, model for Neural Network.
        # They are always a trio of files: in_scale, out_scale binary files and a model_quantile HDF5 file.

        out_scale = load(os.environ.get('OUT_SCALE_FILE'))
        in_scale = load(os.environ.get('IN_SCALE_FILE'))
        hdf5 = keras.models.load_model(os.environ.get('HDF5FILE'), custom_objects={'loss': qloss}, compile=False)

        # out_scale = load("./resources/out_scale_final_09242021.bin")
        # in_scale = load("./resources/in_scale_final_09242021.bin")
        # hdf5 = keras.models.load_model("./resources/shells_model_final_v6_09242021.h5", custom_objects={'loss': qloss},
        #                                compile=False)

        keys, rows = read_db_inputs(sdate, edate)

        # List of the channels
        channels = ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']

        # Bmirrors
        Bmirrors = np.floor(2.591e+04 * (Ls ** -2.98))

        # Initialize map_data and index
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

        print('Doing nn')

        outdat = run_nn(map_data, channels, Kp10[:], Kp_max[:], out_scale, in_scale, hdf5, L=Ls, Bmirrors=Bmirrors,
                        Energies=Energies)

        print('Done with nn')

    except Exception as e:
        # If there are any exceptions then log the error
        print(e)
        logging.error(e)

    return outdat
