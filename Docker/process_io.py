import logging
import sqlite3 as sl

import keras
import numpy as np
from joblib import load

import shells_web_utils as swu


# ----------------------- Basic functions ----------------------------
# --------------------------------------------------------------------

def qloss(y_true, y_pred):
    qs = [0.25, 0.5, 0.75]
    q = np.constant(np.array([qs]), dtype=np.float32)
    e = y_true - y_pred
    v = np.maximum(q * e, (q - 1) * e)
    return keras.backend.mean(v)


def read_db_inputs(sdate=None, edate=None):
    conn = None
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


# ------------------- The main process_data function -----------------
# --------------------------------------------------------------------

def process_data(sdate=None, edate=None):
    try:
        # ---------------- Set up mapping info -------------------

        # --------------------- Set up nn ------------------------
        # load in transforms,model for Neural Network
        # They are always a trio of files: in_scale, out_scale binary files
        # and a model_quantile HDF5 file.

        # out_scale = load(os.environ.get('OUT_SCALE_FILE'))
        # in_scale = load(os.environ.get('IN_SCALE_FILE'))
        # m = keras.models.load_model(os.environ.get('HDF5FILE'), custom_objects={'loss': qloss}, compile=False)

        # No module named 'sklearn', it seems it is deprecated with scikit-learn.
        # Installed conda install scikit-learn
        out_scale = load("../src/out_scale_final_09242021.bin")
        in_scale = load("../src/in_scale_final_09242021.bin")

        hdf5 = keras.models.load_model("../src/shells_model_final_v6_09242021.h5", custom_objects={'loss': qloss},
                                       compile=False)

        keys, rows = read_db_inputs(sdate, edate)

        # List of the channels
        channels = ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']

        # Energies, Bmirrors, Ls based on Seths analysis
        Energies = np.arange(200., 3000., 200.)
        Ls = 4.0
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
        outdat = swu.run_nn(map_data, channels, Kp10[:], Kp_max[:], out_scale, in_scale, hdf5,
                            L=Ls,
                            Bmirrors=Bmirrors,
                            Energies=Energies)
        # outdat['dims'] = ['time', 'L']
        # outdat['sat'] = sat
        print('Done with nn')

    except Exception as e:
        # If there are any exceptions then log the error
        print(e)
        logging.error(e)


process_data('2022-01-10T20:16:20.967250Z', '2022-01-10T20:16:21.967250Z')
