from pathlib import Path
import sys
import json
import os
import netCDF4 as nc4
import datetime as dt
import numpy as np

app_path = os.path.join(os.path.dirname( __file__ ), '..')
sys.path.insert(0, app_path)  # take precedence over any other in path
from app import create_app

def test_config():
    # TEST: check that when the app is created with the default config it is not in test mode
    # and when it is created with test_config it is in test mode
    print('Testing config file being passed correctly')
    print(create_app("default_config").testing)
    assert not create_app("default_config").testing
    assert create_app("test_config").testing


def test_request_example2(client):

    # TEST: Check that the output from the app matches the expected output
    # from running process_POES with nn = True
    # The code to create the test output file is called make_shells_testoutput.py
    # and it creates a files called shells_neural20220101.nc which has the
    # expected output for 20220101 and n15
    print('Testing that app output matches previous shells output with one'
          'pitch angle')

    # First read in the netcdf file that the test will compare to
    fname = 'shells_neural20220101.nc'
    test_data = nc4.Dataset(fname, 'r')

    # Then create a list of times from the file

    times1 = [dt.datetime.utcfromtimestamp(x / 1000) for x in test_data['time']]
    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times1]

    # The x,y,z and pitch angels won't matter for this test because
    # when in test mode, it forces the L value to be whatever is in the
    # test_config.py TESTL variable. xyz just needs to be the right len

    xyz = list()
    for tco in range(0,len(times2)):
        xyz.append([4,1,0])
    energies = [200]
    ddict = {"time":times2,
             "energies":energies,
             "xyz":xyz,
             "sys": "GEO",
             "pitch_angles":[40]
            }
    data = json.dumps(ddict)

    app = create_app(test_config="test_config")

    #data = ddict
    response = app.test_client().post("/shells_io",data=data,content_type='application/json')

    # Now we have to compare test_data for the same L and energy
    # Get the right index for the L value we fixed in the app
    # Todo figure out how to set this directory
    with open("../test_config.py",'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0:5]=='TESTL':
            Lval = float(line[7::])

    #config = configparser.ConfigParser()
    #config.read("test_config.py")
    #Lval = config["TESTL"]
    print(Lval)
    print(test_data['L'][:])
    Lind = np.where(test_data['L'][:] ==Lval)[0]
    temp = response.json


    for E in energies:
        col = 'E flux '+str(E)
        test_one = test_data[col][:,Lind]
        app_one = temp['E flux 200'][:]

        assert np.sum(np.abs(test_one-app_one))<.001

def test_request_with_multiple_Bmirrors(client):
    # TEST: The purpose here is to test that if multiple pitch angles
    # or Bmirrors are requested then it still works.
    # The test bit will emulate the correct magepehem response for
    # based on the number of pitch angles passed

    print('Testing that app output matches previous shells output with one'
          'pitch angle')

    # First read in the netcdf file that the test will compare to
    fname = 'shells_neural20220101.nc'
    test_data = nc4.Dataset(fname, 'r')

    # Then create a list of times from the file

    times1 = [dt.datetime.utcfromtimestamp(x / 1000) for x in test_data['time']]
    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times1]

    # The x,y,z and pitch angels won't matter for this test because
    # when in test mode, it forces the L value to be whatever is in the
    # test_config.py TESTL variable. xyz just needs to be the right len

    xyz = list()
    for tco in range(0,len(times2)):
        xyz.append([4,1,0])
    energies = [200]
    pas = [40,50]
    ddict = {"time":times2,
             "energies":energies,
             "xyz":xyz,
             "sys":"GEO",
             "pitch_angles":pas
            }
    data = json.dumps(ddict)

    app = create_app(test_config="test_config")

    #data = ddict
    response = app.test_client().post("/shells_io",data=data,content_type='application/json')
    # Now we have to compare test_data for the same L and energy
    # Get the right index for the L value we fixed in the app
    # Todo figure out how to set this directory

    print(response)
    with open("../test_config.py",'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0:5]=='TESTL':
            Lval = float(line[7::])

    #config = configparser.ConfigParser()
    #config.read("test_config.py")
    #Lval = config["TESTL"]
    print(Lval)
    print(test_data['L'][:])
    Lind = np.where(test_data['L'][:] ==Lval)[0]
    temp = response.json

    # Check that for each pitch angle its the same as test_one
    for pco in range(0,len(pas)):
        col = 'E flux '+str(energies[0])
        # Have to get the two datasets in the right format
        test_one = np.ndarray.flatten(test_data[col][:,Lind])
        app_temp = np.array(temp['E flux 200'][:])
        #print(np.shape(app_temp[:,pco]))
        app_one = app_temp[:,pco]

        assert np.sum(np.abs(test_one-app_one))<.001