from pathlib import Path
import sys
import json
import os
import netCDF4 as nc4
import datetime as dt
import numpy as np
from dotenv import load_dotenv

app_path = os.path.join(os.path.dirname( __file__ ), '..')
sys.path.insert(0, app_path)  # take precedence over any other in path
from app import create_app

# TESTS in this file
# 1) test_config:
#   TEST: check that when the app is created with the default config it is not in test mode
#   and when it is created with test_config it is in test mode
# 2) test_request_example2
#   TEST: Check that the output from the app matches the expected output
#   from running process_POES with nn = True
# 3) test_request_with_multiple_Bmirrors(client):
#   TEST: Test that if multiple pitch angles
#   or Bmirrors are requested then it still works.
# 4) test_request_with_LShells(client):
#   TEST: Check that the right data is returned when the endpoint
#   is called that allows you to pass L shells.

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

    # Then create a list of times from the test file

    times1 = [dt.datetime.utcfromtimestamp(x / 1000) for x in test_data['time']]
    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times1]

    # The x,y,z and pitch angels won't matter for this test because
    # when in test mode, it forces the L value to be whatever is in the
    # test_config.py TESTL variable instead of using the magephem service.
    # xyz just needs to be the right len

    # Create a list of locations lists
    xyz = list()
    for tco in range(0,len(times2)):
        xyz.append([4,1,0])
    energies = [200]

    # Just give one pitch angle value
    ddict = {"time":times2,
             "energies":energies,
             "xyz":xyz,
             "sys": "GEO",
             "pitch_angles":[40]
            }
    data = json.dumps(ddict)

    app = create_app(test_config="test_config") # Create the app in test mode

    response = app.test_client().post("/shells_io",data=data,content_type='application/json')

    # Now compare test_data for the same L and energy
    # Get the right index for the L value we fixed in the app
    # Todo figure out how to set this directory
    with open("../test_config.py",'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0:5]=='TESTL':
            Lval = float(line[7::])

    Lind = np.where(test_data['L'][:] ==Lval)[0]
    temp = response.json

    for E in energies:
        col = 'E flux '+str(E)
        test_one = test_data[col][:,Lind] # This is the test data at the requested energy and L
        app_one = temp['E flux 200'][:] #This is the same for the app

        assert np.sum(np.abs(test_one-app_one))<.001

def test_request_with_multiple_Bmirrors(client):
    # TEST: Test that if multiple pitch angles
    # or Bmirrors are requested then it still works.
    # The test bit will emulate the correct magepehem response
    # based on the number of pitch angles passed

    print('Testing that app output matches previous shells output with multiple'
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

def test_request_with_LShells(client):
    # TEST: Check that the right data is returned when the endpoint
    # is called that allows you to pass L shells.
    # First read in the netcdf file that the test will compare to
    fname = 'shells_neural20220101.nc'
    test_data = nc4.Dataset(fname, 'r')

    # Then create a list of times from the file

    times1 = [dt.datetime.utcfromtimestamp(x / 1000) for x in test_data['time']]
    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times1]

    # The x,y,z and pitch angels won't matter for this test because
    # when in test mode, it forces the L value to be whatever is in the
    # test_config.py TESTL variable. xyz just needs to be the right len
    Ls = [3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.25, 5.5, 5.75, 6., 6.25]
    xyz = list()
    for tco in range(0,len(times2)):
        xyz.append([4,1,0])
    energies = [200]
    Bm = [980.0,772.0,619.0,504.0,416.0,347.0,293.0,249.0,214.0,185.0,
                                          161.0,141.0,124.0,110.0]
    ddict = {"time":times2,
             "energies":energies,
             "L":Ls,
             "Bmirror":Bm
            }
    data = json.dumps(ddict)

    app = create_app(test_config="test_config")

    response = app.test_client().post("/shells_io_L", data=data, content_type='application/json')
    temp = response.json
    #print(temp)
    print(np.shape(temp['E flux 200'][:]))

    for pco in range(0,len(energies)):
        col = 'E flux '+str(energies[pco])
        # Have to get the two datasets in the right format
        test_one = test_data[col][:]
        app_one = np.array(temp[col][:])

        assert np.sum(np.abs(test_one-app_one))<.001

def test_A_HAPI_request(client):
    # TEST: Test that the HAPI request returns the right poes data
    #
    from Docker import process_inputs as pi
    sdate = dt.datetime(2023,8,17,0,0,0)
    times = [sdate+dt.timedelta(minutes=x*5) for x in range(0,256)]
    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times]
    load_dotenv(".env", verbose=True)
    server = os.environ.get('HAPI_SERVER')
    dataset = os.environ.get('HAPI_DATASET')
    hdata = pi.read_hapi_inputs(times2,server,dataset)

    # Check that data is returned
    dl,dw = np.shape(hdata['mep_ele_tel90_flux_e1'][:])
    if (dl>0) & (dw==29):
        test=1

    assert test==1

def test_with_hapi_data(client):
    # TEST: Check that the output from the app matches the expected output
    # from running process_POES with nn = True
    # The code to create the test output file is called make_shells_testoutput.py
    # and it creates a files called shells_neural20220101.nc which has the
    # expected output for 20220101 and n15
    print('Testing that the app works with data from HAPI')

    # First read in the netcdf file that the test will compare to
    #fname = 'shells_neural20220101.nc'
    #test_data = nc4.Dataset(fname, 'r')

    # Then create a list of times from the test file
    sdate = dt.datetime(2023,8,17,0,0,0)
    times1 = [sdate+dt.timedelta(minutes=x*5) for x in range(0,256)]

    times2 = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in times1]

    # The x,y,z and pitch angels won't matter for this test because
    # when in test mode, it forces the L value to be whatever is in the
    # test_config.py TESTL variable instead of using the magephem service.
    # xyz just needs to be the right len

    # Create a list of locations lists
    xyz = list()
    for tco in range(0,len(times2)):
        xyz.append([4,1,0])
    energies = [200]

    # Just give one pitch angle value
    ddict = {"time":times2,
             "energies":energies,
             "xyz":xyz,
             "sys": "GEO",
             "pitch_angles":[40]
            }
    data = json.dumps(ddict)

    app = create_app(test_config="test_hapi_config") # Create the app in test mode

    response = app.test_client().post("/shells_io",data=data,content_type='application/json')
    temp = response.json

    # Check that it is the right len etc
    # This is using a fixed location and it is using the test mode
    # for magephem so it returns a fixed L and Bmirror for 1 energy and Bmirror
    # Returns e flux that is len (256)
    # This just checks that the right len data is returned
    if len(temp['E flux 200'])==256:
        lcheck=1

    assert lcheck == 1

    print('Here')
