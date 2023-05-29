import json
import requests
import os

import numpy as np
import process_inputs as pi
from flask.views import MethodView
from flask_smorest import Blueprint
from flask import current_app
from schemas import InSchema

blp = Blueprint("I/O", __name__, description="Operations for the shells model")

@blp.route("/shells")
class IOList(MethodView):
    @blp.arguments(InSchema)
    @blp.response(201)
    # InSchema is defined in schema.py
    # There is no fixed output schema in blp.response because the output varies
    def post(self, io_data):
        # The post method takes times, locations, pitch angles in io_data,
        # changes them to magnetic coordinates using magephem service
        # It calls process_data (in process_inputs.py) that gets the shells inputs
        # and then returns shells electron flux output for those input times,
        # locations and pitch angles

        # io_data should have
        # 1) list of times ( ex ['2022-01-01T00:00:00.000Z','2022-01-01T00:00:00.000Z']
        # 2) list of [x,y,z] for those times (ex [[3,1,1],[3,1,2]])
        # 3) list of energies (to be used for all times) (ex [200,400,600] in keV)
        # 4) list of pitch angles (to be used for all times) (ex [[20,40]])
        # Todo: Later we will likely update this so you can pass Ls and Bms
        # instead of xyz and pitch angles

        # "MAGEPHEM" is defined in the .env file test and in the top dir
        url = os.environ.get("MAGEPHEM")

        # Check if the user input xyz (or list of Ls and Bms)
        if 'xyz' in io_data:
            # User inputs date/time, x,y,z and pitch angles
            # Create the json input structure for magephem request
            magephem_input = {
                # list of dates in YYYY-MM-DDTHH:MM:SS.mmm(uuu)Z format
                # example: ["2022-01-10T17:05:00.967250Z", "2022-01-10T18:18:00.967250Z"]
                "dates": io_data["time"],

                # list of input 3-D coordinate sets (nested list)
                # example: [[1,2,3],[4,5,6]]
                "X": io_data["xyz"],

                # scalar input pitch angle
                # example: [10.0, 45.0, 90.0]
                # If pitch angle is a single value (i.e. 10)
                # Then it uses that value for all the times
                # If it's a 1D list [10,45] then it assumes one for each time
                # If its 2D [[10,45]] then it assumes its both for each time step
                # We pass a 2D list so it alwyas returns a list of lists for Bm and L
                # even if there is just one pitch angle passed
                # Todo allow sys to be passed by the user
                # Todo check what kext should be
                "alpha": [io_data["pitch_angles"]],
                "kext": "opq",
                "sys": io_data["sys"],
                "outputs": ["Bm", "L"]
            }

            # Convert x,y,z to Ls, Bm using magephem request:
            # (time1,x1,y1,z1) -> (L1,Bm1,Bm2)
            # (time2,x2,y2,z2) -> (L2,Bm1,Bm2)
            # The output json data will have L and Bm for each time

            if current_app.config["TESTING"]:
                # For testing we make up a set of L values and Bm values
                # make_shells_testoutput.py was used to create the test output
                # It calls process_SHELLS which creates outputs for
                # Ls = np.arange(3., 6.3, .25) and
                # Bmirrors = [np.floor(2.591e+04 * (L ** -2.98)) for L in Ls]
                # And it runs if for N15 for (2022,1,1),dt.datetime(2022,1,2)
                # The output file is at the time cadence of NOAA 15
                # Check what pitch angles are passed and create magephem_reponses
                # that mathc

                magephem_response = {}
                # The TESTL value is set in the test_config.py
                Lval = current_app.config["TESTL"]
                magephem_response["L"] = [[Lval]]*len(io_data["time"])
                Bmirror = np.floor(2.591e+04 * (Lval ** -2.98))

                if len(io_data["pitch_angles"])>1:
                    Bvals = [Bmirror] * len(io_data["pitch_angles"])
                    magephem_response["Bm"] = [Bvals] * len(io_data["time"])
                else:
                    magephem_response["Bm"] = [[Bmirror]]*len(io_data["time"])
            else:
                # Todo: check the response and if it gets a 500 or other errors
                # then it needs to exit
                magephem_response = requests.post(url, json=magephem_input).json()
        else:
            # Todo add the ability to pass Ls and Bms from the input
            pass

        # Here we pass a list of times, a list of Ls for each time,
        # and a list of [Bms] for each time
        # Bm will have multiple values if multiple pitch angles are requested
        # and the list of energies requeste
        output = pi.process_data(io_data["time"], magephem_response["L"], magephem_response["Bm"], io_data["energies"])

        # The output will be a json dict
        # Todo figure out what exactly should be output other than flux
        # output has time, L, E flux 200, etc

        # return the user reqested pitch angles
        output['pitch_angles']= io_data["pitch_angles"]

        # Write Pretty-Print JSON data to file
        #with open("output.json", "w") as write_file:
        #    json.dump(output2, write_file, indent=4)

        # To make this work and return the output dict as a json
        # object there is no defined schema for the response
        # We can't use a schema because the data cols in the returned ouput
        # dict depend on the energies requested, so it can't have a fixed schema
        # If you pass the output any other way then it is hard to access
        return output

