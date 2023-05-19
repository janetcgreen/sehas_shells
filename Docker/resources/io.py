import json
import requests
import os
import uuid

import numpy as np
import process_inputs as pi
from flask.views import MethodView
from flask_smorest import Blueprint
from schemas import IOSchema

blp = Blueprint("I/O", __name__, description="Operations on I/O db")


@blp.route("/io")
class IOList(MethodView):
    @blp.arguments(IOSchema)
    @blp.response(201, IOSchema)
    def post(self, io_data):

        url = os.environ.get("MAGEPHEM")

        # User inputs date/time, x,y,z and pitch angles
        # Create the right json input structure for magephem request
        magephem_input = {
            # list of dates in YYYY-MM-DDTHH:MM:SS.mmm(uuu)Z format
            # example: ["2022-01-10T17:05:00.967250Z", "2022-01-10T18:18:00.967250Z"]
            "dates": io_data["time"],

            # list of input 3-D coordinate sets (nested list)
            # example: [[1,2,3],[4,5,6]]
            "X": io_data["xyz"],

            # scalar input pitch angle
            # example: [10.0, 45.0, 90.0]
            "alpha": io_data["pitch_angles"],
            "kext": "opq",
            "sys": "GDZ",
            "outputs": ["Bm", "L"]
        }

        # Convert x,y,z to Ls, Bm using magephem request:
        # (time1,x1,y1,z1) -> (L1,Bm1)
        # (time2,x2,y2,z2) -> (L2,Bm2)
        # The output json structure will have L and Bm for each time
        try:
            magephem_response = requests.post(url, json=magephem_input).json()
            print('magephem_response: ', magephem_response)
        except Exception as e:
            print('e : ', e)

        output = pi.process_data(io_data["time"], magephem_response["L"], magephem_response["Bm"], io_data["energies"])

        # Pretty-Print JSON
        json_output = {**io_data, "json_output": output}

        # Write Pretty-Print JSON data to file
        with open("output.json", "w") as write_file:
            json.dump(output, write_file, indent=4)

        return json_output