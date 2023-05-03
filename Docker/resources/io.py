import json
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
        io_id = uuid.uuid4().hex
        io = {**io_data, "id": io_id}

        # req_times="2022-01-10T17:05:00.967250Z, 2022-01-10T18:18:00.967250Z, 2022-01-10T19:11:00.967250Z, 2022-01-10T20:20:00.967250Z"
        # convert string to dt list using list comprehension + split() + datetime.strptime()
        # from datetime import datetime
        # time_dt = [datetime.strptime(idx, '%Y-%m-%dT%H:%M:%S.%fZ') for idx in time_str]
        time = list(io_data["time"].split(', '))
        print('req times: ', time)

        # energies = np.arange(200., 3000., 200.)
        # energies = "200., 400., 600., 800., 1000., 1200., 1400., 1600., 1800., 2000., 2200., 2400., 2600., 2800."
        # convert string to float list using list comprehension + split() + float()
        # energies = [float(idx) for idx in io_data["Energies"].split(', ')]
        energies = io_data["Energies"]
        print('energies: ', energies)

        Ls = [4., 4.]
        print('Ls: ', Ls)

        Bmirrors = [np.floor(2.591e+04 * (L ** -2.98)) for L in Ls]
        print('Bmirrors: ', Bmirrors)

        # User inputs time,X,Y,Z,pitch angles
        # Create the right json input structure
        url = 'http:/localhost:23760'
        data_input = {
            # list of dates in YYYY-MM-DDTHH:MM:SS.mmm(uuu)Z format
            # example: ["2020-12-31T12:59:49.0Z", "2020-12-31T12:59:59.0Z"]
            'dates': time,
            # list of input 3-D coordinate sets (nested list)
            # example: [[1,2,3],[4,5,6],[7,8,9]]
            'X': io_data["xyz"],
            # scalar input pitch angle
            # example: [10.0, 45.0, 90.0]
            'alpha': io_data["pitch_angles"]
        }
        # Convert into json
        json_input = json.dumps(data_input)
        # Write Pretty-Print JSON data to file
        with open("input.json", "w") as write_file:
            json.dump(data_input, write_file, indent=4)

        # # Change X,Y,Z to Ls, Bm using magephem and python requests
        # # (time1,x1,y1,z1) -> (L1,Bm1)
        # # (time2,x2,y2,z2) -> (L2,Bm2)
        # # The output json structure will have L and Bm for each time
        # output = requests.post(url, json=json_input)
        #
        # # parse output, the result is a Python dictionary
        # data_output = json.loads(output)

        # output = pi.process_data(time, data_output["L"], data_output["Bm"], energies)
        output = pi.process_data(time, Ls, Bmirrors, energies)

        # Pretty-Print JSON
        json_output = {**io_data, "json_output": json.dumps(output)}

        # Write Pretty-Print JSON data to file
        with open("output.json", "w") as write_file:
            json.dump(output, write_file, indent=4)

        return json_output
