import json
import uuid

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
        energies = [float(idx) for idx in io_data["Energies"].split(', ')]
        print('energies: ', energies)

        Ls = [4., 4.]
        print('Ls: ', Ls)

        output = pi.process_data(time, Ls, energies)

        # Pretty-Print JSON
        json_data = {**io_data, "json_data": json.dumps(output, indent=0)}

        # Write Pretty-Print JSON data to file
        with open("output.json", "w") as write_file:
            json.dump(output, write_file, indent=4)

        return json_data
