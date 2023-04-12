import uuid

import numpy as np
import process_inputs as pi
from flask.views import MethodView
from flask_smorest import Blueprint
from schemas import IOSchema

blp = Blueprint("I/O", __name__, description="Operations on I/O db")


@blp.route("/io")
class IOList(MethodView):
    @blp.response(200, IOSchema(many=True))
    def get(self):
        # Ls and Energies
        Ls = np.array([4])
        Energies = np.arange(200., 3000., 200.)

        output = pi.process_data('2022-01-10T20:16:20.967250Z', '2022-01-10T20:16:21.967250Z', Ls, Energies)
        print(output)

        return output

    @blp.arguments(IOSchema)
    @blp.response(201, IOSchema)
    def post(self, io_data):
        io_id = uuid.uuid4().hex
        io = {**io_data, "id": io_id}

        # time="2022-01-10T20:16:20.967250Z, 2022-01-10T20:16:21.967250Z"
        # convert string to dt list using list comprehension + split() + datetime.strptime()
        # from datetime import datetime
        # time_dt = [datetime.strptime(idx, '%Y-%m-%dT%H:%M:%S.%fZ') for idx in time_str]
        time = list(io_data["time"].split(', '))
        print('time: ', time)

        # energies = np.arange(200., 3000., 200.)
        # energies = "200., 400., 600., 800., 1000., 1200., 1400., 1600., 1800., 2000., 2200., 2400., 2600., 2800."
        # convert string to float list using list comprehension + split() + float()
        energies = [float(idx) for idx in io_data["Energies"].split(', ')]
        print('energies: ', energies)

        Ls = np.array([4])
        print('Ls: ', Ls)

        output = pi.process_data(min(time), max(time), Ls, energies)
        print('THE END! ', output)

        return output
