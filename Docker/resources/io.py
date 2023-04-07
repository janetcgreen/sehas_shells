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

        Ls = np.array([4])
        Energies = np.arange(200., 3000., 200.)

        time = list(io_data["time"].split(','))
        # from datetime import datetime
        # time_dt = [datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ') for time in time_str]

        output = pi.process_data(min(time), max(time), Ls, Energies)
        print('THE END! ', output)

        return output
