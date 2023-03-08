import numpy as np
from marshmallow import Schema, fields


class PropSchema(Schema):  # inputs and outputs
    id = fields.Str(dump_only=True)
    time = fields.Int(required=True)  # string -> dt
    x = fields.Float(required=True)
    y = fields.Float(required=True)
    z = fields.Float(required=True)
    Lshell = fields.Float(required=False)  # output
    energy = fields.Float(required=True)
    pitch_angle = fields.Float(required=True)
    Bmirror = fields.Float(required=False)  # output
    sat_id = fields.Str(required=True)


class PropUpdateSchema(Schema):
    # Turn each xyz location into Lshell (this will be done by calling  the Aerospace code.
    # To start we will just set a fixed Lshell value for each location (L=4))
    Lshell = 4

    # Turn pitch angle into magnetic field strength.
    # For now we will assume fixed values for each location
    Bmirror = np.floor(2.591e+04 * (Lshell ** -2.98))


class SatelliteSchema(Schema):
    id = fields.Str(dump_only=True)
    name = fields.Str(required=True)
