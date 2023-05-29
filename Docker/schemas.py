from marshmallow import Schema, fields


class InSchema(Schema):
    # Inputs to SHELLS app
    # There are two ways the app can be run
    # 1) with a list of requested times,locations for a set of energies and local PAs
    #    The locations will be changed to an L and each PA changed to Bm
    # 2) with a list of requested times and a list of energies,  Ls, and Bmirrors
    time = fields.List(fields.Str(), required=True)  # list of times (dt format)
    xyz = fields.List(fields.List(fields.Float()), required=True)  # list of  x,y,z corresponding to each time (floats)
    sys = fields.Str(required =True)
    energies = fields.List(fields.Float(), required=True)  # list of energies (floats)
    pitch_angles = fields.List(fields.Float(), required=True)  # list of pitch angles (floats)
    # outputs
    # L = fields.Str(required=False)  # based on x,y,z
    # Bmirrors = fields.Str(required=False) # based on x,y,z

class OutSchema(Schema):
    output = fields.Str(required=False)



