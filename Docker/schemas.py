from marshmallow import Schema, fields


class IOSchema(Schema):
    # inputs
    time = fields.List(fields.Str(), required=True)  # list of times (dt format)
    xyz = fields.List(fields.List(fields.Float()), required=True)  # list of  x,y,z corresponding to each time (floats)
    energies = fields.List(fields.Float(), required=True)  # list of energies (floats)
    pitch_angles = fields.List(fields.Float(), required=True)  # list of pitch angles (floats)

    # outputs
    # L = fields.Str(required=False)  # based on x,y,z
    # Bmirrors = fields.Str(required=False) # based on x,y,z
    json_output = fields.Str(required=False)
