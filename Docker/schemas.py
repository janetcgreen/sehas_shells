from marshmallow import Schema, fields


class IOSchema(Schema):
    # inputs
    time = fields.Str(required=True)  # dt format:  list of times
    x = fields.Float(required=True)  # list of x corresponding to each time
    y = fields.Float(required=True)  # list of y corresponding to each time
    z = fields.Float(required=True)  # list of  z corresponding to each time
    xyz = fields.List(fields.List(fields.Float()), required=True)  # list of  x,y,z corresponding to each time (floats)
    Energies = fields.List(fields.Float(), required=True)  # list of energies (floats)
    pitch_angles = fields.List(fields.Float(), required=True)  # list of pitch angles (floats)
    # outputs
    # L = fields.Str(required=False)  # based on x,y,z
    # Bmirrors = fields.Str(required=False) # based on x,y,z
    json_output = fields.Str(required=False)
