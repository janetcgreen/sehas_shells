from marshmallow import Schema, fields


class IOSchema(Schema):
    # inputs
    time = fields.Str(required=True)  # dt format:  list of times
    x = fields.Float(required=True)  # list of x corresponding to each time
    y = fields.Float(required=True)  # list of y corresponding to each time
    z = fields.Float(required=True)  # list of  z corresponding to each time
    Energies = fields.Str(required=True)  # list of floats
    pitch_angle = fields.Float(required=True)  # list
    # outputs
    L = fields.Str(required=False)  # based on x,y,z
    Bmirrors = fields.Str(required=False)
