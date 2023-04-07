from marshmallow import Schema, fields


class IOSchema(Schema):
    # inputs
    time = fields.Str(required=True)  # dt format:  list of times
    x = fields.Float(required=True)  # list of x corresponding to each time
    y = fields.Float(required=True)  # list of y corresponding to each time
    z = fields.Float(required=True)  # list of  z corresponding to each time
    energy = fields.Float(required=True)  # list
    pitch_angle = fields.Float(required=True)  # list
    # outputs
    Lshell = fields.Float(required=False)  # based on x,y,z
    Bmirror = fields.Float(required=False)
