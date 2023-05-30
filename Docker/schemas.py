from marshmallow import Schema, fields

class InSchema(Schema):
    # Inputs to SHELLS app
    # There are two ways the app can be run
    # 1) with a list of requested times,locations for a set of energies and local PAs
    #    The locations will be changed to an L and each PA changed to Bm
    # 2) with a list of requested times and a list of energies,  Ls, and Bmirrors
    time = fields.List(fields.Str(), required=True,
                       description="list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'",
                       example=["2022-01-01T00:00:00.000Z","2022-01-01T01:00:00.000Z"])  # list of times (dt format)
    xyz = fields.List(fields.List(fields.Float()), required=True,
                      description = "list of 3-D locations for each time",
                      example=[[4,1,1],[4.5,1,1]])  # list of  x,y,z corresponding to each time (floats)
    sys = fields.Str(required =True,
                     description="coordinate system for xyz locations: GDZ,GEO,GSM,GSE,SM,GEI,MAG,SPH",
                     example = 'GEO')
    energies = fields.List(fields.Float(), required=True,
                           description="1-D list of energies (keV) for the returned electron flux",
                           example=[500,600,700,1000])  # list of energies (floats)
    pitch_angles = fields.List(fields.Float(), required=True,
                               description="1-D list of local pitch angles for the returned electron flux",
                               example = [50,60,70])  # list of pitch angles (floats)
    # outputs
    # L = fields.Str(required=False)  # based on x,y,z
    # Bmirrors = fields.Str(required=False) # based on x,y,z

class OutSchema(Schema):
    output = fields.Str(required=False)



