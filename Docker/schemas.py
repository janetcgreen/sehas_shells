from marshmallow import Schema, fields


class BaseSchema(Schema):
    class Meta:
        ordered = True

class InSchema(BaseSchema):
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
                           description="1-D list of energies (keV) for the returned electron flux"
                                       "or a negative energy for integral flux above that value",
                           example=[500,600,700,1000])  # list of energies (floats)
    pitch_angles = fields.List(fields.Float(), required=True,
                               description="1-D list of local pitch angles (degrees) for the returned electron flux"
                                           "or a [-1] for omnidirecional flux",
                               example = [50,60,70])  # list of pitch angles (floats)
    # outputs
    # L = fields.Str(required=False)  # based on x,y,z
    # Bmirrors = fields.Str(required=False) # based on x,y,z

class InLSchema(BaseSchema):
    # Inputs to SHELLS app
    # There are two ways the app can be run
    # 1) with a list of requested times,locations for a set of energies and local PAs
    #    The locations will be changed to an L and each PA changed to Bm
    # 2) with a list of requested times and a list of energies,  Ls, and Bmirrors

    time = fields.List(fields.Str(), required=True,
                       description="list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'",
                       example=["2022-01-01T00:00:00.000Z","2022-01-01T01:00:00.000Z"])  # list of times (dt format)
    L = fields.List(fields.Float(), required=True,
                      description = "Fixed 1-D list of L shells for the returned electron flux",
                      example=[3.,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.,5.25,5.5,5.75,6.,6.25])  # list of  x,y,z corresponding to each time (floats)
    Bmirror = fields.List(fields.Float(), required=True,
                               description="Fixed list of mirror point magnetic fields for each L shell."
                                           "If a 1-D list is passed then the Bmirror must correspond to each L shell"
                                           "(ex [5000,6000])."
                                           "If a 2-D list is passed then multiple Bmirror can be set for each L shell"
                                           "(ex [[5000,6000],[5000,6000]]",
                               example = [980.0,772.0,619.0,504.0,416.0,347.0,293.0,249.0,214.0,185.0,
                                          161.0,141.0,124.0,110.0])
    energies = fields.List(fields.Float(), required=True,
                           description="Fixed 1-D list of energies (keV) for the returned electron flux",
                           example=[500,600,700,1000])  # list of energies (floats)






