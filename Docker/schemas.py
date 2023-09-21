from marshmallow import Schema, fields, validate, ValidationError
import datetime as dt

class BaseSchema(Schema):
    class Meta:
        ordered = True
def in_range(value):
    for val in value:
        if ((val>90)| (val<-1)):
            raise ValidationError("pitch_angles exceed valid range (-1 to 90)")

def E_in_range(value):
    for val in value:
        if ((abs(val)>3000)| (abs(val)<200)):
            raise ValidationError("energy outside valid range (200 to 3000)")

def L_in_range(value):
    for val in value:
        if ((val<3.0)| (val>6.3)):
            raise ValidationError("L outside valid range (3 to 6.3)")

def ascending(value):
    if len(value)>1:
        res=all(i < j for i, j in zip(value[:], value[1:]))
        if res is False:
            raise ValidationError("Values must be ascending")
def checktime(value):
    try:
        test = dt.datetime.strptime(value[0],'%Y-%m-%dT%H:%M:%S.%fZ')
    except:
        raise ValidationError("Time format must be YYYY-MM-DDTHH:MM:SS.fffuuuZ")
class InSchema(BaseSchema):

    # Inputs to SHELLS app
    # There are two ways the app can be run
    # 1) with a list of requested times,locations for a set of energies and local PAs
    #    The locations will be changed to an L and each PA changed to Bm
    # 2) with a list of requested times and a list of energies,  Ls, and Bmirrors

    time = fields.List(fields.Str(), required=True,
                description="list of dates and times (10,000 point limit) with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'",
                example=["2022-01-01T00:00:00.000Z","2022-01-01T01:00:00.000Z"],
                validate=validate.And(validate.Length(min=1, max=10000),
                                      checktime))  # list of times (dt format)
    xyz = fields.List(fields.List(fields.Float()), required=True,
                description = "list of 3-D locations for each time (10,000 point limit)",
                example=[[4,1,1],[4.5,1,1]],
                validate=validate.Length(min=1, max=10000))  # list of  x,y,z corresponding to each time (floats)
    sys = fields.Str(required =True,
                description="coordinate system for locations: GDZ,GEO,GSM,GSE,SM,GEI,MAG,SPH",
                example = 'GEO',
                validate=validate.And(validate.Length(min=1, max=3,error='Invalid coordinate system'),
                          validate.OneOf(choices=['GDZ','GEO','GSM','GSE','SM','GEI','MAG','SPH'],
                                         error="Invalid Cooridnate System")))
    energies = fields.List(fields.Float(), required=True,
                description="1-D list of energies (200-3000 keV, max 15 values) for the returned electron flux"
                            "with increasing values or a negative energy for integral flux above that value",
                example=[500,600,700,1000],
                validate=validate.And(validate.Length(min=1, max=15), E_in_range, ascending))# list of energies (floats)
    pitch_angles = fields.List(fields.Float(),
                required=True,
                description="1-D list of local pitch angles (0-90 degrees, max 15 values) for the returned electron flux"
                            "with increasing values or a [-1] for omnidirecional flux",
                example = [50,60,70],
                validate=validate.And(validate.Length(min=1, max=15), in_range, ascending)) # list of pitch angles (floats)
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
                description="list of dates and times (10,000 point limit) with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'",
                example=["2022-01-01T00:00:00.000Z","2022-01-01T01:00:00.000Z"],
                validate=validate.And(validate.Length(min=1, max=10000),checktime))  # list of times (dt format)
    L = fields.List(fields.Float(), required=True,
                description = "Fixed 1-D list of L shells (3.0-6.3, 15 max values) for the returned electron flux",
                example=[3.,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.,5.25,5.5,5.75,6.,6.25],
                validate=validate.And(validate.Length(min=1, max=15),L_in_range))
    Bmirror = fields.List(fields.Float(), required=True,
                description="Fixed list of mirror point magnetic fields for each L shell."
                "If a 1-D list is passed then the Bmirror must correspond to each L shell"
                "(ex [5000,6000])."
                "If a 2-D list is passed then multiple Bmirror can be set for each L shell"
                "(ex [[5000,6000],[5000,6000]]",
                example = [980.0,772.0,619.0,504.0,416.0,347.0,293.0,249.0,214.0,185.0,
                            161.0,141.0,124.0,110.0])
    energies = fields.List(fields.Float(), required=True,
                description="1-D list of energies (200-3000 keV, max 15 values) for the returned electron flux"
                            "with increasing values or a negative energy for integral flux above that value",
                example=[500,600,700,1000],
                validate=validate.And(validate.Length(min=1, max=15),E_in_range,ascending) ) # list of energies (floats)






