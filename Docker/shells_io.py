import json
import requests
import os

import numpy as np
import process_inputs as pi
from flask.views import MethodView
from flask_smorest import Blueprint
from flask import current_app
from schemas import InSchema
from schemas import InLSchema

blp = Blueprint("I/O", __name__, description="Operations for the shells model")

def integ_exp(temp_flux,Es):
    '''
    PURPOSE: To integrate electron flux assuming an exponential spectrum
    :param temp_flux:
    :param Es: list of energies (keV)
    :return:
    '''
    r,c = np.shape(temp_flux)
    J0 = np.zeros((r,c ), dtype=float)  # This is J0 for the exponential
    E0 = np.zeros((r, c), dtype=float)  # This is E0 for the exponential
    Jint = np.zeros((r, c), dtype=float)
    integral = np.zeros((r), dtype=float)
    for co in range(0,c-1):
        E0[:, co] = (Es[co] - Es[co + 1]) / (np.log(temp_flux[:, co + 1]/np.log(temp_flux[:, co])))
        # Check for Nans and set them to 0
        J0[:, co] = temp_flux[:, co] * np.exp(Es[co] / E0[:, co])
        Jtemp = J0[:, co] * E0[:, co] * np.exp(-1.0 * Es[co] / E0[:, co]) \
                - J0[:, co] * E0[:, co] * np.exp(-1.0 * Es[co + 1] / E0[:, co])
        jinds = np.where(np.isnan(Jtemp))
        Jtemp[jinds[0]] = 0
        jinds = np.where(Jtemp < 0)
        Jtemp[jinds[0]] = 0
        Jint[:, co] = Jtemp
        integral[:] += Jint[:, co]
    return integral

@blp.route("/shells_io")
class IOList(MethodView):
    @blp.arguments(InSchema)
    @blp.response(200)
    # InSchema is defined in schema.py
    # There is no fixed output schema in blp.response because the output varies
    def post(self, io_data):
        """Provides SHELLS model electron flux for a set of times, locations,
         local pitch angles, and energies

        Returns electron flux from the SHELLS model (valid between L=3-6.3)
        for input times, locations, local pitch angles, and energies (200-3000 kev)<br>

        Required inputs:<br>
        - time: list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'
        - xyz: list of 3-D locations for each time
        - sys: coordinate system for xyz locations
            + Supports GDZ, GEO, GSM, GSE, SM, GEI, MAG, SPH.
            + GDZ - geodetic as alt (km), latitude (deg), longitude (deg).
            + GEO - Cartesian geographic (RE).
            + GSM - Cartesian geocentric solar magnetospheric (RE).
            + SM - Cartesian solar magnetospheric (RE).
            + GEI - Cartesian geocentric Earth inertical (RE).
            + MAG - Cartesian magnetic.
            + SPH - Spherical geographic coordinates as radius (RE), latitude (deg), longitude (deg).
        - pitch_angles: 1-D list of local pitch angles (0-90 deg) for the returned electron flux
            or [-1] for omnidirectional flux
        - energies: 1-D list of energies (between 200-3000 keV) for the returned electron flux
            or a single negative energy for integral flux above that energy i.e. [-200]<br>
        <br>
        Outputs: (dictionary of arrays that includes user inputs) <br>
        - time: the same input list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'
        - xyz[time]: the same input list of 3-D locations for each time
        - pitch_angles: same as input list
        - Energies: same as input list
        - Bmirrors[time,pitch_angles]: Bmirror values for requested pitch angles and locations
        - L[time,pitch_angles]: L shells for requested pitch angles and locations
        - Kp[time]: Kp value for each time from the CCMC HAPI server
        - Kpmax[time]: The maximum Kp value in the last 3 days for each time step
        - E flux[time,pitch_angles,energies] electron flux #/cm2-s-str-keV as a function of
           time, pitch angles, and requested energies
           If omnidirectional flux is requested the returned E flux is a function of time and
           energy, i.e. E flux[time,energies] #/cm2-s-keV
           If integral flux is requested the returned E flux is as a function of time and
           pitch angle, i.e. E flux[time,pitch_angle] #/cm2-s-str
        - upper q [time, pitch_angles, energies] upper quartile of electron flux #/cm2-s-str-keV
           for each E flux
        - lower q [time, pitch_angles, energies] lower quartile of electron flux #/cm2-s-str-keV
           for each E flux
        """

        # The post method takes times, locations, pitch angles in io_data,
        # changes them to magnetic coordinates using magephem service
        # It calls process_data (in process_inputs.py) that gets the shells inputs
        # and then returns shells electron flux output for those input times,
        # locations and pitch angles

        # io_data should have
        # 1) list of times ( ex ['2022-01-01T00:00:00.000Z','2022-01-01T00:00:00.000Z']
        # 2) list of [x,y,z] for those times (ex [[3,1,1],[3,1,2]])
        # 3) list of energies (to be used for all times) (ex [200,400,600] in keV)
        # 4) list of pitch angles (to be used for all times) (ex [[20,40]])
        # 5) sys coordinate system

        # First check if the pitch angles are [-1]
        # In that case, return omni flux integrated over pitch angles by
        # first getting the flux at a fixed set of pitch angles and setting a flag
        # to integrate the returned shells flux
        if io_data["pitch_angles"][0] == -1:
            pstep=5
            #pstep=20
            io_data["pitch_angles"] = list(range(pstep,91,pstep))
            omni = True
        else:
            omni = False

        # Check if integral flux is needed which is signified by a single negative Energy
        if io_data["energies"][0]<0:
            # Create a list of energies in log space from the given value to 3000 keV
            # Make sure there is reasonable sampling from the passed energy
            # up to the max energy (3000 keV) and at least two energies to integrate
            estep=.15
            #estep = .5
            if (np.log10(3000)-np.log10(np.abs(io_data['energies'][0])))<=estep:
                Es = [np.abs(io_data['energies'][0]),3000]
            else:
                Es = list(10**np.arange(np.log10(np.abs(io_data['energies'][0])), np.log10(3000),estep))
                # Set the last point to the last valid energy (3000)
                if Es[-1]<3000:
                    Es = Es +[3000]
            io_data["energies"] = Es
            integral = True # Set a flag to do the integration at the end
        else:
            integral = False

        # "MAGEPHEM" is defined in the .env file test and in the top dir
        url = os.environ.get("MAGEPHEM")
    
        # User inputs date/time, x,y,z and pitch angles
        # Create the json input structure for magephem request
        magephem_input = {
            # list of dates in YYYY-MM-DDTHH:MM:SS.mmm(uuu)Z format
            # example: ["2022-01-10T17:05:00.967250Z", "2022-01-10T18:18:00.967250Z"]
            "dates": io_data["time"],

            # list of input 3-D coordinate sets (nested list)
            # example: [[1,2,3],[4,5,6]]
            "X": io_data["xyz"],
            "sys": io_data["sys"],

            # scalar input pitch angle
            # example: [10.0, 45.0, 90.0]
            # If pitch angle is a single value (i.e. 10)
            # Then it uses that value for all the times
            # If it's a 1D list [10,45] then it assumes one value for each time
            # If its 2D [[10,45]] then it assumes its both for each time step
            # We pass a 2D list so it alwyas returns a list of lists for Bm and L
            # even if there is just one pitch angle passed

            "alpha": [io_data["pitch_angles"]],
            "kext": "opq",
            "outputs": ["Bm", "L"]
            }

        # Convert x,y,z to Ls, Bm using magephem request:
        # (time1,x1,y1,z1) -> (L1,Bm1,Bm2)
        # (time2,x2,y2,z2) -> (L2,Bm1,Bm2)
        # The output json data will have L and Bm for each time

        if current_app.config["TESTING"]:
            # For testing, make up a set of L values and Bm values that
            # work for a fixed output testing file so we can test with
            # the serverless flask and not use the magephem service
            # make_shells_testoutput.py was used to create the test output
            # It calls process_SHELLS which creates outputs for
            # Ls = np.arange(3., 6.3, .25) and
            # Bmirrors = [np.floor(2.591e+04 * (L ** -2.98)) for L in Ls]
            # And it runs it for N15 for (2022,1,1),dt.datetime(2022,1,2)
            # The time cadence of the file is at that of NOAA 15

            magephem_response = {}
            # The TESTL value is set in the test_config.py
            Lval = current_app.config["TESTL"] # Its set to 4 for now
            Bmirror = np.floor(2.591e+04 * (Lval ** -2.98))
            # Create a faxe Bmirror response
            if len(io_data["pitch_angles"])>1:
                Bvals = [Bmirror] * len(io_data["pitch_angles"])
                Lvals = [Lval]* len(io_data["pitch_angles"])
                magephem_response["Bm"] = [Bvals] * len(io_data["time"])
                magephem_response["L"] = [Lvals] * len(io_data["time"])
            else:
                magephem_response["Bm"] = [[Bmirror]]*len(io_data["time"])
                magephem_response["L"] = [[Lval]] * len(io_data["time"])
        else:

            resp = requests.post(url, json=magephem_input)
            magephem_response = resp.json()

            # Check the magephem response and if it gets a 500 or other errors
            # then exit with error code
            if resp.status_code != 200:
                # The magephem "hardcoded" post responses are 200 ("successful operation") and 405 ("Invalid input")
                # If an error occurs, we'll check first if the response is 405. If not, we'll check the other
                # "standard" responses
                if resp.status_code == 405:
                    error_desc = "magephem error 405: Invalid input"
                else:
                    error_desc = "magephem error {}: {}".format(magephem_response['status'], magephem_response['title'])
                return error_desc

        # Here we pass a list of times, a list of Ls for each time i.e. [[L1]],
        # and a list of [Bms] for each time [[Bm]]
        # Bm and L will have multiple values if multiple pitch angles are requested
        # and the list of energies requested
        #print('Processing data')
        output = pi.process_data(io_data["time"], magephem_response["L"], magephem_response["Bm"], io_data["energies"])

        # The output will be a json dict
        # output has time, L, E flux[time,pa,Energy], Kp,Kpmax
        # return the user reqested pitch angles as well
        # outdat['L'], outdat['Bmirrors'], outdat['Energies'], Kpmax, Kp are all set in process_data

        # For the omni case the pitch_angles returned are the ones used for integration
        # For the integral case, the energies returned are the ones used for integration
        output['pitch_angles']= io_data["pitch_angles"]
        eflux_fill=-1.0e31
        if omni ==True:
            # Integrate the flux and quartiles over pitch angles
            for col in ['E flux','upper q','lower q']:
                oflux = np.full((len(output['time']), len(output['Energies'])), dtype=float, fill_value=eflux_fill)
                temp_flux = np.array(output[col][:])
                for eco,E in enumerate(output['Energies']):
                    # Do the pitch angle integration for each energy
                    # If a pitch angle is in the loss cone, magephem returns -e31.
                    # The flux for those points is set to 0 after the neural network runs
                    # If it is out of bounds then it is -e31
                    oflux[:,eco] = (pstep*4*np.pi*np.sum((temp_flux[:,:,eco]),axis=1))
                    # If any are negative then flag the omni
                    badinds = np.any(temp_flux[:,:, eco]<0,axis=1)
                    oflux[badinds,eco]=eflux_fill
                output[col] = oflux.tolist()

        # Check if we also need an integral flux
        if (integral ==True):
            # Assume an exponential spectrum for the flux
            for col in ['E flux', 'upper q', 'lower q']:
                temp_flux = np.array(output[col][:])
                # If its omnidirectional then E flux is [time,energies]
                if len(np.shape(output[col][:]))==2:
                    #oflux = np.full((len(output['time'])), dtype=float, fill_value=-1)
                    oflux = integ_exp(temp_flux, output['Energies'])
                    badinds = np.any(temp_flux[:,:]<0,axis=1)
                    oflux[badinds]=eflux_fill
                else:
                    # Otherwise it is [time,pa,E]
                    oflux = np.full((len(output['time']), len(output['pitch_angles'])), dtype=float, fill_value=eflux_fill)
                    for pco,p in enumerate(output['pitch_angles']):
                        oflux[:,pco] = integ_exp(temp_flux[:,pco,:],output['Energies'])
                        badinds = np.any(temp_flux[:,pco, :] < 0, axis=1)
                        oflux[badinds,pco] = eflux_fill

                output[col] = oflux.tolist()

        print('Here')

        # To make this work and return the output dict as a json
        # object there is no defined schema for the response
        # We can't use a schema because the data cols in the returned ouput
        # dict depend on the requests, so it can't have a fixed schema
        # If you pass the output any other way then it is hard to access

        return output

@blp.route("/shells_io_L")
class IOList(MethodView):
    @blp.arguments(InLSchema)
    @blp.response(200)
    # InSchema is defined in schema.py
    # There is no fixed output schema in blp.response because the output varies
    def post(self, io_data):
        """Provides electron flux from the SHELLS model

        Returns electron flux from the SHELLS model for input times, Ls,
        Bmirrors, and energies<br>

        Required inputs:<br>
        - time: list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'
        - Ls: 1-D list of L shells for returned electron flux (ex [5,6])
        - Bmirrors: 1-D list of mirror point magnetic fields (nT) for each L shell.
        (ex [100,150]).
        - energies: 1-D list of energies (between 200-3000 keV) for the returned electron flux<br>
        <br>
        Outputs: (dictionary of arrays that includes user inputs) <br>
        - time: the same input list of dates and times with format 'YYYY-MM-DDTHH:MM:SS.fffuuuZ'
        - xyz: the same input list of 3-D locations for each time
        - energies: same as input values
        - Bmirrors[time,L]: Bmirror values (nT) for the requested pitch angles and locations
        - L[time,Bmirrors]: L shells for the requested pitch angles and locations
        - Kp[time]: Kp value for each time from the CCMC HAPI server
        - Kpmax[time]: The maximum Kp value in the last 3 days
        - E flux[time,L,energies] electron flux #/cm2-s-str-keV as a function of
           time, Bmirror/L, and requested energies
        - upper q [time, L,energies] upper quartile of the electron flux #/cm2-s-str-keV
        - lower q [time, L, energies] lower quartile of the electron flux #/cm2-s-str-keV
        """

        # The post method takes times, Lshells, Bmirros  in io_data,
        # It calls process_data (in process_inputs.py) that gets the shells inputs
        # and then returns shells electron flux output for those input times,
        # locations and pitch angles

        # io_data should have
        # 1) list of times ( ex ['2022-01-01T00:00:00.000Z','2022-01-01T00:00:00.000Z']
        # 2) list of [Lshells] to be used for all those times (ex[5,6])
        # 3) list of energies (to be used for all times) (ex [200,400,600] in keV)
        # 4) list of Bmirrors (to be used for all times) (ex [20,40])

        # In the other case we have a list of Ls and Bms for each time that are len(pitchangles)
        # In this case we have to replicate the L shells for each time
        invals = {}
        invals["L"] =[io_data["L"]]*len(io_data["time"])
        invals["Bmirror"] = [io_data["Bmirror"]] * len(io_data["time"])

        output = pi.process_data(io_data["time"], invals["L"], invals["Bmirror"], io_data["energies"])

        # To make this work and return the output dict as a json
        # object there is no defined schema for the response
        # We can't use a schema because the data cols in the returned ouput
        # dict depend on the energies requested, so it can't have a fixed schema
        # If you pass the output any other way then it is hard to access
        return output

