import argparse
import os
import datetime as dt
import requests
from propagator import Propagator
import sys
import json
import csv
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
app_path = os.path.join(os.path.dirname( __file__ ), '..','Docker')
sys.path.insert(0, app_path)  # take precedence over any other in path
from app import create_app

def valid_date(s):
    '''
    ---------------------------------------------------------------
    PURPOSE: To check that a valid date is entered as an input

    :param s (str) a date in the format Y-m-d or Y-m-d H:M:S
    ---------------------------------------------------------------
    '''

    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d")
        return test
    except:
        pass
    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return test
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)
def make_times(tsdate,tedate,tstep)  :
    t =tsdate
    times = []
    while t<tedate:
        times.append(t)
        t=t+dt.timedelta(minutes=tstep)

    return(times)

def get_TLES(url,group,format,sat):
    celes_url = url
    try:
        session = requests.Session()
        response = session.get(celes_url, params={
        "GROUP": group,
        "FORMAT": format
        })
        if response.status_code==200:
            TLES = response.iter_lines(decode_unicode=True)
            for line in TLES:
                if sat in line:
                    tle1 = next(TLES)
                    tle2 = next(TLES)
        else:
            tle1=None
            tle2=None
    except:
        tle1=None
        tle2=None

    return tle1,tle2

def get_GPS_start(fname):
    '''
    PURPOSE: To read the existing file of GPS data and get the last time processed
    :param fname:
    :return:
    '''
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ltime = row[0]

    return dt.datetime.strptime(ltime,'%Y-%m-%dT%H:%M:%S.%fZ')

def read_old_file(fname):
    '''
    PURPOSE to read in data from any existing file
    :param fname:
    :return:
    '''
    # First check if a file exists
    if os.path.isfile(fname):

        df = pd.read_csv(fname)
        odict = df.to_dict(orient='list')
    else:
        odict = None

    return odict

def getLpass(L,dist=200,prom=.5):
    '''
     Creates an arary with pass number for each data point that can be used to more easily average data or plot
     it pass by pass. Limit to between 0 and 30 because weird things happen at large L.

     :param L(data column)      L value that we are using to define a pass
     :param dist(int)           Required distance in datapoints between peaks
     :param prom (float)        The prominance defines how high it has to be to be considered a peak
     :return passes (list)      A list with passnumbers from 0 to ... for each dfatapoint
     :return allbreaks (list)   List of the indices that define the breaks between passes

    Usage: if the data is netcdf4 returned from poes_utils
     getLpass(poes['L_IGRF][:],dist=200,prom=.5):
     if the data is a numpy array
     getLpass(data['L_IGRF'][:],dist=200,prom=.5'''

    if isinstance(L, np.ma.MaskedArray):
        Ldata = L.data
    else:
        Ldata = L
    goodinds= np.where((Ldata[:]>0) & (Ldata[:]<30))[0]
    # This method works best for POES because at low L the values repeat so differencing
    # gives weird results
    peaks = find_peaks(Ldata[goodinds],distance=dist,prominence=prom) # Find the maxima
    valleys = find_peaks(-1*(Ldata[goodinds]),distance=dist,prominence=prom) # Find the minima
    #plt.plot(L.data[goodinds])
    #plt.plot(peaks[0],L.data[goodinds[peaks[0]]],'*')
    #plt.plot(valleys[0], L.data[goodinds[valleys[0]]], '+')
    allbreaks = np.sort(np.append(goodinds[peaks[0]], goodinds[valleys[0]]))
    #pbreaks = 0 * Ldata
    pbreaks = np.zeros((len(Ldata)),dtype=float)
    pbreaks[allbreaks] = 1
    passes = np.cumsum(pbreaks)

    return passes,allbreaks
def make_GPS_shells(sdate_all, edate, sat, sh_url, realtime=1,tstep=5,ndays = 7,
                    Es=[500,2000],outdir=os.getcwd(),outname ='GPS_SHELLS_',testing=1):
    '''
    PURPOSE: To create a plot an output file of SHELLS electron fluxes along a GPS satellite orbit.
    The code is expected to be run on a cron or other scheduler at a regular cadence (i.e. hourly)
    It outputs a rolling ndays (default 7) day file of electron fluxes that are plotted as an Lbin
    plot.

    In theory, the code could be run in realtime or reprocessing mode( with a fixed start and end date).
    However, it is currently only setup to run in realtime because the TLEs at celestrak used to
    create the GPS satellite trajectory are only the most recent one. A historical archive of TLEs is
    available through spacetrak but would require an account with a user name and password to be set up.

    In realtime mode (realtime=1) data are added to the output file up to the current date every
    time the code is called. It creates a rolling file that is always ndays long from the current time
    back ndays.

    :param sdate_all (datetime): start date for reprocessing data (not used currently)
    :param edate (datetime): end date for reprocessing data (not used currently)
    :param sat (str): the GPS satellite name to use
    :param realtime (0 or 1): If this is 1 then data will be created as a rolling nday file
    :param tstep (int): time cadence of data to output in minutes
    :param ndays (int): the number of days of data to create
    :param outdir (str): the output directory of the plot and data
    :param testing (0 or 1): flag used to test the code
    :return:
    '''

    tform='%Y-%m-%dT%H:%M:%S.%fZ'# format used for tie
    fname = outname+str(ndays)+'day.txt' # output file name

    #--------------------------Set start,end, and time array to process-------------------
    # Get the start date and end date of new data that needs to be processed
    # and the new start date of the rolling file of ndays

    # Start date for the whole file
    fsdate = (dt.datetime.utcnow()-dt.timedelta(days=ndays))

    if realtime==1:
        # If its realtime mode then add to the output file starting from the last data point
        try:
            # Checks the output file to get the last time as a datetime
            lasttime = get_GPS_start(fname)
            # start at the first time step after the last one in the file
            sdate = lasttime+dt.timedelta(minutes=tstep)
            # Todo : add a check to see when data was last processed
            # If the last time is more than ndays ago then start at ndays
            # so that it doesn't accidentally process many days of data because
            # the process was stopped and restarted
            if sdate<fsdate:
                sdate = (dt.datetime.utcnow()-dt.timedelta(days=ndays)).replace(minute=0,second=0,microsecond=0)
        except:
            # If that doesn't work then start at the current day minus ndays
            # and at the nearest x minute time step
            # When the code is initially run with no file then it starts here
            sdate = (dt.datetime.utcnow()-dt.timedelta(days=ndays)).replace(minute=0,second=0,microsecond=0)

        # The end date to process is always the current date
        edate = dt.datetime.utcnow()
    else:
        # In theory, reprocessing is possible. However, historial TLEs are neededd
        # that are not available at celestrak. It will be up to CCMC to see
        # if they want to get a spacetrak account to make this work
        # Otherwise past data would be reprocessed with TLEs that are not
        # that are close to the requested dates
        sdate = sdate_all

    # Create the time array between sdate and edate at timestep tstep
    times=make_times(sdate,edate,tstep)

    #------------------------ CREATE ORBIT-----------------------------
    # Create the orbit for those times using TLEs from celestrak https://celestrak.org/
    # celestrak has been providing orbital info for many decades

    # Get the most recent GPS TLE info from celstrak for all the GPS sats
    group = "GPS-OPS"
    format="TLE"
    celes_url = "https://celestrak.org/NORAD/elements/gp.php"

    # Returns the most recent two lines for the requested sat
    # None is returned if the satellite is not found or something goes wrong
    tle1,tle2 = get_TLES(celes_url,group,format,sat)

    # Create xyz location in GEI coords for each timestep
    # If it can't get tles for some reason then it just quits
    if tle1 is not None:
        prop = Propagator(Propagator.GRAVITY_WGS72) # Create the location object
        loc = []
        for t in times:
            # This gives GEI coords which is acceptable for the fast-magephem code
            loc1,vel=prop.position(t.year, t.month, t.day, t.hour, t.minute, t.second, tle1, tle2)
            loc.append(loc1)
    else:
        sys.exit("Can't get tles: Exiting")

    # The default sat used is ns73 which is PRN 32
    # Steve Morley's paper gives the "ns" satellite name for some
    # of the GPS satellites. Its not clear why LANL uses an "ns" name
    # while NORAD uses PRN
    # 41019	ns73	2015-062A NAVSTAR 75 (USA 265), also known as GPS BIIF-11

    #--------------------------- GET SHELLS OUTPUT------------------------
    # Now connect to SHELLS service to get the output along the sat for 90
    # degree pitch angles and the requested energies (default 500, 2000 keV)
    # Reformat the input data
    shells_io = {}
    shells_io["time"] = [x.strftime(tform) for x in times]
    shells_io["xyz"] = loc
    shells_io["sys"] = "GEI"
    # The code assumes only one pitch angle and will have to be modified
    # to write out more cols if needed
    shells_io["pitch_angles"] = [90]
    shells_io["energies"] = Es

    shells_inputs = json.dumps(shells_io)

    # Get the shells data for the new times
    if testing==1:
        # If its testing mode then use the flask serverless mode
        # In this mode, the magephem service isn't actually called
        # but just returns a mock up of fixed L shells
        app = create_app(test_config="test_config")
        response = app.test_client().post("/shells_io",data=shells_inputs,content_type='application/json')
    else:
        # Call the shells service which requires the magaphem service to be running as well
        response = requests.post(sh_url, json=shells_inputs).json()

    #------------------------ WRITE OUTPUT FILE and PLOT ---------------------
    # To write the output file, first read in any data that is already processed and
    # tadd any data returned in response from the shells service

    #This makes a dict of lists of the old data and returns none if there is no file
    odata = read_old_file(fname)

    # If data is returned for the new data then add them together
    # This is kind of tricky because the output from the service is a list
    # of lists. But the output from the file is just a list
    if (response.status_code>=200) & (response.status_code<=250):
        ndata1 = response.json
        # flatten the data returned
        ndata = {key: list(np.array(ndata1.get(key, [])).flat)  for key in ndata1.keys()}
        if odata is not None:
            shdata = {key: odata.get(key, []) + list(ndata.get(key, [])) for key in odata.keys()}
        else:
            shdata = ndata

    else:
        shdata = odata

    # If theres data then write it to a file
    if shdata is not None:
        newinds = [i for i in range(0,len(shdata['time'])) if dt.datetime.strptime(shdata['time'][i],tform)>fsdate]
        fdata = {key: shdata.get(key, [newinds]) for key in shdata.keys()}
        ekeys = [x for x in list(shdata.keys()) if ('E flux' in x)]
        Bkeys = [x for x in list(shdata.keys()) if ('Bmirrors' in x)]
        skeys = ['time','L']+ekeys+Bkeys
        with open(os.path.join(outdir,fname), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(skeys)
            for ico in range(0,len(fdata['time'][:])):
                if dt.datetime.strptime(fdata['time'][ico],'%Y-%m-%dT%H:%M:%S.%fZ')>fsdate:
                    row1 = [fdata['time'][ico],fdata['L'][ico]] # Time and L
                    row2 = ["{0:.5f}".format(fdata[k][ico]) for k in skeys[2::]]
                    row = row1+row2
                    writer.writerow(row)
        # And make a plot
        # Bin the data on orbit and L

        if testing==1:
            # For testing, the Lshells returned are all constant so use z to get something
            # like passes
            # Have to add an offset to z because getLpass gets rid of zeros
            #z = np.array(loc)[:,2]
            #zmin=np.abs(np.nanmin(z))
            #z = (z+zmin)/2000
            npoints = int(60 / tstep) # SEt passes to one hour
            passes = [int(x/npoints) for x in range(0,len(shdata['time']))]
            #passes, breaks = getLpass(z,dist=npoints)
        else:
            # getLpass looks for peaks and valleys
            # dist is the number of points away that something has to be to
            # be considered another peak. That will depend on tstep
            # 20 minutes for GPS is npoints=int(20/tstep)
            npoints = int(20 / tstep)
            passes, breaks = getLpass(shdata['L'][:],dist=npoints)
        xbin=np.arange(3,6.5,.25)
        ybin = np.arange(min(passes), max(passes) + 1)
        # shdata returns a list of lists
        # reconfigure the data into numpy array
        data = np.zeros((len(ekeys)+1,len(shdata['time'][:])),dtype=float)
        # Dealing with timezones in python is hideous.
        # We need time in cime in order to get the average time of each pass after binning
        # In theory this should create the datetime for the string, make it utc and then get a timestamp
        # So when we go back to datetime with utcfromtimestamp it should still be the same
        data[0,:] = [dt.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=dt.timezone.utc).timestamp() for x in shdata['time'][:]]
        for co,key in enumerate(ekeys):
            # Todo check what squeeze does to 2 D array
            data[co+1,:] = np.array(shdata[key][:]).squeeze()
        #data = np.array(shdata[ekeys[0]]).squeeze()
        bin_data = stats.binned_statistic_2d(np.array(shdata['L']).squeeze(), passes, data, statistic=np.nanmean, bins=[xbin, ybin])
        # make a plot with up to 4 energies

        tbins= [dt.datetime.utcfromtimestamp(x) for x in np.nanmean(bin_data.statistic[0,:,:],axis=0)]
        plt.set_cmap('jet')
        pco = 1
        fco = 1
        for co,key in enumerate(['time']+ekeys):
            plt.figure(fco)
            if ('q' not in key) & ('flux' in key):
                ax = plt.subplot(2,1,pco)
                plt.pcolormesh(tbins,xbin[0:-1],bin_data.statistic[co,:,:])
                cbar=plt.colorbar()
                cbar.set_label('log10(#/cm2-s-str-keV)', labelpad=10,rotation=270,fontsize=8)
                plt.ylabel('L')
                date_format = mdates.DateFormatter('%y-%m-%d')
                ax.xaxis.set_major_formatter(date_format)
                plt.xticks(rotation=50)
                n=2
                [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
                plt.title(key+ ' keV')
                plt.tight_layout()

                pco=pco+1
                if pco>2:
                    # If theres more than 2 energies to plot then make a new fig
                    fco = fco+1
                    pco=1
                    plt.savefig(os.path.join(outdir, outname + 'fig' + str(fco) + '.png'))
            plt.savefig(os.path.join(outdir, outname + 'fig' + str(fco) + '.png'))
    print("Done")

if __name__ == "__main__":
    '''
    PURPOSE: To generate a dataset of the shells electron flux output along
    a GPS-like orbit that updates in real time. Also make a plot. For this
    case, we will make a running weekly data file that updates whenever its called.
    The challenge is that we only have the most recent TLEs from celestrack.
    
    The script can be run in two ways, with a fixed start and end date 
    (using the -s and -e inputs) or in real time with the -rt flag.
    If the -s and -e inputs are used than a dataset of daily csv files
    will be created between the start and end dates. if -rt is used then
    a daily file will be created for the current day  
    
    # In order to create the GPS output
    1) get the most recent TLE
    2) Create the trajectory
    3) Call the app to get the E flux
    '''

    parser = argparse.ArgumentParser('This program creates SHELLS electron fluxes'
                                     'along a GPS-like orbit')
    parser.add_argument('-s', "--startdate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=False,
                        default=None,
                        type=valid_date)
    parser.add_argument('-e', "--enddate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=False,
                        default=None,
                        type=valid_date)
    parser.add_argument('-sat', "--satellite",
                        help="The GPS satellite to get",
                        default="PRN 32")
    parser.add_argument('-u', "--url",
                        help="The url for the shells service",
                        required=False, default="http://172.17.0.3:5005/shells_io/")
    parser.add_argument('-rt', "--realtime", action='store_true', default=True)
    parser.add_argument('-c', "--cadence",
                        help="The time cadence in minutes for the output data",
                        default=5)
    parser.add_argument('-d', "--days",
                        help="The number of days in the plot and file for the output data",
                        default=7)
    parser.add_argument('-es', "--energies",
                        help="The electron energies to create",
                        nargs='+',
                        default=[500,2000])
    parser.add_argument('-od', "--outdir",
                        help="The local directory to put the output files",
                        required=False, default=os.path.join(os.getcwd()))
    parser.add_argument('-on', "--outname",
                        help="The base name of the the output files",
                        required=False, default=os.path.join(os.getcwd()))
    parser.add_argument('-t', "--testing", action='store_true', default=True)

    args = parser.parse_args()


    #----------------------------------------------------------------

    x = make_GPS_shells(sdate_all=args.startdate, edate=args.enddate,
                        sat = args.satellite,sh_url=args.url,realtime=args.realtime,
                        tstep=args.cadence,ndays=args.days, Es =args.energies,
                        outdir=args.outdir,outname=args.outname,
                        testing=args.testing)