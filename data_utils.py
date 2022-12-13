__author__ = 'Janet Green'
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import leapseconds
import requests
#import odc_util
import math
from scipy.signal import find_peaks
import urllib
import h5py
import glob
#sys.path.insert(1, '/Users/janet/PycharmProjects/common/')
import poes_utils as pu

def check_slash(cdir):
    # This just checks that there is a slash at the end of the directory inputs so nothing
    # goes wonky
    # Find if it is '\' or '/
    if cdir.find('/')>-1:
        slash = '/'
    else:
        slash = '\\'
    if cdir[-1]!=slash:
        cdir = cdir+slash

    return cdir

def find_files(sat,sdate,edate,root_dir, name_fmt,lower_dir = None):
    import glob
    # PURPOSE: To find all the data files between sdate and edate in a variable but fairly
    # common directory structure given by the inputs. Allowing some specification of the
    # directory structure with common directories like year make it faster than just searching
    # through all files in the top root_dir where there are often many, many files.
    # INPUTS:
    #  sat   :      string name of sat. Assumes that the sat name in the directory structure
    #               is the same as in the file format. This assumption may not always hold which
    #               means a different lower dir with the SAT name would have to be passed
    #               example for HEO sat = 'F1'
    #  sdate :      datetime object with the start day of requested data
    #  edate :      datetime object with the end day of requested data
    #  root_dir:    string with the top leve dirctory (no keywords
    #  lower_dir:   string with the lower level directory structure using fillable keywords
    #               example for HEO 15s data lower_dir = 'SAT/exp/YYYY'
    # OUTPUTS:      List of files to read

    #NOTE: The HEO data is weird in that there are ~2 files per data at random times

    n_days = abs((edate - sdate).days) # Number of days from sdate to look through
    file_pattern_list = list()
    lower_dir_list = list()

    for nco in range(0,n_days):
        # First replace the keywords in the name_fmt
        tempfile = name_fmt
        newtime = sdate+dt.timedelta(days = nco)
        strvals = {'YYYY':str(newtime.year),'yy':str(newtime.year)[-2::],'MM':str(newtime.month).zfill(2),'SAT':sat,
                   'DD':str(newtime.day).zfill(2),'ddd':str(newtime.timetuple().tm_yday).zfill(3)}
        for val in strvals.keys():
            tempfile =tempfile.replace(val, strvals[val])

        # Add the lower directory structure if requested
        if lower_dir:
            # If there is a second directory structure passed then fill in the key words and create
            # a list of those for each file. This is to make it simpler when searching for different types of
            # data under the same root dir but different lower structures which often is the case.
            tempdir = lower_dir
            for val in strvals.keys():
                tempdir = tempdir.replace(val, strvals[val])
            lower_dir_list.append(tempdir)

        file_pattern_list.append(tempfile)

    # Now search for all the matching files in the given directory
    final_files = list()
    for fco in range(0,len(file_pattern_list)):
        # Create a list of all the files in the expected directory
        if lower_dir:
            all_files = glob.glob(root_dir+lower_dir_list[fco]+file_pattern_list[fco], recursive=True)
        else:
            all_files = glob.glob(file_pattern_list[fco],recursive=True)
        final_files.extend(all_files)
    return final_files
    # First


def read_ascii_data(files, header_lines=0, vars=None,delim=None,datatype = None):
    # PURPOSE: To read HEO data
    # INPUTS:
    #   files: A list of files to read
    #   vars: the vars you want to read in. If None then all the data is passed back
    #
    # OUTPUTS: A numpy array with the data
    #----------------------------------------------------------------------------------

    # Find the columns to get from datatype
    dtype_list = list()
    col_list = list()
    if vars:
        # For var list need column index in file, but dtype list must match vars
        for var in vars:
            i = datatype.names.index(var)
            col_list.append(i)
            dtype_list.append((var, datatype[i]))
    else:
        # If no var list, use all columns
        col_list = range(0, len(datatype.names))
        dtype_list = datatype

    # Now we want to find the cols of just vars
    # data = np.zeros(( 0, len( datatype.names )), dtype = datatype )
    data = np.zeros((0, len(col_list)), dtype=dtype_list)

    for file in files:
        data = np.append(data,np.loadtxt(file,dtype=dtype_list,delimiter=delim,skiprows=header_lines,usecols=col_list))

    return(data)


def get_HEO_text(sat, sdate, edate ,root_dir,lower_dir = None, vars = None):
    # PURPOSE: To create a numpy array of the HEO data in the time period from sdate to
    # edate for the satellite, sat. First it looks in the directories root_dir and
    # lower dir (if given). Then it reads the files and appends them into one
    # numpy array with dtype given by the column names in the first line.
    # There are two kinds of HEO data 'exp' (15 sec averages) and Lbin (.25 Lbin averages)
    # This will read either one but it assumes they are in different directories

    # INPUTS:
    # sat       : name of satellite ex. 'f1', 'f3'                      (REQUIRED)
    # sdate     : datetime start of data to get                         (REQUIRED)
    # edate     : datetime end of data to get                           (REQUIRED)
    # root_dir  : top directory to search for data ex. '/data/          (REQUIRED)
    # lower_dir : directory under top directory with KEYWORDs like YYYY (Optional)
    #               ex. 'SAT/YYYY'
    # vars      : list of vars to get ex. ['Year','Month']              (Optional

    # Quick check to see if there is a slash at the end of the directories
    root_dir= check_slash(root_dir)
    if lower_dir:
        lower_dir=check_slash(lower_dir)

    # The file name format for HEO text data is fixed regrdless of whether it is exp or Lbin
    # If you want to get exp dat then use lower_dir = 'SAT/exp/YYYY/'
    # If you want to get Lbin data then user lower_dir = 'SAT/Lbin/YYYY/'

    heo_fmt = 'SAT_YYYYddd_*'

    # first find the files
    files = find_files(sat, sdate, edate, root_dir, heo_fmt, lower_dir=lower_dir)

    # Check if data is returned
    if len(files)<1:
        print('No data for ',sat, 'between ',sdate.strftime("%Y/%m/%d"),' and ',edate.strftime("%Y/%m/%d"))
        return

    # read_ascii is a generic code to read in a typical ascii dataset
    # These are the inputs needed for HEO
    # For the HEO data the header is just one line
    heo_header = 1

    # Read the first line of the first file to get the column names
    with open(files[0],'r') as f:
        head_dat = (f.readline()).split()
        f.close()

    # Then create the heo_type
    heo_type=list()
    for col in head_dat:
        heo_type.append((col,'f'))

    #This will read and concatenate them all together

    # Need to order the files in time
    files.sort()
    alldata = read_ascii_data(files,header_lines=heo_header,vars = vars,datatype =np.dtype(heo_type))

    return alldata

def get_ICO_text( sdate, edate ,root_dir, lower_dir = 'ICO_YYYY', vars = None ):
    # PURPOSE: To create a numpy array of the ICO data in the time period from startYr to
    # endYr. The directory structure must follow the format {root path}/ICO_{year}, for
    # the passed range of years.
    # It reads the files and appends them into one
    # numpy array with dtype given by the column names in the first line.

    # INPUTS:
    # startDate   : start date of data to get                         (REQUIRED)
    # endDate     : end date of data to get                           (REQUIRED)
    # root_dir  : top directory to search for data ex. '/data/        (REQUIRED)
    # vars      : list of vars to get ex. ['Year','Month']            (Optional)

    # Quick check to see if there is a slash at the end of the directories
    root_dir= check_slash(root_dir)
    if lower_dir:
        lower_dir=check_slash(lower_dir)

    # The file name format for ICO text data is fixed "ico_yyyydoy_v04.l2"
    # The directory structure is assumed to be {root path}/ICO_YYYY

    ico_fmt = 'ico_YYYYddd_*'

    # first find the files
    #files = find_files( startDate, endDate, root_dir )
    # The ICO data is in a  7 day ascii format which makes it
    # really hard to find the data you want without opening extra files. Ugh.

    # So just to be safe, I add 7 days to sdate to make sure we get the start
    # And then I reduce it down to the requested time at the end.
    # first find the files

    new_sdate = sdate-dt.timedelta(days = 7)
    files = find_files( ' ', new_sdate, edate, root_dir, ico_fmt, lower_dir=lower_dir)

    # Check if data is returned
    if len(files)<1:
        print('No data between ' + str(sdate) + ' and ' + str(edate))
        return

    # read_ascii is a generic code to read in a typical ascii dataset
    # These are the inputs needed for HEO
    # For the HEO data the header is just one line
    ico_header = 1

    ico_type=list()

    # Read the first line of the first file to get the column names
    with open(files[0],'r') as f:
        head_dat = (f.readline()).split()
        f.close()

    # Then create the heo_type (column 1 is year - integer)

    for col in head_dat:
        if len(ico_type) == 0 :
            ico_type.append(( col, 'i'))
        else :
            ico_type.append((col,'f'))

    # Need to order the files in time
    files.sort()

    # This will read and concatenate them all together

    alldata = read_ascii_data( files, header_lines=ico_header, vars = vars, datatype = np.dtype( ico_type ))

    time1 = [dt.datetime(alldata['Year'][x],1,1)+ dt.timedelta(days=alldata['DecDay'][x]-1) for x in range(0,len(alldata['Year']))]

    ginds = np.where((np.array(time1)>sdate) & (np.array(time1)<edate))[0]
    #time = [time1[x] for x in ginds]
    # Normally if there is no data then None is returned
    # But if ginds = nothing then [] is returned
    # This makes it so None is returned
    if len(ginds)<1:
        finaldat = None
    else:
        finaldat = alldata[ginds]

    return finaldat

    #return alldata

def getLpassHEO(Ldata):
    # PURPOSE: to define the passes based on the Orbit status
    # Input: either the 'Orbit_Status_IGRF' or 'Orbit_Status_OP'
    goodinds = np.where((Ldata>0) & (Ldata<100))[0]
    dL = np.diff(Ldata[goodinds])
    allbreaks = goodinds[np.where(np.diff(np.sign(dL)))]
    #obreaks = 0 * Ldata
    obreaks = np.zeros(len(Ldata),dtype=float)
    obreaks[allbreaks] = 1
    passes = np.cumsum(obreaks)

    return passes, allbreaks
def getLpassHEO2(L,dist=200,prom=.5):
    #Purpose: To create an arary with pass number for each data point
    # that can be used to more easily average data or plot it pass by pass
    # Limit to between 0 and 100 because weird things happen at large L
    # Need to fix this so that it works on numpy arrays and numpy masked arrays
    # Usage: if the data is netcdf4 returned from poes_utils
    # getLpass(poes['L_IGRF][:],dist=200,prom=.5):
    # if the data is a numpy array
    # getLpass(data['L_IGRF'][:],dist=200,prom=.5

    if isinstance(L, np.ma.MaskedArray):
        Ldata = L.data
    else:
        Ldata = L
    goodinds= np.where((Ldata[:]>0) & (Ldata[:]<100))[0]
    peaks = find_peaks(Ldata[goodinds],distance=dist,prominence=prom) # Find the maxima
    valleys = find_peaks(-1*(Ldata[goodinds]),distance=dist,prominence=prom) # Find the minima
    #plt.plot(L.data[goodinds])
    #plt.plot(peaks[0],L.data[goodinds[peaks[0]]],'*')
    #plt.plot(valleys[0], L.data[goodinds[valleys[0]]], '+')
    allbreaks = np.sort(np.append(goodinds[peaks[0]], goodinds[valleys[0]]))
    pbreaks = np.zeros(len(Ldata), dtype=float)
    #pbreaks = 0 * Ldata
    pbreaks[allbreaks] = 1
    passes = np.cumsum(pbreaks)

    return passes,allbreaks
def get_dsts( sdate, edate ) :
    # PURPOSE: retrieves hourly DST index values between the passed start and end dates
    # from the omniweb (nasa) data retrieval web site. Strips off header and footer info
    # returning only the date-time and dst value lists

    dstVarNum = "40"
    sdateStr = sdate.strftime( "%Y%m%d" )
    edateStr = edate.strftime( "%Y%m%d" )
    dstFileStr = "./DST_" + sdateStr + "_" + edateStr + ".txt"

    queryStr = " --post-data \"activity=retrieve&res=hour&spacecraft=omni2&start_date=" + \
       sdateStr + "&end_date=" + edateStr + \
       "&vars=" + dstVarNum + \
       "&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0" + \
       "&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=\" " + \
       "https://omniweb.sci.gsfc.nasa.gov/cgi/nx1.cgi -O " + dstFileStr

    os.system( "wget" + queryStr )

    dsts = []
    isHeader = True
    with open( dstFileStr ) as fp:
        for cnt, line in enumerate(fp):
            if line.startswith( "YEAR" ) :
                isHeader = False
            elif isHeader :
                continue
            else :
                if line.startswith("<") :
                    break
                dst_parts = [ int(i) for i in line.split() ]
                dsts.append( dst_parts )

    return( dsts )

def get_omni( sdate, edate, vars, hires = None ) :
    '''
    PURPOSE: retrieves omnidata values between the passed start and end dates
    from the omniweb (nasa) data retrieval web site. Strips off header and footer info
    returning only the date-time and dst value lists
    vars is a list of variables requested

    If hourly data is needed then hires =None
    If minute data is needed then hires will be hires='1min' or hires='5min;

    :param sdate (datetime):
    :param edate (datetime):
    :param vars (list(str)): list of variables to retrieve
    :param hires:
    :return:
    '''
    # The available variables for hi and lo res are different
    if hires is None:
        allvars =['Bartels Roation Number','IMF Spacecraft ID','Plasma Spacecraft ID','Fine Scale Points in IMF avgs',
              'Fine Scale Points in Plasma Avgs','IMF Magnitude Avg','Magnitude, Avg IMF Vr',
              'Lat. of Avg. IMF','Long. of Avg. IMF','Bx, GSE/GSM','By, GSE','Bz, GSE',
              'By, GSM','Bz, GSM','Sigma in IMF Magnitude Avg','Sigma in IMF Vector Avg',
              'Sigma Bx, nT','Sigma By, nT','Sigma Bz, nT','Proton Temperature, K','Proton Density, n/cc',
              'Flow Speed, km/sec','Flow Longitude, deg','Flow Latitude, deg','Alpha/Proton Density Ratio','Flow Pressure, nPa',
              'Sigma-T','Sigma-Np','Sigma-V','Sigma-Flow-Longitude','Sigma-Flow-Latitude','Sigma-Alpha/Proton Ratio',
              'Ey - Electric Field, mV/m','Plasma Beta','Alfven Mach Number','Kp*10 Index',
              'R Sunspot Number','Dst Index, nT','AE Index, nT',
              'Proton Flux* > 1 MeV','Proton Flux* > 2 MeV','Proton Flux* > 4 MeV','Proton Flux* >10 MeV',
              'Proton Flux* >30 MeV','Proton Flux* >60 MeV','Magnetospheric Flux Flag','ap index, nT','Solar index F10.7',
              'Polar Cap (PCN) index','AL Index, nT','AU Index, nT','Magnetosonic Mach Number','Lyman Alpha','Proton Quazy-Invariant']
        varnums = [allvars.index(s)+3 for s in allvars if any(xs in s for xs in vars)]
    else:
        allvars =['IMF Spacecraft ID','Plasma Spacecraft ID','Fine Scale Points in IMF avgs',
              'Fine Scale Points in Plasma Avgs','Percent interpolated','Timeshift','Sigma Timeshift',
                  'Sigma Min_var_vector','Time btwn observations',
                'IMF Magnitude Avg','Bx, GSE/GSM','By, GSE','Bz, GSE',
              'By, GSM','Bz, GSM','Sigma in IMF Magnitude Avg','Sigma in IMF Vector Avg',
              'Flow Speed, km/sec','Vx Velocity,km/s','Vy Velocity, km/s','Vz Velocity, km/s',
                  'Proton Density, n/cc','Proton Temperature, K',
                'Flow Pressure, nPa',
              'Ey - Electric Field, mV/m','Plasma Beta','Alfven Mach Number',
              'S/C, Xgse','S/C, Ygse','S/c, Zgse',
              'BSN location, Xgse,Re','BSN location, Ygse','BSN location, Zgse','AE Index, nT',
               'AL-index, nT','AU-index, nT','SYM/D, nT','SYM/H, nT','ASY/D, nT','ASY/H, nT',
              'Proton Flux* >10 MeV',
              'Proton Flux* >30 MeV','Proton Flux* >60 MeV']

        varnums = [allvars.index(s)+4 for s in allvars if any(xs in s for xs in vars)]
    #varstring =''
    #for varnum in varnums:
    #    varstring = varstring +'&vars='+str(int(varnum))

    sdateStr = sdate.strftime( "%Y%m%d" )
    edateStr = edate.strftime( "%Y%m%d" )
    #dstFileStr = os.path.join(os.getcwd(),"OMNI_" + sdateStr + "_" + edateStr + ".txt")

    # JGREEN: 03/19/2021 Changed this to use requests and get the data directly
    postdata = {}
    varlist =list()
    postdata['activity'] = 'retrieve'
    postdata['start_date'] = sdateStr
    postdata['end_date'] = edateStr
    for varnum in varnums:
        varlist.append(str(varnum))  
    postdata['vars'] = varlist

    if hires is None:
        # Get the lo res data
        #queryStr = " --post-data \"activity=retrieve&res=hour&spacecraft=omni2&start_date=" + \
        #sdateStr + "&end_date=" + edateStr + \
        #varstring + \
        #"&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0" + \
        #"&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=\" " + \
        #"https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi -O " + dstFileStr
        
        #postdata['activity']='retrieve'
        postdata['res']='hour'
        postdata['spacecraft'] = 'omni2'
        #postdata['start_date'] = sdateStr
        #postdata['end_date'] = edateStr
        #varlist=list()
        #for varnum in varnums:
        #    varlist.append(str(varnum))
        #postdata['vars']=varlist
    else:
        if hires =='5min':
            postdata['res'] = hires
            postdata['spacecraft'] = 'omni_'+hires+'_def'
            #queryStr = " --post-data \"activity=retrieve&res=5min&spacecraft=omni_"+hires+"_def&start_date=" + \
            #       sdateStr + "&end_date=" + edateStr + \
            #       varstring + \
            #       "&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0" + \
            #       "&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=\" " + \
            #       "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi -O " + dstFileStr
        else:
            #postdata['res'] = hires
            postdata['spacecraft'] = 'omni_'+hires+'_def'
            #queryStr = " --post-data \"activity=retrieve&spacecraft=omni_"+hires+"_def&start_date=" + \
            #       sdateStr + "&end_date=" + edateStr + \
            #       varstring + \
            #       "&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0" + \
            #       "&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=\" " + \
            #       "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi -O " + dstFileStr


    #os.system( "wget" + queryStr )
    url = 'https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi'
    x = requests.post(url, data=postdata)
    dsts = []
    isHeader = True
    for line in x.text.splitlines():
        if line.startswith( "YEAR" ) | line.startswith( "YYYY" ):
            isHeader = False
        elif isHeader :
            continue
        else :
            if line.startswith("<") :
                break
            dst_parts = [ float(i) for i in line.split() ]
            dsts.append( dst_parts )
    #with open( dstFileStr ) as fp:
    #    for cnt, line in enumerate(fp):
    #        if line.startswith( "YEAR" ) | line.startswith( "YYYY" ):
    #            isHeader = False
    #        elif isHeader :
    #            continue
    #        else :
    #            if line.startswith("<") :
    #                break
    #            dst_parts = [ float(i) for i in line.split() ]
    #            dsts.append( dst_parts )
    #os.remove(dstFileStr)
    return( dsts )

def get_omni_max( sdate, edate, vars, hires = None ,days=3) :
    '''
    PURPOSE: To get the requested variables from omni data along with the
    maximum in the last number of days specified by days
    :param sdate (datenum): start date
    :param edate (datenum): end date
    :param vars (list): list of variables to get ['Kp*10']
    :param hires: If none then it will get the hourly low res data. If hires='5 min' then get that data
    :param days (int): number of days prior to find the max value
    :return fdata (dict): Dictionary with time, the returned variables and the max in days
    '''
    omdata = get_omni( sdate-dt.timedelta(days=days), edate, vars, hires = None )

    # Todo add a check here if there is no data returned
    fdata={}
    fdata['time'] = np.array([dt.datetime(int(omdata[co][0]), 1, 1, int(omdata[co][2])) + dt.timedelta(
        days=int(omdata[co][1]) - 1) for co in np.arange(0, len(omdata))])

    # Get each of the vars
    for omco in range(0, len(vars)):
        # First three cols are time ones return a list of the values
        fdata[vars[omco]] = np.array([omdata[co][3 + omco] for co in np.arange(0, len(omdata))])

    for omco in range(0, len(vars)):
        fdata[vars[omco]+'_max'+'_'+str(days)+'d'] = np.zeros((0),dtype=float)

    for co in range(0,len(omdata)):
        for omco in range(0, len(vars)):
            mvar = vars[omco] + '_max' + '_' + str(days) + 'd'
            time = fdata['time']
            linds = np.where((time>(time[co]-dt.timedelta(days=days))) &(time<=(time[co])) )[0]
            maxval = np.max(fdata[vars[omco]][linds])
            fdata[mvar] = np.append(fdata[mvar][:],maxval)
    
    return(fdata)

def get_meanDST( dsts ):
    # PURPOSE: return the mean dst value in a list of lists of year doy hr dst

    meanDST = 0.0
    for dstData in dsts :
        meanDST += dstData[3] / len(dsts)

    return meanDST

def get_minDST( dsts ):
    # PURPOSE: return the min dst value in a list of lists of year doy hr dst

    minDST = 99999.9
    for dstData in dsts :
        if dstData[3] < minDST :
            minDST = dstData[3]

    return minDST

def get_precedingDST( dsts, atDateTime ):
    # PURPOSE: return dst value preceding the passed timestamp in a list of lists of year doy hr dst

    yr = atDateTime.year
    doy = atDateTime.timetuple().tm_yday
    hr = atDateTime.hour

    precedingDST = 0.0
    for dstData in dsts :
        if dstData[0] < yr :
            precedingDST = dstData[3]
        elif dstData[0] == yr :
            if dstData[1] < doy :
                precedingDST = dstData[3]
            elif dstData[1] == doy :
                if dstData[2] < hr :
                    precedingDST = dstData[3]
                else :
                    break
            else :
                break
        else :
            break

    return precedingDST


def get_Kp_rt( sdate, edate, days = None) :
    '''
    PURPOSE: To get the real time Kp from GFZ
    :param sdate (datetime):
    :param edate (datetime):
    :param days (int): Number of days for finding the max value
    :return:
    '''
    # This is for data that is 1 month from current day
    gfzsite = 'ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/'
    file = 'Kp_ap_nowcast.txt'
    # The potsdam files have YY MM DD hh.h hh.m (mid time) days days Kp Ap D
    # This only goes back 30 days
    Kp = []
    r = urllib.request.urlopen(gfzsite+file)
    ftext = r.read().decode("utf-8") # Read the returned data
    for line in ftext.splitlines():
        #print(line)
        if line.startswith( '#'):
            continue
        else:
            if line.startswith("<") :
                break
            Kp_parts = [ float(i) for i in line.split() ]
            Kp.append( Kp_parts )
    # Create a datetime array
    Kpdate1 = np.array([dt.datetime(np.int(Kp[co][0]), int(Kp[co][1]),int(Kp[co][2]),int(Kp[co][3]))  for co
              in np.arange(0, len(Kp))])
    # Get the indices within start and stop
    #ginds = [co for co in np.arange(0, len(Kp)) if ((Kpdate1[co]>=sdate) & (Kpdate1[co]<=edate))]
    ginds = np.where((Kpdate1>=sdate) & (Kpdate1<edate))[0]
    
    Kpdata = np.array([int(Kp[ginds[co]][7]*10) for co in np.arange(0, len(ginds))])
    Apdata = np.array([int(Kp[ginds[co]][8]) for co in np.arange(0, len(ginds))])
    Kpdate = [Kpdate1[ginds[co]] for co in np.arange(0, len(ginds))]
    Kpmax = np.zeros((0),dtype= float)
    for co in range(0,len(ginds)):
        # Get the indices
        minds = np.where((Kpdate1>=Kpdate1[co]-dt.timedelta(days=days)) & (Kpdate1<=Kpdate1[co]))[0]
        maxval = np.max(Kpdata[minds])
        Kpmax = np.append(Kpmax,maxval)
    #Kptime1 = pu.unixtime(Kpdate)
    #Kptime = [1000 * co for co in Kptime1]
    
    return Kpdate, Kpdata, Kpmax, Apdata
        

def get_GPS_text(sat, sdate, edate ,root_dir,lower_dir = None, vars = None, version=''):
    # PURPOSE: To create a numpy array of the GPS data in the time period from sdate to
    # edate for the satellite, sat. First it looks in the directories root_dir and
    # lower dir (if given). Then it reads the files and appends them into one
    # numpy array with dtype given by the column names in the first line.

    # INPUTS:
    # sat       : name of satellite ex.                       (REQUIRED)
    # sdate     : datetime start of data to get                         (REQUIRED)
    # edate     : datetime end of data to get                           (REQUIRED)
    # root_dir  : top directory to search for data ex. '/data/          (REQUIRED)
    # lower_dir : directory under top directory with KEYWORDs like YYYY (Optional)
    #               ex. 'SAT/YYYY'
    # vars      : list of vars to get ex. ['Year','Month']              (Optional

    # Quick check to see if there is a slash at the end of the directories
    root_dir= check_slash(root_dir)
    if lower_dir:
        lower_dir=check_slash(lower_dir)

    gps_fmt = 'SAT_yyMMDD_*'+version+'*'

    # The GPS data is in a god awful 7 day ascii format which makes it
    # really hard to find the data you want without opening extra files. Ugh.

    # So just to be safe, I add 7 days to sdate to make sure we get the start
    # And then I reduce it down to the requested time at the end.
    # first find the files
    new_sdate = sdate-dt.timedelta(days = 6)
    files = find_files(sat, new_sdate, edate, root_dir, gps_fmt, lower_dir=lower_dir)
    files.sort()

    # Check if data is returned
    if len(files)<1:
        print('No data for ',sat, 'between ',new_sdate.strftime("%Y/%m/%d"),' and ',edate.strftime("%Y/%m/%d"))
        return

    # read_ascii is a generic code to read in a typical ascii dataset
    # These are the inputs needed for GPS
    # For the GPS data the header is commented out so I think it is 0
    gps_header = 0

    # The first many lines of the GPS data have the colum info in json format
    # but they are commented out. So I have to read all the lines into a string variable
    # and then try to read json to create the columns
    head_dat =''
    with open(files[0],'r') as f:
        temp_dat = (f.readline())
        while temp_dat[0]=='#':
            head_dat = head_dat + (temp_dat[1:-1])
            temp_dat = (f.readline())
        f.close()

    # This is all the header info but each key in the dict has a
    # separate "ELEMENT_NAMES" with the actual columns
    col_info = json.loads(head_dat)
    cols = list()
    for key in col_info:
        if type(col_info[key])==dict:
            cols.extend(col_info[key]['ELEMENT_NAMES'])

    # Then create the heo_type
    gps_type=list()
    for col in cols:
        gps_type.append((col,'f'))

    #This will read and concatenate them all together
    alldata = read_ascii_data(files,header_lines=gps_header,vars = vars,datatype =np.dtype(gps_type))

    gpstimes = [dt.datetime(alldata['year'][x],1,1)+ dt.timedelta(days=alldata['decimal_day'][x]-1) for x in range(0,len(alldata['year']))]
    time1 = [leapseconds.gps_to_utc(x) for x in gpstimes]

    ginds = np.where((np.array(time1)>sdate) & (np.array(time1)<edate))[0]
    time = [time1[x] for x in ginds]

    return time,alldata[ginds]

# -----------------------------------------------------------------------------
#       SAMPEX data utils
#------------------------------------------------------------------------------

def get_SAMP_file(direc,sdate,edate,fstr=None):
    # The SAMPEX MAST data is stored by Bartels rotation
    # First list all the files for the year
    if fstr is not None:
        # This allows selection of orbit files with PSS
        fname = '**/*'+fstr+'*'+sdate.strftime('%Y')+'*.txt'
    else:
        fname = '**/*RSSet*'+sdate.strftime('%Y')+'*.txt'
    filelist = glob.glob(direc+ fname, recursive=True)
    allfiles=[]
    for file in filelist:
        fdoy_start = dt.datetime(sdate.year,1,1)+dt.timedelta(days=int(file[-15:-12])-1)
        fdoy_end = dt.datetime(sdate.year,1,1)+dt.timedelta(int(file[-7:-4])-1)
        if (((sdate>=fdoy_start) & (sdate<fdoy_end)) | ((edate>=fdoy_start) & (edate<fdoy_end))):
            allfiles.append(file)

    return allfiles

def get_SAMPEX_data(sdate,edate,direc,ins=None,cols=None, ocols = None):
    '''PURPOSE: To retrieve SAMPEX data from the 6 sec files along
    with orbit and Lshell

    :param sdate: (datetime) start to get data
    :param edate: (datetime) end to get data
    :param direc (str) the top directory to look for data
    :param ins (str) The instrumnet to get data for i.e. 'MAST'. If an ins
                     is passed then only cols that start with that will be returned
    :param cols (list(str)) If cols are passed then only those cols will be returned
    :param ocols list(str) If ocols are passed then those cols will be returned from the orbit file
    :return:
    '''
    # Make sure that if cols or ocols are passed they are a list
    if cols is not None:
        if not isinstance(cols, list):
            cols = [cols]
    if ocols is not None:
        if not isinstance(ocols, list):
            ocols = [ocols]
            
    # The SAMPEX data is stored in HDF5 files with 27 days for each solar rotation.
    # The HDF files are unreadable so we are using the text files
    # Oh good lord, really?
    # First find the datafile
    allfiles = get_SAMP_file(direc, sdate, edate)
    allfiles.sort()

    # Need to check that files are returned
    if len(allfiles)>0:
        allcols = ['year', 'doy','sec','subcom','HILT HE1 4.3-9 MeV/n','HILT HE2 9-38 MeV/n',
                 'HILT HZ1 Z>6 8.2-42 MeV/n','HILT HZ2 Z>6 42-220 MeV/n','HILT Mux1','HILT Mux2',
                 'HILT Idle_1','HILT Idl_2','LICA L4','LICA L3','LICA L2','LICA L1','LICA Triple',
                 'LICA Double','LICA Stop','LICA Start','LICA Cal','Lica proton','LICA Lo_pro','LICA hi pro',
                 'MAST Z1_sec','MAST ADC_OR','MAST Live','MAST CNO >150 Mev/n','MAST Z1','MAST He 8-15 MeV/n',
                 'MAST Z>6, 17-19.5 MeV/n','MAST Z>6 19.5-23 MeV/n','MAST Z>6 23-31.1 MeV/n',
                 'MAST Z>6 31.1-51.9 MeV/n','MAST Z>6 51.9-76.5 MeV/n','MAST Z>6 76.5-113 MeV/n',
                 'MAST Z>6 113-156 MeV/n','MAST Mux1','MAST Mux2','MAST Mux3','MAST Z1Rx', 'MAST Z2Rx',
                 'PET P 28-60 MeV/n','PET E 4-15 MeV','PET P 19-28 MeV/n',
                 'PET E 2-6 MeV','PET EWG','PET live','PET PEN','PET RNG','PET Mux1','PET Mux2']
        #
        fcols = allcols
        if ins is not None:
            # Just get the cols for that instrument
            fcols = [i for i in allcols if ins in i]
        if cols is not None:
            # Else get all the cols
            fcols = cols

        # This gets the indices of the cols to get.
        # Todo: If its out of order witll loadtxt work?
        # Do I need to sort first and then redo fcols
        colinds = [i for i, val in enumerate(allcols) if val in fcols]

        # Need to make sure year,doy, sec are there
        # This is just in case it was not in the requested cols because
        # people may assume time is included
        for co in np.arange(2,-1,-1):
            if co not in colinds:
                colinds = [co]+colinds
                fcols.insert(0,allcols[co])
        fco = 0

        # Create the dtype for a structured array
        dtype1 = []
        for colco in colinds:
            typ = (allcols[colco],'f8')
            dtype1.append(typ)
        dtype1 = np.dtype(dtype1)
        for file in allfiles:
            # Need to deal with multiple files
            # Todo Add something to open the file and find BEGIN DATA so you know
            # how many rows to skip. It may change.
            if fco ==0:
                alldata = np.loadtxt(file, skiprows=63, usecols=tuple(colinds) )
            else:
                data = np.loadtxt(file, skiprows=63, usecols=tuple(colinds) )
                alldata = np.append(alldata,data,axis = 0)
            fco =fco+1

        # create a time col
        time = [ dt.datetime(int(alldata[x,fcols.index('year')]),1,1) +dt.timedelta(days = int(alldata[x,fcols.index('doy')]-1),seconds =int(alldata[x,fcols.index('sec')])) for x in range(0,len(alldata))]

        #-------- Orbit ---------------------
        # Now check of orbit data is requested
        # Todo: make one function to get Orbit or data files
        if ocols is not None:

            # Get the right files
            ofiles = get_SAMP_file(direc, sdate, edate, fstr='PSSet')
            # And sort the files
            ofiles.sort()

            # These are all the cols available
            allocols = ['Year','doy','sec','sec psst','flag','orbit','GEO_R','GEO_Lon','GEO_Lat',
                        'alt (km)','GEI_X','GEI_Y','GEI_Z','GEI_VX','GEI_VY','GEI_VZ',
                        'ECD_Radius','ECD_Lon','ECD_Lat','ECD_MLT','L_Shell','B_Mag','MLT']

            # Get the indices of the cols requested in ocols
            colinds = [i for i, val in enumerate(allocols) if val in ocols]
            #colinds.sort()

            #Check that cols 0-2 are in the list because those are needed for time
            for co in np.arange(2, -1, -1):
                # If they are not there, then add then to colinds and ocols
                if co not in colinds:
                    colinds = [co] + colinds
                    ocols.insert(0, allocols[co])
                    
            fco = 0
            for file in ofiles:
                # Need to deal with multiple files
                if fco == 0:
                    allodata = np.loadtxt(file, skiprows=60, usecols=tuple(colinds))
                else:
                    data = np.loadtxt(file, skiprows=60, usecols=tuple(colinds))
                    allodata = np.append(allodata, data,axis=0)
                fco = fco + 1
            # Now add allodata to all data
            # If it is just one column you can't use append which is dumb
            if len(colinds)>1:
                # First check that they are the same length because sometimes the data times jump
                if len(allodata)!=len(alldata):
                    # I have to interpoalte every point to time which is a pain in the but
                    odtime = [dt.datetime(int(allodata[x, fcols.index('year')]), 1, 1) + 
                              dt.timedelta(days=int(allodata[x, fcols.index('doy')] - 1),
                        seconds=int(allodata[x, fcols.index('sec')])) for x in range(0, len(allodata))]
                    otime = pu.unixtime(odtime)
                    dtime = pu.unixtime(time)
                    for colco in range(3,len(colinds)):
                        newcol = np.interp(dtime,otime,allodata[:,colco])
                        alldata = np.insert(alldata, np.shape(alldata)[1], newcol, axis=1)
                        
                else:
                    alldata = np.append(alldata,allodata[:,3::],axis=1)
            else:
                alldata = np.insert(alldata, len(fcols), allodata, axis=1)
            fcols.extend([allocols[x] for x in colinds[3::]])
        # This is surprisingly fast
        sinds=[i for i, val in enumerate(time) if ((val-sdate).total_seconds()>0)]
        sind=sinds[0]
        einds = [i for i, val in enumerate(time) if ((val - edate).total_seconds() > 0)]
        eind=einds[0]
        time = time[sind:eind]
        alldata = alldata[sind:eind, :]

    else:
        time = None
        alldata = None
        fcols = None
    return time, alldata, fcols


def main(arg):
    # PURPOSE: The purpose of these utilities is to be able to read in and work with
    # ascii data sets like HEO and GPS. To show how they work we make some simple plots here
    # of the two kinds of data
    # INPUTS: None

    # These are needed to define where the data are
    root_dir = '/Users/janet/PycharmProjects/data/mag.gmu.edu/ftp/users/obrien/heo/ascii'
    lowdir = 'SAT/exp/YYYY'

    # This will get whatever data files are in the root_dir/lower_dir but it assumes there is
    # just one kind of data (i.e. Lbin or exp) in there

    sat = 'f1'
    sdate = dt.datetime(2006,1,1)
    edate = dt.datetime(2006,1,3)

    hdata = get_HEO_text(sat,sdate,edate,root_dir,lower_dir = lowdir)

    root_dir = '/Users/janet/PycharmProjects/data/ico/'
    lowdir ='ICO_YYYY/'
    idata = get_ICO_text(sdate,edate,root_dir,lower_dir = lowdir)

    dstList = get_dsts( sdate, edate )

    meandst = get_meanDST( dstList )

    mindst = get_minDST( dstList )

    atdate = dt.datetime( 2006, 1, 2 )
    precedingdst = get_precedingDST(dstList, atdate)

    if (hdata):
        plt.plot(hdata['Prot1'])

    root_dir = '/Users/janet/PycharmProjects/data/' \
               'www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/gps/data/'
    lowdir = 'SAT/'
    sat = 'ns53'
    # May add vars = here to just get some data
    gtime,gdata = get_GPS_text(sat, sdate, edate, root_dir, lower_dir=lowdir)
    plt.plot(gdata['Prot1'])


if __name__ == '__main__':
    main(sys.argv[1:])
