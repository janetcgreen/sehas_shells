# TLE propagator 
# 
# Utilizes the SGP4 propagator for python to create inertial (ECI) position and velocity
# at a passed time. Supports position generation for individual times, as well as for
# a range of times at a passed cadence. Also supports conversion of individual position
# and velocity coordinates to earth centered, earth fixed coordinates as radius (km),
# latitude (dg) and longitude (dg)
#
# Instance methods:
# Propagator.position(y,m,d,h,m,s,line1,line2) - Given time, tle; compute position, velocity 
# Propagator.positionDt(datetime,line1,line2)  - Given time, tle; compute position, velocity
# Propagator.eci2ecef(dateTime,x,y,z)          - Given time, eci pos, compute ecef position
# Propagator.addTLE(dateTime,line1,line2)      - Add time, tle to list for propagation over time interval
# Propagator.clearTLEs()                       - Empties internal TLE list
# Propagator.propagate(start,end,cadence)      - Given time window, cadence, using TLE list, computes positions and velocities 
#
# Modification history:
# 12/30/3030 Created (extracted from makeChargeHaz.py)

import abc
import datetime as dt
import math
from jdcal import gcal2jd
import numpy as np
from sgp4.earth_gravity import wgs72, wgs72old, wgs84
#from sgp4.api import Satrec
from sgp4.io import twoline2rv
import logging
from _operator import pos
from tleHandler import Tle

class Propagator(object, metaclass=abc.ABCMeta):
   
   # String names of gravitational constants supported in SGP4
   GRAVITY_WGS72 = "wgs72"
   GRAVITY_WGS72_OLD = "wgs72old"
   GRAVITY_WGS84 = "wgs84"

   # This runs when an instance is declared
   def __init__(self, gravity):
       
      self.gravity = gravity
      self.tles = []
      
   # Compute orbit position at a single time instant 
   # Returns position, velocity in inertial coordinates (km, km/s)
   def position(self, yr, mon, day, hr, min, sec, tleLine1, tleLine2):
      
      gc = self.__getGravityConstants( self.gravity )
      
      satellite = twoline2rv( tleLine1, tleLine2, gc )
      
      position, velocity = satellite.propagate( yr, mon, day, hr, min, sec )
      
      return position, velocity

   # Compute orbit position at a single time instant (passing dateTime)   
   # Returns position, velocity in inertial coordinates (km, km/s)
   def positionDt(self, atDateTime, tleLine1, tleLine2 ):
      
      return self.position( atDateTime.year, atDateTime.month, atDateTime.day, 
                            atDateTime.hour, atDateTime.minute, atDateTime.second, 
                            tleLine1, tleLine2 )
      
   # Inertial to ECEF position conversion
   # returns radius (km), latitude (Dg), longitude (Dg)
   def eci2ecef(self, atDateTime, xeci, yeci, zeci):
   
      r = np.sqrt( xeci * xeci + yeci * yeci + zeci * zeci )
       
      lat = math.asin( zeci / r ) * 180.0 / np.pi # lat in degrees

      # This is from http://aa.usno.navy.mil/faq/docs/GAST.php
      # Julian date is the fractional days since BC sometime
      # This gives the Julian date for midnight of the calendar date
      # so you have to add the time
      # For some reason it gives 2 values that you add together

      JDpiece = gcal2jd( atDateTime.year, atDateTime.month, atDateTime.day )   # These are the two pieces of the Julian date
      fracday = ( atDateTime.hour * 60.0 + atDateTime.minute ) / ( 24 * 60.0 ) # fraction of day to add
      JD =JDpiece[0] + JDpiece[1] + fracday                                    # The actual Julian day
      D = JD - 2451545.0
      T = D / 36525.0
      D0 = JD - fracday - 2451545.0 # Coefficient for getting the sidereal day
      GMST1 = 6.697374558 + 0.06570982441908*D0 + 1.00273790935 * fracday * 24.0 + 0.000026 * T * T
      GMST = GMST1 - np.floor( GMST1 / 24 ) * 24
      # This is the sideral time (hours) that I've checked with other online calcualtor
      lon = ( math.atan2( yeci, xeci ) - GMST * np.pi / 12.0 ) * 180.0 / np.pi

      return r, lat, lon
   
   
   # Build collection of TLEs for propagation over time
   def addTLE(self, atDateTime, tleLine1, tleLine2):
      
      tleItem = Tle()
      tleItem.dateTime = atDateTime.timestamp()
      tleItem.line1 = tleLine1
      tleItem.line2 = tleLine2
      self.tles.append( tleItem )
      self.tles.sort( key=Propagator.sortTlesByDate )
      
   def clearTLEs(self):
      
      self.tles.clear()
      
   # Compute orbit position at a single time instant (passing dateTime)   
   # Returns position, velocity in inertial coordinates (km, km/s)
   
   def positionDtUsingTLEList(self, atDateTime ):
      
      dateTimestamp = int( atDateTime.timestamp())

      tleIndex = 0 # TLEs sorted ascending by time

      if len(self.tles) == 0 :
         logging.exception('NO TLES set in propagator')
         raise
      
      else :
         if self.tles[0].dateTime > dateTimestamp :
             logging.exception('NO TLES preceding position time')
             raise
         
      # find TLE to use
     
      for atTle in range( tleIndex, len( self.tles )) :
        
         if self.tles[ atTle ].dateTime <= dateTimestamp :
             tleIndex = atTle
         else :
             break;
        
      return self.position( atDateTime.year, atDateTime.month, atDateTime.day, 
                            atDateTime.hour, atDateTime.minute, atDateTime.second, 
                            self.tles[ tleIndex ].line1,  self.tles[ tleIndex ].line2 )
      
      
   # Generate list of lists of time, position and velocity over a passed time
   # window, over a passed cadence. Returns [ [t,x,y,z,u,v,w], ... ]
   def propagate(self, startDateTime, endDateTime, cadenceSeconds ):

      startDateTimestamp = int( startDateTime.timestamp())
      endDateTimestamp = int( endDateTime.timestamp())
      
      orbit = []
      
      # Generate timesteps from start to end, by cadence
      
      diffSeconds = endDateTimestamp - startDateTimestamp
      times = [ startDateTimestamp + deltaT for deltaT in range( 0, diffSeconds, cadenceSeconds ) ]
      
      # For each timestep, find TLE, compute position
      
      tleIndex = 0 # sorted ascending by time

      if len(self.tles) == 0 :
         logging.exception('NO TLES set in propagator')
         raise
      
      else :
         if self.tles[0].dateTime > startDateTimestamp :
             logging.exception('NO TLES preceding propagation time')
             raise
      
      for atTime in times :
         
         # find TLE to use
         
         for atTle in range( tleIndex, len( self.tles )) :
            
            if self.tles[ atTle ].dateTime <= atTime :
               tleIndex = atTle
            else :
               break;
            
         # compute position
         
         atDateTime = dt.datetime.utcfromtimestamp( atTime )
         pos, vel = self.positionDt( atDateTime, self.tles[ tleIndex ].line1,  self.tles[ tleIndex ].line2 )
         
         orbitItem = [ atDateTime ]
         orbitItem.extend( list( pos ) )
         orbitItem.extend( list( vel) )

         orbit.append( orbitItem )
         
      return orbit
      
   # Sort Tles by date
   def sortTlesByDate( listItem ):
      return listItem.dateTime
   
   # Private method to convert gravity constant names into SGP4 objects
   def __getGravityConstants(self, gravityString ):

      if self.gravity == "wgs72old" :
         return wgs72old
      elif self.gravity == "wgs84" :
         return wgs84
      else :
         return wgs72    # default


# Extract yy,ddd.fffff from tle line 1, convert to datetime
def tle2dt( line1, line2 ):
   
   yy = line1[18:20]
   doy = line1[20:23]
   pctDay = '0' + line1[23:32]
   
   yyyy = 2000 + int(yy)
   if yyyy > dt.date.today().year :
      yyyy = 1990 + int(yy)
      
   atDt = dt.datetime( yyyy, 1, 1) + dt.timedelta( int(doy) - 1)
   
   secs = float( pctDay) * 86400.0
   atDt = atDt + dt.timedelta( seconds = secs)
   
   return atDt

      
# Command-line access
if __name__ == "__main__":
   
   #  PARSE COMMAND LINE ARGUMENTS
   import argparse

   parser = argparse.ArgumentParser('Propagates TLEs into ephemeris using SGP4')
   parser.add_argument('-s', "--startdate",
                        help="The Start Date - format YYYY-MM-DD HH:MM:SS ",
                        required=False )
   parser.add_argument('-e', "--enddate",
                        help="The Start Date - format YYYY-MM-DD HH:MM:SS ",
                        required=False )
   parser.add_argument('-c', "--cadenceSecs",
                        help="Propagation cadence between start and end (in seconds)",
                        type=int,
                        required=False )
   parser.add_argument('-g', "--gravity",
                       help="Gravity string constant (defined in Propagator)",
                       required=False, default=Propagator.GRAVITY_WGS72 )
   parser.add_argument('-t', "--tles",
                        help="List of TLE lines (must be multiples of 2)",
                        nargs="+", type=str,
                        required=False )
   args = parser.parse_args()
    
   if args.startdate is None or \
      args.tles is None or \
      len(args.tles) < 2 or \
      len(args.tles) % 2 != 0 :
      print("Running from command line requires start date and at least one set of tle lines")
      
   elif args.enddate is not None :
      
      if args.cadenceSecs is None :
         print("Propagation over time requires cadence")
         
      else :
         
         # propagate over a time window
         
         startAt = dt.datetime.strptime( args.startdate, '%Y-%m-%d %H:%M:%S')
         endAt = dt.datetime.strptime( args.enddate, '%Y-%m-%d %H:%M:%S')
         cadence = int( args.cadenceSecs )
         
         p = Propagator( args.gravity )
         for i in range(0, len(args.tles), 2) :
            atDateTime = tle2dt( args.tles[i], args.tles[i+1] )
            p.addTLE( atDateTime, args.tles[i], args.tles[i+1] )
            
         results = p.propagate( startAt, endAt, cadence )
         
         print( results )
   
   else :
      
      # propagate a position at an instant in time
      
      startAt = dt.datetime.strptime( args.startdate, '%Y-%m-%d %H:%M:%S')
      p = Propagator( args.gravity )
      pos, velocity = p.positionDt( startAt, args.tles[0], args.tles[1] )
      
      print( pos, velocity )
