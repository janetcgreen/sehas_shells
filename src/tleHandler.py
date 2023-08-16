import datetime as dt
import time
import os
import mysql.connector
import logging
from numpy.distutils.fcompiler import none

class Tle :
   
   def __init__( self ):
      
      self.noradId = 0
      self.dateTime = 0 # unix timestamp
      self.line1 = ""
      self.line2 = ""


class TleHandler :
   
   #
   # Instance initialization
   #
   def __init__( self ):
      
      return
      
   #
   # Read most recent tle for noradId, preceding passed date-time
   #
   def readPrecedingTLE( self, cnx, noradId, atDateTime ):
      
      tle = None
      
      try :

         cursor = cnx.cursor()
         
         query = "SELECT t.unixDateTime_utc, t.line1, t.line2 " + \
                 "FROM TwoLineElementTbl t " + \
                 "WHERE t.unixDateTime_utc = " + \
                 "( SELECT t2.unixDateTime_utc FROM TwoLineElementTbl t2 " + \
                 "  WHERE t2.noradId = %s AND t2.unixDateTime_utc < %s " + \
                 "ORDER BY t2.unixDateTime_utc DESC LIMIT 1 )"

         cursor.execute( query, ( noradId, atDateTime, ))
   
         result = cursor.fetchone()
         if result is not None :

            tle = Tle()
            tle.noradId = noradId
            tle.dateTime = result[0]
            tle.line1 = result[1]
            tle.line2 = result[2] 
   
         cursor.close()
         
         return tle
   
      except Exception as e :
         logging.exception( e )
         cursor.close()
         raise
   
   #
   # Read all tles for a noradId within a passed date-time window. Optionally include
   # the most recent tle preceding the window.
   #
   def readTLEsBetween( self, cnx, noradId, startDateTime, endDateTime, includePrecedingTLE ):
      
      tleList = []
      
      try :

         cursor = cnx.cursor()
         
         query = "SELECT unixDateTime_utc, line1, line2 " + \
                 "FROM TwoLineElementTbl " + \
                 "WHERE noradId = %s AND " + \
                 "unixDateTime_utc >= %s AND unixDateTime_utc < %s " + \
                 "ORDER BY unixDateTime_utc ASC"

         cursor.execute( query, ( noradId, startDateTime, endDateTime ))
   
         for ( unixDateTime_utc, line1, line2 ) in cursor :

            tle = Tle()
            tle.noradId = noradId
            tle.dateTime = unixDateTime_utc
            tle.line1 = line1
            tle.line2 = line2 
            
            tleList.append( tle )
   
         cursor.close()
         
         # optionally include tle preceding time window
         
         if includePrecedingTLE :
            tle = self.readPrecedingTLE( cnx, noradId, startDateTime )
            if tle is not None:
               tleList.insert( 0, tle )
         
         return tleList
   
      except Exception as e :
         logging.exception( e )
         cursor.close()
         raise
   
   
   def writeTLEs( self, cnx, tleList ):
      
      # TBD
      return None
   
