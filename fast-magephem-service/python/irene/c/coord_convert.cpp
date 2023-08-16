//
// coord_convert.cpp
//
// C wrapper of AER Coordinate transform retrieving many (Y1,Y2,Y3) vectors from many (X1,X2,X3) vectors in one call
// for a passed set of times and positions and the desired coordinate transform
//
// input:
//   szDbPath      - NULL terminated string containing the path and filename of the igrfDB.h5 database
//                   used by the magfield model
//   piNumTimes    - ptr to integer count of time values passed (ptr only for fortran compatibility)
//   pdTimes       - mjd, decimal
//   piSysAxesX     - ptr to integer Irbemlib INPUT coordinate system id of passed position, as follows:
//                   0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
//   piSysAxesY     - ptr to integer Irbemlib OUTPUT coordinate system id of passed position, as follows:
//   pdX1,pdX2,pdX3   - arrays of iNumtimes position coordinates
// output: 
//   (arrays must be sized correctly by caller)
//   pdY1,pdY2,pdY3   - arrays of iNumtimes position coordinates
// bad data are returned as NaN

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath> /* NAN */
using namespace std;

#include "GenericModel.h"
typedef S3Tuple S3Coord;
#include "CMagfield.h"
#include "CoordXform.h"
#include "GeoSpaceTime.h"

#include "coord_convert.h"

static unique_ptr<CMagfield> pModel;

const double dEarthRadiusKM = 6371.2;

extern "C"
{

  int sysAxes_enum(const int* piSysAxes) {
    // convert int sysAxes to emfCoordSys enum typecast to int
    // return -1 on fail

    // set output coordinate system (only need to do this once)
    switch (*piSysAxes) {

      case 0: // GDZ alt(km), lat(dg), lon(dg)
        return (int)GDZinKM;

      case 1: // GEO cartesian (RE)
        return (int)GEOinRE;

      case 2: // GEO solar mag (RE)
        return (int)GSMinRE;

      case 3: // GEO solar ecliptic (RE)
        return (int)GSEinRE;

      case 4: // Solar magnetic (RE)
        return (int)SMinRE;

      case 5: // GEO equitorial inertial (RE)
        return (int)GEIinRE;

      case 6: // Geomagnetic (RE)
        return (int)MAGinRE;

      case 7: // Spherical (RE, colat dg, lon dg)
        return (int)SPHinRE;

      case 8: // Radial lat lon (Re, lat dg, lon dg)
        return (int)RLLinRE;

      default:
        return -1;
      }
  }

  int coord_convert( const char*   szDbPath,    // path & file to IGRFDb.h5
		     const int*    piNumTimes, 
		     const double* pdTimes,     // [iNumTimes]
		     const int*    piSysAxesX,
		     const int*    piSysAxesY,
		     const double* pdX1,         // [iNumTimes]
		     const double* pdX2,         // [iNumTimes]
		     const double* pdX3,         // [iNumTimes]
		     double* pdY1,         // [iNumTimes]
		     double* pdY2,         // [iNumTimes]
		     double* pdY3         // [iNumTimes]
		     )
  {
    if (piNumTimes == NULL || pdTimes == NULL || piSysAxesX == NULL || 
	piSysAxesY == NULL || pdX1 == NULL || pdX2 == NULL || pdX3 == NULL ||
	pdY1 == NULL || pdY2 == NULL || pdY3 == NULL)
      {
        return ( (int) eInvalidNullPointer );
      }

    eGENERIC_ERROR_CODE errG=eNoError;
    eMAGFIELD_ERROR_CODE errM=emfNoError;

    // If this is the first call, instantiate and initialize the magfield model

    if (pModel.get() == NULL) {

      pModel = unique_ptr<CMagfield> (new CMagfield());

      errG = pModel->Initialize(szDbPath);
      if (errG != eNoError) {
        return (int)errG;
      }

      errM = pModel->setMainField(emfFastIGRF);
      if (errM != emfNoError) {
        return (int)errM;
      }

      errM = pModel->setExternalField(eefOlsonPfitzer);
      if (errM != emfNoError) {
        return (int)errM;
      }
    }

    // Loop through the timestamps and call the magfield model
    // once to retrieve data for each time, position and set of pitch angles.

    S3Coord inPos, outPos;
    emfCoordSys eInCoordSys, eOutCoordSys;
    bool bWarningsReturned = false;

    int sys = sysAxes_enum(piSysAxesY);
    if (sys < 0) return emfConvertCoordFailed;
    eOutCoordSys = (emfCoordSys)sys;

    for (int iTime = 0; ((iTime < *piNumTimes) && (errM == emfNoError)); iTime++) {


      // do this every time in case eInCoordSys gets modified
      sys = sysAxes_enum(piSysAxesX);
      if (sys < 0) return emfConvertCoordFailed;
      eInCoordSys = (emfCoordSys)sys;

      pModel->updateTime( pdTimes[iTime] );

      inPos.x = pdX1[iTime];
      inPos.y = pdX2[iTime];
      inPos.z = pdX3[iTime];

      if (errM == emfNoError) {
	errM = pModel->convertCoord(eInCoordSys,eOutCoordSys,inPos,outPos);
      }

      if (errM == emfWarning) {
        bWarningsReturned = true;
        errM = emfNoError;
      }
      else if (errM == emfOutOfRange) {
	outPos.x = NAN;
	outPos.y = NAN;
	outPos.z = NAN;
        errM = emfNoError;
      }
      
      // store in output array
      pdY1[iTime] = outPos.x;
      pdY2[iTime] = outPos.y;
      pdY3[iTime] = outPos.z;

    }

    // flag warnings if no more serious errors occurred

    if (bWarningsReturned && errM == emfNoError)
      errM = emfWarning;
    
    return ( (int) errM );
  }

} // extern "C"

