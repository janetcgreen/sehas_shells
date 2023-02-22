//
// get_field
// get_field_opq - wrapper that assumes pikext = OPQ
//
// C wrapper of AER Magfield model for retrieving Bx, By, Bz in a single call
// for a passed set of times, positions and pitch angles. Supports OPQ, IGRF, T89
// NOTE: T89 support is not available at this time
//
// input: (pointers used for fortran compatability only)
//   szDbPath       - NULL terminated string containing the path and filename of the igrfDB.h5 database
//                    used by the magfield model
//   piNumTimes     - pointer to integer count of time values passed
//   pdTimes        - mjd, decimal
//   piSysAxes      - pointer to integer Irbemlib coordinate system id of passed position, as follows:
//                    0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
//   pdX1,pdX2,pdX3 - arrays of iNumtimes position coordinates
//   pikext         - pointer to integer Irbemlib external field model as follows:
//                    0:IGRF, 4:T89, 5:OPQ
//   pdMagInput     - array of magnetic field inputs, (iNumTime x 25), row major, ignored by static fields
//                    pdMagInput[*,0] = Kp*10

// 
// output: 
//   (arrays must be sized correctly by caller)
//   pdBGEO        - array of 3 * iNumTimes B components Bx By Bz in row major order by time (nT)
//   pdBmag        - array of iNumTimes B magnitude value for each time (nT)
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
#include <cmath>
using namespace std;

#include "GenericModel.h"
typedef S3Tuple S3Coord;
#include "CMagfield.h"
#include "CoordXform.h"
#include "GeoSpaceTime.h"

#include "mag_util.h"
#include "coord_convert.h"
#include "get_field.h"

static auto_ptr<CMagfield> pModel;
static int pModel_kext = -1; // identifies prior external field model

const double dEarthRadiusKM = 6371.2;

extern "C"
{

  int get_field(const char*   szDbPath,    // path & file to IGRFDb.h5
		const int*    piNumTimes, 
		const double* pdTimes,     // [iNumTimes]
		const int*    piSysAxes,
		const double* pdX1,         // [iNumTimes]
		const double* pdX2,         // [iNumTimes]
		const double* pdX3,         // [iNumTimes]
		const int*    pikext,
		const double* pdMagInputs,  // [iNumTimes x 25] row major order
		double*       pdBGEO,      // [iNumTimes x 3] row major order
		double*       pdBmag      // [iNumTimes]
		)
  {
    if (piNumTimes == NULL ||pdTimes == NULL || piSysAxes == NULL || 
        pdX1 == NULL || pdX2 == NULL || pdX3 == NULL ||
        pikext == NULL  ||
        pdBGEO == NULL || pdBmag == NULL)
      {
        return ( (int) eInvalidNullPointer );
      }


    eGENERIC_ERROR_CODE errG=eNoError;
    eMAGFIELD_ERROR_CODE errM=emfNoError;

    // If this is the first call, instantiate and initialize the magfield model

    if ((pModel.get() == NULL) || (pModel_kext != *pikext)) {

      pModel = auto_ptr<CMagfield> (new CMagfield());
      pModel_kext = *pikext; // save to check next call

      errG = pModel->Initialize(szDbPath);
      if (errG != eNoError) {
        return (int)errG;
      }

      errM = pModel->setMainField(emfFastIGRF);
      if (errM != emfNoError) {
        return (int)errM;
      }

      switch(*pikext) {
      case KEXT_IGRF : // IGRF
	errM = pModel->setExternalField(eefNone);
	break;
      case KEXT_T89: // T89
	errM = pModel->setExternalField(eefTsyganenko89);
	break;
      case KEXT_OPQ: // OPQ
	errM = pModel->setExternalField(eefOlsonPfitzer);
	break;
      default:
        return (int)eInitializationFailed;
      }

      if (errM != emfNoError) {
        return (int)errM;
      }
    }

    // Loop through the timestamps and call the magfield model
    // once to retrieve data for each time, position and set of pitch angles.

    S3Coord inPos, geoPos;
    S3Tuple Bxyz;
    emfCoordSys eInCoordSys;
    bool bWarningsReturned = false;
    double Bmag;

    for (int iTime = 0; ((iTime < *piNumTimes) && (errM == emfNoError)); iTime++) {

      // Convert Irbemlib coordinate systems to magfield enums
      // do this every time in case eInCoordSys gets modified
      int sys = sysAxes_enum(piSysAxes);
      if (sys < 0) return emfConvertCoordFailed;
      eInCoordSys = (emfCoordSys)sys;

      pModel->updateTime( pdTimes[iTime] );

      if (*pikext == KEXT_T89) { // set Kp
	if (pdMagInputs == NULL) return ( (int) eInvalidNullPointer ); // requires Kp
	double dKp = pdMagInputs[iTime*MAGINPUTS_WIDTH + 0]/10; // first column is Kp*10
	if (dKp < 0) return (int)emfBadKpAeBin;
	int iKp=-1;
	if (dKp > 4.5) {
	  iKp = 5;
	} else {
	  iKp = (int)(dKp+0.5); // round to nearest int
	}
	errM = pModel->setTsyg89KpDriven(iKp); // set Kp
      }

      inPos.x = pdX1[iTime];
      inPos.y = pdX2[iTime];
      inPos.z = pdX3[iTime];

      if (errM == emfNoError) {
        errM = pModel->convertCoord( eInCoordSys, GEOinKM, inPos, &geoPos ); // get GEO in km
      }

      if (errM == emfNoError || errM == emfWarning) {
	errM = pModel->computeBfield(geoPos,&Bxyz,&Bmag);
      }

      if (errM == emfWarning) {
        bWarningsReturned = true;
        errM = emfNoError;
      }
      else if (errM == emfOutOfRange) {
        Bxyz.x = NAN;
        Bxyz.y = NAN;
        Bxyz.z = NAN;
        Bmag = NAN;
        errM = emfNoError;
      }

      if (errM == emfNoError) {

        // process response data

        pdBGEO[iTime*3 + 0] = Bxyz.x;
        pdBGEO[iTime*3 + 1] = Bxyz.y;
        pdBGEO[iTime*3 + 2] = Bxyz.z;
	pdBmag[iTime] = Bmag;

      }
    }

    // flag warnings if no more serious errors occurred

    if (bWarningsReturned && errM == emfNoError)
      errM = emfWarning;
    
    return ( (int) errM );
  }

  int get_field_opq(const char*   szDbPath,    // path & file to IGRFDb.h5
		    const int*    piNumTimes, 
		    const double* pdTimes,     // [iNumTimes]
		    const int*    piSysAxes,
		    const double* pdX1,         // [iNumTimes]
		    const double* pdX2,         // [iNumTimes]
		    const double* pdX3,         // [iNumTimes]
		    double*       pdBGEO,      // [iNumTimes x 3] row major order
		    double*       pdBmag      // [iNumTimes]
		    )
  {
    // wrapper for get_field, with kext = OPQ
    int kext = KEXT_OPQ;
    double *pdMagInputs=NULL;
    return get_field(szDbPath,piNumTimes,pdTimes,piSysAxes,pdX1,pdX2,pdX3,&kext,pdMagInputs,pdBGEO,pdBmag);
  }

} // extern "C"

