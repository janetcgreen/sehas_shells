//
// make_invariant_integral.cpp
// was make_invariant_integral_opq.cpp
//
// C wrapper of AER Magfield model for retrieving Bx, By, Bz, Bmin and I in a single call
// for a passed set of times, positions and pitch angles.
//
// input:
//   szDbPath      - NULL terminated string containing the path and filename of the igrfDB.h5 database
//                   used by the magfield model
//   piNumTimes    - ptr to integer count of time values passed (ptr only for fortran compatibility)
//   pdTimes       - mjd, decimal
//   piSysAxes     - ptr to integer Irbemlib coordinate system id of passed position, as follows:
//                   0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
//   pdX1,pdX2,pdX3   - arrays of iNumtimes position coordinates
//   piNumAngles   - ptr to integer count of number of pitch angles per time passed
//   piFixedAngles - ptr to integer flag (if 1: one set of pitch angles passed)
//                                       (if 0: one set of pitch angles per time passed (iNumTimes*iNumAngles)
//   pdAngles      - array of double n pitch angles (where n defined by piFixedAngles,piNumAngles,piNumtimes)
//   pikext         - pointer to integer Irbemlib external field model as follows:
//                    0:IGRF, 4:T89, 5:OPQ (constants provided in mag_util.h)
//   pdMagInput     - array of magnetic field inputs, (iNumTime x 25), row major, ignored by static fields
//                    pdMagInput[*,0] = Kp*10
//
// output: 
//   (arrays must be sized correctly by caller)
//   pdBGEO        - array of 3 * iNumTimes B components Bx By Bz in row major order by time
//   pdBmin        - array of iNumTimes B minimum value for each time
//   pdBminXyz     - array of 3 * iNumTimes B minimum position for each time (xyz)
//   pdI           - array of iNumTimes * iNumAngles I values
//   for all output arrays, NaN indicates bad/missing data

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
#include "make_invariant_integral.h"

static auto_ptr<CMagfield> pModel;
static int pModel_kext = -1; // identifies prior external field model

const double dEarthRadiusKM = 6371.2;
const double dIflag = 100.0; // flag value for I
const double dbad_data = NAN; // flag value for I

// computeAdInvariants wants pitch angles in descending order, so we have to sort and unsort w/ indices
void populate_and_sort_pitchangles(const double * pdAngles, const int iNumAngles, ivector &viAngleIndices, dvector &vdPitchAnglesSorted);
// populates viAngleIndices and VdPitchAnglesSorted from pdAngles w/ sort

// declare the primary fortran subroutine
// need to grab onera lib code, go in and edit to make the following skip L*
extern "C"
{

  int make_invariant_integral(const char*   szDbPath,    // path & file to IGRFDb.h5
           const int*    piNumTimes, 
           const double* pdTimes,     // [iNumTimes]
           const int*    piSysAxes,
           const double* pdX1,         // [iNumTimes]
           const double* pdX2,         // [iNumTimes]
           const double* pdX3,         // [iNumTimes]
           const int*    piNumAngles,
           const int*    piFixedAngles,
           const double* pdAngles,    // [iNumAngles]
	   const int*    pikext,
	   const double* pdMagInputs,  // [iNumTimes x 25] row major order
           double*       pdBGEO,      // [iNumTimes x 3] row major order
           double*       pdBmin,      // [iNumTimes]
	   double*       pdBminXyz,   // [iNumTimes x 3] row major order
           double*       pdI          // [iNumTimes x iNumAngles] row major order
           )
  {
    if (piNumTimes == NULL ||pdTimes == NULL || piSysAxes == NULL || 
        pdX1 == NULL || pdX2 == NULL || pdX3 == NULL ||
        piNumAngles == NULL || piFixedAngles == NULL || pdAngles == NULL || 
        pdBGEO == NULL || pdBmin == NULL || pdBminXyz == NULL || pdI == NULL)
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

    S3Coord inPos, sphPos, geoPos;
    S3Tuple Bxyz;
    emfCoordSys eInCoordSys;
    bool bWarningsReturned = false;
    // these are only true when using OPQ
    bool iUseOriginal = (*pikext==KEXT_OPQ);
    bool iOrigExternFieldOn = (*pikext==KEXT_OPQ);

    dvector vdPitchAnglesSorted; // sorted pitch angles
    ivector viAngleIndices; // after sort vdPitchAnglesSorted[i] == vdPitchAnglesSorted[viAngleIndicesSorted[i]]

    dvector vdI;
    dvector vdLm;
    dvector vdB;
    double  dBmin;
    double  dBminXyz[3];

    // Note: if *piFixedAngles = 0, angles passed for each coord
    //       else, only 1 set of pitch angles passed
    if (*piFixedAngles) { // pass pitch angles in a vector -- precompute if they're fixed
      populate_and_sort_pitchangles(pdAngles,*piNumAngles,viAngleIndices,vdPitchAnglesSorted);
    }

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

      // pass pitch angles in a vector
      // Note: if *piFixedAngles = 0, angles passed for each coord
      //       else, only 1 set of pitch angles passed

      if ((*piFixedAngles) == 0) { // compute the sorted pitch angles here (otherwise, precomputed above)
        populate_and_sort_pitchangles(pdAngles+iTime*(*piNumAngles),*piNumAngles,viAngleIndices,vdPitchAnglesSorted);
      }

      // Retrieve the data from the model
      // Note: computeBfield needs coords in GEO, computeAdInvariants in RLL

      if (errM == emfNoError) {
        errM = pModel->convertCoord( eInCoordSys, RLLinKM, inPos, &sphPos );
      }

      if (errM == emfNoError || errM == emfWarning)
          errM = pModel->computeAdInvariants( pdTimes[iTime], sphPos, vdPitchAnglesSorted, 
                                              iUseOriginal, iOrigExternFieldOn, &dBmin, dBminXyz, &vdB, &vdLm, &vdI );

      if (errM == emfWarning) {
        bWarningsReturned = true;
        errM = emfNoError;
      }
      else if (errM == emfOutOfRange) {
        vdB[0] = 0.0;
        vdB[1] = 0.0;
        vdB[2] = 0.0;
        dBmin  = 0.0;
        errM = emfNoError;
      }

      if (errM == emfNoError) {

        // process response data

        pdBGEO[iTime*3 + 0] = vdB[0];
        pdBGEO[iTime*3 + 1] = vdB[1];
        pdBGEO[iTime*3 + 2] = vdB[2];
        pdBmin[iTime] = dBmin;
        pdBminXyz[iTime*3 + 0] = dBminXyz[0];
        pdBminXyz[iTime*3 + 1] = dBminXyz[1];
        pdBminXyz[iTime*3 + 2] = dBminXyz[2];

        for (int iAngle=0; iAngle<(*piNumAngles); iAngle++) {
          if (vdI[iAngle] == dIflag) vdI[iAngle] = dbad_data; // replace I==100 with bad_data
            pdI[iTime*(*piNumAngles)+viAngleIndices[iAngle]] = vdI[iAngle];
        }

        // cleanup for next call

        vdLm.clear();
        vdI.clear();
      }
    }

    // flag warnings if no more serious errors occurred

    if (bWarningsReturned && errM == emfNoError)
      errM = emfWarning;
    
    return ( (int) errM );
  }

  int make_invariant_integral_opq(const char*   szDbPath,    // path & file to IGRFDb.h5
           const int*    piNumTimes, 
           const double* pdTimes,     // [iNumTimes]
           const int*    piSysAxes,
           const double* pdX1,         // [iNumTimes]
           const double* pdX2,         // [iNumTimes]
           const double* pdX3,         // [iNumTimes]
           const int*    piNumAngles,
           const int*    piFixedAngles,
           const double* pdAngles,    // [iNumAngles]
           double*       pdBGEO,      // [iNumTimes x 3] row major order
           double*       pdBmin,      // [iNumTimes]
	   double*       pdBminXyz,   // [iNumTimes x 3] row major order
           double*       pdI          // [iNumTimes x iNumAngles] row major order
           )
  {
    // wrapper for make_invariant_integral, with kext = OPQ
    int kext = KEXT_OPQ;
    double *pdMagInputs=NULL;
    return make_invariant_integral(szDbPath,piNumTimes,pdTimes,piSysAxes,
				   pdX1,pdX2,pdX3,piNumAngles,piFixedAngles,
				   pdAngles,&kext,pdMagInputs,pdBGEO,pdBmin,pdBminXyz,pdI);
  }
} // extern "C"


// because sort doesn't support extra arguments, and C++ does not allow nested functions,
// we have to declare vdPitchAngles as a global.
// the alternative appears to be to define a special class to handle this sort
// but that corrupts the namespace just as much with a new class instead of a new variable.
dvector vdPitchAngles; // temp variable, storing one "row" of pitch angles
bool sortIndicesDescending( int iA, int iB ) {return(vdPitchAngles[iA]>vdPitchAngles[iB]);};

void populate_and_sort_pitchangles(const double * pdAngles, const int iNumAngles, ivector &viAngleIndices, dvector &vdPitchAnglesSorted) {
  // populate viAngleIndices and vdPitchAnglesSorted with pdAngles
  // iNumAngles gives count of pitch angles pointed to by pdAngles
  //  vdPitchAnglesSorted: // sorted pitch angles
  //  viAngleIndices:  after sort vdPitchAnglesSorted[i] == vdPitchAnglesSorted[viAngleIndicesSorted[i]]

  // reset vectors
  vdPitchAngles.clear(); // this is a global
  vdPitchAnglesSorted.clear();
  viAngleIndices.clear();
  
  // populate vectors
  for (int iAngle=0; iAngle< iNumAngles; iAngle++) {
    vdPitchAngles.push_back( pdAngles[iAngle] ); /* copy from pointer */
    viAngleIndices.push_back(iAngle); // set AngleIndices to 1:N
  }
  
  // sort indices for desending pitch angles
  sort( viAngleIndices.begin(), viAngleIndices.end(), sortIndicesDescending);
  
  // apply sort to pitch angles
  for (int iAngle=0; iAngle< iNumAngles; iAngle++) {
    vdPitchAnglesSorted.push_back(vdPitchAngles[viAngleIndices[iAngle]]);
  }
}

