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
//
// also supplies utility funciton sysAxes_enum that
// converts an integer sysaxes to an emfCoordSys enum
// input:
//   piSysAxes     - ptr to integer Irbemlib INPUT coordinate system id of passed position, as follows:
//                   0:GDZ, 1:GEO, 2:GSM, 3:GSE, 4:SM, 5:GEI, 6:MAG, 7:SPH, 8:RLL
// output:
// eOutCoordSys - pointer to int typecase of resulting emfCoordSys enum
// -1 on fail


#ifndef COORD_CONVERT_H
#define COORD_CONVERT_H

#ifdef __cplusplus
extern "C" {
#endif

  int sysAxes_enum(const int* piSysAxes);
  
  int coord_convert(const char*   szDbPath,    // path & file to IGRFDb.h5
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
		    );
  
#ifdef __cplusplus
}
#endif

#endif
