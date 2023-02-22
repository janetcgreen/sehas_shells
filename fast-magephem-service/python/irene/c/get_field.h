//
// get_field
// get_field_opq - wrapper that assumes pikext = OPQ
//
// C wrapper of AER Magfield model for retrieving Bx, By, Bz in a single call
// for a passed set of times, positions and pitch angles. Supports OPQ, IGRF, T89
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
//                    0:IGRF, 4:T89, 5:OPQ (constants provided in mag_util.h)
//   pdMagInput     - array of magnetic field inputs, (iNumTime x 25), row major, ignored by static fields
//                    pdMagInput[*,0] = Kp*10
//                    width 25 is defined by MAGINPUTS_WIDTH constant in mag_util.h
// 
// output: 
//   (arrays must be sized correctly by caller)
//   pdBGEO        - array of 3 * iNumTimes B components Bx By Bz in row major order by time (nT)
//   pdBmag        - array of iNumTimes B magnitude value for each time (nT)
// bad data are returned as NaN

#ifndef GET_FIELD_H
#define GET_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

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
		    );

  int get_field_opq(const char*   szDbPath,    // path & file to IGRFDb.h5
		    const int*    piNumTimes, 
		    const double* pdTimes,     // [iNumTimes]
		    const int*    piSysAxes,
		    const double* pdX1,         // [iNumTimes]
		    const double* pdX2,         // [iNumTimes]
		    const double* pdX3,         // [iNumTimes]
		    double*       pdBGEO,      // [iNumTimes x 3] row major order
		    double*       pdBmag      // [iNumTimes]
		    );

#ifdef __cplusplus
}
#endif

#endif
