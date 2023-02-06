//
// make_invariant_integral.h
// formerly make_invariant_integral_opq.h
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
//                   1:GDZ, 2:GEO, 3:GSM, 4:GSE, 5:SM, 6:GEI, 7:MAG, 8:SPH, 9:RLL
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
//   pdBminXyz     - array of 3 * iNumTimes Bmin xyz coordinates for each time
//   pdI           - array of iNumTimes * iNumAngles I values
// bad data are returned as NaN

#ifndef MAKE_INVARIANT_INTEGRAL_H
#define MAKE_INVARIANT_INTEGRAL_H

#ifdef __cplusplus
extern "C" {
#endif

  int make_invariant_integral(const char*   szDbPath,
			      const int*    piNumtimes, 
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
			      double*       pdBminxyz,   // [iNumTimes x 3] row major order
			      double*       pdI          // [iNumTimes x iNumAngles] row major order
			      );

  int make_invariant_integral_opq(const char*   szDbPath,
				  const int*    piNumtimes, 
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
				  double*       pdBminxyz,   // [iNumTimes x 3] row major order
				  double*       pdI          // [iNumTimes x iNumAngles] row major order
				  );

#ifdef __cplusplus
}
#endif

#endif
