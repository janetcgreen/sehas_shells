# This Makefile assumes the following local soft links exist:
#  models_include to ${SPWX}/Models/include
#  common_include to ${SPWX}/Common/include
#  lib to wherever libcmagfield.so and libspwxcommon.so live (e.g., a bin/linux folder or ~/lib)
# "make install" will make all the needed DLLs and install them in irene/lib (../lib)
# "make libmag_util.so" will make just the one DLL and not install it
# "make clean" will clean up the compiler and editor detritus
libmag_util.so: coord_convert.cpp make_invariant_integral.cpp get_field.cpp
	g++ -g -fPIC -shared -Wl,-rpath,${CURDIR}/lib -o $@ $^ -Imodels_include -Icommon_include -Llib -lcmagfield -lspwxcommon -lCoordXform -ladiabat -lhdf5 -lgfortran -lrt

install: libmag_util.so
	\cp libmag_util.so ../lib/

clean:
	\rm -f *.o *~


