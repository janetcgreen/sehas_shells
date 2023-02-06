README file for Python prototype of IRENE - International Radiation Belt Near Earth

(A successor to AE9/AP9)

Package Version:
Beta for eventual release version 2.0

(c) 2015-2021 The Aerospace Corporation

========================

See LICENSE for terms of use

Applications:

ae9ap9_see_kernel.py -- Application for creating proton SEE kernels (CSDA approximation)

ae9ap9_ic_kernel.py -- Application for creating internal charging kernels (slab to hemisphere approximation)

ae9ap9_dd_kernel.py -- Application for creating proton displacement damage kernels (CSDA approximation)


Directory structure:

./docs - documentation

./examples - contains a python script (examples.py) with several examples, and .png output figures

./irene - the main IRENE python module

./irene/lib - external libraries (DLLs) needed by the python code

./irene/c - C code and makefile for DLLs (see DEPENDENCIES below)

./irene/data - data used by the IRENE model

./irene/data/kernels - XML kernel and shielding library files used by the IRENE model

---------------

DEPENDENCIES:

The python library is written toward Python 3.x.

Most of the dependencies are standard libraries such as os, sys,
numpy, scipy. However, the kernels depend on xmltodict, which often
requies a manual install via pip or similar.

The python library depends on external interface to the magnetic field
and coordinate transform routines provided by the main AE9/AP9 C++
code base. the irene/c folder contains source codes and a make file to
build these dependencies. The make file uses the Gnu Compiler
Collection (gcc). Note that some soft links must be set up first
before the make will succeed. Once the soft links are set up, running
"make install" in irene/c will create and install the needed DLL
(libmag_utils.so)

The following soft links are used for building the external interface
and for testing:

irene/c/lib (points to wherever the libcmagfield.so and libspwxcommon.so reside (e.g., a bin/linux folder or ~/lib)

irene/c/common_include (points to the SpWx_Ae9Ap9/Common/include in the AE9/AP9 source tree)

irene/c/models_include (points to the SpWx_Ae9Ap9/Models/include in the AE9/AP9 source tree)

---------------

CONTACT:

Paul O'Brien, paul.obrien@aero.org

The Aerospace Corporation
