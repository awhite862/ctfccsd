#!/bin/bash
CXXFLAGS="-g -O2"
LIB_PATH="-L$MKLROOT/lib/intel64" LIBS="-lmkl_intel_lp64 -lmkl_avx2 -lmkl_sequential -lmkl_core -lpthread -lm" LD_LIB_PATH="-L$MKLROOT/lib/intel64" LD_LIBS="-lmkl_intel_lp64 -lmkl_avx2 -lmkl_sequential -lmkl_core -lpthread -lm" ../configure --build-hptt
