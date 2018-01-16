### INSTALLATION TARGET DIRECTORY (for make install)
INSTALL_DIR = /usr/local/

### LINK TIME LIBRARIES AND FLAGS
#libraries and flags for link time (irrelevant if only building CTF lib and not examples/tests)
LIB_PATH     = -L/usr/local/apps/intel/mkl/lib/intel64 -L/home/qssun/workspace/program/ctf/build-dbg/hptt/lib
#LIB_FILES    = -lmkl_intel_lp64 -lmkl_avx2 -lmkl_gf_thread -lmkl_core -lpthread -lm -Wl,-Bstatic -lhptt -Wl,-Bdynamic
LIB_FILES    = -lmkl_intel_lp64 -lmkl_avx2 -lmkl_core -lpthread -lm -Wl,-Bstatic -lhptt -Wl,-Bdynamic
LINKFLAGS    = 
LD_LIB_PATH  = /usr/local/apps/intel/mkl/lib/intel64:/home/qssun/workspace/program/ctf/build-dbg/hptt/lib
SO_LIB_PATH  = -L/usr/local/apps/intel/mkl/lib/intel64 -L/home/qssun/workspace/program/ctf/build-dbg/hptt/lib -Wl,-rpath=/usr/local/apps/intel/mkl/lib/intel64 -Wl,-rpath=/home/qssun/workspace/program/ctf/build-dbg/hptt/lib
#SO_LIB_FILES = -lmkl_intel_lp64 -lmkl_avx2 -lmkl_gf_thread -lmkl_core -lpthread -lm -Wl,-Bdynamic -lhptt
SO_LIB_FILES = -lmkl_intel_lp64 -lmkl_avx2 -lmkl_core -lpthread -lm -Wl,-Bdynamic -lhptt
LDFLAGS      = 


### COMPILE TIME INCLUDES AND FLAGS
#C++ compiler 
CXX         = mpicxx
#includes for compile time
INCLUDES    =  -I/home/qssun/workspace/program/ctf/build-dbg/hptt/include
#optimization flags, some intel compiler versions may run into errors when using -fast or -ipo
CXXFLAGS    = -O3 -std=c++0x -fopenmp -Wall 
#command to make library out of object files
AR          = ar

#macros to be defined throughout the code, use the below in combination with appropriate external libraries
#Include in DEFS -DUSE_LAPACK to build with LAPACK functionality, 
#                -DUSE_SCALAPACK to build with SCALAPACK functionality
#                -DUSE_MKL to build with MKL batched GEMM and sparse matrix kernels
#                -DUSE_HPTT to build with optimized tensor transposition routines from HPTT library
DEFS        = -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -DUSE_LAPACK -DUSE_HPTT 

### Optional: sparse MKL BLAS Routines
#uncomment below to enable MKL BLAS routines if setting up MKL manually
#DEFS       += -DUSE_MKL=1


### Optional: PROFILING AND TUNING
#uncomment below to enable performance profiling
#DEFS       += -DPROFILE -DPMPI
#uncomment below to enable automatic performance tuning (loses reproducibility of results)
#Note: -DTUNE requires lapack (include -mkl or -llapack in LIBS) and the inclusion of above performance profiling flags
#DEFS       += -DTUNE

### Optional: DEBUGGING AND VERBOSITY
#uncomment below to enable CTF execution output (1 for basic contraction information on start-up and contractions)
#DEFS       += -DVERBOSE=1
#uncomment to set debug level to dump information about mapping and internal CTF actions and activate asserts
#DEFS       += -DDEBUG=1

### FULL COMPILE COMMAND AND LIBRARIES
#used to compile all plain C++ files
FCXX        = $(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES)
#link-line for all executables
LIBS        = $(LIB_PATH) $(LIB_FILES) $(LINKFLAGS)
#compiler for CUDA files (used to compile CUDA code only when -DOFFLOAD and -DUSE_CUDA are in DEFS, otherwise should be same as FCXX with -x c++)
OFFLOAD_CXX = $(CXX) -x c++ $(CXXFLAGS) $(DEFS) $(INCLUDES)
