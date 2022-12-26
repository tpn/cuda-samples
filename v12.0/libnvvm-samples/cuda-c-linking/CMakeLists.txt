################################################################################
#
# Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################

add_executable(cuda-c-linking cuda-c-linking.cpp)
target_link_libraries(cuda-c-linking ${NVVM_LIB} ${CUDA_LIB})

# See https://llvm.org/docs/CMake.html#developing-llvm-passes-out-of-source
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(cuda-c-linking ${llvm_libs})

if (UNIX)
  set_target_properties(cuda-c-linking PROPERTIES
                        LINK_FLAGS "-Wl,-rpath,${LIBNVVM_RPATH}")
endif()

##############################
### Math Lib
##############################

### The math library can be built to support a number of GPU platforms using the 
### common compute_60 architecture.
### For a collection of specific architecutres, the compiler options
### "-gencode=compute_XX,code=sm_XX"... can be used multiple times in the same
### nvcc invocation.
### The result is bundled into a system library using the 'nvcc -lib' feature.
### >> nvcc -m64 -arch=compute_60 -dc math-funcs.cu -o math-funcs64.o
### >> nvcc -m64 -lib math-funcs64.o -o libmathfuncs64.a

enable_language(CUDA)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18.0) 
    set(CMAKE_CUDA_ARCHITECTURES 60)
endif ()
add_library(mathfuncs64 STATIC math-funcs.cu)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc") # -dc: Build for device only. 
set_target_properties(mathfuncs64 PROPERTIES PREFIX "lib" OUTPUT_NAME "mathfuncs64" SUFFIX ".a" CUDA_SEPERABLE_COMPILATION ON)
install(TARGETS cuda-c-linking mathfuncs64 DESTINATION bin)
