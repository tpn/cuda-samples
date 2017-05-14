/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


#ifndef __SURFACE_INDIRECT_FUNCTIONS_H__
#define __SURFACE_INDIRECT_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

#include "builtin_types.h"
#include "host_defines.h"
#include "vector_functions.h"

/*******************************************************************************
*                                                                              *
* 1D Surface indirect read functions
*                                                                              *
*******************************************************************************/

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf1Dread(T *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf1Dread(char *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_char");
__device__ __cudart_builtin__ void surf1Dread(signed char *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_schar");
__device__ __cudart_builtin__ void surf1Dread(char1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_char1");
__device__ __cudart_builtin__ void surf1Dread(unsigned char *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uchar");
__device__ __cudart_builtin__ void surf1Dread(uchar1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uchar1");
__device__ __cudart_builtin__ void surf1Dread(short *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_short");
__device__ __cudart_builtin__ void surf1Dread(short1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_short1");
__device__ __cudart_builtin__ void surf1Dread(unsigned short *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ushort");
__device__ __cudart_builtin__ void surf1Dread(ushort1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ushort1");
__device__ __cudart_builtin__ void surf1Dread(int *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_int");
__device__ __cudart_builtin__ void surf1Dread(int1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_int1");
__device__ __cudart_builtin__ void surf1Dread(unsigned int *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uint");
__device__ __cudart_builtin__ void surf1Dread(uint1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uint1");
__device__ __cudart_builtin__ void surf1Dread(long long *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_longlong");
__device__ __cudart_builtin__ void surf1Dread(longlong1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_longlong1");
__device__ __cudart_builtin__ void surf1Dread(unsigned long long *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ulonglong");
__device__ __cudart_builtin__ void surf1Dread(ulonglong1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ulonglong1");
__device__ __cudart_builtin__ void surf1Dread(float *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_float");
__device__ __cudart_builtin__ void surf1Dread(float1 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_float1");

__device__ __cudart_builtin__ void surf1Dread(char2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_char2");
__device__ __cudart_builtin__ void surf1Dread(uchar2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uchar2");
__device__ __cudart_builtin__ void surf1Dread(short2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_short2");
__device__ __cudart_builtin__ void surf1Dread(ushort2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ushort2");
__device__ __cudart_builtin__ void surf1Dread(int2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_int2");
__device__ __cudart_builtin__ void surf1Dread(uint2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uint2");
__device__ __cudart_builtin__ void surf1Dread(longlong2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_longlong2");
__device__ __cudart_builtin__ void surf1Dread(ulonglong2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ulonglong2");
__device__ __cudart_builtin__ void surf1Dread(float2 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_float2");

__device__ __cudart_builtin__ void surf1Dread(char4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_char4");
__device__ __cudart_builtin__ void surf1Dread(uchar4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uchar4");
__device__ __cudart_builtin__ void surf1Dread(short4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_short4");
__device__ __cudart_builtin__ void surf1Dread(ushort4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_ushort4");
__device__ __cudart_builtin__ void surf1Dread(int4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_int4");
__device__ __cudart_builtin__ void surf1Dread(uint4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_uint4");
__device__ __cudart_builtin__ void surf1Dread(float4 *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf1Dread_float4");

#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
   T ret;
   surf1Dread(&ret, surfObject, x, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}


/*******************************************************************************
*                                                                              *
* 2D Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf2Dread(T *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf2Dread(char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_char");
__device__ __cudart_builtin__ void surf2Dread(signed char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_schar");
__device__ __cudart_builtin__ void surf2Dread(char1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_char1");
__device__ __cudart_builtin__ void surf2Dread(unsigned char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uchar");
__device__ __cudart_builtin__ void surf2Dread(uchar1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uchar1");
__device__ __cudart_builtin__ void surf2Dread(short *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_short");
__device__ __cudart_builtin__ void surf2Dread(short1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_short1");
__device__ __cudart_builtin__ void surf2Dread(unsigned short *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ushort");
__device__ __cudart_builtin__ void surf2Dread(ushort1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ushort1");
__device__ __cudart_builtin__ void surf2Dread(int *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_int");
__device__ __cudart_builtin__ void surf2Dread(int1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_int1");
__device__ __cudart_builtin__ void surf2Dread(unsigned int *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uint");
__device__ __cudart_builtin__ void surf2Dread(uint1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uint1");
__device__ __cudart_builtin__ void surf2Dread(long long *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_longlong");
__device__ __cudart_builtin__ void surf2Dread(longlong1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_longlong1");
__device__ __cudart_builtin__ void surf2Dread(unsigned long long *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ulonglong");
__device__ __cudart_builtin__ void surf2Dread(ulonglong1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ulonglong1");
__device__ __cudart_builtin__ void surf2Dread(float *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_float");
__device__ __cudart_builtin__ void surf2Dread(float1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_float1");

__device__ __cudart_builtin__ void surf2Dread(char2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_char2");
__device__ __cudart_builtin__ void surf2Dread(uchar2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uchar2");
__device__ __cudart_builtin__ void surf2Dread(short2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_short2");
__device__ __cudart_builtin__ void surf2Dread(ushort2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ushort2");
__device__ __cudart_builtin__ void surf2Dread(int2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_int2");
__device__ __cudart_builtin__ void surf2Dread(uint2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uint2");

__device__ __cudart_builtin__ void surf2Dread(longlong2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_longlong2");
__device__ __cudart_builtin__ void surf2Dread(ulonglong2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ulonglong2");
__device__ __cudart_builtin__ void surf2Dread(float2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_float2");

__device__ __cudart_builtin__ void surf2Dread(char4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_char4");
__device__ __cudart_builtin__ void surf2Dread(uchar4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uchar4");
__device__ __cudart_builtin__ void surf2Dread(short4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_short4");
__device__ __cudart_builtin__ void surf2Dread(ushort4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_ushort4");
__device__ __cudart_builtin__ void surf2Dread(int4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_int4");
__device__ __cudart_builtin__ void surf2Dread(uint4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_uint4");
__device__ __cudart_builtin__ void surf2Dread(float4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2Dread_float4");

#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf2Dread(&ret, surfObject, x, y, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}


/*******************************************************************************
*                                                                              *
* 3D Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf3Dread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf3Dread(char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_char");
__device__ __cudart_builtin__ void surf3Dread(signed char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_schar");
__device__ __cudart_builtin__ void surf3Dread(char1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_char1");
__device__ __cudart_builtin__ void surf3Dread(unsigned char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uchar");
__device__ __cudart_builtin__ void surf3Dread(uchar1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uchar1");
__device__ __cudart_builtin__ void surf3Dread(short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_short");
__device__ __cudart_builtin__ void surf3Dread(short1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_short1");
__device__ __cudart_builtin__ void surf3Dread(unsigned short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ushort");
__device__ __cudart_builtin__ void surf3Dread(ushort1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ushort1");
__device__ __cudart_builtin__ void surf3Dread(int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_int");
__device__ __cudart_builtin__ void surf3Dread(int1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_int1");
__device__ __cudart_builtin__ void surf3Dread(unsigned int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uint");
__device__ __cudart_builtin__ void surf3Dread(uint1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uint1");
__device__ __cudart_builtin__ void surf3Dread(long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_longlong");
__device__ __cudart_builtin__ void surf3Dread(longlong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_longlong1");
__device__ __cudart_builtin__ void surf3Dread(unsigned long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ulonglong");
__device__ __cudart_builtin__ void surf3Dread(ulonglong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ulonglong1");
__device__ __cudart_builtin__ void surf3Dread(float *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_float");
__device__ __cudart_builtin__ void surf3Dread(float1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_float1");

__device__ __cudart_builtin__ void surf3Dread(char2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_char2");
__device__ __cudart_builtin__ void surf3Dread(uchar2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uchar2");
__device__ __cudart_builtin__ void surf3Dread(short2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_short2");
__device__ __cudart_builtin__ void surf3Dread(ushort2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ushort2");
__device__ __cudart_builtin__ void surf3Dread(int2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_int2");
__device__ __cudart_builtin__ void surf3Dread(uint2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uint2");
__device__ __cudart_builtin__ void surf3Dread(longlong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_longlong2");
__device__ __cudart_builtin__ void surf3Dread(ulonglong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ulonglong2");
__device__ __cudart_builtin__ void surf3Dread(float2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_float2");

__device__ __cudart_builtin__ void surf3Dread(char4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_char4");
__device__ __cudart_builtin__ void surf3Dread(uchar4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uchar4");
__device__ __cudart_builtin__ void surf3Dread(short4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_short4");
__device__ __cudart_builtin__ void surf3Dread(ushort4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_ushort4");
__device__ __cudart_builtin__ void surf3Dread(int4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_int4");
__device__ __cudart_builtin__ void surf3Dread(uint4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_uint4");
__device__ __cudart_builtin__ void surf3Dread(float4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf3Dread_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf3Dread(&ret, surfObject, x, y, z, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf1DLayeredread(T *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf1DLayeredread(char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_char");
__device__ __cudart_builtin__ void surf1DLayeredread(signed char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_schar");
__device__ __cudart_builtin__ void surf1DLayeredread(char1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_char1");
__device__ __cudart_builtin__ void surf1DLayeredread(unsigned char *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uchar");
__device__ __cudart_builtin__ void surf1DLayeredread(uchar1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uchar1");
__device__ __cudart_builtin__ void surf1DLayeredread(short *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_short");
__device__ __cudart_builtin__ void surf1DLayeredread(short1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_short1");
__device__ __cudart_builtin__ void surf1DLayeredread(unsigned short *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ushort");
__device__ __cudart_builtin__ void surf1DLayeredread(ushort1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ushort1");
__device__ __cudart_builtin__ void surf1DLayeredread(int *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_int");
__device__ __cudart_builtin__ void surf1DLayeredread(int1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_int1");
__device__ __cudart_builtin__ void surf1DLayeredread(unsigned int *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uint");
__device__ __cudart_builtin__ void surf1DLayeredread(uint1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uint1");
__device__ __cudart_builtin__ void surf1DLayeredread(long long *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_longlong");
__device__ __cudart_builtin__ void surf1DLayeredread(longlong1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_longlong1");
__device__ __cudart_builtin__ void surf1DLayeredread(unsigned long long *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ulonglong");
__device__ __cudart_builtin__ void surf1DLayeredread(ulonglong1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ulonglong1");
__device__ __cudart_builtin__ void surf1DLayeredread(float *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_float");
__device__ __cudart_builtin__ void surf1DLayeredread(float1 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_float1");

__device__ __cudart_builtin__ void surf1DLayeredread(char2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_char2");
__device__ __cudart_builtin__ void surf1DLayeredread(uchar2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uchar2");
__device__ __cudart_builtin__ void surf1DLayeredread(short2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_short2");
__device__ __cudart_builtin__ void surf1DLayeredread(ushort2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ushort2");
__device__ __cudart_builtin__ void surf1DLayeredread(int2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_int2");
__device__ __cudart_builtin__ void surf1DLayeredread(uint2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uint2");
__device__ __cudart_builtin__ void surf1DLayeredread(longlong2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_longlong2");
__device__ __cudart_builtin__ void surf1DLayeredread(ulonglong2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ulonglong2");
__device__ __cudart_builtin__ void surf1DLayeredread(float2 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_float2");


__device__ __cudart_builtin__ void surf1DLayeredread(char4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_char4");
__device__ __cudart_builtin__ void surf1DLayeredread(uchar4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uchar4");
__device__ __cudart_builtin__ void surf1DLayeredread(short4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_short4");
__device__ __cudart_builtin__ void surf1DLayeredread(ushort4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_ushort4");
__device__ __cudart_builtin__ void surf1DLayeredread(int4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_int4");
__device__ __cudart_builtin__ void surf1DLayeredread(uint4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_uint4");
__device__ __cudart_builtin__ void surf1DLayeredread(float4 *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode= cudaBoundaryModeTrap) asm("__isurf1DLayeredread_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf2DLayeredread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf2DLayeredread(char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_char");
__device__ __cudart_builtin__ void surf2DLayeredread(signed char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_schar");
__device__ __cudart_builtin__ void surf2DLayeredread(char1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_char1");
__device__ __cudart_builtin__ void surf2DLayeredread(unsigned char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uchar");
__device__ __cudart_builtin__ void surf2DLayeredread(uchar1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uchar1");
__device__ __cudart_builtin__ void surf2DLayeredread(short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_short");
__device__ __cudart_builtin__ void surf2DLayeredread(short1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_short1");
__device__ __cudart_builtin__ void surf2DLayeredread(unsigned short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ushort");
__device__ __cudart_builtin__ void surf2DLayeredread(ushort1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ushort1");
__device__ __cudart_builtin__ void surf2DLayeredread(int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_int");
__device__ __cudart_builtin__ void surf2DLayeredread(int1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_int1");
__device__ __cudart_builtin__ void surf2DLayeredread(unsigned int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uint");
__device__ __cudart_builtin__ void surf2DLayeredread(uint1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uint1");
__device__ __cudart_builtin__ void surf2DLayeredread(long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_longlong");
__device__ __cudart_builtin__ void surf2DLayeredread(longlong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_longlong1");
__device__ __cudart_builtin__ void surf2DLayeredread(unsigned long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ulonglong");
__device__ __cudart_builtin__ void surf2DLayeredread(ulonglong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ulonglong1");
__device__ __cudart_builtin__ void surf2DLayeredread(float *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_float");
__device__ __cudart_builtin__ void surf2DLayeredread(float1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_float1");

__device__ __cudart_builtin__ void surf2DLayeredread(char2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_char2");
__device__ __cudart_builtin__ void surf2DLayeredread(uchar2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uchar2");
__device__ __cudart_builtin__ void surf2DLayeredread(short2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_short2");
__device__ __cudart_builtin__ void surf2DLayeredread(ushort2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ushort2");
__device__ __cudart_builtin__ void surf2DLayeredread(int2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_int2");
__device__ __cudart_builtin__ void surf2DLayeredread(uint2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uint2");
__device__ __cudart_builtin__ void surf2DLayeredread(longlong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_longlong2");
__device__ __cudart_builtin__ void surf2DLayeredread(ulonglong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ulonglong2");
__device__ __cudart_builtin__ void surf2DLayeredread(float2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_float2");

__device__ __cudart_builtin__ void surf2DLayeredread(char4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_char4");
__device__ __cudart_builtin__ void surf2DLayeredread(uchar4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uchar4");
__device__ __cudart_builtin__ void surf2DLayeredread(short4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_short4");
__device__ __cudart_builtin__ void surf2DLayeredread(ushort4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_ushort4");
__device__ __cudart_builtin__ void surf2DLayeredread(int4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_int4");
__device__ __cudart_builtin__ void surf2DLayeredread(uint4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_uint4");
__device__ __cudart_builtin__ void surf2DLayeredread(float4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredread_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surfCubemapread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surfCubemapread(char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_char");
__device__ __cudart_builtin__ void surfCubemapread(signed char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_schar");
__device__ __cudart_builtin__ void surfCubemapread(char1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_char1");
__device__ __cudart_builtin__ void surfCubemapread(unsigned char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uchar");
__device__ __cudart_builtin__ void surfCubemapread(uchar1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uchar1");
__device__ __cudart_builtin__ void surfCubemapread(short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_short");
__device__ __cudart_builtin__ void surfCubemapread(short1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_short1");
__device__ __cudart_builtin__ void surfCubemapread(unsigned short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ushort");
__device__ __cudart_builtin__ void surfCubemapread(ushort1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ushort1");
__device__ __cudart_builtin__ void surfCubemapread(int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_int");
__device__ __cudart_builtin__ void surfCubemapread(int1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_int1");
__device__ __cudart_builtin__ void surfCubemapread(unsigned int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uint");
__device__ __cudart_builtin__ void surfCubemapread(uint1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uint1");
__device__ __cudart_builtin__ void surfCubemapread(long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_longlong");
__device__ __cudart_builtin__ void surfCubemapread(longlong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_longlong1");
__device__ __cudart_builtin__ void surfCubemapread(unsigned long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ulonglong");
__device__ __cudart_builtin__ void surfCubemapread(ulonglong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ulonglong1");
__device__ __cudart_builtin__ void surfCubemapread(float *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_float");
__device__ __cudart_builtin__ void surfCubemapread(float1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_float1");

__device__ __cudart_builtin__ void surfCubemapread(char2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_char2");
__device__ __cudart_builtin__ void surfCubemapread(uchar2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uchar2");
__device__ __cudart_builtin__ void surfCubemapread(short2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_short2");
__device__ __cudart_builtin__ void surfCubemapread(ushort2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ushort2");
__device__ __cudart_builtin__ void surfCubemapread(int2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_int2");
__device__ __cudart_builtin__ void surfCubemapread(uint2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uint2");
__device__ __cudart_builtin__ void surfCubemapread(longlong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_longlong2");
__device__ __cudart_builtin__ void surfCubemapread(ulonglong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ulonglong2");
__device__ __cudart_builtin__ void surfCubemapread(float2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_float2");

__device__ __cudart_builtin__ void surfCubemapread(char4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_char4");
__device__ __cudart_builtin__ void surfCubemapread(uchar4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uchar4");
__device__ __cudart_builtin__ void surfCubemapread(short4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_short4");
__device__ __cudart_builtin__ void surfCubemapread(ushort4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_ushort4");
__device__ __cudart_builtin__ void surfCubemapread(int4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_int4");
__device__ __cudart_builtin__ void surfCubemapread(uint4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_uint4");
__device__ __cudart_builtin__ void surfCubemapread(float4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapread_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surfCubemapread(&ret, surfObject, face, x, y, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surfCubemapLayeredread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surfCubemapLayeredread(char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_char");
__device__ __cudart_builtin__ void surfCubemapLayeredread(signed char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_schar");
__device__ __cudart_builtin__ void surfCubemapLayeredread(char1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_char1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(unsigned char *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uchar");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uchar1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uchar1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_short");
__device__ __cudart_builtin__ void surfCubemapLayeredread(short1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_short1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(unsigned short *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ushort");
__device__ __cudart_builtin__ void surfCubemapLayeredread(ushort1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ushort1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_int");
__device__ __cudart_builtin__ void surfCubemapLayeredread(int1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_int1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(unsigned int *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uint");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uint1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uint1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_longlong");
__device__ __cudart_builtin__ void surfCubemapLayeredread(longlong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_longlong1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(unsigned long long *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ulonglong");
__device__ __cudart_builtin__ void surfCubemapLayeredread(ulonglong1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ulonglong1");
__device__ __cudart_builtin__ void surfCubemapLayeredread(float *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_float");
__device__ __cudart_builtin__ void surfCubemapLayeredread(float1 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_float1");

__device__ __cudart_builtin__ void surfCubemapLayeredread(char2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_char2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uchar2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uchar2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(short2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_short2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(ushort2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ushort2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(int2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_int2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uint2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uint2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(longlong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_longlong2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(ulonglong2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ulonglong2");
__device__ __cudart_builtin__ void surfCubemapLayeredread(float2 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_float2");

__device__ __cudart_builtin__ void surfCubemapLayeredread(char4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_char4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uchar4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uchar4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(short4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_short4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(ushort4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_ushort4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(int4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_int4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(uint4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_uint4");
__device__ __cudart_builtin__ void surfCubemapLayeredread(float4 *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurfCubemapLayeredread_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

/*******************************************************************************
*                                                                              *
* 1D Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf1Dwrite(T, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf1Dwrite(char, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_char");
__device__ __cudart_builtin__ void surf1Dwrite(signed char, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_schar");
__device__ __cudart_builtin__ void surf1Dwrite(char1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_char1");
__device__ __cudart_builtin__ void surf1Dwrite(unsigned char, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uchar");
__device__ __cudart_builtin__ void surf1Dwrite(uchar1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uchar1");
__device__ __cudart_builtin__ void surf1Dwrite(short, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_short");
__device__ __cudart_builtin__ void surf1Dwrite(short1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_short1");
__device__ __cudart_builtin__ void surf1Dwrite(unsigned short, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ushort");
__device__ __cudart_builtin__ void surf1Dwrite(ushort1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ushort1");
__device__ __cudart_builtin__ void surf1Dwrite(int, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_int");
__device__ __cudart_builtin__ void surf1Dwrite(int1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_int1");
__device__ __cudart_builtin__ void surf1Dwrite(unsigned int, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uint");
__device__ __cudart_builtin__ void surf1Dwrite(uint1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uint1");
__device__ __cudart_builtin__ void surf1Dwrite(long long, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_longlong");
__device__ __cudart_builtin__ void surf1Dwrite(longlong1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_longlong1");
__device__ __cudart_builtin__ void surf1Dwrite(unsigned long long, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ulonglong");
__device__ __cudart_builtin__ void surf1Dwrite(ulonglong1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ulonglong1");
__device__ __cudart_builtin__ void surf1Dwrite(float, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_float");
__device__ __cudart_builtin__ void surf1Dwrite(float1, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_float1");

__device__ __cudart_builtin__ void surf1Dwrite(char2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_char2");
__device__ __cudart_builtin__ void surf1Dwrite(uchar2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uchar2");
__device__ __cudart_builtin__ void surf1Dwrite(short2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_short2");
__device__ __cudart_builtin__ void surf1Dwrite(ushort2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ushort2");
__device__ __cudart_builtin__ void surf1Dwrite(int2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_int2");
__device__ __cudart_builtin__ void surf1Dwrite(uint2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uint2");
__device__ __cudart_builtin__ void surf1Dwrite(longlong2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_longlong2");
__device__ __cudart_builtin__ void surf1Dwrite(ulonglong2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ulonglong2");
__device__ __cudart_builtin__ void surf1Dwrite(float2, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_float2");

__device__ __cudart_builtin__ void surf1Dwrite(char4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_char4");
__device__ __cudart_builtin__ void surf1Dwrite(uchar4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uchar4");
__device__ __cudart_builtin__ void surf1Dwrite(short4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_short4");
__device__ __cudart_builtin__ void surf1Dwrite(ushort4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_ushort4");
__device__ __cudart_builtin__ void surf1Dwrite(int4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_int4");
__device__ __cudart_builtin__ void surf1Dwrite(uint4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_uint4");
__device__ __cudart_builtin__ void surf1Dwrite(float4, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1Dwrite_float4");
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
* 2D Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf2Dwrite(T, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf2Dwrite(char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_char");
__device__ __cudart_builtin__ void surf2Dwrite(signed char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_schar");
__device__ __cudart_builtin__ void surf2Dwrite(char1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_char1");
__device__ __cudart_builtin__ void surf2Dwrite(unsigned char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uchar");
__device__ __cudart_builtin__ void surf2Dwrite(uchar1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uchar1");
__device__ __cudart_builtin__ void surf2Dwrite(short, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_short");
__device__ __cudart_builtin__ void surf2Dwrite(short1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_short1");
__device__ __cudart_builtin__ void surf2Dwrite(unsigned short, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ushort");
__device__ __cudart_builtin__ void surf2Dwrite(ushort1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ushort1");
__device__ __cudart_builtin__ void surf2Dwrite(int, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_int");
__device__ __cudart_builtin__ void surf2Dwrite(int1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_int1");
__device__ __cudart_builtin__ void surf2Dwrite(unsigned int, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uint");
__device__ __cudart_builtin__ void surf2Dwrite(uint1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uint1");
__device__ __cudart_builtin__ void surf2Dwrite(long long, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_longlong");
__device__ __cudart_builtin__ void surf2Dwrite(longlong1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_longlong1");
__device__ __cudart_builtin__ void surf2Dwrite(unsigned long long, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ulonglong");
__device__ __cudart_builtin__ void surf2Dwrite(ulonglong1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ulonglong1");
__device__ __cudart_builtin__ void surf2Dwrite(float, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_float");
__device__ __cudart_builtin__ void surf2Dwrite(float1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_float1");

__device__ __cudart_builtin__ void surf2Dwrite(char2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_char2");
__device__ __cudart_builtin__ void surf2Dwrite(uchar2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uchar2");
__device__ __cudart_builtin__ void surf2Dwrite(short2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_short2");
__device__ __cudart_builtin__ void surf2Dwrite(ushort2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ushort2");
__device__ __cudart_builtin__ void surf2Dwrite(int2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_int2");
__device__ __cudart_builtin__ void surf2Dwrite(uint2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uint2");
__device__ __cudart_builtin__ void surf2Dwrite(longlong2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_longlong2");
__device__ __cudart_builtin__ void surf2Dwrite(ulonglong2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ulonglong2");
__device__ __cudart_builtin__ void surf2Dwrite(float2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_float2");

__device__ __cudart_builtin__ void surf2Dwrite(char4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_char4");
__device__ __cudart_builtin__ void surf2Dwrite(uchar4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uchar4");
__device__ __cudart_builtin__ void surf2Dwrite(short4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_short4");
__device__ __cudart_builtin__ void surf2Dwrite(ushort4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_ushort4");
__device__ __cudart_builtin__ void surf2Dwrite(int4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_int4");
__device__ __cudart_builtin__ void surf2Dwrite(uint4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_uint4");
__device__ __cudart_builtin__ void surf2Dwrite(float4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf2Dwrite_float4");
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
* 3D Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf3Dwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf3Dwrite(char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_char");
__device__ __cudart_builtin__ void surf3Dwrite(signed char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_schar");
__device__ __cudart_builtin__ void surf3Dwrite(char1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_char1");
__device__ __cudart_builtin__ void surf3Dwrite(unsigned char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uchar");
__device__ __cudart_builtin__ void surf3Dwrite(uchar1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uchar1");
__device__ __cudart_builtin__ void surf3Dwrite(short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_short");
__device__ __cudart_builtin__ void surf3Dwrite(short1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_short1");
__device__ __cudart_builtin__ void surf3Dwrite(unsigned short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ushort");
__device__ __cudart_builtin__ void surf3Dwrite(ushort1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ushort1");
__device__ __cudart_builtin__ void surf3Dwrite(int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_int");
__device__ __cudart_builtin__ void surf3Dwrite(int1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_int1");
__device__ __cudart_builtin__ void surf3Dwrite(unsigned int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uint");
__device__ __cudart_builtin__ void surf3Dwrite(uint1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uint1");
__device__ __cudart_builtin__ void surf3Dwrite(long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_longlong");
__device__ __cudart_builtin__ void surf3Dwrite(longlong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_longlong1");
__device__ __cudart_builtin__ void surf3Dwrite(unsigned long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ulonglong");
__device__ __cudart_builtin__ void surf3Dwrite(ulonglong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ulonglong1");
__device__ __cudart_builtin__ void surf3Dwrite(float, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_float");
__device__ __cudart_builtin__ void surf3Dwrite(float1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_float1");


__device__ __cudart_builtin__ void surf3Dwrite(char2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_char2");
__device__ __cudart_builtin__ void surf3Dwrite(uchar2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uchar2");
__device__ __cudart_builtin__ void surf3Dwrite(short2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_short2");
__device__ __cudart_builtin__ void surf3Dwrite(ushort2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ushort2");
__device__ __cudart_builtin__ void surf3Dwrite(int2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_int2");
__device__ __cudart_builtin__ void surf3Dwrite(uint2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uint2");
__device__ __cudart_builtin__ void surf3Dwrite(longlong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_longlong2");
__device__ __cudart_builtin__ void surf3Dwrite(ulonglong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ulonglong2");
__device__ __cudart_builtin__ void surf3Dwrite(float2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_float2");

__device__ __cudart_builtin__ void surf3Dwrite(char4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_char4");
__device__ __cudart_builtin__ void surf3Dwrite(uchar4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uchar4");
__device__ __cudart_builtin__ void surf3Dwrite(short4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_short4");
__device__ __cudart_builtin__ void surf3Dwrite(ushort4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_ushort4");
__device__ __cudart_builtin__ void surf3Dwrite(int4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_int4");
__device__ __cudart_builtin__ void surf3Dwrite(uint4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_uint4");
__device__ __cudart_builtin__ void surf3Dwrite(float4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf3Dwrite_float4");
#endif /* __CUDA_ARCH__ */


/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf1DLayeredwrite(T, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */

__device__ __cudart_builtin__ void surf1DLayeredwrite(char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_char");
__device__ __cudart_builtin__ void surf1DLayeredwrite(signed char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_schar");
__device__ __cudart_builtin__ void surf1DLayeredwrite(char1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_char1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned char, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uchar");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uchar1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(short, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_short");
__device__ __cudart_builtin__ void surf1DLayeredwrite(short1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_short1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned short, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ushort");
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ushort1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(int, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_int");
__device__ __cudart_builtin__ void surf1DLayeredwrite(int1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_int1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned int, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uint");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uint1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(long long, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_longlong");
__device__ __cudart_builtin__ void surf1DLayeredwrite(longlong1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_longlong1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned long long, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ulonglong");
__device__ __cudart_builtin__ void surf1DLayeredwrite(ulonglong1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ulonglong1");
__device__ __cudart_builtin__ void surf1DLayeredwrite(float, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_float");
__device__ __cudart_builtin__ void surf1DLayeredwrite(float1, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_float1");

__device__ __cudart_builtin__ void surf1DLayeredwrite(char2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_char2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uchar2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(short2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_short2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ushort2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(int2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_int2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uint2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(longlong2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_longlong2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(ulonglong2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ulonglong2");
__device__ __cudart_builtin__ void surf1DLayeredwrite(float2, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_float2");

__device__ __cudart_builtin__ void surf1DLayeredwrite(char4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_char4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uchar4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(short4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_short4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_ushort4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(int4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_int4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_uint4");
__device__ __cudart_builtin__ void surf1DLayeredwrite(float4, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurf1DLayeredwrite_float4");
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surf2DLayeredwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surf2DLayeredwrite(char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_char");
__device__ __cudart_builtin__ void surf2DLayeredwrite(signed char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_schar");
__device__ __cudart_builtin__ void surf2DLayeredwrite(char1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_char1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uchar");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uchar1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_short");
__device__ __cudart_builtin__ void surf2DLayeredwrite(short1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_short1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ushort");
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ushort1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_int");
__device__ __cudart_builtin__ void surf2DLayeredwrite(int1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_int1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uint");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uint1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_longlong");
__device__ __cudart_builtin__ void surf2DLayeredwrite(longlong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_longlong1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ulonglong");
__device__ __cudart_builtin__ void surf2DLayeredwrite(ulonglong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ulonglong1");
__device__ __cudart_builtin__ void surf2DLayeredwrite(float, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_float");
__device__ __cudart_builtin__ void surf2DLayeredwrite(float1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_float1");

__device__ __cudart_builtin__ void surf2DLayeredwrite(char2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_char2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uchar2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(short2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_short2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ushort2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(int2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_int2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uint2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(longlong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_longlong2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(ulonglong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ulonglong2");
__device__ __cudart_builtin__ void surf2DLayeredwrite(float2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_float2");

__device__ __cudart_builtin__ void surf2DLayeredwrite(char4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_char4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uchar4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(short4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_short4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_ushort4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(int4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_int4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_uint4");
__device__ __cudart_builtin__ void surf2DLayeredwrite(float4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) asm("__isurf2DLayeredwrite_float4");
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surfCubemapwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surfCubemapwrite(char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_char");
__device__ __cudart_builtin__ void surfCubemapwrite(signed char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_schar");
__device__ __cudart_builtin__ void surfCubemapwrite(char1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_char1");
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uchar");
__device__ __cudart_builtin__ void surfCubemapwrite(uchar1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uchar1");
__device__ __cudart_builtin__ void surfCubemapwrite(short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_short");
__device__ __cudart_builtin__ void surfCubemapwrite(short1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_short1");
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ushort");
__device__ __cudart_builtin__ void surfCubemapwrite(ushort1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ushort1");
__device__ __cudart_builtin__ void surfCubemapwrite(int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_int");
__device__ __cudart_builtin__ void surfCubemapwrite(int1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_int1");
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uint");
__device__ __cudart_builtin__ void surfCubemapwrite(uint1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uint1");
__device__ __cudart_builtin__ void surfCubemapwrite(long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_longlong");
__device__ __cudart_builtin__ void surfCubemapwrite(longlong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_longlong1");
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ulonglong");
__device__ __cudart_builtin__ void surfCubemapwrite(ulonglong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ulonglong1");
__device__ __cudart_builtin__ void surfCubemapwrite(float, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_float");
__device__ __cudart_builtin__ void surfCubemapwrite(float1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_float1");

__device__ __cudart_builtin__ void surfCubemapwrite(char2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_char2");
__device__ __cudart_builtin__ void surfCubemapwrite(uchar2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uchar2");
__device__ __cudart_builtin__ void surfCubemapwrite(short2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_short2");
__device__ __cudart_builtin__ void surfCubemapwrite(ushort2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ushort2");
__device__ __cudart_builtin__ void surfCubemapwrite(int2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_int2");
__device__ __cudart_builtin__ void surfCubemapwrite(uint2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uint2");
__device__ __cudart_builtin__ void surfCubemapwrite(longlong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_longlong2");
__device__ __cudart_builtin__ void surfCubemapwrite(ulonglong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ulonglong2");
__device__ __cudart_builtin__ void surfCubemapwrite(float2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_float2");

__device__ __cudart_builtin__ void surfCubemapwrite(char4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_char4");
__device__ __cudart_builtin__ void surfCubemapwrite(uchar4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uchar4");
__device__ __cudart_builtin__ void surfCubemapwrite(short4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_short4");
__device__ __cudart_builtin__ void surfCubemapwrite(ushort4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_ushort4");
__device__ __cudart_builtin__ void surfCubemapwrite(int4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_int4");
__device__ __cudart_builtin__ void surfCubemapwrite(uint4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_uint4");
__device__ __cudart_builtin__ void surfCubemapwrite(float4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapwrite_float4");
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void surfCubemapLayeredwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) { }
#else /* __CUDA_ARCH__ */
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_char");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(signed char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_schar");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_char1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned char, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uchar");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uchar1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_short");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_short1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned short, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ushort");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ushort1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_int");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_int1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned int, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uint");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uint1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_longlong");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(longlong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_longlong1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned long long, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ulonglong");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ulonglong1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ulonglong1");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_float");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float1, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_float1");

__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_char2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uchar2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_short2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ushort2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_int2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uint2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(longlong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_longlong2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ulonglong2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ulonglong2");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float2, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_float2");

__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_char4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uchar4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_short4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_ushort4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_int4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_uint4");
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float4, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)  asm("__isurfCubemapLayeredwrite_float4");
#endif /* __CUDA_ARCH__ */

#endif // __cplusplus && __CUDACC__

#endif // __SURFACE_INDIRECT_FUNCTIONS_H__


