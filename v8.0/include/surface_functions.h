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

#if !defined(__SURFACE_FUNCTIONS_H__)
#define __SURFACE_FUNCTIONS_H__


#ifdef __CUDACC_INTEGRATED__
#define __ASM_IF_INTEGRATED(X) asm(X)
#else /* !__CUDACC_INTEGRATED__ */
#define __ASM_IF_INTEGRATED(X)
#endif  /* __CUDACC_INTEGRATED__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "cuda_surface_types.h"
#include "host_defines.h"
#include "surface_types.h"
#include "vector_functions.h"
#include "vector_types.h"


#if defined(__CUDA_ARCH__)
/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern __device__ __device_builtin__ uchar1     __surf1Dreadc1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadc1");
extern __device__ __device_builtin__ uchar2     __surf1Dreadc2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadc2");
extern __device__ __device_builtin__ uchar4     __surf1Dreadc4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadc4");
extern __device__ __device_builtin__ ushort1    __surf1Dreads1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreads1");
extern __device__ __device_builtin__ ushort2    __surf1Dreads2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreads2");
extern __device__ __device_builtin__ ushort4    __surf1Dreads4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreads4");
extern __device__ __device_builtin__ uint1      __surf1Dreadu1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadu1");
extern __device__ __device_builtin__ uint2      __surf1Dreadu2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadu2");
extern __device__ __device_builtin__ uint4      __surf1Dreadu4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadu4");
extern __device__ __device_builtin__ ulonglong1 __surf1Dreadl1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadl1");
extern __device__ __device_builtin__ ulonglong2 __surf1Dreadl2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dreadl2");

#define __surfModeSwitch(surf, x, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf1Dread##type(surf, x, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1Dread##type(surf, x, cudaBoundaryModeClamp) : \
                                           __surf1Dread##type(surf, x, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */
}

template<class T>
__device__ __forceinline__  T surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)    
  T tmp;
  
  surf1Dread(&tmp, surf, x, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  *res = surf1Dread<T>(surf, x, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_char") ;
template<> __device__ __cudart_builtin__ signed char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uchar4") ;
template<> __device__ __cudart_builtin__ short surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ushort4") ;
template<> __device__ __cudart_builtin__ int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uint") ;
template<> __device__ __cudart_builtin__ int1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_ulonglong2") ;
#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surf1Dread_long_as_int(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_long_as_int");
  return (long int)__surf1Dread_long_as_int(surf, x, mode);
}

template<>
__device__ __forceinline__ unsigned long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned __surf1Dread_ulong_as_uint(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_ulong_as_uint");
  return (long unsigned)__surf1Dread_ulong_as_uint(surf, x, mode);
}

template<>
__device__ __forceinline__ long1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surf1Dread_long1_as_int1(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_long1_as_int1");
  int1 v = __surf1Dread_long1_as_int1(surf, x, mode);
  return make_long1((long int)v.x);
}

template<>
__device__ __forceinline__ ulong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surf1Dread_ulong1_as_uint1(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_ulong1_as_uint1");
  uint1 v = __surf1Dread_ulong1_as_uint1(surf, x, mode);
  return make_ulong1((unsigned long int)v.x);
}

template<>
__device__ __forceinline__ long2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surf1Dread_long2_as_int2(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_long2_as_int2");
  int2 tmp = __surf1Dread_long2_as_int2(surf, x, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surf1Dread_ulong2_as_uint2(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_ulong2_as_uint2");
  uint2 tmp = __surf1Dread_ulong2_as_uint2(surf, x, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surf1Dread_long4_as_int4(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_long4_as_int4");
  int4 tmp = __surf1Dread_long4_as_int4(surf, x, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surf1Dread_ulong4_as_uint4(surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dread_ulong4_as_uint4");
  uint4 tmp = __surf1Dread_ulong4_as_uint4(surf, x, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__ */
template<> __device__ __cudart_builtin__ float surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_float") ;
template<> __device__ __cudart_builtin__ float1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_float1") ;
template<> __device__ __cudart_builtin__ float2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_float2") ;
template<> __device__ __cudart_builtin__ float4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ uchar1     __surf2Dreadc1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadc1") ;
extern __device__ __device_builtin__ uchar2     __surf2Dreadc2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadc2") ;
extern __device__ __device_builtin__ uchar4     __surf2Dreadc4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadc4") ;
extern __device__ __device_builtin__ ushort1    __surf2Dreads1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreads1") ;
extern __device__ __device_builtin__ ushort2    __surf2Dreads2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreads2") ;
extern __device__ __device_builtin__ ushort4    __surf2Dreads4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreads4") ;
extern __device__ __device_builtin__ uint1      __surf2Dreadu1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadu1") ;
extern __device__ __device_builtin__ uint2      __surf2Dreadu2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadu2") ;
extern __device__ __device_builtin__ uint4      __surf2Dreadu4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadu4") ;
extern __device__ __device_builtin__ ulonglong1 __surf2Dreadl1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadl1") ;
extern __device__ __device_builtin__ ulonglong2 __surf2Dreadl2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dreadl2");

#define __surfModeSwitch(surf, x, y, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2Dread##type(surf, x, y, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2Dread##type(surf, x, y, cudaBoundaryModeClamp) : \
                                           __surf2Dread##type(surf, x, y, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  T tmp;
  
  surf2Dread(&tmp, surf, x, y, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surf2Dread<T>(surf, x, y, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_char") ;
template<> __device__ __cudart_builtin__ signed char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uchar4") ;
template<> __device__ __cudart_builtin__ short surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ushort4") ;
template<> __device__ __cudart_builtin__ int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uint") ;
template<> __device__ __cudart_builtin__ int1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ulonglong1") ;
template<>__device__ __cudart_builtin__ longlong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_ulonglong2") ;

#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surf1Dread_long_as_int(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_long_as_int");
  return __surf1Dread_long_as_int(surf, x, y, mode);
}

template<>
__device__ __forceinline__ unsigned long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surf1Dread_ulong_as_uint(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_ulong_as_uint");
  return __surf1Dread_ulong_as_uint(surf, x, y, mode);
}

template<>
__device__ __forceinline__ long1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surf2Dread_long1_as_int1(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_long1_as_int1");
  int1 tmp = __surf2Dread_long1_as_int1(surf, x, y, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surf2Dread_ulong1_as_uint1(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_ulong1_as_uint1");
  uint1 tmp = __surf2Dread_ulong1_as_uint1(surf, x, y, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surf2Dread_long2_as_int2(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_long2_as_int2");
  int2 tmp = __surf2Dread_long2_as_int2(surf, x, y, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surf2Dread_ulong2_as_uint2(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_ulong2_as_uint2");
  uint2 tmp = __surf2Dread_ulong2_as_uint2(surf, x, y, mode);
  return make_ulong2((long unsigned int)tmp.x, (long unsigned int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surf2Dread_long4_as_int4(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_long4_as_int4");
  int4 tmp = __surf2Dread_long4_as_int4(surf, x, y, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surf2Dread_ulong4_as_uint4(surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dread_ulong4_as_uint4");
  uint4 tmp = __surf2Dread_ulong4_as_uint4(surf, x, y, mode);
  return make_ulong4((unsigned long)tmp.x, (unsigned long)tmp.y, (unsigned long)tmp.z, (unsigned long)tmp.w);
}
#endif /* !__LP64__  */

template<> __device__ __cudart_builtin__ float surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_float") ;
template<> __device__ __cudart_builtin__ float1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_float1") ;
template<> __device__ __cudart_builtin__ float2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_float2") ;
template<> __device__ __cudart_builtin__ float4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ uchar1     __surf3Dreadc1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadc1") ;
extern __device__ __device_builtin__ uchar2     __surf3Dreadc2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadc2") ;
extern __device__ __device_builtin__ uchar4     __surf3Dreadc4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadc4") ;
extern __device__ __device_builtin__ ushort1    __surf3Dreads1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreads1") ;
extern __device__ __device_builtin__ ushort2    __surf3Dreads2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreads2") ;
extern __device__ __device_builtin__ ushort4    __surf3Dreads4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreads4") ;
extern __device__ __device_builtin__ uint1      __surf3Dreadu1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadu1") ;
extern __device__ __device_builtin__ uint2      __surf3Dreadu2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadu2") ;
extern __device__ __device_builtin__ uint4      __surf3Dreadu4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadu4") ;
extern __device__ __device_builtin__ ulonglong1 __surf3Dreadl1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadl1") ;
extern __device__ __device_builtin__ ulonglong2 __surf3Dreadl2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dreadl2") ;

#define __surfModeSwitch(surf, x, y, z, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf3Dread##type(surf, x, y, z, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf3Dread##type(surf, x, y, z, cudaBoundaryModeClamp) : \
                                           __surf3Dread##type(surf, x, y, z, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, z, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, z, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, z, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, z, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, z, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  T tmp;
  
  surf3Dread(&tmp, surf, x, y, z, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */
}

template<class T>
__device__ __forceinline__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surf3Dread<T>(surf, x, y, z, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_char") ;
template<> __device__ __cudart_builtin__ signed char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uchar4") ;
template<> __device__ __cudart_builtin__ short surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ushort4") ;
template<> __device__ __cudart_builtin__ int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uint") ;
template<> __device__ __cudart_builtin__ int1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_ulonglong2") ;
#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surf3Dread_long_as_int(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_long_as_int");
  return __surf3Dread_long_as_int(surf, x, y, z, mode);
}

template<>
__device__ __forceinline__ unsigned long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surf3Dread_ulong_as_uint(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_ulong_as_uint");
  return __surf3Dread_ulong_as_uint(surf, x, y, z, mode);
}

template<>
__device__ __forceinline__ long1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surf3Dread_long1_as_int1(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_long1_as_int1");
  int1 tmp = __surf3Dread_long1_as_int1(surf, x, y, z, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surf3Dread_ulong1_as_uint1(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_ulong1_as_uint1");
  uint1 tmp = __surf3Dread_ulong1_as_uint1(surf, x, y, z, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surf3Dread_long2_as_int2(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_long2_as_int2");
  int2 tmp = __surf3Dread_long2_as_int2(surf, x, y, z, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surf3Dread_ulong2_as_uint2(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_ulong2_as_uint2");
  uint2 tmp = __surf3Dread_ulong2_as_uint2(surf, x, y, z, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surf3Dread_long4_as_int4(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_long4_as_int4");
  int4 tmp = __surf3Dread_long4_as_int4(surf, x, y, z, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surf3Dread_ulong4_as_uint4(surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dread_ulong4_as_uint4");
  uint4 tmp = __surf3Dread_ulong4_as_uint4(surf, x, y, z, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__ */

template<> __device__ __cudart_builtin__ float surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_float") ;
template<> __device__ __cudart_builtin__ float1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_float1") ;
template<> __device__ __cudart_builtin__ float2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_float2") ;
template<> __device__ __cudart_builtin__ float4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dread_float4") ;
#endif /* __CUDA_ARCH__ */
#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ uchar1     __surf1DLayeredreadc1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadc1");
extern __device__ __device_builtin__ uchar2     __surf1DLayeredreadc2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadc2");
extern __device__ __device_builtin__ uchar4     __surf1DLayeredreadc4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadc4");
extern __device__ __device_builtin__ ushort1    __surf1DLayeredreads1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreads1");
extern __device__ __device_builtin__ ushort2    __surf1DLayeredreads2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreads2");
extern __device__ __device_builtin__ ushort4    __surf1DLayeredreads4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreads4");
extern __device__ __device_builtin__ uint1      __surf1DLayeredreadu1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadu1");
extern __device__ __device_builtin__ uint2      __surf1DLayeredreadu2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadu2");
extern __device__ __device_builtin__ uint4      __surf1DLayeredreadu4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadu4");
extern __device__ __device_builtin__ ulonglong1 __surf1DLayeredreadl1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadl1");
extern __device__ __device_builtin__ ulonglong2 __surf1DLayeredreadl2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredreadl2");

#define __surfModeSwitch(surf, x, layer, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeClamp) : \
                                           __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, layer, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, layer, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, layer, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, layer, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, layer, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  T tmp;
  
  surf1DLayeredread(&tmp, surf, x, layer, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surf1DLayeredread<T>(surf, x, layer, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_char") ;
template<> __device__ __cudart_builtin__ signed char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uchar4") ;
template<> __device__ __cudart_builtin__ short surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ushort4") ;
template<> __device__ __cudart_builtin__ int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uint") ;
template<> __device__ __cudart_builtin__ int1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_ulonglong2") ;

#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surf1DLayeredread_long_as_int(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_long_as_int");
  return (long int)__surf1DLayeredread_long_as_int(surf, x, layer, mode);
}

template<>
__device__ __forceinline__ unsigned long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surf1DLayeredread_ulong_as_uint(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_ulong_as_uint");
  return (unsigned long int)__surf1DLayeredread_ulong_as_uint(surf, x, layer, mode);
}

template<>
__device__ __forceinline__ long1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surf1DLayeredread_long1_as_int1(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_long1_as_int1");
  int1 tmp = __surf1DLayeredread_long1_as_int1(surf, x, layer, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surf1DLayeredread_ulong1_as_uint1(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_ulong1_as_uint1");
  uint1 tmp = __surf1DLayeredread_ulong1_as_uint1(surf, x, layer, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surf1DLayeredread_long2_as_int2(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_long2_as_int2");
  int2 tmp = __surf1DLayeredread_long2_as_int2(surf, x, layer, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surf1DLayeredread_ulong2_as_uint2(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_ulong2_as_uint2");
  uint2 tmp = __surf1DLayeredread_ulong2_as_uint2(surf, x, layer, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surf1DLayeredread_long4_as_int4(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_long4_as_int4");
  int4 tmp = __surf1DLayeredread_long4_as_int4(surf, x, layer, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surf1DLayeredread_ulong4_as_uint4(surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredread_ulong4_as_uint4");
  uint4 tmp = __surf1DLayeredread_ulong4_as_uint4(surf, x, layer, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__  */

template<> __device__ __cudart_builtin__ float surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_float") ;
template<> __device__ __cudart_builtin__ float1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_float1") ;
template<> __device__ __cudart_builtin__ float2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_float2") ;
template<> __device__ __cudart_builtin__ float4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ uchar1     __surf2DLayeredreadc1(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc1");
extern __device__ __device_builtin__ uchar2     __surf2DLayeredreadc2(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc2") ;
extern __device__ __device_builtin__ uchar4     __surf2DLayeredreadc4(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc4") ;
extern __device__ __device_builtin__ ushort1    __surf2DLayeredreads1(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads1") ;
extern __device__ __device_builtin__ ushort2    __surf2DLayeredreads2(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads2") ;
extern __device__ __device_builtin__ ushort4    __surf2DLayeredreads4(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads4") ;
extern __device__ __device_builtin__ uint1      __surf2DLayeredreadu1(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu1") ;
extern __device__ __device_builtin__ uint2      __surf2DLayeredreadu2(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu2") ;
extern __device__ __device_builtin__ uint4      __surf2DLayeredreadu4(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu4") ;
extern __device__ __device_builtin__ ulonglong1 __surf2DLayeredreadl1(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl1") ;
extern __device__ __device_builtin__ ulonglong2 __surf2DLayeredreadl2(surface<void, cudaSurfaceType2DLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl2") ;

extern __device__ __device_builtin__ uchar1     __surf2DLayeredreadc1(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc1");
extern __device__ __device_builtin__ uchar2     __surf2DLayeredreadc2(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc2") ;
extern __device__ __device_builtin__ uchar4     __surf2DLayeredreadc4(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc4") ;
extern __device__ __device_builtin__ ushort1    __surf2DLayeredreads1(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads1") ;
extern __device__ __device_builtin__ ushort2    __surf2DLayeredreads2(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads2") ;
extern __device__ __device_builtin__ ushort4    __surf2DLayeredreads4(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads4") ;
extern __device__ __device_builtin__ uint1      __surf2DLayeredreadu1(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu1") ;
extern __device__ __device_builtin__ uint2      __surf2DLayeredreadu2(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu2") ;
extern __device__ __device_builtin__ uint4      __surf2DLayeredreadu4(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu4") ;
extern __device__ __device_builtin__ ulonglong1 __surf2DLayeredreadl1(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl1") ;
extern __device__ __device_builtin__ ulonglong2 __surf2DLayeredreadl2(surface<void, cudaSurfaceTypeCubemap> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl2") ;

extern __device__ __device_builtin__ uchar1     __surf2DLayeredreadc1(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc1");
extern __device__ __device_builtin__ uchar2     __surf2DLayeredreadc2(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc2") ;
extern __device__ __device_builtin__ uchar4     __surf2DLayeredreadc4(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadc4") ;
extern __device__ __device_builtin__ ushort1    __surf2DLayeredreads1(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads1") ;
extern __device__ __device_builtin__ ushort2    __surf2DLayeredreads2(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads2") ;
extern __device__ __device_builtin__ ushort4    __surf2DLayeredreads4(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreads4") ;
extern __device__ __device_builtin__ uint1      __surf2DLayeredreadu1(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu1") ;
extern __device__ __device_builtin__ uint2      __surf2DLayeredreadu2(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu2") ;
extern __device__ __device_builtin__ uint4      __surf2DLayeredreadu4(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadu4") ;
extern __device__ __device_builtin__ ulonglong1 __surf2DLayeredreadl1(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl1") ;
extern __device__ __device_builtin__ ulonglong2 __surf2DLayeredreadl2(surface<void, cudaSurfaceTypeCubemapLayered> t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredreadl2") ;

#define __surfModeSwitch(surf, x, y, layer, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, layer, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, layer, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, layer, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, layer, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, layer, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  T tmp;
  
  surf2DLayeredread(&tmp, surf, x, y, layer, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surf2DLayeredread<T>(surf, x, y, layer, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_char") ;
template<> __device__ __cudart_builtin__ signed char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uchar4") ;
template<> __device__ __cudart_builtin__ short surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ushort4") ;
template<> __device__ __cudart_builtin__ int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uint") ;
template<> __device__ __cudart_builtin__ int1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_ulonglong2") ;

#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surf2DLayeredread_long_as_int(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_long_as_int");
  return (long int)__surf2DLayeredread_long_as_int(surf, x, y, layer, mode);
}

template<>
__device__ __forceinline__ unsigned long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surf2DLayeredread_ulong_as_uint(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_ulong_as_uint");
  return (unsigned long int)__surf2DLayeredread_ulong_as_uint(surf, x, y, layer, mode);
}

template<>
__device__ __forceinline__ long1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surf2DLayeredread_long1_as_int1(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_long1_as_int1");
  int1 tmp = __surf2DLayeredread_long1_as_int1(surf, x, y, layer, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surf2DLayeredread_ulong1_as_uint1(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_ulong1_as_uint1");
  uint1 tmp = __surf2DLayeredread_ulong1_as_uint1(surf, x, y, layer, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surf2DLayeredread_long2_as_int2(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_long2_as_int2");
  int2 tmp = __surf2DLayeredread_long2_as_int2(surf, x, y, layer, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surf2DLayeredread_ulong2_as_uint2(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_ulong2_as_uint2");
  uint2 tmp = __surf2DLayeredread_ulong2_as_uint2(surf, x, y, layer, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surf2DLayeredread_long4_as_int4(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_long4_as_int4");
  int4 tmp = __surf2DLayeredread_long4_as_int4(surf, x, y, layer, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surf2DLayeredread_ulong4_as_uint4(surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredread_ulong4_as_uint4");
  uint4 tmp = __surf2DLayeredread_ulong4_as_uint4(surf, x, y, layer, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__  */

template<> __device__ __cudart_builtin__ float surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_float") ;
template<> __device__ __cudart_builtin__ float1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_float1") ;
template<> __device__ __cudart_builtin__ float2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_float2") ;
template<> __device__ __cudart_builtin__ float4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf2DLayeredread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics 
#define __surfModeSwitch(surf, x, y, face, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, face, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, face, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, face, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, face, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, face, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  T tmp;
  
  surfCubemapread(&tmp, surf, x, y, face, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surfCubemapread<T>(surf, x, y, face, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_char") ;
template<> __device__ __cudart_builtin__ signed char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uchar4") ;
template<> __device__ __cudart_builtin__ short surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ushort4") ;
template<> __device__ __cudart_builtin__ int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uint") ;
template<> __device__ __cudart_builtin__ int1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_ulonglong2") ;

#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surfCubemapread_long_as_int(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_long_as_int");
  return (long int)__surfCubemapread_long_as_int(surf, x, y, face, mode);
}

template<>
__device__ __forceinline__ unsigned long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surfCubemapread_ulong_as_uint(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_ulong_as_uint");
  return (unsigned long)__surfCubemapread_ulong_as_uint(surf, x, y, face, mode);
}

template<>
__device__ __forceinline__ long1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surfCubemapread_long1_as_int1(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_long1_as_int1");
  int1 tmp = __surfCubemapread_long1_as_int1(surf, x, y, face, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surfCubemapread_ulong1_as_uint1(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_ulong1_as_uint1");
  uint1 tmp = __surfCubemapread_ulong1_as_uint1(surf, x, y, face, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surfCubemapread_long2_as_int2(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_long2_as_int2");
  int2 tmp = __surfCubemapread_long2_as_int2(surf, x, y, face, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surfCubemapread_ulong2_as_uint2(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_ulong2_as_uint2");
  uint2 tmp = __surfCubemapread_ulong2_as_uint2(surf, x, y, face, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surfCubemapread_long4_as_int4(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_long4_as_int4");
  int4 tmp = __surfCubemapread_long4_as_int4(surf, x, y, face, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surfCubemapread_ulong4_as_uint4(surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapread_ulong4_as_uint4");
  uint4 tmp = __surfCubemapread_ulong4_as_uint4(surf, x, y, face, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__  */

template<> __device__ __cudart_builtin__ float surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_float") ;
template<> __device__ __cudart_builtin__ float1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_float1") ;
template<> __device__ __cudart_builtin__ float2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_float2") ;
template<> __device__ __cudart_builtin__ float4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)

#define __surfModeSwitch(surf, x, y, layerFace, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, layerFace, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, layerFace, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ T surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  T tmp;
  
  surfCubemapLayeredread(&tmp, surf, x, y, layerFace, (int)sizeof(T), mode);
  
  return tmp;
#endif /* __CUDA_ARCH__ */  
}

template<class T>
__device__ __forceinline__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  *res = surfCubemapLayeredread<T>(surf, x, y, layerFace, mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
template<> __device__ __cudart_builtin__ char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_char") ;
template<> __device__ __cudart_builtin__ signed char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_schar") ;
template<> __device__ __cudart_builtin__ unsigned char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uchar") ;
template<> __device__ __cudart_builtin__ char1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_char1") ;
template<> __device__ __cudart_builtin__ uchar1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uchar1") ;
template<> __device__ __cudart_builtin__ char2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_char2") ;
template<> __device__ __cudart_builtin__ uchar2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uchar2") ;
template<> __device__ __cudart_builtin__ char4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_char4") ;
template<> __device__ __cudart_builtin__ uchar4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uchar4") ;
template<> __device__ __cudart_builtin__ short surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_short") ;
template<> __device__ __cudart_builtin__ unsigned short surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ushort") ;
template<> __device__ __cudart_builtin__ short1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_short1") ;
template<> __device__ __cudart_builtin__ ushort1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ushort1") ;
template<> __device__ __cudart_builtin__ short2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_short2") ;
template<> __device__ __cudart_builtin__ ushort2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ushort2") ;
template<> __device__ __cudart_builtin__ short4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_short4") ;
template<> __device__ __cudart_builtin__ ushort4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ushort4") ;
template<> __device__ __cudart_builtin__ int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_int") ;
template<> __device__ __cudart_builtin__ unsigned int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uint") ;
template<> __device__ __cudart_builtin__ int1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_int1") ;
template<> __device__ __cudart_builtin__ uint1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uint1") ;
template<> __device__ __cudart_builtin__ int2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_int2") ;
template<> __device__ __cudart_builtin__ uint2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uint2") ;
template<> __device__ __cudart_builtin__ int4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_int4") ;
template<> __device__ __cudart_builtin__ uint4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_uint4") ;
template<> __device__ __cudart_builtin__ long long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_longlong") ;
template<> __device__ __cudart_builtin__ unsigned long long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ulonglong") ;
template<> __device__ __cudart_builtin__ longlong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_longlong1") ;
template<> __device__ __cudart_builtin__ ulonglong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ulonglong1") ;
template<> __device__ __cudart_builtin__ longlong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_longlong2") ;
template<> __device__ __cudart_builtin__ ulonglong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_ulonglong2") ;

#if !defined(__LP64__)
template<>
__device__ __forceinline__ long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int __surfCubemapLayeredread_long_as_int(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_long_as_int");
  return (long int)__surfCubemapLayeredread_long_as_int(surf, x, y, layerFace, mode);
}

template<>
__device__ __forceinline__ unsigned long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ unsigned int __surfCubemapLayeredread_ulong_as_uint(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_ulong_as_uint");
  return (unsigned long int)__surfCubemapLayeredread_ulong_as_uint(surf, x, y, layerFace, mode);
}

template<>
__device__ __forceinline__ long1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int1 __surfCubemapLayeredread_long1_as_int1(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_long1_as_int1");
  int1 tmp = __surfCubemapLayeredread_long1_as_int1(surf, x, y, layerFace, mode);
  return make_long1((long int)tmp.x);
}

template<>
__device__ __forceinline__ ulong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint1 __surfCubemapLayeredread_ulong1_as_uint1(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_ulong1_as_uint1");
  uint1 tmp = __surfCubemapLayeredread_ulong1_as_uint1(surf, x, y, layerFace, mode);
  return make_ulong1((unsigned long int)tmp.x);
}

template<>
__device__ __forceinline__ long2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int2 __surfCubemapLayeredread_long2_as_int2(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_long2_as_int2");
  int2 tmp = __surfCubemapLayeredread_long2_as_int2(surf, x, y, layerFace, mode);
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__device__ __forceinline__ ulong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint2 __surfCubemapLayeredread_ulong2_as_uint2(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_ulong2_as_uint2");
  uint2 tmp = __surfCubemapLayeredread_ulong2_as_uint2(surf, x, y, layerFace, mode);
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__device__ __forceinline__ long4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ int4 __surfCubemapLayeredread_long4_as_int4(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_long4_as_int4");
  int4 tmp = __surfCubemapLayeredread_long4_as_int4(surf, x, y, layerFace, mode);
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__device__ __forceinline__ ulong4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ uint4 __surfCubemapLayeredread_ulong4_as_uint4(surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredread_ulong4_as_uint4");
  uint4 tmp = __surfCubemapLayeredread_ulong4_as_uint4(surf, x, y, layerFace, mode);
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}
#endif /* !__LP64__  */

template<> __device__ __cudart_builtin__ float surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_float") ;
template<> __device__ __cudart_builtin__ float1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_float1") ;
template<> __device__ __cudart_builtin__ float2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_float2") ;
template<> __device__ __cudart_builtin__ float4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode) asm("__surfCubemapLayeredread_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ void __surf1Dwritec1(    uchar1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwritec1");
extern __device__ __device_builtin__ void __surf1Dwritec2(    uchar2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwritec2");
extern __device__ __device_builtin__ void __surf1Dwritec4(    uchar4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwritec4");
extern __device__ __device_builtin__ void __surf1Dwrites1(   ushort1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwrites1");
extern __device__ __device_builtin__ void __surf1Dwrites2(   ushort2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwrites2");
extern __device__ __device_builtin__ void __surf1Dwrites4(   ushort4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwrites4");
extern __device__ __device_builtin__ void __surf1Dwriteu1(     uint1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwriteu1");
extern __device__ __device_builtin__ void __surf1Dwriteu2(     uint2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwriteu2");
extern __device__ __device_builtin__ void __surf1Dwriteu4(     uint4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwriteu4");
extern __device__ __device_builtin__ void __surf1Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwritel1");
extern __device__ __device_builtin__ void __surf1Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode) asm("__surf1Dwritel2");

#define __surfModeSwitch(val, surf, x, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeClamp) : \
                                           __surf1Dwrite##type(val, surf, x, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  surf1Dwrite(val, surf, x, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */  
}

#ifdef __CUDA_ARCH__
__device__ __cudart_builtin__ void surf1Dwrite(char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_char") ;
__device__ __cudart_builtin__ void surf1Dwrite(signed char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_schar") ;
__device__ __cudart_builtin__ void surf1Dwrite(unsigned char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uchar") ;
__device__ __cudart_builtin__ void surf1Dwrite(char1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_char1") ;
__device__ __cudart_builtin__ void surf1Dwrite(uchar1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uchar1") ;
__device__ __cudart_builtin__ void surf1Dwrite(char2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_char2") ;
__device__ __cudart_builtin__ void surf1Dwrite(uchar2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uchar2") ;
__device__ __cudart_builtin__ void surf1Dwrite(char4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_char4") ;
__device__ __cudart_builtin__ void surf1Dwrite(uchar4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uchar4") ;
__device__ __cudart_builtin__ void surf1Dwrite(short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_short") ;
__device__ __cudart_builtin__ void surf1Dwrite(unsigned short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ushort") ;
__device__ __cudart_builtin__ void surf1Dwrite(short1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_short1") ;
__device__ __cudart_builtin__ void surf1Dwrite(ushort1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ushort1") ;
__device__ __cudart_builtin__ void surf1Dwrite(short2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_short2") ;
__device__ __cudart_builtin__ void surf1Dwrite(ushort2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ushort2") ;
__device__ __cudart_builtin__ void surf1Dwrite(short4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_short4") ;
__device__ __cudart_builtin__ void surf1Dwrite(ushort4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ushort4") ;
__device__ __cudart_builtin__ void surf1Dwrite(int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_int") ;
__device__ __cudart_builtin__ void surf1Dwrite(unsigned int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uint") ;
__device__ __cudart_builtin__ void surf1Dwrite(int1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_int1") ;
__device__ __cudart_builtin__ void surf1Dwrite(uint1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uint1") ;
__device__ __cudart_builtin__ void surf1Dwrite(int2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_int2") ;
__device__ __cudart_builtin__ void surf1Dwrite(uint2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uint2") ;
__device__ __cudart_builtin__ void surf1Dwrite(int4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_int4") ;
__device__ __cudart_builtin__ void surf1Dwrite(uint4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_uint4") ;
__device__ __cudart_builtin__ void surf1Dwrite(long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_longlong") ;
__device__ __cudart_builtin__ void surf1Dwrite(unsigned long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ulonglong") ;
__device__ __cudart_builtin__ void surf1Dwrite(longlong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_longlong1") ;
__device__ __cudart_builtin__ void surf1Dwrite(ulonglong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surf1Dwrite(longlong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_longlong2") ;
__device__ __cudart_builtin__ void surf1Dwrite(ulonglong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surf1Dwrite(long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_long_as_int(int, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_long_as_int");
  __surf1Dwrite_long_as_int((int)val, surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(unsigned long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_ulong_as_uint");
  __surf1Dwrite_ulong_as_uint((unsigned int)val, surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(long1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_long1_as_int1(int1, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_long1_as_int1");
  __surf1Dwrite_long1_as_int1(make_int1((int)val.x), surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(ulong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_ulong1_as_uint1");
  __surf1Dwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(long2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_long2_as_int2(int2, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_long2_as_int2");
  __surf1Dwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(ulong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_ulong2_as_uint2");
  __surf1Dwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(long4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_long4_as_int4(int4, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_long4_as_int4");
  __surf1Dwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, mode);
}

static __device__ __forceinline__ void surf1Dwrite(ulong4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1Dwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceType1D>, int, enum cudaSurfaceBoundaryMode) asm("__surf1Dwrite_ulong4_as_uint4");
  __surf1Dwrite_ulong4_as_uint4(make_uint4((unsigned)val.x, (unsigned)val.y, (unsigned)val.z, (unsigned)val.w), surf, x, mode);
}
#endif /* !__LP64__  */
__device__ __cudart_builtin__ void surf1Dwrite(float val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_float") ;
__device__ __cudart_builtin__ void surf1Dwrite(float1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_float1") ;
__device__ __cudart_builtin__ void surf1Dwrite(float2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_float2") ;
__device__ __cudart_builtin__ void surf1Dwrite(float4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1Dwrite_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ void __surf2Dwritec1(    uchar1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwritec1") ;
extern __device__ __device_builtin__ void __surf2Dwritec2(    uchar2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwritec2") ;
extern __device__ __device_builtin__ void __surf2Dwritec4(    uchar4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwritec4") ;
extern __device__ __device_builtin__ void __surf2Dwrites1(   ushort1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwrites1") ;
extern __device__ __device_builtin__ void __surf2Dwrites2(   ushort2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwrites2") ;
extern __device__ __device_builtin__ void __surf2Dwrites4(   ushort4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwrites4") ;
extern __device__ __device_builtin__ void __surf2Dwriteu1(     uint1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwriteu1") ;
extern __device__ __device_builtin__ void __surf2Dwriteu2(     uint2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwriteu2") ;
extern __device__ __device_builtin__ void __surf2Dwriteu4(     uint4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwriteu4") ;
extern __device__ __device_builtin__ void __surf2Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwritel1") ;
extern __device__ __device_builtin__ void __surf2Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode) asm("__surf2Dwritel2") ;

#define __surfModeSwitch(val, surf, x, y, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeClamp) : \
                                           __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__  void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  surf2Dwrite(val, surf, x, y, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */  
}
#ifdef __CUDA_ARCH__ 
__device__ __cudart_builtin__ void surf2Dwrite(char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_char") ;
__device__ __cudart_builtin__ void surf2Dwrite(signed char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_schar") ;
__device__ __cudart_builtin__ void surf2Dwrite(unsigned char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uchar") ;
__device__ __cudart_builtin__ void surf2Dwrite(char1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_char1") ;
__device__ __cudart_builtin__ void surf2Dwrite(uchar1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uchar1") ;
__device__ __cudart_builtin__ void surf2Dwrite(char2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_char2") ;
__device__ __cudart_builtin__ void surf2Dwrite(uchar2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uchar2") ;
__device__ __cudart_builtin__ void surf2Dwrite(char4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_char4") ;
__device__ __cudart_builtin__ void surf2Dwrite(uchar4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uchar4") ;
__device__ __cudart_builtin__ void surf2Dwrite(short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_short") ;
__device__ __cudart_builtin__ void surf2Dwrite(unsigned short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ushort") ;
__device__ __cudart_builtin__ void surf2Dwrite(short1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_short1") ;
__device__ __cudart_builtin__ void surf2Dwrite(ushort1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ushort1") ;
__device__ __cudart_builtin__ void surf2Dwrite(short2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_short2") ;
__device__ __cudart_builtin__ void surf2Dwrite(ushort2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ushort2") ;
__device__ __cudart_builtin__ void surf2Dwrite(short4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_short4") ;
__device__ __cudart_builtin__ void surf2Dwrite(ushort4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ushort4") ;
__device__ __cudart_builtin__ void surf2Dwrite(int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_int") ;
__device__ __cudart_builtin__ void surf2Dwrite(unsigned int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uint") ;
__device__ __cudart_builtin__ void surf2Dwrite(int1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_int1") ;
__device__ __cudart_builtin__ void surf2Dwrite(uint1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uint1") ;
__device__ __cudart_builtin__ void surf2Dwrite(int2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_int2") ;
__device__ __cudart_builtin__ void surf2Dwrite(uint2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uint2") ;
__device__ __cudart_builtin__ void surf2Dwrite(int4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_int4") ;
__device__ __cudart_builtin__ void surf2Dwrite(uint4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_uint4") ;
__device__ __cudart_builtin__ void surf2Dwrite(long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_longlong") ;
__device__ __cudart_builtin__ void surf2Dwrite(unsigned long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ulonglong") ;
__device__ __cudart_builtin__ void surf2Dwrite(longlong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_longlong1") ;
__device__ __cudart_builtin__ void surf2Dwrite(ulonglong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surf2Dwrite(longlong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_longlong2") ;
__device__ __cudart_builtin__ void surf2Dwrite(ulonglong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surf2Dwrite(long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_long_as_int(int, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_long_as_int");
  __surf2Dwrite_long_as_int((int)val, surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(unsigned long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_ulong_as_uint");
  __surf2Dwrite_ulong_as_uint((unsigned int)val, surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(long1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_long1_as_int1(int1, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_long1_as_int1");
  __surf2Dwrite_long1_as_int1(make_int1((int)val.x), surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(ulong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_ulong1_as_uint1");
  __surf2Dwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(long2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_long2_as_int2(int2, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_long2_as_int2");
  __surf2Dwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(ulong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_ulong2_as_uint2");
  __surf2Dwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(long4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_long4_as_int4(int4, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_long4_as_int4");
  __surf2Dwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, y, mode);
}

static __device__ __forceinline__ void surf2Dwrite(ulong4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2Dwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceType2D>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2Dwrite_ulong4_as_uint4");
  __surf2Dwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode);
}
#endif /* !__LP64__  && defined(__CUDACC_RTC__) */
__device__ __cudart_builtin__ void surf2Dwrite(float val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_float") ;
__device__ __cudart_builtin__ void surf2Dwrite(float1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_float1") ;
__device__ __cudart_builtin__ void surf2Dwrite(float2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_float2") ;
__device__ __cudart_builtin__ void surf2Dwrite(float4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2Dwrite_float4") ;
#endif /* __CUDA_ARCH__ */
#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ void __surf3Dwritec1(    uchar1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwritec1") ;
extern __device__ __device_builtin__ void __surf3Dwritec2(    uchar2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwritec2") ;
extern __device__ __device_builtin__ void __surf3Dwritec4(    uchar4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwritec4") ;
extern __device__ __device_builtin__ void __surf3Dwrites1(   ushort1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwrites1") ;
extern __device__ __device_builtin__ void __surf3Dwrites2(   ushort2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwrites2") ;
extern __device__ __device_builtin__ void __surf3Dwrites4(   ushort4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwrites4") ;
extern __device__ __device_builtin__ void __surf3Dwriteu1(     uint1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwriteu1") ;
extern __device__ __device_builtin__ void __surf3Dwriteu2(     uint2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwriteu2") ;
extern __device__ __device_builtin__ void __surf3Dwriteu4(     uint4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwriteu4") ;
extern __device__ __device_builtin__ void __surf3Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwritel1") ;
extern __device__ __device_builtin__ void __surf3Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode) asm("__surf3Dwritel2") ;

#define __surfModeSwitch(val, surf, x, y, z, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeClamp) : \
                                           __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, z, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, z, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, z, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, z, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, z, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  surf3Dwrite(val, surf, x, y, z, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
__device__ __cudart_builtin__ void surf3Dwrite(char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_char") ;
__device__ __cudart_builtin__ void surf3Dwrite(signed char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_schar") ;
__device__ __cudart_builtin__ void surf3Dwrite(unsigned char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uchar") ;
__device__ __cudart_builtin__ void surf3Dwrite(char1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_char1") ;
__device__ __cudart_builtin__ void surf3Dwrite(uchar1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uchar1") ;
__device__ __cudart_builtin__ void surf3Dwrite(char2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_char2") ;
__device__ __cudart_builtin__ void surf3Dwrite(uchar2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uchar2") ;
__device__ __cudart_builtin__ void surf3Dwrite(char4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_char4") ;
__device__ __cudart_builtin__ void surf3Dwrite(uchar4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uchar4") ;
__device__ __cudart_builtin__ void surf3Dwrite(short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_short") ;
__device__ __cudart_builtin__ void surf3Dwrite(unsigned short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ushort") ;
__device__ __cudart_builtin__ void surf3Dwrite(short1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_short1") ;
__device__ __cudart_builtin__ void surf3Dwrite(ushort1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ushort1") ;
__device__ __cudart_builtin__ void surf3Dwrite(short2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_short2") ;
__device__ __cudart_builtin__ void surf3Dwrite(ushort2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ushort2") ;
__device__ __cudart_builtin__ void surf3Dwrite(short4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_short4") ;
__device__ __cudart_builtin__ void surf3Dwrite(ushort4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ushort4") ;
__device__ __cudart_builtin__ void surf3Dwrite(int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_int") ;
__device__ __cudart_builtin__ void surf3Dwrite(unsigned int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uint") ;
__device__ __cudart_builtin__ void surf3Dwrite(int1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_int1") ;
__device__ __cudart_builtin__ void surf3Dwrite(uint1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uint1") ;
__device__ __cudart_builtin__ void surf3Dwrite(int2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_int2") ;
__device__ __cudart_builtin__ void surf3Dwrite(uint2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uint2") ;
__device__ __cudart_builtin__ void surf3Dwrite(int4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_int4") ;
__device__ __cudart_builtin__ void surf3Dwrite(uint4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_uint4") ;
__device__ __cudart_builtin__ void surf3Dwrite(long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_longlong") ;
__device__ __cudart_builtin__ void surf3Dwrite(unsigned long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ulonglong") ;
__device__ __cudart_builtin__ void surf3Dwrite(longlong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_longlong1") ;
__device__ __cudart_builtin__ void surf3Dwrite(ulonglong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surf3Dwrite(longlong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_longlong2") ;
__device__ __cudart_builtin__ void surf3Dwrite(ulonglong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surf3Dwrite(long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_long_as_int(int, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_long_as_int");
  __surf3Dwrite_long_as_int((int)val, surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(unsigned long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_ulong_as_uint");
  __surf3Dwrite_ulong_as_uint((unsigned int)val, surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(long1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_long1_as_int1(int1, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_long1_as_int1");
  __surf3Dwrite_long1_as_int1(make_int1((int)val.x), surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(ulong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_ulong1_as_uint1");
  __surf3Dwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(long2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_long2_as_int2(int2, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_long2_as_int2");
  __surf3Dwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(ulong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_ulong2_as_uint2");
  __surf3Dwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(long4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_long4_as_int4(int4, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_long4_as_int4");
  __surf3Dwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, y, z, mode);
}

static __device__ __forceinline__ void surf3Dwrite(ulong4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf3Dwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceType3D>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf3Dwrite_ulong4_as_uint4");
  __surf3Dwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode);
}
#endif /* !__LP64__  */
__device__ __cudart_builtin__ void surf3Dwrite(float val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_float") ;
__device__ __cudart_builtin__ void surf3Dwrite(float1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_float1") ;
__device__ __cudart_builtin__ void surf3Dwrite(float2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_float2") ;
__device__ __cudart_builtin__ void surf3Dwrite(float4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf3Dwrite_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__)
extern __device__ __device_builtin__ void __surf1DLayeredwritec1(    uchar1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwritec1") ;
extern __device__ __device_builtin__ void __surf1DLayeredwritec2(    uchar2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwritec2") ;
extern __device__ __device_builtin__ void __surf1DLayeredwritec4(    uchar4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwritec4") ;
extern __device__ __device_builtin__ void __surf1DLayeredwrites1(   ushort1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwrites1") ;
extern __device__ __device_builtin__ void __surf1DLayeredwrites2(   ushort2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwrites2") ;
extern __device__ __device_builtin__ void __surf1DLayeredwrites4(   ushort4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwrites4") ;
extern __device__ __device_builtin__ void __surf1DLayeredwriteu1(     uint1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwriteu1") ;
extern __device__ __device_builtin__ void __surf1DLayeredwriteu2(     uint2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwriteu2") ;
extern __device__ __device_builtin__ void __surf1DLayeredwriteu4(     uint4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwriteu4") ;
extern __device__ __device_builtin__ void __surf1DLayeredwritel1(ulonglong1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwritel1") ;
extern __device__ __device_builtin__ void __surf1DLayeredwritel2(ulonglong2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode) asm("__surf1DLayeredwritel2") ;


#define __surfModeSwitch(val, surf, x, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeClamp) : \
                                           __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
static __device__ __forceinline__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, layer, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, layer, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, layer, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, layer, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, layer, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
static __device__ __forceinline__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  surf1DLayeredwrite(val, surf, x, layer, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */  
}

#if defined(__CUDA_ARCH__)
__device__ __cudart_builtin__ void surf1DLayeredwrite(char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_char") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(signed char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_schar") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uchar") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(char1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_char1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uchar1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(char2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_char2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uchar2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(char4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_char4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uchar4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_short") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ushort") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(short1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_short1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ushort1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(short2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_short2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ushort2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(short4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_short4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ushort4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_int") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uint") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(int1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_int1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uint1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(int2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_int2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uint2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(int4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_int4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(uint4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_uint4") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_longlong") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ulonglong") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_longlong1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_longlong2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surf1DLayeredwrite(long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_long_as_int(int, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_long_as_int");
  __surf1DLayeredwrite_long_as_int((int)val, surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_ulong_as_uint");
  __surf1DLayeredwrite_ulong_as_uint((unsigned int)val, surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(long1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_long1_as_int1(int1, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_long1_as_int1");
  __surf1DLayeredwrite_long1_as_int1(make_int1((int)val.x), surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_ulong1_as_uint1");
  __surf1DLayeredwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(long2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_long2_as_int2(int2, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_long2_as_int2");
  __surf1DLayeredwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_ulong2_as_uint2");
  __surf1DLayeredwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(long4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_long4_as_int4(int4, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_long4_as_int4");
  __surf1DLayeredwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, layer, mode);
}

static __device__ __forceinline__ void surf1DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf1DLayeredwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceType1DLayered>, int, int, enum cudaSurfaceBoundaryMode) asm("__surf1DLayeredwrite_ulong4_as_uint4");
  __surf1DLayeredwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ void surf1DLayeredwrite(float val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_float") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(float1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_float1") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(float2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_float2") ;
__device__ __cudart_builtin__ void surf1DLayeredwrite(float4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf1DLayeredwrite_float4") ;
#endif /* __CUDA_ARCH__ */
#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__) 
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec1(    uchar1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwritec1") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec2(    uchar2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwritec2") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec4(    uchar4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwritec4") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites1(   ushort1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwrites1") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites2(   ushort2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwrites2") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites4(   ushort4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwrites4") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu1(     uint1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwriteu1") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu2(     uint2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwriteu2") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu4(     uint4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwriteu4") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritel1(ulonglong1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwritel1") ;
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritel2(ulonglong2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode) __ASM_IF_INTEGRATED("__surf2DLayeredwritel2") ;


#define __surfModeSwitch(val, surf, x, y, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
 __device__ __forceinline__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__) 
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, layer, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, layer, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, layer, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, layer, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, layer, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
__device__ __forceinline__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)   
  surf2DLayeredwrite(val, surf, x, y, layer, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */
}

#if defined(__CUDA_ARCH__) 
__device__ __cudart_builtin__ void surf2DLayeredwrite(char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_char") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(signed char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_schar") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uchar") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(char1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_char1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uchar1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(char2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_char2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uchar2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(char4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_char4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uchar4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_short") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ushort") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(short1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_short1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ushort1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(short2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_short2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ushort2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(short4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_short4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ushort4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_int") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uint") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(int1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_int1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uint1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(int2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_int2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uint2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(int4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_int4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(uint4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_uint4") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_longlong") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ulonglong") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_longlong1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_longlong2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surf2DLayeredwrite(long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_long_as_int(int, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_long_as_int");
  __surf2DLayeredwrite_long_as_int((int)val, surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_ulong_as_uint(int, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_ulong_as_uint");
  __surf2DLayeredwrite_ulong_as_uint((unsigned int)val, surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(long1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_long1_as_int1(int1, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_long1_as_int1");
  __surf2DLayeredwrite_long1_as_int1(make_int1((int)val.x), surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_ulong1_as_uint1");
  __surf2DLayeredwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(long2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_long2_as_int2(int2, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_long2_as_int2");
  __surf2DLayeredwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_ulong2_as_uint2");
  __surf2DLayeredwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(long4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_long4_as_int4(int4, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_long4_as_int4");
  __surf2DLayeredwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, y, layer, mode);
}

static __device__ __forceinline__ void surf2DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surf2DLayeredwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceType2DLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surf2DLayeredwrite_ulong4_as_uint4");
  __surf2DLayeredwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode);
}

#endif /* !__LP64__ */
__device__ __cudart_builtin__ void surf2DLayeredwrite(float val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_float") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(float1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_float1") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(float2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_float2") ;
__device__ __cudart_builtin__ void surf2DLayeredwrite(float4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surf2DLayeredwrite_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__)
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, face, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
__device__ __forceinline__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, face, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, face, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, face, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, face, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, face, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */
}

template<class T>
__device__ __forceinline__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)
  surfCubemapwrite(val, surf, x, y, face, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */
}

#if defined(__CUDA_ARCH__)
__device__ __cudart_builtin__ void surfCubemapwrite(char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_char") ;
__device__ __cudart_builtin__ void surfCubemapwrite(signed char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_schar") ;
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uchar") ;
__device__ __cudart_builtin__ void surfCubemapwrite(char1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_char1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uchar1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(char2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_char2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uchar2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(char4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_char4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uchar4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_short") ;
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ushort") ;
__device__ __cudart_builtin__ void surfCubemapwrite(short1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_short1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ushort1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(short2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_short2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ushort2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(short4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_short4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ushort4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_int") ;
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uint") ;
__device__ __cudart_builtin__ void surfCubemapwrite(int1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_int1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uint1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uint1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(int2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_int2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uint2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uint2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(int4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_int4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(uint4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_uint4") ;
__device__ __cudart_builtin__ void surfCubemapwrite(long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_longlong") ;
__device__ __cudart_builtin__ void surfCubemapwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ulonglong") ;
__device__ __cudart_builtin__ void surfCubemapwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_longlong1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_longlong2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surfCubemapwrite(long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_long_as_int(int, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_long_as_int");
  __surfCubemapwrite_long_as_int((int)val, surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_ulong_as_uint");
  __surfCubemapwrite_ulong_as_uint((unsigned int)val, surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(long1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_long1_as_int1(int1, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_long1_as_int1");
  __surfCubemapwrite_long1_as_int1(make_int1((int)val.x), surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_ulong1_as_uint1");
  __surfCubemapwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(long2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_long2_as_int2(int2, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_long2_as_int2");
  __surfCubemapwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_ulong2_as_uint2");
  __surfCubemapwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(long4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_long4_as_int4(int4, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_long4_as_int4");
  __surfCubemapwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, y, face, mode);
}

static __device__ __forceinline__ void surfCubemapwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceTypeCubemap>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapwrite_ulong4_as_uint4");
  __surfCubemapwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ void surfCubemapwrite(float val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_float") ;
__device__ __cudart_builtin__ void surfCubemapwrite(float1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_float1") ;
__device__ __cudart_builtin__ void surfCubemapwrite(float2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_float2") ;
__device__ __cudart_builtin__ void surfCubemapwrite(float4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapwrite_float4") ;
#endif /* __CUDA_ARCH__ */

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__)
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, layerFace, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeTrap ))
#endif /* __CUDA_ARCH__ */

template<class T>
static __device__ __forceinline__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, layerFace, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, layerFace, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, layerFace, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, layerFace, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, layerFace, mode, u4)) :
              (void)0;
#endif /* __CUDA_ARCH__ */              
}

template<class T>
static __device__ __forceinline__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#if defined(__CUDA_ARCH__)  
  surfCubemapLayeredwrite(val, surf, x, y, layerFace, (int)sizeof(T), mode);
#endif /* __CUDA_ARCH__ */
}

#if defined(__CUDA_ARCH__)
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_char") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(signed char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_schar") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uchar") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_char1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uchar1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_char2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uchar2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(char4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_char4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uchar4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_short") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ushort") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_short1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ushort1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_short2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ushort2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(short4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_short4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ushort4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_int") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uint") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_int1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uint1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_int2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uint2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(int4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_int4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(uint4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_uint4") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_longlong") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ulonglong") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_longlong1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ulonglong1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_longlong2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_ulonglong2") ;
#if !defined(__LP64__)
static __device__ __forceinline__ void surfCubemapLayeredwrite(long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_long_as_int(int, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_long_as_int");
  __surfCubemapLayeredwrite_long_as_int((int)val, surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_ulong_as_uint(unsigned int, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_ulong_as_uint");
  __surfCubemapLayeredwrite_ulong_as_uint((unsigned int)val, surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(long1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_long1_as_int1(int1, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_long1_as_int1");
  __surfCubemapLayeredwrite_long1_as_int1(make_int1((int)val.x), surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_ulong1_as_uint1(uint1, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_ulong1_as_uint1");
  __surfCubemapLayeredwrite_ulong1_as_uint1(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(long2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_long2_as_int2(int2, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_long2_as_int2");
  __surfCubemapLayeredwrite_long2_as_int2(make_int2((int)val.x, (int)val.y), surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_ulong2_as_uint2(uint2, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_ulong2_as_uint2");
  __surfCubemapLayeredwrite_ulong2_as_uint2(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(long4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_long4_as_int4(int4, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_long4_as_int4");
  __surfCubemapLayeredwrite_long4_as_int4(make_int4((int)val.x, (int)val.y, (int)val.z, (int)val.w), surf, x, y, layerFace, mode);
}

static __device__ __forceinline__ void surfCubemapLayeredwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __device__ __cudart_builtin__ void __surfCubemapLayeredwrite_ulong4_as_uint4(uint4, surface<void, cudaSurfaceTypeCubemapLayered>, int, int, int, enum cudaSurfaceBoundaryMode) asm("__surfCubemapLayeredwrite_ulong4_as_uint4");
  __surfCubemapLayeredwrite_ulong4_as_uint4(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_float") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_float1") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_float2") ;
__device__ __cudart_builtin__ void surfCubemapLayeredwrite(float4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) asm("__surfCubemapLayeredwrite_float4") ;
#endif /* __CUDA_ARCH__ */
#undef __surfModeSwitch
/*******************************************************************************
                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#elif defined(__CUDABE__)

extern uchar1     __surf1Dreadc1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1Dreadc2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1Dreadc4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1Dreads1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1Dreads2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1Dreads4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1Dreadu1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1Dreadu2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1Dreadu4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1Dreadl1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1Dreadl2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2Dreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2Dreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2Dreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2Dreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2Dreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2Dreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2Dreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2Dreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2Dreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2Dreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2Dreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf3Dreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf3Dreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf3Dreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf3Dreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf3Dreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf3Dreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf3Dreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf3Dreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf3Dreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf3Dreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf3Dreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf1DLayeredreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1DLayeredreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1DLayeredreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1DLayeredreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1DLayeredreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1DLayeredreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1DLayeredreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1DLayeredreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1DLayeredreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1DLayeredreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1DLayeredreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2DLayeredreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2DLayeredreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2DLayeredreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2DLayeredreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2DLayeredreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2DLayeredreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2DLayeredreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2DLayeredreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2DLayeredreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2DLayeredreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2DLayeredreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec1(    uchar1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec2(    uchar2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec4(    uchar4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites1(   ushort1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites2(   ushort2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites4(   ushort4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu1(     uint1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu2(     uint2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu4(     uint4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel1(ulonglong1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel2(ulonglong2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec1(    uchar1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec2(    uchar2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec4(    uchar4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites1(   ushort1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites2(   ushort2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites4(   ushort4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu1(     uint1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu2(     uint2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu4(     uint4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel1(ulonglong1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel2(ulonglong2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec1(    uchar1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec2(    uchar2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec4(    uchar4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites1(   ushort1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites2(   ushort2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites4(   ushort4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu1(     uint1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu2(     uint2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu4(     uint4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
#endif /* __cplusplus && __CUDACC__ */
#endif /* !__SURFACE_FUNCTIONS_H__ */
