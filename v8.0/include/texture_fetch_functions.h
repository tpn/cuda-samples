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

#if !defined(__TEXTURE_FETCH_FUNCTIONS_H__)
#define __TEXTURE_FETCH_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "cuda_texture_types.h"
#include "host_defines.h"
#include "texture_types.h"
#include "vector_functions.h"
#include "vector_types.h"

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1Dfetch(texture<T, cudaTextureType1D, cudaReadModeElementType>, int) {  }

template <typename T>
struct __nv_tex_rmnf_ret { };

template <> struct __nv_tex_rmnf_ret<char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<signed char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<char1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<uchar1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<short1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<ushort1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<char2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<uchar2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<short2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<ushort2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<char4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<uchar4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<short4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1Dfetch(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>, int) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1Dfetch(texture<char, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1Dfetch(texture<char, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1Dfetch(texture<signed char, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_schar") ;
__device__ __cudart_builtin__ unsigned char tex1Dfetch(texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uchar") ;
__device__ __cudart_builtin__ char1 tex1Dfetch(texture<char1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_char1") ;
__device__ __cudart_builtin__ uchar1 tex1Dfetch(texture<uchar1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uchar1") ;
__device__ __cudart_builtin__ char2 tex1Dfetch(texture<char2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_char2") ;
__device__ __cudart_builtin__ uchar2 tex1Dfetch(texture<uchar2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uchar2") ;
__device__ __cudart_builtin__ char4 tex1Dfetch(texture<char4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_char4") ;
__device__ __cudart_builtin__ uchar4 tex1Dfetch(texture<uchar4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uchar4") ;

__device__ __cudart_builtin__ short tex1Dfetch(texture<short, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_short") ;
__device__ __cudart_builtin__ unsigned short tex1Dfetch(texture<unsigned short, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_ushort") ;
__device__ __cudart_builtin__ short1 tex1Dfetch(texture<short1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_short1") ;
__device__ __cudart_builtin__ ushort1 tex1Dfetch(texture<ushort1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_ushort1") ;
__device__ __cudart_builtin__ short2 tex1Dfetch(texture<short2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_short2") ;
__device__ __cudart_builtin__ ushort2 tex1Dfetch(texture<ushort2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_ushort2") ;
__device__ __cudart_builtin__ short4 tex1Dfetch(texture<short4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_short4") ;
__device__ __cudart_builtin__ ushort4 tex1Dfetch(texture<ushort4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_ushort4") ;

__device__ __cudart_builtin__ int tex1Dfetch(texture<int, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_int") ;
__device__ __cudart_builtin__ unsigned int tex1Dfetch(texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uint") ;
__device__ __cudart_builtin__ int1 tex1Dfetch(texture<int1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_int1") ;
__device__ __cudart_builtin__ uint1 tex1Dfetch(texture<uint1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uint1") ;
__device__ __cudart_builtin__ int2 tex1Dfetch(texture<int2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_int2") ;
__device__ __cudart_builtin__ uint2 tex1Dfetch(texture<uint2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uint2") ;
__device__ __cudart_builtin__ int4 tex1Dfetch(texture<int4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_int4") ;
__device__ __cudart_builtin__ uint4 tex1Dfetch(texture<uint4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1Dfetch(texture<long, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ int __tex1Dfetch_long_as_int(texture<long, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_long_as_int");
  return __tex1Dfetch_long_as_int(t, x);
}

static __device__ __forceinline__ unsigned long tex1Dfetch(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ unsigned __tex1Dfetch_ulong_as_uint(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_ulong_as_uint");
  return __tex1Dfetch_ulong_as_uint(t, x);
}

static __device__ __forceinline__ long1 tex1Dfetch(texture<long1, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ int1  __tex1Dfetch_long1_as_int1(texture<long1, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_long1_as_int1");
  int1 v = __tex1Dfetch_long1_as_int1(t, x);
  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1Dfetch(texture<ulong1, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ uint1  __tex1Dfetch_ulong1_as_uint1(texture<ulong1, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_ulong1_as_uint1");
  uint1 v = __tex1Dfetch_ulong1_as_uint1(t, x);
  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1Dfetch(texture<long2, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ int2  __tex1Dfetch_long2_as_int2(texture<long2, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_long2_as_int2");
  int2 v = __tex1Dfetch_long2_as_int2(t, x);
  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1Dfetch(texture<ulong2, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ uint2  __tex1Dfetch_ulong2_as_uint2(texture<ulong2, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_ulong2_as_uint2");
  uint2 v = __tex1Dfetch_ulong2_as_uint2(t, x);
  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1Dfetch(texture<long4, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ int4  __tex1Dfetch_long4_as_int4(texture<long4, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_long4_as_int4");
  int4 v = __tex1Dfetch_long4_as_int4(t, x);
  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1Dfetch(texture<ulong4, cudaTextureType1D, cudaReadModeElementType> t, int x)
{
  __device__ __cudart_builtin__ uint4  __tex1Dfetch_ulong4_as_uint4(texture<ulong4, cudaTextureType1D, cudaReadModeElementType>, int) asm("__tex1Dfetch_ulong4_as_uint4");
  uint4 v = __tex1Dfetch_ulong4_as_uint4(t, x);
  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1Dfetch(texture<float, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_float") ;
__device__ __cudart_builtin__ float1 tex1Dfetch(texture<float1, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_float1") ;
__device__ __cudart_builtin__ float2 tex1Dfetch(texture<float2, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_float2") ;
__device__ __cudart_builtin__ float4 tex1Dfetch(texture<float4, cudaTextureType1D, cudaReadModeElementType> t, int x) asm("__tex1Dfetch_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1Dfetch(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x)  asm("__tex1Dfetch_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1Dfetch(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x)  asm("__tex1Dfetch_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1Dfetch(texture<signed char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x)  asm("__tex1Dfetch_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1Dfetch(texture<unsigned char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x)  asm("__tex1Dfetch_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1Dfetch(texture<char1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x)  asm("__tex1Dfetch_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1Dfetch(texture<uchar1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1Dfetch(texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1Dfetch(texture<uchar2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1Dfetch(texture<char4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_char4") ;

__device__ __cudart_builtin__ float4 tex1Dfetch(texture<uchar4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_uchar4") ;
__device__ __cudart_builtin__ float tex1Dfetch(texture<short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_short") ;
__device__ __cudart_builtin__ float tex1Dfetch(texture<unsigned short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1Dfetch(texture<short1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1Dfetch(texture<ushort1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1Dfetch(texture<short2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1Dfetch(texture<ushort2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1Dfetch(texture<short4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1Dfetch(texture<ushort4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, int x) asm("__tex1Dfetch_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */


#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1D(texture<T, cudaTextureType1D, cudaReadModeElementType>, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1D(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1D(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1D(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1D(texture<signed char, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_schar") ;
__device__ __cudart_builtin__ unsigned char tex1D(texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uchar") ;
__device__ __cudart_builtin__ char1 tex1D(texture<char1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_char1") ;
__device__ __cudart_builtin__ uchar1 tex1D(texture<uchar1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uchar1") ;
__device__ __cudart_builtin__ char2 tex1D(texture<char2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_char2") ;
__device__ __cudart_builtin__ uchar2 tex1D(texture<uchar2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uchar2") ;
__device__ __cudart_builtin__ char4 tex1D(texture<char4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_char4") ;
__device__ __cudart_builtin__ uchar4 tex1D(texture<uchar4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uchar4") ;

__device__ __cudart_builtin__ short tex1D(texture<short, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_short") ;
__device__ __cudart_builtin__ unsigned short tex1D(texture<unsigned short, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_ushort") ;
__device__ __cudart_builtin__ short1 tex1D(texture<short1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_short1") ;
__device__ __cudart_builtin__ ushort1 tex1D(texture<ushort1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_ushort1") ;
__device__ __cudart_builtin__ short2 tex1D(texture<short2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_short2") ;
__device__ __cudart_builtin__ ushort2 tex1D(texture<ushort2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_ushort2") ;
__device__ __cudart_builtin__ short4 tex1D(texture<short4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_short4") ;
__device__ __cudart_builtin__ ushort4 tex1D(texture<ushort4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_ushort4") ;


__device__ __cudart_builtin__ int tex1D(texture<int, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_int") ;
__device__ __cudart_builtin__ unsigned int tex1D(texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uint") ;
__device__ __cudart_builtin__ int1 tex1D(texture<int1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_int1") ;
__device__ __cudart_builtin__ uint1 tex1D(texture<uint1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uint1") ;
__device__ __cudart_builtin__ int2 tex1D(texture<int2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_int2") ;
__device__ __cudart_builtin__ uint2 tex1D(texture<uint2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uint2") ;
__device__ __cudart_builtin__ int4 tex1D(texture<int4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_int4") ;
__device__ __cudart_builtin__ uint4 tex1D(texture<uint4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1D(texture<long, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ int __tex1D_long_as_int(texture<long, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_long_as_int");
  return __tex1D_long_as_int(t, x);
}

static __device__ __forceinline__ unsigned long tex1D(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ unsigned __tex1D_ulong_as_uint(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_ulong_as_uint");
  return __tex1D_ulong_as_uint(t, x);
}

static __device__ __forceinline__ long1 tex1D(texture<long1, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ int1 __tex1D_long1_as_int1(texture<long1, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_long1_as_int1");
  int1 v = __tex1D_long1_as_int1(t, x);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1D(texture<ulong1, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ uint1 __tex1D_ulong1_as_uint1(texture<ulong1, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_ulong1_as_uint1");
  uint1 v = __tex1D_ulong1_as_uint1(t, x);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1D(texture<long2, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ int2 __tex1D_long2_as_int2(texture<long2, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_long2_as_int2");
  int2 v = __tex1D_long2_as_int2(t, x);
  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1D(texture<ulong2, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ uint2 __tex1D_ulong2_as_uint2(texture<ulong2, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_ulong2_as_uint2");
  uint2 v = __tex1D_ulong2_as_uint2(t, x);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1D(texture<long4, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ int4 __tex1D_long4_as_int4(texture<long4, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_long4_as_int4");
  int4 v = __tex1D_long4_as_int4(t, x);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1D(texture<ulong4, cudaTextureType1D, cudaReadModeElementType> t, float x)
{
  __device__ __cudart_builtin__ uint4 __tex1D_ulong4_as_uint4(texture<ulong4, cudaTextureType1D, cudaReadModeElementType>, float)  asm("__tex1D_ulong4_as_uint4");
  uint4 v = __tex1D_ulong4_as_uint4(t, x);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1D(texture<float, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_float") ;
__device__ __cudart_builtin__ float1 tex1D(texture<float1, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_float1") ;
__device__ __cudart_builtin__ float2 tex1D(texture<float2, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_float2") ;
__device__ __cudart_builtin__ float4 tex1D(texture<float4, cudaTextureType1D, cudaReadModeElementType> t, float x) asm("__tex1D_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1D(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1D(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1D(texture<signed char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1D(texture<unsigned char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1D(texture<char1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1D(texture<uchar1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1D(texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1D(texture<uchar2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1D(texture<char4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1D(texture<uchar4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex1D(texture<short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_short") ;
__device__ __cudart_builtin__ float tex1D(texture<unsigned short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1D(texture<short1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1D(texture<ushort1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1D(texture<short2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1D(texture<ushort2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1D(texture<short4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1D(texture<ushort4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x) asm("__tex1D_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */


#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2D(texture<T, cudaTextureType2D, cudaReadModeElementType>, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2D(texture<T, cudaTextureType2D, cudaReadModeNormalizedFloat>, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2D(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2D(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2D(texture<signed char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_schar") ;
__device__ __cudart_builtin__ unsigned char tex2D(texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uchar") ;
__device__ __cudart_builtin__ char1 tex2D(texture<char1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_char1") ;
__device__ __cudart_builtin__ uchar1 tex2D(texture<uchar1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uchar1") ;
__device__ __cudart_builtin__ char2 tex2D(texture<char2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_char2") ;
__device__ __cudart_builtin__ uchar2 tex2D(texture<uchar2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uchar2") ;
__device__ __cudart_builtin__ char4 tex2D(texture<char4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_char4") ;
__device__ __cudart_builtin__ uchar4 tex2D(texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uchar4") ;

__device__ __cudart_builtin__ short tex2D(texture<short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_short") ;
__device__ __cudart_builtin__ unsigned short tex2D(texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_ushort") ;
__device__ __cudart_builtin__ short1 tex2D(texture<short1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_short1") ;
__device__ __cudart_builtin__ ushort1 tex2D(texture<ushort1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_ushort1") ;
__device__ __cudart_builtin__ short2 tex2D(texture<short2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_short2") ;
__device__ __cudart_builtin__ ushort2 tex2D(texture<ushort2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_ushort2") ;
__device__ __cudart_builtin__ short4 tex2D(texture<short4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_short4") ;
__device__ __cudart_builtin__ ushort4 tex2D(texture<ushort4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_ushort4") ;

__device__ __cudart_builtin__ int tex2D(texture<int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_int") ;
__device__ __cudart_builtin__ unsigned int tex2D(texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uint") ;
__device__ __cudart_builtin__ int1 tex2D(texture<int1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_int1") ;
__device__ __cudart_builtin__ uint1 tex2D(texture<uint1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uint1") ;
__device__ __cudart_builtin__ int2 tex2D(texture<int2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_int2") ;
__device__ __cudart_builtin__ uint2 tex2D(texture<uint2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uint2") ;
__device__ __cudart_builtin__ int4 tex2D(texture<int4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_int4") ;
__device__ __cudart_builtin__ uint4 tex2D(texture<uint4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2D(texture<long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__ int __tex2D_long_as_int(texture<long, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_long_as_int");
  return __tex2D_long_as_int(t, x, y);
}


static __device__ __forceinline__ unsigned long tex2D(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__ unsigned int __tex2D_ulong_as_uint(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_ulong_as_uint");
  return __tex2D_ulong_as_uint(t, x, y);
}

static __device__ __forceinline__ long1 tex2D(texture<long1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  int1 __tex2D_long1_as_int1(texture<long1, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_long1_as_int1");
  int1 v = __tex2D_long1_as_int1(t, x, y);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2D(texture<ulong1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  uint1 __tex2D_ulong1_as_uint1(texture<ulong1, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_ulong1_as_uint1");
  uint1 v = __tex2D_ulong1_as_uint1(t, x, y);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2D(texture<long2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  int2 __tex2D_long2_as_int2(texture<long2, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_long2_as_int2");
  int2 v = __tex2D_long2_as_int2(t, x, y);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2D(texture<ulong2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  uint2 __tex2D_ulong2_as_uint2(texture<ulong2, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_ulong2_as_uint2");
  uint2 v = __tex2D_ulong2_as_uint2(t, x, y);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2D(texture<long4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  int4 __tex2D_long4_as_int4(texture<long4, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_long4_as_int4");
  int4 v = __tex2D_long4_as_int4(t, x, y);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2D(texture<ulong4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y)
{
  __device__ __cudart_builtin__  uint4 __tex2D_ulong4_as_uint4(texture<ulong4, cudaTextureType2D, cudaReadModeElementType>, float, float)  asm("__tex2D_ulong4_as_uint4");
  uint4 v = __tex2D_ulong4_as_uint4(t, x, y);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */

__device__ __cudart_builtin__ float tex2D(texture<float, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_float") ;
__device__ __cudart_builtin__ float1 tex2D(texture<float1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_float1") ;
__device__ __cudart_builtin__ float2 tex2D(texture<float2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_float2") ;
__device__ __cudart_builtin__ float4 tex2D(texture<float4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y) asm("__tex2D_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2D(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2D(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2D(texture<signed char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2D(texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2D(texture<char1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2D(texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2D(texture<char2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2D(texture<uchar2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2D(texture<char4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2D(texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_uchar4") ;
__device__ __cudart_builtin__ float tex2D(texture<short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_short") ;
__device__ __cudart_builtin__ float tex2D(texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2D(texture<short1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2D(texture<ushort1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2D(texture<short2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2D(texture<ushort2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2D(texture<short4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2D(texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y) asm("__tex2D_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1DLayered(texture<T, cudaTextureType1DLayered, cudaReadModeElementType>, float, int) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1DLayered(texture<T, cudaTextureType1DLayered, cudaReadModeNormalizedFloat>, float, int) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1DLayered(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1DLayered(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1DLayered(texture<signed char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_schar") ;
__device__ __cudart_builtin__ unsigned char tex1DLayered(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uchar") ;
__device__ __cudart_builtin__ char1 tex1DLayered(texture<char1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_char1") ;
__device__ __cudart_builtin__ uchar1 tex1DLayered(texture<uchar1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uchar1") ;
__device__ __cudart_builtin__ char2 tex1DLayered(texture<char2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_char2") ;
__device__ __cudart_builtin__ uchar2 tex1DLayered(texture<uchar2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uchar2") ;
__device__ __cudart_builtin__ char4 tex1DLayered(texture<char4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_char4") ;
__device__ __cudart_builtin__ uchar4 tex1DLayered(texture<uchar4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uchar4") ;

__device__ __cudart_builtin__ short tex1DLayered(texture<short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_short") ;
__device__ __cudart_builtin__ unsigned short tex1DLayered(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_ushort") ;
__device__ __cudart_builtin__ short1 tex1DLayered(texture<short1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_short1") ;
__device__ __cudart_builtin__ ushort1 tex1DLayered(texture<ushort1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_ushort1") ;
__device__ __cudart_builtin__ short2 tex1DLayered(texture<short2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_short2") ;
__device__ __cudart_builtin__ ushort2 tex1DLayered(texture<ushort2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_ushort2") ;
__device__ __cudart_builtin__ short4 tex1DLayered(texture<short4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_short4") ;
__device__ __cudart_builtin__ ushort4 tex1DLayered(texture<ushort4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_ushort4") ;

__device__ __cudart_builtin__ int tex1DLayered(texture<int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_int") ;
__device__ __cudart_builtin__ unsigned int tex1DLayered(texture<unsigned int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uint") ;
__device__ __cudart_builtin__ int1 tex1DLayered(texture<int1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_int1") ;
__device__ __cudart_builtin__ uint1 tex1DLayered(texture<uint1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uint1") ;
__device__ __cudart_builtin__ int2 tex1DLayered(texture<int2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_int2") ;
__device__ __cudart_builtin__ uint2 tex1DLayered(texture<uint2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uint2") ;
__device__ __cudart_builtin__ int4 tex1DLayered(texture<int4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_int4") ;
__device__ __cudart_builtin__ uint4 tex1DLayered(texture<uint4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1DLayered(texture<long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  int __tex1DLayered_long_as_int(texture<long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_long_as_int");
  return __tex1DLayered_long_as_int(t, x, layer);
}

static __device__ __forceinline__ unsigned long tex1DLayered(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  unsigned __tex1DLayered_ulong_as_uint(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_ulong_as_uint");
  return __tex1DLayered_ulong_as_uint(t, x, layer);
}

static __device__ __forceinline__ long1 tex1DLayered(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  int1 __tex1DLayered_long1_as_int1(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_long1_as_int1");
  int1 v = __tex1DLayered_long1_as_int1(t, x, layer);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1DLayered(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  uint1 __tex1DLayered_ulong1_as_uint1(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_ulong1_as_uint1");
  uint1 v = __tex1DLayered_ulong1_as_uint1(t, x, layer);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1DLayered(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  int2 __tex1DLayered_long2_as_int2(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_long2_as_int2");
  int2 v = __tex1DLayered_long2_as_int2(t, x, layer);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1DLayered(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  uint2 __tex1DLayered_ulong2_as_uint2(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_ulong2_as_uint2");
  uint2 v = __tex1DLayered_ulong2_as_uint2(t, x, layer);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1DLayered(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  int4 __tex1DLayered_long4_as_int4(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_long4_as_int4");
  int4 v = __tex1DLayered_long4_as_int4(t, x, layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1DLayered(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer)
{
  __device__ __cudart_builtin__  uint4 __tex1DLayered_ulong4_as_uint4(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int)  asm("__tex1DLayered_ulong4_as_uint4");
  uint4 v = __tex1DLayered_ulong4_as_uint4(t, x, layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1DLayered(texture<float, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_float") ;
__device__ __cudart_builtin__ float1 tex1DLayered(texture<float1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_float1") ;
__device__ __cudart_builtin__ float2 tex1DLayered(texture<float2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_float2") ;
__device__ __cudart_builtin__ float4 tex1DLayered(texture<float4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer) asm("__tex1DLayered_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1DLayered(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1DLayered(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1DLayered(texture<signed char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1DLayered(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1DLayered(texture<char1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1DLayered(texture<uchar1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1DLayered(texture<char2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1DLayered(texture<uchar2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1DLayered(texture<char4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1DLayered(texture<uchar4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex1DLayered(texture<short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_short") ;
__device__ __cudart_builtin__ float tex1DLayered(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1DLayered(texture<short1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1DLayered(texture<ushort1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1DLayered(texture<short2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1DLayered(texture<ushort2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1DLayered(texture<short4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1DLayered(texture<ushort4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer) asm("__tex1DLayered_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2DLayered(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2DLayered(texture<T, cudaTextureType2DLayered, cudaReadModeNormalizedFloat>, float, float, int) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2DLayered(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2DLayered(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2DLayered(texture<signed char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_schar") ;
__device__ __cudart_builtin__ unsigned char tex2DLayered(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uchar") ;
__device__ __cudart_builtin__ char1 tex2DLayered(texture<char1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_char1") ;
__device__ __cudart_builtin__ uchar1 tex2DLayered(texture<uchar1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uchar1") ;
__device__ __cudart_builtin__ char2 tex2DLayered(texture<char2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_char2") ;
__device__ __cudart_builtin__ uchar2 tex2DLayered(texture<uchar2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uchar2") ;
__device__ __cudart_builtin__ char4 tex2DLayered(texture<char4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_char4") ;
__device__ __cudart_builtin__ uchar4 tex2DLayered(texture<uchar4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uchar4") ;

__device__ __cudart_builtin__ short tex2DLayered(texture<short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_short") ;
__device__ __cudart_builtin__ unsigned short tex2DLayered(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_ushort") ;
__device__ __cudart_builtin__ short1 tex2DLayered(texture<short1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_short1") ;
__device__ __cudart_builtin__ ushort1 tex2DLayered(texture<ushort1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_ushort1") ;
__device__ __cudart_builtin__ short2 tex2DLayered(texture<short2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_short2") ;
__device__ __cudart_builtin__ ushort2 tex2DLayered(texture<ushort2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_ushort2") ;
__device__ __cudart_builtin__ short4 tex2DLayered(texture<short4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_short4") ;
__device__ __cudart_builtin__ ushort4 tex2DLayered(texture<ushort4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_ushort4") ;

__device__ __cudart_builtin__ int tex2DLayered(texture<int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_int") ;
__device__ __cudart_builtin__ unsigned int tex2DLayered(texture<unsigned int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uint") ;
__device__ __cudart_builtin__ int1 tex2DLayered(texture<int1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_int1") ;
__device__ __cudart_builtin__ uint1 tex2DLayered(texture<uint1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uint1") ;
__device__ __cudart_builtin__ int2 tex2DLayered(texture<int2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_int2") ;
__device__ __cudart_builtin__ uint2 tex2DLayered(texture<uint2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uint2") ;
__device__ __cudart_builtin__ int4 tex2DLayered(texture<int4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_int4") ;
__device__ __cudart_builtin__ uint4 tex2DLayered(texture<uint4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2DLayered(texture<long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  int __tex2DLayered_long_as_int(texture<long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_long_as_int");

  return __tex2DLayered_long_as_int(t, x, y, layer);
}

static __device__ __forceinline__ unsigned long tex2DLayered(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  unsigned int __tex2DLayered_ulong_as_uint(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_ulong_as_uint");

  return __tex2DLayered_ulong_as_uint(t, x, y, layer);
}

static __device__ __forceinline__ long1 tex2DLayered(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  int1 __tex2DLayered_long1_as_int1(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_long1_as_int1");
  int1 v = __tex2DLayered_long1_as_int1(t, x, y, layer);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2DLayered(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  uint1 __tex2DLayered_ulong1_as_uint1(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_ulong1_as_uint1");
  uint1 v = __tex2DLayered_ulong1_as_uint1(t, x, y, layer);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2DLayered(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  int2 __tex2DLayered_long2_as_int2(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_long2_as_int2");
  int2 v = __tex2DLayered_long2_as_int2(t, x, y, layer);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2DLayered(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  uint2 __tex2DLayered_ulong2_as_uint2(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_ulong2_as_uint2");
  uint2 v = __tex2DLayered_ulong2_as_uint2(t, x, y, layer);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2DLayered(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  int4 __tex2DLayered_long4_as_int4(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_long4_as_int4");
  int4 v = __tex2DLayered_long4_as_int4(t, x, y, layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2DLayered(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer)
{
  __device__ __cudart_builtin__  uint4 __tex2DLayered_ulong4_as_uint4(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int)  asm("__tex2DLayered_ulong4_as_uint4");
  uint4 v = __tex2DLayered_ulong4_as_uint4(t, x, y, layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex2DLayered(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_float") ;
__device__ __cudart_builtin__ float1 tex2DLayered(texture<float1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_float1") ;
__device__ __cudart_builtin__ float2 tex2DLayered(texture<float2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_float2") ;
__device__ __cudart_builtin__ float4 tex2DLayered(texture<float4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer) asm("__tex2DLayered_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2DLayered(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2DLayered(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2DLayered(texture<signed char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2DLayered(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2DLayered(texture<char1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2DLayered(texture<uchar1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2DLayered(texture<char2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2DLayered(texture<uchar2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2DLayered(texture<char4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2DLayered(texture<uchar4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex2DLayered(texture<short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_short") ;
__device__ __cudart_builtin__ float tex2DLayered(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2DLayered(texture<short1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2DLayered(texture<ushort1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2DLayered(texture<short2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2DLayered(texture<ushort2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2DLayered(texture<short4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2DLayered(texture<ushort4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer) asm("__tex2DLayered_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */


#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex3D(texture<T, cudaTextureType3D, cudaReadModeElementType>, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex3D(texture<T, cudaTextureType3D, cudaReadModeNormalizedFloat>, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex3D(texture<char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex3D(texture<char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex3D(texture<signed char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_schar") ;
__device__ __cudart_builtin__ unsigned char tex3D(texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uchar") ;
__device__ __cudart_builtin__ char1 tex3D(texture<char1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_char1") ;
__device__ __cudart_builtin__ uchar1 tex3D(texture<uchar1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uchar1") ;
__device__ __cudart_builtin__ char2 tex3D(texture<char2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_char2") ;
__device__ __cudart_builtin__ uchar2 tex3D(texture<uchar2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uchar2") ;
__device__ __cudart_builtin__ char4 tex3D(texture<char4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_char4") ;
__device__ __cudart_builtin__ uchar4 tex3D(texture<uchar4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uchar4") ;

__device__ __cudart_builtin__ short tex3D(texture<short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_short") ;
__device__ __cudart_builtin__ unsigned short tex3D(texture<unsigned short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_ushort") ;
__device__ __cudart_builtin__ short1 tex3D(texture<short1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_short1") ;
__device__ __cudart_builtin__ ushort1 tex3D(texture<ushort1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_ushort1") ;
__device__ __cudart_builtin__ short2 tex3D(texture<short2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_short2") ;
__device__ __cudart_builtin__ ushort2 tex3D(texture<ushort2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_ushort2") ;
__device__ __cudart_builtin__ short4 tex3D(texture<short4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_short4") ;
__device__ __cudart_builtin__ ushort4 tex3D(texture<ushort4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_ushort4") ;

__device__ __cudart_builtin__ int tex3D(texture<int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_int") ;
__device__ __cudart_builtin__ unsigned int tex3D(texture<unsigned int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uint") ;
__device__ __cudart_builtin__ int1 tex3D(texture<int1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_int1") ;
__device__ __cudart_builtin__ uint1 tex3D(texture<uint1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uint1") ;
__device__ __cudart_builtin__ int2 tex3D(texture<int2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_int2") ;
__device__ __cudart_builtin__ uint2 tex3D(texture<uint2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uint2") ;
__device__ __cudart_builtin__ int4 tex3D(texture<int4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_int4") ;
__device__ __cudart_builtin__ uint4 tex3D(texture<uint4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex3D(texture<long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int __tex3D_long_as_int(texture<long, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_long_as_int");
  return __tex3D_long_as_int(t, x, y, z);
}

static __device__ __forceinline__ unsigned long tex3D(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  unsigned int __tex3D_ulong_as_uint(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_ulong_as_uint");
  return __tex3D_ulong_as_uint(t, x, y, z);
}

static __device__ __forceinline__ long1 tex3D(texture<long1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int1 __tex3D_long1_as_int1(texture<long1, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_long1_as_int1");
  int1 v = __tex3D_long1_as_int1(t, x, y, z);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex3D(texture<ulong1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint1 __tex3D_ulong1_as_uint1(texture<ulong1, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_ulong1_as_uint1");
  uint1 v = __tex3D_ulong1_as_uint1(t, x, y, z);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex3D(texture<long2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int2 __tex3D_long2_as_int2(texture<long2, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_long2_as_int2");
  int2 v = __tex3D_long2_as_int2(t, x, y, z);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex3D(texture<ulong2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint2 __tex3D_ulong2_as_uint2(texture<ulong2, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_ulong2_as_uint2");
  uint2 v = __tex3D_ulong2_as_uint2(t, x, y, z);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex3D(texture<long4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int4 __tex3D_long4_as_int4(texture<long4, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_long4_as_int4");
  int4 v = __tex3D_long4_as_int4(t, x, y, z);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex3D(texture<ulong4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint4 __tex3D_ulong4_as_uint4(texture<ulong4, cudaTextureType3D, cudaReadModeElementType>, float, float, float)  asm("__tex3D_ulong4_as_uint4");
  uint4 v = __tex3D_ulong4_as_uint4(t, x, y, z);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex3D(texture<float, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_float") ;
__device__ __cudart_builtin__ float1 tex3D(texture<float1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_float1") ;
__device__ __cudart_builtin__ float2 tex3D(texture<float2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_float2") ;
__device__ __cudart_builtin__ float4 tex3D(texture<float4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z) asm("__tex3D_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex3D(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex3D(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex3D(texture<signed char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_schar") ;
__device__ __cudart_builtin__ float tex3D(texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex3D(texture<char1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex3D(texture<uchar1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex3D(texture<char2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex3D(texture<uchar2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex3D(texture<char4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex3D(texture<uchar4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex3D(texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_short") ;
__device__ __cudart_builtin__ float tex3D(texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex3D(texture<short1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex3D(texture<ushort1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex3D(texture<short2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex3D(texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex3D(texture<short4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex3D(texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__tex3D_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T texCubemap(texture<T, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type texCubemap(texture<T, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat>, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char texCubemap(texture<char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char texCubemap(texture<char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char texCubemap(texture<signed char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_schar") ;
__device__ __cudart_builtin__ unsigned char texCubemap(texture<unsigned char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uchar") ;
__device__ __cudart_builtin__ char1 texCubemap(texture<char1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_char1") ;
__device__ __cudart_builtin__ uchar1 texCubemap(texture<uchar1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uchar1") ;
__device__ __cudart_builtin__ char2 texCubemap(texture<char2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_char2") ;
__device__ __cudart_builtin__ uchar2 texCubemap(texture<uchar2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uchar2") ;
__device__ __cudart_builtin__ char4 texCubemap(texture<char4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_char4") ;
__device__ __cudart_builtin__ uchar4 texCubemap(texture<uchar4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uchar4") ;

__device__ __cudart_builtin__ short texCubemap(texture<short, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_short") ;
__device__ __cudart_builtin__ unsigned short texCubemap(texture<unsigned short, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_ushort") ;
__device__ __cudart_builtin__ short1 texCubemap(texture<short1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_short1") ;
__device__ __cudart_builtin__ ushort1 texCubemap(texture<ushort1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_ushort1") ;
__device__ __cudart_builtin__ short2 texCubemap(texture<short2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_short2") ;
__device__ __cudart_builtin__ ushort2 texCubemap(texture<ushort2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_ushort2") ;
__device__ __cudart_builtin__ short4 texCubemap(texture<short4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_short4") ;
__device__ __cudart_builtin__ ushort4 texCubemap(texture<ushort4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_ushort4") ;

__device__ __cudart_builtin__ int texCubemap(texture<int, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_int") ;
__device__ __cudart_builtin__ unsigned int texCubemap(texture<unsigned int, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uint") ;
__device__ __cudart_builtin__ int1 texCubemap(texture<int1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_int1") ;
__device__ __cudart_builtin__ uint1 texCubemap(texture<uint1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uint1") ;
__device__ __cudart_builtin__ int2 texCubemap(texture<int2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_int2") ;
__device__ __cudart_builtin__ uint2 texCubemap(texture<uint2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uint2") ;
__device__ __cudart_builtin__ int4 texCubemap(texture<int4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_int4") ;
__device__ __cudart_builtin__ uint4 texCubemap(texture<uint4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long texCubemap(texture<long, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int __texCubemap_long_as_int(texture<long, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_long_as_int");
  return __texCubemap_long_as_int(t, x, y, z);
}

static __device__ __forceinline__ unsigned long texCubemap(texture<unsigned long, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  unsigned int __texCubemap_ulong_as_uint(texture<unsigned long, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_ulong_as_uint");
  return __texCubemap_ulong_as_uint(t, x, y, z);
}

static __device__ __forceinline__ long1 texCubemap(texture<long1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int1 __texCubemap_long1_as_int1(texture<long1, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_long1_as_int1");
  int1 v = __texCubemap_long1_as_int1(t, x, y, z);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 texCubemap(texture<ulong1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint1 __texCubemap_ulong1_as_uint1(texture<ulong1, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_ulong1_as_uint1");
  uint1 v = __texCubemap_ulong1_as_uint1(t, x, y, z);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 texCubemap(texture<long2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int2 __texCubemap_long2_as_int2(texture<long2, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_long2_as_int2");
  int2 v = __texCubemap_long2_as_int2(t, x, y, z);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 texCubemap(texture<ulong2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint2 __texCubemap_ulong2_as_uint2(texture<ulong2, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_ulong2_as_uint2");
  uint2 v = __texCubemap_ulong2_as_uint2(t, x, y, z);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 texCubemap(texture<long4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  int4 __texCubemap_long4_as_int4(texture<long4, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_long4_as_int4");
  int4 v = __texCubemap_long4_as_int4(t, x, y, z);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 texCubemap(texture<ulong4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z)
{
  __device__ __cudart_builtin__  uint4 __texCubemap_ulong4_as_uint4(texture<ulong4, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float)  asm("__texCubemap_ulong4_as_uint4");
  uint4 v = __texCubemap_ulong4_as_uint4(t, x, y, z);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float texCubemap(texture<float, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_float") ;
__device__ __cudart_builtin__ float1 texCubemap(texture<float1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_float1") ;
__device__ __cudart_builtin__ float2 texCubemap(texture<float2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_float2") ;
__device__ __cudart_builtin__ float4 texCubemap(texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z) asm("__texCubemap_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float texCubemap(texture<char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float texCubemap(texture<char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float texCubemap(texture<signed char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_schar") ;
__device__ __cudart_builtin__ float texCubemap(texture<unsigned char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 texCubemap(texture<char1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_char1") ;
__device__ __cudart_builtin__ float1 texCubemap(texture<uchar1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 texCubemap(texture<char2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_char2") ;
__device__ __cudart_builtin__ float2 texCubemap(texture<uchar2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 texCubemap(texture<char4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_char4") ;
__device__ __cudart_builtin__ float4 texCubemap(texture<uchar4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_uchar4") ;

__device__ __cudart_builtin__ float texCubemap(texture<short, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_short") ;
__device__ __cudart_builtin__ float texCubemap(texture<unsigned short, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 texCubemap(texture<short1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_short1") ;
__device__ __cudart_builtin__ float1 texCubemap(texture<ushort1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 texCubemap(texture<short2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_short2") ;
__device__ __cudart_builtin__ float2 texCubemap(texture<ushort2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 texCubemap(texture<short4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_short4") ;
__device__ __cudart_builtin__ float4 texCubemap(texture<ushort4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z) asm("__texCubemap_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T texCubemapLayered(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type texCubemapLayered(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat>, float, float, float, int) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char texCubemapLayered(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char texCubemapLayered(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char texCubemapLayered(texture<signed char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_schar") ;
__device__ __cudart_builtin__ unsigned char texCubemapLayered(texture<unsigned char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uchar") ;
__device__ __cudart_builtin__ char1 texCubemapLayered(texture<char1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_char1") ;
__device__ __cudart_builtin__ uchar1 texCubemapLayered(texture<uchar1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uchar1") ;
__device__ __cudart_builtin__ char2 texCubemapLayered(texture<char2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_char2") ;
__device__ __cudart_builtin__ uchar2 texCubemapLayered(texture<uchar2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uchar2") ;
__device__ __cudart_builtin__ char4 texCubemapLayered(texture<char4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_char4") ;
__device__ __cudart_builtin__ uchar4 texCubemapLayered(texture<uchar4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uchar4") ;

__device__ __cudart_builtin__ short texCubemapLayered(texture<short, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_short") ;
__device__ __cudart_builtin__ unsigned short texCubemapLayered(texture<unsigned short, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_ushort") ;
__device__ __cudart_builtin__ short1 texCubemapLayered(texture<short1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_short1") ;
__device__ __cudart_builtin__ ushort1 texCubemapLayered(texture<ushort1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_ushort1") ;
__device__ __cudart_builtin__ short2 texCubemapLayered(texture<short2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_short2") ;
__device__ __cudart_builtin__ ushort2 texCubemapLayered(texture<ushort2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_ushort2") ;
__device__ __cudart_builtin__ short4 texCubemapLayered(texture<short4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_short4") ;
__device__ __cudart_builtin__ ushort4 texCubemapLayered(texture<ushort4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_ushort4") ;

__device__ __cudart_builtin__ int texCubemapLayered(texture<int, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_int") ;
__device__ __cudart_builtin__ unsigned int texCubemapLayered(texture<unsigned int, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uint") ;
__device__ __cudart_builtin__ int1 texCubemapLayered(texture<int1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_int1") ;
__device__ __cudart_builtin__ uint1 texCubemapLayered(texture<uint1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uint1") ;
__device__ __cudart_builtin__ int2 texCubemapLayered(texture<int2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_int2") ;
__device__ __cudart_builtin__ uint2 texCubemapLayered(texture<uint2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uint2") ;
__device__ __cudart_builtin__ int4 texCubemapLayered(texture<int4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_int4") ;
__device__ __cudart_builtin__ uint4 texCubemapLayered(texture<uint4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long texCubemapLayered(texture<long, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  int __texCubemapLayered_long_as_int(texture<long, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_long_as_int");
  return __texCubemapLayered_long_as_int(t, x, y, z, layer);
}

static __device__ __forceinline__ unsigned long texCubemapLayered(texture<unsigned long, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  unsigned __texCubemapLayered_ulong_as_uint(texture<unsigned long, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_ulong_as_uint");
  return __texCubemapLayered_ulong_as_uint(t, x, y, z, layer);
}

static __device__ __forceinline__ long1 texCubemapLayered(texture<long1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  int1 __texCubemapLayered_long1_as_int1(texture<long1, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_long1_as_int1");
  int1 v =  __texCubemapLayered_long1_as_int1(t, x, y, z, layer);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 texCubemapLayered(texture<ulong1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  uint1 __texCubemapLayered_ulong1_as_uint1(texture<ulong1, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_ulong1_as_uint1");
  uint1 v =  __texCubemapLayered_ulong1_as_uint1(t, x, y, z, layer);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 texCubemapLayered(texture<long2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  int2 __texCubemapLayered_long2_as_int2(texture<long2, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_long2_as_int2");
  int2 v =  __texCubemapLayered_long2_as_int2(t, x, y, z, layer);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 texCubemapLayered(texture<ulong2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  uint2 __texCubemapLayered_ulong2_as_uint2(texture<ulong2, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_ulong2_as_uint2");
  uint2 v =  __texCubemapLayered_ulong2_as_uint2(t, x, y, z, layer);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 texCubemapLayered(texture<long4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  int4 __texCubemapLayered_long4_as_int4(texture<long4, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_long4_as_int4");
  int4 v =  __texCubemapLayered_long4_as_int4(t, x, y, z, layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 texCubemapLayered(texture<ulong4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer)
{
  __device__ __cudart_builtin__  uint4 __texCubemapLayered_ulong4_as_uint4(texture<ulong4, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int)  asm("__texCubemapLayered_ulong4_as_uint4");
  uint4 v =  __texCubemapLayered_ulong4_as_uint4(t, x, y, z, layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float texCubemapLayered(texture<float, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_float") ;
__device__ __cudart_builtin__ float1 texCubemapLayered(texture<float1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_float1") ;
__device__ __cudart_builtin__ float2 texCubemapLayered(texture<float2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_float2") ;
__device__ __cudart_builtin__ float4 texCubemapLayered(texture<float4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer) asm("__texCubemapLayered_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float texCubemapLayered(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float texCubemapLayered(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float texCubemapLayered(texture<signed char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_schar") ;
__device__ __cudart_builtin__ float texCubemapLayered(texture<unsigned char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 texCubemapLayered(texture<char1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_char1") ;
__device__ __cudart_builtin__ float1 texCubemapLayered(texture<uchar1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 texCubemapLayered(texture<char2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_char2") ;
__device__ __cudart_builtin__ float2 texCubemapLayered(texture<uchar2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 texCubemapLayered(texture<char4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_char4") ;
__device__ __cudart_builtin__ float4 texCubemapLayered(texture<uchar4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_uchar4") ;

__device__ __cudart_builtin__ float texCubemapLayered(texture<short, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_short") ;
__device__ __cudart_builtin__ float texCubemapLayered(texture<unsigned short, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 texCubemapLayered(texture<short1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_short1") ;
__device__ __cudart_builtin__ float1 texCubemapLayered(texture<ushort1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 texCubemapLayered(texture<short2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_short2") ;
__device__ __cudart_builtin__ float2 texCubemapLayered(texture<ushort2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 texCubemapLayered(texture<short4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_short4") ;
__device__ __cudart_builtin__ float4 texCubemapLayered(texture<ushort4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer) asm("__texCubemapLayered_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__

template <typename T>
struct __nv_tex2dgather_ret { };
template <> struct __nv_tex2dgather_ret<char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<signed char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char1> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char2> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char3> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char4> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<unsigned char> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar1> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar2> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar3> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar4> { typedef uchar4 type; };

template <> struct __nv_tex2dgather_ret<short> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short1> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short2> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short3> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short4> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<unsigned short> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort1> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort2> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort3> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort4> { typedef ushort4 type; };

template <> struct __nv_tex2dgather_ret<int> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int1> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int2> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int3> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int4> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<unsigned int> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint1> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint2> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint3> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint4> { typedef uint4 type; };

template <> struct __nv_tex2dgather_ret<float> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float1> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float2> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float3> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float4> { typedef float4 type; };

template <typename T>
static __device__ typename __nv_tex2dgather_ret<T>::type tex2Dgather(texture<T, cudaTextureType2D, cudaReadModeElementType>, float, float, int=0) {  }

template <typename T>
static __device__ float4 tex2Dgather(texture<T, cudaTextureType2D, cudaReadModeNormalizedFloat>, float, float, int = 0) {  }
#else /* __CUDA_ARCH__ */

__device__ __cudart_builtin__ char4 tex2Dgather(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_char") ;
__device__ __cudart_builtin__ char4 tex2Dgather(texture<signed char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_schar") ;
__device__ __cudart_builtin__ uchar4 tex2Dgather(texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uchar") ;
__device__ __cudart_builtin__ char4 tex2Dgather(texture<char1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_char1") ;
__device__ __cudart_builtin__ uchar4 tex2Dgather(texture<uchar1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uchar1") ;
__device__ __cudart_builtin__ char4 tex2Dgather(texture<char2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_char2") ;
__device__ __cudart_builtin__ uchar4 tex2Dgather(texture<uchar2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uchar2") ;
__device__ __cudart_builtin__ char4 tex2Dgather(texture<char3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_char3") ;
__device__ __cudart_builtin__ uchar4 tex2Dgather(texture<uchar3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uchar3") ;
__device__ __cudart_builtin__ char4 tex2Dgather(texture<char4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_char4") ;
__device__ __cudart_builtin__ uchar4 tex2Dgather(texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uchar4") ;
__device__ __cudart_builtin__ short4 tex2Dgather(texture<signed short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_short") ;
__device__ __cudart_builtin__ ushort4 tex2Dgather(texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_ushort") ;
__device__ __cudart_builtin__ short4 tex2Dgather(texture<short1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_short1") ;
__device__ __cudart_builtin__ ushort4 tex2Dgather(texture<ushort1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_ushort1") ;
__device__ __cudart_builtin__ short4 tex2Dgather(texture<short2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_short2") ;
__device__ __cudart_builtin__ ushort4 tex2Dgather(texture<ushort2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_ushort2") ;
__device__ __cudart_builtin__ short4 tex2Dgather(texture<short3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_short3") ;
__device__ __cudart_builtin__ ushort4 tex2Dgather(texture<ushort3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_ushort3") ;
__device__ __cudart_builtin__ short4 tex2Dgather(texture<short4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_short4") ;
__device__ __cudart_builtin__ ushort4 tex2Dgather(texture<ushort4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_ushort4") ;
__device__ __cudart_builtin__ int4 tex2Dgather(texture<signed int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_int") ;
__device__ __cudart_builtin__ uint4 tex2Dgather(texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uint") ;
__device__ __cudart_builtin__ int4 tex2Dgather(texture<int1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_int1") ;
__device__ __cudart_builtin__ uint4 tex2Dgather(texture<uint1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uint1") ;
__device__ __cudart_builtin__ int4 tex2Dgather(texture<int2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_int2") ;
__device__ __cudart_builtin__ uint4 tex2Dgather(texture<uint2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uint2") ;
__device__ __cudart_builtin__ int4 tex2Dgather(texture<int3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_int3") ;
__device__ __cudart_builtin__ uint4 tex2Dgather(texture<uint3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uint3") ;
__device__ __cudart_builtin__ int4 tex2Dgather(texture<int4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_int4") ;
__device__ __cudart_builtin__ uint4 tex2Dgather(texture<uint4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_uint4") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<float, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_float") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<float1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_float1") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<float2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_float2") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<float3, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_float3") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<float4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, int comp = 0) asm("__tex2Dgather_float4") ;


__device__ __cudart_builtin__ float4 tex2Dgather(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_char") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<signed char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_schar") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_uchar") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<char1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_char1") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_uchar1") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<char2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_char2") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<uchar2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<char3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_char3") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<uchar3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_uchar3") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<char4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_uchar4") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<signed short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_short") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_ushort") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<short1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_short1") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<ushort1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_ushort1") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<short2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_short2") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<ushort2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<short3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_short3") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<ushort3, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_ushort3") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<short4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2Dgather(texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0) asm("__tex2Dgather_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1DLod(texture<T, cudaTextureType1D, cudaReadModeElementType>, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1DLod(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1DLod(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1DLod(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1DLod(texture<signed char, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_schar") ;
__device__ __cudart_builtin__ unsigned char tex1DLod(texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uchar") ;
__device__ __cudart_builtin__ char1 tex1DLod(texture<char1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_char1") ;
__device__ __cudart_builtin__ uchar1 tex1DLod(texture<uchar1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uchar1") ;
__device__ __cudart_builtin__ char2 tex1DLod(texture<char2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_char2") ;
__device__ __cudart_builtin__ uchar2 tex1DLod(texture<uchar2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uchar2") ;
__device__ __cudart_builtin__ char4 tex1DLod(texture<char4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_char4") ;
__device__ __cudart_builtin__ uchar4 tex1DLod(texture<uchar4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uchar4") ;

__device__ __cudart_builtin__ short tex1DLod(texture<short, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_short") ;
__device__ __cudart_builtin__ unsigned short tex1DLod(texture<unsigned short, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_ushort") ;
__device__ __cudart_builtin__ short1 tex1DLod(texture<short1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_short1") ;
__device__ __cudart_builtin__ ushort1 tex1DLod(texture<ushort1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_ushort1") ;
__device__ __cudart_builtin__ short2 tex1DLod(texture<short2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_short2") ;
__device__ __cudart_builtin__ ushort2 tex1DLod(texture<ushort2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_ushort2") ;
__device__ __cudart_builtin__ short4 tex1DLod(texture<short4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_short4") ;
__device__ __cudart_builtin__ ushort4 tex1DLod(texture<ushort4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_ushort4") ;

__device__ __cudart_builtin__ int tex1DLod(texture<int, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_int") ;
__device__ __cudart_builtin__ unsigned int tex1DLod(texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uint") ;
__device__ __cudart_builtin__ int1 tex1DLod(texture<int1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_int1") ;
__device__ __cudart_builtin__ uint1 tex1DLod(texture<uint1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uint1") ;
__device__ __cudart_builtin__ int2 tex1DLod(texture<int2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_int2") ;
__device__ __cudart_builtin__ uint2 tex1DLod(texture<uint2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uint2") ;
__device__ __cudart_builtin__ int4 tex1DLod(texture<int4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_int4") ;
__device__ __cudart_builtin__ uint4 tex1DLod(texture<uint4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1DLod(texture<long, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  int __tex1DLod_long_as_int(texture<long, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_long_as_int");
  return __tex1DLod_long_as_int(t, x, level);
}

static __device__ __forceinline__ unsigned long tex1DLod(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  unsigned __tex1DLod_ulong_as_uint(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_ulong_as_uint");
  return __tex1DLod_ulong_as_uint(t, x, level);
}

static __device__ __forceinline__ long1 tex1DLod(texture<long1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  int1 __tex1DLod_long1_as_int1(texture<long1, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_long1_as_int1");
  int1 v = __tex1DLod_long1_as_int1(t, x, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1DLod(texture<ulong1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  uint1 __tex1DLod_ulong1_as_uint1(texture<ulong1, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_ulong1_as_uint1");
  uint1 v = __tex1DLod_ulong1_as_uint1(t, x, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1DLod(texture<long2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  int2 __tex1DLod_long2_as_int2(texture<long2, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_long2_as_int2");
  int2 v = __tex1DLod_long2_as_int2(t, x, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1DLod(texture<ulong2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  uint2 __tex1DLod_ulong2_as_uint2(texture<ulong2, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_ulong2_as_uint2");
  uint2 v = __tex1DLod_ulong2_as_uint2(t, x, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1DLod(texture<long4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  int4 __tex1DLod_long4_as_int4(texture<long4, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_long4_as_int4");
  int4 v = __tex1DLod_long4_as_int4(t, x, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1DLod(texture<ulong4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level)
{
  __device__ __cudart_builtin__  uint4 __tex1DLod_ulong4_as_uint4(texture<ulong4, cudaTextureType1D, cudaReadModeElementType>, float, float)  asm("__tex1DLod_ulong4_as_uint4");
  uint4 v = __tex1DLod_ulong4_as_uint4(t, x, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1DLod(texture<float, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_float") ;
__device__ __cudart_builtin__ float1 tex1DLod(texture<float1, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_float1") ;
__device__ __cudart_builtin__ float2 tex1DLod(texture<float2, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_float2") ;
__device__ __cudart_builtin__ float4 tex1DLod(texture<float4, cudaTextureType1D, cudaReadModeElementType> t, float x, float level) asm("__tex1DLod_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1DLod(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1DLod(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1DLod(texture<signed char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1DLod(texture<unsigned char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1DLod(texture<char1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1DLod(texture<uchar1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1DLod(texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1DLod(texture<uchar2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1DLod(texture<char4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1DLod(texture<uchar4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_uchar4") ;
__device__ __cudart_builtin__ float tex1DLod(texture<short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_short") ;
__device__ __cudart_builtin__ float tex1DLod(texture<unsigned short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1DLod(texture<short1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1DLod(texture<ushort1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1DLod(texture<short2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1DLod(texture<ushort2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1DLod(texture<short4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1DLod(texture<ushort4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float level) asm("__tex1DLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2DLod(texture<T, cudaTextureType2D, cudaReadModeElementType>, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2DLod(texture<T, cudaTextureType2D, cudaReadModeNormalizedFloat>, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2DLod(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2DLod(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2DLod(texture<signed char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_schar") ;
__device__ __cudart_builtin__ unsigned char tex2DLod(texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uchar") ;
__device__ __cudart_builtin__ char1 tex2DLod(texture<char1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_char1") ;
__device__ __cudart_builtin__ uchar1 tex2DLod(texture<uchar1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uchar1") ;
__device__ __cudart_builtin__ char2 tex2DLod(texture<char2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_char2") ;
__device__ __cudart_builtin__ uchar2 tex2DLod(texture<uchar2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uchar2") ;
__device__ __cudart_builtin__ char4 tex2DLod(texture<char4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_char4") ;
__device__ __cudart_builtin__ uchar4 tex2DLod(texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uchar4") ;

__device__ __cudart_builtin__ short tex2DLod(texture<short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_short") ;
__device__ __cudart_builtin__ unsigned short tex2DLod(texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_ushort") ;
__device__ __cudart_builtin__ short1 tex2DLod(texture<short1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_short1") ;
__device__ __cudart_builtin__ ushort1 tex2DLod(texture<ushort1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_ushort1") ;
__device__ __cudart_builtin__ short2 tex2DLod(texture<short2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_short2") ;
__device__ __cudart_builtin__ ushort2 tex2DLod(texture<ushort2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_ushort2") ;
__device__ __cudart_builtin__ short4 tex2DLod(texture<short4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_short4") ;
__device__ __cudart_builtin__ ushort4 tex2DLod(texture<ushort4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_ushort4") ;

__device__ __cudart_builtin__ int tex2DLod(texture<int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_int") ;
__device__ __cudart_builtin__ unsigned int tex2DLod(texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uint") ;
__device__ __cudart_builtin__ int1 tex2DLod(texture<int1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_int1") ;
__device__ __cudart_builtin__ uint1 tex2DLod(texture<uint1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uint1") ;
__device__ __cudart_builtin__ int2 tex2DLod(texture<int2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_int2") ;
__device__ __cudart_builtin__ uint2 tex2DLod(texture<uint2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uint2") ;
__device__ __cudart_builtin__ int4 tex2DLod(texture<int4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_int4") ;
__device__ __cudart_builtin__ uint4 tex2DLod(texture<uint4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2DLod(texture<long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  int __tex2DLod_long_as_int(texture<long, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_long_as_int");
  return __tex2DLod_long_as_int(t, x, y, level);
}

static __device__ __forceinline__ unsigned long tex2DLod(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  unsigned int __tex2DLod_ulong_as_uint(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_ulong_as_uint");
  return __tex2DLod_ulong_as_uint(t, x, y, level);
}

static __device__ __forceinline__ long1 tex2DLod(texture<long1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  int1 __tex2DLod_long1_as_int1(texture<long1, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_long1_as_int1");
  int1 v = __tex2DLod_long1_as_int1(t, x, y, level);
  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2DLod(texture<ulong1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  uint1 __tex2DLod_ulong1_as_uint1(texture<ulong1, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_ulong1_as_uint1");
  uint1 v = __tex2DLod_ulong1_as_uint1(t, x, y, level);
  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2DLod(texture<long2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  int2 __tex2DLod_long2_as_int2(texture<long2, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_long2_as_int2");
  int2 v = __tex2DLod_long2_as_int2(t, x, y, level);
  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2DLod(texture<ulong2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  uint2 __tex2DLod_ulong2_as_uint2(texture<ulong2, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_ulong2_as_uint2");
  uint2 v = __tex2DLod_ulong2_as_uint2(t, x, y, level);
  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2DLod(texture<long4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  int4 __tex2DLod_long4_as_int4(texture<long4, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_long4_as_int4");
  int4 v = __tex2DLod_long4_as_int4(t, x, y, level);
  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2DLod(texture<ulong4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level)
{
  __device__ __cudart_builtin__  uint4 __tex2DLod_ulong4_as_uint4(texture<ulong4, cudaTextureType2D, cudaReadModeElementType>, float, float, float)  asm("__tex2DLod_ulong4_as_uint4");
  uint4 v = __tex2DLod_ulong4_as_uint4(t, x, y, level);
  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */

__device__ __cudart_builtin__ float tex2DLod(texture<float, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_float") ;
__device__ __cudart_builtin__ float1 tex2DLod(texture<float1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_float1") ;
__device__ __cudart_builtin__ float2 tex2DLod(texture<float2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_float2") ;
__device__ __cudart_builtin__ float4 tex2DLod(texture<float4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float level) asm("__tex2DLod_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2DLod(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2DLod(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2DLod(texture<signed char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2DLod(texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2DLod(texture<char1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2DLod(texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2DLod(texture<char2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2DLod(texture<uchar2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2DLod(texture<char4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2DLod(texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_uchar4") ;
__device__ __cudart_builtin__ float tex2DLod(texture<short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_short") ;
__device__ __cudart_builtin__ float tex2DLod(texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2DLod(texture<short1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2DLod(texture<ushort1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2DLod(texture<short2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2DLod(texture<ushort2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2DLod(texture<short4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2DLod(texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float level) asm("__tex2DLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */


#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1DLayeredLod(texture<T, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1DLayeredLod(texture<T, cudaTextureType1DLayered, cudaReadModeNormalizedFloat>, float, int, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1DLayeredLod(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1DLayeredLod(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1DLayeredLod(texture<signed char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_schar") ;
__device__ __cudart_builtin__ unsigned char tex1DLayeredLod(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uchar") ;
__device__ __cudart_builtin__ char1 tex1DLayeredLod(texture<char1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_char1") ;
__device__ __cudart_builtin__ uchar1 tex1DLayeredLod(texture<uchar1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uchar1") ;
__device__ __cudart_builtin__ char2 tex1DLayeredLod(texture<char2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_char2") ;
__device__ __cudart_builtin__ uchar2 tex1DLayeredLod(texture<uchar2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uchar2") ;
__device__ __cudart_builtin__ char4 tex1DLayeredLod(texture<char4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_char4") ;
__device__ __cudart_builtin__ uchar4 tex1DLayeredLod(texture<uchar4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uchar4") ;

__device__ __cudart_builtin__ short tex1DLayeredLod(texture<short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_short") ;
__device__ __cudart_builtin__ unsigned short tex1DLayeredLod(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_ushort") ;
__device__ __cudart_builtin__ short1 tex1DLayeredLod(texture<short1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_short1") ;
__device__ __cudart_builtin__ ushort1 tex1DLayeredLod(texture<ushort1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_ushort1") ;
__device__ __cudart_builtin__ short2 tex1DLayeredLod(texture<short2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_short2") ;
__device__ __cudart_builtin__ ushort2 tex1DLayeredLod(texture<ushort2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_ushort2") ;
__device__ __cudart_builtin__ short4 tex1DLayeredLod(texture<short4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_short4") ;
__device__ __cudart_builtin__ ushort4 tex1DLayeredLod(texture<ushort4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_ushort4") ;

__device__ __cudart_builtin__ int tex1DLayeredLod(texture<int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_int") ;
__device__ __cudart_builtin__ unsigned int tex1DLayeredLod(texture<unsigned int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uint") ;
__device__ __cudart_builtin__ int1 tex1DLayeredLod(texture<int1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_int1") ;
__device__ __cudart_builtin__ uint1 tex1DLayeredLod(texture<uint1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uint1") ;
__device__ __cudart_builtin__ int2 tex1DLayeredLod(texture<int2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_int2") ;
__device__ __cudart_builtin__ uint2 tex1DLayeredLod(texture<uint2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uint2") ;
__device__ __cudart_builtin__ int4 tex1DLayeredLod(texture<int4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_int4") ;
__device__ __cudart_builtin__ uint4 tex1DLayeredLod(texture<uint4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1DLayeredLod(texture<long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  int __tex1DLayeredLod_long_as_int(texture<long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_long_as_int");
  return __tex1DLayeredLod_long_as_int(t, x, layer, level);
}

static __device__ __forceinline__ unsigned long tex1DLayeredLod(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  unsigned int __tex1DLayeredLod_ulong_as_uint(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_ulong_as_uint");
  return __tex1DLayeredLod_ulong_as_uint(t, x, layer, level);
}

static __device__ __forceinline__ long1 tex1DLayeredLod(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  int1 __tex1DLayeredLod_long1_as_int1(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_long1_as_int1");
  int1 v = __tex1DLayeredLod_long1_as_int1(t, x, layer, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1DLayeredLod(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  uint1 __tex1DLayeredLod_ulong1_as_uint1(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_ulong1_as_uint1");
  uint1 v = __tex1DLayeredLod_ulong1_as_uint1(t, x, layer, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1DLayeredLod(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  int2 __tex1DLayeredLod_long2_as_int2(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_long2_as_int2");
  int2 v = __tex1DLayeredLod_long2_as_int2(t, x, layer, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1DLayeredLod(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  uint2 __tex1DLayeredLod_ulong2_as_uint2(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_ulong2_as_uint2");
  uint2 v = __tex1DLayeredLod_ulong2_as_uint2(t, x, layer, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1DLayeredLod(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  int4 __tex1DLayeredLod_long4_as_int4(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_long4_as_int4");
  int4 v = __tex1DLayeredLod_long4_as_int4(t, x, layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1DLayeredLod(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level)
{
  __device__ __cudart_builtin__  uint4 __tex1DLayeredLod_ulong4_as_uint4(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float)  asm("__tex1DLayeredLod_ulong4_as_uint4");
  uint4 v = __tex1DLayeredLod_ulong4_as_uint4(t, x, layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<float, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_float") ;
__device__ __cudart_builtin__ float1 tex1DLayeredLod(texture<float1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_float1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredLod(texture<float2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_float2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredLod(texture<float4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float level) asm("__tex1DLayeredLod_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<signed char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1DLayeredLod(texture<char1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1DLayeredLod(texture<uchar1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredLod(texture<char2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1DLayeredLod(texture<uchar2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredLod(texture<char4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1DLayeredLod(texture<uchar4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex1DLayeredLod(texture<short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_short") ;
__device__ __cudart_builtin__ float tex1DLayeredLod(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1DLayeredLod(texture<short1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1DLayeredLod(texture<ushort1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredLod(texture<short2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1DLayeredLod(texture<ushort2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredLod(texture<short4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1DLayeredLod(texture<ushort4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float level) asm("__tex1DLayeredLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2DLayeredLod(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2DLayeredLod(texture<T, cudaTextureType2DLayered, cudaReadModeNormalizedFloat>, float, float, int, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2DLayeredLod(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2DLayeredLod(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2DLayeredLod(texture<signed char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_schar") ;
__device__ __cudart_builtin__ unsigned char tex2DLayeredLod(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uchar") ;
__device__ __cudart_builtin__ char1 tex2DLayeredLod(texture<char1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_char1") ;
__device__ __cudart_builtin__ uchar1 tex2DLayeredLod(texture<uchar1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uchar1") ;
__device__ __cudart_builtin__ char2 tex2DLayeredLod(texture<char2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_char2") ;
__device__ __cudart_builtin__ uchar2 tex2DLayeredLod(texture<uchar2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uchar2") ;
__device__ __cudart_builtin__ char4 tex2DLayeredLod(texture<char4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_char4") ;
__device__ __cudart_builtin__ uchar4 tex2DLayeredLod(texture<uchar4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uchar4") ;

__device__ __cudart_builtin__ short tex2DLayeredLod(texture<short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_short") ;
__device__ __cudart_builtin__ unsigned short tex2DLayeredLod(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_ushort") ;
__device__ __cudart_builtin__ short1 tex2DLayeredLod(texture<short1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_short1") ;
__device__ __cudart_builtin__ ushort1 tex2DLayeredLod(texture<ushort1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_ushort1") ;
__device__ __cudart_builtin__ short2 tex2DLayeredLod(texture<short2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_short2") ;
__device__ __cudart_builtin__ ushort2 tex2DLayeredLod(texture<ushort2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_ushort2") ;
__device__ __cudart_builtin__ short4 tex2DLayeredLod(texture<short4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_short4") ;
__device__ __cudart_builtin__ ushort4 tex2DLayeredLod(texture<ushort4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_ushort4") ;

__device__ __cudart_builtin__ int tex2DLayeredLod(texture<int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_int") ;
__device__ __cudart_builtin__ unsigned int tex2DLayeredLod(texture<unsigned int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uint") ;
__device__ __cudart_builtin__ int1 tex2DLayeredLod(texture<int1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_int1") ;
__device__ __cudart_builtin__ uint1 tex2DLayeredLod(texture<uint1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uint1") ;
__device__ __cudart_builtin__ int2 tex2DLayeredLod(texture<int2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_int2") ;
__device__ __cudart_builtin__ uint2 tex2DLayeredLod(texture<uint2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uint2") ;
__device__ __cudart_builtin__ int4 tex2DLayeredLod(texture<int4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_int4") ;
__device__ __cudart_builtin__ uint4 tex2DLayeredLod(texture<uint4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2DLayeredLod(texture<long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  int __tex2DLayeredLod_long_as_int(texture<long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_long_as_int");
  return __tex2DLayeredLod_long_as_int(t, x, y, layer, level);
}

static __device__ __forceinline__ unsigned long tex2DLayeredLod(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  unsigned int __tex2DLayeredLod_ulong_as_uint(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_ulong_as_uint");
  return __tex2DLayeredLod_ulong_as_uint(t, x, y, layer, level);
}

static __device__ __forceinline__ long1 tex2DLayeredLod(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  int1 __tex2DLayeredLod_long1_as_int1(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_long1_as_int1");
  int1 v = __tex2DLayeredLod_long1_as_int1(t, x, y, layer, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2DLayeredLod(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  uint1 __tex2DLayeredLod_ulong1_as_uint1(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_ulong1_as_uint1");
  uint1 v = __tex2DLayeredLod_ulong1_as_uint1(t, x, y, layer, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2DLayeredLod(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  int2 __tex2DLayeredLod_long2_as_int2(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_long2_as_int2");
  int2 v = __tex2DLayeredLod_long2_as_int2(t, x, y, layer, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2DLayeredLod(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  uint2 __tex2DLayeredLod_ulong2_as_uint2(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_ulong2_as_uint2");
  uint2 v = __tex2DLayeredLod_ulong2_as_uint2(t, x, y, layer, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2DLayeredLod(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  int4 __tex2DLayeredLod_long4_as_int4(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_long4_as_int4");
  int4 v = __tex2DLayeredLod_long4_as_int4(t, x, y, layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2DLayeredLod(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level)
{
  __device__ __cudart_builtin__  uint4 __tex2DLayeredLod_ulong4_as_uint4(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float)  asm("__tex2DLayeredLod_ulong4_as_uint4");
  uint4 v = __tex2DLayeredLod_ulong4_as_uint4(t, x, y, layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_float") ;
__device__ __cudart_builtin__ float1 tex2DLayeredLod(texture<float1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_float1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredLod(texture<float2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_float2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredLod(texture<float4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<signed char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2DLayeredLod(texture<char1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2DLayeredLod(texture<uchar1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredLod(texture<char2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2DLayeredLod(texture<uchar2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredLod(texture<char4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2DLayeredLod(texture<uchar4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex2DLayeredLod(texture<short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_short") ;
__device__ __cudart_builtin__ float tex2DLayeredLod(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2DLayeredLod(texture<short1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2DLayeredLod(texture<ushort1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredLod(texture<short2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2DLayeredLod(texture<ushort2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredLod(texture<short4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2DLayeredLod(texture<ushort4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level) asm("__tex2DLayeredLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex3DLod(texture<T, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex3DLod(texture<T, cudaTextureType3D, cudaReadModeNormalizedFloat>, float, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex3DLod(texture<char, cudaTextureType3D,  cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex3DLod(texture<char, cudaTextureType3D,  cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex3DLod(texture<signed char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_schar") ;
__device__ __cudart_builtin__ unsigned char tex3DLod(texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uchar") ;
__device__ __cudart_builtin__ char1 tex3DLod(texture<char1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_char1") ;
__device__ __cudart_builtin__ uchar1 tex3DLod(texture<uchar1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uchar1") ;
__device__ __cudart_builtin__ char2 tex3DLod(texture<char2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_char2") ;
__device__ __cudart_builtin__ uchar2 tex3DLod(texture<uchar2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uchar2") ;
__device__ __cudart_builtin__ char4 tex3DLod(texture<char4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_char4") ;
__device__ __cudart_builtin__ uchar4 tex3DLod(texture<uchar4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uchar4") ;

__device__ __cudart_builtin__ short tex3DLod(texture<short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_short") ;
__device__ __cudart_builtin__ unsigned short tex3DLod(texture<unsigned short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_ushort") ;
__device__ __cudart_builtin__ short1 tex3DLod(texture<short1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_short1") ;
__device__ __cudart_builtin__ ushort1 tex3DLod(texture<ushort1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_ushort1") ;
__device__ __cudart_builtin__ short2 tex3DLod(texture<short2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_short2") ;
__device__ __cudart_builtin__ ushort2 tex3DLod(texture<ushort2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_ushort2") ;
__device__ __cudart_builtin__ short4 tex3DLod(texture<short4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_short4") ;
__device__ __cudart_builtin__ ushort4 tex3DLod(texture<ushort4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_ushort4") ;

__device__ __cudart_builtin__ int tex3DLod(texture<int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_int") ;
__device__ __cudart_builtin__ unsigned int tex3DLod(texture<unsigned int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uint") ;
__device__ __cudart_builtin__ int1 tex3DLod(texture<int1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_int1") ;
__device__ __cudart_builtin__ uint1 tex3DLod(texture<uint1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uint1") ;
__device__ __cudart_builtin__ int2 tex3DLod(texture<int2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_int2") ;
__device__ __cudart_builtin__ uint2 tex3DLod(texture<uint2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uint2") ;
__device__ __cudart_builtin__ int4 tex3DLod(texture<int4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_int4") ;
__device__ __cudart_builtin__ uint4 tex3DLod(texture<uint4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex3DLod(texture<long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int __tex3DLod_long_as_int(texture<long, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_long_as_int");
  return __tex3DLod_long_as_int(t, x, y, z, level);
}

static __device__ __forceinline__ unsigned long tex3DLod(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  unsigned int __tex3DLod_ulong_as_uint(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_ulong_as_uint");
  return __tex3DLod_ulong_as_uint(t, x, y, z, level);
}

static __device__ __forceinline__ long1 tex3DLod(texture<long1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int1 __tex3DLod_long1_as_int1(texture<long1, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_long1_as_int1");
  int1 v = __tex3DLod_long1_as_int1(t, x, y, z, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex3DLod(texture<ulong1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint1 __tex3DLod_ulong1_as_uint1(texture<ulong1, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_ulong1_as_uint1");
  uint1 v = __tex3DLod_ulong1_as_uint1(t, x, y, z, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex3DLod(texture<long2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int2 __tex3DLod_long2_as_int2(texture<long2, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_long2_as_int2");
  int2 v = __tex3DLod_long2_as_int2(t, x, y, z, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex3DLod(texture<ulong2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint2 __tex3DLod_ulong2_as_uint2(texture<ulong2, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_ulong2_as_uint2");
  uint2 v = __tex3DLod_ulong2_as_uint2(t, x, y, z, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex3DLod(texture<long4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int4 __tex3DLod_long4_as_int4(texture<long4, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_long4_as_int4");
  int4 v = __tex3DLod_long4_as_int4(t, x, y, z, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex3DLod(texture<ulong4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint4 __tex3DLod_ulong4_as_uint4(texture<ulong4, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float)  asm("__tex3DLod_ulong4_as_uint4");
  uint4 v = __tex3DLod_ulong4_as_uint4(t, x, y, z, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex3DLod(texture<float, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_float") ;
__device__ __cudart_builtin__ float1 tex3DLod(texture<float1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_float1") ;
__device__ __cudart_builtin__ float2 tex3DLod(texture<float2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_float2") ;
__device__ __cudart_builtin__ float4 tex3DLod(texture<float4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__tex3DLod_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex3DLod(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex3DLod(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex3DLod(texture<signed char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_schar") ;
__device__ __cudart_builtin__ float tex3DLod(texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex3DLod(texture<char1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex3DLod(texture<uchar1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex3DLod(texture<char2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex3DLod(texture<uchar2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex3DLod(texture<char4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex3DLod(texture<uchar4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex3DLod(texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_short") ;
__device__ __cudart_builtin__ float tex3DLod(texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex3DLod(texture<short1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex3DLod(texture<ushort1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex3DLod(texture<short2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex3DLod(texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex3DLod(texture<short4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex3DLod(texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__tex3DLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T texCubemapLod(texture<T, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type texCubemapLod(texture<T, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat>, float, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char texCubemapLod(texture<char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char texCubemapLod(texture<char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char texCubemapLod(texture<signed char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_schar") ;
__device__ __cudart_builtin__ unsigned char texCubemapLod(texture<unsigned char, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uchar") ;
__device__ __cudart_builtin__ char1 texCubemapLod(texture<char1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_char1") ;
__device__ __cudart_builtin__ uchar1 texCubemapLod(texture<uchar1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uchar1") ;
__device__ __cudart_builtin__ char2 texCubemapLod(texture<char2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_char2") ;
__device__ __cudart_builtin__ uchar2 texCubemapLod(texture<uchar2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uchar2") ;
__device__ __cudart_builtin__ char4 texCubemapLod(texture<char4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_char4") ;
__device__ __cudart_builtin__ uchar4 texCubemapLod(texture<uchar4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uchar4") ;

__device__ __cudart_builtin__ short texCubemapLod(texture<short, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_short") ;
__device__ __cudart_builtin__ unsigned short texCubemapLod(texture<unsigned short, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_ushort") ;
__device__ __cudart_builtin__ short1 texCubemapLod(texture<short1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_short1") ;
__device__ __cudart_builtin__ ushort1 texCubemapLod(texture<ushort1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_ushort1") ;
__device__ __cudart_builtin__ short2 texCubemapLod(texture<short2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_short2") ;
__device__ __cudart_builtin__ ushort2 texCubemapLod(texture<ushort2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_ushort2") ;
__device__ __cudart_builtin__ short4 texCubemapLod(texture<short4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_short4") ;
__device__ __cudart_builtin__ ushort4 texCubemapLod(texture<ushort4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_ushort4") ;

__device__ __cudart_builtin__ int texCubemapLod(texture<int, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_int") ;
__device__ __cudart_builtin__ unsigned int texCubemapLod(texture<unsigned int, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uint") ;
__device__ __cudart_builtin__ int1 texCubemapLod(texture<int1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_int1") ;
__device__ __cudart_builtin__ uint1 texCubemapLod(texture<uint1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uint1") ;
__device__ __cudart_builtin__ int2 texCubemapLod(texture<int2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_int2") ;
__device__ __cudart_builtin__ uint2 texCubemapLod(texture<uint2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uint2") ;
__device__ __cudart_builtin__ int4 texCubemapLod(texture<int4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_int4") ;
__device__ __cudart_builtin__ uint4 texCubemapLod(texture<uint4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long texCubemapLod(texture<long, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int __texCubemapLod_long_as_int(texture<long, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_long_as_int");
  return __texCubemapLod_long_as_int(t, x, y, z, level);
}

static __device__ __forceinline__ unsigned long texCubemapLod(texture<unsigned long, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  unsigned int __texCubemapLod_ulong_as_uint(texture<unsigned long, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_ulong_as_uint");
  return __texCubemapLod_ulong_as_uint(t, x, y, z, level);
}

static __device__ __forceinline__ long1 texCubemapLod(texture<long1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int1 __texCubemapLod_long1_as_int1(texture<long1, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_long1_as_int1");
  int1 v = __texCubemapLod_long1_as_int1(t, x, y, z, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 texCubemapLod(texture<ulong1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint1 __texCubemapLod_ulong1_as_uint1(texture<ulong1, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_ulong1_as_uint1");
  uint1 v = __texCubemapLod_ulong1_as_uint1(t, x, y, z, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 texCubemapLod(texture<long2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int2 __texCubemapLod_long2_as_int2(texture<long2, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_long2_as_int2");
  int2 v = __texCubemapLod_long2_as_int2(t, x, y, z, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 texCubemapLod(texture<ulong2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint2 __texCubemapLod_ulong2_as_uint2(texture<ulong2, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_ulong2_as_uint2");
  uint2 v = __texCubemapLod_ulong2_as_uint2(t, x, y, z, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 texCubemapLod(texture<long4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  int4 __texCubemapLod_long4_as_int4(texture<long4, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_long4_as_int4");
  int4 v = __texCubemapLod_long4_as_int4(t, x, y, z, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 texCubemapLod(texture<ulong4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level)
{
  __device__ __cudart_builtin__  uint4 __texCubemapLod_ulong4_as_uint4(texture<ulong4, cudaTextureTypeCubemap, cudaReadModeElementType>, float, float, float, float)  asm("__texCubemapLod_ulong4_as_uint4");
  uint4 v = __texCubemapLod_ulong4_as_uint4(t, x, y, z, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float texCubemapLod(texture<float, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_float") ;
__device__ __cudart_builtin__ float1 texCubemapLod(texture<float1, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_float1") ;
__device__ __cudart_builtin__ float2 texCubemapLod(texture<float2, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_float2") ;
__device__ __cudart_builtin__ float4 texCubemapLod(texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> t, float x, float y, float z, float level) asm("__texCubemapLod_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float texCubemapLod(texture<char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float texCubemapLod(texture<char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float texCubemapLod(texture<signed char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_schar") ;
__device__ __cudart_builtin__ float texCubemapLod(texture<unsigned char, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 texCubemapLod(texture<char1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 texCubemapLod(texture<uchar1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 texCubemapLod(texture<char2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 texCubemapLod(texture<uchar2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 texCubemapLod(texture<char4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 texCubemapLod(texture<uchar4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_uchar4") ;

__device__ __cudart_builtin__ float texCubemapLod(texture<short, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_short") ;
__device__ __cudart_builtin__ float texCubemapLod(texture<unsigned short, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 texCubemapLod(texture<short1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 texCubemapLod(texture<ushort1, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 texCubemapLod(texture<short2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 texCubemapLod(texture<ushort2, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 texCubemapLod(texture<short4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 texCubemapLod(texture<ushort4, cudaTextureTypeCubemap, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level) asm("__texCubemapLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T texCubemapLayeredLod(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type texCubemapLayeredLod(texture<T, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat>, float, float, float, int, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char texCubemapLayeredLod(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char texCubemapLayeredLod(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char texCubemapLayeredLod(texture<signed char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_schar") ;
__device__ __cudart_builtin__ unsigned char texCubemapLayeredLod(texture<unsigned char, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uchar") ;
__device__ __cudart_builtin__ char1 texCubemapLayeredLod(texture<char1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_char1") ;
__device__ __cudart_builtin__ uchar1 texCubemapLayeredLod(texture<uchar1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uchar1") ;
__device__ __cudart_builtin__ char2 texCubemapLayeredLod(texture<char2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_char2") ;
__device__ __cudart_builtin__ uchar2 texCubemapLayeredLod(texture<uchar2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uchar2") ;
__device__ __cudart_builtin__ char4 texCubemapLayeredLod(texture<char4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_char4") ;
__device__ __cudart_builtin__ uchar4 texCubemapLayeredLod(texture<uchar4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uchar4") ;

__device__ __cudart_builtin__ short texCubemapLayeredLod(texture<short, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_short") ;
__device__ __cudart_builtin__ unsigned short texCubemapLayeredLod(texture<unsigned short, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_ushort") ;
__device__ __cudart_builtin__ short1 texCubemapLayeredLod(texture<short1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_short1") ;
__device__ __cudart_builtin__ ushort1 texCubemapLayeredLod(texture<ushort1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_ushort1") ;
__device__ __cudart_builtin__ short2 texCubemapLayeredLod(texture<short2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_short2") ;
__device__ __cudart_builtin__ ushort2 texCubemapLayeredLod(texture<ushort2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_ushort2") ;
__device__ __cudart_builtin__ short4 texCubemapLayeredLod(texture<short4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_short4") ;
__device__ __cudart_builtin__ ushort4 texCubemapLayeredLod(texture<ushort4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_ushort4") ;

__device__ __cudart_builtin__ int texCubemapLayeredLod(texture<int, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_int") ;
__device__ __cudart_builtin__ unsigned int texCubemapLayeredLod(texture<unsigned int, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uint") ;
__device__ __cudart_builtin__ int1 texCubemapLayeredLod(texture<int1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_int1") ;
__device__ __cudart_builtin__ uint1 texCubemapLayeredLod(texture<uint1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uint1") ;
__device__ __cudart_builtin__ int2 texCubemapLayeredLod(texture<int2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_int2") ;
__device__ __cudart_builtin__ uint2 texCubemapLayeredLod(texture<uint2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uint2") ;
__device__ __cudart_builtin__ int4 texCubemapLayeredLod(texture<int4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_int4") ;
__device__ __cudart_builtin__ uint4 texCubemapLayeredLod(texture<uint4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long texCubemapLayeredLod(texture<long, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  int __texCubemapLayeredLod_long_as_int(texture<long, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_long_as_int");
  return  __texCubemapLayeredLod_long_as_int(t, x, y, z, layer, level);
}

static __device__ __forceinline__ unsigned long texCubemapLayeredLod(texture<unsigned long, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  unsigned int __texCubemapLayeredLod_ulong_as_uint(texture<unsigned long, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_ulong_as_uint");
  return  __texCubemapLayeredLod_ulong_as_uint(t, x, y, z, layer, level);
}

static __device__ __forceinline__ long1 texCubemapLayeredLod(texture<long1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  int1 __texCubemapLayeredLod_long1_as_int1(texture<long1, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_long1_as_int1");
  int1 v = __texCubemapLayeredLod_long1_as_int1(t, x, y, z, layer, level);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 texCubemapLayeredLod(texture<ulong1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  uint1 __texCubemapLayeredLod_ulong1_as_uint1(texture<ulong1, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_ulong1_as_uint1");
  uint1 v = __texCubemapLayeredLod_ulong1_as_uint1(t, x, y, z, layer, level);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 texCubemapLayeredLod(texture<long2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  int2 __texCubemapLayeredLod_long2_as_int2(texture<long2, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_long2_as_int2");
  int2 v = __texCubemapLayeredLod_long2_as_int2(t, x, y, z, layer, level);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 texCubemapLayeredLod(texture<ulong2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  uint2 __texCubemapLayeredLod_ulong2_as_uint2(texture<ulong2, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_ulong2_as_uint2");
  uint2 v = __texCubemapLayeredLod_ulong2_as_uint2(t, x, y, z, layer, level);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 texCubemapLayeredLod(texture<long4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  int4 __texCubemapLayeredLod_long4_as_int4(texture<long4, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_long4_as_int4");
  int4 v = __texCubemapLayeredLod_long4_as_int4(t, x, y, z, layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 texCubemapLayeredLod(texture<ulong4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  __device__ __cudart_builtin__  uint4 __texCubemapLayeredLod_ulong4_as_uint4(texture<ulong4, cudaTextureTypeCubemapLayered, cudaReadModeElementType>, float, float, float, int, float)  asm("__texCubemapLayeredLod_ulong4_as_uint4");
  uint4 v = __texCubemapLayeredLod_ulong4_as_uint4(t, x, y, z, layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<float, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_float") ;
__device__ __cudart_builtin__ float1 texCubemapLayeredLod(texture<float1, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_float1") ;
__device__ __cudart_builtin__ float2 texCubemapLayeredLod(texture<float2, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_float2") ;
__device__ __cudart_builtin__ float4 texCubemapLayeredLod(texture<float4, cudaTextureTypeCubemapLayered, cudaReadModeElementType> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<signed char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_schar") ;
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<unsigned char, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 texCubemapLayeredLod(texture<char1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_char1") ;
__device__ __cudart_builtin__ float1 texCubemapLayeredLod(texture<uchar1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 texCubemapLayeredLod(texture<char2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_char2") ;
__device__ __cudart_builtin__ float2 texCubemapLayeredLod(texture<uchar2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 texCubemapLayeredLod(texture<char4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_char4") ;
__device__ __cudart_builtin__ float4 texCubemapLayeredLod(texture<uchar4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_uchar4") ;

__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<short, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_short") ;
__device__ __cudart_builtin__ float texCubemapLayeredLod(texture<unsigned short, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 texCubemapLayeredLod(texture<short1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_short1") ;
__device__ __cudart_builtin__ float1 texCubemapLayeredLod(texture<ushort1, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 texCubemapLayeredLod(texture<short2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_short2") ;
__device__ __cudart_builtin__ float2 texCubemapLayeredLod(texture<ushort2, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 texCubemapLayeredLod(texture<short4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_short4") ;
__device__ __cudart_builtin__ float4 texCubemapLayeredLod(texture<ushort4, cudaTextureTypeCubemapLayered, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level) asm("__texCubemapLayeredLod_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1DGrad(texture<T, cudaTextureType1D, cudaReadModeElementType>, float, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1DGrad(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>, float, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1DGrad(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1DGrad(texture<char, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1DGrad(texture<signed char, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_schar") ;
__device__ __cudart_builtin__ unsigned char tex1DGrad(texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uchar") ;
__device__ __cudart_builtin__ char1 tex1DGrad(texture<char1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_char1") ;
__device__ __cudart_builtin__ uchar1 tex1DGrad(texture<uchar1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uchar1") ;
__device__ __cudart_builtin__ char2 tex1DGrad(texture<char2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_char2") ;
__device__ __cudart_builtin__ uchar2 tex1DGrad(texture<uchar2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uchar2") ;
__device__ __cudart_builtin__ char4 tex1DGrad(texture<char4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_char4") ;
__device__ __cudart_builtin__ uchar4 tex1DGrad(texture<uchar4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uchar4") ;

__device__ __cudart_builtin__ short tex1DGrad(texture<short, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_short") ;
__device__ __cudart_builtin__ unsigned short tex1DGrad(texture<unsigned short, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_ushort") ;
__device__ __cudart_builtin__ short1 tex1DGrad(texture<short1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_short1") ;
__device__ __cudart_builtin__ ushort1 tex1DGrad(texture<ushort1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_ushort1") ;
__device__ __cudart_builtin__ short2 tex1DGrad(texture<short2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_short2") ;
__device__ __cudart_builtin__ ushort2 tex1DGrad(texture<ushort2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_ushort2") ;
__device__ __cudart_builtin__ short4 tex1DGrad(texture<short4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_short4") ;
__device__ __cudart_builtin__ ushort4 tex1DGrad(texture<ushort4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_ushort4") ;

__device__ __cudart_builtin__ int tex1DGrad(texture<int, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_int") ;
__device__ __cudart_builtin__ unsigned int tex1DGrad(texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uint") ;
__device__ __cudart_builtin__ int1 tex1DGrad(texture<int1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_int1") ;
__device__ __cudart_builtin__ uint1 tex1DGrad(texture<uint1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uint1") ;
__device__ __cudart_builtin__ int2 tex1DGrad(texture<int2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_int2") ;
__device__ __cudart_builtin__ uint2 tex1DGrad(texture<uint2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uint2") ;
__device__ __cudart_builtin__ int4 tex1DGrad(texture<int4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_int4") ;
__device__ __cudart_builtin__ uint4 tex1DGrad(texture<uint4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1DGrad(texture<long, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int __tex1DGrad_long_as_int(texture<long, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_long_as_int");
  return __tex1DGrad_long_as_int(t, x, dPdx, dPdy);
}

static __device__ __forceinline__ unsigned long tex1DGrad(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  unsigned __tex1DGrad_ulong_as_uint(texture<unsigned long, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_ulong_as_uint");
  return __tex1DGrad_ulong_as_uint(t, x, dPdx, dPdy);
}

static __device__ __forceinline__ long1 tex1DGrad(texture<long1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int1 __tex1DGrad_long1_as_int1(texture<long1, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_long1_as_int1");
  int1 v = __tex1DGrad_long1_as_int1(t, x, dPdx, dPdy);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1DGrad(texture<ulong1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint1 __tex1DGrad_ulong1_as_uint1(texture<ulong1, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_ulong1_as_uint1");
  uint1 v = __tex1DGrad_ulong1_as_uint1(t, x, dPdx, dPdy);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1DGrad(texture<long2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int2 __tex1DGrad_long2_as_int2(texture<long2, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_long2_as_int2");
  int2 v = __tex1DGrad_long2_as_int2(t, x, dPdx, dPdy);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1DGrad(texture<ulong2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint2 __tex1DGrad_ulong2_as_uint2(texture<ulong2, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_ulong2_as_uint2");
  uint2 v = __tex1DGrad_ulong2_as_uint2(t, x, dPdx, dPdy);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1DGrad(texture<long4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int4 __tex1DGrad_long4_as_int4(texture<long4, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_long4_as_int4");
  int4 v = __tex1DGrad_long4_as_int4(t, x, dPdx, dPdy);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1DGrad(texture<ulong4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint4 __tex1DGrad_ulong4_as_uint4(texture<ulong4, cudaTextureType1D, cudaReadModeElementType>, float, float, float)  asm("__tex1DGrad_ulong4_as_uint4");
  uint4 v = __tex1DGrad_ulong4_as_uint4(t, x, dPdx, dPdy);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1DGrad(texture<float, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_float") ;
__device__ __cudart_builtin__ float1 tex1DGrad(texture<float1, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_float1") ;
__device__ __cudart_builtin__ float2 tex1DGrad(texture<float2, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_float2") ;
__device__ __cudart_builtin__ float4 tex1DGrad(texture<float4, cudaTextureType1D, cudaReadModeElementType> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1DGrad(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1DGrad(texture<char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1DGrad(texture<signed char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1DGrad(texture<unsigned char, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1DGrad(texture<char1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1DGrad(texture<uchar1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1DGrad(texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1DGrad(texture<uchar2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1DGrad(texture<char4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1DGrad(texture<uchar4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex1DGrad(texture<short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_short") ;
__device__ __cudart_builtin__ float tex1DGrad(texture<unsigned short, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1DGrad(texture<short1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1DGrad(texture<ushort1, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1DGrad(texture<short2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1DGrad(texture<ushort2, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1DGrad(texture<short4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1DGrad(texture<ushort4, cudaTextureType1D, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy) asm("__tex1DGrad_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2DGrad(texture<T, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2DGrad(texture<T, cudaTextureType2D, cudaReadModeNormalizedFloat>, float, float, float2, float2) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2DGrad(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2DGrad(texture<char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2DGrad(texture<signed char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_schar") ;
__device__ __cudart_builtin__ unsigned char tex2DGrad(texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uchar") ;
__device__ __cudart_builtin__ char1 tex2DGrad(texture<char1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_char1") ;
__device__ __cudart_builtin__ uchar1 tex2DGrad(texture<uchar1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uchar1") ;
__device__ __cudart_builtin__ char2 tex2DGrad(texture<char2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_char2") ;
__device__ __cudart_builtin__ uchar2 tex2DGrad(texture<uchar2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uchar2") ;
__device__ __cudart_builtin__ char4 tex2DGrad(texture<char4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_char4") ;
__device__ __cudart_builtin__ uchar4 tex2DGrad(texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uchar4") ;

__device__ __cudart_builtin__ short tex2DGrad(texture<short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_short") ;
__device__ __cudart_builtin__ unsigned short tex2DGrad(texture<unsigned short, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_ushort") ;
__device__ __cudart_builtin__ short1 tex2DGrad(texture<short1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_short1") ;
__device__ __cudart_builtin__ ushort1 tex2DGrad(texture<ushort1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_ushort1") ;
__device__ __cudart_builtin__ short2 tex2DGrad(texture<short2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_short2") ;
__device__ __cudart_builtin__ ushort2 tex2DGrad(texture<ushort2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_ushort2") ;
__device__ __cudart_builtin__ short4 tex2DGrad(texture<short4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_short4") ;
__device__ __cudart_builtin__ ushort4 tex2DGrad(texture<ushort4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_ushort4") ;

__device__ __cudart_builtin__ int tex2DGrad(texture<int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_int") ;
__device__ __cudart_builtin__ unsigned int tex2DGrad(texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uint") ;
__device__ __cudart_builtin__ int1 tex2DGrad(texture<int1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_int1") ;
__device__ __cudart_builtin__ uint1 tex2DGrad(texture<uint1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uint1") ;
__device__ __cudart_builtin__ int2 tex2DGrad(texture<int2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_int2") ;
__device__ __cudart_builtin__ uint2 tex2DGrad(texture<uint2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uint2") ;
__device__ __cudart_builtin__ int4 tex2DGrad(texture<int4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_int4") ;
__device__ __cudart_builtin__ uint4 tex2DGrad(texture<uint4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2DGrad(texture<long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int __tex2DGrad_long_as_int(texture<long, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_long_as_int");
  return __tex2DGrad_long_as_int(t, x, y, dPdx, dPdy);
}

static __device__ __forceinline__ unsigned long tex2DGrad(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__ unsigned int __tex2DGrad_ulong_as_uint(texture<unsigned long, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_ulong_as_uint");
  return __tex2DGrad_ulong_as_uint(t, x, y, dPdx, dPdy);
}

static __device__ __forceinline__ long1 tex2DGrad(texture<long1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int1 __tex2DGrad_long1_as_int1(texture<long1, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_long1_as_int1");
  int1 v = __tex2DGrad_long1_as_int1(t, x, y, dPdx, dPdy);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2DGrad(texture<ulong1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint1 __tex2DGrad_ulong1_as_uint1(texture<ulong1, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_ulong1_as_uint1");
  uint1 v = __tex2DGrad_ulong1_as_uint1(t, x, y, dPdx, dPdy);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2DGrad(texture<long2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int2 __tex2DGrad_long2_as_int2(texture<long2, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_long2_as_int2");
  int2 v = __tex2DGrad_long2_as_int2(t, x, y, dPdx, dPdy);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2DGrad(texture<ulong2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint2 __tex2DGrad_ulong2_as_uint2(texture<ulong2, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_ulong2_as_uint2");
  uint2 v = __tex2DGrad_ulong2_as_uint2(t, x, y, dPdx, dPdy);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2DGrad(texture<long4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int4 __tex2DGrad_long4_as_int4(texture<long4, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_long4_as_int4");
  int4 v = __tex2DGrad_long4_as_int4(t, x, y, dPdx, dPdy);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2DGrad(texture<ulong4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint4 __tex2DGrad_ulong4_as_uint4(texture<ulong4, cudaTextureType2D, cudaReadModeElementType>, float, float, float2, float2)  asm("__tex2DGrad_ulong4_as_uint4");
  uint4 v = __tex2DGrad_ulong4_as_uint4(t, x, y, dPdx, dPdy);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex2DGrad(texture<float, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_float") ;
__device__ __cudart_builtin__ float1 tex2DGrad(texture<float1, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_float1") ;
__device__ __cudart_builtin__ float2 tex2DGrad(texture<float2, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_float2") ;
__device__ __cudart_builtin__ float4 tex2DGrad(texture<float4, cudaTextureType2D, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2DGrad(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2DGrad(texture<char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2DGrad(texture<signed char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2DGrad(texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2DGrad(texture<char1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2DGrad(texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2DGrad(texture<char2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2DGrad(texture<uchar2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2DGrad(texture<char4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2DGrad(texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex2DGrad(texture<short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_short") ;
__device__ __cudart_builtin__ float tex2DGrad(texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2DGrad(texture<short1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2DGrad(texture<ushort1, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2DGrad(texture<short2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2DGrad(texture<ushort2, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2DGrad(texture<short4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2DGrad(texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy) asm("__tex2DGrad_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex1DLayeredGrad(texture<T, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex1DLayeredGrad(texture<T, cudaTextureType1DLayered, cudaReadModeNormalizedFloat>, float, int, float, float) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex1DLayeredGrad(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex1DLayeredGrad(texture<char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex1DLayeredGrad(texture<signed char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_schar") ;
__device__ __cudart_builtin__ unsigned char tex1DLayeredGrad(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uchar") ;
__device__ __cudart_builtin__ char1 tex1DLayeredGrad(texture<char1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_char1") ;
__device__ __cudart_builtin__ uchar1 tex1DLayeredGrad(texture<uchar1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uchar1") ;
__device__ __cudart_builtin__ char2 tex1DLayeredGrad(texture<char2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_char2") ;
__device__ __cudart_builtin__ uchar2 tex1DLayeredGrad(texture<uchar2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uchar2") ;
__device__ __cudart_builtin__ char4 tex1DLayeredGrad(texture<char4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_char4") ;
__device__ __cudart_builtin__ uchar4 tex1DLayeredGrad(texture<uchar4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uchar4") ;

__device__ __cudart_builtin__ short tex1DLayeredGrad(texture<short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_short") ;
__device__ __cudart_builtin__ unsigned short tex1DLayeredGrad(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_ushort") ;
__device__ __cudart_builtin__ short1 tex1DLayeredGrad(texture<short1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_short1") ;
__device__ __cudart_builtin__ ushort1 tex1DLayeredGrad(texture<ushort1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_ushort1") ;
__device__ __cudart_builtin__ short2 tex1DLayeredGrad(texture<short2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_short2") ;
__device__ __cudart_builtin__ ushort2 tex1DLayeredGrad(texture<ushort2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_ushort2") ;
__device__ __cudart_builtin__ short4 tex1DLayeredGrad(texture<short4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_short4") ;
__device__ __cudart_builtin__ ushort4 tex1DLayeredGrad(texture<ushort4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_ushort4") ;

__device__ __cudart_builtin__ int tex1DLayeredGrad(texture<int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_int") ;
__device__ __cudart_builtin__ unsigned int tex1DLayeredGrad(texture<unsigned int, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uint") ;
__device__ __cudart_builtin__ int1 tex1DLayeredGrad(texture<int1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_int1") ;
__device__ __cudart_builtin__ uint1 tex1DLayeredGrad(texture<uint1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uint1") ;
__device__ __cudart_builtin__ int2 tex1DLayeredGrad(texture<int2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_int2") ;
__device__ __cudart_builtin__ uint2 tex1DLayeredGrad(texture<uint2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uint2") ;
__device__ __cudart_builtin__ int4 tex1DLayeredGrad(texture<int4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_int4") ;
__device__ __cudart_builtin__ uint4 tex1DLayeredGrad(texture<uint4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex1DLayeredGrad(texture<long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int __tex1DLayeredGrad_long_as_int(texture<long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_long_as_int");
  return __tex1DLayeredGrad_long_as_int(t, x, layer, dPdx, dPdy);
}

static __device__ __forceinline__ unsigned long tex1DLayeredGrad(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  unsigned int __tex1DLayeredGrad_ulong_as_uint(texture<unsigned long, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_ulong_as_uint");
  return __tex1DLayeredGrad_ulong_as_uint(t, x, layer, dPdx, dPdy);
}

static __device__ __forceinline__ long1 tex1DLayeredGrad(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int1 __tex1DLayeredGrad_long1_as_int1(texture<long1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_long1_as_int1");
  int1 v = __tex1DLayeredGrad_long1_as_int1(t, x, layer, dPdx, dPdy);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex1DLayeredGrad(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint1 __tex1DLayeredGrad_ulong1_as_uint1(texture<ulong1, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_ulong1_as_uint1");
  uint1 v = __tex1DLayeredGrad_ulong1_as_uint1(t, x, layer, dPdx, dPdy);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex1DLayeredGrad(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int2 __tex1DLayeredGrad_long2_as_int2(texture<long2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_long2_as_int2");
  int2 v = __tex1DLayeredGrad_long2_as_int2(t, x, layer, dPdx, dPdy);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex1DLayeredGrad(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint2 __tex1DLayeredGrad_ulong2_as_uint2(texture<ulong2, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_ulong2_as_uint2");
  uint2 v = __tex1DLayeredGrad_ulong2_as_uint2(t, x, layer, dPdx, dPdy);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex1DLayeredGrad(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  int4 __tex1DLayeredGrad_long4_as_int4(texture<long4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_long4_as_int4");
  int4 v = __tex1DLayeredGrad_long4_as_int4(t, x, layer, dPdx, dPdy);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex1DLayeredGrad(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  __device__ __cudart_builtin__  uint4 __tex1DLayeredGrad_ulong4_as_uint4(texture<ulong4, cudaTextureType1DLayered, cudaReadModeElementType>, float, int, float, float)  asm("__tex1DLayeredGrad_ulong4_as_uint4");
  uint4 v = __tex1DLayeredGrad_ulong4_as_uint4(t, x, layer, dPdx, dPdy);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<float, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_float") ;
__device__ __cudart_builtin__ float1 tex1DLayeredGrad(texture<float1, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_float1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredGrad(texture<float2, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_float2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredGrad(texture<float4, cudaTextureType1DLayered, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_float4") ;


#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<signed char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_schar") ;
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<unsigned char, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex1DLayeredGrad(texture<char1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex1DLayeredGrad(texture<uchar1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredGrad(texture<char2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex1DLayeredGrad(texture<uchar2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredGrad(texture<char4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex1DLayeredGrad(texture<uchar4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_short") ;
__device__ __cudart_builtin__ float tex1DLayeredGrad(texture<unsigned short, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex1DLayeredGrad(texture<short1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex1DLayeredGrad(texture<ushort1, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex1DLayeredGrad(texture<short2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex1DLayeredGrad(texture<ushort2, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex1DLayeredGrad(texture<short4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex1DLayeredGrad(texture<ushort4, cudaTextureType1DLayered, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy) asm("__tex1DLayeredGrad_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex2DLayeredGrad(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex2DLayeredGrad(texture<T, cudaTextureType2DLayered, cudaReadModeNormalizedFloat>, float, float, int, float2, float2) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex2DLayeredGrad(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex2DLayeredGrad(texture<char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex2DLayeredGrad(texture<signed char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_schar") ;
__device__ __cudart_builtin__ unsigned char tex2DLayeredGrad(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uchar") ;
__device__ __cudart_builtin__ char1 tex2DLayeredGrad(texture<char1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_char1") ;
__device__ __cudart_builtin__ uchar1 tex2DLayeredGrad(texture<uchar1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uchar1") ;
__device__ __cudart_builtin__ char2 tex2DLayeredGrad(texture<char2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_char2") ;
__device__ __cudart_builtin__ uchar2 tex2DLayeredGrad(texture<uchar2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uchar2") ;
__device__ __cudart_builtin__ char4 tex2DLayeredGrad(texture<char4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_char4") ;
__device__ __cudart_builtin__ uchar4 tex2DLayeredGrad(texture<uchar4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uchar4") ;

__device__ __cudart_builtin__ short tex2DLayeredGrad(texture<short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_short") ;
__device__ __cudart_builtin__ unsigned short tex2DLayeredGrad(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_ushort") ;
__device__ __cudart_builtin__ short1 tex2DLayeredGrad(texture<short1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_short1") ;
__device__ __cudart_builtin__ ushort1 tex2DLayeredGrad(texture<ushort1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_ushort1") ;
__device__ __cudart_builtin__ short2 tex2DLayeredGrad(texture<short2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_short2") ;
__device__ __cudart_builtin__ ushort2 tex2DLayeredGrad(texture<ushort2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_ushort2") ;
__device__ __cudart_builtin__ short4 tex2DLayeredGrad(texture<short4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_short4") ;
__device__ __cudart_builtin__ ushort4 tex2DLayeredGrad(texture<ushort4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_ushort4") ;

__device__ __cudart_builtin__ int tex2DLayeredGrad(texture<int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_int") ;
__device__ __cudart_builtin__ unsigned int tex2DLayeredGrad(texture<unsigned int, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uint") ;
__device__ __cudart_builtin__ int1 tex2DLayeredGrad(texture<int1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_int1") ;
__device__ __cudart_builtin__ uint1 tex2DLayeredGrad(texture<uint1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uint1") ;
__device__ __cudart_builtin__ int2 tex2DLayeredGrad(texture<int2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_int2") ;
__device__ __cudart_builtin__ uint2 tex2DLayeredGrad(texture<uint2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uint2") ;
__device__ __cudart_builtin__ int4 tex2DLayeredGrad(texture<int4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_int4") ;
__device__ __cudart_builtin__ uint4 tex2DLayeredGrad(texture<uint4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex2DLayeredGrad(texture<long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int __tex2DLayeredGrad_long_as_int(texture<long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_long_as_int");
  return  __tex2DLayeredGrad_long_as_int(t, x, y, layer, dPdx, dPdy);
}

static __device__ __forceinline__ unsigned long tex2DLayeredGrad(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  unsigned int __tex2DLayeredGrad_ulong_as_uint(texture<unsigned long, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_ulong_as_uint");
  return  __tex2DLayeredGrad_ulong_as_uint(t, x, y, layer, dPdx, dPdy);
}

static __device__ __forceinline__ long1 tex2DLayeredGrad(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int1 __tex2DLayeredGrad_long1_as_int1(texture<long1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_long1_as_int1");
  int1 v = __tex2DLayeredGrad_long1_as_int1(t, x, y, layer, dPdx, dPdy);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex2DLayeredGrad(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint1 __tex2DLayeredGrad_ulong1_as_uint1(texture<ulong1, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_ulong1_as_uint1");
  uint1 v = __tex2DLayeredGrad_ulong1_as_uint1(t, x, y, layer, dPdx, dPdy);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex2DLayeredGrad(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int2 __tex2DLayeredGrad_long2_as_int2(texture<long2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_long2_as_int2");
  int2 v = __tex2DLayeredGrad_long2_as_int2(t, x, y, layer, dPdx, dPdy);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex2DLayeredGrad(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint2 __tex2DLayeredGrad_ulong2_as_uint2(texture<ulong2, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_ulong2_as_uint2");
  uint2 v = __tex2DLayeredGrad_ulong2_as_uint2(t, x, y, layer, dPdx, dPdy);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex2DLayeredGrad(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  int4 __tex2DLayeredGrad_long4_as_int4(texture<long4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_long4_as_int4");
  int4 v = __tex2DLayeredGrad_long4_as_int4(t, x, y, layer, dPdx, dPdy);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex2DLayeredGrad(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  __device__ __cudart_builtin__  uint4 __tex2DLayeredGrad_ulong4_as_uint4(texture<ulong4, cudaTextureType2DLayered, cudaReadModeElementType>, float, float, int, float2, float2)  asm("__tex2DLayeredGrad_ulong4_as_uint4");
  uint4 v = __tex2DLayeredGrad_ulong4_as_uint4(t, x, y, layer, dPdx, dPdy);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_float") ;
__device__ __cudart_builtin__ float1 tex2DLayeredGrad(texture<float1, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_float1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredGrad(texture<float2, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_float2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredGrad(texture<float4, cudaTextureType2DLayered, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<signed char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_schar") ;
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<unsigned char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex2DLayeredGrad(texture<char1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex2DLayeredGrad(texture<uchar1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredGrad(texture<char2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex2DLayeredGrad(texture<uchar2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredGrad(texture<char4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex2DLayeredGrad(texture<uchar4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_short") ;
__device__ __cudart_builtin__ float tex2DLayeredGrad(texture<unsigned short, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex2DLayeredGrad(texture<short1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex2DLayeredGrad(texture<ushort1, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex2DLayeredGrad(texture<short2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex2DLayeredGrad(texture<ushort2, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex2DLayeredGrad(texture<short4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex2DLayeredGrad(texture<ushort4, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy) asm("__tex2DLayeredGrad_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#ifndef __CUDA_ARCH__
template <typename T>
static __device__ T tex3DGrad(texture<T, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4) {  }

template <typename T>
static __device__ typename __nv_tex_rmnf_ret<T>::type tex3DGrad(texture<T, cudaTextureType3D, cudaReadModeNormalizedFloat>, float, float, float, float4, float4) {  }
#else /* __CUDA_ARCH__ */
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ char tex3DGrad(texture<char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_char_as_uchar") ;
#else
__device__ __cudart_builtin__ char tex3DGrad(texture<char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ signed char tex3DGrad(texture<signed char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_schar") ;
__device__ __cudart_builtin__ unsigned char tex3DGrad(texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uchar") ;
__device__ __cudart_builtin__ char1 tex3DGrad(texture<char1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_char1") ;
__device__ __cudart_builtin__ uchar1 tex3DGrad(texture<uchar1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uchar1") ;
__device__ __cudart_builtin__ char2 tex3DGrad(texture<char2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_char2") ;
__device__ __cudart_builtin__ uchar2 tex3DGrad(texture<uchar2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uchar2") ;
__device__ __cudart_builtin__ char4 tex3DGrad(texture<char4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_char4") ;
__device__ __cudart_builtin__ uchar4 tex3DGrad(texture<uchar4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uchar4") ;

__device__ __cudart_builtin__ short tex3DGrad(texture<short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_short") ;
__device__ __cudart_builtin__ unsigned short tex3DGrad(texture<unsigned short, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_ushort") ;
__device__ __cudart_builtin__ short1 tex3DGrad(texture<short1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_short1") ;
__device__ __cudart_builtin__ ushort1 tex3DGrad(texture<ushort1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_ushort1") ;
__device__ __cudart_builtin__ short2 tex3DGrad(texture<short2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_short2") ;
__device__ __cudart_builtin__ ushort2 tex3DGrad(texture<ushort2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_ushort2") ;
__device__ __cudart_builtin__ short4 tex3DGrad(texture<short4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_short4") ;
__device__ __cudart_builtin__ ushort4 tex3DGrad(texture<ushort4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_ushort4") ;

__device__ __cudart_builtin__ int tex3DGrad(texture<int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_int") ;
__device__ __cudart_builtin__ unsigned int tex3DGrad(texture<unsigned int, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uint") ;
__device__ __cudart_builtin__ int1 tex3DGrad(texture<int1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_int1") ;
__device__ __cudart_builtin__ uint1 tex3DGrad(texture<uint1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uint1") ;
__device__ __cudart_builtin__ int2 tex3DGrad(texture<int2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_int2") ;
__device__ __cudart_builtin__ uint2 tex3DGrad(texture<uint2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uint2") ;
__device__ __cudart_builtin__ int4 tex3DGrad(texture<int4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_int4") ;
__device__ __cudart_builtin__ uint4 tex3DGrad(texture<uint4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_uint4") ;
#if !defined(__LP64__)
static __device__ __forceinline__ long tex3DGrad(texture<long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  int __tex3DGrad_long_as_int(texture<long, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_long_as_int");
  return __tex3DGrad_long_as_int(t, x, y, z, dPdx, dPdy);
}

static __device__ __forceinline__ unsigned long tex3DGrad(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  unsigned int __tex3DGrad_ulong_as_uint(texture<unsigned long, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_ulong_as_uint");
  return __tex3DGrad_ulong_as_uint(t, x, y, z, dPdx, dPdy);

}

static __device__ __forceinline__ long1 tex3DGrad(texture<long1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  int1 __tex3DGrad_long1_as_int1(texture<long1, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_long1_as_int1");
  int1 v = __tex3DGrad_long1_as_int1(t, x, y, z, dPdx, dPdy);

  return make_long1(v.x);
}

static __device__ __forceinline__ ulong1 tex3DGrad(texture<ulong1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  uint1 __tex3DGrad_ulong1_as_uint1(texture<ulong1, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_ulong1_as_uint1");
  uint1 v = __tex3DGrad_ulong1_as_uint1(t, x, y, z, dPdx, dPdy);

  return make_ulong1(v.x);
}

static __device__ __forceinline__ long2 tex3DGrad(texture<long2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  int2 __tex3DGrad_long2_as_int2(texture<long2, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_long2_as_int2");
  int2 v = __tex3DGrad_long2_as_int2(t, x, y, z, dPdx, dPdy);

  return make_long2(v.x, v.y);
}

static __device__ __forceinline__ ulong2 tex3DGrad(texture<ulong2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  uint2 __tex3DGrad_ulong2_as_uint2(texture<ulong2, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_ulong2_as_uint2");
  uint2 v = __tex3DGrad_ulong2_as_uint2(t, x, y, z, dPdx, dPdy);

  return make_ulong2(v.x, v.y);
}

static __device__ __forceinline__ long4 tex3DGrad(texture<long4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  int4 __tex3DGrad_long4_as_int4(texture<long4, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_long4_as_int4");
  int4 v = __tex3DGrad_long4_as_int4(t, x, y, z, dPdx, dPdy);

  return make_long4(v.x, v.y, v.z, v.w);
}

static __device__ __forceinline__ ulong4 tex3DGrad(texture<ulong4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  __device__ __cudart_builtin__  uint4 __tex3DGrad_ulong4_as_uint4(texture<ulong4, cudaTextureType3D, cudaReadModeElementType>, float, float, float, float4, float4)  asm("__tex3DGrad_ulong4_as_uint4");
  uint4 v = __tex3DGrad_ulong4_as_uint4(t, x, y, z, dPdx, dPdy);

  return make_ulong4(v.x, v.y, v.z, v.w);
}
#endif /* !__LP64__ */
__device__ __cudart_builtin__ float tex3DGrad(texture<float, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_float") ;
__device__ __cudart_builtin__ float1 tex3DGrad(texture<float1, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_float1") ;
__device__ __cudart_builtin__ float2 tex3DGrad(texture<float2, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_float2") ;
__device__ __cudart_builtin__ float4 tex3DGrad(texture<float4, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_float4") ;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ float tex3DGrad(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_char_as_uchar") ;
#else
__device__ __cudart_builtin__ float tex3DGrad(texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_char_as_schar") ;
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ float tex3DGrad(texture<signed char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_schar") ;
__device__ __cudart_builtin__ float tex3DGrad(texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_uchar") ;
__device__ __cudart_builtin__ float1 tex3DGrad(texture<char1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_char1") ;
__device__ __cudart_builtin__ float1 tex3DGrad(texture<uchar1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_uchar1") ;
__device__ __cudart_builtin__ float2 tex3DGrad(texture<char2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_char2") ;
__device__ __cudart_builtin__ float2 tex3DGrad(texture<uchar2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_uchar2") ;
__device__ __cudart_builtin__ float4 tex3DGrad(texture<char4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_char4") ;
__device__ __cudart_builtin__ float4 tex3DGrad(texture<uchar4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_uchar4") ;

__device__ __cudart_builtin__ float tex3DGrad(texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_short") ;
__device__ __cudart_builtin__ float tex3DGrad(texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_ushort") ;
__device__ __cudart_builtin__ float1 tex3DGrad(texture<short1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_short1") ;
__device__ __cudart_builtin__ float1 tex3DGrad(texture<ushort1, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_ushort1") ;
__device__ __cudart_builtin__ float2 tex3DGrad(texture<short2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_short2") ;
__device__ __cudart_builtin__ float2 tex3DGrad(texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_ushort2") ;
__device__ __cudart_builtin__ float4 tex3DGrad(texture<short4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_short4") ;
__device__ __cudart_builtin__ float4 tex3DGrad(texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy) asm("__tex3DGrad_rmnf_ushort4") ;
#endif /* __CUDA_ARCH__ */

#endif /* __cplusplus && __CUDACC__ */

#endif /* !__TEXTURE_FETCH_FUNCTIONS_H__ */

