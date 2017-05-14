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


#ifndef __TEXTURE_INDIRECT_FUNCTIONS_H__
#define __TEXTURE_INDIRECT_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

#include "builtin_types.h"
#include "host_defines.h"
#include "vector_functions.h"

/*******************************************************************************
*                                                                              *
* 1D Linear Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1Dfetch(T *, cudaTextureObject_t, int) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1Dfetch(char *, cudaTextureObject_t, int) asm("__itex1Dfetch_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1Dfetch(char *, cudaTextureObject_t, int) asm("__itex1Dfetch_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex1Dfetch(signed char *, cudaTextureObject_t, int) asm("__itex1Dfetch_schar");
__device__ __cudart_builtin__ void tex1Dfetch(char1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_char1");
__device__ __cudart_builtin__ void tex1Dfetch(char2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_char2");
__device__ __cudart_builtin__ void tex1Dfetch(char4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_char4");
__device__ __cudart_builtin__ void tex1Dfetch(unsigned char *, cudaTextureObject_t, int) asm("__itex1Dfetch_uchar");
__device__ __cudart_builtin__ void tex1Dfetch(uchar1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uchar1");
__device__ __cudart_builtin__ void tex1Dfetch(uchar2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uchar2");
__device__ __cudart_builtin__ void tex1Dfetch(uchar4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uchar4");

__device__ __cudart_builtin__ void tex1Dfetch(short *, cudaTextureObject_t, int) asm("__itex1Dfetch_short");
__device__ __cudart_builtin__ void tex1Dfetch(short1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_short1");
__device__ __cudart_builtin__ void tex1Dfetch(short2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_short2");
__device__ __cudart_builtin__ void tex1Dfetch(short4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_short4");
__device__ __cudart_builtin__ void tex1Dfetch(unsigned short *, cudaTextureObject_t, int) asm("__itex1Dfetch_ushort");
__device__ __cudart_builtin__ void tex1Dfetch(ushort1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ushort1");
__device__ __cudart_builtin__ void tex1Dfetch(ushort2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ushort2");
__device__ __cudart_builtin__ void tex1Dfetch(ushort4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ushort4");

__device__ __cudart_builtin__ void tex1Dfetch(int *, cudaTextureObject_t, int) asm("__itex1Dfetch_int");
__device__ __cudart_builtin__ void tex1Dfetch(int1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_int1");
__device__ __cudart_builtin__ void tex1Dfetch(int2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_int2");
__device__ __cudart_builtin__ void tex1Dfetch(int4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_int4");
__device__ __cudart_builtin__ void tex1Dfetch(unsigned int *, cudaTextureObject_t, int) asm("__itex1Dfetch_uint");
__device__ __cudart_builtin__ void tex1Dfetch(uint1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uint1");
__device__ __cudart_builtin__ void tex1Dfetch(uint2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uint2");
__device__ __cudart_builtin__ void tex1Dfetch(uint4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1Dfetch(long *, cudaTextureObject_t, int) asm("__itex1Dfetch_long");
__device__ __cudart_builtin__ void tex1Dfetch(long1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_long1");
__device__ __cudart_builtin__ void tex1Dfetch(long2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_long2");
__device__ __cudart_builtin__ void tex1Dfetch(long4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_long4");
__device__ __cudart_builtin__ void tex1Dfetch(unsigned long *, cudaTextureObject_t, int) asm("__itex1Dfetch_ulong");
__device__ __cudart_builtin__ void tex1Dfetch(ulong1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ulong1");
__device__ __cudart_builtin__ void tex1Dfetch(ulong2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ulong2");
__device__ __cudart_builtin__ void tex1Dfetch(ulong4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1Dfetch(float *, cudaTextureObject_t, int) asm("__itex1Dfetch_float");
__device__ __cudart_builtin__ void tex1Dfetch(float1 *, cudaTextureObject_t, int) asm("__itex1Dfetch_float1");
__device__ __cudart_builtin__ void tex1Dfetch(float2 *, cudaTextureObject_t, int) asm("__itex1Dfetch_float2");
__device__ __cudart_builtin__ void tex1Dfetch(float4 *, cudaTextureObject_t, int) asm("__itex1Dfetch_float4");
#endif /* !__CUDA_ARCH__ */


template <class T>
static __device__ T tex1Dfetch(cudaTextureObject_t texObject, int x)
{
  T ret;
  tex1Dfetch(&ret, texObject, x);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1D(T *, cudaTextureObject_t, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1D(char *, cudaTextureObject_t, float) asm("__itex1D_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1D(char *, cudaTextureObject_t, float) asm("__itex1D_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */

__device__ __cudart_builtin__ void tex1D(signed char *, cudaTextureObject_t, float) asm("__itex1D_schar");
__device__ __cudart_builtin__ void tex1D(char1 *, cudaTextureObject_t, float) asm("__itex1D_char1");
__device__ __cudart_builtin__ void tex1D(char2 *, cudaTextureObject_t, float) asm("__itex1D_char2");
__device__ __cudart_builtin__ void tex1D(char4 *, cudaTextureObject_t, float) asm("__itex1D_char4");
__device__ __cudart_builtin__ void tex1D(unsigned char *, cudaTextureObject_t, float) asm("__itex1D_uchar");
__device__ __cudart_builtin__ void tex1D(uchar1 *, cudaTextureObject_t, float) asm("__itex1D_uchar1");
__device__ __cudart_builtin__ void tex1D(uchar2 *, cudaTextureObject_t, float) asm("__itex1D_uchar2");
__device__ __cudart_builtin__ void tex1D(uchar4 *, cudaTextureObject_t, float) asm("__itex1D_uchar4");

__device__ __cudart_builtin__ void tex1D(short *, cudaTextureObject_t, float) asm("__itex1D_short");
__device__ __cudart_builtin__ void tex1D(short1 *, cudaTextureObject_t, float) asm("__itex1D_short1");
__device__ __cudart_builtin__ void tex1D(short2 *, cudaTextureObject_t, float) asm("__itex1D_short2");
__device__ __cudart_builtin__ void tex1D(short4 *, cudaTextureObject_t, float) asm("__itex1D_short4");
__device__ __cudart_builtin__ void tex1D(unsigned short *, cudaTextureObject_t, float) asm("__itex1D_ushort");
__device__ __cudart_builtin__ void tex1D(ushort1 *, cudaTextureObject_t, float) asm("__itex1D_ushort1");
__device__ __cudart_builtin__ void tex1D(ushort2 *, cudaTextureObject_t, float) asm("__itex1D_ushort2");
__device__ __cudart_builtin__ void tex1D(ushort4 *, cudaTextureObject_t, float) asm("__itex1D_ushort4");


__device__ __cudart_builtin__ void tex1D(int *, cudaTextureObject_t, float) asm("__itex1D_int");
__device__ __cudart_builtin__ void tex1D(int1 *, cudaTextureObject_t, float) asm("__itex1D_int1");
__device__ __cudart_builtin__ void tex1D(int2 *, cudaTextureObject_t, float) asm("__itex1D_int2");
__device__ __cudart_builtin__ void tex1D(int4 *, cudaTextureObject_t, float) asm("__itex1D_int4");
__device__ __cudart_builtin__ void tex1D(unsigned int *, cudaTextureObject_t, float) asm("__itex1D_uint");
__device__ __cudart_builtin__ void tex1D(uint1 *, cudaTextureObject_t, float) asm("__itex1D_uint1");
__device__ __cudart_builtin__ void tex1D(uint2 *, cudaTextureObject_t, float) asm("__itex1D_uint2");
__device__ __cudart_builtin__ void tex1D(uint4 *, cudaTextureObject_t, float) asm("__itex1D_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1D(long *, cudaTextureObject_t, float) asm("__itex1D_long");
__device__ __cudart_builtin__ void tex1D(long1 *, cudaTextureObject_t, float) asm("__itex1D_long1");
__device__ __cudart_builtin__ void tex1D(long2 *, cudaTextureObject_t, float) asm("__itex1D_long2");
__device__ __cudart_builtin__ void tex1D(long4 *, cudaTextureObject_t, float) asm("__itex1D_long4");
__device__ __cudart_builtin__ void tex1D(unsigned long *, cudaTextureObject_t, float) asm("__itex1D_ulong");
__device__ __cudart_builtin__ void tex1D(ulong1 *, cudaTextureObject_t, float) asm("__itex1D_ulong1");
__device__ __cudart_builtin__ void tex1D(ulong2 *, cudaTextureObject_t, float) asm("__itex1D_ulong2");
__device__ __cudart_builtin__ void tex1D(ulong4 *, cudaTextureObject_t, float) asm("__itex1D_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1D(float *, cudaTextureObject_t, float) asm("__itex1D_float");
__device__ __cudart_builtin__ void tex1D(float1 *, cudaTextureObject_t, float) asm("__itex1D_float1");
__device__ __cudart_builtin__ void tex1D(float2 *, cudaTextureObject_t, float) asm("__itex1D_float2");
__device__ __cudart_builtin__ void tex1D(float4 *, cudaTextureObject_t, float) asm("__itex1D_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1D(cudaTextureObject_t texObject, float x)
{
  T ret;
  tex1D(&ret, texObject, x);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 2D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2D(T *, cudaTextureObject_t, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2D(char *, cudaTextureObject_t, float, float) asm("__itex2D_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2D(char *, cudaTextureObject_t, float, float) asm("__itex2D_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex2D(signed char *, cudaTextureObject_t, float, float) asm("__itex2D_schar");
__device__ __cudart_builtin__ void tex2D(char1 *, cudaTextureObject_t, float, float) asm("__itex2D_char1");
__device__ __cudart_builtin__ void tex2D(char2 *, cudaTextureObject_t, float, float) asm("__itex2D_char2");
__device__ __cudart_builtin__ void tex2D(char4 *, cudaTextureObject_t, float, float) asm("__itex2D_char4");
__device__ __cudart_builtin__ void tex2D(unsigned char *, cudaTextureObject_t, float, float) asm("__itex2D_uchar");
__device__ __cudart_builtin__ void tex2D(uchar1 *, cudaTextureObject_t, float, float) asm("__itex2D_uchar1");
__device__ __cudart_builtin__ void tex2D(uchar2 *, cudaTextureObject_t, float, float) asm("__itex2D_uchar2");
__device__ __cudart_builtin__ void tex2D(uchar4 *, cudaTextureObject_t, float, float) asm("__itex2D_uchar4");

__device__ __cudart_builtin__ void tex2D(short *, cudaTextureObject_t, float, float) asm("__itex2D_short");
__device__ __cudart_builtin__ void tex2D(short1 *, cudaTextureObject_t, float, float) asm("__itex2D_short1");
__device__ __cudart_builtin__ void tex2D(short2 *, cudaTextureObject_t, float, float) asm("__itex2D_short2");
__device__ __cudart_builtin__ void tex2D(short4 *, cudaTextureObject_t, float, float) asm("__itex2D_short4");
__device__ __cudart_builtin__ void tex2D(unsigned short *, cudaTextureObject_t, float, float) asm("__itex2D_ushort");
__device__ __cudart_builtin__ void tex2D(ushort1 *, cudaTextureObject_t, float, float) asm("__itex2D_ushort1");
__device__ __cudart_builtin__ void tex2D(ushort2 *, cudaTextureObject_t, float, float) asm("__itex2D_ushort2");
__device__ __cudart_builtin__ void tex2D(ushort4 *, cudaTextureObject_t, float, float) asm("__itex2D_ushort4");

__device__ __cudart_builtin__ void tex2D(int *, cudaTextureObject_t, float, float) asm("__itex2D_int");
__device__ __cudart_builtin__ void tex2D(int1 *, cudaTextureObject_t, float, float) asm("__itex2D_int1");
__device__ __cudart_builtin__ void tex2D(int2 *, cudaTextureObject_t, float, float) asm("__itex2D_int2");
__device__ __cudart_builtin__ void tex2D(int4 *, cudaTextureObject_t, float, float) asm("__itex2D_int4");
__device__ __cudart_builtin__ void tex2D(unsigned int *, cudaTextureObject_t, float, float) asm("__itex2D_uint");
__device__ __cudart_builtin__ void tex2D(uint1 *, cudaTextureObject_t, float, float) asm("__itex2D_uint1");
__device__ __cudart_builtin__ void tex2D(uint2 *, cudaTextureObject_t, float, float) asm("__itex2D_uint2");
__device__ __cudart_builtin__ void tex2D(uint4 *, cudaTextureObject_t, float, float) asm("__itex2D_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2D(long *, cudaTextureObject_t, float, float) asm("__itex2D_long");
__device__ __cudart_builtin__ void tex2D(long1 *, cudaTextureObject_t, float, float) asm("__itex2D_long1");
__device__ __cudart_builtin__ void tex2D(long2 *, cudaTextureObject_t, float, float) asm("__itex2D_long2");
__device__ __cudart_builtin__ void tex2D(long4 *, cudaTextureObject_t, float, float) asm("__itex2D_long4");
__device__ __cudart_builtin__ void tex2D(unsigned long *, cudaTextureObject_t, float, float) asm("__itex2D_ulong");
__device__ __cudart_builtin__ void tex2D(ulong1 *, cudaTextureObject_t, float, float) asm("__itex2D_ulong1");
__device__ __cudart_builtin__ void tex2D(ulong2 *, cudaTextureObject_t, float, float) asm("__itex2D_ulong2");
__device__ __cudart_builtin__ void tex2D(ulong4 *, cudaTextureObject_t, float, float) asm("__itex2D_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2D(float *, cudaTextureObject_t, float, float) asm("__itex2D_float");
__device__ __cudart_builtin__ void tex2D(float1 *, cudaTextureObject_t, float, float) asm("__itex2D_float1");
__device__ __cudart_builtin__ void tex2D(float2 *, cudaTextureObject_t, float, float) asm("__itex2D_float2");
__device__ __cudart_builtin__ void tex2D(float4 *, cudaTextureObject_t, float, float) asm("__itex2D_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2D(cudaTextureObject_t texObject, float x, float y)
{
  T ret;
  tex2D(&ret, texObject, x, y);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 3D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex3D(T *, cudaTextureObject_t, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex3D(char *, cudaTextureObject_t, float, float, float) asm("__itex3D_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex3D(char *, cudaTextureObject_t, float, float, float) asm("__itex3D_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex3D(signed char *, cudaTextureObject_t, float, float, float) asm("__itex3D_schar");
__device__ __cudart_builtin__ void tex3D(char1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_char1");
__device__ __cudart_builtin__ void tex3D(char2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_char2");
__device__ __cudart_builtin__ void tex3D(char4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_char4");
__device__ __cudart_builtin__ void tex3D(unsigned char *, cudaTextureObject_t, float, float, float) asm("__itex3D_uchar");
__device__ __cudart_builtin__ void tex3D(uchar1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uchar1");
__device__ __cudart_builtin__ void tex3D(uchar2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uchar2");
__device__ __cudart_builtin__ void tex3D(uchar4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uchar4");

__device__ __cudart_builtin__ void tex3D(short *, cudaTextureObject_t, float, float, float) asm("__itex3D_short");
__device__ __cudart_builtin__ void tex3D(short1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_short1");
__device__ __cudart_builtin__ void tex3D(short2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_short2");
__device__ __cudart_builtin__ void tex3D(short4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_short4");
__device__ __cudart_builtin__ void tex3D(unsigned short *, cudaTextureObject_t, float, float, float) asm("__itex3D_ushort");
__device__ __cudart_builtin__ void tex3D(ushort1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ushort1");
__device__ __cudart_builtin__ void tex3D(ushort2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ushort2");
__device__ __cudart_builtin__ void tex3D(ushort4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ushort4");

__device__ __cudart_builtin__ void tex3D(int *, cudaTextureObject_t, float, float, float) asm("__itex3D_int");
__device__ __cudart_builtin__ void tex3D(int1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_int1");
__device__ __cudart_builtin__ void tex3D(int2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_int2");
__device__ __cudart_builtin__ void tex3D(int4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_int4");
__device__ __cudart_builtin__ void tex3D(unsigned int *, cudaTextureObject_t, float, float, float) asm("__itex3D_uint");
__device__ __cudart_builtin__ void tex3D(uint1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uint1");
__device__ __cudart_builtin__ void tex3D(uint2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uint2");
__device__ __cudart_builtin__ void tex3D(uint4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex3D(long *, cudaTextureObject_t, float, float, float) asm("__itex3D_long");
__device__ __cudart_builtin__ void tex3D(long1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_long1");
__device__ __cudart_builtin__ void tex3D(long2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_long2");
__device__ __cudart_builtin__ void tex3D(long4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_long4");
__device__ __cudart_builtin__ void tex3D(unsigned long *, cudaTextureObject_t, float, float, float) asm("__itex3D_ulong");
__device__ __cudart_builtin__ void tex3D(ulong1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ulong1");
__device__ __cudart_builtin__ void tex3D(ulong2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ulong2");
__device__ __cudart_builtin__ void tex3D(ulong4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex3D(float *, cudaTextureObject_t, float, float, float) asm("__itex3D_float");
__device__ __cudart_builtin__ void tex3D(float1 *, cudaTextureObject_t, float, float, float) asm("__itex3D_float1");
__device__ __cudart_builtin__ void tex3D(float2 *, cudaTextureObject_t, float, float, float) asm("__itex3D_float2");
__device__ __cudart_builtin__ void tex3D(float4 *, cudaTextureObject_t, float, float, float) asm("__itex3D_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex3D(cudaTextureObject_t texObject, float x, float y, float z)
{
  T ret;
  tex3D(&ret, texObject, x, y, z);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 1D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1DLayered(T *, cudaTextureObject_t, float, int) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1DLayered(char *, cudaTextureObject_t, float, int) asm("__itex1DLayered_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1DLayered(char *, cudaTextureObject_t, float, int) asm("__itex1DLayered_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex1DLayered(signed char *, cudaTextureObject_t, float, int) asm("__itex1DLayered_schar");
__device__ __cudart_builtin__ void tex1DLayered(char1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_char1");
__device__ __cudart_builtin__ void tex1DLayered(char2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_char2");
__device__ __cudart_builtin__ void tex1DLayered(char4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_char4");
__device__ __cudart_builtin__ void tex1DLayered(unsigned char *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uchar");
__device__ __cudart_builtin__ void tex1DLayered(uchar1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uchar1");
__device__ __cudart_builtin__ void tex1DLayered(uchar2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uchar2");
__device__ __cudart_builtin__ void tex1DLayered(uchar4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uchar4");

__device__ __cudart_builtin__ void tex1DLayered(short *, cudaTextureObject_t, float, int) asm("__itex1DLayered_short");
__device__ __cudart_builtin__ void tex1DLayered(short1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_short1");
__device__ __cudart_builtin__ void tex1DLayered(short2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_short2");
__device__ __cudart_builtin__ void tex1DLayered(short4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_short4");
__device__ __cudart_builtin__ void tex1DLayered(unsigned short *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ushort");
__device__ __cudart_builtin__ void tex1DLayered(ushort1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ushort1");
__device__ __cudart_builtin__ void tex1DLayered(ushort2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ushort2");
__device__ __cudart_builtin__ void tex1DLayered(ushort4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ushort4");

__device__ __cudart_builtin__ void tex1DLayered(int *, cudaTextureObject_t, float, int) asm("__itex1DLayered_int");
__device__ __cudart_builtin__ void tex1DLayered(int1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_int1");
__device__ __cudart_builtin__ void tex1DLayered(int2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_int2");
__device__ __cudart_builtin__ void tex1DLayered(int4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_int4");
__device__ __cudart_builtin__ void tex1DLayered(unsigned int *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uint");
__device__ __cudart_builtin__ void tex1DLayered(uint1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uint1");
__device__ __cudart_builtin__ void tex1DLayered(uint2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uint2");
__device__ __cudart_builtin__ void tex1DLayered(uint4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1DLayered(long *, cudaTextureObject_t, float, int) asm("__itex1DLayered_long");
__device__ __cudart_builtin__ void tex1DLayered(long1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_long1");
__device__ __cudart_builtin__ void tex1DLayered(long2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_long2");
__device__ __cudart_builtin__ void tex1DLayered(long4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_long4");
__device__ __cudart_builtin__ void tex1DLayered(unsigned long *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ulong");
__device__ __cudart_builtin__ void tex1DLayered(ulong1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ulong1");
__device__ __cudart_builtin__ void tex1DLayered(ulong2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ulong2");
__device__ __cudart_builtin__ void tex1DLayered(ulong4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1DLayered(float *, cudaTextureObject_t, float, int) asm("__itex1DLayered_float");
__device__ __cudart_builtin__ void tex1DLayered(float1 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_float1");
__device__ __cudart_builtin__ void tex1DLayered(float2 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_float2");
__device__ __cudart_builtin__ void tex1DLayered(float4 *, cudaTextureObject_t, float, int) asm("__itex1DLayered_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1DLayered(cudaTextureObject_t texObject, float x, int layer)
{
  T ret;
  tex1DLayered(&ret, texObject, x, layer);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 2D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2DLayered(T *, cudaTextureObject_t, float, float, int) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2DLayered(char *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2DLayered(char *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex2DLayered(signed char *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_schar");
__device__ __cudart_builtin__ void tex2DLayered(char1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_char1");
__device__ __cudart_builtin__ void tex2DLayered(char2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_char2");
__device__ __cudart_builtin__ void tex2DLayered(char4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_char4");
__device__ __cudart_builtin__ void tex2DLayered(unsigned char *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uchar");
__device__ __cudart_builtin__ void tex2DLayered(uchar1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uchar1");
__device__ __cudart_builtin__ void tex2DLayered(uchar2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uchar2");
__device__ __cudart_builtin__ void tex2DLayered(uchar4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uchar4");

__device__ __cudart_builtin__ void tex2DLayered(short *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_short");
__device__ __cudart_builtin__ void tex2DLayered(short1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_short1");
__device__ __cudart_builtin__ void tex2DLayered(short2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_short2");
__device__ __cudart_builtin__ void tex2DLayered(short4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_short4");
__device__ __cudart_builtin__ void tex2DLayered(unsigned short *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ushort");
__device__ __cudart_builtin__ void tex2DLayered(ushort1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ushort1");
__device__ __cudart_builtin__ void tex2DLayered(ushort2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ushort2");
__device__ __cudart_builtin__ void tex2DLayered(ushort4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ushort4");

__device__ __cudart_builtin__ void tex2DLayered(int *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_int");
__device__ __cudart_builtin__ void tex2DLayered(int1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_int1");
__device__ __cudart_builtin__ void tex2DLayered(int2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_int2");
__device__ __cudart_builtin__ void tex2DLayered(int4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_int4");
__device__ __cudart_builtin__ void tex2DLayered(unsigned int *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uint");
__device__ __cudart_builtin__ void tex2DLayered(uint1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uint1");
__device__ __cudart_builtin__ void tex2DLayered(uint2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uint2");
__device__ __cudart_builtin__ void tex2DLayered(uint4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2DLayered(long *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_long");
__device__ __cudart_builtin__ void tex2DLayered(long1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_long1");
__device__ __cudart_builtin__ void tex2DLayered(long2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_long2");
__device__ __cudart_builtin__ void tex2DLayered(long4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_long4");
__device__ __cudart_builtin__ void tex2DLayered(unsigned long *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ulong");
__device__ __cudart_builtin__ void tex2DLayered(ulong1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ulong1");
__device__ __cudart_builtin__ void tex2DLayered(ulong2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ulong2");
__device__ __cudart_builtin__ void tex2DLayered(ulong4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2DLayered(float *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_float");
__device__ __cudart_builtin__ void tex2DLayered(float1 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_float1");
__device__ __cudart_builtin__ void tex2DLayered(float2 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_float2");
__device__ __cudart_builtin__ void tex2DLayered(float4 *, cudaTextureObject_t, float, float, int) asm("__itex2DLayered_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer)
{
  T ret;
  tex2DLayered(&ret, texObject, x, y, layer);
  return ret;
}


/*******************************************************************************
*                                                                              *
* Cubemap Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void texCubemap(T *, cudaTextureObject_t, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void texCubemap(char *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void texCubemap(char *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void texCubemap(signed char *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_schar");
__device__ __cudart_builtin__ void texCubemap(char1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_char1");
__device__ __cudart_builtin__ void texCubemap(char2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_char2");
__device__ __cudart_builtin__ void texCubemap(char4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_char4");
__device__ __cudart_builtin__ void texCubemap(unsigned char *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uchar");
__device__ __cudart_builtin__ void texCubemap(uchar1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uchar1");
__device__ __cudart_builtin__ void texCubemap(uchar2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uchar2");
__device__ __cudart_builtin__ void texCubemap(uchar4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uchar4");

__device__ __cudart_builtin__ void texCubemap(short *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_short");
__device__ __cudart_builtin__ void texCubemap(short1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_short1");
__device__ __cudart_builtin__ void texCubemap(short2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_short2");
__device__ __cudart_builtin__ void texCubemap(short4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_short4");
__device__ __cudart_builtin__ void texCubemap(unsigned short *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ushort");
__device__ __cudart_builtin__ void texCubemap(ushort1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ushort1");
__device__ __cudart_builtin__ void texCubemap(ushort2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ushort2");
__device__ __cudart_builtin__ void texCubemap(ushort4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ushort4");

__device__ __cudart_builtin__ void texCubemap(int *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_int");
__device__ __cudart_builtin__ void texCubemap(int1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_int1");
__device__ __cudart_builtin__ void texCubemap(int2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_int2");
__device__ __cudart_builtin__ void texCubemap(int4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_int4");
__device__ __cudart_builtin__ void texCubemap(unsigned int *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uint");
__device__ __cudart_builtin__ void texCubemap(uint1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uint1");
__device__ __cudart_builtin__ void texCubemap(uint2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uint2");
__device__ __cudart_builtin__ void texCubemap(uint4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void texCubemap(long *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_long");
__device__ __cudart_builtin__ void texCubemap(long1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_long1");
__device__ __cudart_builtin__ void texCubemap(long2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_long2");
__device__ __cudart_builtin__ void texCubemap(long4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_long4");
__device__ __cudart_builtin__ void texCubemap(unsigned long *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ulong");
__device__ __cudart_builtin__ void texCubemap(ulong1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ulong1");
__device__ __cudart_builtin__ void texCubemap(ulong2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ulong2");
__device__ __cudart_builtin__ void texCubemap(ulong4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void texCubemap(float *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_float");
__device__ __cudart_builtin__ void texCubemap(float1 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_float1");
__device__ __cudart_builtin__ void texCubemap(float2 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_float2");
__device__ __cudart_builtin__ void texCubemap(float4 *, cudaTextureObject_t, float, float, float) asm("__itexCubemap_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T texCubemap(cudaTextureObject_t texObject, float x, float y, float z)
{
  T ret;
  texCubemap(&ret, texObject, x, y, z);
  return ret;
}


/*******************************************************************************
*                                                                              *
* Cubemap Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void texCubemapLayered(T *, cudaTextureObject_t, float, float, float, int) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void texCubemapLayered(char *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void texCubemapLayered(char *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */

__device__ __cudart_builtin__ void texCubemapLayered(signed char *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_schar");
__device__ __cudart_builtin__ void texCubemapLayered(char1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_char1");
__device__ __cudart_builtin__ void texCubemapLayered(char2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_char2");
__device__ __cudart_builtin__ void texCubemapLayered(char4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_char4");
__device__ __cudart_builtin__ void texCubemapLayered(unsigned char *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uchar");
__device__ __cudart_builtin__ void texCubemapLayered(uchar1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uchar1");
__device__ __cudart_builtin__ void texCubemapLayered(uchar2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uchar2");
__device__ __cudart_builtin__ void texCubemapLayered(uchar4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uchar4");

__device__ __cudart_builtin__ void texCubemapLayered(short *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_short");
__device__ __cudart_builtin__ void texCubemapLayered(short1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_short1");
__device__ __cudart_builtin__ void texCubemapLayered(short2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_short2");
__device__ __cudart_builtin__ void texCubemapLayered(short4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_short4");
__device__ __cudart_builtin__ void texCubemapLayered(unsigned short *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ushort");
__device__ __cudart_builtin__ void texCubemapLayered(ushort1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ushort1");
__device__ __cudart_builtin__ void texCubemapLayered(ushort2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ushort2");
__device__ __cudart_builtin__ void texCubemapLayered(ushort4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ushort4");

__device__ __cudart_builtin__ void texCubemapLayered(int *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_int");
__device__ __cudart_builtin__ void texCubemapLayered(int1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_int1");
__device__ __cudart_builtin__ void texCubemapLayered(int2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_int2");
__device__ __cudart_builtin__ void texCubemapLayered(int4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_int4");
__device__ __cudart_builtin__ void texCubemapLayered(unsigned int *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uint");
__device__ __cudart_builtin__ void texCubemapLayered(uint1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uint1");
__device__ __cudart_builtin__ void texCubemapLayered(uint2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uint2");
__device__ __cudart_builtin__ void texCubemapLayered(uint4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void texCubemapLayered(long *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_long");
__device__ __cudart_builtin__ void texCubemapLayered(long1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_long1");
__device__ __cudart_builtin__ void texCubemapLayered(long2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_long2");
__device__ __cudart_builtin__ void texCubemapLayered(long4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_long4");
__device__ __cudart_builtin__ void texCubemapLayered(unsigned long *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ulong");
__device__ __cudart_builtin__ void texCubemapLayered(ulong1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ulong1");
__device__ __cudart_builtin__ void texCubemapLayered(ulong2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ulong2");
__device__ __cudart_builtin__ void texCubemapLayered(ulong4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void texCubemapLayered(float *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_float");
__device__ __cudart_builtin__ void texCubemapLayered(float1 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_float1");
__device__ __cudart_builtin__ void texCubemapLayered(float2 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_float2");
__device__ __cudart_builtin__ void texCubemapLayered(float4 *, cudaTextureObject_t, float, float, float, int) asm("__itexCubemapLayered_float4");

#endif /* __CUDA_ARCH__ */
template <class T>
static __device__ T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer)
{
  T ret;
  texCubemapLayered(&ret, texObject, x, y, z, layer);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 2D Texture indirect gather functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2Dgather(T *, cudaTextureObject_t, float, float, int = 0) { }
#else /* __CUDA_ARCH__ */

__device__ __cudart_builtin__ void tex2Dgather(char *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_char");
__device__ __cudart_builtin__ void tex2Dgather(signed char *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_schar");
__device__ __cudart_builtin__ void tex2Dgather(char1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_char1");
__device__ __cudart_builtin__ void tex2Dgather(char2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_char2");
__device__ __cudart_builtin__ void tex2Dgather(char4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_char4");
__device__ __cudart_builtin__ void tex2Dgather(unsigned char *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uchar");
__device__ __cudart_builtin__ void tex2Dgather(uchar1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uchar1");
__device__ __cudart_builtin__ void tex2Dgather(uchar2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uchar2");
__device__ __cudart_builtin__ void tex2Dgather(uchar4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uchar4");

__device__ __cudart_builtin__ void tex2Dgather(short *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_short");
__device__ __cudart_builtin__ void tex2Dgather(short1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_short1");
__device__ __cudart_builtin__ void tex2Dgather(short2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_short2");
__device__ __cudart_builtin__ void tex2Dgather(short4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_short4");
__device__ __cudart_builtin__ void tex2Dgather(unsigned short *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ushort");
__device__ __cudart_builtin__ void tex2Dgather(ushort1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ushort1");
__device__ __cudart_builtin__ void tex2Dgather(ushort2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ushort2");
__device__ __cudart_builtin__ void tex2Dgather(ushort4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ushort4");

__device__ __cudart_builtin__ void tex2Dgather(int *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_int");
__device__ __cudart_builtin__ void tex2Dgather(int1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_int1");
__device__ __cudart_builtin__ void tex2Dgather(int2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_int2");
__device__ __cudart_builtin__ void tex2Dgather(int4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_int4");
__device__ __cudart_builtin__ void tex2Dgather(unsigned int *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uint");
__device__ __cudart_builtin__ void tex2Dgather(uint1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uint1");
__device__ __cudart_builtin__ void tex2Dgather(uint2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uint2");
__device__ __cudart_builtin__ void tex2Dgather(uint4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_uint4");
#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2Dgather(long *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_long");
__device__ __cudart_builtin__ void tex2Dgather(long1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_long1");
__device__ __cudart_builtin__ void tex2Dgather(long2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_long2");
__device__ __cudart_builtin__ void tex2Dgather(long4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_long4");
__device__ __cudart_builtin__ void tex2Dgather(unsigned long *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ulong");
__device__ __cudart_builtin__ void tex2Dgather(ulong1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ulong1");
__device__ __cudart_builtin__ void tex2Dgather(ulong2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ulong2");
__device__ __cudart_builtin__ void tex2Dgather(ulong4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_ulong4");
#endif /* !defined(__LP64__) */
__device__ __cudart_builtin__ void tex2Dgather(float *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_float");
__device__ __cudart_builtin__ void tex2Dgather(float1 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_float1");
__device__ __cudart_builtin__ void tex2Dgather(float2 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_float2");
__device__ __cudart_builtin__ void tex2Dgather(float4 *, cudaTextureObject_t, float, float, int = 0) asm("__itex2Dgather_float4");

#endif /* __CUDA_ARCH__ */
template <class T>
static __device__ T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0)
{
  T ret;
  tex2Dgather(&ret, to, x, y, comp);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 1D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1DLod(T *, cudaTextureObject_t, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1DLod(char *, cudaTextureObject_t, float, float) asm("__itex1DLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1DLod(char *, cudaTextureObject_t, float, float) asm("__itex1DLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex1DLod(signed char *, cudaTextureObject_t, float, float) asm("__itex1DLod_schar");
__device__ __cudart_builtin__ void tex1DLod(char1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_char1");
__device__ __cudart_builtin__ void tex1DLod(char2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_char2");
__device__ __cudart_builtin__ void tex1DLod(char4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_char4");
__device__ __cudart_builtin__ void tex1DLod(unsigned char *, cudaTextureObject_t, float, float) asm("__itex1DLod_uchar");
__device__ __cudart_builtin__ void tex1DLod(uchar1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uchar1");
__device__ __cudart_builtin__ void tex1DLod(uchar2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uchar2");
__device__ __cudart_builtin__ void tex1DLod(uchar4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uchar4");

__device__ __cudart_builtin__ void tex1DLod(short *, cudaTextureObject_t, float, float) asm("__itex1DLod_short");
__device__ __cudart_builtin__ void tex1DLod(short1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_short1");
__device__ __cudart_builtin__ void tex1DLod(short2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_short2");
__device__ __cudart_builtin__ void tex1DLod(short4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_short4");
__device__ __cudart_builtin__ void tex1DLod(unsigned short *, cudaTextureObject_t, float, float) asm("__itex1DLod_ushort");
__device__ __cudart_builtin__ void tex1DLod(ushort1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ushort1");
__device__ __cudart_builtin__ void tex1DLod(ushort2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ushort2");
__device__ __cudart_builtin__ void tex1DLod(ushort4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ushort4");

__device__ __cudart_builtin__ void tex1DLod(int *, cudaTextureObject_t, float, float) asm("__itex1DLod_int");
__device__ __cudart_builtin__ void tex1DLod(int1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_int1");
__device__ __cudart_builtin__ void tex1DLod(int2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_int2");
__device__ __cudart_builtin__ void tex1DLod(int4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_int4");
__device__ __cudart_builtin__ void tex1DLod(unsigned int *, cudaTextureObject_t, float, float) asm("__itex1DLod_uint");
__device__ __cudart_builtin__ void tex1DLod(uint1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uint1");
__device__ __cudart_builtin__ void tex1DLod(uint2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uint2");
__device__ __cudart_builtin__ void tex1DLod(uint4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1DLod(long *, cudaTextureObject_t, float, float) asm("__itex1DLod_long");
__device__ __cudart_builtin__ void tex1DLod(long1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_long1");
__device__ __cudart_builtin__ void tex1DLod(long2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_long2");
__device__ __cudart_builtin__ void tex1DLod(long4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_long4");
__device__ __cudart_builtin__ void tex1DLod(unsigned long *, cudaTextureObject_t, float, float) asm("__itex1DLod_ulong");
__device__ __cudart_builtin__ void tex1DLod(ulong1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ulong1");
__device__ __cudart_builtin__ void tex1DLod(ulong2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ulong2");
__device__ __cudart_builtin__ void tex1DLod(ulong4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1DLod(float *, cudaTextureObject_t, float, float) asm("__itex1DLod_float");
__device__ __cudart_builtin__ void tex1DLod(float1 *, cudaTextureObject_t, float, float) asm("__itex1DLod_float1");
__device__ __cudart_builtin__ void tex1DLod(float2 *, cudaTextureObject_t, float, float) asm("__itex1DLod_float2");
__device__ __cudart_builtin__ void tex1DLod(float4 *, cudaTextureObject_t, float, float) asm("__itex1DLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1DLod(cudaTextureObject_t texObject, float x, float level)
{
  T ret;
  tex1DLod(&ret, texObject, x, level);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 2D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2DLod(T *, cudaTextureObject_t, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2DLod(char *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2DLod(char *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */

__device__ __cudart_builtin__ void tex2DLod(signed char *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_schar");
__device__ __cudart_builtin__ void tex2DLod(char1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_char1");
__device__ __cudart_builtin__ void tex2DLod(char2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_char2");
__device__ __cudart_builtin__ void tex2DLod(char4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_char4");
__device__ __cudart_builtin__ void tex2DLod(unsigned char *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uchar");
__device__ __cudart_builtin__ void tex2DLod(uchar1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uchar1");
__device__ __cudart_builtin__ void tex2DLod(uchar2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uchar2");
__device__ __cudart_builtin__ void tex2DLod(uchar4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uchar4");

__device__ __cudart_builtin__ void tex2DLod(short *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_short");
__device__ __cudart_builtin__ void tex2DLod(short1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_short1");
__device__ __cudart_builtin__ void tex2DLod(short2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_short2");
__device__ __cudart_builtin__ void tex2DLod(short4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_short4");
__device__ __cudart_builtin__ void tex2DLod(unsigned short *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ushort");
__device__ __cudart_builtin__ void tex2DLod(ushort1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ushort1");
__device__ __cudart_builtin__ void tex2DLod(ushort2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ushort2");
__device__ __cudart_builtin__ void tex2DLod(ushort4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ushort4");

__device__ __cudart_builtin__ void tex2DLod(int *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_int");
__device__ __cudart_builtin__ void tex2DLod(int1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_int1");
__device__ __cudart_builtin__ void tex2DLod(int2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_int2");
__device__ __cudart_builtin__ void tex2DLod(int4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_int4");
__device__ __cudart_builtin__ void tex2DLod(unsigned int *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uint");
__device__ __cudart_builtin__ void tex2DLod(uint1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uint1");
__device__ __cudart_builtin__ void tex2DLod(uint2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uint2");
__device__ __cudart_builtin__ void tex2DLod(uint4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2DLod(long *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_long");
__device__ __cudart_builtin__ void tex2DLod(long1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_long1");
__device__ __cudart_builtin__ void tex2DLod(long2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_long2");
__device__ __cudart_builtin__ void tex2DLod(long4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_long4");
__device__ __cudart_builtin__ void tex2DLod(unsigned long *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ulong");
__device__ __cudart_builtin__ void tex2DLod(ulong1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ulong1");
__device__ __cudart_builtin__ void tex2DLod(ulong2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ulong2");
__device__ __cudart_builtin__ void tex2DLod(ulong4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2DLod(float *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_float");
__device__ __cudart_builtin__ void tex2DLod(float1 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_float1");
__device__ __cudart_builtin__ void tex2DLod(float2 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_float2");
__device__ __cudart_builtin__ void tex2DLod(float4 *, cudaTextureObject_t, float, float, float) asm("__itex2DLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level)
{
  T ret;
  tex2DLod(&ret, texObject, x, y, level);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 3D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex3DLod(T *, cudaTextureObject_t, float, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex3DLod(char *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex3DLod(char *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex3DLod(signed char *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_schar");
__device__ __cudart_builtin__ void tex3DLod(char1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_char1");
__device__ __cudart_builtin__ void tex3DLod(char2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_char2");
__device__ __cudart_builtin__ void tex3DLod(char4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_char4");
__device__ __cudart_builtin__ void tex3DLod(unsigned char *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uchar");
__device__ __cudart_builtin__ void tex3DLod(uchar1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uchar1");
__device__ __cudart_builtin__ void tex3DLod(uchar2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uchar2");
__device__ __cudart_builtin__ void tex3DLod(uchar4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uchar4");

__device__ __cudart_builtin__ void tex3DLod(short *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_short");
__device__ __cudart_builtin__ void tex3DLod(short1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_short1");
__device__ __cudart_builtin__ void tex3DLod(short2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_short2");
__device__ __cudart_builtin__ void tex3DLod(short4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_short4");
__device__ __cudart_builtin__ void tex3DLod(unsigned short *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ushort");
__device__ __cudart_builtin__ void tex3DLod(ushort1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ushort1");
__device__ __cudart_builtin__ void tex3DLod(ushort2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ushort2");
__device__ __cudart_builtin__ void tex3DLod(ushort4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ushort4");

__device__ __cudart_builtin__ void tex3DLod(int *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_int");
__device__ __cudart_builtin__ void tex3DLod(int1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_int1");
__device__ __cudart_builtin__ void tex3DLod(int2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_int2");
__device__ __cudart_builtin__ void tex3DLod(int4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_int4");
__device__ __cudart_builtin__ void tex3DLod(unsigned int *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uint");
__device__ __cudart_builtin__ void tex3DLod(uint1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uint1");
__device__ __cudart_builtin__ void tex3DLod(uint2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uint2");
__device__ __cudart_builtin__ void tex3DLod(uint4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex3DLod(long *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_long");
__device__ __cudart_builtin__ void tex3DLod(long1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_long1");
__device__ __cudart_builtin__ void tex3DLod(long2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_long2");
__device__ __cudart_builtin__ void tex3DLod(long4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_long4");
__device__ __cudart_builtin__ void tex3DLod(unsigned long *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ulong");
__device__ __cudart_builtin__ void tex3DLod(ulong1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ulong1");
__device__ __cudart_builtin__ void tex3DLod(ulong2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ulong2");
__device__ __cudart_builtin__ void tex3DLod(ulong4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex3DLod(float *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_float");
__device__ __cudart_builtin__ void tex3DLod(float1 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_float1");
__device__ __cudart_builtin__ void tex3DLod(float2 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_float2");
__device__ __cudart_builtin__ void tex3DLod(float4 *, cudaTextureObject_t texObject, float, float, float, float) asm("__itex3DLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
  T ret;
  tex3DLod(&ret, texObject, x, y, z, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1DLayeredLod(T *, cudaTextureObject_t, float, int, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1DLayeredLod(char *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1DLayeredLod(char *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex1DLayeredLod(signed char *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_schar");
__device__ __cudart_builtin__ void tex1DLayeredLod(char1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_char1");
__device__ __cudart_builtin__ void tex1DLayeredLod(char2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_char2");
__device__ __cudart_builtin__ void tex1DLayeredLod(char4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_char4");
__device__ __cudart_builtin__ void tex1DLayeredLod(unsigned char *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uchar");
__device__ __cudart_builtin__ void tex1DLayeredLod(uchar1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uchar1");
__device__ __cudart_builtin__ void tex1DLayeredLod(uchar2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uchar2");
__device__ __cudart_builtin__ void tex1DLayeredLod(uchar4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uchar4");

__device__ __cudart_builtin__ void tex1DLayeredLod(short *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_short");
__device__ __cudart_builtin__ void tex1DLayeredLod(short1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_short1");
__device__ __cudart_builtin__ void tex1DLayeredLod(short2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_short2");
__device__ __cudart_builtin__ void tex1DLayeredLod(short4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_short4");
__device__ __cudart_builtin__ void tex1DLayeredLod(unsigned short *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ushort");
__device__ __cudart_builtin__ void tex1DLayeredLod(ushort1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ushort1");
__device__ __cudart_builtin__ void tex1DLayeredLod(ushort2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ushort2");
__device__ __cudart_builtin__ void tex1DLayeredLod(ushort4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ushort4");

__device__ __cudart_builtin__ void tex1DLayeredLod(int *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_int");
__device__ __cudart_builtin__ void tex1DLayeredLod(int1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_int1");
__device__ __cudart_builtin__ void tex1DLayeredLod(int2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_int2");
__device__ __cudart_builtin__ void tex1DLayeredLod(int4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_int4");
__device__ __cudart_builtin__ void tex1DLayeredLod(unsigned int *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uint");
__device__ __cudart_builtin__ void tex1DLayeredLod(uint1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uint1");
__device__ __cudart_builtin__ void tex1DLayeredLod(uint2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uint2");
__device__ __cudart_builtin__ void tex1DLayeredLod(uint4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1DLayeredLod(long *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_long");
__device__ __cudart_builtin__ void tex1DLayeredLod(long1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_long1");
__device__ __cudart_builtin__ void tex1DLayeredLod(long2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_long2");
__device__ __cudart_builtin__ void tex1DLayeredLod(long4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_long4");
__device__ __cudart_builtin__ void tex1DLayeredLod(unsigned long *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ulong");
__device__ __cudart_builtin__ void tex1DLayeredLod(ulong1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ulong1");
__device__ __cudart_builtin__ void tex1DLayeredLod(ulong2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ulong2");
__device__ __cudart_builtin__ void tex1DLayeredLod(ulong4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1DLayeredLod(float *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_float");
__device__ __cudart_builtin__ void tex1DLayeredLod(float1 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_float1");
__device__ __cudart_builtin__ void tex1DLayeredLod(float2 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_float2");
__device__ __cudart_builtin__ void tex1DLayeredLod(float4 *, cudaTextureObject_t, float, int, float) asm("__itex1DLayeredLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level)
{
  T ret;
  tex1DLayeredLod(&ret, texObject, x, layer, level);
  return ret;
}


/*******************************************************************************
*                                                                              *
* 2D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2DLayeredLod(T *, cudaTextureObject_t, float, float, int, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2DLayeredLod(char *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2DLayeredLod(char *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex2DLayeredLod(signed char *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_schar");
__device__ __cudart_builtin__ void tex2DLayeredLod(char1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_char1");
__device__ __cudart_builtin__ void tex2DLayeredLod(char2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_char2");
__device__ __cudart_builtin__ void tex2DLayeredLod(char4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_char4");
__device__ __cudart_builtin__ void tex2DLayeredLod(unsigned char *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uchar");
__device__ __cudart_builtin__ void tex2DLayeredLod(uchar1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uchar1");
__device__ __cudart_builtin__ void tex2DLayeredLod(uchar2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uchar2");
__device__ __cudart_builtin__ void tex2DLayeredLod(uchar4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uchar4");

__device__ __cudart_builtin__ void tex2DLayeredLod(short *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_short");
__device__ __cudart_builtin__ void tex2DLayeredLod(short1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_short1");
__device__ __cudart_builtin__ void tex2DLayeredLod(short2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_short2");
__device__ __cudart_builtin__ void tex2DLayeredLod(short4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_short4");
__device__ __cudart_builtin__ void tex2DLayeredLod(unsigned short *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ushort");
__device__ __cudart_builtin__ void tex2DLayeredLod(ushort1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ushort1");
__device__ __cudart_builtin__ void tex2DLayeredLod(ushort2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ushort2");
__device__ __cudart_builtin__ void tex2DLayeredLod(ushort4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ushort4");

__device__ __cudart_builtin__ void tex2DLayeredLod(int *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_int");
__device__ __cudart_builtin__ void tex2DLayeredLod(int1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_int1");
__device__ __cudart_builtin__ void tex2DLayeredLod(int2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_int2");
__device__ __cudart_builtin__ void tex2DLayeredLod(int4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_int4");
__device__ __cudart_builtin__ void tex2DLayeredLod(unsigned int *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uint");
__device__ __cudart_builtin__ void tex2DLayeredLod(uint1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uint1");
__device__ __cudart_builtin__ void tex2DLayeredLod(uint2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uint2");
__device__ __cudart_builtin__ void tex2DLayeredLod(uint4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2DLayeredLod(long *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_long");
__device__ __cudart_builtin__ void tex2DLayeredLod(long1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_long1");
__device__ __cudart_builtin__ void tex2DLayeredLod(long2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_long2");
__device__ __cudart_builtin__ void tex2DLayeredLod(long4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_long4");
__device__ __cudart_builtin__ void tex2DLayeredLod(unsigned long *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ulong");
__device__ __cudart_builtin__ void tex2DLayeredLod(ulong1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ulong1");
__device__ __cudart_builtin__ void tex2DLayeredLod(ulong2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ulong2");
__device__ __cudart_builtin__ void tex2DLayeredLod(ulong4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2DLayeredLod(float *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_float");
__device__ __cudart_builtin__ void tex2DLayeredLod(float1 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_float1");
__device__ __cudart_builtin__ void tex2DLayeredLod(float2 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_float2");
__device__ __cudart_builtin__ void tex2DLayeredLod(float4 *, cudaTextureObject_t, float, float, int, float) asm("__itex2DLayeredLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level)
{
  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void texCubemapLod(T *, cudaTextureObject_t, float, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void texCubemapLod(char *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void texCubemapLod(char *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void texCubemapLod(signed char *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_schar");
__device__ __cudart_builtin__ void texCubemapLod(char1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_char1");
__device__ __cudart_builtin__ void texCubemapLod(char2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_char2");
__device__ __cudart_builtin__ void texCubemapLod(char4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_char4");
__device__ __cudart_builtin__ void texCubemapLod(unsigned char *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uchar");
__device__ __cudart_builtin__ void texCubemapLod(uchar1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uchar1");
__device__ __cudart_builtin__ void texCubemapLod(uchar2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uchar2");
__device__ __cudart_builtin__ void texCubemapLod(uchar4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uchar4");

__device__ __cudart_builtin__ void texCubemapLod(short *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_short");
__device__ __cudart_builtin__ void texCubemapLod(short1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_short1");
__device__ __cudart_builtin__ void texCubemapLod(short2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_short2");
__device__ __cudart_builtin__ void texCubemapLod(short4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_short4");
__device__ __cudart_builtin__ void texCubemapLod(unsigned short *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ushort");
__device__ __cudart_builtin__ void texCubemapLod(ushort1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ushort1");
__device__ __cudart_builtin__ void texCubemapLod(ushort2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ushort2");
__device__ __cudart_builtin__ void texCubemapLod(ushort4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ushort4");

__device__ __cudart_builtin__ void texCubemapLod(int *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_int");
__device__ __cudart_builtin__ void texCubemapLod(int1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_int1");
__device__ __cudart_builtin__ void texCubemapLod(int2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_int2");
__device__ __cudart_builtin__ void texCubemapLod(int4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_int4");
__device__ __cudart_builtin__ void texCubemapLod(unsigned int *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uint");
__device__ __cudart_builtin__ void texCubemapLod(uint1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uint1");
__device__ __cudart_builtin__ void texCubemapLod(uint2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uint2");
__device__ __cudart_builtin__ void texCubemapLod(uint4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void texCubemapLod(long *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_long");
__device__ __cudart_builtin__ void texCubemapLod(long1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_long1");
__device__ __cudart_builtin__ void texCubemapLod(long2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_long2");
__device__ __cudart_builtin__ void texCubemapLod(long4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_long4");
__device__ __cudart_builtin__ void texCubemapLod(unsigned long *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ulong");
__device__ __cudart_builtin__ void texCubemapLod(ulong1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ulong1");
__device__ __cudart_builtin__ void texCubemapLod(ulong2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ulong2");
__device__ __cudart_builtin__ void texCubemapLod(ulong4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void texCubemapLod(float *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_float");
__device__ __cudart_builtin__ void texCubemapLod(float1 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_float1");
__device__ __cudart_builtin__ void texCubemapLod(float2 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_float2");
__device__ __cudart_builtin__ void texCubemapLod(float4 *, cudaTextureObject_t, float, float, float, float) asm("__itexCubemapLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
  T ret;
  texCubemapLod(&ret, texObject, x, y, z, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void texCubemapLayeredLod(T *, cudaTextureObject_t, float, float, float, int, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void texCubemapLayeredLod(char *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void texCubemapLayeredLod(char *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void texCubemapLayeredLod(signed char *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_schar");
__device__ __cudart_builtin__ void texCubemapLayeredLod(char1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_char1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(char2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_char2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(char4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_char4");
__device__ __cudart_builtin__ void texCubemapLayeredLod(unsigned char *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uchar");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uchar1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uchar1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uchar2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uchar2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uchar4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uchar4");

__device__ __cudart_builtin__ void texCubemapLayeredLod(short *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_short");
__device__ __cudart_builtin__ void texCubemapLayeredLod(short1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_short1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(short2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_short2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(short4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_short4");
__device__ __cudart_builtin__ void texCubemapLayeredLod(unsigned short *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ushort");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ushort1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ushort1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ushort2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ushort2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ushort4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ushort4");

__device__ __cudart_builtin__ void texCubemapLayeredLod(int *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_int");
__device__ __cudart_builtin__ void texCubemapLayeredLod(int1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_int1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(int2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_int2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(int4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_int4");
__device__ __cudart_builtin__ void texCubemapLayeredLod(unsigned int *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uint");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uint1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uint1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uint2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uint2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(uint4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void texCubemapLayeredLod(long *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_long");
__device__ __cudart_builtin__ void texCubemapLayeredLod(long1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_long1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(long2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_long2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(long4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_long4");
__device__ __cudart_builtin__ void texCubemapLayeredLod(unsigned long *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ulong");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ulong1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ulong1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ulong2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ulong2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(ulong4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void texCubemapLayeredLod(float *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_float");
__device__ __cudart_builtin__ void texCubemapLayeredLod(float1 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_float1");
__device__ __cudart_builtin__ void texCubemapLayeredLod(float2 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_float2");
__device__ __cudart_builtin__ void texCubemapLayeredLod(float4 *, cudaTextureObject_t, float, float, float, int, float) asm("__itexCubemapLayeredLod_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  T ret;
  texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1DGrad(T *, cudaTextureObject_t, float, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1DGrad(char *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1DGrad(char *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */

__device__ __cudart_builtin__ void tex1DGrad(signed char *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_schar");
__device__ __cudart_builtin__ void tex1DGrad(char1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_char1");
__device__ __cudart_builtin__ void tex1DGrad(char2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_char2");
__device__ __cudart_builtin__ void tex1DGrad(char4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_char4");
__device__ __cudart_builtin__ void tex1DGrad(unsigned char *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uchar");
__device__ __cudart_builtin__ void tex1DGrad(uchar1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uchar1");
__device__ __cudart_builtin__ void tex1DGrad(uchar2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uchar2");
__device__ __cudart_builtin__ void tex1DGrad(uchar4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uchar4");

__device__ __cudart_builtin__ void tex1DGrad(short *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_short");
__device__ __cudart_builtin__ void tex1DGrad(short1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_short1");
__device__ __cudart_builtin__ void tex1DGrad(short2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_short2");
__device__ __cudart_builtin__ void tex1DGrad(short4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_short4");
__device__ __cudart_builtin__ void tex1DGrad(unsigned short *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ushort");
__device__ __cudart_builtin__ void tex1DGrad(ushort1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ushort1");
__device__ __cudart_builtin__ void tex1DGrad(ushort2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ushort2");
__device__ __cudart_builtin__ void tex1DGrad(ushort4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ushort4");

__device__ __cudart_builtin__ void tex1DGrad(int *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_int");
__device__ __cudart_builtin__ void tex1DGrad(int1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_int1");
__device__ __cudart_builtin__ void tex1DGrad(int2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_int2");
__device__ __cudart_builtin__ void tex1DGrad(int4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_int4");
__device__ __cudart_builtin__ void tex1DGrad(unsigned int *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uint");
__device__ __cudart_builtin__ void tex1DGrad(uint1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uint1");
__device__ __cudart_builtin__ void tex1DGrad(uint2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uint2");
__device__ __cudart_builtin__ void tex1DGrad(uint4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1DGrad(long *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_long");
__device__ __cudart_builtin__ void tex1DGrad(long1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_long1");
__device__ __cudart_builtin__ void tex1DGrad(long2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_long2");
__device__ __cudart_builtin__ void tex1DGrad(long4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_long4");
__device__ __cudart_builtin__ void tex1DGrad(unsigned long *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ulong");
__device__ __cudart_builtin__ void tex1DGrad(ulong1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ulong1");
__device__ __cudart_builtin__ void tex1DGrad(ulong2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ulong2");
__device__ __cudart_builtin__ void tex1DGrad(ulong4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1DGrad(float *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_float");
__device__ __cudart_builtin__ void tex1DGrad(float1 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_float1");
__device__ __cudart_builtin__ void tex1DGrad(float2 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_float2");
__device__ __cudart_builtin__ void tex1DGrad(float4 *, cudaTextureObject_t, float, float, float) asm("__itex1DGrad_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  T ret;
  tex1DGrad(&ret, texObject, x, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2DGrad(T *, cudaTextureObject_t, float, float, float2, float2) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2DGrad(char *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2DGrad(char *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex2DGrad(signed char *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_schar");
__device__ __cudart_builtin__ void tex2DGrad(char1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_char1");
__device__ __cudart_builtin__ void tex2DGrad(char2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_char2");
__device__ __cudart_builtin__ void tex2DGrad(char4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_char4");
__device__ __cudart_builtin__ void tex2DGrad(unsigned char *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uchar");
__device__ __cudart_builtin__ void tex2DGrad(uchar1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uchar1");
__device__ __cudart_builtin__ void tex2DGrad(uchar2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uchar2");
__device__ __cudart_builtin__ void tex2DGrad(uchar4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uchar4");

__device__ __cudart_builtin__ void tex2DGrad(short *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_short");
__device__ __cudart_builtin__ void tex2DGrad(short1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_short1");
__device__ __cudart_builtin__ void tex2DGrad(short2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_short2");
__device__ __cudart_builtin__ void tex2DGrad(short4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_short4");
__device__ __cudart_builtin__ void tex2DGrad(unsigned short *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ushort");
__device__ __cudart_builtin__ void tex2DGrad(ushort1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ushort1");
__device__ __cudart_builtin__ void tex2DGrad(ushort2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ushort2");
__device__ __cudart_builtin__ void tex2DGrad(ushort4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ushort4");

__device__ __cudart_builtin__ void tex2DGrad(int *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_int");
__device__ __cudart_builtin__ void tex2DGrad(int1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_int1");
__device__ __cudart_builtin__ void tex2DGrad(int2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_int2");
__device__ __cudart_builtin__ void tex2DGrad(int4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_int4");
__device__ __cudart_builtin__ void tex2DGrad(unsigned int *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uint");
__device__ __cudart_builtin__ void tex2DGrad(uint1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uint1");
__device__ __cudart_builtin__ void tex2DGrad(uint2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uint2");
__device__ __cudart_builtin__ void tex2DGrad(uint4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2DGrad(long *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_long");
__device__ __cudart_builtin__ void tex2DGrad(long1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_long1");
__device__ __cudart_builtin__ void tex2DGrad(long2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_long2");
__device__ __cudart_builtin__ void tex2DGrad(long4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_long4");
__device__ __cudart_builtin__ void tex2DGrad(unsigned long *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ulong");
__device__ __cudart_builtin__ void tex2DGrad(ulong1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ulong1");
__device__ __cudart_builtin__ void tex2DGrad(ulong2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ulong2");
__device__ __cudart_builtin__ void tex2DGrad(ulong4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2DGrad(float *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_float");
__device__ __cudart_builtin__ void tex2DGrad(float1 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_float1");
__device__ __cudart_builtin__ void tex2DGrad(float2 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_float2");
__device__ __cudart_builtin__ void tex2DGrad(float4 *, cudaTextureObject_t, float, float, float2, float2) asm("__itex2DGrad_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 3D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex3DGrad(T *, cudaTextureObject_t, float, float, float, float4, float4) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex3DGrad(char *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex3DGrad(char *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex3DGrad(signed char *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_schar");
__device__ __cudart_builtin__ void tex3DGrad(char1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_char1");
__device__ __cudart_builtin__ void tex3DGrad(char2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_char2");
__device__ __cudart_builtin__ void tex3DGrad(char4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_char4");
__device__ __cudart_builtin__ void tex3DGrad(unsigned char *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uchar");
__device__ __cudart_builtin__ void tex3DGrad(uchar1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uchar1");
__device__ __cudart_builtin__ void tex3DGrad(uchar2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uchar2");
__device__ __cudart_builtin__ void tex3DGrad(uchar4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uchar4");

__device__ __cudart_builtin__ void tex3DGrad(short *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_short");
__device__ __cudart_builtin__ void tex3DGrad(short1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_short1");
__device__ __cudart_builtin__ void tex3DGrad(short2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_short2");
__device__ __cudart_builtin__ void tex3DGrad(short4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_short4");
__device__ __cudart_builtin__ void tex3DGrad(unsigned short *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ushort");
__device__ __cudart_builtin__ void tex3DGrad(ushort1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ushort1");
__device__ __cudart_builtin__ void tex3DGrad(ushort2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ushort2");
__device__ __cudart_builtin__ void tex3DGrad(ushort4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ushort4");

__device__ __cudart_builtin__ void tex3DGrad(int *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_int");
__device__ __cudart_builtin__ void tex3DGrad(int1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_int1");
__device__ __cudart_builtin__ void tex3DGrad(int2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_int2");
__device__ __cudart_builtin__ void tex3DGrad(int4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_int4");
__device__ __cudart_builtin__ void tex3DGrad(unsigned int *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uint");
__device__ __cudart_builtin__ void tex3DGrad(uint1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uint1");
__device__ __cudart_builtin__ void tex3DGrad(uint2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uint2");
__device__ __cudart_builtin__ void tex3DGrad(uint4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex3DGrad(long *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_long");
__device__ __cudart_builtin__ void tex3DGrad(long1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_long1");
__device__ __cudart_builtin__ void tex3DGrad(long2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_long2");
__device__ __cudart_builtin__ void tex3DGrad(long4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_long4");
__device__ __cudart_builtin__ void tex3DGrad(unsigned long *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ulong");
__device__ __cudart_builtin__ void tex3DGrad(ulong1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ulong1");
__device__ __cudart_builtin__ void tex3DGrad(ulong2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ulong2");
__device__ __cudart_builtin__ void tex3DGrad(ulong4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex3DGrad(float *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_float");
__device__ __cudart_builtin__ void tex3DGrad(float1 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_float1");
__device__ __cudart_builtin__ void tex3DGrad(float2 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_float2");
__device__ __cudart_builtin__ void tex3DGrad(float4 *, cudaTextureObject_t,float, float, float, float4, float4) asm("__itex3DGrad_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex1DLayeredGrad(T *, cudaTextureObject_t, float, int, float, float) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex1DLayeredGrad(char *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex1DLayeredGrad(char *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */

__device__ __cudart_builtin__ void tex1DLayeredGrad(signed char *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_schar");
__device__ __cudart_builtin__ void tex1DLayeredGrad(char1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_char1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(char2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_char2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(char4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_char4");
__device__ __cudart_builtin__ void tex1DLayeredGrad(unsigned char *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uchar");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uchar1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uchar1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uchar2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uchar2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uchar4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uchar4");

__device__ __cudart_builtin__ void tex1DLayeredGrad(short *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_short");
__device__ __cudart_builtin__ void tex1DLayeredGrad(short1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_short1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(short2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_short2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(short4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_short4");
__device__ __cudart_builtin__ void tex1DLayeredGrad(unsigned short *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ushort");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ushort1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ushort1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ushort2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ushort2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ushort4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ushort4");

__device__ __cudart_builtin__ void tex1DLayeredGrad(int *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_int");
__device__ __cudart_builtin__ void tex1DLayeredGrad(int1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_int1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(int2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_int2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(int4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_int4");
__device__ __cudart_builtin__ void tex1DLayeredGrad(unsigned int *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uint");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uint1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uint1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uint2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uint2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(uint4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex1DLayeredGrad(long *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_long");
__device__ __cudart_builtin__ void tex1DLayeredGrad(long1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_long1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(long2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_long2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(long4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_long4");
__device__ __cudart_builtin__ void tex1DLayeredGrad(unsigned long *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ulong");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ulong1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ulong1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ulong2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ulong2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(ulong4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex1DLayeredGrad(float *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_float");
__device__ __cudart_builtin__ void tex1DLayeredGrad(float1 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_float1");
__device__ __cudart_builtin__ void tex1DLayeredGrad(float2 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_float2");
__device__ __cudart_builtin__ void tex1DLayeredGrad(float4 *, cudaTextureObject_t, float, int, float, float) asm("__itex1DLayeredGrad_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  T ret;
  tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/
#ifndef __CUDA_ARCH__
template <typename T>
static __device__ void tex2DLayeredGrad(T *, cudaTextureObject_t, float, float, int, float2, float2) { }
#else /* __CUDA_ARCH__ */

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
__device__ __cudart_builtin__ void tex2DLayeredGrad(char *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_char_as_uchar");
#else  /* !(defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)) */
__device__ __cudart_builtin__ void tex2DLayeredGrad(char *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_char_as_schar");
#endif /* defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__) */
__device__ __cudart_builtin__ void tex2DLayeredGrad(signed char *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_schar");
__device__ __cudart_builtin__ void tex2DLayeredGrad(char1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_char1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(char2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_char2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(char4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_char4");
__device__ __cudart_builtin__ void tex2DLayeredGrad(unsigned char *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uchar");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uchar1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uchar1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uchar2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uchar2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uchar4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uchar4");

__device__ __cudart_builtin__ void tex2DLayeredGrad(short *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_short");
__device__ __cudart_builtin__ void tex2DLayeredGrad(short1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_short1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(short2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_short2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(short4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_short4");
__device__ __cudart_builtin__ void tex2DLayeredGrad(unsigned short *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ushort");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ushort1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ushort1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ushort2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ushort2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ushort4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ushort4");

__device__ __cudart_builtin__ void tex2DLayeredGrad(int *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_int");
__device__ __cudart_builtin__ void tex2DLayeredGrad(int1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_int1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(int2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_int2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(int4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_int4");
__device__ __cudart_builtin__ void tex2DLayeredGrad(unsigned int *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uint");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uint1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uint1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uint2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uint2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(uint4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_uint4");

#if !defined(__LP64__)
__device__ __cudart_builtin__ void tex2DLayeredGrad(long *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_long");
__device__ __cudart_builtin__ void tex2DLayeredGrad(long1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_long1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(long2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_long2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(long4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_long4");
__device__ __cudart_builtin__ void tex2DLayeredGrad(unsigned long *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ulong");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ulong1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ulong1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ulong2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ulong2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(ulong4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_ulong4");
#endif /* !__LP64__ */

__device__ __cudart_builtin__ void tex2DLayeredGrad(float *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_float");
__device__ __cudart_builtin__ void tex2DLayeredGrad(float1 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_float1");
__device__ __cudart_builtin__ void tex2DLayeredGrad(float2 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_float2");
__device__ __cudart_builtin__ void tex2DLayeredGrad(float4 *, cudaTextureObject_t, float, float, int, float2, float2) asm("__itex2DLayeredGrad_float4");
#endif /* __CUDA_ARCH__ */

template <class T>
static __device__ T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy);
  return ret;
}
#endif // __cplusplus && __CUDACC__
#endif // __TEXTURE_INDIRECT_FUNCTIONS_H__
