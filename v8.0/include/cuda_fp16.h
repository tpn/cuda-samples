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

/**
 * \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
 * This section describes half precision intrinsic functions that are
 * only supported in device code.
 */

/**
 * \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion And Data Movement
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

/**
 * \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 */

#ifndef CUDA_FP16_H_JNESTUG4
#define CUDA_FP16_H_JNESTUG4

typedef struct __align__(2) {
   unsigned short x;
} __half;

typedef struct __align__(4) {
   unsigned int x;
} __half2;

#ifndef CUDA_NO_HALF
typedef __half half;
typedef __half2 half2;
#endif /*CUDA_NO_HALF*/

#if defined(__CUDACC__)

#if !defined(__cplusplus)
#include <stdbool.h>
#endif /*!defined(__cplusplus)*/

#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts float number to half precision in round-to-nearest-even mode
 * and returns \p half with converted value.
 *
 * Converts float number \p a to half precision in round-to-nearest-even mode.
 *
 * \return Returns \p half result with converted value.
 */
__CUDA_FP16_DECL__ __half __float2half(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts float number to half precision in round-towards-zero mode
 * and returns \p half with converted value.
 *
 * Converts float number \p a to half precision in round-towards-zero mode.
 *
 * \return Returns \p half result with converted value.
 */
__CUDA_FP16_DECL__ __half __float2half_rz(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts float number to half precision in round-down mode
 * and returns \p half with converted value.
 *
 * Converts float number \p a to half precision in round-down mode.
 *
 * \return Returns \p half result with converted value.
 */
__CUDA_FP16_DECL__ __half __float2half_rd(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts float number to half precision in round-up mode
 * and returns \p half with converted value.
 *
 * Converts float number \p a to half precision in round-up mode.
 *
 * \return Returns \p half result with converted value.
 */
__CUDA_FP16_DECL__ __half __float2half_ru(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts \p half number to float.
 *
 * Converts half number \p a to float.
 *
 * \return Returns float result with converted value.
 */
__CUDA_FP16_DECL__ float __half2float(const __half a);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed integer in round-to-nearest-even mode.
 *
 * Convert the half-precision floating point value \p h to a signed integer in
 * round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ int __half2int_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed integer in round-towards-zero mode.
 *
 * Convert the half-precision floating point value \p h to a signed integer in
 * round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ int __half2int_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to a signed integer in
 * round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ int __half2int_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to a signed integer in
 * round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ int __half2int_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed integer to a half in round-to-nearest-even mode.
 *
 * Convert the signed integer value \p i to a half-precision floating point
 * value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __int2half_rn(int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed integer to a half in round-towards-zero mode.
 *
 * Convert the signed integer value \p i to a half-precision floating point
 * value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __int2half_rz(int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed integer to a half in round-down mode.
 *
 * Convert the signed integer value \p i to a half-precision floating point
 * value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __int2half_rd(int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed integer to a half in round-up mode.
 *
 * Convert the signed integer value \p i to a half-precision floating point
 * value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __int2half_ru(int i);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed short integer in round-to-nearest-even
 * mode.
 *
 * Convert the half-precision floating point value \p h to a signed short
 * integer in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ short int __half2short_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed short integer in round-towards-zero mode.
 *
 * Convert the half-precision floating point value \p h to a signed short
 * integer in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ short int __half2short_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed short integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to a signed short
 * integer in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ short int __half2short_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed short integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to a signed short
 * integer in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ short int __half2short_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed short integer to a half in round-to-nearest-even
 * mode.
 *
 * Convert the signed short integer value \p i to a half-precision floating
 * point value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __short2half_rn(short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed short integer to a half in round-towards-zero mode.
 *
 * Convert the signed short integer value \p i to a half-precision floating
 * point value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __short2half_rz(short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed short integer to a half in round-down mode.
 *
 * Convert the signed short integer value \p i to a half-precision floating
 * point value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __short2half_rd(short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed short integer to a half in round-up mode.
 *
 * Convert the signed short integer value \p i to a half-precision floating
 * point value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __short2half_ru(short int i);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned integer in round-to-nearest-even mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned integer
 * in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned int __half2uint_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned integer in round-towards-zero mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned integer
 * in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned int __half2uint_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned integer
 * in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned integer
 * in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned integer to a half in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p i to a half-precision floating point
 * value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __uint2half_rn(unsigned int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned integer to a half in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p i to a half-precision floating point
 * value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __uint2half_rz(unsigned int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned integer to a half in round-down mode.
 *
 * Convert the unsigned integer value \p i to a half-precision floating point
 * value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __uint2half_rd(unsigned int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned integer to a half in round-up mode.
 *
 * Convert the unsigned integer value \p i to a half-precision floating point
 * value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __uint2half_ru(unsigned int i);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned short integer in round-to-nearest-even
 * mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned short
 * integer in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned short integer in round-towards-zero
 * mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned short
 * integer in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned short integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned short
 * integer in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned short integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned short
 * integer in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned short integer to a half in round-to-nearest-even
 * mode.
 *
 * Convert the unsigned short integer value \p i to a half-precision floating
 * point value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ushort2half_rn(unsigned short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned short integer to a half in round-towards-zero
 * mode.
 *
 * Convert the unsigned short integer value \p i to a half-precision floating
 * point value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ushort2half_rz(unsigned short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned short integer to a half in round-down mode.
 *
 * Convert the unsigned short integer value \p i to a half-precision floating
 * point value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ushort2half_rd(unsigned short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned short integer to a half in round-up mode.
 *
 * Convert the unsigned short integer value \p i to a half-precision floating
 * point value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ushort2half_ru(unsigned short int i);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned 64-bit integer in round-to-nearest-even
 * mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned 64-bit
 * integer in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned 64-bit integer in round-towards-zero
 * mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned 64-bit
 * integer in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned 64-bit integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned 64-bit
 * integer in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to an unsigned 64-bit integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to an unsigned 64-bit
 * integer in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned 64-bit integer to a half in round-to-nearest-even
 * mode.
 *
 * Convert the unsigned 64-bit integer value \p i to a half-precision floating
 * point value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ull2half_rn(unsigned long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned 64-bit integer to a half in round-towards-zero
 * mode.
 *
 * Convert the unsigned 64-bit integer value \p i to a half-precision floating
 * point value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ull2half_rz(unsigned long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned 64-bit integer to a half in round-down mode.
 *
 * Convert the unsigned 64-bit integer value \p i to a half-precision floating
 * point value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ull2half_rd(unsigned long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert an unsigned 64-bit integer to a half in round-up mode.
 *
 * Convert the unsigned 64-bit integer value \p i to a half-precision floating
 * point value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ull2half_ru(unsigned long long int i);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed 64-bit integer in round-to-nearest-even
 * mode.
 *
 * Convert the half-precision floating point value \p h to a signed 64-bit
 * integer in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ long long int __half2ll_rn(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed 64-bit integer in round-towards-zero mode.
 *
 * Convert the half-precision floating point value \p h to a signed 64-bit
 * integer in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ long long int __half2ll_rz(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed 64-bit integer in round-down mode.
 *
 * Convert the half-precision floating point value \p h to a signed 64-bit
 * integer in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ long long int __half2ll_rd(__half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a half to a signed 64-bit integer in round-up mode.
 *
 * Convert the half-precision floating point value \p h to a signed 64-bit
 * integer in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ long long int __half2ll_ru(__half h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed 64-bit integer to a half in round-to-nearest-even
 * mode.
 *
 * Convert the signed 64-bit integer value \p i to a half-precision floating
 * point value in round-to-nearest-even mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ll2half_rn(long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed 64-bit integer to a half in round-towards-zero mode.
 *
 * Convert the signed 64-bit integer value \p i to a half-precision floating
 * point value in round-towards-zero mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ll2half_rz(long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed 64-bit integer to a half in round-down mode.
 *
 * Convert the signed 64-bit integer value \p i to a half-precision floating
 * point value in round-down mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ll2half_rd(long long int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Convert a signed 64-bit integer to a half in round-up mode.
 *
 * Convert the signed 64-bit integer value \p i to a half-precision floating
 * point value in round-up mode.
 *
 * \return Returns converted value.
 */
__CUDA_FP16_DECL__ __half __ll2half_ru(long long int i);

/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Truncate input argument to the integral part.
 *
 * Round \p h to the nearest integer value that does not exceed \p h in
 * magnitude.
 *
 * \return Returns truncated integer value.
 */
__CUDA_FP16_DECL__ __half htrunc(const __half h);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p h.
 *
 * \return Returns ceiling expressed as a half-precision floating point number.
 */
__CUDA_FP16_DECL__ __half hceil(const __half h);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculate the largest integer less than or equal to \p h.
 *
 * Calculate the largest integer value which is less than or equal to \p h.
 *
 * \return Returns floor expressed as half-precision floating point number.
 */
__CUDA_FP16_DECL__ __half hfloor(const __half h);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Round input to nearest integer value in half-precision floating point
 * number.
 *
 * Round \p h to the nearest integer value in half-precision floating point
 * format, with halfway cases rounded to the nearest even integer value.
 *
 * \return Returns rounded integer value expressed as half-precision floating
 * point number.
 */
__CUDA_FP16_DECL__ __half hrint(const __half h);

/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Truncate \p half2 vector input argument to the integral part.
 *
 * Round each component of vector \p h to the nearest integer value that does
 * not exceed \p h in magnitude.
 *
 * \return Returns \p half2 vector truncated integer value.
 */
__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculate \p half2 vector ceiling of the input argument.
 *
 * For each component of vector \p h compute the smallest integer value not less
 * than \p h.
 *
 * \return Returns \p half2 vector ceiling expressed as a pair of half-precision
 * floating point numbers.
 */
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculate the largest integer less than or equal to \p h.
 *
 * For each component of vector \p h calculate the largest integer value which
 * is less than or equal to \p h.
 *
 * \return Returns \p half2 vector floor expressed as a pair of half-precision
 * floating point number.
 */
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Round input to nearest integer value in half-precision floating point
 * number.
 *
 * Round each component of \p half2 vector \p h to the nearest integer value in
 * half-precision floating point format, with halfway cases rounded to the
 * nearest even integer value.
 *
 * \return Returns \p half2 vector of rounded integer values expressed as
 * half-precision floating point numbers.
 */
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts input to half precision in round-to-nearest-even mode and
 * populates both halves of \p half2 with converted value.
 *
 * Converts input \p a to half precision in round-to-nearest-even mode and
 * populates both halves of \p half2 with converted value.
 *
 * \return Returns \p half2 with both halves equal to the converted half
 * precision number.
 */
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both input floats to half precision in round-to-nearest-even
 * mode and returns \p half2 with converted values.
 *
 * Converts both input floats to half precision in round-to-nearest-even mode
 * and combines the results into one \p half2 number. Low 16 bits of the return
 * value correspond to the input \p a, high 16 bits correspond to the input \p
 * b.
 *
 * \return Returns \p half2 which has corresponding halves equal to the
 * converted input floats.
 */
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both components of float2 number to half precision in
 * round-to-nearest-even mode and returns \p half2 with converted values.
 *
 * Converts both components of float2 to half precision in round-to-nearest
 * mode and combines the results into one \p half2 number. Low 16 bits of the
 * return value correspond to \p a.x and high 16 bits of the return value
 * correspond to \p a.y.
 *
 * \return Returns \p half2 which has corresponding halves equal to the
 * converted float2 components.
 */
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both halves of \p half2 to float2 and returns the result.
 *
 * Converts both halves of \p half2 input \p a to float2 and returns the
 * result.
 *
 * \return Returns converted float2.
 */
__CUDA_FP16_DECL__ float2 __half22float2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts low 16 bits of \p half2 to float and returns the result
 *
 * Converts low 16 bits of \p half2 input \p a to 32 bit floating point number
 * and returns the result.
 *
 * \return Returns low 16 bits of \p a converted to float.
 */
__CUDA_FP16_DECL__ float __low2float(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Returns \p half2 with both halves equal to the input value.
 *
 * Returns \p half2 number with both halves equal to the input \p a \p half
 * number.
 *
 * \return Returns \p half2 with both halves equal to the input \p a.
 */
__CUDA_FP16_DECL__ __half2 __half2half2(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts high 16 bits of \p half2 to float and returns the result
 *
 * Converts high 16 bits of \p half2 input \p a to 32 bit floating point number
 * and returns the result.
 *
 * \return Returns high 16 bits of \p a converted to float.
 */
__CUDA_FP16_DECL__ float __high2float(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Swaps both halves of the \p half2 input.
 *
 * Swaps both halves of the \p half2 input and returns a new \p half2 number
 * with swapped halves.
 *
 * \return Returns \p half2 with halves swapped.
 */
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Extracts low 16 bits from each of the two \p half2 inputs and combines
 * into one \p half2 number.
 *
 * Extracts low 16 bits from each of the two \p half2 inputs and combines into
 * one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
 * the return value, low 16 bits from input \p b is stored in high 16 bits of
 * the return value.
 *
 * \return Returns \p half2 which contains low 16 bits from \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Extracts high 16 bits from each of the two \p half2 inputs and
 * combines into one \p half2 number.
 *
 * Extracts high 16 bits from each of the two \p half2 inputs and combines into
 * one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
 * the return value, high 16 bits from input \p b is stored in high 16 bits of
 * the return value.
 *
 * \return Returns \p half2 which contains high 16 bits from \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Returns high 16 bits of \p half2 input.
 *
 * Returns high 16 bits of \p half2 input \p a.
 *
 * \return Returns \p half which contains high 16 bits of the input.
 */
__CUDA_FP16_DECL__ __half __high2half(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Returns low 16 bits of \p half2 input.
 *
 * Returns low 16 bits of \p half2 input \p a.
 *
 * \return Returns \p half which contains low 16 bits of the input.
 */
__CUDA_FP16_DECL__ __half __low2half(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Checks if the input \p half number is infinite.
 *
 * Checks if the input \p half number \p a is infinite.
 *
 * \return Returns -1 iff \p a is equal to negative infinity, 1 iff \p a is
 * equal to positive infinity and 0 otherwise.
 */
__CUDA_FP16_DECL__ int __hisinf(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Combines two \p half numbers into one \p half2 number.
 *
 * Combines two input \p half number \p a and \p b into one \p half2 number.
 * Input \p a is stored in low 16 bits of the return value, input \p b is stored
 * in high 16 bits of the return value.
 *
 * \return Returns \p half2 number which has one half equal to \p a and the
 * other to \p b.
 */
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Extracts low 16 bits from \p half2 input.
 *
 * Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
 * number which has both halves equal to the extracted bits.
 *
 * \return Returns \p half2 with both halves equal to low 16 bits from the
 * input.
 */
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Extracts high 16 bits from \p half2 input.
 *
 * Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
 * number which has both halves equal to the extracted bits.
 *
 * \return Returns \p half2 with both halves equal to high 16 bits from the
 * input.
 */
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 a);

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Reinterprets bits in a \p half as a signed short integer.
 *
 * Reinterprets the bits in the half-precision floating point value \p h
 * as a signed short integer.
 *
 * \return Returns reinterpreted value.
 */
__CUDA_FP16_DECL__ short int __half_as_short(const __half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Reinterprets bits in a \p half as an unsigned short integer.
 *
 * Reinterprets the bits in the half-precision floating point value \p h
 * as an unsigned short integer.
 *
 * \return Returns reinterpreted value.
 */
__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Reinterprets bits in a signed short integer as a \p half.
 *
 * Reinterprets the bits in the signed short integer value \p i as a
 * half-precision floating point value.
 *
 * \return Returns reinterpreted value.
 */
__CUDA_FP16_DECL__ __half __short_as_half(const short int i);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Reinterprets bits in an unsigned short integer as a \p half.
 *
 * Reinterprets the bits in the unsigned short integer value \p i as a
 * half-precision floating point value.
 *
 * \return Returns reinterpreted value.
 */
__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i);

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif
__CUDA_FP16_DECL__ __half2 __shfl(__half2 var, int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_up(__half2 var, unsigned int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_down(__half2 var, unsigned int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_xor(__half2 var, int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half __shfl(__half var, int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half __shfl_up(__half var, unsigned int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half __shfl_down(__half var, unsigned int delta, int width=warpSize);
__CUDA_FP16_DECL__ __half __shfl_xor(__half var, int delta, int width=warpSize);
#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif
#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__) */

#if defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldcg(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldca(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldcs(const __half *ptr);
#endif /*defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )*/

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs half2 vector if-equal comparison.
 *
 * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the \p half2 vector result of if-equal comparison of vectors
 * \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector not-equal comparison.
 *
 * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the \p half2 vector result of not-equal comparison of vectors
 * \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector less-equal comparison.
 *
 * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the \p half2 vector result of less-equal comparison of
 * vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector greater-equal comparison.
 *
 * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the \p half2 vector result of greater-equal comparison of
 * vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector less-than comparison.
 *
 * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the \p half2 vector result of less-than comparison of vectors
 * \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector greater-than comparison.
 *
 * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate false results.
 *
 * \return Returns the half2 vector result of greater-than comparison of vectors
 * \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered if-equal comparison.
 *
 * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered if-equal comparison
 * of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered not-equal comparison.
 *
 * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered not-equal comparison
 * of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered less-equal comparison.
 *
 * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered less-equal comparison
 * of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered greater-equal comparison.
 *
 * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered greater-equal
 * comparison of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered less-than comparison.
 *
 * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered less-than comparison
 * of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered greater-than comparison.
 *
 * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
 * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
 * NaN inputs generate true results.
 *
 * \return Returns the \p half2 vector result of unordered greater-than
 * comparison of vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Determine whether \p half2 argument is a NaN.
 *
 * Determine whether each half of input \p half2 number \p a is a NaN.
 *
 * \return Returns \p half2 which has the corresponding \p half results set to
 * 1.0 for true, or 0.0 for false.
 */
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector addition in round-to-nearest-even mode.
 *
 * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
 * mode.
 *
 * \return Returns the \p half2 vector result of adding vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
 *
 * Subtracts \p half2 input vector \p b from input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns the \p half2 vector result of subtraction vector \p b from \p
 * a.
 */
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
 *
 * Performs \p half2 vector multiplication of inputs \p a and \p b, in
 * round-to-nearest-even mode.
 *
 * \return Returns the \p half2 vector result of multiplying vectors \p a and \p
 * b.
 */
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half2 vector division in round-to-nearest-even mode.
 *
 * Divides \p half2 input vector \p a by input vector \p b in round-to-nearest
 * mode.
 *
 * \return Returns the \p half2 vector result of division \p a by \p b.
 */
__CUDA_FP16_DECL__ __half2 h2div(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector addition in round-to-nearest-even mode, with
 * saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
 * mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
 * +0.0.
 *
 * \return Returns the \p half2 vector result of adding vectors \p a and \p b
 * with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector subtraction in round-to-nearest-even mode,
 * with saturation to [0.0, 1.0].
 *
 * Subtracts \p half2 input vector \p b from input vector \p a in
 * round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
 * results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of subtraction vector \p b from \p
 * a with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector multiplication in round-to-nearest-even mode,
 * with saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector multiplication of inputs \p a and \p b, in
 * round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
 * results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of multiplying vectors \p a and \p
 * b with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
 * mode.
 *
 * Performs \p half2 vector multiply on inputs \p a and \p b,
 * then performs a \p half2 vector add of the result with \p c,
 * rounding the result once in round-to-nearest-even mode.
 *
 * \return Returns the \p half2 vector result of the fused multiply-add
 * operation on vectors \p a, \p b, and \p c.
 */
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
 * mode, with saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector multiply on inputs \p a and \p b,
 * then performs a \p half2 vector add of the result with \p c,
 * rounding the result once in round-to-nearest-even mode, and clamps the
 * results to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of the fused multiply-add
 * operation on vectors \p a, \p b, and \p c with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Negates both halves of the input \p half2 number and returns the
 * result.
 *
 * Negates both halves of the input \p half2 number \p a and returns the result.
 *
 * \return Returns \p half2 number with both halves negated.
 */
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half addition in round-to-nearest-even mode.
 *
 * Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
 * mode.
 *
 * \return Returns the \p half result of adding \p a and \p b.
 */
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half subtraction in round-to-nearest-even mode.
 *
 * Subtracts \p half input \p b from input \p a in round-to-nearest
 * mode.
 *
 * \return Returns the \p half result of subtraction \p b from \p a.
 */
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half multiplication in round-to-nearest-even mode.
 *
 * Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
 * mode.
 *
 * \return Returns the \p half result of multiplying \p a and \p b.
 */
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half division in round-to-nearest-even mode.
 *
 * Divides \p half input \p a by input \p b in round-to-nearest
 * mode.
 *
 * \return Returns the \p half result of division \p a by \p b.
 */
__CUDA_FP16_DECL__  __half hdiv(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half addition in round-to-nearest-even mode, with
 * saturation to [0.0, 1.0].
 *
 * Performs \p half add of inputs \p a and \p b, in round-to-nearest-even mode,
 * and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half result of adding \p a and \p b with saturation.
 */
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half subtraction in round-to-nearest-even mode, with
 * saturation to [0.0, 1.0].
 *
 * Subtracts \p half input \p b from input \p a in round-to-nearest
 * mode,
 * and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half result of subtraction \p b from \p a
 * with saturation.
 */
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half multiplication in round-to-nearest-even mode, with
 * saturation to [0.0, 1.0].
 *
 * Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
 * mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
 * +0.0.
 *
 * \return Returns the \p half result of multiplying \p a and \p b with
 * saturation.
 */
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half fused multiply-add in round-to-nearest-even mode.
 *
 * Performs \p half multiply on inputs \p a and \p b,
 * then performs a \p half add of the result with \p c,
 * rounding the result once in round-to-nearest-even mode.
 *
 * \return Returns the \p half result of the fused multiply-add operation on \p
 * a, \p b, and \p c.
 */
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half fused multiply-add in round-to-nearest-even mode,
 * with saturation to [0.0, 1.0].
 *
 * Performs \p half multiply on inputs \p a and \p b,
 * then performs a \p half add of the result with \p c,
 * rounding the result once in round-to-nearest-even mode, and clamps the result
 * to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half result of the fused multiply-add operation on \p
 * a, \p b, and \p c with saturation.
 */
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Negates input \p half number and returns the result.
 *
 * Negates input \p half number and returns the result.
 *
 * \return Returns negated \p half input \p a.
 */
__CUDA_FP16_DECL__ __half __hneg(const __half a);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector if-equal comparison, and returns boolean true
 * iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half if-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of if-equal comparison
 * of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector not-equal comparison, and returns boolean
 * true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half not-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of not-equal comparison
 * of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector less-equal comparison, and returns boolean
 * true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half less-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of less-equal comparison
 * of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector greater-equal comparison, and returns boolean
 * true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half greater-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of greater-equal
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector less-than comparison, and returns boolean
 * true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half less-than comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of less-than comparison
 * of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector greater-than comparison, and returns boolean
 * true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half greater-than comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate false results.
 *
 * \return Returns boolean true if both \p half results of greater-than
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered if-equal comparison, and returns
 * boolean true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half if-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered if-equal
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered not-equal comparison, and returns
 * boolean true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half not-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered not-equal
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered less-equal comparison, and returns
 * boolean true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half less-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered less-equal
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered greater-equal comparison, and
 * returns boolean true iff both \p half results are true, boolean false
 * otherwise.
 *
 * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half greater-equal comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered
 * greater-equal comparison of vectors \p a and \p b are true, boolean false
 * otherwise.
 */
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered less-than comparison, and returns
 * boolean true iff both \p half results are true, boolean false otherwise.
 *
 * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half less-than comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered less-than
 * comparison of vectors \p a and \p b are true, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_COMPARISON
 * \brief Performs \p half2 vector unordered greater-than comparison, and
 * returns boolean true iff both \p half results are true, boolean false
 * otherwise.
 *
 * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
 * The bool result is set to true only if both \p half greater-than comparisons
 * evaluate to true, or false otherwise.
 * NaN inputs generate true results.
 *
 * \return Returns boolean true if both \p half results of unordered
 * greater-than comparison of vectors \p a and \p b are true, boolean false
 * otherwise.
 */
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half if-equal comparison.
 *
 * Performs \p half if-equal comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of if-equal comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half not-equal comparison.
 *
 * Performs \p half not-equal comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of not-equal comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half less-equal comparison.
 *
 * Performs \p half less-equal comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of less-equal comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half greater-equal comparison.
 *
 * Performs \p half greater-equal comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of greater-equal comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half less-than comparison.
 *
 * Performs \p half less-than comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of less-than comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half greater-than comparison.
 *
 * Performs \p half greater-than comparison of inputs \p a and \p b.
 * NaN inputs generate false results.
 *
 * \return Returns boolean result of greater-than comparison of \p a and \p b.
 */
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered if-equal comparison.
 *
 * Performs \p half if-equal comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered if-equal comparison of \p a and
 * \p b.
 */
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered not-equal comparison.
 *
 * Performs \p half not-equal comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered not-equal comparison of \p a and
 * \p b.
 */
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered less-equal comparison.
 *
 * Performs \p half less-equal comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered less-equal comparison of \p a and
 * \p b.
 */
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered greater-equal comparison.
 *
 * Performs \p half greater-equal comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered greater-equal comparison of \p a
 * and \p b.
 */
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered less-than comparison.
 *
 * Performs \p half less-than comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered less-than comparison of \p a and
 * \p b.
 */
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Performs \p half unordered greater-than comparison.
 *
 * Performs \p half greater-than comparison of inputs \p a and \p b.
 * NaN inputs generate true results.
 *
 * \return Returns boolean result of unordered greater-than comparison of \p a
 * and \p b.
 */
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_COMPARISON
 * \brief Determine whether \p half argument is a NaN.
 *
 * Determine whether \p half value \p a is a NaN.
 *
 * \return Returns boolean true iff argument is a NaN, boolean false otherwise.
 */
__CUDA_FP16_DECL__ bool __hisnan(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half square root in round-to-nearest-even mode.
 *
 * Calculates \p half square root of input \p a in round-to-nearest-even mode.
 *
 * \return Returns \p half square root of \p a.
 */
__CUDA_FP16_DECL__ __half hsqrt(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half reciprocal square root in round-to-nearest-even
 * mode.
 *
 * Calculates \p half reciprocal square root of input \p a in round-to-nearest
 * mode.
 *
 * \return Returns \p half reciprocal square root of \p a.
 */
__CUDA_FP16_DECL__ __half hrsqrt(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half reciprocal in round-to-nearest-even mode.
 *
 * Calculates \p half reciprocal of input \p a in round-to-nearest-even mode.
 *
 * \return Returns \p half reciprocal of \p a.
 */
__CUDA_FP16_DECL__ __half hrcp(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half natural logarithm in round-to-nearest-even mode.
 *
 * Calculates \p half natural logarithm of input \p a in round-to-nearest-even
 * mode.
 *
 * \return Returns \p half natural logarithm of \p a.
 */
__CUDA_FP16_DECL__ __half hlog(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half binary logarithm in round-to-nearest-even mode.
 *
 * Calculates \p half binary logarithm of input \p a in round-to-nearest-even
 * mode.
 *
 * \return Returns \p half binary logarithm of \p a.
 */
__CUDA_FP16_DECL__ __half hlog2(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half decimal logarithm in round-to-nearest-even mode.
 *
 * Calculates \p half decimal logarithm of input \p a in round-to-nearest-even
 * mode.
 *
 * \return Returns \p half decimal logarithm of \p a.
 */
__CUDA_FP16_DECL__ __half hlog10(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half natural exponential function in round-to-nearest
 * mode.
 *
 * Calculates \p half natural exponential function of input \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half natural exponential function of \p a.
 */
__CUDA_FP16_DECL__ __half hexp(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half binary exponential function in round-to-nearest
 * mode.
 *
 * Calculates \p half binary exponential function of input \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half binary exponential function of \p a.
 */
__CUDA_FP16_DECL__ __half hexp2(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half decimal exponential function in round-to-nearest
 * mode.
 *
 * Calculates \p half decimal exponential function of input \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half decimal exponential function of \p a.
 */
__CUDA_FP16_DECL__ __half hexp10(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half cosine in round-to-nearest-even mode.
 *
 * Calculates \p half cosine of input \p a in round-to-nearest-even mode.
 *
 * \return Returns \p half cosine of \p a.
 */
__CUDA_FP16_DECL__ __half hcos(const __half a);
/**
 * \ingroup CUDA_MATH__HALF_FUNCTIONS
 * \brief Calculates \p half sine in round-to-nearest-even mode.
 *
 * Calculates \p half sine of input \p a in round-to-nearest-even mode.
 *
 * \return Returns \p half sine of \p a.
 */
__CUDA_FP16_DECL__ __half hsin(const __half a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector square root in round-to-nearest-even mode.
 *
 * Calculates \p half2 square root of input vector \p a in round-to-nearest
 * mode.
 *
 * \return Returns \p half2 square root of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector reciprocal square root in round-to-nearest
 * mode.
 *
 * Calculates \p half2 reciprocal square root of input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 reciprocal square root of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector reciprocal in round-to-nearest-even mode.
 *
 * Calculates \p half2 reciprocal of input vector \p a in round-to-nearest-even
 * mode.
 *
 * \return Returns \p half2 reciprocal of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector natural logarithm in round-to-nearest-even
 * mode.
 *
 * Calculates \p half2 natural logarithm of input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 natural logarithm of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector binary logarithm in round-to-nearest-even
 * mode.
 *
 * Calculates \p half2 binary logarithm of input vector \p a in round-to-nearest
 * mode.
 *
 * \return Returns \p half2 binary logarithm of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector decimal logarithm in round-to-nearest-even
 * mode.
 *
 * Calculates \p half2 decimal logarithm of input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 decimal logarithm of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector exponential function in round-to-nearest
 * mode.
 *
 * Calculates \p half2 exponential function of input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 exponential function of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector binary exponential function in
 * round-to-nearest-even mode.
 *
 * Calculates \p half2 binary exponential function of input vector \p a in
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 binary exponential function of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector decimal exponential function in
 * round-to-nearest-even mode.
 *
 * Calculates \p half2 decimal exponential function of input vector \p a in 
 * round-to-nearest-even mode.
 *
 * \return Returns \p half2 decimal exponential function of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector cosine in round-to-nearest-even mode.
 *
 * Calculates \p half2 cosine of input vector \p a in round-to-nearest-even
 * mode.
 *
 * \return Returns \p half2 cosine of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF2_FUNCTIONS
 * \brief Calculates \p half2 vector sine in round-to-nearest-even mode.
 *
 * Calculates \p half2 sine of input vector \p a in round-to-nearest-even mode.
 *
 * \return Returns \p half2 sine of vector \p a.
 */
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a);

#endif /*if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/

__CUDA_FP16_DECL__ int __half2int_rn(__half h)
{
   int i;
   asm("cvt.rni.s32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ int __half2int_rz(__half h)
{
   int i;
   asm("cvt.rzi.s32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ int __half2int_rd(__half h)
{
   int i;
   asm("cvt.rmi.s32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ int __half2int_ru(__half h)
{
   int i;
   asm("cvt.rpi.s32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __int2half_rn(int i)
{
   __half h;
   asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __int2half_rz(int i)
{
   __half h;
   asm("cvt.rz.f16.s32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __int2half_rd(int i)
{
   __half h;
   asm("cvt.rm.f16.s32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __int2half_ru(int i)
{
   __half h;
   asm("cvt.rp.f16.s32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}

__CUDA_FP16_DECL__ short int __half2short_rn(__half h)
{
   short int i;
   asm("cvt.rni.s16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ short int __half2short_rz(__half h)
{
   short int i;
   asm("cvt.rzi.s16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ short int __half2short_rd(__half h)
{
   short int i;
   asm("cvt.rmi.s16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ short int __half2short_ru(__half h)
{
   short int i;
   asm("cvt.rpi.s16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __short2half_rn(short int i)
{
   __half h;
   asm("cvt.rn.f16.s16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __short2half_rz(short int i)
{
   __half h;
   asm("cvt.rz.f16.s16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __short2half_rd(short int i)
{
   __half h;
   asm("cvt.rm.f16.s16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __short2half_ru(short int i)
{
   __half h;
   asm("cvt.rp.f16.s16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}

__CUDA_FP16_DECL__ unsigned int __half2uint_rn(__half h)
{
   unsigned int i;
   asm("cvt.rni.u32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_rz(__half h)
{
   unsigned int i;
   asm("cvt.rzi.u32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(__half h)
{
   unsigned int i;
   asm("cvt.rmi.u32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(__half h)
{
   unsigned int i;
   asm("cvt.rpi.u32.f16 %0, %1;" : "=r"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __uint2half_rn(unsigned int i)
{
   __half h;
   asm("cvt.rn.f16.u32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __uint2half_rz(unsigned int i)
{
   __half h;
   asm("cvt.rz.f16.u32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __uint2half_rd(unsigned int i)
{
   __half h;
   asm("cvt.rm.f16.u32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __uint2half_ru(unsigned int i)
{
   __half h;
   asm("cvt.rp.f16.u32 %0, %1;" : "=h"(h.x) : "r"(i));
   return h;
}

__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(__half h)
{
   unsigned short int i;
   asm("cvt.rni.u16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rz(__half h)
{
   unsigned short int i;
   asm("cvt.rzi.u16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(__half h)
{
   unsigned short int i;
   asm("cvt.rmi.u16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(__half h)
{
   unsigned short int i;
   asm("cvt.rpi.u16.f16 %0, %1;" : "=h"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __ushort2half_rn(unsigned short int i)
{
   __half h;
   asm("cvt.rn.f16.u16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_rz(unsigned short int i)
{
   __half h;
   asm("cvt.rz.f16.u16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_rd(unsigned short int i)
{
   __half h;
   asm("cvt.rm.f16.u16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_ru(unsigned short int i)
{
   __half h;
   asm("cvt.rp.f16.u16 %0, %1;" : "=h"(h.x) : "h"(i));
   return h;
}

__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(__half h)
{
   unsigned long long int i;
   asm("cvt.rni.u64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rz(__half h)
{
   unsigned long long int i;
   asm("cvt.rzi.u64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(__half h)
{
   unsigned long long int i;
   asm("cvt.rmi.u64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(__half h)
{
   unsigned long long int i;
   asm("cvt.rpi.u64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __ull2half_rn(unsigned long long int i)
{
   __half h;
   asm("cvt.rn.f16.u64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ull2half_rz(unsigned long long int i)
{
   __half h;
   asm("cvt.rz.f16.u64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ull2half_rd(unsigned long long int i)
{
   __half h;
   asm("cvt.rm.f16.u64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ull2half_ru(unsigned long long int i)
{
   __half h;
   asm("cvt.rp.f16.u64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}

__CUDA_FP16_DECL__ long long int __half2ll_rn(__half h)
{
   long long int i;
   asm("cvt.rni.s64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_rz(__half h)
{
   long long int i;
   asm("cvt.rzi.s64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_rd(__half h)
{
   long long int i;
   asm("cvt.rmi.s64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_ru(__half h)
{
   long long int i;
   asm("cvt.rpi.s64.f16 %0, %1;" : "=l"(i) : "h"(h.x));
   return i;
}
__CUDA_FP16_DECL__ __half __ll2half_rn(long long int i)
{
   __half h;
   asm("cvt.rn.f16.s64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ll2half_rz(long long int i)
{
   __half h;
   asm("cvt.rz.f16.s64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ll2half_rd(long long int i)
{
   __half h;
   asm("cvt.rm.f16.s64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}
__CUDA_FP16_DECL__ __half __ll2half_ru(long long int i)
{
   __half h;
   asm("cvt.rp.f16.s64 %0, %1;" : "=h"(h.x) : "l"(i));
   return h;
}

__CUDA_FP16_DECL__ __half htrunc(const __half h)
{
   __half r;
   asm("cvt.rzi.f16.f16 %0, %1;" : "=h"(r.x) : "h"(h.x));
   return r;
}
__CUDA_FP16_DECL__ __half hceil(const __half h)
{
   __half r;
   asm("cvt.rpi.f16.f16 %0, %1;" : "=h"(r.x) : "h"(h.x));
   return r;
}
__CUDA_FP16_DECL__ __half hfloor(const __half h)
{
   __half r;
   asm("cvt.rmi.f16.f16 %0, %1;" : "=h"(r.x) : "h"(h.x));
   return r;
}
__CUDA_FP16_DECL__ __half hrint(const __half h)
{
   __half r;
   asm("cvt.rni.f16.f16 %0, %1;" : "=h"(r.x) : "h"(h.x));
   return r;
}

__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  cvt.rzi.f16.f16 low, low;\n"
       "  cvt.rzi.f16.f16 high, high;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  cvt.rpi.f16.f16 low, low;\n"
       "  cvt.rpi.f16.f16 high, high;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  cvt.rmi.f16.f16 low, low;\n"
       "  cvt.rmi.f16.f16 high, high;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  cvt.rni.f16.f16 low, low;\n"
       "  cvt.rni.f16.f16 high, high;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "r"(h.x));
   return val;
}

__CUDA_FP16_DECL__ float2 __half22float2(const __half2 l)
{
   float hi_float;
   float lo_float;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(l.x));

   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(l.x));

   return make_float2(lo_float, hi_float);
}
__CUDA_FP16_DECL__ __half __float2half(const float f)
{
   __half val;
   asm volatile("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half_rz(const float f)
{
   __half val;
   asm volatile("{  cvt.rz.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half_rd(const float f)
{
   __half val;
   asm volatile("{  cvt.rm.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half_ru(const float f)
{
   __half val;
   asm volatile("{  cvt.rp.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ float __half2float(const __half h)
{
   float val;
   asm volatile("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float f)
{
   __half2 val;
   asm("{.reg .f16 low;\n"
       "  cvt.rn.f16.f32 low, %1;\n"
       "  mov.b32 %0, {low,low};}\n" : "=r"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float f1, const float f2)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  cvt.rn.f16.f32 low, %1;\n"
       "  cvt.rn.f16.f32 high, %2;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "f"(f1), "f"(f2));
   return val;
}
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 f)
{
   __half2 val = __floats2half2_rn(f.x,f.y);
   return val;
}
__CUDA_FP16_DECL__ float __low2float(const __half2 l)
{
   float val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(val) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ float __high2float(const __half2 l)
{
   float val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(val) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 l, const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
       "  mov.b32 {alow,ahigh}, %1;\n"
       "  mov.b32 {blow,bhigh}, %2;\n"
       "  mov.b32 %0, {alow,blow};}\n" : "=r"(val.x) : "r"(l.x), "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 l, const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
       "  mov.b32 {alow,ahigh}, %1;\n"
       "  mov.b32 {blow,bhigh}, %2;\n"
       "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(val.x) : "r"(l.x), "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half __low2half(const __half2 h)
{
   __half ret;
   asm("{.reg .f16 low,high;\n"
       " mov.b32 {low,high}, %1;\n"
       " mov.b16 %0, low;}" : "=h"(ret.x) : "r"(h.x));
   return ret;
}
__CUDA_FP16_DECL__ int __hisinf(const __half a)
{
   if ( a.x == 0xFC00 )
      return -1;
   if ( a.x == 0x7C00 )
      return 1;
   return 0;
}
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 l)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {low,low};}\n" : "=r"(val.x) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 l)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {high,high};}\n" : "=r"(val.x) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half __high2half(const __half2 h)
{
   __half ret;
   asm("{.reg .f16 low,high;\n"
       " mov.b32 {low,high}, %1;\n"
       " mov.b16 %0, high;}" : "=h"(ret.x) : "r"(h.x));
   return ret;
}
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half l, const __half h)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%2};}\n"
       : "=r"(val.x) : "h"(l.x), "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __half2half2(const __half lh)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%1};}\n"
       : "=r"(val.x) : "h"(lh.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 lh)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {high,low};}\n" : "=r"(val.x) : "r"(lh.x));
   return val;
}
__CUDA_FP16_DECL__ short int __half_as_short(const __half h)
{
   return (short int)h.x;
}
__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h)
{
   return h.x;
}
__CUDA_FP16_DECL__ __half __short_as_half(const short int i)
{
   __half h;
   h.x = (unsigned short int)i;
   return h;
}
__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i)
{
   __half h;
   h.x = i;
   return h;
}

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
/******************************************************************************
 *                           __half, __half2 warp shuffle                     *
 ******************************************************************************/
#define SHUFFLE_HALF2_MACRO(name) do {\
   __half2 r; \
   asm("{"#name" %0,%1,%2,%3;\n}" \
       :"=r"(r.x): "r"(var.x), "r"(delta), "r"(c)); \
   return r; \
} while(0);
__CUDA_FP16_DECL__ __half2 __shfl(__half2 var, int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.idx.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_up(__half2 var, unsigned int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = (warpSize-width) << 8;
   SHUFFLE_HALF2_MACRO(shfl.up.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_down(__half2 var, unsigned int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.down.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_xor(__half2 var, int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.bfly.b32);
}
#undef SHUFFLE_HALF2_MACRO
__CUDA_FP16_DECL__ __half __shfl(__half var, int delta, int width)
{
   __half2 temp1 = __halves2half2(var, var);
   __half2 temp2 = __shfl(temp1, delta, width);
   return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_up(__half var, unsigned int delta, int width)
{
   __half2 temp1 = __halves2half2(var, var);
   __half2 temp2 = __shfl_up(temp1, delta, width);
   return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_down(__half var, unsigned int delta, int width)
{
   __half2 temp1 = __halves2half2(var, var);
   __half2 temp2 = __shfl_down(temp1, delta, width);
   return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_xor(__half var, int delta, int width)
{
   __half2 temp1 = __halves2half2(var, var);
   __half2 temp2 = __shfl_xor(temp1, delta, width);
   return __low2half(temp2);
}
#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)*/
/******************************************************************************
 *               __half and __half2 __ldg,__ldcg,__ldca,__ldcs                *
 ******************************************************************************/

#if defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *ptr)
{
   __half2 ret;
   asm volatile ("ld.global.nc.b32 %0, [%1];"  : "=r"(ret.x): __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr)
{
   __half ret;
   asm volatile ("ld.global.nc.b16 %0, [%1];"  : "=h"(ret.x) : __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *ptr)
{
   __half2 ret;
   asm volatile ("ld.global.cg.b32 %0, [%1];"  : "=r"(ret.x): __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half __ldcg(const __half *ptr)
{
   __half ret;
   asm volatile ("ld.global.cg.b16 %0, [%1];"  : "=h"(ret.x) : __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *ptr)
{
   __half2 ret;
   asm volatile ("ld.global.ca.b32 %0, [%1];"  : "=r"(ret.x): __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half __ldca(const __half *ptr)
{
   __half ret;
   asm volatile ("ld.global.ca.b16 %0, [%1];"  : "=h"(ret.x) : __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *ptr)
{
   __half2 ret;
   asm volatile ("ld.global.cs.b32 %0, [%1];"  : "=r"(ret.x): __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half __ldcs(const __half *ptr)
{
   __half ret;
   asm volatile ("ld.global.cs.b16 %0, [%1];"  : "=h"(ret.x) : __LDG_PTR (ptr));
   return ret;
}
#undef __LDG_PTR
#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/******************************************************************************
 *                             __half2 comparison                             *
 ******************************************************************************/
#define COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef COMPARISON_OP_HALF2_MACRO
#define BOOL_COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   if (val.x == 0x3C003C00) \
      return true; \
   else  \
      return false; \
} while(0);
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef BOOL_COMPARISON_OP_HALF2_MACRO
/******************************************************************************
 *                             __half comparison                              *
 ******************************************************************************/
#define COMPARISON_OP_HALF_MACRO(name) do {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp."#name".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(a.x), "h"(b.x)); \
   return val ? true : false; \
} while(0);
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(eq);
}
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ne);
}
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(le);
}
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ge);
}
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(lt);
}
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(gt);
}
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(equ);
}
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(neu);
}
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(leu);
}
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(geu);
}
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ltu);
}
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(gtu);
}
#undef COMPARISON_OP_HALF_MACRO
/******************************************************************************
 *                            __half2 arithmetic                             *
 ******************************************************************************/
#define BINARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(add);
}
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(sub);
}
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(mul);
}
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(mul.sat);
}
#undef BINARY_OP_HALF2_MACRO
#define TERNARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x),"r"(c.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)
{
   TERNARY_OP_HALF2_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c)
{
   TERNARY_OP_HALF2_MACRO(fma.rn.sat);
}
#undef TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half2 h2div(__half2 a, __half2 b) {
   __half ha, hb;

   ha = __low2half(a);
   hb = __low2half(b);

   __half v1 = hdiv(ha, hb);

   ha = __high2half(a);
   hb = __high2half(b);

   __half v2 = hdiv(ha, hb);

   return __halves2half2(v1, v2);
}
/******************************************************************************
 *                             __half arithmetic                             *
 ******************************************************************************/
#define BINARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2;\n}" \
        :"=h"(val.x) : "h"(a.x),"h"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(add);
}
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(sub);
}
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(mul);
}
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(mul.sat);
}
#undef BINARY_OP_HALF_MACRO
#define TERNARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2,%3;\n}" \
        :"=h"(val.x) : "h"(a.x),"h"(b.x),"h"(c.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
{
   TERNARY_OP_HALF_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
{
   TERNARY_OP_HALF_MACRO(fma.rn.sat);
}
#undef TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half hdiv(__half a, __half b) {
   __half v, abs, den;
   den.x = 0x008F;
   float fa, fb, fv, rcp;

   fa = __half2float(a);
   fb = __half2float(b);

   asm volatile("{rcp.approx.f32 %0, %1;\n}" :"=f"(rcp):"f"(fb));

   fv = rcp * fa;

   v = __float2half(fv);
   abs.x = (unsigned short) (((unsigned int)v.x) & 0x00007FFF);
   if (__hlt(abs, den) && (!(abs.x == 0x0000))) {
      float err = __fmaf_rn(-fb, fv, fa);
      fv = __fmaf_rn(rcp, err, fv);
      v = __float2half(fv);
   }
   return v;
}

/******************************************************************************
 *                             __half2 functions                  *
 ******************************************************************************/
#define SPEC_CASE2(i,r, spc, ulp) \
   "{.reg.b32 spc, ulp, p;\n"\
   "  mov.b32 spc,"#spc";\n"\
   "  mov.b32 ulp,"#ulp";\n"\
   "  set.eq.f16x2.f16x2 p,"#i", spc;\n"\
   "  fma.rn.f16x2 "#r",p,ulp,"#r";\n}\n"
#define SPEC_CASE(i,r, spc, ulp) \
   "{.reg.b16 spc, ulp, p;\n"\
   "  mov.b16 spc,"#spc";\n"\
   "  mov.b16 ulp,"#ulp";\n"\
   "  set.eq.f16.f16 p,"#i", spc;\n"\
   "  fma.rn.f16 "#r",p,ulp,"#r";\n}\n"
#define APPROX_FCAST(fun) do {\
   __half val;\
   asm volatile("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  cvt.f32.f16     f,r;      \n"\
                "  "#fun".approx.f32   f,f;  \n"\
                "  cvt.rn.f16.f32      r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(val.x) : "h"(a.x));\
   return val;\
} while(0);
#define APPROX_FCAST2(fun) do {\
   __half2 val;\
   asm volatile("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  cvt.f32.f16     fl, hl;         \n"\
                "  cvt.f32.f16     fu, hu;         \n"\
                "  "#fun".approx.f32   fl, fl;     \n"\
                "  "#fun".approx.f32   fu, fu;     \n"\
                "  cvt.rn.f16.f32      hl, fl;     \n"\
                "  cvt.rn.f16.f32      hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(val.x) : "r"(a.x));       \
   return val;\
} while(0);
static __device__ __forceinline__ float __float_simpl_sinf(float);
static __device__ __forceinline__ float __float_simpl_cosf(float);
__CUDA_FP16_DECL__ __half __hsin_internal(const __half a) {
   __half val;
   float f = __half2float(a);
   f = __float_simpl_sinf(f);
   val.x = __float2half_rn(f);
   return val;
}
__CUDA_FP16_DECL__ __half hsin(const __half a) {
   __half r = __hsin_internal(a);
   asm volatile("{\n\t"
                "  .reg.b16 i,r,t;     \n\t"
                "  mov.b16 r, %0;      \n\t"
                "  mov.b16 i, %1;      \n\t"
                "  mov.b16 t, 0x8000;  \n\t"
                "  and.b16 t,r,t;      \n\t"
                SPEC_CASE(i,r,0X32B3, 0x0800)
                SPEC_CASE(i,r,0X5CB0, 0x1000)
                SPEC_CASE(i,r,0XB2B3, 0x8800)
                SPEC_CASE(i,r,0XDCB0, 0x9000)
                "  or.b16  r,r,t;      \n\t"
                "  mov.b16 %0, r;      \n"
                "}\n" : "+h"(r.x): "h"(a.x));
   return r;
}
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a) {
   __half l = __low2half(a);
   __half h = __high2half(a);
   __half2 r = __halves2half2(__hsin_internal(l), __hsin_internal(h));
   asm volatile("{\n\t"
                "  .reg.b32 i,r,t;             \n\t"
                "  mov.b32 r, %0;              \n\t"
                "  mov.b32 i, %1;              \n\t"
                "  and.b32 t, r, 0x80008000;   \n\t"
                SPEC_CASE2(i,r,0X32B332B3, 0x08000800)
                SPEC_CASE2(i,r,0X5CB05CB0, 0x10001000)
                SPEC_CASE2(i,r,0XB2B3B2B3, 0x88008800)
                SPEC_CASE2(i,r,0XDCB0DCB0, 0x90009000)
                "  or.b32  r, r, t;            \n\t"
                "  mov.b32 %0, r;              \n"
                "}\n" : "+r"(r.x): "r"(a.x));
   return r;
}
__CUDA_FP16_DECL__ __half __hcos_internal(const __half a) {
   __half val;
   float f = __half2float(a);
   f = __float_simpl_cosf(f);
   val.x = __float2half_rn(f);
   return val;
}
__CUDA_FP16_DECL__ __half hcos(const __half a) {
   __half r = __hcos_internal(a);
   asm volatile("{\n\t"
                "  .reg.b16 i,r;        \n\t"
                "  mov.b16 r, %0;       \n\t"
                "  mov.b16 i, %1;       \n\t"
                SPEC_CASE(i,r,0X2B7C, 0x1000)
                SPEC_CASE(i,r,0XAB7C, 0x1000)
                "  mov.b16 %0, r;       \n"
                "}\n" : "+h"(r.x): "h"(a.x));
   return r;
}
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a) {
   __half l = __low2half(a);
   __half h = __high2half(a);
   __half2 r = __halves2half2(__hcos_internal(l), __hcos_internal(h));
   asm volatile("{\n\t"
                "  .reg.b32 i,r;   \n\t"
                "  mov.b32 r, %0;  \n\t"
                "  mov.b32 i, %1;  \n\t"
                SPEC_CASE2(i,r,0X2B7C2B7C, 0x10001000)
                SPEC_CASE2(i,r,0XAB7CAB7C, 0x10001000)
                "  mov.b32 %0, r;  \n"
                "}\n" : "+r"(r.x): "r"(a.x));
   return r;
}
static __device__ __forceinline__ float __internal_trig_reduction_kernel(float a, int *quadrant)
{
   float j, t;
   int q;
   q = __float2int_rn (a * 0.636619772f);
   j = (float)q;
   t = __fmaf_rn (-j, 1.5707962512969971e+000f, a);
   t = __fmaf_rn (-j, 7.5497894158615964e-008f, t);
   *quadrant = q;
   return t;
}
static __device__ __forceinline__ float __internal_sin_cos_kernel(float x, int i)
{
   float x2, z;
   x2 = x*x;

   if (i & 1) {
      z  =                   2.44331571e-5f;
      z  = __fmaf_rn (z, x2, -1.38873163e-3f);
   } else {
      z  =                  -1.95152959e-4f;
      z  = __fmaf_rn (z, x2,  8.33216087e-3f);
   }
   if (i & 1) {
      z  = __fmaf_rn (z, x2,  4.16666457e-2f);
      z  = __fmaf_rn (z, x2, -5.00000000e-1f);
   } else {
      z  = __fmaf_rn (z, x2, -1.66666546e-1f);
      z  = __fmaf_rn (z, x2,  0.0f);
   }
   x = __fmaf_rn (z, x, x);
   if (i & 1) x = __fmaf_rn (z, x2, 1.0f);
   if (i & 2) x = __fmaf_rn (x, -1.0f, 0.0f);
   return x;
}
static __device__ __forceinline__ float __float_simpl_sinf(float a)
{
   float z;
   int i;
   if (isinf(a)) {
      a = a * 0.0f;
   }
   a = __internal_trig_reduction_kernel(a, &i);
   z = __internal_sin_cos_kernel(a, i);
   return z;
}
static __device__ __forceinline__ float __float_simpl_cosf(float a)
{
   float z;
   int i;
   if (isinf(a)) {
      a = a * 0.0f;
   }
   a = __internal_trig_reduction_kernel(a, &i);
   i++;
   z = __internal_sin_cos_kernel(a, i);
   return z;
}
__CUDA_FP16_DECL__ __half hexp(const __half a) {
   __half val;
   asm volatile("{.reg.b32         f, C;           \n"
                " .reg.b16         h,r;            \n"
                "  mov.b16         h,%1;           \n"
                "  cvt.f32.f16     f,h;            \n"
                "  mov.b32         C, 0x3fb8aa3b;  \n"
                "  mul.f32         f,f,C;          \n"
                "  ex2.approx.f32      f,f;        \n"
                "  cvt.rn.f16.f32      r,f;        \n"
                SPEC_CASE(h,r,0X1F79, 0x9400)
                SPEC_CASE(h,r,0X25CF, 0x9400)
                SPEC_CASE(h,r,0XC13B, 0x0400)
                SPEC_CASE(h,r,0XC1EF, 0x0200)
                "  mov.b16         %0,r;           \n"
                "}": "=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;         \n"
                " .reg.b32         h,r,fl,fu, C;   \n"
                "  mov.b32         {hl, hu}, %1;   \n"
                "  mov.b32         h, %1;          \n"
                "  cvt.f32.f16     fl, hl;         \n"
                "  cvt.f32.f16     fu, hu;         \n"
                "  mov.b32         C, 0x3fb8aa3b;  \n"
                "  mul.f32         fl,fl,C;        \n"
                "  mul.f32         fu,fu,C;        \n"
                "  ex2.approx.f32      fl, fl;     \n"
                "  ex2.approx.f32      fu, fu;     \n"
                "  cvt.rn.f16.f32      hl, fl;     \n"
                "  cvt.rn.f16.f32      hu, fu;     \n"
                "  mov.b32         r, {hl, hu};    \n"
                SPEC_CASE2(h,r,0X1F791F79, 0x94009400)
                SPEC_CASE2(h,r,0X25CF25CF, 0x94009400)
                SPEC_CASE2(h,r,0XC13BC13B, 0x04000400)
                SPEC_CASE2(h,r,0XC1EFC1EF, 0x02000200)
                "  mov.b32         %0, r;  \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half hexp2(const __half a) {
   __half val;
   asm volatile("{.reg.b32         f, ULP;         \n"
                " .reg.b16         r;              \n"
                "  mov.b16         r,%1;           \n"
                "  cvt.f32.f16     f,r;            \n"
                "  ex2.approx.f32      f,f;        \n"
                "  mov.b32         ULP, 0x33800000;\n"
                "  fma.rn.f32      f,f,ULP,f;      \n"
                "  cvt.rn.f16.f32      r,f;        \n"
                "  mov.b16         %0,r;           \n"
                "}": "=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;         \n"
                " .reg.b32         fl, fu, ULP;    \n"
                "  mov.b32         {hl, hu}, %1;   \n"
                "  cvt.f32.f16     fl, hl;         \n"
                "  cvt.f32.f16     fu, hu;         \n"
                "  ex2.approx.f32      fl, fl;     \n"
                "  ex2.approx.f32      fu, fu;     \n"
                "  mov.b32         ULP, 0x33800000;\n"
                "  fma.rn.f32      fl,fl,ULP,fl;   \n"
                "  fma.rn.f32      fu,fu,ULP,fu;   \n"
                "  cvt.rn.f16.f32      hl, fl;     \n"
                "  cvt.rn.f16.f32      hu, fu;     \n"
                "  mov.b32         %0, {hl, hu};   \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half hexp10(const __half a) {
   __half val;
   asm volatile("{.reg.b16         h,r;            \n"
                " .reg.b32         f, C;           \n"
                "  mov.b16         h, %1;          \n"
                "  cvt.f32.f16     f, h;           \n"
                "  mov.b32         C, 0x40549A78;  \n"
                "  mul.f32         f,f,C;          \n"
                "  ex2.approx.f32      f, f;       \n"
                "  cvt.rn.f16.f32      r, f;       \n"
                SPEC_CASE(h,r,0x34DE, 0x9800)
                SPEC_CASE(h,r,0x9766, 0x9000)
                SPEC_CASE(h,r,0x9972, 0x1000)
                SPEC_CASE(h,r,0xA5C4, 0x1000)
                SPEC_CASE(h,r,0xBF0A, 0x8100)
                "  mov.b16         %0, r;          \n"
                "}":"=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;         \n"
                " .reg.b32         h,r,fl,fu, C;   \n"
                "  mov.b32         {hl, hu}, %1;   \n"
                "  mov.b32         h, %1;          \n"
                "  cvt.f32.f16     fl, hl;         \n"
                "  cvt.f32.f16     fu, hu;         \n"
                "  mov.b32         C, 0x40549A78;  \n"
                "  mul.f32         fl,fl,C;        \n"
                "  mul.f32         fu,fu,C;        \n"
                "  ex2.approx.f32      fl, fl;     \n"
                "  ex2.approx.f32      fu, fu;     \n"
                "  cvt.rn.f16.f32      hl, fl;     \n"
                "  cvt.rn.f16.f32      hu, fu;     \n"
                "  mov.b32         r, {hl, hu};    \n"
                SPEC_CASE2(h,r,0x34DE34DE, 0x98009800)
                SPEC_CASE2(h,r,0x97669766, 0x90009000)
                SPEC_CASE2(h,r,0x99729972, 0x10001000)
                SPEC_CASE2(h,r,0xA5C4A5C4, 0x10001000)
                SPEC_CASE2(h,r,0xBF0ABF0A, 0x81008100)
                "  mov.b32         %0, r;  \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half hlog2(const __half a) {
   __half val;
   asm volatile("{.reg.b16         h, r;           \n"
                " .reg.b32         f;              \n"
                "  mov.b16         h, %1;          \n"
                "  cvt.f32.f16     f, h;           \n"
                "  lg2.approx.f32      f, f;       \n"
                "  cvt.rn.f16.f32      r, f;       \n"
                SPEC_CASE(r,r, 0xA2E2, 0x8080)
                SPEC_CASE(r,r, 0xBF46, 0x9400)
                "  mov.b16         %0, r;          \n"
                "}":"=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;         \n"
                " .reg.b32         fl, fu, r, p;   \n"
                "  mov.b32         {hl, hu}, %1;   \n"
                "  cvt.f32.f16     fl, hl;         \n"
                "  cvt.f32.f16     fu, hu;         \n"
                "  lg2.approx.f32      fl, fl;     \n"
                "  lg2.approx.f32      fu, fu;     \n"
                "  cvt.rn.f16.f32      hl, fl;     \n"
                "  cvt.rn.f16.f32      hu, fu;     \n"
                "  mov.b32         r, {hl, hu};    \n"
                SPEC_CASE2(r,r, 0xA2E2A2E2, 0x80808080)
                SPEC_CASE2(r,r, 0xBF46BF46, 0x94009400)
                "  mov.b32         %0, r;          \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half hlog(const __half a) {
   __half val;
   asm volatile("{.reg.b32         f, C;           \n"
                " .reg.b16         r,h;            \n"
                "  mov.b16         h,%1;           \n"
                "  cvt.f32.f16     f,h;            \n"
                "  lg2.approx.f32      f,f;        \n"
                "  mov.b32         C, 0x3f317218;  \n"
                "  mul.f32         f,f,C;          \n"
                "  cvt.rn.f16.f32      r,f;        \n"
                SPEC_CASE(h,r, 0X160D, 0x9C00)
                SPEC_CASE(h,r, 0X3BFE, 0x8010)
                SPEC_CASE(h,r, 0X3C0B, 0x8080)
                SPEC_CASE(h,r, 0X6051, 0x1C00)
                "  mov.b16         %0,r;           \n"
                "}": "=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;             \n"
                " .reg.b32         r, fl, fu, C, h;    \n"
                "  mov.b32         {hl, hu}, %1;       \n"
                "  mov.b32         h, %1;              \n"
                "  cvt.f32.f16     fl, hl;             \n"
                "  cvt.f32.f16     fu, hu;             \n"
                "  lg2.approx.f32      fl, fl;         \n"
                "  lg2.approx.f32      fu, fu;         \n"
                "  mov.b32         C, 0x3f317218;      \n"
                "  mul.f32         fl,fl,C;            \n"
                "  mul.f32         fu,fu,C;            \n"
                "  cvt.rn.f16.f32      hl, fl;         \n"
                "  cvt.rn.f16.f32      hu, fu;         \n"
                "  mov.b32         r, {hl, hu};        \n"
                SPEC_CASE2(h,r, 0X160D160D, 0x9C009C00)
                SPEC_CASE2(h,r, 0X3BFE3BFE, 0x80108010)
                SPEC_CASE2(h,r, 0X3C0B3C0B, 0x80808080)
                SPEC_CASE2(h,r, 0X60516051, 0x1C001C00)
                "  mov.b32         %0, r;              \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half hlog10(const __half a) {
   __half val;
   asm volatile("{.reg.b16         h, r;           \n"
                " .reg.b32         f, C;           \n"
                "  mov.b16         h, %1;          \n"
                "  cvt.f32.f16     f, h;           \n"
                "  lg2.approx.f32      f, f;       \n"
                "  mov.b32         C, 0x3E9A209B;  \n"
                "  mul.f32         f,f,C;          \n"
                "  cvt.rn.f16.f32      r, f;       \n"
                SPEC_CASE(h,r,0x338F, 0x1000)
                SPEC_CASE(h,r,0x33F8, 0x9000)
                SPEC_CASE(h,r,0x57E1, 0x9800)
                SPEC_CASE(h,r,0x719D, 0x9C00)
                "  mov.b16         %0, r;          \n"
                "}":"=h"(val.x) : "h"(a.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a) {
   __half2 val;
   asm volatile("{.reg.b16         hl, hu;             \n"
                " .reg.b32         r, fl, fu, C, h;    \n"
                "  mov.b32         {hl, hu}, %1;       \n"
                "  mov.b32         h, %1;              \n"
                "  cvt.f32.f16     fl, hl;             \n"
                "  cvt.f32.f16     fu, hu;             \n"
                "  lg2.approx.f32      fl, fl;         \n"
                "  lg2.approx.f32      fu, fu;         \n"
                "  mov.b32         C, 0x3E9A209B;      \n"
                "  mul.f32         fl,fl,C;            \n"
                "  mul.f32         fu,fu,C;            \n"
                "  cvt.rn.f16.f32      hl, fl;         \n"
                "  cvt.rn.f16.f32      hu, fu;         \n"
                "  mov.b32         r, {hl, hu};        \n"
                SPEC_CASE2(h,r,0x338F338F, 0x10001000)
                SPEC_CASE2(h,r,0x33F833F8, 0x90009000)
                SPEC_CASE2(h,r,0x57E157E1, 0x98009800)
                SPEC_CASE2(h,r,0x719D719D, 0x9C009C00)
                "  mov.b32         %0, r;              \n"
                "}":"=r"(val.x) : "r"(a.x));
   return val;
}
#undef SPEC_CASE2
#undef SPEC_CASE
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a) {
   APPROX_FCAST2(rcp);
}
__CUDA_FP16_DECL__ __half hrcp(const __half a) {
   APPROX_FCAST(rcp);
}
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a) {
   APPROX_FCAST2(rsqrt);
}
__CUDA_FP16_DECL__ __half hrsqrt(const __half a) {
   APPROX_FCAST(rsqrt);
}
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a) {
   APPROX_FCAST2(sqrt);
}
__CUDA_FP16_DECL__ __half hsqrt(const __half a) {
   APPROX_FCAST(sqrt);
}
#undef APPROX_FCAST
#undef APPROX_FCAST2
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a)
{
   __half2 r;
   asm( "{set.nan.f16x2.f16x2 %0,%1,%2;\n}"
        :"=r"(r.x) : "r"(a.x),"r"(a.x));
   return r;
}
__CUDA_FP16_DECL__ bool __hisnan(const __half a)
{
   __half r;
   asm( "{set.nan.f16.f16 %0,%1,%2;\n}"
        :"=h"(r.x) : "h"(a.x),"h"(a.x));
   if (r.x == 0)
      return false;
   else return true;
}
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a)
{
   __half2 zero = __float2half2_rn(0.0);
   return __hsub2(zero,a);
}
__CUDA_FP16_DECL__ __half __hneg(const __half a)
{
   __half zero;
   zero = __float2half(0.0);
   return __hsub(zero,a);
}
#endif /*__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
#undef __CUDA_FP16_DECL__
#endif /*defined(__CUDACC__)*/
#endif /* end of include guard: CUDA_FP16_H_JNESTUG4 */
