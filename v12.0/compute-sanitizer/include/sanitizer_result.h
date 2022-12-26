/*
 * Copyright 2018-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__SANITIZER_RESULT_H__)
#define __SANITIZER_RESULT_H__

#ifndef SANITIZERAPI
#ifdef _WIN32
#define SANITIZERAPI __stdcall
#else
#define SANITIZERAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \defgroup SANITIZER_RESULT_API Sanitizer Result Codes
 * Error and result codes returned by Sanitizer functions.
 * @{
 */

/**
 * \addtogroup SANITIZER_RESULT_API
 * @{
 */

/**
 * \brief Sanitizer result codes.
 *
 * Error and result codes returned by Sanitizer functions.
 */
typedef enum {
    /**
     * No error.
     */
    SANITIZER_SUCCESS                             = 0,

    /**
     * One or more of the parameters is invalid.
     */
    SANITIZER_ERROR_INVALID_PARAMETER             = 1,

    /**
     * The device does not correspond to a valid CUDA device.
     */
    SANITIZER_ERROR_INVALID_DEVICE                = 2,

    /**
     * The context is NULL or not valid.
     */
    SANITIZER_ERROR_INVALID_CONTEXT               = 3,

    /**
     * The domain ID is invalid.
     */
    SANITIZER_ERROR_INVALID_DOMAIN_ID             = 4,

    /**
     * The callback ID is invalid.
     */
    SANITIZER_ERROR_INVALID_CALLBACK_ID           = 5,

    /**
     * The current operation cannot be performed due to dependency on
     * other factors.
     */
    SANITIZER_ERROR_INVALID_OPERATION             = 6,

    /**
     * Unable to allocate enough memory to perform the requested
     * operation.
     */
    SANITIZER_ERROR_OUT_OF_MEMORY                 = 7,

    /**
     * The output buffer size is not sufficient to return all
     * requested data.
     */
    SANITIZER_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT = 8,

    /**
     * API is not implemented.
     */
    SANITIZER_ERROR_API_NOT_IMPLEMENTED           = 9,

    /**
     * The maximum limit is reached.
     */
    SANITIZER_ERROR_MAX_LIMIT_REACHED             = 10,

    /**
     * The object is not ready to perform the requested operation.
     */
    SANITIZER_ERROR_NOT_READY                     = 11,

    /**
     * The current operation is not compatible with the current state
     * of the object.
     */
    SANITIZER_ERROR_NOT_COMPATIBLE                = 12,

    /**
     * Sanitizer is unable to initialize its connection to the CUDA
     * driver.
     */
    SANITIZER_ERROR_NOT_INITIALIZED               = 13,

    /**
     * The attempted operation is not supported on the current system
     * or device
     */
    SANITIZER_ERROR_NOT_SUPPORTED                 = 14,

    /**
     * The attempted device operation has a parameter not in device memory
     */
    SANITIZER_ERROR_ADDRESS_NOT_IN_DEVICE_MEMORY  = 15,

    /**
     * An unknown internal error has occurred.
     */
    SANITIZER_ERROR_UNKNOWN                       = 999,

    SANITIZER_ERROR_FORCE_INT                     = 0x7fffffff
} SanitizerResult;

/**
 * Get the descriptive string for a SanitizerResult.
 *
 * Return the descriptive string for a SanitizerResult in \p *str.
 * \note \b Thread-safety: this function is thread-safe.
 *
 * \param result The result to get the string for
 * \param str Returns the string
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p str is NULL or \p
 * result is not a valid SanitizerResult.
 */
SanitizerResult SANITIZERAPI sanitizerGetResultString(SanitizerResult result,
                                                      const char **str);

/** @} */ /* END SANITIZER_RESULT_API */

#if defined(__cplusplus)
}
#endif

#endif /* __SANITIZER_RESULT_H__ */
