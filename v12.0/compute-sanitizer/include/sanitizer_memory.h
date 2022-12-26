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

#if !defined(__SANITIZER_MEMORY_H__)
#define __SANITIZER_MEMORY_H__

#include <sanitizer_result.h>
#include <sanitizer_stream.h>

#include <cuda.h>

#include <stdint.h>

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
 * \defgroup SANITIZER_MEMORY_API Sanitizer Memory API
 * Functions, types, and enums that implement the Sanitizer Memory API.
 * @{
 */

/**
 * \addtogroup SANITIZER_MEMORY_API
 * @{
 */

/**
 * \brief Allocate memory on the device
 *
 * Equivalent of cudaMalloc that can be called within a callback function.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param ctx Context for the allocation. If NULL, the current context will be used.
 * \param devPtr Pointer to allocated device memory
 * \param size Allocation size in bytes
 */
SanitizerResult SANITIZERAPI sanitizerAlloc(CUcontext ctx,
                                            void** devPtr,
                                            size_t size);

/**
 * \brief Allocate host pinned memory
 *
 * Equivalent of cudaMallocHost that can be called within a callback function.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param ctx Context for the allocation. If NULL, the current context will be used.
 * \param devPtr Pointer to allocated host memory
 * \param size Allocation size in bytes
 */
SanitizerResult SANITIZERAPI sanitizerAllocHost(CUcontext ctx,
                                                void** devPtr,
                                                size_t size);
/**
 * \brief Frees memory on the device
 *
 * Equivalent of cudaFree that can be called within a callback function.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param ctx Context for the allocation. If NULL, the current context will be used.
 * \param devPtr Device pointer to memory to free
 */
SanitizerResult SANITIZERAPI sanitizerFree(CUcontext ctx,
                                           void* devPtr);
/**
 * \brief Frees host memory
 *
 * Equivalent of cudaFreeHost that can be called within a callback function.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param ctx Context for the allocation. If NULL, the current context will be used.
 * \param devPtr Host pointer to memory to free
 */
SanitizerResult SANITIZERAPI sanitizerFreeHost(CUcontext ctx,
                                               void* devPtr);


/**
 * \brief Copies data from host to device
 *
 * Equivalent of cudaMemcpyAsync that can be called within a callback function.
 * The function will return once the pageable buffer has been copied to the
 * staging memory for DMA transfer to device memory, but the DMA to final
 * destination may not have completed.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param dst Destination memory address
 * \param src Source memory address
 * \param count Size in bytes to copy
 * \param stream Stream handle. If NULL, the NULL stream will be used.
 */
SanitizerResult SANITIZERAPI sanitizerMemcpyHostToDeviceAsync(void* dst,
                                                              void* src,
                                                              size_t count,
                                                              Sanitizer_StreamHandle stream);

/**
 * \brief Copies data from device to host
 *
 * Equivalent of cudaMemcpy that can be called within a callback function.
 * The function will return once the copy has completed.
 * If the function is called from a SANITIZER_CB_DOMAIN_LAUNCH,
 * SANITIZER_CB_DOMAIN_MEMCPY or SANITIZER_CB_DOMAIN_MEMSET callback,
 * only pinned host memory may be used as destination.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param dst Destination memory address
 * \param src Source memory address
 * \param count Size in bytes to copy
 * \param stream Stream handle. If NULL, the NULL stream will be used.
 */
SanitizerResult SANITIZERAPI sanitizerMemcpyDeviceToHost(void* dst,
                                                         void* src,
                                                         size_t count,
                                                         Sanitizer_StreamHandle stream);

/**
 * \brief Initializes or sets device memory to a value.
 *
 * Equivalent of cudaMemset that can be called within a callback function.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param devPtr Pointer to device memory
 * \param value value to set for each byte of specified memory
 * \param count Size in bytes to set
 * \param stream Stream handle. If NULL, the NULL stream will be used.
 */
SanitizerResult SANITIZERAPI sanitizerMemset(void* devPtr,
                                             int value,
                                             size_t count,
                                             Sanitizer_StreamHandle stream);

/** @} */ /* END SANITIZER_MEMORY_API */

#if defined(__cplusplus)
}
#endif

#endif /* __SANITIZER_MEMORY_H__ */
