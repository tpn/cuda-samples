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

#if !defined(__SANITIZER_PATCHING_H__)
#define __SANITIZER_PATCHING_H__

#include <sanitizer_memory.h>
#include <sanitizer_result.h>

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
 * \defgroup SANITIZER_PATCHING_API Sanitizer Patching API
 * Functions, types, and enums that implement the Sanitizer Patching API.
 * @{
 */

/**
 * \addtogroup SANITIZER_PATCHING_API
 * @{
 */

typedef struct Sanitizer_Launch_st *Sanitizer_LaunchHandle;

/**
 * \brief Load a module containing patches that can be used by the
 * patching API.
 *
 * \note \b Thread-safety: an API user must serialize access to
 * sanitizerAddPatchesFromFile, sanitizerAddPatches, sanitizerPatchInstructions,
 * and sanitizerPatchModule. For example if sanitizerAddPatchesFromFile(filename)
 * and sanitizerPatchInstruction(*, *, cbName) are called concurrently and
 * cbName is intended to be found in the loaded module, the results are
 * undefined.
 *
 * \note The patches loaded are only valid for the specified CUDA context.
 *
 * \param filename Path to the module file. This API supports the same module
 * formats as the cuModuleLoad function from the CUDA driver API.
 * \param ctx CUDA context in which to load the patches. If ctx is NULL, the
 * current context will be used.
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_NOT_INITIALIZED if unable to initialize the sanitizer
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p filename is not a path to
 * a valid CUDA module.
 */
SanitizerResult SANITIZERAPI sanitizerAddPatchesFromFile(const char* filename,
                                                         CUcontext ctx);

/**
 * \brief Load a module containing patches that can be used by the
 * patching API.
 *
 * \note \b Thread-safety: an API user must serialize access to
 * sanitizerAddPatchesFromFile, sanitizerAddPatches, sanitizerPatchInstructions,
 * and sanitizerPatchModule. For example if sanitizerAddPatches(image) and
 * sanitizerPatchInstruction(*, *, cbName) are called concurrently and cbName
 * is intended to be found in the loaded image, the results are undefined.
 *
 * \note The patches loaded are only valid for the specified CUDA context.
 *
 * \param image Pointer to module data to load. This API supports the same
 * module formats as the cuModuleLoadData and cuModuleLoadFatBinary functions
 * from the CUDA driver API.
 * \param ctx CUDA context in which to load the patches. If ctx is NULL, the
 * current context will be used.
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_NOT_INITIALIZED if unable to initialize the sanitizer
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p image does not point to a
 * valid CUDA module.
 */
SanitizerResult SANITIZERAPI sanitizerAddPatches(const void* image,
                                                 CUcontext ctx);

/**
 * \brief Sanitizer patch result codes
 *
 * Error and result codes returned by Sanitizer patches.
 * If a patch returns SANITIZER_PATCH_ERROR, the thread
 * will be exited. On Volta and newer architectures, the
 * full warp which the thread belongs to will be exited.
 */
typedef enum {
    /**
     * No error.
     */
    SANITIZER_PATCH_SUCCESS                 = 0,

    /**
     * An error was detected in the patch.
     */
    SANITIZER_PATCH_ERROR                   = 1,

    SANITIZER_PATCH_FORCE_INT               = 0x7fffffff
} SanitizerPatchResult;

/**
 * \brief Function type for a CUDA block enter callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the entry point of the block
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackBlockEnter)(void* userdata, uint64_t pc);

/**
 * \brief Function type for a CUDA block exit callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackBlockExit)(void* userdata, uint64_t pc);

/**
 * \brief Flags describing a memory access
 *
 * Flags describing a memory access. These values are to be or-combined in the
 * value of \b flags for a SanitizerCallbackMemoryAccess callback.
 */
typedef enum {
    /**
     * Empty flag.
     */
    SANITIZER_MEMORY_DEVICE_FLAG_NONE      = 0,

    /**
     * Specifies that the access is a read.
     */
    SANITIZER_MEMORY_DEVICE_FLAG_READ      = 0x1,

    /**
     * Specifies that the access is a write.
     */
    SANITIZER_MEMORY_DEVICE_FLAG_WRITE     = 0x2,

    /**
     * Specifies that the access is a system-scoped atomic.
     */
    SANITIZER_MEMORY_DEVICE_FLAG_ATOMSYS   = 0x4,

    SANITIZER_MEMORY_DEVICE_FLAG_FORCE_INT = 0x7fffffff
} Sanitizer_DeviceMemoryFlags;

/**
 * \brief Function type for a memory access callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p ptr is the address of the memory being accessed.
 * For local or shared memory access, this is the offset within the local
 * or shared memory window.
 * \p accessSize is the size of the access in bytes. Valid values are 1, 2, 4,
 * 8, and 16.
 * \p flags contains information about the type of access. See
 * Sanitizer_DeviceMemoryFlags to interpret this value.
 * \p newValue is a pointer to the new value being written if the acces is a
 * write. If the access is a read or an atomic, the pointer will be NULL.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackMemoryAccess)(void* userdata, uint64_t pc, void* ptr, uint32_t accessSize, uint32_t flags, const void* newValue);

/**
 * \brief Flags describing a barrier
 *
 * Flags describing a barrier. These values are to be or-combined in the
 * value of \b flags for a SanitizerCallbackBarrier callback.
 */
typedef enum {
    /**
     * Empty flag.
     */
    SANITIZER_BARRIER_FLAG_NONE              = 0,

    /**
     * Specifies that the barrier can be called unaligned.
     * This flag is only valid on SM 7.0 and above.
     */
    SANITIZER_BARRIER_FLAG_UNALIGNED_ALLOWED = 0x1,

    SANITIZER_BARRIER_FLAG_FORCE_INT         = 0x7fffffff
} Sanitizer_BarrierFlags;

/**
 * \brief Function type for a barrier callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p barIndex is the barrier index.
 * \p threadCount is the number of expected threads (must be a multiple of the warp size).
 * \p flags contains information about the barrier.
 * See Sanitizer_BarrierFlags to interpret this value.
 * 0 means that all threads are participating in the barrier.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackBarrier)(void* userdata, uint64_t pc, uint32_t barIndex, uint32_t threadCount, uint32_t flags);

/**
 * \brief Function type for a syncwarp callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p mask is the thread mask passed to __syncwarp().
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackSyncwarp)(void* userdata, uint64_t pc, uint32_t mask);

/**
 * \brief Function type for a shfl callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 *
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackShfl)(void* userdata, uint64_t pc);

/**
 * \brief Flags describing a function call
 *
 * Flags describing a function call. These values are to be or-combined in the
 * value of \b flags for a SanitizerCallbackCall callback.
 */
typedef enum {
    /**
     * Empty flag.
     */
    SANITIZER_CALL_FLAG_NONE              = 0,

    /**
     * Specifies that barriers within this function call can be called unaligned.
     * This flag is only valid on SM 7.0 and above.
     */
    SANITIZER_CALL_FLAG_UNALIGNED_ALLOWED = 0x1,

    SANITIZER_CALL_FLAG_FORCE_INT         = 0x7fffffff
} Sanitizer_CallFlags;

/**
 * \brief Function type for a function call callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p targetPc is the PC where the called function is located.
 * \p flags contains information about the function call.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackCall)(void* userdata, uint64_t pc, uint64_t targetPc, uint32_t flags);

/**
 * \brief Function type for a function return callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 *
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackRet)(void* userdata, uint64_t pc);

/**
 * \brief Function type for a device-side malloc call.
 *
 * \note This is called after the call has completed.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p allocatedPtr is the pointer returned by device-side malloc
 * \p allocatedSize is the size requested by the user to device-side malloc.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackDeviceSideMalloc)(void* userdata, uint64_t pc, void* allocatedPtr, uint64_t allocatedSize);

/**
 * \brief Function type for a device-side free call.
 *
 * \note This is called prior to the actual call.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p ptr is the pointer passed to device-side free.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackDeviceSideFree)(void* userdata, uint64_t pc, void* ptr);

/**
 * \brief CUDA Barrier action kind.
 *
 * Refer to the CUDA Barrier interface section of the CUDA toolkit documentation for
 * a more extensive description of these actions.
 */
typedef enum {

    /**
     * Invalid action ID.
     */
    SANITIZER_CUDA_BARRIER_INVALID                = 0,

    /**
     * Barrier initialization.
     */
    SANITIZER_CUDA_BARRIER_INIT                   = 1,

    /**
     * Barrier arrive operation. On Hopper and newer architectures,
     * barrier data is the count argument to the arrive-on operation.
     */
    SANITIZER_CUDA_BARRIER_ARRIVE                 = 2,

    /**
     * Barrier arrive and drop operation. On Hopper and newer architectures,
     * barrier data is the count argument to the arrive-on operation.
     */
    SANITIZER_CUDA_BARRIER_ARRIVE_DROP            = 3,

    /**
     * Barrier arrive operation without phase completion.
     * Barrier data is the count argument to the arrive-on operation.
     */
    SANITIZER_CUDA_BARRIER_ARRIVE_NOCOMPLETE      = 4,

    /**
     * Barrier arrive and drop operation without phase completion.
     * Barrier data is the count argument to the arrive-on operation.
     */
    SANITIZER_CUDA_BARRIER_ARRIVE_DROP_NOCOMPLETE = 5,

    /**
     * Barrier wait operation.
     */
    SANITIZER_CUDA_BARRIER_WAIT                   = 6,

    /**
     * Barrier invalidation.
     */
    SANITIZER_CUDA_BARRIER_INVALIDATE             = 7,

    SANITIZER_CUDA_BARRIER_FORCE_INT              = 0x7fffffff
} Sanitizer_CudaBarrierInstructionKind;

/**
 * \brief Function type for a CUDA Barrier action callback.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p barrier Barrier address which can be used as a unique identifier
 * \p kind Barrier action type. See \ref Sanitizer_CudaBarrierInstructionKind
 * \p data Barrier data. This is specific to each action type, refer to \ref Sanitizer_CudaBarrierInstructionKind
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackCudaBarrier)(void* userdata, uint64_t pc, void* barrier, uint32_t kind, uint32_t data);

/**
 * \brief Function type for a global to shared memory asynchronous copy.
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p src is the address of the global memory being read.
 * This can be NULL if src-size is 0.
 * \p dst is the address of the shared memory being written.
 * This is an offset within the shared memory window
 * \p accessSize is the size of the access in bytes. Valid values are 4, 8 and 16.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackMemcpyAsync)(void* userdata, uint64_t pc, void* src, uint32_t dst, uint32_t accessSize);

/**
 * \brief Function type for a pipeline commit
 *
 * This can be generated by a pipeline::producer_commit (C++ API), a pipeline_commit (C API)
 * or a cp.async.commit_group (PTX API).
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackPipelineCommit)(void* userdata, uint64_t pc);

/**
 * \brief Function type for a pipeline wait
 *
 * This can be generated by a pipeline::consumer_wait (C++ API), a pipeline_wait_prior (C API),
 * cp.async.wait_group or cp.async.wait_all (PTX API).
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p groups is the number of groups the pipeline will wait for. 0 is used to wait for all groups.
 */
typedef SanitizerPatchResult (SANITIZERAPI *SanitizerCallbackPipelineWait)(void* userdata, uint64_t pc, uint32_t groups);

/**
 * \brief Function type for a matrix shared memory access callback
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p address is the address of the shared memory being read or written.
 * This is an offset within the shared memory window
 * \p accessSize is the size of the access in bytes. Valid value is 16.
 * \p flags contains information about the type of access. See
 * Sanitizer_DeviceMemoryFlags to interpret this value.
 * \p count is the number of matrices accessed.
 * \p newValue is a pointer to the new value being written if the acces is a
 * write. If the access is a read or an atomic, the pointer will be NULL.
 */
typedef SanitizerPatchResult(SANITIZERAPI* SanitizerCallbackMatrixMemoryAccess)(void* userdata, uint64_t pc, uint32_t address, uint32_t accessSize, uint32_t flags, uint32_t count, const void* pNewValue);

/**
 * \brief Cache control action
 */
typedef enum {

    /**
     * Invalid action ID.
     */
    SANITIZER_CACHE_CONTROL_INVALID           = 0,

    /**
     * Prefetch to L1.
     */
    SANITIZER_CACHE_CONTROL_L1_PREFETCH       = 1,

    /**
     * Prefetch to L2.
     */
    SANITIZER_CACHE_CONTROL_L2_PREFETCH       = 2,

    SANITIZER_CACHE_CONTROL_FORCE_INT         = 0x7fffffff
} Sanitizer_CacheControlInstructionKind;

/**
 * \brief Function type for a cache control instruction callback
 *
 * \p userdata is a pointer to user data. See \ref sanitizerPatchModule
 * \p pc is the program counter of the patched instruction
 * \p address is the address of the memory being controlled
 * \p kind is the type of cache control. See \ref Sanitizer_CacheControlInstructionKind
 */
typedef SanitizerPatchResult(SANITIZERAPI* SanitizerCallbackCacheControl)(void* userdata, uint64_t pc, void* address, Sanitizer_CacheControlInstructionKind kind);

/**
 * \brief Instrumentation.
 *
 * Instrumentation. Every entry represent an instruction type or a function
 * call where a callback patch can be inserted.
 */
typedef enum {
    /**
     * Invalid instruction ID.
     */
    SANITIZER_INSTRUCTION_INVALID              = 0,

    /**
     * CUDA block enter. This is called prior to any user code. The type of the
     * callback must be SanitizerCallbackBlockEnter.
     */
    SANITIZER_INSTRUCTION_BLOCK_ENTER          = 1,

    /**
     * CUDA block exit. This is called after all user code has executed. The type of
     * the callback must be SanitizerCallbackBlockExit.
     */
    SANITIZER_INSTRUCTION_BLOCK_EXIT           = 2,

    /**
     * Global Memory Access. This can be a store, load or atomic operation. The type
     * of the callback must be SanitizerCallbackMemoryAccess.
     */
    SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS = 3,

    /**
     * Shared Memory Access. This can be a store, load or atomic operation. The type
     * of the callback must be SanitizerCallbackMemoryAccess.
     */
    SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS = 4,

    /**
     * Local Memory Access. This can be a store or load operation. The type
     * of the callback must be SanitizerCallbackMemoryAccess.
     */
    SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS  = 5,

    /**
     * Barrier. The type of the callback must be SanitizerCallbackBarrier.
     */
    SANITIZER_INSTRUCTION_BARRIER              = 6,

    /**
     * Syncwarp. The type of the callback must be SanitizerCallbackSyncwarp.
     */
    SANITIZER_INSTRUCTION_SYNCWARP             = 7,

    /**
     * Shfl. The type of the callback must be SanitizerCallbackShfl.
     */
    SANITIZER_INSTRUCTION_SHFL                 = 8,

    /**
     * Function call. The type of the callback must be SanitizerCallbackCall.
     */
    SANITIZER_INSTRUCTION_CALL                 = 9,

    /**
     * Function return. The type of the callback must be SanitizerCallbackRet.
     */
    SANITIZER_INSTRUCTION_RET                  = 10,

    /**
     * Device-side malloc. The type of the callback must be
     * SanitizerCallbackDeviceSideMalloc.
     */
    SANITIZER_INSTRUCTION_DEVICE_SIDE_MALLOC   = 11,

    /**
     * Device-side free. The type of the callback must be
     * SanitizerCallbackDeviceSideFree.
     */
    SANITIZER_INSTRUCTION_DEVICE_SIDE_FREE     = 12,

    /**
     * CUDA Barrier operation. The type of the callback must be
     * SanitizerCallbackCudaBarrier.
     */
    SANITIZER_INSTRUCTION_CUDA_BARRIER         = 13,

    /**
     * Global to shared memory asynchronous copy. The type of the
     * callback must be SanitizerCallbackMemcpyAsync.
     */
    SANITIZER_INSTRUCTION_MEMCPY_ASYNC         = 14,

    /**
     * Pipeline commit. The type of the callback must be
     * SanitizerCallbackPipelineCommit.
     */
    SANITIZER_INSTRUCTION_PIPELINE_COMMIT      = 15,

    /**
     * Pipeline wait. The type of the callback must be
     * SanitizerCallbackPipelineWait.
     */
    SANITIZER_INSTRUCTION_PIPELINE_WAIT        = 16,

    /**
     * Remote Shared Memory Access. This can be a store or load operation. The
     * type of the callback must be SanitizerCallbackMemoryAccess.
     */
    SANITIZER_INSTRUCTION_REMOTE_SHARED_MEMORY_ACCESS = 17,

    /**
     * Device-side aligned malloc. The type of the callback must be
     * SanitizerCallbackDeviceSideMalloc.
     */
    SANITIZER_INSTRUCTION_DEVICE_ALIGNED_MALLOC       = 18,

    /**
     * Matrix shared memory access. The type of the callback must
     * be SanitizerCallbackMatrixMemoryAccess.
     */
    SANITIZER_INSTRUCTION_MATRIX_MEMORY_ACCESS = 19,

    /**
     * Cache control instruction. The type of the callback must
     * be SanitizerCallbackCacheControl.
     */
    SANITIZER_INSTRUCTION_CACHE_CONTROL        = 20,

    SANITIZER_INSTRUCTION_FORCE_INT            = 0x7fffffff
} Sanitizer_InstructionId;

/**
 * \brief Set instrumentation points and patches to be applied in a module.
 *
 * Mark that all instrumentation points matching instructionId are to be
 * patched in order to call the device function identified by
 * deviceCallbackName. It is up to the API client to ensure that this
 * device callback exists and match the correct callback format for
 * this instrumentation point.
 * \note \b Thread-safety: an API user must serialize access to
 * sanitizerAddPatchesFromFile, sanitizerAddPatches, sanitizerPatchInstructions,
 * and sanitizerPatchModule. For example if sanitizerAddPatches(fileName) and
 * sanitizerPatchInstruction(*, *, cbName) are called concurrently and cbName
 * is intended to be found in the loaded module, the results are undefined.
 *
 * \param instructionId Instrumentation point for which to insert patches
 * \param module CUDA module to instrument
 * \param deviceCallbackName Name of the device function callback that the
 * inserted patch will call at the instrumented points. This function is
 * expected to be found in code previously loaded by sanitizerAddPatchesFromFile
 * or sanitizerAddPatches.
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_NOT_INITIALIZED if unable to initialize the sanitizer
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p module is not a CUDA module
 * or if \p deviceCallbackName function cannot be located.
 */
SanitizerResult SANITIZERAPI sanitizerPatchInstructions(const Sanitizer_InstructionId instructionId,
                                                        CUmodule module,
                                                        const char* deviceCallbackName);

/**
 *
 * \brief Perform the actual instrumentation of a module.
 *
 * Perform the instrumentation of a CUDA module based on previous calls to
 * sanitizerPatchInstructions. This function also specifies the device memory
 * buffer to be passed in as userdata to all callback functions.
 * \note \b Thread-safety: an API user must serialize access to
 * sanitizerAddPatchesFromFile, sanitizerAddPatches, sanitizerPatchInstructions,
 * and sanitizerPatchModule. For example if sanitizerPatchModule(mod, *) and
 * sanitizerPatchInstruction(*, mod, *) are called concurrently, the results
 * are undefined.
 *
 * \param module CUDA module to instrument
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p module is not a CUDA module
 */
SanitizerResult SANITIZERAPI sanitizerPatchModule(CUmodule module);

/**
 * \brief Specifies the user data pointer for callbacks
 *
 * Mark all subsequent launches of \p kernel to use \p userdata
 * pointer as the device memory buffer to pass in to callback functions.
 *
 * \param kernel CUDA function to link to user data. Callbacks in subsequent
 * launches on this kernel will use \p userdata as callback data.
 * \param userdata Device memory buffer. This data will be passed to callback
 * functions via the \p userdata parameter.
 *
 * \retval SANITIZER_SUCCESS on success
 */
SanitizerResult SANITIZERAPI sanitizerSetCallbackData(CUfunction kernel,
                                                      const void* userdata);

/**
 * \brief Specifies the user data pointer for callbacks
 *
 * Mark \p launch to use \p userdata pointer as the device memory buffer
 * to pass in to callback functions. This function is only available if
 * the driver version is 455 or newer.
 *
 * \param launch Kernel launch to link to user data. Callbacks in this kernel
 * launch will use \p userdata as callback data.
 * \param kernel CUDA function associated with the kernel launch.
 * \param stream CUDA stream associated with the stream launch.
 * \param userdata Device memory buffer. This data will be passed to callback
 * functions via the \p userdata parameter.
 *
 * \retval SANITIZER_SUCCESS on success
 */
SanitizerResult SANITIZERAPI sanitizerSetLaunchCallbackData(Sanitizer_LaunchHandle launch,
                                                            CUfunction kernel,
                                                            Sanitizer_StreamHandle stream,
                                                            const void* userdata);

/**
 *
 * \brief Remove existing instrumentation of a module
 *
 * Remove any instrumentation of a CUDA module performed by previous calls
 * to sanitizerPatchModule.
 * \note \b Thread-safety: an API user must serialize access to
 * sanitizerPatchModule and sanitizerUnpatchModule on the same module.
 * For example, if sanitizerPatchModule(mod) and sanitizerUnpatchModule(mod)
 * are called concurrently, the results are undefined.
 *
 * \param module CUDA module on which to remove instrumentation
 *
 * \retval SANITIZER_SUCCESS on success
 */
SanitizerResult SANITIZERAPI sanitizerUnpatchModule(CUmodule module);

/**
 *
 * \brief Get PC and size of a CUDA function
 *
 * \param[in] module CUDA module containing the function
 * \param[in] deviceCallbackName CUDA function name
 * \param[out] pc Function start program counter (PC) returned
 * \param[out] size Function size in bytes returned
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p functionName function
 * cannot be located, if pc is NULL or if size is NULL.
 *
 */
SanitizerResult SANITIZERAPI sanitizerGetFunctionPcAndSize(CUmodule module,
                                                           const char* functionName,
                                                           uint64_t* pc,
                                                           uint64_t* size);

/**
 *
 * \brief Get PC and size of a device callback
 *
 * \param[in] ctx CUDA context in which the patches were loaded.
 * If ctx is NULL, the current context will be used.
 * \param[in] deviceCallbackName device function callback name
 * \param[out] pc Callback PC returned
 * \param[out] size Callback size returned
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p deviceCallbackName function
 * cannot be located, if pc is NULL or if size is NULL.
 *
 */
SanitizerResult SANITIZERAPI sanitizerGetCallbackPcAndSize(CUcontext ctx,
                                                           const char* deviceCallbackName,
                                                           uint64_t* pc,
                                                           uint64_t* size);

typedef enum {
    /**
     * The function is not loaded.
     */
    SANITIZER_FUNCTION_NOT_LOADED           = 0x0,

    /**
     * The function is being loaded.
     */
    SANITIZER_FUNCTION_PARTIALLY_LOADED     = 0x1,

    /**
     * The function is fully loaded.
     */
    SANITIZER_FUNCTION_LOADED               = 0x2,

    SANITIZER_FUNCTION_LOADED_FORCE_INT     = 0x7fffffff
} Sanitizer_FunctionLoadedStatus;

/**
 *
 * \brief Get the loading status of a function. Requires a driver version >=515.
 *
 * \param[in] func CUDA function for which the loading status is queried.
 * \param[out] loadingStatus Loading status returned
 *
 * \retval SANITIZER_SUCCESS on success
 * \retval SANITIZER_ERROR_INVALID_PARAMETER if \p func is NULL or if
 * loadingStatus is NULL.
 * \retval SANITIZER_ERROR_NOT_SUPPORTED if the loading status cannot be queried
 * with this driver version.
 *
 */
SanitizerResult SANITIZERAPI sanitizerGetFunctionLoadedStatus(CUfunction func,
                                                              Sanitizer_FunctionLoadedStatus* loadingStatus);


/** @} */ /* END SANITIZER_PATCHING_API */

#if defined(__cplusplus)
}
#endif

#endif /* __SANITIZER_PATCHING_H__ */
