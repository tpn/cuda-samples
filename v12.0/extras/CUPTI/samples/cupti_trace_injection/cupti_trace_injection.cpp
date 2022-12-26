/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved.
 *
 * CUPTI based tracing injection to trace any CUDA application.
 * This sample demonstrates how to use activity
 * and callback APIs in the injection code.
 * Refer to the README.txt file for usage.
 *
 * Workflow in brief:
 *
 *  After the initialization routine returns, the application resumes running,
 *  with the registered callbacks triggering as expected.
 *  Subscribed to ProfilerStart and ProfilerStop callbacks. These callbacks
 *  control the collection of profiling data.
 *
 *  ProfilerStart callback:
 *      Start the collection by enabling activities. Also enable callback for
 *      the API cudaDeviceReset to flush activity buffers.
 *
 *  ProfilerStop callback:
 *      Get all the activity buffers which have all the activity records completed
 *      by using cuptiActivityFlushAll() API and then disable cudaDeviceReset callback
 *      and all the activities to stop collection.
 *
 *  atExitHandler:
 *      Register to the atexit handler to get all the activity buffers including the ones
 *      which have incomplete activity records by using force flush API
 *      cuptiActivityFlushAll(1).
 */
#include <cuda.h>
#include <cupti.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include "detours.h"
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

// Variable related to initialize injection .
std::mutex initializeInjectionMutex;

// Macros
#define IS_ACTIVITY_SELECTED(activitySelect, activityKind)                               \
    (activitySelect & (1LL << activityKind))

#define SELECT_ACTIVITY(activitySelect, activityKind)                                    \
    (activitySelect |= (1LL << activityKind))

#define CUPTI_CALL(call)                                                                 \
    do {                                                                                 \
        CUptiResult _status = call;                                                      \
        if (_status != CUPTI_SUCCESS) {                                                  \
            const char *errstr;                                                          \
            cuptiGetResultString(_status, &errstr);                                      \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",         \
                    __FILE__, __LINE__, #call, errstr);                                  \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

#define BUF_SIZE (8 * 1024 * 1024) // 8MB
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                      \
    (((uintptr_t)(buffer) & ((align)-1))                                                 \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))                    \
         : (buffer))

// Global Structure

typedef struct {
    volatile uint32_t initialized;
    CUpti_SubscriberHandle subscriber;
    int tracingEnabled;
    uint64_t profileMode;
} injGlobalControl;

injGlobalControl globalControl;

// Function Declarations

static CUptiResult cuptiInitialize(void);

static void atExitHandler(void);

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, void *cbInfo);

// Function Definitions

static void globalControlInit(void) {
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.tracingEnabled = 0;
    globalControl.profileMode = 0;
}

#ifdef _WIN32
typedef void(WINAPI *rtlExitUserProcess_t)(uint32_t exitCode);
rtlExitUserProcess_t Real_RtlExitUserProcess = NULL;

// Detour_RtlExitUserProcess
void WINAPI Detour_RtlExitUserProcess(uint32_t exitCode) {
    atExitHandler();

    Real_RtlExitUserProcess(exitCode);
}
#endif

void registerAtExitHandler(void) {
#ifdef _WIN32
    {
        // It's unsafe to use atexit(), static destructors, DllMain PROCESS_DETACH, etc.
        // because there's no way to guarantee the CUDA driver is still in a valid state
        // when you get to those, due to the undefined order of dynamic library tear-down
        // during process destruction.
        // Also, the first thing the Windows kernel does when any thread in a process
        // calls exit() is to immediately terminate all other threads, without any kind of
        // synchronization.
        // So the only valid time to do any in-process cleanup at exit() is before control
        // is passed to the kernel. Use Detours to intercept a low-level ntdll.dll
        // function "RtlExitUserProcess".
        int detourStatus = 0;
        FARPROC proc;

        // ntdll.dll will always be loaded, no need to load the library
        HMODULE ntDll = GetModuleHandle(TEXT("ntdll.dll"));
        if (!ntDll) {
            detourStatus = 1;
            goto DetourError;
        }

        proc = GetProcAddress(ntDll, "RtlExitUserProcess");
        if (!proc) {
            detourStatus = 1;
            goto DetourError;
        }
        Real_RtlExitUserProcess = (rtlExitUserProcess_t)proc;

        // Begin a detour transaction
        if (DetourTransactionBegin() != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        if (DetourUpdateThread(GetCurrentThread()) != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        DetourSetIgnoreTooSmall(TRUE);

        if (DetourAttach((void **)&Real_RtlExitUserProcess,
                         (void *)Detour_RtlExitUserProcess) != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        // Commit the transaction
        if (DetourTransactionCommit() != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }
    DetourError:
        if (detourStatus != 0) {
            atexit(&atExitHandler);
        }
    }
#else
    atexit(&atExitHandler);
#endif
}

static void atExitHandler(void) {
    CUPTI_CALL(cuptiGetLastError());

    // Force flush
    if (globalControl.tracingEnabled) {
        CUPTI_CALL(cuptiActivityFlushAll(1));
    }
}

static CUptiResult unsubscribeAllCallbacks(void) {
    if (globalControl.subscriber) {
        CUPTI_CALL(cuptiEnableAllDomains(0, globalControl.subscriber));
        CUPTI_CALL(cuptiUnsubscribe(globalControl.subscriber));
        globalControl.subscriber = NULL;
    }
    return CUPTI_SUCCESS;
}

static const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
    switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
        return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
        return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
        return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
        return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
        return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        return "HtoH";
    default:
        break;
    }

    return "<unknown>";
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                     size_t *maxNumRecords) {
    uint8_t *rawBuffer;

    *size = BUF_SIZE;
    rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL) {
        printf("Error: Out of memory.\n");
        exit(-1);
    }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                                     size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;
    size_t dropped;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            CUpti_ActivityKind kind = record->kind;

            switch (kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                const char *kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL)
                                             ? "KERNEL"
                                             : "CONC KERNEL";
                CUpti_ActivityKernel6 *kernel = (CUpti_ActivityKernel6 *)record;
                printf("%s \"%s\"  device %u, context %u, stream %u, correlation %u\n",
                       kindString, kernel->name, kernel->deviceId, kernel->contextId,
                       kernel->streamId, kernel->correlationId);
                printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static "
                       "%u, dynamic %u)\n",
                       kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX,
                       kernel->blockY, kernel->blockZ, kernel->staticSharedMemory,
                       kernel->dynamicSharedMemory);
                break;
            }
            case CUPTI_ACTIVITY_KIND_DRIVER: {
                CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
                printf("DRIVER cbid=%u  process %u, thread %u, correlation %u\n",
                       api->cbid, api->processId, api->threadId, api->correlationId);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy5 *memcpy = (CUpti_ActivityMemcpy5 *)record;
                printf("MEMCPY %s device %u, context %u, stream %u, size %llu, "
                       "correlation %u\n",
                       getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
                       memcpy->deviceId, memcpy->contextId, memcpy->streamId,
                       (unsigned long long)memcpy->bytes, memcpy->correlationId);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                CUpti_ActivityMemset4 *memset = (CUpti_ActivityMemset4 *)record;
                printf("MEMSET value=%u  device %u, context %u, stream %u, correlation %u\n",
                      memset->value, memset->deviceId, memset->contextId, memset->streamId,
                      memset->correlationId);
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
                printf("RUNTIME cbid=%u process %u, thread %u, correlation %u\n",
                       api->cbid, api->processId, api->threadId, api->correlationId);
                break;
            }
            default:
                break;
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            CUPTI_CALL(status);
        }
    } while (1);

    // Report any records dropped from the queue
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
        printf("Dropped %u activity records.\n", (unsigned int)dropped);
    }
    free(buffer);
}

static CUptiResult selectActivities() {
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_DRIVER);
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_RUNTIME);
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_MEMSET);
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_MEMCPY);

    return CUPTI_SUCCESS;
}

static CUptiResult enableCuptiActivity(CUcontext ctx) {
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));
    CUPTI_CALL(selectActivities());

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i) {
        if (IS_ACTIVITY_SELECTED(globalControl.profileMode, i)) {
            // If context is NULL activities are being enabled after CUDA initialization
            // else the activities are being enabled on cudaProfilerStart API
            if (ctx == NULL) {
                CUPTI_CALL(cuptiActivityEnable((CUpti_ActivityKind)i));
            } else {
                // Since some activities are not supported at context mode, enable them in
                // global mode if context mode fails
                result = cuptiActivityEnableContext(ctx, (CUpti_ActivityKind)i);

                if (result == CUPTI_ERROR_INVALID_KIND) {
                    cuptiGetLastError();
                    result = cuptiActivityEnable((CUpti_ActivityKind)i);
                } else if (result != CUPTI_SUCCESS) {
                    CUPTI_CALL(result);
                }
            }
        }
    }

    return result;
}

static CUptiResult cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber,
                              (CUpti_CallbackFunc)callbackHandler, NULL));

    // Subscribe Driver  callback to call onProfilerStartstop function
    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuProfilerStart));
    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuProfilerStop));

    // Enable CUPTI activities
    CUPTI_CALL(enableCuptiActivity(NULL));

    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    return status;
}

static CUptiResult onCudaDeviceReset(void) {
    // Flush all queues
    CUPTI_CALL(cuptiActivityFlushAll(0));

    return CUPTI_SUCCESS;
}

static CUptiResult onProfilerStart(CUcontext context) {
    if (context == NULL) {
        // Don't do anything if context is NULL
        return CUPTI_SUCCESS;
    }

    CUPTI_CALL(enableCuptiActivity(context));

    return CUPTI_SUCCESS;
}

static CUptiResult disableCuptiActivity(CUcontext ctx) {
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiEnableCallback(0, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i) {
        if (IS_ACTIVITY_SELECTED(globalControl.profileMode, i)) {
            // Since some activities are not supported at context mode, disable them in
            // global mode if context mode fails
            result = cuptiActivityDisableContext(ctx, (CUpti_ActivityKind)i);

            if (result == CUPTI_ERROR_INVALID_KIND) {
                cuptiGetLastError();
                CUPTI_CALL(cuptiActivityDisable((CUpti_ActivityKind)i));
            } else if (result != CUPTI_SUCCESS) {
                CUPTI_CALL(result);
            }
        }
    }

    return CUPTI_SUCCESS;
}

static CUptiResult onProfilerStop(CUcontext context) {
    if (context == NULL) {
        // Don't do anything if context is NULL
        return CUPTI_SUCCESS;
    }

    CUPTI_CALL(cuptiActivityFlushAll(0));
    CUPTI_CALL(disableCuptiActivity(context));

    return CUPTI_SUCCESS;
}

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, void *cbdata) {
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;

    // Check last error
    CUPTI_CALL(cuptiGetLastError());

    switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API: {
        switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuProfilerStart: {
            /* We start profiling collection on exit of the API. */
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                onProfilerStart(cbInfo->context);
            }
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuProfilerStop: {
            /* We stop profiling collection on entry of the API. */
            if (cbInfo->callbackSite == CUPTI_API_ENTER) {
                onProfilerStop(cbInfo->context);
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    case CUPTI_CB_DOMAIN_RUNTIME_API: {
        switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020: {
            if (cbInfo->callbackSite == CUPTI_API_ENTER) {
                CUPTI_CALL(onCudaDeviceReset());
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

#ifdef _WIN32
extern "C" __declspec(dllexport) int InitializeInjection(void)
#else
extern "C" int InitializeInjection(void)
#endif
{
    if (globalControl.initialized) {
        return 1;
    }

    initializeInjectionMutex.lock();

    // Init globalControl
    globalControlInit();

    registerAtExitHandler();

    // Initialize CUPTI
    if (cuptiInitialize() != CUPTI_SUCCESS) {
        printf("Error: Cupti Initilization failed.\n");
        unsubscribeAllCallbacks();
        exit(EXIT_FAILURE);
    }
    globalControl.tracingEnabled = 1;
    globalControl.initialized = 1;
    initializeInjectionMutex.unlock();
    return 1;
}
