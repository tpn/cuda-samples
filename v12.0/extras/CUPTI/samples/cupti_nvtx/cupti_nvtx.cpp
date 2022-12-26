/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to output NVTX ranges.
 * The sample adds NVTX ranges around a simple vector addition app
 * NVTX functionality shown in the sample:
 *  Subscribe to NVTX callbacks and get NVTX records
 *  Create domain, add start/end and push/pop ranges w.r.t the domain
 *  Register string against a domain
 *  Naming of CUDA resources
 *
 * Before running the sample set the NVTX_INJECTION64_PATH
 * environment variable pointing to the CUPTI Library.
 * For Linux:
 *    export NVTX_INJECTION64_PATH=<full_path>/libcupti.so
 * For Windows:
 *    set NVTX_INJECTION64_PATH=<full_path>/cupti.dll
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include "cupti.h"

// Standard NVTX headers
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

// Includes definition of the callback structures to use for NVTX with CUPTI
#include "generated_nvtx_meta.h"

#define CUPTI_CALL(call)                                                         \
    do {                                                                         \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS) {                                          \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(EXIT_FAILURE);                                                            \
        }                                                                        \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

const char * getName(const char *name) {
    if (name == NULL) {
        return "<null>";
    }
    return name;
}

const char * getDomainName(const char *name) {
    if (name == NULL) {
        return "<default domain>";
    }
    return name;
}


const char * getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
    switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
        return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD:
        return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
        return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM:
        return "STREAM";
    default:
        break;
    }

    return "<unknown>";
}

uint32_t getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id) {
    switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
        return id->pt.processId;
    case CUPTI_ACTIVITY_OBJECT_THREAD:
        return id->pt.threadId;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
        return id->dcs.deviceId;
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return id->dcs.contextId;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
        return id->dcs.streamId;
    default:
        break;
    }

    return 0xffffffff;
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    uint8_t *b;

    *size = BUF_SIZE;
    b = (uint8_t *)malloc(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(b, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL) {
        printf("Error: out of memory\n");
        exit(EXIT_FAILURE);
    }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                switch(record->kind) {
                    case CUPTI_ACTIVITY_KIND_MARKER:
                    {
                        CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
                        printf("MARKER  id %u [ %llu ], name %s, domain %s\n",
                                marker->id, (unsigned long long) marker->timestamp, getName(marker->name), getDomainName(marker->domain));
                        break;
                    }
                    case CUPTI_ACTIVITY_KIND_NAME:
                    {
                        CUpti_ActivityName *name = (CUpti_ActivityName *) record;
                        switch (name->objectKind)
                        {
                            case CUPTI_ACTIVITY_OBJECT_CONTEXT:
                                printf("NAME %s id %u %s id %u, name %s\n",
                                    getActivityObjectKindString(name->objectKind),
                                    getActivityObjectKindId(name->objectKind, &name->objectId),
                                    getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
                                    getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                                    getName(name->name));
                                break;
                            case CUPTI_ACTIVITY_OBJECT_STREAM:
                                printf("NAME %s id %u %s id %u %s id %u, name %s\n",
                                    getActivityObjectKindString(name->objectKind),
                                    getActivityObjectKindId(name->objectKind, &name->objectId),
                                    getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
                                    getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
                                    getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
                                    getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                                    getName(name->name));
                                break;
                            default:
                                printf("NAME %s id %u, name %s\n",
                                    getActivityObjectKindString(name->objectKind),
                                    getActivityObjectKindId(name->objectKind, &name->objectId),
                                    getName(name->name));
                                break;
                        }
                    }
                    default:
                        break;
                }
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // Report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
    }
}

static void CUPTIAPI
nvtxCallback(void *userdata, CUpti_CallbackDomain domain,
             CUpti_CallbackId cbid, const void *cbdata)
{
    CUpti_NvtxData* data = (CUpti_NvtxData*)cbdata;

    switch (cbid) {
        case CUPTI_CBID_NVTX_nvtxDomainCreateA: {
            // Get the parameters passed to the NVTX function
            nvtxDomainCreateA_params* params = (nvtxDomainCreateA_params*)data->functionParams;
            // Get the return value of the NVTX function
            nvtxDomainHandle_t* domainHandle = (nvtxDomainHandle_t*)data->functionReturnValue;
            break;
        }
        case CUPTI_CBID_NVTX_nvtxMarkEx: {
            nvtxMarkEx_params* params = (nvtxMarkEx_params*)data->functionParams;
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx: {
            nvtxDomainMarkEx_params* params = (nvtxDomainMarkEx_params*)data->functionParams;
            break;
        }
        // Add more NVTX callbacks, refer "generated_nvtx_meta.h" for all NVTX callbacks
        // If there is no return value for the NVTX function, functionReturnValue is NULL.
        default:
            break;
    }

    return;
}

void initTrace() {
    CUpti_SubscriberHandle subscriber;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)nvtxCallback, NULL));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX));

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    // For NVTX markers (Marker, Domain, Start/End ranges, Push/Pop ranges, Registered Strings)
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    // For naming CUDA resources (Threads, Devices, Contexts, Streams)
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
}

void finiTrace() {
   // Force flush any remaining activity buffers before termination of the application
   CUPTI_CALL(cuptiActivityFlushAll(1));
}