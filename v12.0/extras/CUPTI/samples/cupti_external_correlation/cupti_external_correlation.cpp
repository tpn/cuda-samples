/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to show how to do external correlation.
 * The sample pushes external ids in a simple vector addition
 * application showing how you can externally correlate different
 * phases of the code. In this sample it is broken into
 * initialization, execution and cleanup showing how you can
 * correlate all the APIs invloved in these 3 phases in the app.
 *
 * Psuedo code:
 * cuptiActivityPushExternalCorrelationId()
 * ExternalAPI() -> (runs bunch of CUDA APIs/ launches activity on GPU)
 * cuptiActivityPopExternalCorrelationId()
 * All CUDA activity activities within this range will generate external correlation
 * record which then can be used to correlate it with the external API
 */

#include <map>
#include <vector>
#include <stdlib.h>

#include "cupti_external_correlation.h"

// Map to book-keep the correlation ids mapped to the external ids
static std::map<uint64_t, std::vector<uint32_t> > externalCorrelationMap;

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
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
                    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
                    {
                        CUpti_ActivityExternalCorrelation *corr = (CUpti_ActivityExternalCorrelation *)record;
                        printf("\nEXTERNAL_CORRELATION: correlation %u, external %llu\n",
                                corr->correlationId,
                                (unsigned long long) corr->externalId);

                        //  Map the correlation ids to external ids for correlation
                        auto externalMapIter = externalCorrelationMap.find(corr->externalId);
                        if (externalMapIter == externalCorrelationMap.end()) {
                            std::vector<uint32_t> correlationVector;
                            correlationVector.push_back((uint32_t)corr->correlationId);
                            externalCorrelationMap.insert({(uint64_t)corr->externalId, correlationVector });
                        } else {
                            externalCorrelationMap[(uint64_t)corr->externalId].push_back((uint32_t)corr->correlationId);
                        }

                        break;
                    }
                    case CUPTI_ACTIVITY_KIND_RUNTIME:
                    {
                        CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
                        const char* name = NULL;
                        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &name);
                        printf("RUNTIME API %s: cbid %u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                                name,
                                api->cbid,
                                (unsigned long long) (api->start),
                                (unsigned long long) (api->end),
                                api->processId, api->threadId, api->correlationId);
                        break;
                    }
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                    {
                        CUpti_ActivityKernel9 *kernel = (CUpti_ActivityKernel9 *)record;
                        printf("CONCURRENT_KERNEL %s: [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
                                kernel->name,
                                (unsigned long long) (kernel->start),
                                (unsigned long long) (kernel->end),
                                kernel->deviceId, kernel->contextId, kernel->streamId,
                                kernel->correlationId);
                        break;
                    }
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                    {
                        CUpti_ActivityMemcpy5 *memcpy = (CUpti_ActivityMemcpy5 *) record;
                        printf("MEMCPY %s: [ %llu - %llu ] device %u, context %u, stream %u, size %llu, correlation %u\n",
                                getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
                                (unsigned long long) (memcpy->start),
                                (unsigned long long) (memcpy->end),
                                memcpy->deviceId, memcpy->contextId, memcpy->streamId,
                                (unsigned long long)memcpy->bytes, memcpy->correlationId);
                        break;
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

void initTrace() {
    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    // Enable CUDA runtime activity kinds for CUPTI to provide correlation ids
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    // Enable external correlation activtiy kind
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    // Enable activity kinds to trace GPU activities kernel and memcpy
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
}

void finiTrace() {
    // Force flush any remaining activity buffers before termination of the application
    CUPTI_CALL(cuptiActivityFlushAll(1));

    // Print the summary of extenal ids mapping with the correaltion ids
    printf ("\n=== SUMMARY ===");
    for(auto externalIdIter : externalCorrelationMap) {
        printf("\nExternal Id: %llu: ", (unsigned long long)externalIdIter.first);
        if (externalIdIter.first == INITIALIZATION_EXTERNAL_ID) {
            printf("INITIALIZATION_EXTERNAL_ID");
        }
        else if (externalIdIter.first == EXECUTION_EXTERNAL_ID) {
            printf("EXECUTION_EXTERNAL_ID");
        }
        else if (externalIdIter.first == CLEANUP_EXTERNAL_ID) {
            printf("CLEANUP_EXTERNAL_ID");
        }

        printf("\n  Correlate to CUPTI records with correlation ids: ");
        auto correlationIter = externalIdIter.second;
        for (auto correlationId : correlationIter) {
            printf("%u ", correlationId);
        }
        printf("\n");
    }
}