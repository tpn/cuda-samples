/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample to show how to correlate CUDA APIs with the corresponding GPU
 * activities using the correlation-id field in the activity records.
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <stdlib.h>
using namespace std;

#define CUPTI_CALL(call)                                                        \
    do                                                                          \
    {                                                                           \
        CUptiResult _status = call;                                             \
        if (_status != CUPTI_SUCCESS)                                           \
        {                                                                       \
            const char* errstr;                                                 \
            cuptiGetResultString(_status, &errstr);                             \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",\
                    __FILE__, __LINE__, #call, errstr);                         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

// Map to store correlation id and its corresponding activity record
static std::map<uint32_t, CUpti_Activity*> correlationMap;
static std::map<uint32_t, CUpti_Activity*> connectionMap;

// Iterator to traverse the map
static std::map<uint32_t, CUpti_Activity*>::iterator iter;

static const char*
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
    switch (kind)
    {
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

// Store the runtime and driver API records in connectionMap and others in correlationMap
static void
storeActivity(CUpti_Activity* record)
{
    switch (record->kind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy5* memcpy = (CUpti_ActivityMemcpy5*)record;
            correlationMap[memcpy->correlationId] = record;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            CUpti_ActivityMemset4* memset = (CUpti_ActivityMemset4*)record;
            correlationMap[memset->correlationId] = record;
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)record;
            correlationMap[kernel->correlationId] = record;
            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER: {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*)record;
            connectionMap[api->correlationId] = record;
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*)record;
            connectionMap[api->correlationId] = record;
            break;
        }
        default:
            break;
    }
}
static void
printApiActivity(CUpti_ActivityAPI* api)
{
    // Print runtime or driver activity records
    if (api->kind == CUPTI_ACTIVITY_KIND_DRIVER)
    {
        printf(" DRIVER cbid=%u process %u, thread %u, correlation %u\n\n",
                api->cbid, api->processId, api->threadId, api->correlationId);
    }
    else
    {
        printf("RUNTIME cbid=%u process %u, thread %u, correlation %u\n\n",
                 api->cbid, api->processId, api->threadId, api->correlationId);
    }
}
static void
printActivity()
{
    iter = correlationMap.begin();
    // Iterate over the map using Iterator till end.
    while (iter != correlationMap.end())
    {
        // iter->first is correlation id and iter->second is activity record
        switch (iter->second->kind)
        {
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy5* memcpy = (CUpti_ActivityMemcpy5*)iter->second;

                // Check whether memcpy correlation id is present in connection map
                if (connectionMap.find(memcpy->correlationId) != connectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", memcpy->correlationId);
                    printf("MEMCPY %s device %u, context %u, stream %u, size %llu, correlation %u\n",
                             getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind), memcpy->deviceId,
                             memcpy->contextId, memcpy->streamId, (unsigned long long)memcpy->bytes, memcpy->correlationId);
                    CUpti_ActivityAPI* api = (CUpti_ActivityAPI*)connectionMap[memcpy->correlationId];
                    if(api != NULL)
                    {
                        printApiActivity(api);
                    }
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                CUpti_ActivityMemset4* memset = (CUpti_ActivityMemset4*)iter->second;

                // Check whether memset correlation id is present in connection map
                if (connectionMap.find(memset->correlationId) != connectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", memset->correlationId);
                    printf("MEMSET value=%u  device %u, context %u, stream %u, correlation %u\n",
                             memset->value, memset->deviceId, memset->contextId, memset->streamId,
                             memset->correlationId);
                    CUpti_ActivityAPI* api = (CUpti_ActivityAPI*)connectionMap[memset->correlationId];
                    if(api != NULL)
                    {
                        printApiActivity(api);
                    }
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                const char* kindString = (iter->second->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
                CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)iter->second;

                // Check whether kernel correlation id is present in connection map
                if (connectionMap.find(kernel->correlationId) != connectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", kernel->correlationId);
                    printf("%s \"%s\" device %u, context %u, stream %u, correlation %u\n",
                            kindString, kernel->name, kernel->deviceId, kernel->contextId, kernel->streamId,
                            kernel->correlationId);
                    printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
                             kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX, kernel->blockY, kernel->blockZ,
                             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
                    CUpti_ActivityAPI* api = (CUpti_ActivityAPI*)connectionMap[kernel->correlationId];
                    if(api != NULL)
                    {
                        printApiActivity(api);
                    }

                }
                break;
            }
            default:
                printf("  <unknown>\n");
                break;
        }
        iter++;
    }
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    uint8_t* bfr = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL)
    {
        printf("Error: out of memory\n");
        exit(EXIT_FAILURE);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity* record = NULL;
    if (validSize > 0)
    {
        do
        {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS)
            {
                storeActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
            {
                break;
            }
            else
            {
                CUPTI_CALL(status);
            }
        } while (1);

        // print all the record
        printActivity();

        // Report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0)
        {
            printf("Dropped %u activity records\n", (unsigned int)dropped);
        }
    }
    free(buffer);
}

void initTrace()
{
    // Enable  activity record kinds.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void finiTrace()
{
    // Force flush any remaining activity buffers before termination of the application
    CUPTI_CALL(cuptiActivityFlushAll(1));
}
