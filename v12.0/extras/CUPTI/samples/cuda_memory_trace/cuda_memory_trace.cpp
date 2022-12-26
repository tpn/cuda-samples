/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print trace of CUDA memory operations.
 * The sample also traces CUDA memory operations done via
 * default memory pool.
 *
 */

#include <stdio.h>
#include <string.h>
#include <cupti.h>
#include <stdlib.h>

#define CUPTI_CALL(call)                                                          \
    do {                                                                          \
        CUptiResult _status = call;                                               \
        if (_status != CUPTI_SUCCESS) {                                           \
            const char *errstr;                                                   \
            cuptiGetResultString(_status, &errstr);                               \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
                    __FILE__, __LINE__, #call, errstr);                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                         \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                               \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getKindString(CUpti_ActivityKind kind)
{
    switch (kind) {
    case CUPTI_ACTIVITY_KIND_MEMORY2:
        return "MEMORY2";
    case CUPTI_ACTIVITY_KIND_MEMORY_POOL:
        return "MEMORY_POOL";
    default:
        return "<unknown>";
    }
}

static const char *
getMemoryOperationTypeString(CUpti_ActivityMemoryOperationType type)
{
    switch (type) {
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
            return "ALLOCATE";
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
            return "RELEASE";
        default:
            return "<unknown>";
    }
}

static const char *
getMemoryKindString(CUpti_ActivityMemoryKind kind)
{
    switch (kind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "PAGEABLE";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "PINNED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "DEVICE";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
            return "ARRAY";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
            return "MANAGED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
            return "DEVICE_STATIC";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
            return "MANAGED_STATIC";
        default:
            return "<unknown>";
    }
}

static const char *
getMemoryPoolTypeString(CUpti_ActivityMemoryPoolType type)
{
    switch (type) {
        case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL:
            return "LOCAL";
        case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED:
            return "IMPORTED";
        default:
            return "<unknown>";
    }
}

static const char *
getMemoryPoolOperationTypeString(CUpti_ActivityMemoryPoolOperationType type)
{
    switch (type) {
        case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED:
            return "MEM_POOL_CREATED";
        case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED:
            return "MEM_POOL_DESTROYED";
        case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED:
            return "MEM_POOL_TRIMMED";
        default:
            return "<unknown>";
    }
}

static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind)
    {
        case CUPTI_ACTIVITY_KIND_MEMORY2:
        {
            CUpti_ActivityMemory3 *memory = (CUpti_ActivityMemory3 *)(void *)record;
            printf("  %s [ %llu ] memoryOperationType %s, address %llu, size %llu, pc %llu, memoryKind %s, devId %u, ctxId %u, "
                   "procId %u, corrId %u, streamId %u, isAsync %u, memoryPoolType %s, memoryPoolAddress %llu, memoryPoolThreshold %llu",
                    getKindString(memory->kind),
                    (unsigned long long)memory->timestamp,
                    getMemoryOperationTypeString(memory->memoryOperationType),
                    (unsigned long long)memory->address,
                    (unsigned long long)memory->bytes,
                    (unsigned long long)memory->PC,
                    getMemoryKindString(memory->memoryKind),
                    memory->deviceId,
                    memory->contextId,
                    memory->processId,
                    memory->correlationId,
                    memory->streamId,
                    memory->isAsync,
                    getMemoryPoolTypeString(memory->memoryPoolConfig.memoryPoolType),
                    (unsigned long long)memory->memoryPoolConfig.address,
                    (unsigned long long)memory->memoryPoolConfig.releaseThreshold);

            if (memory->memoryPoolConfig.memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL) {
                printf(", memoryPoolSize: %llu, memoryPoolUtilizedSize: %llu",
                       (unsigned long long)memory->memoryPoolConfig.pool.size,
                       (unsigned long long)memory->memoryPoolConfig.utilizedSize);
            }
            else if (memory->memoryPoolConfig.memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED) {
                printf( ", memoryPoolProcessId: %llu", (unsigned long long)memory->memoryPoolConfig.pool.processId);
            }
            printf( "\n");

            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMORY_POOL:
        {
            CUpti_ActivityMemoryPool2 *memoryPool = (CUpti_ActivityMemoryPool2 *)(void *)record;
            printf("  %s [ %llu ] memoryPoolOperationType %s, memoryPoolType %s, address %llu, size %llu, utilizedSize %llu, "
                   "devId %u, procId %u, corrId %u, releaseThreshold %llu\n",
                    getKindString(memoryPool->kind),
                    (unsigned long long)memoryPool->timestamp,
                    getMemoryPoolOperationTypeString(memoryPool->memoryPoolOperationType),
                    getMemoryPoolTypeString(memoryPool->memoryPoolType),
                    (unsigned long long)memoryPool->address,
                    (unsigned long long)memoryPool->size,
                    (unsigned long long)memoryPool->utilizedSize,
                    memoryPool->deviceId,
                    memoryPool->processId,
                    memoryPool->correlationId,
                    (unsigned long long)memoryPool->releaseThreshold);
        }
        default:
            break;
    }
}

void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);

    if (bfr == NULL) {
        printf("Error: out of memory.\n");
        exit(EXIT_FAILURE);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                printActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Warning: Dropped %u activity records.\n", (unsigned int) dropped);
        }
    }

    free(buffer);
}

void
initTrace()
{
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY_POOL));

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void
finiTrace()
{
    // Force flush any remaining activity buffers before termination of the application
    CUPTI_CALL(cuptiActivityFlushAll(1));
}
