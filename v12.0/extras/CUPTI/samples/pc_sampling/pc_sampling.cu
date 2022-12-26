/*
 * Copyright 2014-2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of pc sampling.
 * This app will work on devices with compute capability 5.2
 * or 6.0 and higher.
 */
#include <cuda.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                      \
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

#define ARRAY_SIZE 32
#define THREADS_PER_BLOCK 32

cudaDeviceProp prop;

__global__ void
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

static const char *
getStallReasonString(CUpti_ActivityPCSamplingStallReason reason)
{
    switch (reason) {
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID:
        return "Invalid";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
        return "Selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
        return "Instruction fetch";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
        return "Execution dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
        return "Memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
        return "Texture";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
        return "Sync";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
        return "Constant memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
        return "Pipe busy";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
        return "Memory throttle";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
        return "Not selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
        return "Other";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING:
        return "Sleeping";
    default:
        break;
    }

    return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
        {
            CUpti_ActivitySourceLocator *sourceLocator = (CUpti_ActivitySourceLocator *)record;
            printf("Source Locator Id %d, File %s Line %d\n", sourceLocator->id, sourceLocator->fileName, sourceLocator->lineNumber);
            break;
        }
        case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
        {
            CUpti_ActivityPCSampling3 *psRecord = (CUpti_ActivityPCSampling3 *)record;

            printf("source %u, functionId %u, pc 0x%llx, corr %u, samples %u",
                  psRecord->sourceLocatorId,
                  psRecord->functionId,
                  (unsigned long long)psRecord->pcOffset,
                  psRecord->correlationId,
                  psRecord->samples
                  );

            // latencySamples Field is valid for devices with compute capability 6.0 and higher.
            if (prop.major >= 6)
            {
                printf(", latency samples %u", psRecord->latencySamples);
            }

            printf(", stallreason %s\n", getStallReasonString(psRecord->stallReason));
            break;
        }
        case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
        {
            CUpti_ActivityPCSamplingRecordInfo *pcsriResult =
                                (CUpti_ActivityPCSamplingRecordInfo *)(void *)record;

            printf("corr %u, totalSamples %llu, droppedSamples %llu, samplingPeriodInCycles %llu\n",
                  pcsriResult->correlationId,
                  (unsigned long long)pcsriResult->totalSamples,
                  (unsigned long long)pcsriResult->droppedSamples,
                  (unsigned long long)pcsriResult->samplingPeriodInCycles);
            break;
        }
        case CUPTI_ACTIVITY_KIND_FUNCTION:
        {
            CUpti_ActivityFunction *fResult =
                (CUpti_ActivityFunction *)record;

            printf("id %u, ctx %u, moduleId %u, functionIndex %u, name %s\n",
                fResult->id,
                fResult->contextId,
                fResult->moduleId,
                fResult->functionIndex,
                fResult->name);
            break;
        }
        default:
            printf("unknown\n");
            break;
    }
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    *size = BUF_SIZE + ALIGN_SIZE;
    *buffer = (uint8_t*) calloc(1, *size);
    *maxNumRecords = 0;
    if (*buffer == NULL) {
        printf("Error: out of memory\n");
        exit(EXIT_FAILURE);
    }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if(status == CUPTI_SUCCESS) {
            printActivity(record);
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        }
        else {
            CUPTI_CALL(status);
        }
    } while (1);

    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
        printf("Dropped %u activity records\n", (unsigned int)dropped);
    }

    free(buffer);
}

void
initTrace()
{
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void
finiTrace()
{
    CUPTI_CALL(cuptiActivityFlushAll(0));
}

static void
do_pass(cudaStream_t stream)
{
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = ARRAY_SIZE * sizeof(int);
    int blocksPerGrid = 0;
    CUpti_ActivityPCSamplingConfig configPC;
    CUcontext cuCtx;

    // Allocate input vectors h_A and h_B in host memory
    // don't bother to initialize
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

    configPC.size = sizeof(CUpti_ActivityPCSamplingConfig);
    configPC.samplingPeriod=CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
    configPC.samplingPeriod2 = 0;
    cuCtxGetCurrent(&cuCtx);
    // configure api needs to be called before activity enable for chips till Pascal
    // and for Volta+ order does not matter
    CUPTI_CALL(cuptiActivityConfigurePCSampling(cuCtx, &configPC));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

    RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    blocksPerGrid = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    VecAdd<<<blocksPerGrid, THREADS_PER_BLOCK, 0, stream>>>(d_A, d_B, d_C, ARRAY_SIZE);

    RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

    if (stream == 0)
        RUNTIME_API_CALL(cudaDeviceSynchronize());
    else
        RUNTIME_API_CALL(cudaStreamSynchronize(stream));

    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
}

int
main(int argc, char *argv[])
{
    int deviceNum = 0;

    // initialize the activity trace
    initTrace();

    RUNTIME_API_CALL(cudaGetDevice(&deviceNum));
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    printf("Device Name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);
    if (!((prop.major > 5) || ((prop.major == 5) && (prop.minor == 2)))) {
        printf("sample is waived on this device, pc sampling is supported on devices with compute capability 5.2 or 6.0 and higher\n");
        exit(EXIT_WAIVED);
    }

    // do pass default stream
    do_pass(0);

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    finiTrace();
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    RUNTIME_API_CALL(cudaDeviceReset());
    exit(EXIT_SUCCESS);
}
