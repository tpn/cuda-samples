/*
 * Copyright 2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of pc sampling
 * with start stop APIs. This app will work on devices with compute
 * capability 7.0 and higher.
 */
#include <cuda.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#if defined(WIN32) || defined(_WIN32)
#define stricmp _stricmp
#else
#define stricmp strcasecmp
#endif

#define RUNTIME_API_CALL(apiFuncCall)                                           \
do {                                                                            \
    cudaError_t _status = apiFuncCall;                                          \
    if (_status != cudaSuccess) {                                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
        exit(EXIT_FAILURE);                                                      \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                            \
do {                                                                            \
    CUresult _status = apiFuncCall;                                             \
    if (_status != CUDA_SUCCESS) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",    \
                __FILE__, __LINE__, #apiFuncCall, _status);                     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define CUPTI_CALL(call)                                                        \
do {                                                                            \
    CUptiResult _status = call;                                                 \
    if (_status != CUPTI_SUCCESS) {                                             \
        const char *errstr;                                                     \
        cuptiGetResultString(_status, &errstr);                                 \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #call, errstr);                             \
        exit(EXIT_FAILURE);                                                      \
    }                                                                           \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define NUM_PC_COLLECT 100
#define ARRAY_SIZE 32000
#define THREADS_PER_BLOCK 256

typedef enum
{
    VECTOR_ADD  = 0,
    VECTOR_SUB  = 1,
    VECTOR_MUL  = 2,
} vectorOp;

// CUDA device kernels
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void VecSub(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] - B[i];
    }
}

__global__ void VecMul(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] * B[i];
    }
}

static void cleanUp(int *h_A, int *h_B, int *h_C, int *d_A, int *d_B, int *d_C)
{
    // Free device memory
    if (d_A)
        RUNTIME_API_CALL(cudaFree(d_A));
    if (d_B)
        RUNTIME_API_CALL(cudaFree(d_B));
    if (d_C)
        RUNTIME_API_CALL(cudaFree(d_C));

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
}

static void initVec(int *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = i;
    }
}

static void vectorOperation(const vectorOp op)
{
    int N = ARRAY_SIZE;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocksPerGrid = 0;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);
    int i, result = 0;

    CUcontext cuCtx;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(h_A);
    h_B = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(h_B);
    h_C = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(h_C);

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

    cuCtxGetCurrent(&cuCtx);

    RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice));

    blocksPerGrid = (N + threads_per_block - 1) / threads_per_block;

    if (op == VECTOR_ADD)
    {
        printf("Launching VecAdd\n");
        VecAdd<<<blocksPerGrid, threads_per_block>>>(d_A, d_B, d_C, N);
    }
    else if (op == VECTOR_SUB)
    {
        printf("Launching VecSub\n");
        VecSub<<<blocksPerGrid, threads_per_block>>>(d_A, d_B, d_C, N);
    }
    else if (op == VECTOR_MUL)
    {
        printf("Launching VecMul\n");
        VecMul<<<blocksPerGrid, threads_per_block>>>(d_A, d_B, d_C, N);
    }
    else
    {
        fprintf(stderr, "error: invalid operation\n");
        exit(EXIT_FAILURE);
    }

    RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost));
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Verify result
    for (i = 0; i < N; ++i)
    {
        if (op == VECTOR_ADD)
        {
            result = h_A[i] + h_B[i];
        }
        else if (op == VECTOR_SUB)
        {
            result = h_A[i] - h_B[i];
        }
        else if (op == VECTOR_MUL)
        {
            result = h_A[i] * h_B[i];
        }
        else
        {
            fprintf(stderr, "error: invalid operation\n");
            exit(EXIT_FAILURE);
        }

        if (h_C[i] != result)
        {
            fprintf(stderr, "error: result verification failed\n");
            exit(EXIT_FAILURE);
        }
    }
    cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
}

inline const char * getStallReason(const uint32_t& stallReasonCount, const uint32_t& pcSamplingStallReasonIndex, uint32_t *pStallReasonIndex, char **pStallReasons)
{
    for (uint32_t i = 0; i < stallReasonCount; i++)
    {
        if (pStallReasonIndex[i] == pcSamplingStallReasonIndex)
        {
            return pStallReasons[i];
        }
    }
    return "ERROR_STALL_REASON_INDEX_NOT_FOUND";
}

void printPCSamplingData(CUpti_PCSamplingData *pPcSamplingData, const uint32_t& stallReasonCount, uint32_t *pStallReasonIndex, char **pStallReasons)
{
    std::cout << "----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----" << std::endl;
    for (size_t i = 0; i < pPcSamplingData->totalNumPcs; i++)
    {
        std::cout << ", pcOffset : 0x"<< std::hex << pPcSamplingData->pPcData[i].pcOffset
                  << ", stallReasonCount: " << std::dec << pPcSamplingData->pPcData[i].stallReasonCount
                  << ", functionName: " << pPcSamplingData->pPcData[i].functionName;
        for (size_t j = 0; j < pPcSamplingData->pPcData[i].stallReasonCount; j++)
        {
            std::cout << ", stallReason: " << getStallReason(stallReasonCount, pPcSamplingData->pPcData[i].stallReason[j].pcSamplingStallReasonIndex, pStallReasonIndex, pStallReasons)
                      << ", samples: " << pPcSamplingData->pPcData[i].stallReason[j].samples;
        }
        std::cout << std::endl;
    }
    std::cout << "Number of PCs remaining to be collected: " << pPcSamplingData->remainingNumPcs << ", ";
    std::cout << "range id: " << pPcSamplingData->rangeId << ", ";
    std::cout << "total samples: " << pPcSamplingData->totalSamples << ", ";
    std::cout << "dropped samples: " << pPcSamplingData->droppedSamples << ", ";
    std::cout << "non user kernels total samples: " << pPcSamplingData->nonUsrKernelsTotalSamples << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;
}

bool run(const CUcontext& cuCtx, const size_t& stallReasonCount, uint32_t *pStallReasonIndex, char **pStallReasons)
{
    CUpti_PCSamplingStartParams pcSamplingStartParams = {};
    pcSamplingStartParams.size = CUpti_PCSamplingStartParamsSize;
    pcSamplingStartParams.ctx = cuCtx;

    CUpti_PCSamplingStopParams pcSamplingStopParams = {};
    pcSamplingStopParams.size = CUpti_PCSamplingStopParamsSize;
    pcSamplingStopParams.ctx = cuCtx;

    // On-demand user buffer to hold collected PC Sampling data in PC-To-Counter format
    CUpti_PCSamplingData pcSamplingData;
    pcSamplingData.size = sizeof(CUpti_PCSamplingData);
    pcSamplingData.collectNumPcs = NUM_PC_COLLECT;
    pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)calloc(pcSamplingData.collectNumPcs, sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData);
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)calloc(stallReasonCount, sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData[i].stallReason);
    }

    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
    pcSamplingGetDataParams.ctx = cuCtx;
    pcSamplingGetDataParams.pcSamplingData = (void *)&pcSamplingData;

    // Kernel outside PC Sampling data collection range
    vectorOperation(VECTOR_MUL);

    // Start PC Sampling
    std::cout << "----- PC sampling start -----" << std::endl;
    CUPTI_CALL(cuptiPCSamplingStart(&pcSamplingStartParams));
    vectorOperation(VECTOR_ADD);
    vectorOperation(VECTOR_SUB);

    // Stop PC Sampling
    CUPTI_CALL(cuptiPCSamplingStop(&pcSamplingStopParams));
    std::cout << "----- PC sampling stop -----" << std::endl;

    // Collect PC Sampling data
    CUPTI_CALL(cuptiPCSamplingGetData(&pcSamplingGetDataParams));
    printPCSamplingData(&pcSamplingData, stallReasonCount, pStallReasonIndex, pStallReasons);

    // Kernel outside PC Sampling data collection range
    vectorOperation(VECTOR_MUL);

    // Start PC Sampling
    std::cout << "----- PC sampling start -----" << std::endl;
    CUPTI_CALL(cuptiPCSamplingStart(&pcSamplingStartParams));
    vectorOperation(VECTOR_MUL);
    vectorOperation(VECTOR_ADD);
    vectorOperation(VECTOR_SUB);

    // Stop PC Sampling
    CUPTI_CALL(cuptiPCSamplingStop(&pcSamplingStopParams));
    std::cout << "----- PC sampling stop -----" << std::endl;

    // Collect PC Sampling data
    CUPTI_CALL(cuptiPCSamplingGetData(&pcSamplingGetDataParams));
    printPCSamplingData(&pcSamplingData, stallReasonCount, pStallReasonIndex, pStallReasons);

    // Kernel outside PC Sampling data collection range
    vectorOperation(VECTOR_MUL);

    // Free memory
    std::unordered_set<char*> functions;
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        if (pcSamplingData.pPcData[i].stallReason)
        {
            free(pcSamplingData.pPcData[i].stallReason);
        }
        
        if (pcSamplingData.pPcData[i].functionName)
        {
            functions.insert(pcSamplingData.pPcData[i].functionName);
        }
    }

    for(auto it = functions.begin(); it != functions.end(); ++it)
    {
        free(*it);
    }
    functions.clear();

    if (pcSamplingData.pPcData)
    {
        free(pcSamplingData.pPcData);
    }
    return true;
}

void doPCSampling(const CUcontext& cuCtx)
{
    // Enable PC Sampling
    CUpti_PCSamplingEnableParams pcSamplingEnableParams = {};
    pcSamplingEnableParams.size = CUpti_PCSamplingEnableParamsSize;
    pcSamplingEnableParams.ctx = cuCtx;
    CUPTI_CALL(cuptiPCSamplingEnable(&pcSamplingEnableParams));

    // Get number of supported counters
    size_t numStallReasons = 0;
    CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {};
    numStallReasonsParams.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    numStallReasonsParams.ctx = cuCtx;
    numStallReasonsParams.numStallReasons = &numStallReasons;
    CUPTI_CALL(cuptiPCSamplingGetNumStallReasons(&numStallReasonsParams));

    char **pStallReasons = (char **)calloc(numStallReasons, sizeof(char*));
    MEMORY_ALLOCATION_CALL(pStallReasons);
    for (size_t i = 0; i < numStallReasons; i++)
    {
        pStallReasons[i] = (char *)calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char));
        MEMORY_ALLOCATION_CALL(pStallReasons[i]);
    }
    uint32_t *pStallReasonIndex = (uint32_t *)calloc(numStallReasons, sizeof(uint32_t));
    MEMORY_ALLOCATION_CALL(pStallReasonIndex);

    CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {};
    stallReasonsParams.size = CUpti_PCSamplingGetStallReasonsParamsSize;
    stallReasonsParams.ctx = cuCtx;
    stallReasonsParams.numStallReasons = numStallReasons;
    stallReasonsParams.stallReasonIndex = pStallReasonIndex;
    stallReasonsParams.stallReasons = pStallReasons;
    CUPTI_CALL(cuptiPCSamplingGetStallReasons(&stallReasonsParams));

    // Buffer to hold collected PC Sampling data in PC-To-Counter format
    CUpti_PCSamplingData pcSamplingData;
    pcSamplingData.size = sizeof(CUpti_PCSamplingData);
    pcSamplingData.collectNumPcs = NUM_PC_COLLECT;
    pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)calloc(pcSamplingData.collectNumPcs, sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData);
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)calloc(numStallReasons, sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData[i].stallReason);
    }

    // PC Sampling configuration attributes
    CUpti_PCSamplingConfigurationInfo enableStartStop = {};
    enableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    enableStartStop.attributeData.enableStartStopControlData.enableStartStopControl = true;

    CUpti_PCSamplingConfigurationInfo samplingDataBuffer = {};
    samplingDataBuffer.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    samplingDataBuffer.attributeData.samplingDataBufferData.samplingDataBuffer = (void *)&pcSamplingData;

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
    pcSamplingConfigurationInfo.push_back(enableStartStop);
    pcSamplingConfigurationInfo.push_back(samplingDataBuffer);

    CUpti_PCSamplingConfigurationInfoParams pcSamplingConfigurationInfoParams = {};
    pcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    pcSamplingConfigurationInfoParams.ctx = cuCtx;
    pcSamplingConfigurationInfoParams.numAttributes = pcSamplingConfigurationInfo.size();
    pcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingConfigurationInfo.data();

    CUPTI_CALL(cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    if(!run(cuCtx, numStallReasons, pStallReasonIndex, pStallReasons))
    {
        std::cout << "Failed to run sample" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Disable PC Sampling
    CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
    pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
    pcSamplingDisableParams.ctx = cuCtx;
    CUPTI_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

    // Free memory
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        if (pcSamplingData.pPcData[i].stallReason)
        {
            free(pcSamplingData.pPcData[i].stallReason);
        }
    }
    if (pcSamplingData.pPcData)
    {
        free(pcSamplingData.pPcData);
    }

    for (size_t i = 0; i < numStallReasons; i++)
    {
        if (pStallReasons[i])
        {
            free(pStallReasons[i]);
        }
    }
    if (pStallReasons)
    {
        free(pStallReasons);
    }
}

int main(int argc, char *argv[])
{
    CUcontext cuCtx;
    cudaDeviceProp prop;
    int deviceNum = 0;

    RUNTIME_API_CALL(cudaGetDevice(&deviceNum));

    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    printf("Device Name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);

    // Initialize profiler API and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    CUPTI_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Sample is waived on this device, Unable to profile on device " << deviceNum << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    cuCtxGetCurrent(&cuCtx);
    DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, deviceNum));
    doPCSampling(cuCtx);
    DRIVER_API_CALL(cuCtxDestroy(cuCtx));

    exit(EXIT_SUCCESS);
}
