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
#include <cuda_runtime.h>
#include <cupti.h>

// Standard NVTX headers
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

#define DRIVER_API_CALL(apiFuncCall)                                            \
do {                                                                            \
    CUresult _status = apiFuncCall;                                             \
    if (_status != CUDA_SUCCESS) {                                              \
        const char* errstr;                                                     \
        cuGetErrorString(_status, &errstr);                                     \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #apiFuncCall, errstr);                      \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                           \
do {                                                                            \
    cudaError_t _status = apiFuncCall;                                          \
    if (_status != cudaSuccess) {                                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

extern void initTrace(void);
extern void finiTrace(void);

__global__ void VecAdd(const int* A, const int* B, int* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

static void initVec(int *vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = i;
}

void vectorAdd() {
    CUcontext context = 0;
    CUdevice device = 0;
    int N = 50000;
    size_t size = N * sizeof (int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *h_A = 0, *h_B = 0, *h_C = 0;
    int *d_A = 0, *d_B = 0, *d_C = 0;

    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    nvtxNameCuDeviceA(device, "CUDA Device 0");

    // Create domain "Vector Addition"
    nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x0000ff;

    // Push range "doPass" on domain "Vector Addition"
    eventAttrib.message.ascii = "vectorAdd";
    nvtxDomainRangePushEx(domain, &eventAttrib);


    // Push range "Allocate host memory" on default domain
    nvtxRangePushA("Allocate host memory");

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*) malloc(size);
    h_B = (int*) malloc(size);
    h_C = (int*) malloc(size);

    if (!h_A || !h_B || !h_C) {
        printf("Error: Out of memory\n");
        return;
    }

    // Pop range "Allocate host memory" on default domain
    nvtxRangePop();

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    nvtxNameCuContextA(context, "CUDA Context");

    // Push range "Allocate device memory" on domain "Vector Addition"
    eventAttrib.message.ascii = "Allocate device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**) &d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**) &d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**) &d_C, size));

    // Pop range on domain - Allocate device memory
    nvtxDomainRangePop(domain);

    // Register string "Memcpy operation"
    nvtxStringHandle_t string = nvtxDomainRegisterStringA(domain, "Memcpy operation");
    // Push range "Memcpy operation" on domain "Vector Addition"
    eventAttrib.message.registered = string;
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Push range "Memcpy operation" on domain "Vector Addition"
    nvtxDomainRangePop(domain);

    // Start range "Launch kernel" on domain "Vector Addition"
    eventAttrib.message.ascii = "Launch kernel";
    nvtxRangeId_t id = nvtxDomainRangeStartEx(domain, &eventAttrib);

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, N);
    DRIVER_API_CALL(cuCtxSynchronize());

    // End range "Launch kernel" on domain "Vector Addition"
    nvtxDomainRangeEnd(domain, id);

    eventAttrib.message.registered = string;
    // Push range "Memcpy operation" on domain "Vector Addition"
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Push range "Memcpy operation" on domain "Vector Addition"
    nvtxDomainRangePop(domain);

    // Push range "Free device memory" on domain "Vector Addition"
    eventAttrib.message.ascii = "Free device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Free device memory
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));

    // Push range "Free device memory" on domain "Vector Addition"
    nvtxDomainRangePop(domain);

    // Push range "Free host memory" on default domain
    nvtxRangePushA("Free host memory");

    // Free host memory
    if (h_A) {
        free(h_A);
    }
    if (h_B) {
        free(h_B);
    }
    if (h_C) {
        free(h_C);
    }

    // Pop range "Free host memory" on default domain
    nvtxRangePop();

    DRIVER_API_CALL(cuCtxSynchronize());
    DRIVER_API_CALL(cuCtxDestroy(context));

    // Pop range "vectorAdd" on domain "Vector Addition"
    nvtxDomainRangePop(domain);
}

int main(int argc, char *argv[]) {
    initTrace();

    DRIVER_API_CALL(cuInit(0));
    vectorAdd();

    finiTrace();

    exit(EXIT_SUCCESS);
}