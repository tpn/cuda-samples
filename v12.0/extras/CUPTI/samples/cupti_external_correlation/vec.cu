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

#include "cupti_external_correlation.h"

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
    uint64_t id = 0;

    DRIVER_API_CALL(cuDeviceGet(&device, 0));

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*) malloc(size);
    h_B = (int*) malloc(size);
    h_C = (int*) malloc(size);

    if (!h_A || !h_B || !h_C) {
        printf("Error: Out of memory\n");
        return;
    }

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);

    // Push external id for the initialization: memory allocation and memcpy operations from host to device
    CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(INITIALIZATION_EXTERNAL_ID)));

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**) &d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**) &d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**) &d_C, size));

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Pop the external id
    CUPTI_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

    // Push external id for the vector addition and copy of results from device to host.
    CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(EXECUTION_EXTERNAL_ID)));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, N);
    DRIVER_API_CALL(cuCtxSynchronize());

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Pop the external id
    CUPTI_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

    // Push external id for the cleanup phase in the code
    CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(CLEANUP_EXTERNAL_ID)));

    // Free device memory
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));

    // Pop the external id
    CUPTI_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

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

    DRIVER_API_CALL(cuCtxSynchronize());
    DRIVER_API_CALL(cuCtxDestroy(context));
}

int main(int argc, char *argv[]) {
    initTrace();

    DRIVER_API_CALL(cuInit(0));
    vectorAdd();

    finiTrace();

    exit(EXIT_SUCCESS);
}