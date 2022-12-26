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
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define DRIVER_API_CALL(apiFuncCall)                                                \
    do {                                                                            \
        CUresult _status = apiFuncCall;                                             \
        if (_status != CUDA_SUCCESS) {                                              \
            const char* errstr;                                                     \
            cuGetErrorString(_status, &errstr);                                     \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                    __FILE__, __LINE__, #apiFuncCall, errstr);                      \
            exit(EXIT_FAILURE);                                                      \
        }                                                                           \
    } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                               \
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

__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

static void
memoryAllocations()
{
    int nelem = 1048576;
    size_t size = nelem * sizeof(int);

    int *h_A, *h_B;
    int *d_A, *d_B;

    // Allocate memory
    RUNTIME_API_CALL(cudaMallocHost((void**)&h_A, size));
    RUNTIME_API_CALL(cudaHostAlloc((void**)&h_B, size, cudaHostAllocPortable));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMallocManaged((void**)&d_B, size, cudaMemAttachGlobal));

    // Free the allocated memory
    RUNTIME_API_CALL(cudaFreeHost(h_A));
    RUNTIME_API_CALL(cudaFreeHost(h_B));
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
}

static void
memoryAllocationsViaMemoryPool()
{
    int nelem = 1048576;
    size_t bytes = nelem * sizeof(float);

    float *a, *b, *c;
    float *d_A, *d_B, *d_C;
    cudaStream_t stream;

    int isMemPoolSupported = 0;
    cudaError_t status = cudaSuccess;
    status = cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, 0);
    // For enhance compatibility cases, the attribute cudaDevAttrMemoryPoolsSupported might not be present
    // return early if Runtime API does not return cudaSuccess
    if (!isMemPoolSupported || status != cudaSuccess) {
        printf("Warning: Waiving execution of memory operations via memory pool as device does not support memory pools.\n");
        return;
    }

    // Allocate and initialize memory on host and device
    a = (float*) malloc(bytes);
    b = (float*) malloc(bytes);
    c = (float*) malloc(bytes);

    for (int n = 0; n < nelem; n++) {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    RUNTIME_API_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Allocate memory using default memory pool
    RUNTIME_API_CALL(cudaMallocAsync(&d_A, bytes, stream));
    RUNTIME_API_CALL(cudaMallocAsync(&d_B, bytes, stream));
    RUNTIME_API_CALL(cudaMallocAsync(&d_C, bytes, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_A, a, bytes, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_B, b, bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem/(float)block.x));
    vectorAddGPU <<< grid, block, 0, stream >>>(d_A, d_B, d_C, nelem);

    // Free the allocated memory
    RUNTIME_API_CALL(cudaFreeAsync(d_A, stream));
    RUNTIME_API_CALL(cudaFreeAsync(d_B, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(c, d_C, bytes, cudaMemcpyDeviceToHost, stream));
    RUNTIME_API_CALL(cudaFree(d_C));

    RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    RUNTIME_API_CALL(cudaStreamDestroy(stream));

    free(a);
    free(b);
    free(c);
}

int
main(int argc, char *argv[])
{
    // Initialize CUPTI
    initTrace();

    // Initialize CUDA
    DRIVER_API_CALL(cuInit(0));

    char deviceName[256];
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
    printf("Device Name: %s\n", deviceName);
    RUNTIME_API_CALL(cudaSetDevice(0));

    memoryAllocations();
    memoryAllocationsViaMemoryPool();

    // Flush CUPTI activity buffers
    finiTrace();

    exit(EXIT_SUCCESS);
}

