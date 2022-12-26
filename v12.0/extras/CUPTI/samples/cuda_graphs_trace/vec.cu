/*
 * Copyright 2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print the trace of CUDA graphs and correlate
 * the graph node launch to the node creation API using CUPTI callbacks.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        const char* errstr;                                                    \
        cuGetErrorString(_status, &errstr);                                    \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define COMPUTE_N 50000

extern void initTrace(void);
extern void finiTrace(void);

// Kernels
__global__ void
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

__global__ void
VecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
}

static void
do_pass(cudaStream_t stream)
{
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;
  cudaKernelNodeParams kernelParams;
  cudaMemcpy3DParms memcpyParams = {0};
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  cudaGraphNode_t nodes[5];

  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  if (!h_A || !h_B || !h_C) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

  RUNTIME_API_CALL(cudaGraphCreate(&graph, 0));

  // Init memcpy params
  memcpyParams.kind = cudaMemcpyHostToDevice;
  memcpyParams.srcPtr.ptr = h_A;
  memcpyParams.dstPtr.ptr = d_A;
  memcpyParams.extent.width = size;
  memcpyParams.extent.height = 1;
  memcpyParams.extent.depth = 1;
  RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[0], graph, NULL, 0, &memcpyParams));

  memcpyParams.srcPtr.ptr = h_B;
  memcpyParams.dstPtr.ptr = d_B;
  RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[1], graph, NULL, 0, &memcpyParams));

  // Init kernel params
  int num = COMPUTE_N;
  void* kernelArgs[] = {(void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&num};
  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
  kernelParams.func = (void *)VecAdd;
  kernelParams.gridDim = dim3(blocksPerGrid, 1, 1);
  kernelParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelParams.sharedMemBytes = 0;
  kernelParams.kernelParams = (void **)kernelArgs;
  kernelParams.extra = NULL;

  RUNTIME_API_CALL(cudaGraphAddKernelNode(&nodes[2], graph, &nodes[0], 2, &kernelParams));

  kernelParams.func = (void *)VecSub;
  RUNTIME_API_CALL(cudaGraphAddKernelNode(&nodes[3], graph, &nodes[2], 1, &kernelParams));

  memcpyParams.kind = cudaMemcpyDeviceToHost;
  memcpyParams.srcPtr.ptr = d_C;
  memcpyParams.dstPtr.ptr = h_C;
  memcpyParams.extent.width = size;
  memcpyParams.extent.height = 1;
  memcpyParams.extent.depth = 1;
  RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[4], graph, &nodes[3], 1, &memcpyParams));

  RUNTIME_API_CALL(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  RUNTIME_API_CALL(cudaGraphLaunch(graphExec, stream));
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
  CUdevice device;
  char deviceName[256];

  // initialize the activity trace
  initTrace();

  DRIVER_API_CALL(cuInit(0));

  DRIVER_API_CALL(cuDeviceGet(&device, 0));
  DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
  printf("Device Name: %s\n", deviceName);

  RUNTIME_API_CALL(cudaSetDevice(0));

  // do pass with user stream
  cudaStream_t stream0;
  RUNTIME_API_CALL(cudaStreamCreate(&stream0));
  do_pass(stream0);

  RUNTIME_API_CALL(cudaDeviceSynchronize());

  // Flush CUPTI buffers before resetting the device.
  // This can also be called in the cudaDeviceReset callback.
  CUPTI_CALL(cuptiActivityFlushAll(0));
  RUNTIME_API_CALL(cudaDeviceReset());

  finiTrace();
  exit(EXIT_SUCCESS);
}

