// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates a very simple use case of the Checkpoint API -
// An array is saved to device, a checkpoint is saved capturing these initial
// values, the device memory is update with a new value, then restored to
// initial value using the previously saved checkpoint.  By validating that
// the device values return the initial values, this demonstrates that the
// checkpoint API worked as expected.

#include <cuda.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

#include <cupti_checkpoint.h>
using namespace NV::Cupti::Checkpoint;

#define CHECKPOINT_API_CALL(apiFuncCall)                                       \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
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

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Basic example of a kernel which may overwrite its own input data
// This is not intended to show how to write a well-designed reduction,
// but to demonstrate that a kernel which modifies its input data can be
// replayed using the checkpoint API and get reproducible results.
//
// Sums n values, returning the total sum in data[0]
__global__ void reduce(float * data, size_t n)
{
    float thd_sum = 0.0;

    // Each thread sums its elements locally
    for (int i = threadIdx.x; i < n; i+= blockDim.x)
    {
        thd_sum += data[i];
    }

    // And saves the per-thread sum back to the thread's first element
    data[threadIdx.x] = thd_sum;

    __syncthreads();

    // Then, thread 0 reduces those per-thread sums to a single value in data[0]
    if (threadIdx.x == 0)
    {
        float total_sum = 0.0;

        size_t set_elems = (blockDim.x < n ? blockDim.x : n);

        for (int i = 0; i < set_elems; i++)
        {
            total_sum += data[i];
        }

        data[0] = total_sum;
    }
}

int main()
{
    CUcontext ctx;

    // Set up a context for device 0
    RUNTIME_API_CALL(cudaSetDevice(0));
    DRIVER_API_CALL(cuCtxCreate(&ctx, 0, 0));

    // Allocate host and device arrays and initialize to known values
    float * d_A;
    size_t el_A = 1024 * 1024;
    size_t sz_A = el_A * sizeof(float);
    RUNTIME_API_CALL(cudaMalloc(&d_A, sz_A));
    MEMORY_ALLOCATION_CALL(d_A);
    float * h_A = (float *)malloc(sz_A);
    MEMORY_ALLOCATION_CALL(h_A);
    for (size_t i = 0; i < el_A; i++)
    {
        h_A[i] = 1.0;
    }
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice));

    cout << "Initially, d_A[0] = " << h_A[0] << endl;

    // Demonstrate a case where calling a kernel repeatedly may cause incorrect
    // behavior due to internally modifying its input data
    cout << "Without checkpoint:" << endl;
    for (int repeat = 0; repeat < 3; repeat++)
    {
        reduce<<<1, 64>>>(d_A, el_A);

        // Test return value - should change each iteration due to not resetting input array
        float ret;
        RUNTIME_API_CALL(cudaMemcpy(&ret, d_A, sizeof(float), cudaMemcpyDeviceToHost));
        cout << "After " << (repeat + 1) << " iteration" << (repeat > 0 ? "s" : "") << ", d_A[0] = " << ret << endl;
    }

    // Re-initialize input array
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice));
    cout << "Reset device array - d_A[0] = " << h_A[0] << endl;

    // Configure a checkpoint object
    CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
    cp.ctx = ctx;
    cp.optimizations = 1;

    float expected;

    cout << "With checkpoint:" << endl;
    for (int repeat = 0; repeat < 3; repeat++)
    {
        // Save or restore the checkpoint as needed
        if (repeat == 0)
        {
            CHECKPOINT_API_CALL(cuptiCheckpointSave(&cp));
        }
        else
        {
            CHECKPOINT_API_CALL(cuptiCheckpointRestore(&cp));
        }

        // Call reduction kernel that modifies its own input
        reduce<<<1, 64>>>(d_A, el_A);

        // Check the output value (d_A[0])
        float ret;
        RUNTIME_API_CALL(cudaMemcpy(&ret, d_A, sizeof(float), cudaMemcpyDeviceToHost));

        // The first call to the kernel produces the expected result - with checkpoint, every subsequent call should also return this
        if (repeat == 0)
        {
            expected = ret;
        }

        cout << "After " << (repeat + 1) << " iteration" << (repeat > 0 ? "s" : "") << ", d_A[0] = " << ret << endl;

        // Verify that this iteration's output value matches the expected value from the first iteration
        if (ret != expected)
        {
            cerr << "Error - repeat " << repeat << " did not match expected value (" << ret << " != " << expected << "), did checkpoint not restore input data correctly?" << endl;
            CHECKPOINT_API_CALL(cuptiCheckpointFree(&cp));
            exit(EXIT_FAILURE);
        }
    }

    // Clean up
    CHECKPOINT_API_CALL(cuptiCheckpointFree(&cp));

    exit(EXIT_SUCCESS);
}

