/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Includes, system
#include <stdio.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda_awbarrier.h>
#include <cooperative_groups.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

namespace nvcuda_namespace = nvcuda::experimental;
namespace cg = cooperative_groups;


#if __CUDA_ARCH__ >= 700
template <bool writeSquareRoot> __device__ void reduceBlockData(nvcuda_namespace::awbarrier &barrier,
                            cg::thread_block_tile<32> &tile32, double &threadSum, double *result)
{
    extern __shared__ double tmp[];

    #pragma unroll
    for (int offset = tile32.size()/2; offset > 0; offset /= 2)
    {
         threadSum += tile32.shfl_down(threadSum, offset);
    }
    if (tile32.thread_rank() == 0)
    {
        tmp[tile32.meta_group_rank()] = threadSum;
    }

    const auto token = barrier.arrive();

    // The warp which would arrive last at the barrier will 
    // perform last round of reduction
    if (tile32.any(token.pending_count() == 1)) {

        double beta  = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;

        #pragma unroll
        for (int offset = tile32.size()/2; offset > 0; offset /= 2)
        {
             beta += tile32.shfl_down(beta, offset);
        }

        if (tile32.thread_rank() == 0)
        {
            if (writeSquareRoot)
                *result = sqrt(beta);
            else
                *result = beta;
        }
    }

    barrier.wait(token);
}
#endif

__global__ void normVecByDotProductAWBarrier(float *vecA, float *vecB, double *partialResults, int size)
{
#if __CUDA_ARCH__ >= 700
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();;
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    __shared__ nvcuda_namespace::awbarrier barrier;

    if (threadIdx.x == 0) {
        nvcuda_namespace::init(&barrier, blockDim.x);
    }

    cg::sync(cta);

    double threadSum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size())
    {
        threadSum += (double) (vecA[i] * vecB[i]);
    }

    // Each thread block performs reduction of partial dotProducts and writes to 
    // global mem.
    reduceBlockData<false>(barrier, tile32, threadSum, &partialResults[blockIdx.x]);

    cg::sync(grid);

    // One block performs the final summation of partial dot products
    // of all the thread blocks and writes the sqrt of final dot product.
    if (blockIdx.x == 0)
    {
        threadSum = 0.0;
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size())
        {
            threadSum += partialResults[i];
        }
        reduceBlockData<true>(barrier, tile32, threadSum, &partialResults[0]);
    }

    cg::sync(grid);

    const double finalValue = partialResults[0];

    // Perform normalization of vecA & vecB.
    for (int i = grid.thread_rank(); i < size; i += grid.size())
    {
        vecA[i] = (float)vecA[i] / finalValue;
        vecB[i] = (float)vecB[i] / finalValue;
    }
#endif
}


int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    int dev = findCudaDevice(argc, (const char **)argv);

    int major = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

    // Arrive-Wait Barrier require a GPU of Volta (SM7X) architecture or higher.
    if (major < 7) {
        printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    int supportsCooperativeLaunch = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&supportsCooperativeLaunch, cudaDevAttrCooperativeLaunch, dev));

    if (!supportsCooperativeLaunch)
    {
        printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, Waiving the run\n", dev);
        exit(EXIT_WAIVED);
    }

    int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

    printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId)
{
    float *vecA, *d_vecA;
    float *vecB, *d_vecB;
    double *d_partialResults;
    int size = 10000000;

    vecA = new float[size];
    vecB = new float[size];

    checkCudaErrors(cudaMalloc(&d_vecA, sizeof(float)*size));
    checkCudaErrors(cudaMalloc(&d_vecB, sizeof(float)*size));

    float baseVal = 2.0;
    for (int i = 0; i < size; i++)
    {
        vecA[i] = vecB[i] = baseVal;
    }

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMemcpyAsync(d_vecA, vecA, sizeof(float)*size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_vecB, vecB, sizeof(float)*size, cudaMemcpyHostToDevice, stream));

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.
    int minGridSize = 0, blockSize = 0;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                            &minGridSize,
                            &blockSize,
                            (void*)normVecByDotProductAWBarrier,
                            0,
                            size));

    int smemSize =  ((blockSize/32)+1) * sizeof(double);

    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

    int multiProcessorCount = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

    minGridSize = multiProcessorCount * numBlocksPerSm;
    checkCudaErrors(cudaMalloc(&d_partialResults, minGridSize*sizeof(double)));

    printf("Launching normVecByDotProductAWBarrier kernel with numBlocks = %d blockSize = %d\n", minGridSize, blockSize);

    dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

    void *kernelArgs[] = {
        (void*)&d_vecA,
        (void*)&d_vecB,
        (void*)&d_partialResults,
        (void*)&size
    };

    checkCudaErrors(cudaLaunchCooperativeKernel((void *)normVecByDotProductAWBarrier, dimGrid, dimBlock, kernelArgs, smemSize, stream));

    checkCudaErrors(cudaMemcpyAsync(vecA, d_vecA, sizeof(float)*size, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    float expectedResult =  (baseVal / sqrt(size*baseVal*baseVal));
    unsigned int matches = 0;
    for (int i=0; i < size; i++)
    {
        if ((vecA[i] - expectedResult) > 0.00001)
        {
            printf("mismatch at i = %d\n", i);
            break;
        }
        else
        {
            matches++;
        }
    }

    //printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
    checkCudaErrors(cudaFree(d_vecA));
    checkCudaErrors(cudaFree(d_vecB));
    checkCudaErrors(cudaFree(d_partialResults));

    delete[] vecA;
    delete[] vecB;
    return matches == size;
}
