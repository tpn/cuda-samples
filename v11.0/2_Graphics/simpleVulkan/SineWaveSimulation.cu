/*
 * Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "SineWaveSimulation.h"
#include <algorithm>
#include <helper_cuda.h>

__global__ void sinewave(float *heightMap, unsigned int width, unsigned int height, float time)
{
    const float freq = 4.0f;
    const size_t stride = gridDim.x * blockDim.x;

    // Iterate through the entire array in a way that is
    // independent of the grid configuration
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < width * height; tid += stride) {
        // Calculate the x, y coordinates
        const size_t y = tid / width;
        const size_t x = tid - y * width;
        // Normalize x, y to [0,1]
        const float u = ((2.0f * x) / width)  - 1.0f;
        const float v = ((2.0f * y) / height) - 1.0f;
        // Calculate the new height value
        const float w = 0.5f * sinf(u * freq + time) * cosf(v * freq + time);
        // Store this new height value
        heightMap[tid] = w;
    }
}

SineWaveSimulation::SineWaveSimulation(size_t width, size_t height) 
                                        : m_heightMap(nullptr), m_width(width), m_height(height)
{
}

void SineWaveSimulation::initCudaLaunchConfig(int device)
{
    cudaDeviceProp prop = {};
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    // We don't need large block sizes, since there's not much inter-thread communication
    m_threads = prop.warpSize;

    // Use the occupancy calculator and fill the gpu as best as we can
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_blocks, sinewave, prop.warpSize, 0));
    m_blocks *= prop.multiProcessorCount;

    // Go ahead and the clamp the blocks to the minimum needed for this height/width
    m_blocks = std::min(m_blocks, (int)((m_width * m_height + m_threads - 1) / m_threads));
}

int SineWaveSimulation::initCuda(uint8_t  *vkDeviceUUID, size_t UUID_SIZE)
{
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the GPU which is selected by Vulkan
    while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);

        if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp((void*)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
            if (ret == 0)
            {
                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                 current_device, deviceProp.name, deviceProp.major,
                 deviceProp.minor);

                return current_device;
            }

        } else {
          devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr,
                "CUDA error:"
                " No Vulkan-CUDA Interop capable GPU found.\n");
        exit(EXIT_FAILURE);
    }

    return -1;
}

SineWaveSimulation::~SineWaveSimulation()
{
    m_heightMap = NULL;
}

void SineWaveSimulation::initSimulation(float *heights)
{
    m_heightMap = heights;
}

void SineWaveSimulation::stepSimulation(float time, cudaStream_t stream)
{
    sinewave <<< m_blocks, m_threads, 0, stream >>> (m_heightMap, m_width, m_height, time);
    getLastCudaError("Failed to launch CUDA simulation");
}
