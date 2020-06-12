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

#pragma once
#ifndef __SINESIM_H__
#define __SINESIM_H__

#include <vector>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "linmath.h"

class SineWaveSimulation
{
    float *m_heightMap;
    size_t m_width, m_height;
    int m_blocks, m_threads;
public:
    SineWaveSimulation(size_t width, size_t height);
    ~SineWaveSimulation();
    void initSimulation(float *heightMap);
    void stepSimulation(float time, cudaStream_t stream = 0);
    void initCudaLaunchConfig(int device);
    int initCuda(uint8_t  *vkDeviceUUID, size_t UUID_SIZE);

    size_t getWidth() const {
        return m_width;
    }
    size_t getHeight() const {
        return m_height;
    }
};

#endif // __SINESIM_H__
