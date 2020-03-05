/////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////

// This example demonstrates how to use the CUDA Direct3D bindings with the
// runtime API.
// Device code.

#ifndef _SIMPLED3D_KERNEL_CU_
#define _SIMPLED3D_KERNEL_CU_

// includes, C string library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param pos  pos in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, __int_as_float(0xff00ff00));
}

extern "C"
void simpleD3DKernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    cudaError_t error = cudaSuccess;

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    kernel<<<grid, block>>>(pos, width, height, time);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("kernel() failed to launch error = %d\n", error);
    }
}

#endif // #ifndef _SIMPLED3D_KERNEL_CU_
