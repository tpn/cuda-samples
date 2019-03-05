/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Paint a 3D texture with a gradient in X (blue) and Z (green), and have every
 * other Z slice have full red.
 */
__global__ void cuda_kernel_texture_3d(unsigned char *surface, int width, int height, int depth, size_t pitch, size_t pitchSlice, float t)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height) return;

    // walk across the Z slices of this texture.  it should be noted that
    // this is far from optimal data access.
    for (int z = 0; z < depth; ++z)
    {
        // get a pointer to this pixel
        unsigned char *pixel = surface + z*pitchSlice + y*pitch + 4*x;
        pixel[0] = (unsigned char)(255.f * (0.5f + 0.5f*cos(t + (x*x + y*y + z*z)*0.0001f *3.14f)));   // red
        pixel[1] = (unsigned char)(255.f * (0.5f + 0.5f*sin(t + (x*x + y*y + z*z)*0.0001f *3.14f)));   // green
        pixel[2] = (unsigned char) 0;  // blue
        pixel[3] = 255; // alpha
    }
}

extern "C"
void cuda_texture_3d(void *surface, int width, int height, int depth, size_t pitch, size_t pitchSlice, float t)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

    cuda_kernel_texture_3d<<<Dg,Db>>>((unsigned char *)surface, width, height, depth, pitch, pitchSlice, t);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}
