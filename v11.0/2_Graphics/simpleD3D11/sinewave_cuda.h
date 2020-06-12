/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef SINEWAVE_CUDA_H
#define SINEWAVE_CUDA_H

#include <stdio.h>
#include "ShaderStructs.h"
#include "helper_cuda.h"

void RunSineWaveKernel(cudaExternalSemaphore_t &extSemaphore, uint64_t &key, unsigned int timeoutMs,
                        size_t mesh_width, size_t mesh_height, Vertex *cudaDevVertptr, cudaStream_t streamToRun);
Vertex* cudaImportVertexBuffer(void*sharedHandle, cudaExternalMemory_t &externalMemory, int meshWidth, int meshHeight);
void cudaImportKeyedMutex(void*sharedHandle, cudaExternalSemaphore_t &extSemaphore);
#endif // !