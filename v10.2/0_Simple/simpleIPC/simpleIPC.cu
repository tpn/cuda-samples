/*
 * Copyright 2018-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates Inter Process Communication
 * using one process per GPU for computation.
 */
#include <vector>
#include "helper_cuda.h"
#include "helper_multiprocess.h"
#include <stdio.h>
#include <stdlib.h>
static const char shmName[] = "simpleIPCshm";
// For direct NVLINK and PCI-E peers, at max 8 simultaneous peers are allowed
// For NVSWITCH connected peers like DGX-2, simultaneous peers are not limited
// in the same way.
#define MAX_DEVICES (32)
#define DATA_SIZE (64ULL << 20ULL)   // 64MB

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

typedef struct shmStruct_st {
    size_t nprocesses;
    int barrier;
    int sense;
    int devices[MAX_DEVICES];
    cudaIpcMemHandle_t memHandle[MAX_DEVICES];
    cudaIpcEventHandle_t eventHandle[MAX_DEVICES];
} shmStruct;

__global__ void simpleKernel(char *ptr, int sz, char val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
        ptr[idx] = val;
    }
}

static void barrierWait(volatile int *barrier, volatile int *sense, unsigned int n)
{
    int count;

    // Check-in
    count = cpu_atomic_add32(barrier, 1);
    if (count == n)     // Last one in
        *sense = 1;
    while(!*sense);

    // Check-out
    count = cpu_atomic_add32(barrier, -1);
    if (count == 0)     // Last one out
        *sense = 0;
    while(*sense);
}

static void childProcess(int id)
{
    volatile shmStruct *shm = NULL;
    cudaStream_t stream;
    sharedMemoryInfo info;
    size_t procCount, i;
    int blocks = 0;
    int threads = 128;
    cudaDeviceProp prop;
    std::vector<void *> ptrs;
    std::vector<cudaEvent_t> events;
    std::vector<char> verification_buffer(DATA_SIZE);

    if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    procCount = shm->nprocesses;

    printf("Process %d: Starting on device %d...\n", id, shm->devices[id]);

    checkCudaErrors(cudaSetDevice(shm->devices[id]));
    checkCudaErrors(cudaGetDeviceProperties(&prop, shm->devices[id]));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, simpleKernel, threads, 0));
    blocks *= prop.multiProcessorCount;

    // Open and track all the allocations and events created in the master
    // process for use later
    for (i = 0; i < procCount; i++) {
        void *ptr = NULL;
        cudaEvent_t event;

        // Notice, we don't need to explicitly enable peer access for
        // allocations on other devices.
        checkCudaErrors(cudaIpcOpenMemHandle(&ptr, *(cudaIpcMemHandle_t*)&shm->memHandle[i], cudaIpcMemLazyEnablePeerAccess));
        checkCudaErrors(cudaIpcOpenEventHandle(&event, *(cudaIpcEventHandle_t*)&shm->eventHandle[i]));

        ptrs.push_back(ptr);
        events.push_back(event);
    }

    // At each iteration of the loop, each sibling process will push work on
    // their respective devices accessing the next peer mapped buffer allocated
    // by the master process (these can come from other sibling processes as well).
    // To coordinate each process' access, we force the stream to wait for the work
    // already accessing this buffer asynchronously through IPC events, allowing
    // the CPU processes to continue to queue more work.
    for (i = 0; i < procCount; i++) {
        size_t bufferId = (i + id) % procCount;
        // Wait for the buffer to be accessed to be ready
        checkCudaErrors(cudaStreamWaitEvent(stream, events[bufferId], 0));
        // Push a simple kernel on it
        simpleKernel<<<blocks, threads, 0, stream>>>((char *)ptrs[bufferId], DATA_SIZE, id);
        checkCudaErrors(cudaGetLastError());
        // Signal that this buffer is ready for the next consumer
        checkCudaErrors(cudaEventRecord(events[bufferId], stream));
        // Wait for all my sibling processes to push this stage of their work
        // before proceeding to the next. This prevents siblings from racing
        // ahead and clobbering the recorded event or waiting on the wrong
        // recorded event.
        barrierWait(&shm->barrier, &shm->sense, (unsigned int)procCount);
        if (id == 0) {
            printf("Step %lld done\n", (unsigned long long)i);
        }
    }

    // Now wait for my buffer to be ready so I can copy it locally and verify it
    checkCudaErrors(cudaStreamWaitEvent(stream, events[id], 0));
    checkCudaErrors(cudaMemcpyAsync(&verification_buffer[0], ptrs[id], DATA_SIZE, cudaMemcpyDeviceToHost, stream));
    // And wait for all the queued up work to complete
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Process %d: verifying...\n", id);

    // The contents should have the id of the sibling just after me
    char compareId = (char)((id + 1) % procCount);
    for (unsigned long long j = 0; j < DATA_SIZE; j++) {
        if (verification_buffer[j] != compareId) {
            printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
                   (int)verification_buffer[j], (int)compareId);
        }
    }

    // Clean up!
    for (i = 0; i < procCount; i++) { 
        checkCudaErrors(cudaIpcCloseMemHandle(ptrs[i]));
        checkCudaErrors(cudaEventDestroy(events[i]));
    }

    checkCudaErrors(cudaStreamDestroy(stream));

    printf("Process %d complete!\n", id);
}

static void parentProcess(char *app)
{
    sharedMemoryInfo info;
    int devCount, i;
    volatile shmStruct *shm = NULL;
    std::vector<void *> ptrs;
    std::vector<cudaEvent_t> events;
    std::vector<Process> processes;

    checkCudaErrors(cudaGetDeviceCount(&devCount));

    if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    memset((void *)shm, 0, sizeof(*shm));

    // Pick all the devices that can access each other's memory for this test
    // Keep in mind that CUDA has minimal support for fork() without a
    // corresponding exec() in the child process, but in this case our
    // spawnProcess will always exec, so no need to worry.
    for (i = 0; i < devCount; i++) {
        bool allPeers = true;
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));

        // CUDA IPC is only supported on devices with unified addressing
        if (!prop.unifiedAddressing) {
			printf("Device %d does not support unified addressing, skipping...\n", i);
			continue;
		}
        // This sample requires two processes accessing each device, so we need
        // to ensure exclusive or prohibited mode is not set
        if (prop.computeMode != cudaComputeModeDefault) {
			printf("Device %d is in an unsupported compute mode for this sample\n", i);
			continue;
		}
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        // CUDA IPC on Windows is only supported on TCC
        if (!prop.tccDriver) {
			printf("Device %d is not in TCC mode\n", i);
			continue;
		}
#endif

        for (int j = 0; j < shm->nprocesses; j++) {
            int canAccessPeerIJ, canAccessPeerJI;
            checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerJI, shm->devices[j], i));
            checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, shm->devices[j]));
            if (!canAccessPeerIJ || !canAccessPeerJI) {
                allPeers = false;
                break;
            }
        }
        if (allPeers) {
            // Enable peers here.  This isn't necessary for IPC, but it will
            // setup the peers for the device.  For systems that only allow 8
            // peers per GPU at a time, this acts to remove devices from CanAccessPeer
            for (int j = 0; j < shm->nprocesses; j++) {
                checkCudaErrors(cudaSetDevice(i));
                checkCudaErrors(cudaDeviceEnablePeerAccess(shm->devices[j], 0));
                checkCudaErrors(cudaSetDevice(shm->devices[j]));
                checkCudaErrors(cudaDeviceEnablePeerAccess(i, 0));
            }
            shm->devices[shm->nprocesses++] = i;
            if (shm->nprocesses >= MAX_DEVICES) break;
        }
		else {
			printf("Device %d is not peer capable with some other selected peers, skipping\n", i);
		}
    }

    if (shm->nprocesses == 0) {
        printf("No CUDA devices support IPC\n");
        exit(EXIT_WAIVED);
    }

    // Now allocate memory and an event for each process and fill the shared
    // memory buffer with the IPC handles to communicate
    for (i = 0; i < shm->nprocesses; i++) {
        void *ptr = NULL;
        cudaEvent_t event;

        checkCudaErrors(cudaSetDevice(shm->devices[i]));
        checkCudaErrors(cudaMalloc(&ptr, DATA_SIZE));
        checkCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm->memHandle[i], ptr));
        checkCudaErrors(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
        checkCudaErrors(cudaIpcGetEventHandle((cudaIpcEventHandle_t *)&shm->eventHandle[i], event));

        ptrs.push_back(ptr);
        events.push_back(event);
    }

    // Launch the child processes!
    for (i = 0; i < shm->nprocesses; i++) {
        char devIdx[10];
        char * const args[] = { app, devIdx , NULL };
        Process process;

        SPRINTF(devIdx, "%d", i);

        if (spawnProcess(&process, app, args)) {
            printf("Failed to create process\n");
            exit(EXIT_FAILURE);
        }

        processes.push_back(process);
    }

    // And wait for them to finish
    for (i = 0; i < processes.size(); i++) {
        if (waitProcess(&processes[i]) != EXIT_SUCCESS) {
            printf("Process %d failed!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Clean up!
    for (i = 0; i < shm->nprocesses; i++) {
        checkCudaErrors(cudaSetDevice(shm->devices[i]));
        checkCudaErrors(cudaEventSynchronize(events[i]));
        checkCudaErrors(cudaEventDestroy(events[i]));
        checkCudaErrors(cudaFree(ptrs[i]));
    }

    sharedMemoryClose(&info);
}

int main(int argc, char **argv)
{
#if defined(__arm__) || defined(__aarch64__)
    printf("Not supported on ARM\n");
    return EXIT_WAIVED;
#else
    if (argc == 1) {
        parentProcess(argv[0]);
    }
    else {
        childProcess(atoi(argv[1]));
    }
    return EXIT_SUCCESS;
#endif
}
