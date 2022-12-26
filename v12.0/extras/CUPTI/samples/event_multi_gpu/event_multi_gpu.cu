/*
* Copyright 2014-2017 NVIDIA Corporation. All rights reserved.
*
* Sample app to demonstrate use of CUPTI library to obtain profiler
* event values on a multi-gpu setup without serializing the compute
* work on the devices.
*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti_events.h>
#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CUPTI_CALL(call)                                                    \
do {                                                                        \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
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

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MAX_DEVICES     32
#define BLOCK_X         32
#define GRID_X          32

// default event
#define EVENT_NAME      "inst_executed"

// dummy kernel
__global__ void kernel() {
    uint64_t i = 0;
    volatile uint64_t limit = 1024 * 128;
    for (i = 0; i < limit; i++) {
    }
}

int
main(int argc, char *argv[])
{
    int deviceCount;
    char deviceName[256];
    CUdevice device[MAX_DEVICES];
    CUcontext context[MAX_DEVICES];
    CUpti_EventGroup eventGroup[MAX_DEVICES];
    CUpti_EventID eventId[MAX_DEVICES];
    size_t bytesRead, valueSize;
    uint32_t numInstances = 0, j = 0;
    uint64_t *eventValues = NULL, eventVal = 0;
    const char *eventName;
    int i = 0;
    uint32_t profile_all = 1;

    printf("Usage: %s [event_name]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(EXIT_WAIVED);
    }

    if (deviceCount < 2) {
        printf("This multi-gpu test is waived on single gpu setup.\n");
        exit(EXIT_WAIVED);
    }

    if (deviceCount > MAX_DEVICES) {
        printf("Found more devices (%d) than handled in the test (%d)\n",
            deviceCount, MAX_DEVICES);
        exit(EXIT_WAIVED);
    }

    if (argc > 1) {
        eventName = argv[1];
    }
    else {
        eventName = EVENT_NAME;
    }

    for (i = 0; i < deviceCount; i++) {
        DRIVER_API_CALL(cuDeviceGet(&device[i], i));

        DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device[i]));

        printf("CUDA Device Name: %s\n", deviceName);
    }

    // create one context per device
    for (i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxCreate(&(context[i]), 0, device[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&(context[i])));
    }

    // enable event profiling on each device
    for (i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        CUPTI_CALL(cuptiSetEventCollectionMode(context[i],
            CUPTI_EVENT_COLLECTION_MODE_KERNEL));
        CUPTI_CALL(cuptiEventGroupCreate(context[i], &eventGroup[i], 0));
        CUPTI_CALL(cuptiEventGetIdFromName(device[i], eventName, &eventId[i]));
        CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup[i], eventId[i]));
        CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup[i],
                                               CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                               sizeof(profile_all), &profile_all));
        CUPTI_CALL(cuptiEventGroupEnable(eventGroup[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // launch kernel on each device
    for (i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        kernel<<<GRID_X, BLOCK_X>>>();

        // don't do any sync here, it's done once
        // work is queued on all devices

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // sync each context now
    for (i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        DRIVER_API_CALL(cuCtxSynchronize());

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // read events
    for (i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        valueSize = sizeof(numInstances);
        CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup[i],
                                               CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                               &valueSize, &numInstances));

        bytesRead = sizeof(uint64_t) * numInstances;
        eventValues = (uint64_t *) malloc(bytesRead);
        if (eventValues == NULL) {
            printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        CUPTI_CALL(cuptiEventGroupReadEvent(eventGroup[i],
            CUPTI_EVENT_READ_FLAG_NONE,
            eventId[i], &bytesRead, eventValues));

        if (bytesRead != (sizeof(uint64_t) * numInstances)) {
            printf("Failed to read value for \"%s\"\n", eventName);
            exit(EXIT_FAILURE);
        }

        for (j = 0; j < numInstances; j++) {
            eventVal += eventValues[j];
        }

        printf("[%d] %s: %llu\n", i, eventName, (unsigned long long)eventVal);

        CUPTI_CALL(cuptiEventGroupDisable(eventGroup[i]));
        CUPTI_CALL(cuptiEventGroupDestroy(eventGroup[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    exit(EXIT_SUCCESS);
}
