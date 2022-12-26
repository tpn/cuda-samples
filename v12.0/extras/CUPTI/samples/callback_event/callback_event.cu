/*
 * Copyright 2010-2017 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler event values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <stdlib.h>

#define EVENT_NAME "inst_executed"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("%s:%d: error %d for CUDA Driver API function '%s'\n",    \
              __FILE__, __LINE__, err, cufunc);                         \
      exit(EXIT_FAILURE);                                                \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
  if (err != CUPTI_SUCCESS)                                             \
    {                                                                   \
      const char *errstr;                                               \
      cuptiGetResultString(err, &errstr);                               \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",          \
              __FILE__, __LINE__, errstr, cuptifunc);                   \
      exit(EXIT_FAILURE);                                               \
    }

typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;

// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

static void
initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

void CUPTIAPI
getEventValueCallback(void *userdata, CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  CUptiResult cuptiErr;
  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
  size_t bytesRead;

  // This callback is enabled only for launch so we shouldn't see anything else.
  if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
  {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    exit(EXIT_FAILURE);
  }

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cudaDeviceSynchronize();
    cuptiErr = cuptiSetEventCollectionMode(cbInfo->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");
    cuptiErr = cuptiEventGroupEnable(traceData->eventData->eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");
  }

  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint32_t numInstances = 0, i;
    uint64_t *values = NULL;
    size_t valueSize = sizeof(numInstances);

    cuptiErr = cuptiEventGroupGetAttribute(traceData->eventData->eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                           &valueSize, &numInstances);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

    bytesRead = sizeof (uint64_t) * numInstances;
    values = (uint64_t *) malloc(bytesRead);
    if (values == NULL) {
        printf("%s:%d: Out of memory\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    cuptiErr = cuptiEventGroupReadEvent(traceData->eventData->eventGroup,
                                        CUPTI_EVENT_READ_FLAG_NONE,
                                        traceData->eventData->eventId,
                                        &bytesRead, values);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");

    traceData->eventVal = 0;
    for (i=0; i<numInstances; i++) {
        traceData->eventVal += values[i];
    }
    free(values);

    cuptiErr = cuptiEventGroupDisable(traceData->eventData->eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");
  }
}

static void
displayEventVal(RuntimeApiTrace_t *trace, const char *eventName)
{
  printf("Event Name : %s \n", eventName);
  printf("Event Value : %llu\n", (unsigned long long) trace->eventVal);
}

static void
cleanUp(int *h_A, int *h_B, int *h_C, int *d_A, int *d_B, int *d_C)
{
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
}

int
main(int argc, char *argv[])
{
  CUcontext context = 0;
  CUdevice dev = 0;
  CUresult err;
  int N = 50000;
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int sum, i;
  int computeCapabilityMajor=0;
  int computeCapabilityMinor=0;
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  int deviceNum;
  int deviceCount;
  char deviceName[256];
  const char *eventName;
  uint32_t profile_all = 1;

  CUptiResult cuptiErr;
  CUpti_SubscriberHandle subscriber;
  cupti_eventData cuptiEvent;
  RuntimeApiTrace_t trace;

  printf("Usage: %s [device_num] [event_name]\n", argv[0]);

  err = cuInit(0);
  CHECK_CU_ERROR(err, "cuInit");

  err = cuDeviceGetCount(&deviceCount);
  CHECK_CU_ERROR(err, "cuDeviceGetCount");

  if (deviceCount == 0) {
    printf("There is no device supporting CUDA.\n");
    exit(EXIT_WAIVED);
  }

  if (argc > 1)
    deviceNum = atoi(argv[1]);
  else
    deviceNum = 0;
  printf("CUDA Device Number: %d\n", deviceNum);

  err = cuDeviceGet(&dev, deviceNum);
  CHECK_CU_ERROR(err, "cuDeviceGet");

  err = cuDeviceGetName(deviceName, 256, dev);
  CHECK_CU_ERROR(err, "cuDeviceGetName");

  printf("CUDA Device Name: %s\n", deviceName);

  err = cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

  err = cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

  printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);
  int deviceComputeCapability = 10 * computeCapabilityMajor + computeCapabilityMinor;
  if(deviceComputeCapability > 72) {
    printf("Sample unsupported on Device with compute capability > 7.2\n");
    exit(EXIT_WAIVED);
  }

  err = cuCtxCreate(&context, 0, dev);
  CHECK_CU_ERROR(err, "cuCtxCreate");


  // Creating event group for profiling
  cuptiErr = cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

  if (argc > 2) {
      eventName = argv[2];
  }
  else {
      eventName = EVENT_NAME;
  }

  cuptiErr = cuptiEventGetIdFromName(dev, eventName, &cuptiEvent.eventId);
  if (cuptiErr != CUPTI_SUCCESS)
    {
      printf("Invalid eventName: %s\n", eventName);
      exit(EXIT_FAILURE);
    }

  cuptiErr = cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

  cuptiErr = cuptiEventGroupSetAttribute(cuptiEvent.eventGroup,
                                         CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                         sizeof(profile_all), &profile_all);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");

  trace.eventData = &cuptiEvent;

  cuptiErr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback , &trace);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiSubscribe");

  cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");
  cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");

  // Allocate input vectors h_A and h_B in host memory
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);

  // Initialize input vectors
  initVec(h_A, N);
  initVec(h_B, N);
  memset(h_C, 0, size);

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  threadsPerBlock = 256;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("Launching kernel: blocks %d, thread/block %d\n",
         blocksPerGrid, threadsPerBlock);

  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (i = 0; i < N; ++i) {
    sum = h_A[i] + h_B[i];
    if (h_C[i] != sum) {
      printf("kernel execution FAILED\n");
      goto Error;
    }
  }

  displayEventVal(&trace, eventName);

  trace.eventData = NULL;

  cuptiErr = cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupRemoveEvent");

  cuptiErr = cuptiEventGroupDestroy(cuptiEvent.eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

  cuptiErr = cuptiUnsubscribe(subscriber);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiUnsubscribe");

  cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
  cudaDeviceSynchronize();
  exit(EXIT_SUCCESS);

 Error:
  cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
  cudaDeviceSynchronize();
  exit(EXIT_FAILURE);
}

