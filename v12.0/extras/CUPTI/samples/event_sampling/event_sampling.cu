/*
 * Copyright 2011-2021 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler
 * event values by sampling.
 */


#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti_events.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("Error %d for CUDA Driver API function '%s'.\n",          \
              err, cufunc);                                             \
      exit(EXIT_FAILURE);                                               \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                       \
  if (err != CUPTI_SUCCESS)                                     \
    {                                                           \
      const char *errstr;                                       \
      cuptiGetResultString(err, &errstr);                       \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",  \
              __FILE__, __LINE__, errstr, cuptifunc);           \
      exit(EXIT_FAILURE);                                       \
    }

#define EVENT_NAME "inst_executed"
#define N 100000
#define ITERATIONS 10000
#define SAMPLE_PERIOD_MS 50

#ifdef _WIN32
HANDLE semaphore;
DWORD ret;
#else
sem_t semaphore;
int ret;
#endif

// used to signal from the compute thread to the sampling thread
static volatile int testComplete = 0;

static CUcontext context;
static CUdevice device;
static const char *eventName;

// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int n = 0 ; n < 100; n++) {
    if (i < size)
      C[i] = A[i] + B[i];
  }
}

static void
initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

void *
sampling_func(void *arg)
{
  CUptiResult cuptiErr;
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
  size_t bytesRead, valueSize;
  uint32_t numInstances = 0, j = 0;
  uint64_t *eventValues = NULL, eventVal = 0;
  uint32_t profile_all = 1;

  cuptiErr = cuptiSetEventCollectionMode(context,
                                         CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

  cuptiErr = cuptiEventGroupCreate(context, &eventGroup, 0);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

  cuptiErr = cuptiEventGetIdFromName(device, eventName, &eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");

  cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

  cuptiErr = cuptiEventGroupSetAttribute(eventGroup,
                                         CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                         sizeof(profile_all), &profile_all);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");

  cuptiErr = cuptiEventGroupEnable(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

  valueSize = sizeof(numInstances);
  cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
                                         CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                         &valueSize, &numInstances);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

  bytesRead = sizeof(uint64_t) * numInstances;
  eventValues = (uint64_t *) malloc(bytesRead);
  if (eventValues == NULL) {
      printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  // Release the semaphore as sampling thread is ready to read events
#ifdef _WIN32
  ret = ReleaseSemaphore(semaphore, 1, NULL);
  if (ret == 0) {
    printf("Failed to release the semaphore\n");
    exit(EXIT_FAILURE);
  }
#else
  ret = sem_post(&semaphore);
  if (ret != 0) {
    printf("Failed to release the semaphore\n");
    exit(EXIT_FAILURE);
  }
#endif

  do {
    cuptiErr = cuptiEventGroupReadEvent(eventGroup,
                                        CUPTI_EVENT_READ_FLAG_NONE,
                                        eventId, &bytesRead, eventValues);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
    if (bytesRead != (sizeof(uint64_t) * numInstances)) {
      printf("Failed to read value for \"%s\"\n", eventName);
      exit(EXIT_FAILURE);
    }

    for (j = 0; j < numInstances; j++) {
      eventVal += eventValues[j];
    }
    printf("%s: %llu\n", eventName, (unsigned long long)eventVal);
#ifdef _WIN32
    Sleep(SAMPLE_PERIOD_MS);
#else
    usleep(SAMPLE_PERIOD_MS * 1000);
#endif
  } while (!testComplete);
  cuptiErr = cuptiEventGroupDisable(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

  cuptiErr = cuptiEventGroupDestroy(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

  free(eventValues);
  return NULL;
}

static void
compute(int iters)
{
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int sum, i;
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;

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

  // Invoke kernel (multiple times to make sure we have time for
  // sampling)
  threadsPerBlock = 256;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  for (i = 0; i < iters; i++) {
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  }

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (i = 0; i < N; ++i) {
    sum = h_A[i] + h_B[i];
    if (h_C[i] != sum) {
      printf("kernel execution FAILED\n");
      exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
}

int
main(int argc, char *argv[])
{
#ifdef _WIN32
  HANDLE hThread;
#else
  int status;
  pthread_t pThread;
#endif
  CUresult err;
  int deviceNum;
  int deviceCount;
  char deviceName[256];
  int major;
  int minor;

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

  err = cuDeviceGet(&device, deviceNum);
  CHECK_CU_ERROR(err, "cuDeviceGet");

  err = cuDeviceGetName(deviceName, 256, device);
  CHECK_CU_ERROR(err, "cuDeviceGetName");

  printf("CUDA Device Name: %s\n", deviceName);

  err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

  err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

  printf("Compute Capability of Device: %d.%d\n", major,minor);
  int deviceComputeCapability = 10 * major + minor;
  if(deviceComputeCapability > 72) {
    printf("Sample unsupported on Device with compute capability > 7.2\n");
    exit(EXIT_WAIVED);
  }

  if (argc > 2) {
    eventName = argv[2];
  }
  else {
    eventName = EVENT_NAME;
  }

  err = cuCtxCreate(&context, 0, device);
  CHECK_CU_ERROR(err, "cuCtxCreate");

  // Create semaphore
#ifdef _WIN32
  semaphore = CreateSemaphore(NULL, 0, 10, NULL);
  if (semaphore == NULL) {
    printf("Failed to create the semaphore\n");
    exit(EXIT_FAILURE);
  }
#else
  ret = sem_init(&semaphore, 0, 0);
  if (ret != 0) {
    printf("Failed to create the semaphore\n");
    exit(EXIT_FAILURE);
  }
#endif

  testComplete = 0;

  printf("Creating sampling thread\n");
#ifdef _WIN32
  hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) sampling_func,
                         NULL, 0, NULL );
  if (!hThread) {
    printf("CreateThread failed\n");
    exit(EXIT_FAILURE);
  }
#else
  status = pthread_create(&pThread, NULL, sampling_func, NULL);
  if (status != 0) {
    perror("pthread_create");
    exit(EXIT_FAILURE);
  }
#endif

  // Wait for sampling thread to be ready for event collection
#ifdef _WIN32
  ret = WaitForSingleObject(semaphore, INFINITE);
  if (ret != WAIT_OBJECT_0) {
    printf("Failed to wait for the semaphore\n");
    exit(EXIT_FAILURE);
  }
#else
  ret = sem_wait(&semaphore);
  if (ret != 0) {
    printf("Failed to wait for the semaphore\n");
    exit(EXIT_FAILURE);
  }
#endif

  // run kernel while sampling
  compute(ITERATIONS);

  // "signal" the sampling thread to exit and wait for it
  testComplete = 1;
#ifdef _WIN32
  WaitForSingleObject(hThread, INFINITE);
#else
  pthread_join(pThread, NULL);
#endif

  cudaDeviceSynchronize();
  exit(EXIT_SUCCESS);
}

