/*
 * Copyright 2014-2017 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print sass to source correlation
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

static CUpti_SubscriberHandle g_subscriber;

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__
void transpose(float *d_Outdata, const float *d_Indata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y+j][threadIdx.x] = d_Indata[(y+j)*width + x];
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    d_Outdata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  // The activity record for source locator contains the ID for the source path,
  // path for the file and the line number in the source.
  case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    {
      CUpti_ActivitySourceLocator *sourceLocator = (CUpti_ActivitySourceLocator *)record;
      printf("SOURCE_LOCATOR SrcLctrId %d, File %s Line %d\n", sourceLocator->id, sourceLocator->fileName, sourceLocator->lineNumber);
      break;
    }
  // The activity record for instruction execution corresponds to a PC of the generated code, it contains the ID for source locator
  // the correlation ID of the kernel to which this record is associated, function ID and pc offset for the instruction.
  case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION:
    {
      CUpti_ActivityInstructionExecution *sourceRecord = (CUpti_ActivityInstructionExecution *)record;
      printf("INSTRUCTION_EXECUTION srcLctr %u, corr %u, functionId %u, pc %x\n",
        sourceRecord->sourceLocatorId, sourceRecord->correlationId, sourceRecord->functionId,
        sourceRecord->pcOffset);
      // number of threads that executed this instruction and number of times the instruction was executed
      printf("notPredOffthread_inst_executed %llu, thread_inst_executed %llu, inst_executed %u\n\n",
        (unsigned long long)sourceRecord->notPredOffThreadsExecuted,
        (unsigned long long)sourceRecord->threadsExecuted, sourceRecord->executed);
      break;
    }
  // function name and corresponding module information
  case CUPTI_ACTIVITY_KIND_FUNCTION:
    {
      CUpti_ActivityFunction *fResult = (CUpti_ActivityFunction *)record;
      printf("FUCTION functionId %u, moduleId %u, name %s\n",
        fResult->id,
        fResult->moduleId,
        fResult->name);
      break;
    }
  default:
    printf("  <unknown>\n");
    exit(EXIT_FAILURE);
    break;
  }
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *b;

  *size = BUF_SIZE;
  b = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(b, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if(status == CUPTI_SUCCESS) {
      printActivity(record);
    }
    else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
    else {
      CUPTI_CALL(status);
    }
  } while (1);

  size_t dropped;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    printf("Dropped %u activity records\n", (unsigned int)dropped);
  }

  free(buffer);
}

#define DUMP_CUBIN 0

void CUPTIAPI dumpCudaModule(CUpti_CallbackId cbid, void *resourceDescriptor)
{
#if DUMP_CUBIN
  const char *pCubin;
  size_t cubinSize;

  // dump the cubin at MODULE_LOADED_STARTING
  CUpti_ModuleResourceData *moduleResourceData = (CUpti_ModuleResourceData *)resourceDescriptor;
#endif

  if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
#if DUMP_CUBIN
    // You can use nvdisasm to dump the SASS from the cubin.
    // Try nvdisasm -b -fun <function_id> sass_to_source.cubin
    pCubin = moduleResourceData->pCubin;
    cubinSize = moduleResourceData->cubinSize;

    FILE *cubin;
    cubin = fopen("sass_source_map.cubin", "wb");
    fwrite(pCubin, sizeof(uint8_t), cubinSize, cubin);
    fclose(cubin);
#endif
  }
  else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    // You can dump the cubin either at MODULE_LOADED or MODULE_UNLOAD_STARTING
  }
}

static void
handleResource(CUpti_CallbackId cbid, const CUpti_ResourceData *resourceData)
{
  if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    dumpCudaModule(cbid, resourceData->resourceDescriptor);
  }
  else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    dumpCudaModule(cbid, resourceData->resourceDescriptor);
  }
}

static void CUPTIAPI
traceCallback(void *userdata, CUpti_CallbackDomain domain,
              CUpti_CallbackId cbid, const void *cbdata)
{
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    handleResource(cbid, (CUpti_ResourceData *)cbdata);
  }
}

void
initTrace()
{
  // do cupti calls before any CUDA call
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  CUPTI_CALL(cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)traceCallback, NULL));
  CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void
finiTrace()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));
  CUPTI_CALL(cuptiUnsubscribe(g_subscriber));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
}

int
main(int argc, char *argv[])
{
  const int nx = 32;
  const int ny = 32;
  const int mem_size = nx*ny*sizeof(float);
  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
  cudaDeviceProp g_deviceProp;

  initTrace();

  RUNTIME_API_CALL(cudaGetDeviceProperties(&g_deviceProp, 0));
  printf("Device Name: %s\n", g_deviceProp.name);

  float *d_X, *d_Y;

  float *h_X = (float*)malloc(mem_size);
  float *h_Y = (float*)malloc(mem_size);
  if (!(h_X && h_Y)) {
    printf("Malloc failed\n");
    exit(EXIT_FAILURE);
  }
  // initialization of host data
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      h_X[j*nx + i] = (float) (j*nx + i);
    }
  }
  RUNTIME_API_CALL(cudaMalloc(&d_X, mem_size));
  RUNTIME_API_CALL(cudaMalloc(&d_Y, mem_size));

  RUNTIME_API_CALL(cudaMemcpy(d_X, h_X, mem_size, cudaMemcpyHostToDevice));

  transpose<<<dimGrid, dimBlock>>>(d_Y, d_X);

  RUNTIME_API_CALL(cudaMemcpy(h_Y, d_Y, mem_size, cudaMemcpyDeviceToHost));

  free(h_X);
  free(h_Y);

  cudaFree(d_X);
  cudaFree(d_Y);

  cudaDeviceSynchronize();
  cudaDeviceReset();

  finiTrace();
  exit(EXIT_SUCCESS);
}

