/*
 * Copyright 2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print the trace of CUDA graphs and correlate
 * the graph node launch to the node creation API using CUPTI callbacks.
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                       \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;

typedef struct {
  const char *funcName;
  uint32_t correlationId;
} ApiData;

typedef std::map<uint64_t, ApiData> nodeIdApiDataMap;
nodeIdApiDataMap nodeIdCorrelationMap;

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy5 *memcpy = (CUpti_ActivityMemcpy5 *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, size %llu, correlation %u, graph ID %u, graph node ID %llu\n",
              getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
              (unsigned long long) (memcpy->start - startTimestamp),
              (unsigned long long) (memcpy->end - startTimestamp),
              memcpy->deviceId, memcpy->contextId, memcpy->streamId,
              (unsigned long long)memcpy->bytes, memcpy->correlationId,
              memcpy->graphId, (unsigned long long)memcpy->graphNodeId);

      // Retrieve the information of the API used to create the node
      nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(memcpy->graphNodeId);
      if (it != nodeIdCorrelationMap.end()) {
          printf("Graph node was created using API %s with correlationId %u\n", it->second.funcName, it->second.correlationId);
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset4 *memset = (CUpti_ActivityMemset4 *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u, graph ID %u, graph node ID %llu\n",
             memset->value,
             (unsigned long long) (memset->start - startTimestamp),
             (unsigned long long) (memset->end - startTimestamp),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId, memset->graphId, (unsigned long long)memset->graphNodeId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel9 *kernel = (CUpti_ActivityKernel9 *) record;
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u), graph ID %u, graph node ID %llu\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory,
             kernel->graphId, (unsigned long long)kernel->graphNodeId);

      // Retrieve the information of the API used to create the node
      nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(kernel->graphNodeId);
      if (it != nodeIdCorrelationMap.end()) {
          printf("Graph node was created using API %s with correlationId %u\n", it->second.funcName, it->second.correlationId);
      }

      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      const char* apiName;
      CUPTI_CALL(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &apiName));
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u, Name %s\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId, apiName);
      break;
    }

  default:
    printf("  <unknown>\n");
    break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

void CUPTIAPI
callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    static const char* funcName;
    static uint32_t correlationId;

    // Check last error
    CUPTI_CALL(cuptiGetLastError());

    switch (domain) {
    case CUPTI_CB_DOMAIN_RESOURCE:
    {
        CUpti_ResourceData *resourceData = (CUpti_ResourceData *)cbInfo;
        switch (cbid)
        {
            case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED:
            {
                // Do not store info for the nodes that are created during graph instantiate
                if (!strncmp(funcName, "cudaGraphInstantiate", strlen("cudaGraphInstantiate"))) {
                    break;
                }
                CUpti_GraphData *cbData = (CUpti_GraphData *) resourceData->resourceDescriptor;
                uint64_t nodeId;

                // Query the graph node ID and store the API correlation id and function name
                CUPTI_CALL(cuptiGetGraphNodeId(cbData->node, &nodeId));
                ApiData apiData;
                apiData.correlationId = correlationId;
                apiData.funcName = funcName;
                nodeIdCorrelationMap[nodeId] = apiData;
                break;
            }
            case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED:
            {
                CUpti_GraphData *cbData = (CUpti_GraphData *) resourceData->resourceDescriptor;
                uint64_t nodeId, originalNodeId;

                // Overwrite the map entry with node ID of the cloned graph node
                CUPTI_CALL(cuptiGetGraphNodeId(cbData->originalNode, &originalNodeId));
                nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(originalNodeId);
                if (it != nodeIdCorrelationMap.end()) {
                    CUPTI_CALL(cuptiGetGraphNodeId(cbData->node, &nodeId));
                    ApiData apiData = it->second;
                    nodeIdCorrelationMap.erase(it);
                    nodeIdCorrelationMap[nodeId] = apiData;
                }
                break;
            }
            default:
                break;
        }
    }
        break;
    case CUPTI_CB_DOMAIN_RUNTIME_API:
    {
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            correlationId = cbInfo->correlationId;
            funcName = cbInfo->functionName;
        }
        break;
    }
    default:
        break;
    }
}

void
initTrace()
{
  CUpti_SubscriberHandle subscriber;

  // Enable activity record kinds.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler , NULL));

  // Enable callbacks for CUDA graph
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED));
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
   // Force flush any remaining activity buffers before termination of the application
   CUPTI_CALL(cuptiActivityFlushAll(1));
}
