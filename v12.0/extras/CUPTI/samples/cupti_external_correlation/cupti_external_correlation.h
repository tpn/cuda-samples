/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to show how to do external correlation.
 * The sample pushes external ids in a simple vector addition
 * application showing how you can externally correlate different
 * phases of the code. In this sample it is broken into
 * initialization, execution and cleanup showing how you can
 * correlate all the APIs invloved in these 3 phases in the app.
 *
 * Psuedo code:
 * cuptiActivityPushExternalCorrelationId()
 * ExternalAPI() -> (runs bunch of CUDA APIs/ launches activity on GPU)
 * cuptiActivityPopExternalCorrelationId()
 * All CUDA activity activities within this range will generate external correlation
 * record which then can be used to correlate it with the external API
 */

#ifndef CUPTI_EXTERNAL_CORRELATION_H
#define CUPTI_EXTERNAL_CORRELATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include <cupti.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CUPTI_CALL(call)                                                        \
do {                                                                            \
    CUptiResult _status = call;                                                 \
    if (_status != CUPTI_SUCCESS) {                                             \
        const char *errstr;                                                     \
        cuptiGetResultString(_status, &errstr);                                 \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #call, errstr);                             \
        exit(EXIT_FAILURE);                                                      \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                            \
do {                                                                            \
    CUresult _status = apiFuncCall;                                             \
    if (_status != CUDA_SUCCESS) {                                              \
        const char* errstr;                                                     \
        cuGetErrorString(_status, &errstr);                                     \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #apiFuncCall, errstr);                      \
        exit(EXIT_FAILURE);                                                               \
    }                                                                           \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                           \
do {                                                                            \
    cudaError_t _status = apiFuncCall;                                          \
    if (_status != cudaSuccess) {                                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
        exit(EXIT_FAILURE);                                                               \
    }                                                                           \
} while (0)

// Enum mapping the external id to the different phases in the vector addition it correlates to.
typedef enum {
    INITIALIZATION_EXTERNAL_ID = 0,
    EXECUTION_EXTERNAL_ID = 1,
    CLEANUP_EXTERNAL_ID = 2,
    MAX_EXTERNAL_ID = 3
} ExternalId;

void initTrace(void);
void finiTrace(void);

#endif // CUPTI_EXTERNAL_CORRELATION_H