// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates two ways to use the CUPTI Profiler API with concurrent kernels.
// By taking the ratio of runtimes for a consecutive series of kernels, compared
// to a series of concurrent kernels, one can difinitively demonstrate that concurrent
// kernels were running while metrics were gathered and the User Replay mechanism was in use.
//
// Example:
// 4 kernel launches, with 1x, 2x, 3x, and 4x amounts of work, each sized to one SM (one warp
// of threads, one thread block).
// When run synchronously, this comes to 10x amount of work.
// When run concurrently, the longest (4x) kernel should be the only measured time (it hides the others).
// Thus w/ 4 kernels, the concurrent : consecutive time ratio should be 4:10.
// On test hardware this does simplify to 3.998:10.  As the test is affected by memory layout, this may not
// hold for certain architectures where, for example, cache sizes may optimize certain kernel calls.
//
// After demonstrating concurrency using multpile streams, this then demonstrates using multiple devices.
// In this 3rd configuration, the same concurrent workload with streams is then duplicated and run
// on each device concurrently using streams.
// In this case, the wallclock time to launch, run, and join the threads should be roughly the same as the
// wallclock time to run the single device case.  If concurrency was not working, the wallcock time
// would be (num devices) times the single device concurrent case.
//
//  * If the multiple devices have different performance, the runtime may be significantly different between
//    devices, but this does not mean concurrent profiling is not happening.

// Standard CUDA, CUPTI, Profiler, NVPW headers
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"
#include "driver_types.h"
#include "nvperf_host.h"

// Make use of example code wrappers for NVPW calls
#include <Eval.h>
using ::NV::Metric::Eval::PrintMetricValues;

#include <Metric.h>
using ::NV::Metric::Config::GetConfigImage;
using ::NV::Metric::Config::GetCounterDataPrefixImage;

#include <Utils.h>
using ::NV::Metric::Utils::GetNVPWResultString;

// Standard STL headers
#include <stdlib.h>
#include <chrono>
#include <cstdint>
#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <string>
using ::std::string;

#include <thread>
using ::std::thread;

#include <vector>
using ::std::vector;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Helpful error handlers for standard CUPTI and CUDA runtime calls
#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
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

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
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

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, NV::Metric::Utils::GetNVPWResultString(_status)); \
    exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Consolidate CUPTI Profiler options into one location
// This contains fields for multiple levels of Profiler API configuration, with only one Session and Config
// More complicated configurations can configure multiple sessions and multiple configs w/in a session:
// Session
//     Config
//     ...
//     Config
// ...
// Session
//     Config
//     ...
//     Config
typedef struct
{
    char const * chipName;     //!< string name of target compute device, needed for NVPW calls
    int device;                //!< compute device number
    int maxLaunchesPerPass;    //!< maximum number of kernel launches in any Pass in this Session
    int maxNumRanges;          //!< maximum number of Ranges that may be encountered in this Session (nested Ranges are multiplicative)
    int maxRangeNameLength;    //!< maximum string length for any Range in this Session
    int maxRangesPerPass;      //!< maximum number of Ranges in any Pass in this Session
    int minNestingLevels;      //!< minimum level to tag a Range within this session, must be >= 1
    int numNestingLevels;      //!< maximum level for nested Ranges within this Session, must be >= 1
    CUpti_ProfilerRange rangeMode; //!< CUPTI_AutoRange or CUPTI_UserRange
    CUpti_ProfilerReplayMode replayMode; //!< CUPTI_KernelReplay, CUPTI_UserReplay, or CUPTI_ApplicationReplay
    CUcontext context;         //!< CUDA driver context, or NULL if default context has already been initialized
} profilingConfig;

// Per-device configuration, buffers, stream and device information, and device pointers
typedef struct
{
    int deviceID;
    profilingConfig config;                 // Each device (or each context) needs its own CUPTI profiling config
    vector<uint8_t> counterDataImage;       // As well as its own profiling data images
    vector<uint8_t> counterDataPrefixImage; // ..
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    vector<cudaStream_t> streams;           // Each device needs its own streams
    vector<double *> d_x;                   // And device memory allocation
    vector<double *> d_y;                   // ..
} perDeviceData;

bool explicitlyInitialized = false;

// Call any needed initialization routines for host or target
void explicitInitialization()
{
    if (explicitlyInitialized == false)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // NVPW required initialization
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        explicitlyInitialized = true;
    }
}

/**
 * \brief Initialize Config, Counter Data Prefix, Counter Data, and Counter Data Scratch Buffer images.
 *
 * This should be run on the target.  In particular, counterAvailabilityImage, counterDataImage, and counterDataScratchBuffer
 * must be generated on the target.  ConfigImage and counterDataPrefixImage may be run on the host, but it may be simpler to do
 * all initialization in one place.
 *
 * @param config [in|out] config struct.  All fields except chipName should be provided.  Returns same struct with chipName filled.
 * @param metricNames [in] vector of Perfworks API metric name strings
 * @param configImage [out] Returns initialized Config image
 * @param counterDataPrefixImage [out] Returns initialized Counter Data Prefix image
 * @param counterDataImage [out] Returns initialized Counter Data image
 * @param counterDataScratchBufferImage [out] Returns initiailized Counter Data Scratch Buffer image
 */
void targetInitProfiling(perDeviceData &deviceData,
                         vector<string> const &metricNames)
{
    // Ensure CUPTI Profiling API & NVPW are initialized
    explicitInitialization();

    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = deviceData.config.context;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Allocate sized counterAvailabilityImage
    vector<uint8_t> counterAvailabilityImage;
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);

    // Initialize counterAvailabilityImage
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Get chip name for the CUDA device
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceData.config.device;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    deviceData.config.chipName = strdup(getChipNameParams.pChipName);

    // Fill in configImage - can be run on host or target
    if (!NV::Metric::Config::GetConfigImage(deviceData.config.chipName, metricNames, deviceData.configImage, counterAvailabilityImage.data()))
    {
        cerr << "Failed to create configImage" << endl;
        exit(EXIT_FAILURE);
    }

    // Fill in counterDataPrefixImage - can be run on host or target
    if (!NV::Metric::Config::GetCounterDataPrefixImage(deviceData.config.chipName, metricNames, deviceData.counterDataPrefixImage, counterAvailabilityImage.data()))
    {
        cerr << "Failed to create counterDataPrefixImage" << endl;
        exit(EXIT_FAILURE);
    }

    // Record counterDataPrefixImage info and other options for sizing the counterDataImage
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = deviceData.counterDataPrefixImage.data();
    counterDataImageOptions.counterDataPrefixSize = deviceData.counterDataPrefixImage.size();
    counterDataImageOptions.maxNumRanges = deviceData.config.maxNumRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = deviceData.config.maxNumRanges;
    counterDataImageOptions.maxRangeNameLength = deviceData.config.maxRangeNameLength;

    // Calculate size of counterDataImage based on counterDataPrefixImage and options
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
    // Create counterDataImage
    deviceData.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    // Initialize counterDataImage
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = deviceData.counterDataImage.size();
    initializeParams.pCounterDataImage = deviceData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = deviceData.counterDataImage.size();
    scratchBufferSizeParams.pCounterDataImage = deviceData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    // Create counterDataScratchBuffer
    deviceData.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    // Initialize counterDataScratchBuffer
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = deviceData.counterDataImage.size();
    initScratchBufferParams.pCounterDataImage = deviceData.counterDataImage.data();
    initScratchBufferParams.counterDataScratchBufferSize = deviceData.counterDataScratchBufferImage.size();;
    initScratchBufferParams.pCounterDataScratchBuffer = deviceData.counterDataScratchBufferImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));
}

void startSession(profilingConfig &config,
                  vector<uint8_t> &counterDataImage,
                  vector<uint8_t> &counterDataScratchBuffer,
                  vector<uint8_t> &configImage)
{
    // Ensure CUPTI Profiling API & NVPW are initialized (only needed if not previously initialized)
    explicitInitialization();

    // Start a session
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = counterDataImage.data();
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = counterDataScratchBuffer.data();
    beginSessionParams.ctx = config.context;
    beginSessionParams.maxLaunchesPerPass = config.maxLaunchesPerPass;
    beginSessionParams.maxRangesPerPass = config.maxRangesPerPass;
    beginSessionParams.pPriv = NULL;
    beginSessionParams.range = config.rangeMode;
    beginSessionParams.replayMode = config.replayMode;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = configImage.data();
    setConfigParams.configSize = configImage.size();
    setConfigParams.passIndex = 0; // Only set for Application Replay mode
    setConfigParams.minNestingLevel = config.minNestingLevels;
    setConfigParams.numNestingLevels = config.numNestingLevels;
    setConfigParams.targetNestingLevel = config.minNestingLevels;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
}

// Simple wrappers for Profiler API routines
void beginPass(profilingConfig const &config)
{
    CUpti_Profiler_BeginPass_Params beginPassParams = { CUpti_Profiler_BeginPass_Params_STRUCT_SIZE };
    beginPassParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
}

void enableProfiling(profilingConfig const &config)
{
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    enableProfilingParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}

void pushRange(profilingConfig const &config, char const * rangeName)
{
    CUpti_Profiler_PushRange_Params pushRangeParams = { CUpti_Profiler_PushRange_Params_STRUCT_SIZE };
    pushRangeParams.ctx = config.context;
    pushRangeParams.pRangeName = rangeName;
    pushRangeParams.rangeNameLength = strlen(rangeName);
    CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
}

void popRange(profilingConfig const &config)
{
    CUpti_Profiler_PopRange_Params popRangeParams = { CUpti_Profiler_PopRange_Params_STRUCT_SIZE };
    popRangeParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
}

void disableProfiling(profilingConfig &config)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    disableProfilingParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
}

bool endPass(profilingConfig & config)
{
    CUpti_Profiler_EndPass_Params endPassParams = { CUpti_Profiler_EndPass_Params_STRUCT_SIZE };
    endPassParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
    return endPassParams.allPassesSubmitted;
}

void endSession(profilingConfig &config)
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    unsetConfigParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    endSessionParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
}

void flushData(profilingConfig &config)
{
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = { CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE };
    flushCounterDataParams.ctx = config.context;
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    if (flushCounterDataParams.numRangesDropped != 0 || flushCounterDataParams.numTraceBytesDropped)
    {
        cerr << "WARNING: " << flushCounterDataParams.numTraceBytesDropped << " trace bytes dropped due to full TraceBuffer" << endl;
        cerr << "WARNING: " << flushCounterDataParams.numRangesDropped << " ranges dropped in pass" << endl;
    }
}

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay)
__global__ void daxpyKernel(int elements, double a, double * x, double * y)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels (streams, when running concurrently)
int const numKernels = 4;
int const numStreams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Wrapper which will launch numKernel kernel calls on a single device
// The device streams vector is used to control which stream each call is made on
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used
void profileKernels(perDeviceData &d, char const * const rangeName, bool serial)
{
    // Switch to desired device
    RUNTIME_API_CALL(cudaSetDevice(d.deviceID));
    DRIVER_API_CALL(cuCtxSetCurrent(d.config.context));

    // Use the same pass structure for multiple streams on a this device
    startSession(d.config, d.counterDataImage, d.counterDataScratchBufferImage, d.configImage);
        int numPasses = 0;
        bool lastPass = false;
        // Perform multiple passes if needed to provide all configured metrics
        // Note that in this mode, kernel input data is not restored to initial values before each pass
        do
        {
            beginPass(d.config);
            numPasses++;
            enableProfiling(d.config);

            // Then, time launching same amount of work in separate streams (or default stream if serial).
            // cuptiProfilerPushRange and PopRange will serialize the kernel launches, so keep the calls outside the concurrent stream launch loop
            pushRange(d.config, rangeName);
                for (unsigned int stream = 0; stream < d.streams.size(); stream++)
                {
                    cudaStream_t streamId = (serial ? 0 : d.streams[stream]);
                    daxpyKernel <<<threadBlocks, threadsPerBlock, 0, streamId>>> (elements[stream], a, d.d_x[stream], d.d_y[stream]);
                }

                // After launching all work, synchronize all streams
                if (serial == false)
                {
                    for (unsigned int stream = 0; stream < d.streams.size(); stream++)
                    {
                        RUNTIME_API_CALL(cudaStreamSynchronize(d.streams[stream]));
                    }
                }
                else
                {
                    RUNTIME_API_CALL(cudaStreamSynchronize(0));
                }
            popRange(d.config);

            disableProfiling(d.config);
            lastPass = endPass(d.config);
        } while (lastPass == false);

        // Flush is required to ensure data is returned from device when running User Replay mode
        flushData(d.config);
    endSession(d.config);
}


int main(int argc, char * argv[])
{
    // These two metrics will demonstrate whether kernels within a Range were run serially or concurrently
    vector<string> metricNames;
    metricNames.push_back("sm__cycles_active.sum");
    metricNames.push_back("sm__cycles_elapsed.max");
    // This metric shows that the same number of flops were executed on each run
    metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum");

    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information
    vector<int> device_ids;

    // Initialize profiler API support before testing device compatibility
    explicitInitialization();

    // Find all devices capable of running CUPTI Profiling
    for (int i = 0; i < numDevices; i++)
    {
        // Get device compatibility
        CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
        params.cuDevice = i;
        CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));
        if (params.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
        {
            // Record device number
            device_ids.push_back(i);
        }
        else
        {
            cerr << "Unable to profile on device " << i << ":" << endl;

            if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                cerr << "\tDevice architecture is not supported" << endl;
            }

            if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                cerr << "\tDevice SLI configuration is not supported" << endl;
            }

            if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                cerr << "\tDevice VGPU configuration is not supported" << endl;
            }
            else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
            {
                cerr << "\tDevice VGPU configuration disabled profiling support" << endl;
            }

            if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                cerr << "\tDevice Confidential Compute configuration is not supported" << endl;
            }

            if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << endl;
            }
        }
    }

    numDevices = device_ids.size();
    cout << "Found " << numDevices << " compatible devices" << endl;

    // Ensure we found at least one device
    if (numDevices == 0)
    {
        cerr << "No devices detected compatible with CUPTI Profiling" << endl;
        exit(EXIT_WAIVED);
    }

    // Initialize kernel input to some known numbers
    vector<double> h_x(blockSize * numKernels);
    vector<double> h_y(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels
    vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++)
    {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number
    for (int stream = 0; stream < numStreams; stream++)
    {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data
    vector<perDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        int device_id = device_ids[device];
        RUNTIME_API_CALL(cudaSetDevice(device_id));
        cout << "Configuring device " << device_id << endl;

        // Required CUPTI Profiling configuration & initialization
        // Can be done ahead of time or immediately before startSession() call
        // Initialization & configuration images can be generated separately, then passed to later calls
        // For simplicity's sake, in this sample, a single config struct is created per device and passed to each CUPTI Profiler API call
        // For more complex cases, each combination of CUPTI Profiler Session and Config requires additional initialization
        profilingConfig config;
        config.device = device_id;         // Device ID, used to get device name for metrics enumeration
        config.maxLaunchesPerPass = 1;     // Must be >= maxRangesPerPass.  Set this to the largest count of kernel launches which may be encountered in any Pass in this Session

        // Device 0 has max of 3 passes; other devices only run one pass in this sample code
        if (device == 0)
        {
            config.maxNumRanges = 3;       // Maximum number of ranges that may be profiled in the current Session
        }
        else
        {
            config.maxNumRanges = 1;       // Maximum number of ranges that may be profiled in the current Session
        }

        config.maxRangeNameLength = 64;    // Max length including NULL terminator of any range name
        config.maxRangesPerPass = 1;       // Max ranges that can be recorded in any Pass in this Session
        config.minNestingLevels = 1;       // Must be >= 1, minimum reported nest level for Ranges in this Session
        config.numNestingLevels = 1;       // Must be >= 1, max height of nested Ranges in this Session
        config.rangeMode = CUPTI_UserRange;// CUPTI_AutoRange or CUPTI_UserRange
        config.replayMode = CUPTI_UserReplay; // CUPTI_KernelReplay, CUPTI_UserReplay, or CUPTI_ApplicationReplay
        DRIVER_API_CALL(cuCtxCreate(&(config.context), 0, device)); // Either set to a context, or may be NULL if a default context has been created
        deviceData[device].config = config;// Save this device config

        // Initialize CUPTI Profiling structures
        targetInitProfiling(deviceData[device], metricNames);

        // Per-stream initialization & memory allocation - copy from constant host array to each device array
        deviceData[device].streams.resize(numStreams);
        deviceData[device].d_x.resize(numStreams);
        deviceData[device].d_y.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_x[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_x[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_y[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_y[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_y[stream], h_x.data(), size, cudaMemcpyHostToDevice));
        }
    }

    //
    // First version - single device, kernel calls serialized on default stream
    //

    // Use wallclock time to measure performance
    auto begin_time = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams, which run serially
    profileKernels(deviceData[0], "single_device_serial", true);

    auto end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    int numBlocks = 0;
    for (int i = 1; i <= numKernels; i++)
    {
        numBlocks += i;
    }
    cout << "It took " << elapsed_serial_ms.count() << "ms on the host to profile " << numKernels << " kernels in serial." << endl;

    //
    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency
    // (Should be limited by the longest running kernel)
    //

    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism
    profileKernels(deviceData[0], "single_device_async", false);

    end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    cout << "It took " << elapsed_single_device_ms.count() << "ms on the host to profile " << numKernels << " kernels on a single device on separate streams." << endl;
    cout << "--> If the separate stream wallclock time is less than the serial version, the streams were profiling concurrently." << endl;

    //
    // Third version - same as the second case, but duplicates the concurrent work across devices to show cross-device concurrency
    // This is done using devices so no serialization is needed between devices
    // (Should have roughly the same wallclock time as second case if the devices have similar performance)
    //

    if (numDevices == 1)
    {
        cout << "Only one compatible device found; skipping the multi-threaded test." << endl;
    }
    else
    {
        cout << "Running on " << numDevices << " devices, one thread per device." << endl;

        // Time creation of the same multiple streams (on multiple devices, if possible)
        vector<::std::thread> threads;
        begin_time = ::std::chrono::high_resolution_clock::now();

        // Now launch parallel thread work, duplicated on one thread per device
        for (int thread = 0; thread < numDevices; thread++)
        {
            threads.push_back(::std::thread(profileKernels, ::std::ref(deviceData[thread]), "multi_device_async", false));
        }

        // Wait for all threads to finish
        for (auto &t: threads)
        {
            t.join();
        }

        // Record time used when launching on multiple devices
        end_time = ::std::chrono::high_resolution_clock::now();
        auto elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
        cout << "It took " << elapsed_multiple_device_ms.count() << "ms on the host to profile the same " << numKernels << " kernels on each of the " << numDevices << " devices in parallel" << endl;
        cout << "--> Wallclock ratio of parallel device launch to single device launch is " << elapsed_multiple_device_ms.count() / static_cast<double>(elapsed_single_device_ms.count()) << endl;
        cout << "--> If the ratio is close to 1, that means there was little overhead to profile in parallel on multiple devices compared to profiling on a single device." << endl;
        cout << "--> If the devices have different performance, the ratio may not be close to one, and this should be limited by the slowest device." << endl;
    }

    // Free stream memory for each device
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_x[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_y[j]));
        }
    }

    // Display metric values
    cout << endl << "Metrics for device #0:" << endl;
    cout << "Look at the sm__cycles_elapsed.max values for each test." << endl;
    cout << "This value represents the time spent on device to run the kernels in each case, and should be longest for the serial range, and roughly equal for the single and multi device concurrent ranges." << endl;
    PrintMetricValues(deviceData[0].config.chipName, deviceData[0].counterDataImage, metricNames);

    // Only display next device info if needed
    if (numDevices > 1)
    {
        cout << endl << "Metrics for the remaining devices only display the multi device async case and should all be similar to the first device's values if the device has similar performance characteristics." << endl;
        cout << "If devices have different performance characteristics, the runtime cycles calculation may vary by device." << endl;
    }
    for (int i = 1; i < numDevices; i++)
    {
        cout << endl << "Metrics for device #" << i << ":" << endl;
        PrintMetricValues(deviceData[i].config.chipName, deviceData[i].counterDataImage, metricNames);
    }

    exit(EXIT_SUCCESS);
}
