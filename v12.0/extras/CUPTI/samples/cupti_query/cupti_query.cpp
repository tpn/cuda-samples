/*
 * Copyright 2010-2019 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to query domains and events supported by device
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32)
#define stricmp _stricmp
#else
#define stricmp strcasecmp
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define NAME_SHORT  64
#define NAME_LONG   128

#define DESC_SHORT  512
#define DESC_LONG   2048
#define CATEGORY_LENGTH  sizeof(CUpti_EventCategory)

static unsigned int optionsFlag = 0;

#define setOptionsFlag(bit)     (optionsFlag |= (1<<bit))
#define isOptionsFlagSet(bit)   ((optionsFlag & (1<<bit)) ? 1 : 0)

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("%s:%d:Error %d for CUDA Driver API function '%s'.\n",    \
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
      exit(EXIT_FAILURE);                                                \
    }

enum enum_options {
    FLAG_DEVICE_ID = 0,
    FLAG_DOMAIN_ID,
    FLAG_GET_DOMAINS,
    FLAG_GET_EVENTS,
    FLAG_GET_METRICS
};

typedef struct ptiDomainData_st {
    CUpti_EventDomainID domainId;       // domain id
    char domainName[NAME_SHORT];        // domain name
    uint32_t profiledInstanceCnt;       // number of domain instances (profiled)
    uint32_t totalInstanceCnt;          // number of domain instances (total)
    CUpti_EventCollectionMethod eventCollectionMethod;
}ptiDomainData;

typedef union {
    CUpti_EventID eventId;          // event id
    CUpti_MetricID metricId;        //metric id
}cuptiId;

typedef struct ptiEventData_st {
    cuptiId Id;
    char eventName[NAME_SHORT];         // event name
    char shortDesc[DESC_SHORT];         // short desc of the event
    char longDesc[DESC_LONG];           // long desc of the event
    CUpti_EventCategory  category;      // category of the event
}ptiData;

static void printUsage() {
    printf("usage: cuptiQuery\n");
    printf("       -help                                            : displays help message\n");
    printf("       -device <dev_id> -getdomains                     : displays supported domains for specified device\n");
    printf("       -device <dev_id> -getmetrics                     : displays supported metrics for specified device\n");
    printf("       -device <dev_id> -domain <domain_id> -getevents  : displays supported events for specified domain and device\n");
    printf("Note: default device is 0 and default domain is first domain for device\n");
}

// add a null terminator to the end of a string if the string
// length equals the maximum length (as in that case there was no
// room to write the null terminator)
static void checkNullTerminator(char *str, size_t len, size_t max_len) {
    if (len >= max_len) {
        str[max_len - 1] = '\0';
    }
}

int enumEventDomains(CUdevice dev) {
    CUptiResult ptiStatus = CUPTI_SUCCESS;
    CUpti_EventDomainID *domainId = NULL;
    ptiDomainData domainData;
    uint32_t maxDomains = 0, i = 0;
    size_t size = 0;

    ptiStatus = cuptiDeviceGetNumEventDomains(dev, &maxDomains);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceGetNumEventDomains");

    if (maxDomains == 0) {
        printf("No domain is exposed by dev = %d\n", dev);
        ptiStatus = CUPTI_ERROR_UNKNOWN;
        goto Exit;
    }

    size = sizeof(CUpti_EventDomainID) * maxDomains;
    domainId = (CUpti_EventDomainID*)malloc(size);
    if (domainId == NULL) {
        printf("Failed to allocate memory to domain ID\n");
        ptiStatus = CUPTI_ERROR_OUT_OF_MEMORY;
        goto Exit;
    }
    memset(domainId, 0, size);

    ptiStatus = cuptiDeviceEnumEventDomains(dev, &size, domainId);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceEnumEventDomains");

    // enum domains
    for (i = 0; i < maxDomains; i++) {
        domainData.domainId = domainId[i];
        // query domain name
        size = NAME_SHORT;
        ptiStatus = cuptiEventDomainGetAttribute(domainData.domainId,
                                                 CUPTI_EVENT_DOMAIN_ATTR_NAME,
                                                 &size,
                                                 (void*)domainData.domainName);
        checkNullTerminator(domainData.domainName, size, NAME_SHORT);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventDomainGetAttribute");

        // query num of profiled instances in the domain
        size = sizeof(domainData.profiledInstanceCnt);
        ptiStatus = cuptiDeviceGetEventDomainAttribute(dev,
                                                 domainData.domainId,
                                                 CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT,
                                                 &size,
                                                 (void*)&domainData.profiledInstanceCnt);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceEventDomainGetAttribute");

        // query total instances in the domain
        size = sizeof(domainData.totalInstanceCnt);
        ptiStatus = cuptiDeviceGetEventDomainAttribute(dev,
                                                 domainData.domainId,
                                                 CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                 &size,
                                                 (void*)&domainData.totalInstanceCnt);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceEventDomainGetAttribute");

        size = sizeof(CUpti_EventCollectionMethod);
        ptiStatus = cuptiEventDomainGetAttribute(domainData.domainId,
                                                 CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD,
                                                 &size,
                                                 (void*)&domainData.eventCollectionMethod);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventDomainGetAttribute");

        printf ("Domain# %u\n", i+1);
        printf ("Id         = %d\n",   domainData.domainId);
        printf ("Name       = %s\n",   domainData.domainName);
        printf ("Profiled instance count = %u\n", domainData.profiledInstanceCnt);
        printf ("Total instance count = %u\n", domainData.totalInstanceCnt);

        printf ("Event collection method = ");
        switch(domainData.eventCollectionMethod)
        {
            case CUPTI_EVENT_COLLECTION_METHOD_PM:
                printf("CUPTI_EVENT_COLLECTION_METHOD_PM\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_SM:
                printf("CUPTI_EVENT_COLLECTION_METHOD_SM\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED:
                printf("CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC:
                printf("CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC\n");
                break;
            default:
                printf("\nError: Invalid event collection method!\n");
                return -1;
        }
    }

Exit:
    if (domainId) {
        free (domainId);
    }

    if (ptiStatus == CUPTI_SUCCESS) {
        return 0;
    }
    else {
        return -1;
    }
}

int enumEvents(CUpti_EventDomainID domainId) {
    ptiData eventData;
    CUptiResult ptiStatus = CUPTI_SUCCESS;
    CUpti_EventID *eventId = NULL;
    uint32_t maxEvents = 0;
    uint32_t i = 0;
    size_t size = 0;

    // query num of events available in the domain
    ptiStatus = cuptiEventDomainGetNumEvents(domainId,
                                             &maxEvents);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventDomainGetNumEvents");

    size = sizeof(CUpti_EventID) * maxEvents;
    eventId = (CUpti_EventID*)malloc(size);
    if (eventId == NULL) {
        printf("Failed to allocate memory to event ID\n");
        ptiStatus = CUPTI_ERROR_OUT_OF_MEMORY;
        goto Exit;
    }
    memset(eventId, 0, size);

    ptiStatus = cuptiEventDomainEnumEvents(domainId,
                                           &size,
                                           eventId);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventDomainEnumEvents");

    // query event info
    for (i = 0; i < maxEvents; i++) {
        eventData.Id.eventId = eventId[i];

        size = NAME_SHORT;
        ptiStatus = cuptiEventGetAttribute(eventData.Id.eventId,
                                           CUPTI_EVENT_ATTR_NAME,
                                           &size,
                                           (uint8_t*)eventData.eventName);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(eventData.eventName, size, NAME_SHORT);

        size = DESC_SHORT;
        ptiStatus = cuptiEventGetAttribute(eventData.Id.eventId,
                                           CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
                                           &size,
                                           (uint8_t*)eventData.shortDesc);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(eventData.shortDesc, size, DESC_SHORT);

        size = DESC_LONG;
        ptiStatus = cuptiEventGetAttribute(eventData.Id.eventId,
                                           CUPTI_EVENT_ATTR_LONG_DESCRIPTION,
                                           &size,
                                           (uint8_t*)eventData.longDesc);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(eventData.longDesc, size, DESC_LONG);

        size = CATEGORY_LENGTH;
        ptiStatus = cuptiEventGetAttribute(eventData.Id.eventId,
                                           CUPTI_EVENT_ATTR_CATEGORY,
                                           &size,
                                           (&eventData.category));
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");

        printf("Event# %u\n", i+1);
        printf("Id        = %d\n", eventData.Id.eventId);
        printf("Name      = %s\n", eventData.eventName);
        printf("Shortdesc = %s\n", eventData.shortDesc);
        printf("Longdesc  = %s\n", eventData.longDesc);

        switch(eventData.category)
        {
            case CUPTI_EVENT_CATEGORY_INSTRUCTION:
                printf("Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_MEMORY:
                printf("Category  = CUPTI_EVENT_CATEGORY_MEMORY\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_CACHE:
                printf("Category  = CUPTI_EVENT_CATEGORY_CACHE\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER:
                printf("Category  = CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_SYSTEM:
                printf("Category  = CUPTI_EVENT_CATEGORY_SYSTEM\n\n");
                break;
            default:
                printf("\n Invalid category!\n");
        }

    }

Exit:
    if (eventId) {
        free (eventId);
    }

    if (ptiStatus == CUPTI_SUCCESS) {
        return 0;
    }
    else {
        return -1;
    }
}

int enumMetrics(CUdevice dev) {
    ptiData metricData;
    CUptiResult ptiStatus = CUPTI_SUCCESS;
    CUpti_MetricID *metricId = NULL;
    uint32_t maxMetrics = 0;
    uint32_t i = 0;
    size_t size = 0;

    ptiStatus = cuptiDeviceGetNumMetrics(dev, &maxMetrics);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceGetNumMetrics");

    size = sizeof(CUpti_EventID) * maxMetrics;
    metricId = (CUpti_MetricID*)malloc(size);
    if (metricId == NULL) {
        printf("Failed to allocate memory to metric ID\n");
        ptiStatus = CUPTI_ERROR_OUT_OF_MEMORY;
        goto Exit;
    }
    memset(metricId, 0, size);

    ptiStatus = cuptiDeviceEnumMetrics(dev, &size, metricId);
    CHECK_CUPTI_ERROR(ptiStatus, "cuptiDeviceEnumMetrics");

    // query metric info
    for (i = 0; i < maxMetrics; i++) {
        metricData.Id.metricId = metricId[i];

        size = NAME_SHORT;
        ptiStatus = cuptiMetricGetAttribute(metricData.Id.metricId,
                                           CUPTI_METRIC_ATTR_NAME,
                                           &size,
                                           (uint8_t*)metricData.eventName);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(metricData.eventName, size, NAME_SHORT);

        size = DESC_SHORT;
        ptiStatus = cuptiMetricGetAttribute(metricData.Id.metricId,
                                           CUPTI_METRIC_ATTR_SHORT_DESCRIPTION,
                                           &size,
                                           (uint8_t*)metricData.shortDesc);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(metricData.shortDesc, size, DESC_SHORT);

        size = DESC_LONG;
        ptiStatus = cuptiMetricGetAttribute(metricData.Id.metricId,
                                           CUPTI_METRIC_ATTR_LONG_DESCRIPTION,
                                           &size,
                                           (uint8_t*)metricData.longDesc);
        CHECK_CUPTI_ERROR(ptiStatus, "cuptiEventGetAttribute");
        checkNullTerminator(metricData.longDesc, size, DESC_LONG);

        printf("Metric# %u\n", i+1);
        printf("Id        = %d\n", metricData.Id.metricId);
        printf("Name      = %s\n", metricData.eventName);
        printf("Shortdesc = %s\n", metricData.shortDesc);
        printf("Longdesc  = %s\n\n", metricData.longDesc);
    }

Exit:
    if (metricId) {
        free (metricId);
    }

    if (ptiStatus == CUPTI_SUCCESS) {
        return 0;
    }
    else {
        return -1;
    }
}

void parseCommandLineArgs(int argc, char *argv[], int &deviceId, CUpti_EventDomainID &domainId)
{
    for(int k=1; k<argc; k++) {
        if ((k+1 < argc) && stricmp(argv[k], "-device") == 0) {
            deviceId = atoi(argv[k+1]);
            setOptionsFlag(FLAG_DEVICE_ID);
            k++;
        }
        else if ((k+1 < argc) && stricmp(argv[k], "-domain") == 0) {
            domainId = (CUpti_EventDomainID)atoi(argv[k+1]);
            setOptionsFlag(FLAG_DOMAIN_ID);
            k++;
        }
        else if ((k < argc) && stricmp(argv[k], "-getdomains") == 0) {
            setOptionsFlag(FLAG_GET_DOMAINS);
        }
        else if (stricmp(argv[k], "-getevents") == 0) {
            setOptionsFlag(FLAG_GET_EVENTS);
        }
        else if (stricmp(argv[k], "-getmetrics") == 0) {
            setOptionsFlag(FLAG_GET_METRICS);
        }
        else if ((stricmp(argv[k], "--help") == 0) ||
                 (stricmp(argv[k], "-help") == 0) ||
                 (stricmp(argv[k], "-h") == 0)) {
            printUsage();
            exit(EXIT_SUCCESS);
        }
        else {
            printf("Invalid/incomplete option %s\n", argv[k]);
        }
    }
}

int main(int argc, char *argv[])
{
    CUdevice dev;
    CUresult err;
    CUptiResult cuptiErr = CUPTI_SUCCESS;
    int ret = 0;
    int deviceId = 0;
    int deviceCount = 0;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    char deviceName[256];
    CUpti_EventDomainID domainId = 0;
    size_t size = 0;

    err = cuInit(0);
    CHECK_CU_ERROR(err, "cuInit");

    err = cuDeviceGetCount(&deviceCount);
    CHECK_CU_ERROR(err, "cuDeviceGetCount");

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        ret = -2;
        goto Exit;
    }

    // parse command line arguments
    parseCommandLineArgs(argc, argv, deviceId, domainId);

    if (!isOptionsFlagSet(FLAG_DEVICE_ID)) {
        // default is device 0
        printf("Assuming default device id 0\n");
        deviceId = 0;
    }

    // show events if no explicit flag is set
    if (!isOptionsFlagSet(FLAG_GET_DOMAINS) &&
        !isOptionsFlagSet(FLAG_GET_EVENTS) &&
        !isOptionsFlagSet(FLAG_GET_METRICS)) {
        setOptionsFlag(FLAG_GET_EVENTS);
    }

    err = cuDeviceGet(&dev, deviceId);
    if (err == CUDA_ERROR_INVALID_DEVICE) {
        printf("Device (%d) out of range\n", deviceId);
        ret=-2;
        goto Exit;
    }
    else {
        CHECK_CU_ERROR(err, "cuDeviceGet");
    }

    err = cuDeviceGetName(deviceName, 256, dev);
    CHECK_CU_ERROR(err, "cuDeviceGetName");

    printf("CUDA Device Id  : %d\n", deviceId);
    printf("CUDA Device Name: %s\n\n", deviceName);

    err = cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

    err = cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    CHECK_CU_ERROR(err, "cuDeviceGetAttribute");

    if (isOptionsFlagSet(FLAG_GET_DOMAINS)) {
        if (enumEventDomains(dev)) {
            printf("enumEventDomains failed\n");
            ret = -1;
            goto Exit;
        }
    }
    else if (isOptionsFlagSet(FLAG_GET_EVENTS)) {
        if (!isOptionsFlagSet(FLAG_DOMAIN_ID)) {
            // query first domain on the device
            size = sizeof(CUpti_EventDomainID);
            cuptiErr = cuptiDeviceEnumEventDomains(dev, &size, (CUpti_EventDomainID *)&domainId);
            CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceEnumEventDomains");

            printf("Assuming default domain id %d\n", domainId);
        }
        else {
            // validate the domain on the device
            CUpti_EventDomainID *domainIdArr = NULL;
            uint32_t maxDomains = 0, i = 0;

            cuptiErr = cuptiDeviceGetNumEventDomains(dev, &maxDomains);
            CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetNumEventDomains");

            if (maxDomains == 0) {
                printf("No domain is exposed by dev = %d\n", dev);
                cuptiErr = CUPTI_ERROR_UNKNOWN;
                ret = -2;
                goto Exit;
            }

            size = sizeof(CUpti_EventDomainID) * maxDomains;
            domainIdArr = (CUpti_EventDomainID*)malloc(size);
            if (domainIdArr == NULL) {
                printf("Failed to allocate memory to domain ID\n");
                cuptiErr = CUPTI_ERROR_OUT_OF_MEMORY;
                ret = -1;
                goto Exit;
            }
            memset(domainIdArr, 0, size);

            // enum domains
            cuptiErr = cuptiDeviceEnumEventDomains(dev, &size, domainIdArr);
            CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceEnumEventDomains");

            for (i = 0; i < maxDomains; i++) {
                if (domainIdArr[i] == domainId) {
                    break;
                }
            }
            free (domainIdArr);

            if (i == maxDomains) {
                printf("Domain Id %d is not supported by device\n", domainId);
                ret = -2;
                goto Exit;
            }
        }

        if (enumEvents(domainId)) {
            printf("enumEvents failed\n");
            ret = -1;
            goto Exit;
        }
    }
    else if (isOptionsFlagSet(FLAG_GET_METRICS)) {
        if(enumMetrics(dev)) {
            printf("enumMetrics failed\n");
            ret = -1;
            goto Exit;
        }
    }

Exit:
    cudaDeviceSynchronize();
     if(ret == -1) {
        exit(EXIT_FAILURE);
    }
    else if (ret ==-2) {
        exit(EXIT_WAIVED);
    }
    else {
        exit(EXIT_SUCCESS);
    }
}
