/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This example demonstrates how to use the Video Decode Library with CUDA
 * bindings to interop between NVCUVID(CUDA) and OpenGL (PBOs).  Post-Processing
 * video (de-interlacing) is supported with this sample.
 */

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA Header includes
#include <cuda.h>
#include <cudaGL.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_cuda_gl.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>

// cudaDecodeGL related helper functions
#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageGL.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

#if !defined(WIN32) && !defined(_WIN32) && !defined(WIN64) && !defined(_WIN64)
typedef unsigned char BYTE;
#define S_OK true;
#endif

const char *sAppName     = "CUDA/OpenGL Video Decode";
const char *sAppFilename = "cudaDecodeGL";
const char *sSDKname     = "cudaDecodeGL";

#define VIDEO_SOURCE_FILE "plush1_720p_10s.m2v"

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    0
#else
#define ENABLE_DEBUG_OUT    0
#endif

StopWatchInterface *frame_timer = NULL,
                    *global_timer = NULL;

int                 g_DeviceID    = 0;
bool                g_bWindowed   = true;
bool                g_bDeviceLost = false;
bool                g_bDone       = false;
bool                g_bRunning    = false;
bool                g_bAutoQuit   = false;
bool                g_bUseVsync   = false;
bool                g_bFrameRepeat= false;
bool                g_bFrameStep  = false;
bool                g_bQAReadback = false;
bool                g_bGLVerify   = false;
bool                g_bFirstFrame = true;
bool                g_bLoop       = false;
bool                g_bUpdateCSC  = true;
bool                g_bUpdateAll  = false;
bool                g_bLinearFiltering = false;
bool                g_bUseDisplay = false; // this flag enables/disables video on the window
bool                g_bUseInterop = true;
bool                g_bReadback   = false;
bool                g_bIsProgressive = true; // assume it is progressive, unless otherwise noted
bool                g_bException = false;
bool                g_bWaived     = false;

int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats

int *pArgc = NULL;
char **pArgv = NULL;

cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
CUvideoctxlock       g_CtxLock = NULL;

float present_fps, decoded_fps, total_time = 0.0f;

// These are CUDA function pointers to the CUDA kernels
CUmoduleManager   *g_pCudaModule;

CUmodule           cuModNV12toARGB       = 0;
CUfunction         g_kernelNV12toARGB    = 0;
CUfunction         g_kernelPassThru      = 0;

CUcontext          g_oContext = 0;
CUdevice           g_oDevice  = 0;

CUstream           g_ReadbackSID = 0, g_KernelSID = 0;

eColorSpace        g_eColorSpace = ITU601;
float              g_nHue        = 0.0f;

// System Memory surface we want to readback to
BYTE          *g_bFrameData[2] = { 0, 0 };
FrameQueue    *g_pFrameQueue   = 0;
VideoSource   *g_pVideoSource  = 0;
VideoParser   *g_pVideoParser  = 0;
VideoDecoder *g_pVideoDecoder = 0;

ImageGL       *g_pImageGL      = 0; // if we're using OpenGL
CUdeviceptr    g_pInteropFrame[2] = { 0, 0 }; // if we're using CUDA malloc

std::string sFileName;

char exec_path[256];

unsigned int g_nWindowWidth  = 0;
unsigned int g_nWindowHeight = 0;

unsigned int g_nClientAreaWidth  = 0;
unsigned int g_nClientAreaHeight = 0;

unsigned int g_nVideoWidth  = 0;
unsigned int g_nVideoHeight = 0;

unsigned int g_FrameCount = 0;
unsigned int g_DecodeFrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;

// Forward declarations
bool initGL(int argc, char **argv, int *pbTCC);
bool initGLTexture(unsigned int nWidth, unsigned int nHeight);
bool loadVideoSource(const char *video_file,
                     unsigned int &width, unsigned int &height,
                     unsigned int &dispWidth, unsigned int &dispHeight);
void initCudaVideo();

void freeCudaResources(bool bDestroyContext);

bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive);
void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
                          CUdeviceptr *ppInteropFrame, size_t pFramePitch,
                          CUmodule cuModNV12toARGB,
                          CUfunction fpCudaKernel, CUstream streamID);
bool drawScene(int field_num);
bool cleanup(bool bDestroyContext);
bool initCudaResources(int argc, char **argv, int *bTCC);

bool renderVideoFrame(int bUseInterop);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef bool (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);
PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;

// This allows us to turn off vsync for OpenGL
void setVSync(int interval)
{
    const GLubyte *extensions = glGetString(GL_EXTENSIONS);

    if (strstr((const char *)extensions, "WGL_EXT_swap_control") == 0)
    {
        return;    // Error: WGL_EXT_swap_control extension not supported on your computer.\n");
    }
    else
    {
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");

        if (wglSwapIntervalEXT)
        {
            wglSwapIntervalEXT(interval);
        }
    }
}

#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#else
void setVSync(int interval)
{
}

#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#endif

void printStatistics()
{
    int   hh, mm, ss, msec;

    present_fps = 1.f / (total_time / (g_FrameCount * 1000.f));
    decoded_fps = 1.f / (total_time / (g_DecodeFrameCount * 1000.f));

    msec = ((int)total_time % 1000);
    ss   = (int)(total_time/1000) % 60;
    mm   = (int)(total_time/(1000*60)) % 60;
    hh   = (int)(total_time/(1000*60*60)) % 60;

    printf("\n[%s] statistics\n", sAppFilename);
    printf("\t Video Length (hh:mm:ss.msec)   = %02d:%02d:%02d.%03d\n", hh, mm, ss, msec);

    printf("\t Frames Presented (inc repeats) = %d\n", g_FrameCount);
    printf("\t Average Present Rate     (fps) = %4.2f\n", present_fps);

    printf("\t Frames Decoded   (hardware)    = %d\n", g_DecodeFrameCount);
    printf("\t Average Rate of Decoding (fps) = %4.2f\n", decoded_fps);
}

void computeFPS(int bUseInterop)
{
    sdkStopTimer(&frame_timer);

    if (g_bRunning)
    {
        g_fpsCount++;
    }

    char sFPS[256];
    std::string sDecodeStatus;

    if (g_bDeviceLost)
    {
        sDecodeStatus = "DeviceLost!\0";
        sprintf(sFPS, "%s [%s] - [%s %d]",
                sSDKname, sDecodeStatus.c_str(),
                (g_bIsProgressive ? "Frame" : "Field"),
                g_DecodeFrameCount);

        if (!g_bQAReadback || g_bGLVerify)
        {
            glutSetWindowTitle(sFPS);
        }

        sdkResetTimer(&frame_timer);
        g_fpsCount = 0;
        return;
    }

    if (!g_pFrameQueue->isDecodeFinished() && !g_bRunning)
    {
        sDecodeStatus = "PAUSE\0";
        sprintf(sFPS, "%s [%s] - [%s %d] - Video Display %s / Vsync %s",
                sAppFilename, sDecodeStatus.c_str(),
                (g_bIsProgressive ? "Frame" : "Field"),
                g_DecodeFrameCount,
                g_bUseDisplay ? "ON" : "OFF",
                g_bUseVsync   ? "ON" : "OFF");

        if (bUseInterop && (!g_bQAReadback || g_bGLVerify))
        {
            glutSetWindowTitle(sFPS);
        }

    }
    else
    {
        if (g_bFrameStep)
        {
            sDecodeStatus = "STEP\0";
        }
        else
        {
            sDecodeStatus = "PLAY\0";
        }
    }

    if (g_fpsCount == g_fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

        sprintf(sFPS, "%s [%s] - [%3.1f fps, %s %d] - Video Display %s / Vsync %s",
                sAppFilename, sDecodeStatus.c_str(), ifps,
                (g_bIsProgressive ? "Frame" : "Field"),
                g_DecodeFrameCount,
                g_bUseDisplay ? "ON" : "OFF",
                g_bUseVsync   ? "ON" : "OFF");

        if (bUseInterop)
        {
            glutSetWindowTitle(sFPS);
        }

        printf("[%s] - [%s: %04d, %04.1f fps, frame time: %04.2f (ms) ]\n",
               sAppFilename,
               (g_bIsProgressive ? "Frame" : "Field"),
               g_FrameCount, ifps, 1000.f/ifps);

        sdkResetTimer(&frame_timer);
        g_fpsCount = 0;
    }

    if ((g_bDone && g_bAutoQuit) || g_pFrameQueue->isDecodeFinished())
    {
        sDecodeStatus = "STOP (End of File)\0";

        // we only want to record this once
        if (total_time == 0.0f)
        {
            total_time = sdkGetTimerValue(&global_timer);
        }

        sdkStopTimer(&global_timer);
    }
    else
    {
        sdkStartTimer(&frame_timer);
    }
}

bool initCudaResources(int argc, char **argv, int *bTCC)
{
    printf("\n");

    for (int i=0; i < argc; i++)
    {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    printf("\n");

    CUdevice cuda_device;

    // Device is specified at the command line, we need to check if this it TCC or not, and then call the
    // appropriate TCC/non-TCC findCudaDevice in order to initialize the CUDA device
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        cuda_device = getCmdLineArgumentInt(argc, (const char **) argv, "device");

        cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
        checkCudaErrors(cuDeviceGetAttribute(bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, cuda_device));

        if (*bTCC)
        {
            char name[100];
            cuDeviceGetName(name, 100, cuda_device);
            printf("CUDA device=%d [%s] is a TCC Compute device\n", cuda_device, name);
            g_bUseInterop = false;
        }
        else
        {
            if (g_bUseInterop)
            {
                initGL(argc, argv, bTCC);
                cuda_device = findCudaGLDeviceDRV(argc, (const char **)argv);
            }
            else
            {
                cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
            }
        }

        if (cuda_device < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }

        checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
    }
    else
    {
        // If we want to use Graphics Interop, then choose the GPU that is capable
        if (g_bUseInterop && !(*bTCC))
        {
            initGL(argc, argv, bTCC);
            // If we have found only TCC devices while trying to init OpenGL
            // don't try to use OpenGL
            if (!(*bTCC))
            {
                cuda_device = findCudaGLDeviceDRV(argc, (const char **)argv);
            }
            else
            {
                cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
                g_bUseInterop = false;
            }
        }
        else
        {
            cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
            g_bUseInterop = false;
        }
        checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
    }

    // get compute capabilities and the devicename
    int major, minor;
    size_t totalGlobalMem;

    char deviceName[256];
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
    printf("> Using GPU Device: %s has SM %d.%d compute capability\n", deviceName, major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
    printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem/(1024*1024));



    // Create CUDA Device w/ GL interop (if not TCC), otherwise CUDA w/o interop (if TCC)
    // (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
    if (g_bUseInterop && !(*bTCC))
    {
        checkCudaErrors(cuGLCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }
    else
    {
        checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }

    // Initialize CUDA related Driver API
    // Determine if we are running on a 32-bit or 64-bit OS and choose the right PTX file
    try
    {
        if (sizeof(void *) == 4)
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi32.ptx", exec_path, 2, 2, 2);
        }
        else
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi64.ptx", exec_path, 2, 2, 2);
        }
    }
    catch (char const *p_file)
    {
        // If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
        printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
        printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
        return false;
    }


    g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi",   &g_kernelNV12toARGB);
    g_pCudaModule->GetCudaFunction("Passthru_drvapi",     &g_kernelPassThru);

    /////////////////Change///////////////////////////
    // Now we create the CUDA resources and the CUDA decoder context
    initCudaVideo();

    if (g_bUseInterop)
    {
        initGLTexture(g_pVideoDecoder->targetWidth(),
                      g_pVideoDecoder->targetHeight());
    }
    else
    {
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[0], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[1], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 2));
    }

    CUcontext cuCurrent = NULL;
    CUresult result = cuCtxPopCurrent(&cuCurrent);

    if (result != CUDA_SUCCESS)
    {
        printf("cuCtxPopCurrent: %d\n", result);
        assert(0);
    }

    /////////////////////////////////////////
    return ((g_pCudaModule && g_pVideoDecoder) ? true : false);
}

bool reinitCudaResources()
{
    // Free resources
    cleanup(false);

    // Reinit VideoSource and Frame Queue
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    /////////////////Change///////////////////////////
    initCudaVideo();
    initGLTexture(g_pVideoDecoder->targetWidth(),
                  g_pVideoDecoder->targetHeight());
    /////////////////////////////////////////

    return S_OK;
}

void displayHelp()
{
    printf("\n");
    printf("%s - Help\n\n", sAppName);
    printf("  %s [parameters] [video_file]\n\n", sAppFilename);
    printf("Program parameters:\n");
    printf("\t-decodecuda   - Use CUDA for MPEG-2 (Available with 64+ CUDA cores)\n");
    printf("\t-decodedxva   - Use VP for MPEG-2, VC-1, H.264 decode.\n");
    printf("\t-decodecuvid  - Use VP for MPEG-2, VC-1, H.264 decode (optimized)\n");
    printf("\t-vsync        - Enable vertical sync.\n");
    printf("\t-novsync      - Disable vertical sync.\n");
    printf("\t-repeatframe  - Enable frame repeats.\n");
    printf("\t-updateall    - always update CSC matrices.\n");
    printf("\t-displayvideo - display video frames on the window\n");
    printf("\t-nointerop    - create the CUDA context w/o using graphics interop\n");
    printf("\t-readback     - enable readback of frames to system memory\n");
    printf("\t-device=n     - choose a specific GPU device to decode video with\n");
}

void parseCommandLineArguments(int argc, char *argv[])
{
    char video_file[256];

    printf("Command Line Arguments:\n");

    for (int n=0; n < argc; n++)
    {
        printf("argv[%d] = %s\n", n, argv[n]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayHelp();
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuda"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodedxva"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferDXVA;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuvid"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "vsync"))
    {
        g_bUseVsync = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "novsync"))
    {
        g_bUseVsync = false;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "repeatframe"))
    {
        g_bFrameRepeat = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "framestep"))
    {
        g_bFrameStep     = true;
        g_bUseDisplay    = true;
        g_fpsLimit = 1;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "updateall"))
    {
        g_bUpdateAll = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "displayvideo"))
    {
        g_bUseDisplay = true;
        g_bUseInterop = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "nointerop"))
    {
        g_bUseInterop = false;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "readback"))
    {
        g_bReadback = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        g_DeviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        //        g_bUseDisplay    = true;
    }

    if (g_bUseDisplay == false)
    {
        g_bQAReadback = true;
    }

    if (g_bLoop == false)
    {
        g_bAutoQuit = true;
    }

    // Search all command file parameters for video files with extensions:
    // mp4, avc, mkv, 264, h264. vc1, wmv, mp2, mpeg2, mpg
    char *file_ext = NULL;

    for (int i=1; i < argc; i++)
    {
        if (getFileExtension(argv[i], &file_ext) > 0)
        {
            strcpy(video_file, argv[i]);
            break;
        }
    }

    // We load the default video file for the CUDA Sample
    if (file_ext == NULL)
    {
        strcpy(video_file, sdkFindFilePath(VIDEO_SOURCE_FILE, argv[0]));
    }

    // Now verify the video file is legit
    FILE *fp = fopen(video_file, "r");

    if (video_file == NULL && fp == NULL)
    {
        printf("[%s]: unable to find file: <%s>\nExiting...\n", sAppFilename, VIDEO_SOURCE_FILE);
        exit(EXIT_FAILURE);
    }

    if (fp)
    {
        fclose(fp);
    }

    // default video file loaded by this sample
    sFileName = video_file;

    // store the current path so we can reinit the CUDA context
    strcpy(exec_path, argv[0]);

    printf("[%s]: input file: <%s>\n", sAppFilename, video_file);
}

int main(int argc, char *argv[])
{

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("[%s]\n", sAppName);

    sdkCreateTimer(&frame_timer);
    sdkResetTimer(&frame_timer);

    sdkCreateTimer(&global_timer);
    sdkResetTimer(&global_timer);

    // parse the command line arguments
    parseCommandLineArguments(argc, argv);

    // Find out the video size
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    // Determine the proper window size needed to create the correct *client* area
    // that is of the size requested by m_dimensions.
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    RECT adjustedWindowSize;
    DWORD dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
    SetRect(&adjustedWindowSize, 0, 0, g_nVideoWidth  , g_nVideoHeight);
    AdjustWindowRect(&adjustedWindowSize, dwWindowStyle, false);
#endif

    g_nVideoWidth   = PAD_ALIGN(g_nVideoWidth   , 0x3F);
    g_nVideoHeight  = PAD_ALIGN(g_nVideoHeight  , 0x0F);

    // Initialize the CUDA Device
    cuInit(0);

    // Initialize CUDA and try to connect with an OpenGL context
    // Other video memory resources will be available
    int bTCC = 0;

    if (initCudaResources(argc, argv, &bTCC) == false)
    {
        g_bAutoQuit = true;
        g_bException = true;
        g_bWaived   = true;
        goto ExitApp;
    }

    g_pVideoSource->start();
    g_bRunning = true;

    sdkStartTimer(&global_timer);
    sdkResetTimer(&global_timer);

    if (!g_bUseInterop)
    {
        // On this case we drive the display with a while loop (no openGL calls)
        bool bQuit = false;

        while (!bQuit)
        {
            bQuit = renderVideoFrame(g_bUseInterop);
        }
    }
    else
    {
        glutMainLoop();
    }

    g_pFrameQueue->endDecode();
    g_pVideoSource->stop();

    computeFPS(g_bUseInterop);
    printStatistics();

ExitApp:
    // clean up CUDA and OpenGL resources
    cleanup(g_bWaived ? false : true);

    if (g_bWaived)
    {
        exit(EXIT_WAIVED);
    }
    else
    {
        exit(g_bException ? EXIT_FAILURE : EXIT_SUCCESS);
    }

    return 0;
}


// display results using OpenGL
void display()
{
    bool bQuit = false;
    bQuit = renderVideoFrame(true);

    if (bQuit)
    {
        g_pFrameQueue->endDecode();
        g_pVideoSource->stop();

        computeFPS(g_bUseInterop);
        printStatistics();

        cleanup(true);
        exit(EXIT_SUCCESS);
    }
}


void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            if (g_pFrameQueue)
            {
                g_pFrameQueue->endDecode();
            }

            if (g_pVideoSource)
            {
                g_pVideoSource->stop();
            }

            computeFPS(g_bUseInterop);
            printStatistics();

            cleanup(true);
            exit(EXIT_FAILURE);
            break;

        case 'F':
        case 'f':
            g_bLinearFiltering = !g_bLinearFiltering;

            if (g_pImageGL)
                g_pImageGL->setTextureFilterMode(g_bLinearFiltering ? GL_LINEAR : GL_NEAREST,
                                                 g_bLinearFiltering ? GL_LINEAR : GL_NEAREST);

            break;

        case ' ':
            g_bRunning = !g_bRunning;
            break;

        default:
            break;
    }

    glutPostRedisplay();
}

void idle()
{
    glutPostRedisplay();
}

void reshape(int window_x, int window_y)
{
    printf("reshape() glViewport(%d, %d, %d, %d)\n", 0, 0, window_x, window_y);

    glViewport(0, 0, window_x, window_y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

// Initialize OpenGL Resources
bool initGL(int argc, char **argv, int *pbTCC)
{
    int dev, device_count = 0;
    bool bSpecifyDevice=false;
    char device_name[256];

    // Check for a min spec of Compute 1.1 capability before running
    checkCudaErrors(cuDeviceGetCount(&device_count));

    for (int i=0; i < argc; i++)
    {
        int string_start = 0;

        while (argv[i][string_start++] != '-');

        char *string_argv = &argv[i][string_start];

        if (!STRNCASECMP(string_argv, "device=", 7))
        {
            bSpecifyDevice = true;
        }
    }

    // If deviceID == 0, and there is more than 1 device, let's find the first available graphics GPU
    if (!bSpecifyDevice && device_count > 0)
    {
        for (int i=0; i < device_count; i++)
        {
            checkCudaErrors(cuDeviceGet(&dev, i));
            checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

            int bSupported = checkCudaCapabilitiesDRV(1, 1, i);

            if (!bSupported)
            {
                printf("  -> GPU: \"%s\" does not meet the minimum spec of SM 1.1\n", device_name);
                printf("  -> A GPU with a minimum compute capability of SM 1.1 or higher is required.\n");
                return false;
            }

#if CUDA_VERSION >= 3020
            checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");
#endif

            if (*pbTCC)
            {
                continue;
            }
            else
            {
                g_DeviceID = i; // we choose an available non-TCC display device
                break;
            }

#else

            // We don't know for sure if this is a TCC device or not, if it is Tesla we will not run
            if (!STRNCASECMP(device_name, "Tesla", 5))
            {
                printf("  \"%s\" does not support %s\n", device_name, sSDKname);
                *pbTCC = 1;
                return false;
            }
            else
            {
                *pbTCC = 0;
            }

#endif
            printf("\n");
        }
    }
    else
    {
        if ((g_DeviceID > (device_count-1)) || (g_DeviceID < 0))
        {
            printf(" >>> Invalid GPU Device ID=%d specified, only %d GPU device(s) are available.<<<\n", g_DeviceID, device_count);
            printf(" >>> Valid GPU ID (n) range is between [%d,%d]...  Exiting... <<<\n", 0, device_count-1);
            return false;
        }

        // We are specifying a GPU device, check to see if it is TCC or not
        checkCudaErrors(cuDeviceGet(&dev, g_DeviceID));
        checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

#if CUDA_VERSION >= 3020
        checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");
#endif
#else

        // We don't know for sure if this is a TCC device or not, if it is Tesla we will not run
        if (!STRNCASECMP(device_name, "Tesla", 5))
        {
            printf("  \"%s\" does not support %s\n", device_name, sSDKname);
            *pbTCC = 1;
            return false;
        }
        else
        {
            *pbTCC = 0;
        }

#endif
    }

    if (!(*pbTCC))
    {
        // initialize GLUT
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(g_nWindowWidth, g_nWindowHeight);
        glutCreateWindow(sAppName);
		reshape(g_nWindowWidth, g_nWindowHeight);

        printf(">> initGL() creating window [%d x %d]\n", g_nWindowWidth, g_nWindowHeight);

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutKeyboardFunc(keyboard);
        glutIdleFunc(idle);

        if (!isGLVersionSupported(1,5) ||
            !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
        {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 1.5\n");
            fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
            fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            return true;
        }

        if (!g_bUseVsync)
        {
            setVSync(0);
        }
    }
    else
    {
        fprintf(stderr, "> %s is decoding w/o visualization\n", sSDKname);
    }

    return true;
}


// Initializes OpenGL Textures (allocation and initialization)
bool
initGLTexture(unsigned int nWidth, unsigned int nHeight)
{
    g_pImageGL = new ImageGL(nWidth, nHeight,
                             nWidth, nHeight,
                             g_bIsProgressive,
                             ImageGL::BGRA_PIXEL_FORMAT);
    g_pImageGL->clear(0x80);

    g_pImageGL->setCUDAcontext(g_oContext);
    g_pImageGL->setCUDAdevice(g_oDevice);
    return true;
}

bool
loadVideoSource(const char *video_file,
                unsigned int &width    , unsigned int &height,
                unsigned int &dispWidth, unsigned int &dispHeight)
{
    std::auto_ptr<FrameQueue> apFrameQueue(new FrameQueue);
    std::auto_ptr<VideoSource> apVideoSource(new VideoSource(video_file, apFrameQueue.get()));

    // retrieve the video source (width,height)
    apVideoSource->getSourceDimensions(width, height);
    apVideoSource->getSourceDimensions(dispWidth, dispHeight);

    std::cout << apVideoSource->format() << std::endl;

    if (g_bFrameRepeat)
    {
        g_iRepeatFactor = (int)(60.0f / ceil((float)apVideoSource->format().frame_rate.numerator / (float)apVideoSource->format().frame_rate.denominator));
        printf("Frame Rate Playback Speed = %d fps\n", 60 / g_iRepeatFactor);
    }

    g_pFrameQueue  = apFrameQueue.release();
    g_pVideoSource = apVideoSource.release();

    if (g_pVideoSource->format().codec == cudaVideoCodec_JPEG ||
        g_pVideoSource->format().codec == cudaVideoCodec_MPEG2)
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    bool IsProgressive = 0;
    g_pVideoSource->getProgressive(IsProgressive);
    return IsProgressive;
}

void
initCudaVideo()
{
    // bind the context lock to the CUDA context
    CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);

    if (result != CUDA_SUCCESS)
    {
        printf("cuvidCtxLockCreate failed: %d\n", result);
        assert(0);
    }

    size_t totalGlobalMem;
    size_t freeMem;

    cuMemGetInfo(&freeMem,&totalGlobalMem);
    printf("  Free memory:     %4.4f MB\n", (float)freeMem/(1024*1024));

    std::auto_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(g_pVideoSource->format(), g_oContext, g_eVideoCreateFlags, g_CtxLock));
    std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue, &g_oContext));
    g_pVideoSource->setParser(*apVideoParser.get());

    g_pVideoParser  = apVideoParser.release();
    g_pVideoDecoder = apVideoDecoder.release();

    // Create a Stream ID for handling Readback
    if (g_bReadback)
    {
        checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
        checkCudaErrors(cuStreamCreate(&g_KernelSID,   0));
        printf(">> initCudaVideo()\n");
        printf("   CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
        printf("   CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID   == 0) ? "Disabled" : "Enabled"), g_KernelSID);
    }
}


void
freeCudaResources(bool bDestroyContext)
{
    if (g_pVideoParser)
    {
        delete g_pVideoParser;
    }

    if (g_pVideoDecoder)
    {
        delete g_pVideoDecoder;
    }

    if (g_pVideoSource)
    {
        delete g_pVideoSource;
    }

    if (g_pFrameQueue)
    {
        delete g_pFrameQueue;
    }

    if (g_ReadbackSID)
    {
        checkCudaErrors(cuStreamDestroy(g_ReadbackSID));
    }

    if (g_KernelSID)
    {
        checkCudaErrors(cuStreamDestroy(g_KernelSID));
    }

    if (g_CtxLock)
    {
        checkCudaErrors(cuvidCtxLockDestroy(g_CtxLock));
    }

    if (g_oContext && bDestroyContext)
    {
        checkCudaErrors(cuCtxDestroy(g_oContext));
        g_oContext = NULL;
    }
}

// Run the Cuda part of the computation (if g_pFrameQueue is empty, then return false)
bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive)
{
    CUVIDPARSERDISPINFO oDisplayInfo;

    if (g_pFrameQueue->dequeue(&oDisplayInfo))
    {
        CCtxAutoLock lck(g_CtxLock);
        // Push the current CUDA context (only if we are using CUDA decoding path)
        CUresult result = cuCtxPushCurrent(g_oContext);

        CUdeviceptr  pDecodedFrame[2] = { 0, 0 };
        CUdeviceptr  pInteropFrame[2] = { 0, 0 };
        int num_fields = (oDisplayInfo.progressive_frame ? (1) : (2+oDisplayInfo.repeat_first_field));
        *pbIsProgressive = oDisplayInfo.progressive_frame;
        g_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;

        for (int active_field=0; active_field<num_fields; active_field++)
        {
            nRepeats = oDisplayInfo.repeat_first_field;
            CUVIDPROCPARAMS oVideoProcessingParameters;
            memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

            oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;
            oVideoProcessingParameters.second_field      = active_field;
            oVideoProcessingParameters.top_field_first   = oDisplayInfo.top_field_first;
            oVideoProcessingParameters.unpaired_field    = (num_fields == 1);

            unsigned int nWidth = 0;
            unsigned int nHeight = 0;
            unsigned int nDecodedPitch = 0;

            // map decoded video frame to CUDA surface
            g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
            nWidth  = PAD_ALIGN(g_pVideoDecoder->targetWidth() , 0x3F);
            nHeight = PAD_ALIGN(g_pVideoDecoder->targetHeight(), 0x0F);
            // map OpenGL PBO or CUDA memory
            size_t pFramePitch = 0;

            // If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
            if (g_bReadback && g_bFirstFrame && g_ReadbackSID)
            {
                CUresult result;
                checkCudaErrors(result = cuMemAllocHost((void **)&g_bFrameData[0], (nDecodedPitch * nHeight * 3 / 2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_bFrameData[1], (nDecodedPitch * nHeight * 3 / 2)));

                g_bFirstFrame = false;

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

            // If streams are enabled, we can perform the readback to the host while the kernel is executing
            if (g_bReadback && g_ReadbackSID)
            {
                CUresult result = cuMemcpyDtoHAsync(g_bFrameData[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), g_ReadbackSID);

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

#if ENABLE_DEBUG_OUT
            printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
                   (oDisplayInfo.progressive_frame ? "Frame" : "Field"),
                   g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

            if (g_pImageGL)
            {
                // map the texture surface
                g_pImageGL->map(&pInteropFrame[active_field], &pFramePitch, active_field);
                pFramePitch = g_nWindowWidth * 4;
            }
            else
            {
                pInteropFrame[active_field] = g_pInteropFrame[active_field];
                pFramePitch = g_pVideoDecoder->targetWidth() * 2;
            }

            // perform post processing on the CUDA surface (performs colors space conversion and post processing)
            // comment this out if we inclue the line of code seen above

            cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field],
                                 pFramePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);

            if (g_pImageGL)
            {
                // unmap the texture surface
                g_pImageGL->unmap(active_field);
            }

            // unmap video frame
            // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
            g_pVideoDecoder->unmapFrame(pDecodedFrame[active_field]);
            // release the frame, so it can be re-used in decoder
            g_pFrameQueue->releaseFrame(&oDisplayInfo);

            g_DecodeFrameCount++;
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
    }
    else
    {
        // Frame Queue has no frames, we don't compute FPS until we start
        return false;
    }

    // check if decoding has come to an end.
    // if yes, signal the app to shut down.
    if (g_pFrameQueue->isDecodeFinished())
    {
        // Let's just stop, and allow the user to quit, so they can at least see the results
        //g_pVideoSource->stop();

        // If we want to loop reload the video file and restart
        if (g_bLoop && !g_bAutoQuit)
        {
            reinitCudaResources();
            g_FrameCount = 0;
            g_DecodeFrameCount = 0;
            g_pVideoSource->start();
        }

        if (g_bAutoQuit)
        {
            g_bDone = true;
        }
    }

    return true;
}

// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
                     CUdeviceptr *ppInteropFrame, size_t pFramePitch,
                     CUmodule cuModNV12toARGB,
                     CUfunction fpCudaKernel, CUstream streamID)
{
    uint32 nWidth  = g_pVideoDecoder->targetWidth();
    uint32 nHeight = g_pVideoDecoder->targetHeight();

    // Upload the Color Space Conversion Matrices
    if (g_bUpdateCSC)
    {
        // CCIR 601/709
        float hueColorSpaceMat[9];
        setColorSpaceMatrix(g_eColorSpace,    hueColorSpaceMat, g_nHue);
        updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);

        if (!g_bUpdateAll)
        {
            g_bUpdateCSC = false;
        }
    }

    // TODO: Stage for handling video post processing

    // Final Stage: NV12toARGB color space conversion
    CUresult eResult;
    eResult = cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch,
                                      *ppInteropFrame, pFramePitch,
                                      nWidth, nHeight, fpCudaKernel, streamID);
}

// Draw the final result on the screen
bool drawScene(int field_num)
{
    bool hr = true;

    // Normal OpenGL rendering code
    // render image
    if (g_pImageGL)
    {
        g_pImageGL->render(field_num);
    }

    return hr;
}

// Release all previously initialized objects
bool cleanup(bool bDestroyContext)
{
    if (bDestroyContext)
    {
        // Attach the CUDA Context (so we may properly free memory)
        checkCudaErrors(cuCtxPushCurrent(g_oContext));

        if (g_pInteropFrame[0])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[0]));
        }

        if (g_pInteropFrame[1])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[1]));
        }

        // Let's free the Frame Data used during readback
        if (g_bReadback)
        {
            if (g_bFrameData[0])
            {
                checkCudaErrors(cuMemFreeHost((void *)g_bFrameData[0]));
                g_bFrameData[0] = NULL;
            }

            if (g_bFrameData[1])
            {
                checkCudaErrors(cuMemFreeHost((void *)g_bFrameData[1]));
                g_bFrameData[1] = NULL;
            }
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
    }

    if (g_pImageGL)
    {
        delete g_pImageGL;
        g_pImageGL = NULL;
    }

    freeCudaResources(bDestroyContext);

    return true;
}

// Launches the CUDA kernels to fill in the texture data
// returns true if it's the end of the decode and false otherwise
bool renderVideoFrame(int bUseInterop)
{
    static unsigned int nRepeatFrame = 0;
    int repeatFactor = g_iRepeatFactor;
    int bIsProgressive = 1, bFPSComputed = 0;
    bool bFramesDecoded = false;

    if (0 != g_pFrameQueue)
    {
        // if not running, we simply don't copy new frames from the decoder
        if (!g_bDeviceLost && g_bRunning)
        {
            bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, bUseInterop, &bIsProgressive);
        }
    }
    else
    {
        return true;
    }

    if (bFramesDecoded)
    {
        g_FrameCount ++;

        while (repeatFactor-- > 0)
        {
            if (g_bUseDisplay && bUseInterop)
            {
                // We will always draw field/frame 0
                drawScene(0);
                glutSwapBuffers();

                if (!repeatFactor)
                {
                    computeFPS(bUseInterop);
                }

                // If interlaced mode, then we will need to draw field 1 separately
                if (!bIsProgressive)
                {
                    drawScene(1);
                    glutSwapBuffers();

                    if (!repeatFactor)
                    {
                        computeFPS(bUseInterop);
                    }
                }

                bFPSComputed = 1;
            }

            // Pass the Windows handle to show Frame Rate on the window title
            if (!bFPSComputed && !repeatFactor)
            {
                computeFPS(bUseInterop);
            }

            if (g_bUseDisplay && bUseInterop)
            {
                glutReportErrors();
            }

        }
    }

    if (bFramesDecoded && g_bFrameStep)
    {
        if (g_bRunning)
        {
            g_bRunning = false;
        }
    }

    if (g_pFrameQueue->isDecodeFinished())
    {
        return true; //quit
    }

    return false;
}

