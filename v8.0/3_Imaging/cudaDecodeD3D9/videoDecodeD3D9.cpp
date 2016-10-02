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
 * bindings to interop between CUDA and DX9 textures for the purpose of post
 * processing video.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#endif
#include <d3dx9.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudad3d9.h>
#include <builtin_types.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda_drvapi.h>    // helper file for CUDA Driver API calls and error checking

// cudaDecodeD3D9 related helper functions
#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageDX.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

// Include files
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>

const char *sAppName     = "CUDA/D3D9 Video Decode";
const char *sAppFilename = "cudaDecodeD3D9";
const char *sSDKname     = "cudaDecodeD3D9";

#define VIDEO_SOURCE_FILE "plush1_720p_10s.m2v"

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    1
#else
#define ENABLE_DEBUG_OUT    0
#endif

StopWatchInterface *frame_timer  = NULL;
StopWatchInterface *global_timer = NULL;

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
bool                g_bFirstFrame = true;
bool                g_bLoop       = false;
bool                g_bUpdateCSC  = true;
bool                g_bUpdateAll  = false;
bool                g_bUseDisplay = false; // this flag enables/disables video on the window
bool                g_bUseInterop = false;
bool                g_bReadback   = false; // this flag enables/disables reading back of a video from a window
bool                g_bIsProgressive = true; // assume it is progressive, unless otherwise noted
bool                g_bException  = false;
bool                g_bWaived     = false;

int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats

int *pArgc = NULL;
char **pArgv = NULL;

cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
CUvideoctxlock       g_CtxLock = NULL;

float present_fps, decoded_fps, total_time = 0.0f;

D3DDISPLAYMODEEX      g_d3ddm;
D3DPRESENT_PARAMETERS g_d3dpp;

IDirect3D9Ex       *g_pD3D; // Used to create the D3DDevice
IDirect3DDevice9Ex *g_pD3DDevice;

// These are CUDA function pointers to the CUDA kernels
CUmoduleManager   *g_pCudaModule;

CUmodule           cuModNV12toARGB  = 0;
CUfunction         gfpNV12toARGB    = 0;
CUfunction         gfpPassthru      = 0;

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

ImageDX       *g_pImageDX      = 0;
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
bool    initD3D9(HWND hWnd, int argc, char **argv, int *pbTCC);
HRESULT initD3D9Surface(unsigned int nWidth, unsigned int nHeight);
HRESULT freeDestSurface();

bool loadVideoSource(const char *video_file,
                     unsigned int &width, unsigned int &height,
                     unsigned int &dispWidth, unsigned int &dispHeight);
void initCudaVideo();

void freeCudaResources(bool bDestroyContext);

bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive);
void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
                          CUdeviceptr *ppTextureData,  size_t nTexturePitch,
                          CUmodule cuModNV12toARGB,
                          CUfunction fpCudaKernel, CUstream streamID);
HRESULT drawScene(int field_num);
HRESULT cleanup(bool bDestroyContext);
HRESULT initCudaResources(int argc, char **argv, int bUseInterop, int bTCC);

bool renderVideoFrame(HWND hWnd, bool bUseInterop);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else // Linux
#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif


LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

bool checkHW(char *name, char *gpuType, int dev)
{
    char deviceName[256];
    checkCudaErrors(cuDeviceGetName(deviceName, 256, dev));

    STRCPY(name, strlen(deviceName), deviceName);

    if (!strnicmp(deviceName, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    CUresult err = cuInit(0);
    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, "Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                STRCPY(firstGraphicsName, strlen(temp), temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        STRCPY(name, strlen(firstGraphicsName), firstGraphicsName);
    }
    else
    {
        STRCPY(name, strlen("this hardware"), "this hardware");
    }

    return nGraphicsGPU;
}

void printStatistics()
{
    int   hh, mm, ss, msec;

    present_fps = 1.f / (total_time / (g_FrameCount * 1000.f));
    decoded_fps = 1.f / (total_time / (g_DecodeFrameCount * 1000.f));

    msec = ((int)total_time % 1000);
    ss   = (int)(total_time/1000) % 60;
    mm   = (int)(total_time/(1000*60)) % 60;
    hh   = (int)(total_time/(1000*60*60)) % 60;

    printf("\n[%s] statistics\n", sSDKname);
    printf("\t Video Length (hh:mm:ss.msec)   = %02d:%02d:%02d.%03d\n", hh, mm, ss, msec);

    printf("\t Frames Presented (inc repeats) = %d\n", g_FrameCount);
    printf("\t Average Present FPS            = %4.2f\n", present_fps);

    printf("\t Frames Decoded   (hardware)    = %d\n", g_DecodeFrameCount);
    printf("\t Average Decoder FPS            = %4.2f\n", decoded_fps);

}

void computeFPS(HWND hWnd, bool bUseInterop)
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
                sAppName, sDecodeStatus.c_str(),
                (g_bIsProgressive ? "Frame" : "Field"),
                g_DecodeFrameCount);

        if (bUseInterop && (!g_bQAReadback))
        {
            SetWindowText(hWnd, sFPS);
            UpdateWindow(hWnd);
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

        // we only want to record this once
        if (total_time == 0.0f)
        {
            total_time = sdkGetTimerValue(&global_timer);
        }

        sdkStopTimer(&global_timer);

        if (bUseInterop && (!g_bQAReadback))
        {
            SetWindowText(hWnd, sFPS);
            UpdateWindow(hWnd);
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

        if (bUseInterop && (!g_bQAReadback))
        {
            SetWindowText(hWnd, sFPS);
            UpdateWindow(hWnd);
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

HRESULT initCudaResources(int argc, char **argv, int bUseInterop, int bTCC)
{
    HRESULT hr = S_OK;

    CUdevice cuda_device;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        cuda_device = getCmdLineArgumentInt(argc, (const char **) argv, "device");

        // If interop is disabled, then we need to create a CUDA context w/o the GL context
        if (bUseInterop && !bTCC)
        {
            cuda_device = findCudaDeviceDRV(argc, (const char **)argv);
        }
        else
        {
            cuda_device = findCudaGLDeviceDRV(argc, (const char **)argv);
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
        if (bUseInterop)
        {
            cuda_device = gpuGetMaxGflopsGLDeviceIdDRV();
            checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
        }
        else
        {
            cuda_device = gpuGetMaxGflopsDeviceIdDRV();
            checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
        }
    }

    // get compute capabilities and the devicename
    int major, minor;
    size_t totalGlobalMem;
    char deviceName[256];
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
    printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
    printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem/(1024*1024));

    // Create CUDA Device w/ D3D9 interop (if WDDM), otherwise CUDA w/o interop (if TCC)
    // (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
    if (bUseInterop)
    {
        checkCudaErrors(cuD3D9CtxCreate(&g_oContext, &g_oDevice, CU_CTX_BLOCKING_SYNC, g_pD3DDevice));
    }
    else
    {
        checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }

    try
    {
        // Initialize CUDA related Driver API (32-bit or 64-bit), depending the platform running
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
        return E_FAIL;
    }

    g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi",   &gfpNV12toARGB);
    g_pCudaModule->GetCudaFunction("Passthru_drvapi",     &gfpPassthru);

    /////////////////Change///////////////////////////
    // Now we create the CUDA resources and the CUDA decoder context
    initCudaVideo();

    if (bUseInterop)
    {
        initD3D9Surface(g_pVideoDecoder->targetWidth(),
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
    return ((g_pCudaModule && g_pVideoDecoder && (g_pImageDX || g_pInteropFrame[0])) ? S_OK : E_FAIL);
}

HRESULT reinitCudaResources()
{
    // Free resources
    cleanup(false);

    // Reinit VideoSource and Frame Queue
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    /////////////////Change///////////////////////////
    initCudaVideo();
    initD3D9Surface(g_pVideoDecoder->targetWidth(),
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
        g_bUseInterop    = true;
        g_fpsLimit = 1;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "updateall"))
    {
        g_bUpdateAll     = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "displayvideo"))
    {
        g_bUseDisplay   = true;
        g_bUseInterop   = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "nointerop"))
    {
        g_bUseInterop   = false;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "readback"))
    {
        g_bReadback     = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        g_DeviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        g_bUseDisplay    = true;
        g_bUseInterop    = true;
    }

    if (g_bUseDisplay == false)
    {
        g_bQAReadback    = true;
        g_bUseInterop    = false;
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
    pArgc = &argc;
    pArgv = argv;

    sdkCreateTimer(&frame_timer);
    sdkResetTimer(&frame_timer);

    sdkCreateTimer(&global_timer);
    sdkResetTimer(&global_timer);

    // parse the command line arguments
    parseCommandLineArguments(argc, argv);

    // create window (after we know the size of the input file size)
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      sAppName, NULL
                    };
    RegisterClassEx(&wc);

    // Find out the video size
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    // figure out the window size we must create to get a *client* area
    // that is of the size requested by m_dimensions.
    RECT adjustedWindowSize;
    DWORD dwWindowStyle;
    HWND hWnd = NULL;

    {
        dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
        SetRect(&adjustedWindowSize, 0, 0, g_nVideoWidth  , g_nVideoHeight);
        AdjustWindowRect(&adjustedWindowSize, dwWindowStyle, false);

        g_nWindowWidth  = adjustedWindowSize.right  - adjustedWindowSize.left;
        g_nWindowHeight = adjustedWindowSize.bottom - adjustedWindowSize.top;

        // Create the application's window
        hWnd = CreateWindow(wc.lpszClassName, sAppName,
                            dwWindowStyle,
                            0, 0,
                            g_nWindowWidth,
                            g_nWindowHeight,
                            NULL, NULL, wc.hInstance, NULL);
    }

    // Initialize CUDA
    cuInit(0);

    int bTCC = 0;

    if (g_bUseInterop)
    {
        // Initialize Direct3D
        if (initD3D9(hWnd, argc, argv, &bTCC) == false)
        {
            g_bAutoQuit = true;
            g_bWaived   = true;
            goto ExitApp;
        }
    }

    // If we are using TCC driver, then always turn off interop
    if (bTCC)
    {
        g_bUseInterop = false;
    }

    // Initialize CUDA/D3D9 context and other video memory resources
    if (initCudaResources(argc, argv, g_bUseInterop, bTCC) == E_FAIL)
    {
        g_bAutoQuit  = true;
        g_bException = true;
        g_bWaived    = true;
        goto ExitApp;
    }

    g_pVideoSource->start();
    g_bRunning = true;

    if (!g_bQAReadback && !bTCC)
    {
        ShowWindow(hWnd, SW_SHOWDEFAULT);
        UpdateWindow(hWnd);
    }

    // the main loop
    sdkStartTimer(&frame_timer);
    sdkStartTimer(&global_timer);
    sdkResetTimer(&global_timer);

    if (!g_bUseInterop)
    {
        // On this case we drive the display with a while loop (no OpenGL calls)
        bool bQuit = false;

        while (g_pVideoSource->isStarted() && !bQuit)
        {
            bQuit = renderVideoFrame(hWnd, g_bUseInterop);
        }
    }
    else
    {
        // Standard windows loop
        while (!g_bDone)
        {
            MSG msg;
            ZeroMemory(&msg, sizeof(msg));

            while (msg.message!=WM_QUIT)
            {
                if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                else
                {
                    renderVideoFrame(hWnd, g_bUseInterop);
                }

                if (g_bAutoQuit && g_bDone)
                {
                    break;
                }
            }
        } // while loop
    }

    g_pFrameQueue->endDecode();
    g_pVideoSource->stop();

    printStatistics();

    // clean up CUDA and D3D resources
ExitApp:
    cleanup(g_bWaived ? false : true);

    if (!g_bQAReadback)
    {
        // Unregister windows class
        UnregisterClass(wc.lpszClassName, wc.hInstance);
    }

    if (g_bAutoQuit)
    {
        PostQuitMessage(0);
    }

    if (hWnd)
    {
        DestroyWindow(hWnd);
    }

    if (g_bWaived)
    {
        exit(EXIT_WAIVED);
    }
    else
    {
        exit(g_bException ? EXIT_FAILURE : EXIT_SUCCESS);
    }
}

// Initialize Direct3D
bool
initD3D9(HWND hWnd, int argc, char **argv, int *pbTCC)
{
    int dev, device_count = 0;
    bool bSpecifyDevice=false;
    char device_name[256];

    // Check for a min spec of Compute 1.1 capability before running
    checkCudaErrors(cuDeviceGetCount(&device_count));

    for (int i=0; i < argc; i++)
    {
        int string_start = 0;

        while (argv[i][string_start] == '-')
        {
            string_start++;
        }

        const char *string_argv = &argv[i][string_start];

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
            printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

            if (*pbTCC)
            {
                continue;
            }
            else
            {
                g_DeviceID = i; // we choose an available WDDM display device
            }

#else

            // We don't know for sure if this is a TCC device or not, if it is Tesla system, we will not run
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

        checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
        printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");
    }

    // Only if we are not using a TCC device will we use this path, otherwise we support using the VP w/o TCC
    HRESULT eResult = S_OK;

    if (!(*pbTCC))
    {
        // Create the D3D object.
        if (S_OK != Direct3DCreate9Ex(D3D_SDK_VERSION, &g_pD3D))
        {
            return false;
        }

        // Get primary display identifier
        D3DADAPTER_IDENTIFIER9 adapterId;
        bool bDeviceFound = false;
        int device;

        // Find the first CUDA capable device
        CUresult cuStatus;

        for (unsigned int g_iAdapter = 0; g_iAdapter < g_pD3D->GetAdapterCount(); g_iAdapter++)
        {
            HRESULT hr = g_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &adapterId);

            if (FAILED(hr))
            {
                continue;
            }

            cuStatus = cuD3D9GetDevice(&device, adapterId.DeviceName);
            printf("> Display Device: \"%s\" %s Direct3D9\n",
                   adapterId.Description,
                   (cuStatus == cudaSuccess) ? "supports" : "does not support");

            if (cudaSuccess == cuStatus)
            {
                bDeviceFound = true;
                break;
            }
        }

        // we check to make sure we have found a cuda-compatible D3D device to work on
        if (!bDeviceFound)
        {
            printf("\n");
            printf("  No CUDA-compatible Direct3D9 device available\n");
            // destroy the D3D device
            g_pD3D->Release();
            g_pD3D = NULL;
            return false;
        }

        // Create the D3D Display Device
        RECT                  rc;
        GetClientRect(hWnd, &rc);
        g_pD3D->GetAdapterDisplayModeEx(D3DADAPTER_DEFAULT, &g_d3ddm, NULL);
        ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));

        g_d3dpp.Windowed               = (g_bQAReadback ? TRUE : g_bWindowed); // fullscreen or windowed?

        g_d3dpp.BackBufferCount        = 1;
        g_d3dpp.SwapEffect             = D3DSWAPEFFECT_DISCARD;
        g_d3dpp.hDeviceWindow          = hWnd;
        g_d3dpp.BackBufferWidth        = rc.right  - rc.left;
        g_d3dpp.BackBufferHeight       = rc.bottom - rc.top;
        g_d3dpp.BackBufferFormat       = g_d3ddm.Format;
        g_d3dpp.FullScreen_RefreshRateInHz = 0; // set to 60 for fullscreen, and also don't forget to set Windowed to FALSE
        g_d3dpp.PresentationInterval   = (g_bUseVsync ? D3DPRESENT_INTERVAL_ONE : D3DPRESENT_INTERVAL_IMMEDIATE);

        if (g_bQAReadback)
        {
            g_d3dpp.Flags              = D3DPRESENTFLAG_VIDEO;    // turn off vsync
        }

        eResult = g_pD3D->CreateDeviceEx(0, D3DDEVTYPE_HAL, hWnd,
                                       D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
                                       &g_d3dpp, NULL, &g_pD3DDevice);
    }
    else
    {
        fprintf(stderr, "> %s is decoding w/o visualization\n", sSDKname);
        eResult = S_OK;
    }

    return (eResult == S_OK);
}

// Initialize Direct3D Textures (allocation and initialization)
HRESULT
initD3D9Surface(unsigned int nWidth, unsigned int nHeight)
{
    g_pImageDX = new ImageDX(g_pD3DDevice,
                             nWidth, nHeight,
                             nWidth, nHeight,
                             g_bIsProgressive,
                             ImageDX::BGRA_PIXEL_FORMAT); // ImageDX::LUMINANCE_PIXEL_FORMAT
    g_pImageDX->clear(0x80);

    g_pImageDX->setCUDAcontext(g_oContext);
    g_pImageDX->setCUDAdevice(g_oDevice);

    return S_OK;
}

HRESULT
freeDestSurface()
{
    if (g_pImageDX)
    {
        delete g_pImageDX;
        g_pImageDX = NULL;
    }

    return S_OK;
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
        g_iRepeatFactor = ftoi((60.0f / ceil((float)apVideoSource->format().frame_rate.numerator / (float)apVideoSource->format().frame_rate.denominator)));
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
    std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue));
    g_pVideoSource->setParser(*apVideoParser.get());

    g_pVideoParser  = apVideoParser.release();
    g_pVideoDecoder = apVideoDecoder.release();

    // Create a Stream ID for handling Readback
    if (g_bReadback)
    {
        checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
        checkCudaErrors(cuStreamCreate(&g_KernelSID,   0));
        printf("> initCudaVideo()\n");
        printf("  CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
        printf("  CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID   == 0) ? "Disabled" : "Enabled"), g_KernelSID);
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
        cuStreamDestroy(g_ReadbackSID);
    }

    if (g_KernelSID)
    {
        cuStreamDestroy(g_KernelSID);
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

// Run the Cuda part of the computation
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

            unsigned int nDecodedPitch = 0;
            unsigned int nWidth = 0;
            unsigned int nHeight = 0;

            // map decoded video frame to CUDA surface
            g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
            nWidth  = g_pVideoDecoder->targetWidth();
            nHeight = g_pVideoDecoder->targetHeight();
            // map DirectX texture to CUDA surface
            size_t nTexturePitch = 0;

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
                }
            }

            // If streams are enabled, we can perform the readback to the host while the kernel is executing
            if (g_bReadback && g_ReadbackSID)
            {
                CUresult result = cuMemcpyDtoHAsync(g_bFrameData[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), g_ReadbackSID);

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                }
            }

#if ENABLE_DEBUG_OUT
            printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
                   (oDisplayInfo.progressive_frame ? "Frame" : "Field"),
                   g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

            if (g_pImageDX)
            {
                // map the texture surface
                g_pImageDX->map(&pInteropFrame[active_field], &nTexturePitch, active_field);
            }
            else
            {
                pInteropFrame[active_field] = g_pInteropFrame[active_field];
                nTexturePitch = g_pVideoDecoder->targetWidth() * 2;
            }

            // perform post processing on the CUDA surface (performs colors space conversion and post processing)
            // comment this out if we inclue the line of code seen above
            cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field], nTexturePitch, g_pCudaModule->getModule(), gfpNV12toARGB, g_KernelSID);

            if (g_pImageDX)
            {
                // unmap the texture surface
                g_pImageDX->unmap(active_field);
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
        return false;
    }

    // check if decoding has come to an end.
    // if yes, signal the app to shut down.
    if (g_pFrameQueue->isDecodeFinished())
    {
        // Let's just stop, and allow the user to quit, so they can at least see the results
        g_pVideoSource->stop();

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
                     CUdeviceptr *ppTextureData,  size_t nTexturePitch,
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
                                      *ppTextureData,  nTexturePitch,
                                      nWidth, nHeight, fpCudaKernel, streamID);
}

// Draw the final result on the screen
HRESULT drawScene(int field_num)
{
    HRESULT hr = S_OK;

    // Begin code to handle case where the D3D gets lost
    if (g_bDeviceLost)
    {
        // test the cooperative level to see if it's okay
        // to render
        if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel()))
        {
            fprintf(stderr, "TestCooperativeLevel = %08x failed, will attempt to reset\n", hr);

            // if the device was truly lost, (i.e., a fullscreen device just lost focus), wait
            // until we get it back

            if (hr == D3DERR_DEVICELOST)
            {
                fprintf(stderr, "TestCooperativeLevel = %08x DeviceLost, will retry next call\n", hr);
                return S_OK;
            }

            // eventually, we will get this return value,
            // indicating that we can now reset the device
            if (hr == D3DERR_DEVICENOTRESET)
            {
                fprintf(stderr, "TestCooperativeLevel = %08x will try to RESET the device\n", hr);
                // if we are windowed, read the desktop mode and use the same format for
                // the back buffer; this effectively turns off color conversion

                if (g_bWindowed)
                {
                    g_pD3D->GetAdapterDisplayModeEx(D3DADAPTER_DEFAULT, &g_d3ddm, NULL);
                    g_d3dpp.BackBufferFormat = g_d3ddm.Format;
                }

                // now try to reset the device
                if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp)))
                {
                    fprintf(stderr, "TestCooperativeLevel = %08x RESET device FAILED\n", hr);
                    return hr;
                }
                else
                {
                    fprintf(stderr, "TestCooperativeLevel = %08x RESET device SUCCESS!\n", hr);
                    // Reinit All other resources including CUDA contexts
                    initCudaResources(0, NULL, true, false);
                    fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n", hr);

                    // we have acquired the device
                    g_bDeviceLost = false;
                }
            }
        }
    }

    // Normal D3D9 rendering code
    if (!g_bDeviceLost)
    {
        // init the scene
        if (g_bUseDisplay)
        {
            g_pD3DDevice->BeginScene();
            g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
            // render image
            g_pImageDX->render(field_num);
            // end the scene
            g_pD3DDevice->EndScene();
        }

        hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

        if (hr == D3DERR_DEVICELOST)
        {
            fprintf(stderr, "drawScene Present = %08x detected D3D DeviceLost\n", hr);
            g_bDeviceLost = true;

            // We check for DeviceLost, if that happens we want to release resources
            g_pFrameQueue->endDecode();
            g_pVideoSource->stop();

            freeCudaResources(false);
        }
    }
    else
    {
        fprintf(stderr, "drawScene (DeviceLost == TRUE), waiting\n");
    }

    return S_OK;
}

// Release all previously initialized objects
HRESULT cleanup(bool bDestroyContext)
{
    // Attach the CUDA Context (so we may properly free memory)
    if (bDestroyContext)
    {
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

    if (g_pImageDX)
    {
        delete g_pImageDX;
        g_pImageDX = NULL;
    }

    freeCudaResources(bDestroyContext);



    // destroy the D3D device
    if (g_pD3DDevice)
    {
        g_pD3DDevice->Release();
        g_pD3DDevice = NULL;
    }

    return S_OK;
}

// Launches the CUDA kernels to fill in the texture data
bool renderVideoFrame(HWND hWnd, bool bUseInterop)
{
    static unsigned int nRepeatFrame = 0;
    int repeatFactor = g_iRepeatFactor;
    int bIsProgressive = 1, bFPSComputed = 0;
    bool bFramesDecoded = false;

    if (0 != g_pFrameQueue)
    {
        // if not running, we simply don't copy
        // new frames from the decoder
        if (!g_bDeviceLost && g_bRunning)
        {
            bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, true, &bIsProgressive);
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
            // draw the scene using the copied textures
            if (g_bUseDisplay && bUseInterop)
            {
                // We will always draw field/frame 0
                drawScene(0);

                if (!repeatFactor)
                {
                    computeFPS(hWnd, bUseInterop);
                }

                // If interlaced mode, then we will need to draw field 1 separately
                if (!bIsProgressive)
                {
                    drawScene(1);

                    if (!repeatFactor)
                    {
                        computeFPS(hWnd, bUseInterop);
                    }
                }

                bFPSComputed = 1;
            }
        }

        // Pass the Windows handle to show Frame Rate on the window title
        if (!bFPSComputed)
        {
            computeFPS(hWnd, bUseInterop);
        }
    }

    if (bFramesDecoded && g_bFrameStep)
    {
        if (g_bRunning)
        {
            g_bRunning = false;
        }
    }

    if (bFramesDecoded == false && g_pFrameQueue->isDecodeFinished())
    {
        return true; //quit
    }

    return false;
}

// The window's message handler
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_KEYDOWN:
            switch (wParam)
            {
                // use ESC to quit application
                case VK_ESCAPE:
                    {
                        g_bDone = true;
                        PostQuitMessage(0);
                        return 0;
                    }
                    break;

                // use space to pause playback
                case VK_SPACE:
                    {
                        g_bRunning = !g_bRunning;
                    }
                    break;
            }

            break;

        // resize the window
        // even though we disable resizing (see next event handler below)
        // we need to be able to handle this event properly because the
        // windowing system calls it.
        case WM_SIZE:
            {
                // store new window size in our globals.
                g_nClientAreaWidth  = GET_X_LPARAM(lParam);
                g_nClientAreaHeight = GET_Y_LPARAM(lParam);

                D3DVIEWPORT9 oViewport;
                oViewport.X = 0;
                oViewport.Y = 0;
                oViewport.Width  = g_nClientAreaWidth;
                oViewport.Height = g_nClientAreaHeight;
                oViewport.MaxZ = 1.0f;
                oViewport.MinZ = 0.0f;

                g_pD3DDevice->SetViewport(&oViewport);

                D3DMATRIX oViewMatrix = {1.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 1.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 1.0f
                                        };
                // scale viewport to be of window width and height
                oViewMatrix._11 = 2.0f/g_nClientAreaWidth;
                oViewMatrix._22 = 2.0f/g_nClientAreaHeight;
                // translate viewport so that lower left corner represents
                // (0, 0) and upper right corner is (width, height)
                oViewMatrix._41 = -1.0f - 1.0f / g_nClientAreaWidth;
                oViewMatrix._42 = -1.0f + 1.0f / g_nClientAreaHeight;

                if (0 != g_pD3DDevice)
                {
                    g_pD3DDevice->SetTransform(D3DTS_VIEW, &oViewMatrix);
                }

                renderVideoFrame(hWnd, g_bUseInterop);

                return 0;   // Jump Back
            }

        // disallow resizing.
        case WM_GETMINMAXINFO:
            {
                MINMAXINFO *pMinMaxInfo = reinterpret_cast<MINMAXINFO *>(lParam);

                pMinMaxInfo->ptMinTrackSize.x = g_nWindowWidth;
                pMinMaxInfo->ptMinTrackSize.y = g_nWindowHeight;

                pMinMaxInfo->ptMaxTrackSize.x = g_nWindowWidth;
                pMinMaxInfo->ptMaxTrackSize.y = g_nWindowHeight;

                return 0;
            }

        case WM_DESTROY:
            g_bDone = true;
            PostQuitMessage(0);
            return 0;

        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

