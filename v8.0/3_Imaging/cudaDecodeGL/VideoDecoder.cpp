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

#include "VideoDecoder.h"

#include "FrameQueue.h"
#include "stdio.h"
#include <cstring>
#include <cassert>
#include <string>

VideoDecoder::VideoDecoder(const CUVIDEOFORMAT &rVideoFormat,
                           CUcontext &rContext,
                           cudaVideoCreateFlags eCreateFlags,
                           CUvideoctxlock &vidCtxLock)
    : m_VidCtxLock(vidCtxLock)
{
    // get a copy of the CUDA context
    m_Context          = rContext;
    m_VideoCreateFlags = eCreateFlags;

    printf("> VideoDecoder::cudaVideoCreateFlags = <%d>", (int)eCreateFlags);

    switch (eCreateFlags)
    {
        case cudaVideoCreate_Default:
            printf("Default (VP)\n");
            break;

        case cudaVideoCreate_PreferCUDA:
            printf("Use CUDA decoder\n");
            break;

        case cudaVideoCreate_PreferDXVA:
            printf("Use DXVA decoder\n");
            break;

        case cudaVideoCreate_PreferCUVID:
            printf("Use CUVID decoder\n");
            break;

        default:
            printf("Unknown value\n");
            break;
    }

    printf("\n");

    // Validate video format.  These are the currently supported formats via NVCUVID
    assert(cudaVideoCodec_MPEG1 == rVideoFormat.codec ||
           cudaVideoCodec_MPEG2 == rVideoFormat.codec ||
           cudaVideoCodec_MPEG4 == rVideoFormat.codec ||
           cudaVideoCodec_VC1   == rVideoFormat.codec ||
           cudaVideoCodec_H264  == rVideoFormat.codec ||
           cudaVideoCodec_JPEG  == rVideoFormat.codec ||
           cudaVideoCodec_YUV420== rVideoFormat.codec ||
           cudaVideoCodec_YV12  == rVideoFormat.codec ||
           cudaVideoCodec_NV12  == rVideoFormat.codec ||
           cudaVideoCodec_YUYV  == rVideoFormat.codec ||
           cudaVideoCodec_UYVY  == rVideoFormat.codec);

    assert(cudaVideoChromaFormat_Monochrome == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_420        == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_422        == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_444        == rVideoFormat.chroma_format);

    // Fill the decoder-create-info struct from the given video-format struct.
    memset(&oVideoDecodeCreateInfo_, 0, sizeof(CUVIDDECODECREATEINFO));
    // Create video decoder
    oVideoDecodeCreateInfo_.CodecType           = rVideoFormat.codec;
    oVideoDecodeCreateInfo_.ulWidth             = rVideoFormat.coded_width;
    oVideoDecodeCreateInfo_.ulHeight            = rVideoFormat.coded_height;
    oVideoDecodeCreateInfo_.ulNumDecodeSurfaces = FrameQueue::cnMaximumSize;

    // Limit decode memory to 24MB (16M pixels at 4:2:0 = 24M bytes)
    while (oVideoDecodeCreateInfo_.ulNumDecodeSurfaces * rVideoFormat.coded_width * rVideoFormat.coded_height > 16*1024*1024)
    {
        oVideoDecodeCreateInfo_.ulNumDecodeSurfaces--;
    }

    oVideoDecodeCreateInfo_.ChromaFormat        = rVideoFormat.chroma_format;
    oVideoDecodeCreateInfo_.OutputFormat        = cudaVideoSurfaceFormat_NV12;
    oVideoDecodeCreateInfo_.DeinterlaceMode     = cudaVideoDeinterlaceMode_Adaptive;

    // No scaling
    oVideoDecodeCreateInfo_.ulTargetWidth       = oVideoDecodeCreateInfo_.ulWidth;
    oVideoDecodeCreateInfo_.ulTargetHeight      = oVideoDecodeCreateInfo_.ulHeight;
    oVideoDecodeCreateInfo_.ulNumOutputSurfaces = MAX_FRAME_COUNT;  // We won't simultaneously map more than 8 surfaces
    oVideoDecodeCreateInfo_.ulCreationFlags     = m_VideoCreateFlags;
    oVideoDecodeCreateInfo_.vidLock             = vidCtxLock;
    // create the decoder
    CUresult oResult = cuvidCreateDecoder(&oDecoder_, &oVideoDecodeCreateInfo_);
    assert(CUDA_SUCCESS == oResult);
}

VideoDecoder::~VideoDecoder()
{
    cuvidDestroyDecoder(oDecoder_);
}

cudaVideoCodec
VideoDecoder::codec()
const
{
    return oVideoDecodeCreateInfo_.CodecType;
}

cudaVideoChromaFormat
VideoDecoder::chromaFormat()
const
{
    return oVideoDecodeCreateInfo_.ChromaFormat;
}

unsigned long
VideoDecoder::maxDecodeSurfaces()
const
{
    return oVideoDecodeCreateInfo_.ulNumDecodeSurfaces;
}

unsigned long
VideoDecoder::frameWidth()
const
{
    return oVideoDecodeCreateInfo_.ulWidth;
}

unsigned long
VideoDecoder::frameHeight()
const
{
    return oVideoDecodeCreateInfo_.ulHeight;
}

unsigned long
VideoDecoder::targetWidth()
const
{
    return oVideoDecodeCreateInfo_.ulTargetWidth;
}

unsigned long
VideoDecoder::targetHeight()
const
{
    return oVideoDecodeCreateInfo_.ulTargetHeight;
}

void
VideoDecoder::decodePicture(CUVIDPICPARAMS *pPictureParameters, CUcontext *pContext)
{
    // Handle CUDA picture decode (this actually calls the hardware VP/CUDA to decode video frames)
    CUresult oResult = cuvidDecodePicture(oDecoder_, pPictureParameters);
    assert(CUDA_SUCCESS == oResult);
}

void
VideoDecoder::mapFrame(int iPictureIndex, CUdeviceptr *ppDevice, unsigned int *pPitch, CUVIDPROCPARAMS *pVideoProcessingParameters)
{
    CUresult oResult = cuvidMapVideoFrame(oDecoder_,
                                          iPictureIndex,
                                          ppDevice,
                                          pPitch, pVideoProcessingParameters);
    assert(CUDA_SUCCESS == oResult);
    assert(0 != *ppDevice);
    assert(0 != *pPitch);
}

void
VideoDecoder::unmapFrame(CUdeviceptr pDevice)
{
    CUresult oResult = cuvidUnmapVideoFrame(oDecoder_, pDevice);
    assert(CUDA_SUCCESS == oResult);
}

