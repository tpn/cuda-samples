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

#ifndef NV_IMAGE_DX
#define NV_IMAGE_DX

#include <cuda.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <d3dx9.h>

const int Format2Bpp[] = { 1, 4, 0 };

class ImageDX
{
    public:
        enum PixelFormat
        {
            LUMINANCE_PIXEL_FORMAT,
            BGRA_PIXEL_FORMAT,
            UNKNOWN_PIXEL_FORMAT
        };

        ImageDX(IDirect3DDevice9 *pDeviceD3D,
                unsigned int nDispWidth, unsigned int nDispHeight,
                unsigned int nTexWidth,  unsigned int nTexHeight,
                bool bIsProgressive,
                PixelFormat ePixelFormat = BGRA_PIXEL_FORMAT);

        // Destructor
        ~ImageDX();

        void
        registerAsCudaResource(int field_num);

        void
        unregisterAsCudaResource(int field_num);

        bool
        isCudaResource()
        const;

        void
        setCUDAcontext(CUcontext oContext);

        void
        setCUDAdevice(CUdevice oDevice);

        int Bpp()
        {
            return Format2Bpp[(int)e_PixFmt_];
        }

        // Map this image's DX surface into CUDA memory space.
        // Parameters:
        //      ppImageData - point to point to image data. On return this
        //          pointer references the mapped data.
        //      pImagePitch - pointer to image pitch. On return of this
        //          pointer contains the pitch of the mapped image surface.
        // Note:
        //      This method will fail, if this image is not a registered CUDA resource.
        void
        map(CUdeviceptr *ppImageData, size_t *pImagePitch, int active_field = 0);

        void
        unmap(int active_field = 0);

        // Clear the image.
        // Parameters:
        //      nClearColor - the luminance value to clear the image to. Default is white.
        // Note:
        //      This method will not work if this image is not registered as a CUDA resource at the
        //      time of this call.
        void
        clear(unsigned char nClearColor = 0xff);

        unsigned int
        width()
        const;

        unsigned int
        height()
        const;

        void
        render(int active_field = 0)
        const;

    private:
        static
        D3DFORMAT
        d3dFormat(PixelFormat ePixelFormat);


        struct VertexStruct
        {
            VertexStruct() {};
            VertexStruct(float nX, float nY, float nZ, float nU, float nV);
            float position[3];
            float texture[2];
        };


        VertexStruct aVertexBuffer_[4];

        static const unsigned int aIndexBuffer_[6];

        IDirect3DDevice9   *pDeviceD3D_;
        IDirect3DTexture9 *pTexture_[2];
        IDirect3DSurface9 *pSurface_[2];

        unsigned int nWidth_;
        unsigned int nHeight_;
        unsigned int nTexWidth_;
        unsigned int nTexHeight_;
        PixelFormat e_PixFmt_;

        bool bIsCudaResource_;
        bool bIsProgressive_;

        CUcontext oContext_;
        CUdevice  oDevice_;
};

#endif // NV_IMAGE_DX