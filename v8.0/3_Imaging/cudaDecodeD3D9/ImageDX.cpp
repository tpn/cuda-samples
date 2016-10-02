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

#include "ImageDX.h"

#include <cuda.h>
#include <cudad3d9.h>

#include <cassert>

#include "helper_cuda_drvapi.h"

const unsigned int ImageDX::aIndexBuffer_[6] = {0, 1, 2, 0, 2, 3};

ImageDX::ImageDX(IDirect3DDevice9 *pDeviceD3D,
                 unsigned int nDispWidth, unsigned int nDispHeight,
                 unsigned int nTexWidth,  unsigned int nTexHeight,
                 bool bIsProgressive,
                 PixelFormat ePixelFormat) :
    pDeviceD3D_(pDeviceD3D)
    , nWidth_(nDispWidth)
    , nHeight_(nDispHeight)
    , nTexWidth_(nTexWidth)
    , nTexHeight_(nTexHeight)
    , bIsProgressive_(bIsProgressive)
    , bIsCudaResource_(false)
{
    assert(0 != pDeviceD3D_);

    int nFrames = bIsProgressive_ ? 1 : 2;

    pTexture_[0] = pTexture_[1] = 0;
    pSurface_[0] = pSurface_[1] = 0;

    for (int field_num = 0; field_num < nFrames; field_num++)
    {
        HRESULT hResult = pDeviceD3D_->CreateTexture(nWidth_, nHeight_, 1, 0, d3dFormat(ePixelFormat), D3DPOOL_DEFAULT, &pTexture_[field_num], 0);
        assert(S_OK == hResult);
        assert(0 != pTexture_[field_num]);
        hResult = pTexture_[field_num]->GetSurfaceLevel(0, &pSurface_[field_num]);
        assert(S_OK == hResult);
        assert(0 != pSurface_[field_num]);
        registerAsCudaResource(field_num);
    }

    aVertexBuffer_[0] = VertexStruct(0.0f,           0.0f,            0.0f, 0.0f, 1.0f);
    aVertexBuffer_[1] = VertexStruct((float)nWidth_, 0.0f,            0.0f, 1.0f, 1.0f);
    aVertexBuffer_[2] = VertexStruct((float)nWidth_, (float)nHeight_, 0.0f, 1.0f, 0.0f);
    aVertexBuffer_[3] = VertexStruct(0.0f, (float)nHeight_, 0.0f, 0.0f, 0.0f);

}

ImageDX::~ImageDX()
{
    int nFrames = bIsProgressive_ ? 1 : 2;

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        unregisterAsCudaResource(field_num);
        pSurface_[field_num]->Release();
        pTexture_[field_num]->Release();
    }
}

void
ImageDX::registerAsCudaResource(int field_num)
{
    // register the Direct3D resources that we'll use
    checkCudaErrors(cuD3D9RegisterResource(pTexture_[field_num], 0));
    getLastCudaDrvErrorMsg("cudaD3D9RegisterResource (pTexture_) failed");

    bIsCudaResource_ = true;

    // we will be write directly to this 2D texture, so we must set the
    // appropriate flags (to eliminate extra copies during map, but unmap will do copies)
    checkCudaErrors(cuD3D9ResourceSetMapFlags(pTexture_[field_num], CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD));
}

void
ImageDX::unregisterAsCudaResource(int field_num)
{
    CUresult result = cuCtxPushCurrent(oContext_);
    checkCudaErrors(cuD3D9UnregisterResource(pTexture_[field_num]));
    bIsCudaResource_ = false;
    cuCtxPopCurrent(NULL);
}

void
ImageDX::setCUDAcontext(CUcontext oContext)
{
    oContext_ = oContext;
    printf("ImageDX::CUcontext = %08x\n", (int)oContext);
}

void
ImageDX::setCUDAdevice(CUdevice oDevice)
{
    oDevice_ = oDevice;
    printf("ImageDX::CUdevice  = %08x\n", (int)oDevice);
}

bool
ImageDX::isCudaResource()
const
{
    return bIsCudaResource_;
}

void
ImageDX::map(CUdeviceptr *ppImageData, size_t *pImagePitch, int active_field)
{
    int nFrames = bIsProgressive_ ? 1 : 2;

    checkCudaErrors(cuD3D9MapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(pTexture_)));
    checkCudaErrors(cuD3D9ResourceGetMappedPointer(ppImageData, pTexture_[active_field], 0, 0));
    assert(0 != *ppImageData);
    checkCudaErrors(cuD3D9ResourceGetMappedPitch(pImagePitch, NULL, pTexture_[active_field], 0, 0));
    assert(0 != *pImagePitch);
}

void
ImageDX::unmap(int active_field)
{
    int nFrames = bIsProgressive_ ? 1 : 2;

    checkCudaErrors(cuD3D9UnmapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(&pTexture_)));
}

void
ImageDX::clear(unsigned char nClearColor)
{
    // Can only be cleared if surface is a CUDA resource
    assert(bIsCudaResource_);

    int nFrames = bIsProgressive_ ? 1 : 2;
    CUdeviceptr  pData = 0;
    size_t       nSize = 0;

    checkCudaErrors(cuD3D9MapResources(nFrames, (IDirect3DResource9 **)&pTexture_));

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        checkCudaErrors(cuD3D9ResourceGetMappedPointer(&pData, pTexture_[field_num], 0, 0));
        assert(0 != pData);
        checkCudaErrors(cuD3D9ResourceGetMappedSize(&nSize, pTexture_[field_num], 0, 0));
        assert(0 != nSize);

        // clear the surface to solid white
        checkCudaErrors(cuMemsetD8(pData, nClearColor, nSize));
    }

    checkCudaErrors(cuD3D9UnmapResources(nFrames, (IDirect3DResource9 **)&pTexture_));
}

unsigned int
ImageDX::width()
const
{
    return nWidth_;
}

unsigned int
ImageDX::height()
const
{
    return nHeight_;
}

void
ImageDX::render(int active_field)
const
{
    // use this index and vertex data throughout
    // initialize the scene
    pDeviceD3D_->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    pDeviceD3D_->SetRenderState(D3DRS_LIGHTING, FALSE);
    // pDeviceD3D_->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
    pDeviceD3D_->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
    pDeviceD3D_->SetFVF(D3DFVF_XYZ|D3DFVF_TEX1|D3DFVF_TEXCOORDSIZE2(0));
    // pDeviceD3D_->SetFVF(D3DFVF_XYZRHW|D3DFVF_TEX1|D3DFVF_TEXCOORDSIZE2(0));
    // draw the 2d texture
    pDeviceD3D_->SetTexture(0, pTexture_[active_field]);
    pDeviceD3D_->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
    pDeviceD3D_->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);
    pDeviceD3D_->SetSamplerState(0, D3DSAMP_BORDERCOLOR, 0xFFFFFFFF); // set border to solid white
    pDeviceD3D_->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, aIndexBuffer_, D3DFMT_INDEX32, aVertexBuffer_, sizeof(VertexStruct));
}

D3DFORMAT
ImageDX::d3dFormat(PixelFormat ePixelFormat)
{
    switch (ePixelFormat)
    {
        case LUMINANCE_PIXEL_FORMAT:
            return D3DFMT_L8;

        case BGRA_PIXEL_FORMAT:
            return D3DFMT_A8R8G8B8;

        case UNKNOWN_PIXEL_FORMAT:
            assert(false);

        default:
            assert(false);
    }

    assert(false);
    return D3DFMT_UNKNOWN;
}

ImageDX::VertexStruct::VertexStruct(float nX, float nY, float nZ, float nU, float nV)
{
    position[0] = nX;
    position[1] = nY;
    position[2] = nZ;

    texture[0]  = nU;
    texture[1]  = nV;
}

