/*
 * Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once
#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////
//! Allocate virtually contiguous memory backed on separate devices
//! @return CUresult error code on failure.
//! @param[out] dptr            Virtual address reserved for allocation
//! @param[out] allocationSize  Actual amount of virtual address space reserved.
//!                             AllocationSize is needed in the free operation.
//! @param[in] size             The minimum size to allocate (will be rounded up to accomodate
//!                             required granularity).
//! @param[in] residentDevices  Specifies what devices the allocation should be striped across.
//! @param[in] mappingDevices   Specifies what devices need to read/write to the allocation.
//! @align                      Additional allignment requirement if desired.
//! @note       The VA mappings will look like the following:
//!
//!     v-stripeSize-v                v-rounding -v
//!     +-----------------------------------------+
//!     |      D1     |      D2     |      D3     |
//!     +-----------------------------------------+
//!     ^-- dptr                      ^-- dptr + size
//!
//! Each device in the residentDevices list will get an equal sized stripe.
//! Excess memory allocated will be  that meets the minimum
//! granularity requirements of all the devices.
//!
//! @note uses cuMemGetAllocationGranularity cuMemCreate cuMemMap and cuMemSetAccess
//!   function calls to organize the va space
//!
//! @note uses cuMemRelease to release the allocationHandle.  The allocation handle
//!   is not needed after its mappings are set up.
////////////////////////////////////////////////////////////////////////////
CUresult
simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t *allocationSize, size_t size,
         const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
         size_t align = 0);


////////////////////////////////////////////////////////////////////////////
//! Frees resources allocated by simpleMallocMultiDeviceMmap
//! @CUresult CUresult error code on failure.
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmap
////////////////////////////////////////////////////////////////////////////
CUresult
simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size);
