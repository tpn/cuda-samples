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
#ifndef VFLOCKING_D3D10_H
#define VFLOCKING_D3D10_H

#pragma once

// simulation parameters
struct Params
{
    float alpha ;
    float upwashX ;
    float upwashY ;
    float wingspan ;
    float dX ;
    float dY ;
    float epsilon ;
    float lambda ; // -0.1073f * wingspan ;
} ;

#endif
