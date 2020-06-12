/*
 * Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj;
} ubo;

layout(location = 0) in float height;
layout(location = 1) in vec2 xyPos;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.modelViewProj * vec4(xyPos.xy, height, 1.0f);
    fragColor = vec3(0.0f, (height + 0.5f), 0.0f);
}
