#pragma once

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

#define GL4METAL_DEBUG 1
#define GL4METAL_VERSION "1.0"

#if GL4METAL_DEBUG
#define GL4METAL_LOG(format, ...) printf("[GL4Metal] " format "\n", ##__VA_ARGS__)
#else
#define GL4METAL_LOG(...)
#endif

#define GL_TEXTURE_BUFFER_FORMAT GL_TEXTURE_BUFFER_FORMAT_ARB

// iOS limits (Apple GPU family 3+ / A9 and newer)
#define GL4METAL_MAX_TEXTURE_SIZE 16384
#define GL4METAL_MAX_3D_TEXTURE_SIZE 2048
#define GL4METAL_MAX_ARRAY_TEXTURE_LAYERS 2048
#define GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS 36
// Metal buffer slots: vertex attributes 0...15, fragment gl_FragCoord 15,
// graphics/compute SSBO and UBO ranges 0...28, SPIRV-Cross OpArrayLength 29,
// and GL loose-uniform blocks 30. Vertex SSBO/UBO slots are limited to 16...28.
#define GL4METAL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 1024

// Metal buffer indices shared by GL buffers and SPIRV-Cross helper resources.
// Vertex attributes occupy vertex slots 0...15; fragment slot 15 is gl_FragCoord.
// Slot 29 carries OpArrayLength byte sizes and slot 30 carries GL loose uniforms.
#define GL4METAL_BUFFER_SIZE_BUFFER_INDEX 29
#define GL4METAL_UNIFORM_BUFFER_INDEX 30
// Metal/iOS supports up to eight color attachments on the target devices.
#define GL4METAL_MAX_FRAGMENT_OUTPUTS 8

// Iris uses GL_MAX_TEXTURE_IMAGE_UNITS as a logical dynamic-sampler allocator
// capacity. Those logical units are virtualized to dense MSL slots at draw time.
// Keep 96 fragment-side logical units plus 16 vertex-side logical units in the
// combined GL namespace; actual active MSL sampler slots remain 16 per stage.
#define GL4METAL_MAX_TEXTURE_UNITS 112
#define GL4METAL_MAX_FRAGMENT_TEXTURE_UNIT_BINDINGS 96
#define GL4METAL_MAX_IMAGE_UNITS 32
#define GL4METAL_MAX_COMPUTE_TEXTURE_UNITS 16
#define GL4METAL_MAX_FRAGMENT_TEXTURE_UNITS 16
#define GL4METAL_MAX_VERTEX_TEXTURE_UNITS 16
#define GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 16
// Reserved fragment-stage buffer slot for the dynamic GL gl_FragCoord coordinate fixup.
#define GL4METAL_FRAGCOORD_BUFFER_INDEX 15

#define GL4METAL_MAX_COLOR_ATTACHMENTS GL4METAL_MAX_FRAGMENT_OUTPUTS