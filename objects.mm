#import "gl4metal.h"
#import "gl4metal_runtime.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <math.h>

// Legacy OpenGL texture formats are not declared by the active GL header,
// but Minecraft/older GL paths may still use them with glTexImage2D.
#ifndef GL_LUMINANCE
#define GL_LUMINANCE 0x1909
#endif
#ifndef GL_LUMINANCE_ALPHA
#define GL_LUMINANCE_ALPHA 0x190A
#endif

#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
#ifndef GL_DEPTH_ATTACHMENT
#define GL_DEPTH_ATTACHMENT 0x8D00
#endif
#ifndef GL_STENCIL_ATTACHMENT
#define GL_STENCIL_ATTACHMENT 0x8D20
#endif
#ifndef GL_DEPTH_STENCIL_ATTACHMENT
#define GL_DEPTH_STENCIL_ATTACHMENT 0x821A
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
#ifndef GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT 0x8CD6
#endif
#ifndef GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT 0x8CD7
#endif
#ifndef GL_DEPTH_COMPONENT
#define GL_DEPTH_COMPONENT 0x1902
#endif
#ifndef GL_DEPTH_COMPONENT16
#define GL_DEPTH_COMPONENT16 0x81A5
#endif
#ifndef GL_DEPTH_COMPONENT24
#define GL_DEPTH_COMPONENT24 0x81A6
#endif
#ifndef GL_DEPTH_COMPONENT32F
#define GL_DEPTH_COMPONENT32F 0x8CAC
#endif
#ifndef GL_DEPTH24_STENCIL8
#define GL_DEPTH24_STENCIL8 0x88F0
#endif
#ifndef GL_STENCIL_INDEX8
#define GL_STENCIL_INDEX8 0x8D48
#endif

// -----------------------------------------------------------------------------
// Texture state / OpenGL -> Metal bridge
// -----------------------------------------------------------------------------
// The GL texture entry points already belong in objects.mm.  Keep that API
// ownership here and expose only small bridge functions to gl4metal.mm.

static const NSUInteger GL4METAL_MAX_TEXTURE_UNITS = 32;

static NSMutableDictionary<NSNumber *, id<MTLTexture>> *gl4metalTextureObjects = nil;
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSDictionary<NSString *, NSNumber *> *> *> *gl4metalTextureLevelInfo = nil;
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSNumber *> *> *gl4metalTextureParams = nil;
static NSMutableDictionary<NSNumber *, NSData *> *gl4metalTextureBorderColors = nil;
static NSMutableDictionary<NSNumber *, id<MTLSamplerState>> *gl4metalTextureSamplers = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalTextureUnitBindings = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalTexture1DUnitBindings = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalTextureTargets = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalSamplerUnitBindings = nil;
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSNumber *> *> *gl4metalSamplerParams = nil;
static NSMutableDictionary<NSNumber *, NSData *> *gl4metalSamplerBorderColors = nil;
static NSMutableDictionary<NSNumber *, id<MTLSamplerState>> *gl4metalSamplerStates = nil;
static NSMutableSet<NSNumber *> *gl4metalTextureObjectNames = nil;
static NSMutableSet<NSNumber *> *gl4metalTextureGeneratedNames = nil;
static NSMutableSet<NSNumber *> *gl4metalSamplerGeneratedNames = nil;
static NSMutableSet<NSNumber *> *gl4metalSamplerObjectNames = nil;
static GLuint gl4metalNextTextureId = 1;
static GLuint gl4metalNextSamplerId = 1;
static GLuint gl4metalActiveTextureUnit = 0;
static GLint gl4metalUnpackAlignment = 4;
static GLint gl4metalUnpackRowLength = 0;
static GLint gl4metalUnpackSkipRows = 0;
static GLint gl4metalUnpackSkipPixels = 0;
static GLint gl4metalPackAlignment = 4;
static GLint gl4metalPackRowLength = 0;
static GLint gl4metalPackSkipRows = 0;
static GLint gl4metalPackSkipPixels = 0;
static GLint gl4metalUnpackSwapBytes = GL_FALSE;
static GLint gl4metalUnpackLSBFirst = GL_FALSE;
static GLint gl4metalPackSwapBytes = GL_FALSE;
static GLint gl4metalPackLSBFirst = GL_FALSE;

// Forward declaration: texture deletion can detach FBO attachments below.
static void gl4metalDetachTextureFromAllFramebuffers(GLuint texture);

static BOOL gl4metalIsDrawFramebufferTarget(GLenum target) {
    return target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER;
}

static BOOL gl4metalIsSamplerScalarParameter(GLenum pname);
static BOOL gl4metalValidSamplerParameter(GLenum pname, GLfloat value);
static NSData *gl4metalGetTextureBorderData(GLuint texture);

// Real framebuffer/renderbuffer state. GL texture names and renderbuffer names
// remain separate namespaces just like OpenGL; the Metal resources are stored
// in their corresponding dictionaries.
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferColor = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferDepth = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferStencil = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferColorRB = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferDepthRB = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferStencilRB = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferColorLevel = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferDepthLevel = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferStencilLevel = nil;
static NSMutableDictionary<NSNumber *, id<MTLTexture>> *gl4metalRenderbufferTextures = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalRenderbufferFormats = nil;
static GLuint gl4metalCurrentRenderbuffer = 0;
static NSMutableSet<NSNumber *> *gl4metalFramebufferGeneratedNames = nil;
static NSMutableSet<NSNumber *> *gl4metalRenderbufferGeneratedNames = nil;
static GLuint gl4metalNextFramebufferId = 1;
static GLuint gl4metalNextRenderbufferId = 1;

static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalEnsureTextureParams(GLuint texture) {
    if (!gl4metalTextureParams) {
        gl4metalTextureParams = [NSMutableDictionary dictionary];
    }

    NSMutableDictionary<NSNumber *, NSNumber *> *params = gl4metalTextureParams[@(texture)];
    if (!params) {
        params = [@{
            @(GL_TEXTURE_MIN_FILTER): @(GL_NEAREST_MIPMAP_LINEAR),
            @(GL_TEXTURE_MAG_FILTER): @(GL_LINEAR),
            @(GL_TEXTURE_WRAP_S): @(GL_REPEAT),
            @(GL_TEXTURE_WRAP_T): @(GL_REPEAT),
            @(GL_TEXTURE_WRAP_R): @(GL_REPEAT),
            @(GL_TEXTURE_MIN_LOD): @(-1000.0f),
            @(GL_TEXTURE_MAX_LOD): @(1000.0f),
            @(GL_TEXTURE_LOD_BIAS): @(0.0f),
            @(GL_TEXTURE_BASE_LEVEL): @0,
            @(GL_TEXTURE_MAX_LEVEL): @1000,
            @(GL_TEXTURE_COMPARE_MODE): @(GL_NONE),
            @(GL_TEXTURE_COMPARE_FUNC): @(GL_LEQUAL)
        } mutableCopy];
        gl4metalTextureParams[@(texture)] = params;
    }
    return params;
}

static void gl4metalEnsureTextureState(void) {
    if (!gl4metalTextureObjects) gl4metalTextureObjects = [NSMutableDictionary dictionary];
    if (!gl4metalTextureLevelInfo) gl4metalTextureLevelInfo = [NSMutableDictionary dictionary];
    if (!gl4metalTextureParams) gl4metalTextureParams = [NSMutableDictionary dictionary];
    if (!gl4metalTextureBorderColors) gl4metalTextureBorderColors = [NSMutableDictionary dictionary];
    if (!gl4metalTextureSamplers) gl4metalTextureSamplers = [NSMutableDictionary dictionary];
    if (!gl4metalTextureUnitBindings) gl4metalTextureUnitBindings = [NSMutableDictionary dictionary];
    if (!gl4metalTexture1DUnitBindings) gl4metalTexture1DUnitBindings = [NSMutableDictionary dictionary];
    if (!gl4metalTextureTargets) gl4metalTextureTargets = [NSMutableDictionary dictionary];
    if (!gl4metalSamplerUnitBindings) gl4metalSamplerUnitBindings = [NSMutableDictionary dictionary];
    if (!gl4metalSamplerParams) gl4metalSamplerParams = [NSMutableDictionary dictionary];
    if (!gl4metalSamplerBorderColors) gl4metalSamplerBorderColors = [NSMutableDictionary dictionary];
    if (!gl4metalSamplerStates) gl4metalSamplerStates = [NSMutableDictionary dictionary];
    if (!gl4metalTextureObjectNames) gl4metalTextureObjectNames = [NSMutableSet set];
    if (!gl4metalTextureGeneratedNames) gl4metalTextureGeneratedNames = [NSMutableSet set];
    if (!gl4metalSamplerGeneratedNames) gl4metalSamplerGeneratedNames = [NSMutableSet set];
    if (!gl4metalSamplerObjectNames) gl4metalSamplerObjectNames = [NSMutableSet set];
}

static GLuint gl4metalTextureBoundToUnit(GLuint unit) {
    gl4metalEnsureTextureState();
    NSNumber *value = gl4metalTextureUnitBindings[@(unit)];
    return value ? value.unsignedIntValue : 0;
}

static GLuint gl4metalTextureBoundToUnitForTarget(GLuint unit, GLenum target) {
    gl4metalEnsureTextureState();
    NSDictionary *bindings = target == GL_TEXTURE_1D
        ? gl4metalTexture1DUnitBindings : gl4metalTextureUnitBindings;
    return [bindings[@(unit)] unsignedIntValue];
}

static size_t gl4metalBytesPerPixelForFormatType(GLenum format, GLenum type) {
    if (type != GL_UNSIGNED_BYTE) return 0;
    switch (format) {
        case GL_RGBA:
        case GL_BGRA:
            return 4;
        case GL_RGB:
        case GL_BGR:
            return 3;
        case GL_RG:
            return 2;
        case GL_RED:
        case GL_GREEN:
        case GL_BLUE:
        case GL_ALPHA:
        case GL_LUMINANCE:
            return 1;
        case GL_LUMINANCE_ALPHA:
            return 2;
        default:
            return 0;
    }
}

static BOOL gl4metalUploadPixelsToTexture(id<MTLTexture> texture,
                                          GLint level,
                                          GLint xoffset,
                                          GLint yoffset,
                                          GLsizei width,
                                          GLsizei height,
                                          GLenum internalFormat,
                                          GLenum format,
                                          GLenum type,
                                          const void *pixels) {
    if (!texture || width <= 0 || height <= 0 || level < 0) return NO;
    if (level >= (GLint)texture.mipmapLevelCount) return NO;

    NSUInteger levelWidth = MAX((NSUInteger)1, texture.width >> level);
    NSUInteger levelHeight = MAX((NSUInteger)1, texture.height >> level);
    if ((NSUInteger)xoffset + (NSUInteger)width > levelWidth) return NO;
    if ((NSUInteger)yoffset + (NSUInteger)height > levelHeight) return NO;

    // Keep texture uploads in the same row order supplied by the GL caller.
    // Minecraft's ordinary 2D textures (Sampler0/Sampler2/etc.) must not be
    // globally flipped here. Framebuffer-origin conversion is handled at the
    // compositor shader for DiffuseSampler instead, because framebuffer-backed
    // textures have a different origin relationship than ordinary GL textures.
    if (!pixels) return YES;

    size_t sourceBpp = gl4metalBytesPerPixelForFormatType(format, type);
    if (sourceBpp == 0) return NO;

    if ((size_t)width > SIZE_MAX / (size_t)height) return NO;
    size_t pixelCount = (size_t)width * (size_t)height;
    if (pixelCount > SIZE_MAX / 4u) return NO;
    size_t sourceRowPixels = gl4metalUnpackRowLength > 0
        ? (size_t)gl4metalUnpackRowLength : (size_t)width;
    if (sourceRowPixels > SIZE_MAX / sourceBpp) return NO;
    size_t unalignedRowBytes = sourceRowPixels * sourceBpp;
    size_t alignment = (size_t)gl4metalUnpackAlignment;
    if (unalignedRowBytes > SIZE_MAX - (alignment - 1)) return NO;
    size_t sourceRowBytes = (unalignedRowBytes + alignment - 1) & ~(alignment - 1);
    if ((size_t)gl4metalUnpackSkipRows > SIZE_MAX / MAX((size_t)1, sourceRowBytes) ||
        (size_t)gl4metalUnpackSkipPixels > SIZE_MAX / sourceBpp) return NO;
    size_t skipRowBytes = (size_t)gl4metalUnpackSkipRows * sourceRowBytes;
    size_t skipPixelBytes = (size_t)gl4metalUnpackSkipPixels * sourceBpp;
    if (skipRowBytes > SIZE_MAX - skipPixelBytes) return NO;
    size_t sourceStart = skipRowBytes + skipPixelBytes;
    size_t inputRowBytes = (size_t)width * sourceBpp;
    if ((size_t)(height - 1) > (SIZE_MAX - sourceStart) / MAX((size_t)1, sourceRowBytes) ||
        inputRowBytes > SIZE_MAX - (sourceStart + (size_t)(height - 1) * sourceRowBytes)) return NO;
    NSMutableData *rgba = [NSMutableData dataWithLength:pixelCount * 4];
    if (!rgba) return NO;

    const uint8_t *src = (const uint8_t *)pixels;
    uint8_t *dst = (uint8_t *)rgba.mutableBytes;

    for (size_t row = 0; row < (size_t)height; ++row) {
        const uint8_t *srcRow = src + sourceStart + row * sourceRowBytes;
        uint8_t *dstRow = dst + row * (size_t)width * 4;

        for (size_t x = 0; x < (size_t)width; ++x) {
            const uint8_t *p = srcRow + x * sourceBpp;
            uint8_t *q = dstRow + x * 4;

            if (format == GL_RGBA && sourceBpp == 4) {
                q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = p[3];
            } else if (format == GL_BGRA && sourceBpp == 4) {
                q[0] = p[2]; q[1] = p[1]; q[2] = p[0]; q[3] = p[3];
            } else if (format == GL_RGB && sourceBpp == 3) {
                q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = 255;
            } else if (format == GL_BGR && sourceBpp == 3) {
                q[0] = p[2]; q[1] = p[1]; q[2] = p[0]; q[3] = 255;
            } else if (format == GL_RED && sourceBpp == 1) {
                q[0] = p[0]; q[1] = 0; q[2] = 0; q[3] = 255;
            } else if (format == GL_GREEN && sourceBpp == 1) {
                q[0] = 0; q[1] = p[0]; q[2] = 0; q[3] = 255;
            } else if (format == GL_BLUE && sourceBpp == 1) {
                q[0] = 0; q[1] = 0; q[2] = p[0]; q[3] = 255;
            } else if (format == GL_ALPHA && sourceBpp == 1) {
                q[0] = 0; q[1] = 0; q[2] = 0; q[3] = p[0];
            } else if (format == GL_RG && sourceBpp == 2) {
                q[0] = p[0]; q[1] = p[1]; q[2] = 0; q[3] = 255;
            } else if (format == GL_LUMINANCE && sourceBpp == 1) {
                q[0] = p[0]; q[1] = p[0]; q[2] = p[0]; q[3] = 255;
            } else if (format == GL_LUMINANCE_ALPHA && sourceBpp == 2) {
                q[0] = p[0]; q[1] = p[0]; q[2] = p[0]; q[3] = p[1];
            } else {
                return NO;
            }
            if (internalFormat == GL_LUMINANCE || internalFormat == GL_LUMINANCE_ALPHA) {
                uint8_t luminance = (uint8_t)lrint((0.30 * q[0]) + (0.59 * q[1]) + (0.11 * q[2]));
                q[0] = q[1] = q[2] = luminance;
                if (internalFormat == GL_LUMINANCE) q[3] = 255;
            } else if (internalFormat == GL_ALPHA) {
                q[0] = q[1] = q[2] = 0;
            } else if (internalFormat == GL_RED || internalFormat == GL_R8) {
                q[1] = q[2] = 0;
                q[3] = 255;
            } else if (internalFormat == GL_RG || internalFormat == GL_RG8) {
                q[2] = 0;
                q[3] = 255;
            } else if (internalFormat == GL_RGB || internalFormat == GL_RGB8 ||
                       internalFormat == GL_RGB565) {
                q[3] = 255;
            }
        }
    }

    // Do not CPU-write an MTLTexture that a previous command buffer may still
    // be sampling or rendering. Stage the pixels into a shared MTLBuffer and
    // encode the update on the current Metal command buffer. If there is no
    // current frame command buffer, submit a dedicated upload command buffer;
    // the same command queue preserves ordering for subsequent rendering.
    id<MTLCommandBuffer> cb = ctx.currentCommandBuffer;
    BOOL dedicated = NO;
    if (!cb) {
        cb = [ctx.commandQueue commandBuffer];
        dedicated = YES;
    }
    if (!cb) return NO;

    id<MTLBuffer> staging = [ctx.device newBufferWithBytes:rgba.bytes
                                                       length:rgba.length
                                                      options:MTLResourceStorageModeShared];
    if (!staging) return NO;

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return NO;

    [blit copyFromBuffer:staging
            sourceOffset:0
        sourceBytesPerRow:(NSUInteger)width * 4
      sourceBytesPerImage:(NSUInteger)width * (NSUInteger)height * 4
               sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
                toTexture:texture
         destinationSlice:0
         destinationLevel:(NSUInteger)level
        destinationOrigin:MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, 0)];
    [blit endEncoding];

    if (dedicated) {
        cb.label = [NSString stringWithFormat:@"gl4metal texture upload"];
        [cb commit];
    }

    if (getenv("GL4METAL_RESOURCE_UPDATE_TRACE")) {
        printf("[gl4metal][RESOURCE_UPDATE] texture upload staged level=%d rect=%d,%d %dx%d dedicated=%d\n",
               level, xoffset, yoffset, width, height, dedicated);
    }
    return YES;
}

static BOOL gl4metalUploadDepthPixels(id<MTLTexture> texture, GLint level,
                                     GLint xoffset, GLint yoffset,
                                     GLsizei width, GLsizei height, GLenum type,
                                     const void *pixels) {
    if (!texture || texture.pixelFormat != MTLPixelFormatDepth32Float || level != 0 ||
        xoffset < 0 || yoffset < 0 || width <= 0 || height <= 0) return NO;
    if ((NSUInteger)xoffset > texture.width || (NSUInteger)yoffset > texture.height ||
        (NSUInteger)width > texture.width - (NSUInteger)xoffset ||
        (NSUInteger)height > texture.height - (NSUInteger)yoffset) return NO;
    if (type != GL_FLOAT && type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) return NO;
    if (!pixels) return YES;
    size_t inputBytes = type == GL_UNSIGNED_SHORT ? sizeof(GLushort) : sizeof(GLuint);
    NSUInteger inputRowPixels = gl4metalUnpackRowLength > 0
        ? (NSUInteger)gl4metalUnpackRowLength : (NSUInteger)width;
    if (inputRowPixels > SIZE_MAX / inputBytes) return NO;
    NSUInteger inputRowBytes = inputRowPixels * inputBytes;
    NSUInteger alignment = (NSUInteger)gl4metalUnpackAlignment;
    if (inputRowBytes > SIZE_MAX - (alignment - 1)) return NO;
    inputRowBytes = (inputRowBytes + alignment - 1) & ~(alignment - 1);
    NSUInteger outputRowBytes = (NSUInteger)width * sizeof(float);
    if ((NSUInteger)height > SIZE_MAX / outputRowBytes) return NO;
    NSMutableData *converted = [NSMutableData dataWithLength:outputRowBytes * (NSUInteger)height];
    if (!converted) return NO;
    if ((NSUInteger)gl4metalUnpackSkipRows > SIZE_MAX / MAX((NSUInteger)1, inputRowBytes) ||
        (NSUInteger)gl4metalUnpackSkipPixels > SIZE_MAX / inputBytes) return NO;
    NSUInteger skipRowBytes = (NSUInteger)gl4metalUnpackSkipRows * inputRowBytes;
    NSUInteger skipPixelBytes = (NSUInteger)gl4metalUnpackSkipPixels * inputBytes;
    if (skipRowBytes > SIZE_MAX - skipPixelBytes) return NO;
    NSUInteger start = skipRowBytes + skipPixelBytes;
    NSUInteger inputWidthBytes = (NSUInteger)width * inputBytes;
    if ((NSUInteger)(height - 1) > (SIZE_MAX - start) / MAX((NSUInteger)1, inputRowBytes) ||
        inputWidthBytes > SIZE_MAX - (start + (NSUInteger)(height - 1) * inputRowBytes)) return NO;
    const uint8_t *source = (const uint8_t *)pixels + start;
    float *destination = (float *)converted.mutableBytes;
    for (GLsizei row = 0; row < height; ++row) {
        const uint8_t *inputRow = source + (NSUInteger)row * inputRowBytes;
        for (GLsizei column = 0; column < width; ++column) {
            float value = 0.0f;
            const uint8_t *component = inputRow + (NSUInteger)column * inputBytes;
            if (type == GL_FLOAT) {
                memcpy(&value, component, sizeof(value));
                if (gl4metalUnpackSwapBytes) {
                    uint8_t *bytes = (uint8_t *)&value;
                    uint8_t temp = bytes[0]; bytes[0] = bytes[3]; bytes[3] = temp;
                    temp = bytes[1]; bytes[1] = bytes[2]; bytes[2] = temp;
                }
            } else if (type == GL_UNSIGNED_SHORT) {
                GLushort v; memcpy(&v, component, sizeof(v));
                if (gl4metalUnpackSwapBytes) v = (GLushort)((v >> 8) | (v << 8));
                value = (float)v / 65535.0f;
            } else {
                GLuint v; memcpy(&v, component, sizeof(v));
                if (gl4metalUnpackSwapBytes)
                    v = ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
                        ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
                value = (float)((double)v / 4294967295.0);
            }
            destination[(NSUInteger)row * (NSUInteger)width + (NSUInteger)column] = fminf(1.0f, fmaxf(0.0f, value));
        }
    }
    id<MTLCommandBuffer> commandBuffer = ctx.currentCommandBuffer;
    BOOL dedicated = NO;
    if (!commandBuffer) {
        commandBuffer = [ctx.commandQueue commandBuffer];
        dedicated = YES;
    }
    id<MTLBuffer> staging = [ctx.device newBufferWithBytes:converted.bytes
                                                    length:converted.length
                                                   options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !staging || !blit) return NO;
    [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:outputRowBytes
        sourceBytesPerImage:converted.length sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
                  toTexture:texture destinationSlice:0 destinationLevel:(NSUInteger)level
         destinationOrigin:MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, 0)];
    [blit endEncoding];
    if (dedicated) [commandBuffer commit];
    return YES;
}

static NSUInteger gl4metalMipLevelCount(GLsizei width, GLsizei height) {
    NSUInteger largest = (NSUInteger)MAX(width, height);
    NSUInteger levels = 1;
    while (largest > 1) {
        largest >>= 1;
        ++levels;
    }
    return levels;
}

static BOOL gl4metalValidReadbackFormatType(GLenum format, GLenum type) {
    BOOL validFormat = NO;
    switch (format) {
        case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_RG:
        case GL_RGB: case GL_BGR: case GL_RGBA: case GL_BGRA:
        case GL_LUMINANCE: case GL_LUMINANCE_ALPHA:
            validFormat = YES;
            break;
        default:
            return NO;
    }
    return validFormat && (type == GL_UNSIGNED_BYTE || type == GL_FLOAT);
}

static GLenum gl4metalReadColorTexture(id<MTLTexture> texture, NSUInteger level,
                                       GLint x, GLint y, GLsizei width, GLsizei height,
                                       GLenum format, GLenum type, void *pixels,
                                       BOOL framebufferOrigin) {
    NSUInteger outputComponents = 0;
    switch (format) {
        case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_LUMINANCE: outputComponents = 1; break;
        case GL_RG: case GL_LUMINANCE_ALPHA: outputComponents = 2; break;
        case GL_RGB: case GL_BGR: outputComponents = 3; break;
        case GL_RGBA: case GL_BGRA: outputComponents = 4; break;
        default: return GL_INVALID_ENUM;
    }
    NSUInteger componentBytes = type == GL_UNSIGNED_BYTE ? 1 : type == GL_FLOAT ? sizeof(GLfloat) : 0;
    if (componentBytes == 0) return GL_INVALID_ENUM;
    size_t outputBpp = outputComponents * componentBytes;
    if (!texture || level >= texture.mipmapLevelCount) return GL_INVALID_OPERATION;
    if (texture.pixelFormat != MTLPixelFormatRGBA8Unorm &&
        texture.pixelFormat != MTLPixelFormatBGRA8Unorm) return GL_INVALID_OPERATION;
    if (width < 0 || height < 0) return GL_INVALID_VALUE;
    if (width == 0 || height == 0) return GL_NO_ERROR;
    if (!pixels) return GL_INVALID_VALUE;
    if (!ctx || !ctx.commandQueue) return GL_INVALID_OPERATION;
    if ([ctx.boundBuffers[@(GL_PIXEL_PACK_BUFFER)] unsignedIntValue] != 0)
        return GL_INVALID_OPERATION; // Client-memory readback only.

    NSUInteger textureWidth = MAX((NSUInteger)1, texture.width >> MIN(level, (NSUInteger)31));
    NSUInteger textureHeight = MAX((NSUInteger)1, texture.height >> MIN(level, (NSUInteger)31));
    GLint clipX0 = MAX(x, 0);
    GLint clipY0 = MAX(y, 0);
    GLint clipX1 = (GLint)MIN((NSUInteger)INT_MAX, (NSUInteger)MAX((int64_t)0, (int64_t)x + width));
    GLint clipY1 = (GLint)MIN((NSUInteger)INT_MAX, (NSUInteger)MAX((int64_t)0, (int64_t)y + height));
    clipX1 = (GLint)MIN((NSUInteger)MAX(clipX1, 0), textureWidth);
    clipY1 = (GLint)MIN((NSUInteger)MAX(clipY1, 0), textureHeight);
    clipX0 = (GLint)MIN((NSUInteger)MAX(clipX0, 0), textureWidth);
    clipY0 = (GLint)MIN((NSUInteger)MAX(clipY0, 0), textureHeight);
    GLsizei copyWidth = MAX(0, clipX1 - clipX0);
    GLsizei copyHeight = MAX(0, clipY1 - clipY0);

    NSUInteger rowPixels = gl4metalPackRowLength > 0 ? (NSUInteger)gl4metalPackRowLength : (NSUInteger)width;
    if (rowPixels > SIZE_MAX / outputBpp) return GL_INVALID_VALUE;
    NSUInteger rawDestinationRow = rowPixels * outputBpp;
    NSUInteger alignment = (NSUInteger)gl4metalPackAlignment;
    if (rawDestinationRow > SIZE_MAX - (alignment - 1)) return GL_INVALID_VALUE;
    NSUInteger destinationRow = (rawDestinationRow + alignment - 1) & ~(alignment - 1);
    NSUInteger skipRows = (NSUInteger)gl4metalPackSkipRows;
    NSUInteger skipPixels = (NSUInteger)gl4metalPackSkipPixels;
    if (skipRows > SIZE_MAX / MAX((NSUInteger)1, destinationRow) ||
        skipPixels > SIZE_MAX / outputBpp) return GL_INVALID_VALUE;
    NSUInteger skipRowBytes = skipRows * destinationRow;
    NSUInteger skipPixelBytes = skipPixels * outputBpp;
    if (skipRowBytes > SIZE_MAX - skipPixelBytes) return GL_INVALID_VALUE;
    NSUInteger destinationStart = skipRowBytes + skipPixelBytes;
    if ((NSUInteger)width > SIZE_MAX / outputBpp) return GL_INVALID_VALUE;
    if ((NSUInteger)height > 0) {
        if ((NSUInteger)(height - 1) > (SIZE_MAX - destinationStart) / MAX((NSUInteger)1, destinationRow))
            return GL_INVALID_VALUE;
        NSUInteger lastRow = destinationStart + (NSUInteger)(height - 1) * destinationRow;
        if ((NSUInteger)width * outputBpp > SIZE_MAX - lastRow) return GL_INVALID_VALUE;
    }
    for (GLsizei row = 0; row < height; ++row) {
        uint8_t *out = (uint8_t *)pixels + destinationStart + (NSUInteger)row * destinationRow;
        memset(out, 0, (size_t)width * outputBpp);
    }
    if (copyWidth == 0 || copyHeight == 0) return GL_NO_ERROR;
    if ((NSUInteger)copyWidth > (SIZE_MAX - 255u) / 4u) return GL_INVALID_VALUE;
    NSUInteger sourceRow = ((NSUInteger)copyWidth * 4u + 255u) & ~255u;
    if ((NSUInteger)copyHeight > SIZE_MAX / sourceRow) return GL_INVALID_VALUE;
    NSUInteger sourceLength = sourceRow * (NSUInteger)copyHeight;
    id<MTLBuffer> staging = [ctx.device newBufferWithLength:sourceLength
                                                    options:MTLResourceStorageModeShared];
    if (!staging) return GL_OUT_OF_MEMORY;

    gl4metalFlushPendingClear();
    if (ctx.currentCommandBuffer) {
        [ctx.currentCommandBuffer commit];
        [ctx.currentCommandBuffer waitUntilCompleted];
        if (ctx.currentCommandBuffer.status == MTLCommandBufferStatusError) {
            ctx.currentCommandBuffer = nil;
            return GL_INVALID_OPERATION;
        }
        ctx.currentCommandBuffer = nil;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !blit) return GL_OUT_OF_MEMORY;
    NSUInteger sourceY = framebufferOrigin
        ? textureHeight - (NSUInteger)clipY1 : (NSUInteger)clipY0;
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:level
             sourceOrigin:MTLOriginMake((NSUInteger)clipX0, sourceY, 0)
               sourceSize:MTLSizeMake((NSUInteger)copyWidth, (NSUInteger)copyHeight, 1)
                 toBuffer:staging
        destinationOffset:0
   destinationBytesPerRow:sourceRow
 destinationBytesPerImage:sourceLength];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) return GL_INVALID_OPERATION;

    const uint8_t *source = (const uint8_t *)staging.contents;
    BOOL sourceBGRA = texture.pixelFormat == MTLPixelFormatBGRA8Unorm;
    for (GLsizei sourceYIndex = 0; sourceYIndex < copyHeight; ++sourceYIndex) {
        GLint glY = framebufferOrigin
            ? clipY1 - 1 - sourceYIndex
            : clipY0 + sourceYIndex;
        GLint destinationY = glY - y;
        uint8_t *outRow = (uint8_t *)pixels + destinationStart + (NSUInteger)destinationY * destinationRow;
        const uint8_t *sourceRowData = source + (NSUInteger)sourceYIndex * sourceRow;
        for (GLsizei px = 0; px < copyWidth; ++px) {
            GLint destinationX = clipX0 + px - x;
            const uint8_t *src = sourceRowData + (NSUInteger)px * 4u;
            uint8_t red = sourceBGRA ? src[2] : src[0];
            uint8_t green = src[1];
            uint8_t blue = sourceBGRA ? src[0] : src[2];
            uint8_t alpha = src[3];
            uint8_t *dst = outRow + (NSUInteger)destinationX * outputBpp;
            uint8_t luminance = (uint8_t)lrint((0.30 * red) + (0.59 * green) + (0.11 * blue));
            GLfloat values[4];
            switch (format) {
                case GL_RGBA: values[0] = red / 255.0f; values[1] = green / 255.0f; values[2] = blue / 255.0f; values[3] = alpha / 255.0f; break;
                case GL_BGRA: values[0] = blue / 255.0f; values[1] = green / 255.0f; values[2] = red / 255.0f; values[3] = alpha / 255.0f; break;
                case GL_RGB: values[0] = red / 255.0f; values[1] = green / 255.0f; values[2] = blue / 255.0f; break;
                case GL_BGR: values[0] = blue / 255.0f; values[1] = green / 255.0f; values[2] = red / 255.0f; break;
                case GL_RG: values[0] = red / 255.0f; values[1] = green / 255.0f; break;
                case GL_RED: values[0] = red / 255.0f; break;
                case GL_GREEN: values[0] = green / 255.0f; break;
                case GL_BLUE: values[0] = blue / 255.0f; break;
                case GL_ALPHA: values[0] = alpha / 255.0f; break;
                case GL_LUMINANCE: values[0] = luminance / 255.0f; break;
                case GL_LUMINANCE_ALPHA: values[0] = luminance / 255.0f; values[1] = alpha / 255.0f; break;
                default: return GL_INVALID_ENUM;
            }
            if (type == GL_FLOAT) {
                memcpy(dst, values, outputBpp);
                if (gl4metalPackSwapBytes) {
                    for (NSUInteger component = 0; component < outputComponents; ++component) {
                        uint8_t *bytes = dst + component * sizeof(GLfloat);
                        uint8_t temp = bytes[0]; bytes[0] = bytes[3]; bytes[3] = temp;
                        temp = bytes[1]; bytes[1] = bytes[2]; bytes[2] = temp;
                    }
                }
            } else {
                for (NSUInteger component = 0; component < outputComponents; ++component)
                    dst[component] = (uint8_t)lrintf(values[component] * 255.0f);
            }
        }
    }
    return GL_NO_ERROR;
}

static GLenum gl4metalReadDepthTexture(id<MTLTexture> texture, NSUInteger level,
                                       GLint x, GLint y, GLsizei width, GLsizei height,
                                       GLenum type, void *pixels, BOOL framebufferOrigin) {
    if (width < 0 || height < 0) return GL_INVALID_VALUE;
    if (type != GL_FLOAT && type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT)
        return GL_INVALID_ENUM;
    if (!texture || level >= texture.mipmapLevelCount ||
        (texture.pixelFormat != MTLPixelFormatDepth32Float &&
         texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)) return GL_INVALID_OPERATION;
    if (width == 0 || height == 0) return GL_NO_ERROR;
    if (!pixels) return GL_INVALID_VALUE;
    if (!ctx || !ctx.commandQueue || [ctx.boundBuffers[@(GL_PIXEL_PACK_BUFFER)] unsignedIntValue] != 0)
        return GL_INVALID_OPERATION;

    NSUInteger componentBytes = type == GL_FLOAT ? sizeof(GLfloat) :
        type == GL_UNSIGNED_SHORT ? sizeof(GLushort) : sizeof(GLuint);
    NSUInteger rowPixels = gl4metalPackRowLength > 0 ? (NSUInteger)gl4metalPackRowLength : (NSUInteger)width;
    if (rowPixels > SIZE_MAX / componentBytes) return GL_INVALID_VALUE;
    NSUInteger rawRow = rowPixels * componentBytes;
    NSUInteger alignment = (NSUInteger)gl4metalPackAlignment;
    if (rawRow > SIZE_MAX - (alignment - 1)) return GL_INVALID_VALUE;
    NSUInteger destinationRow = (rawRow + alignment - 1) & ~(alignment - 1);
    NSUInteger skipRows = (NSUInteger)gl4metalPackSkipRows;
    NSUInteger skipPixels = (NSUInteger)gl4metalPackSkipPixels;
    if (skipRows > SIZE_MAX / MAX((NSUInteger)1, destinationRow) ||
        skipPixels > SIZE_MAX / componentBytes) return GL_INVALID_VALUE;
    NSUInteger skipRowBytes = skipRows * destinationRow;
    NSUInteger skipPixelBytes = skipPixels * componentBytes;
    if (skipRowBytes > SIZE_MAX - skipPixelBytes) return GL_INVALID_VALUE;
    NSUInteger destinationStart = skipRowBytes + skipPixelBytes;
    if ((NSUInteger)width > SIZE_MAX / componentBytes) return GL_INVALID_VALUE;
    if ((NSUInteger)height > 0) {
        if ((NSUInteger)(height - 1) >
            (SIZE_MAX - destinationStart) / MAX((NSUInteger)1, destinationRow)) return GL_INVALID_VALUE;
        NSUInteger lastRow = destinationStart + (NSUInteger)(height - 1) * destinationRow;
        if ((NSUInteger)width * componentBytes > SIZE_MAX - lastRow) return GL_INVALID_VALUE;
    }
    for (GLsizei row = 0; row < height; ++row)
        memset((uint8_t *)pixels + destinationStart + (NSUInteger)row * destinationRow,
               0, (size_t)width * componentBytes);

    NSUInteger textureWidth = MAX((NSUInteger)1, texture.width >> MIN(level, (NSUInteger)31));
    NSUInteger textureHeight = MAX((NSUInteger)1, texture.height >> MIN(level, (NSUInteger)31));
    GLint clipX0 = (GLint)MIN((NSUInteger)MAX(x, 0), textureWidth);
    GLint clipY0 = (GLint)MIN((NSUInteger)MAX(y, 0), textureHeight);
    GLint clipX1 = (GLint)MIN((NSUInteger)MAX((int64_t)0, (int64_t)x + width), textureWidth);
    GLint clipY1 = (GLint)MIN((NSUInteger)MAX((int64_t)0, (int64_t)y + height), textureHeight);
    GLsizei copyWidth = MAX(0, clipX1 - clipX0);
    GLsizei copyHeight = MAX(0, clipY1 - clipY0);
    if (copyWidth == 0 || copyHeight == 0) return GL_NO_ERROR;

    if ((NSUInteger)copyWidth > (SIZE_MAX - 255u) / sizeof(float)) return GL_INVALID_VALUE;
    NSUInteger sourceRow = ((NSUInteger)copyWidth * sizeof(float) + 255u) & ~255u;
    if ((NSUInteger)copyHeight > SIZE_MAX / sourceRow) return GL_INVALID_VALUE;
    NSUInteger sourceLength = sourceRow * (NSUInteger)copyHeight;
    id<MTLBuffer> staging = [ctx.device newBufferWithLength:sourceLength
                                                    options:MTLResourceStorageModeShared];
    if (!staging) return GL_OUT_OF_MEMORY;

    gl4metalFlushPendingClear();
    if (ctx.currentCommandBuffer) {
        [ctx.currentCommandBuffer commit];
        [ctx.currentCommandBuffer waitUntilCompleted];
        if (ctx.currentCommandBuffer.status == MTLCommandBufferStatusError) {
            ctx.currentCommandBuffer = nil;
            return GL_INVALID_OPERATION;
        }
        ctx.currentCommandBuffer = nil;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !blit) return GL_OUT_OF_MEMORY;
    NSUInteger sourceY = framebufferOrigin ? textureHeight - (NSUInteger)clipY1 : (NSUInteger)clipY0;
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:level
             sourceOrigin:MTLOriginMake((NSUInteger)clipX0, sourceY, 0)
               sourceSize:MTLSizeMake((NSUInteger)copyWidth, (NSUInteger)copyHeight, 1)
                 toBuffer:staging
        destinationOffset:0
   destinationBytesPerRow:sourceRow
 destinationBytesPerImage:sourceLength];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) return GL_INVALID_OPERATION;

    const uint8_t *source = (const uint8_t *)staging.contents;
    for (GLsizei sourceRowIndex = 0; sourceRowIndex < copyHeight; ++sourceRowIndex) {
        GLint glY = framebufferOrigin ? clipY1 - 1 - sourceRowIndex : clipY0 + sourceRowIndex;
        uint8_t *destinationRowPtr = (uint8_t *)pixels + destinationStart +
            (NSUInteger)(glY - y) * destinationRow;
        const uint8_t *sourceRowPtr = source + (NSUInteger)sourceRowIndex * sourceRow;
        for (GLsizei column = 0; column < copyWidth; ++column) {
            float depth = 0.0f;
            memcpy(&depth, sourceRowPtr + (NSUInteger)column * sizeof(float), sizeof(depth));
            depth = fminf(1.0f, fmaxf(0.0f, depth));
            uint8_t *out = destinationRowPtr + (NSUInteger)(clipX0 + column - x) * componentBytes;
            if (type == GL_FLOAT) {
                memcpy(out, &depth, sizeof(depth));
            } else if (type == GL_UNSIGNED_SHORT) {
                GLushort value = (GLushort)lrintf(depth * 65535.0f);
                if (gl4metalPackSwapBytes) value = (GLushort)((value >> 8) | (value << 8));
                memcpy(out, &value, sizeof(value));
            } else {
                GLuint value = (GLuint)llround((double)depth * 4294967295.0);
                if (gl4metalPackSwapBytes)
                    value = ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
                        ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
                memcpy(out, &value, sizeof(value));
            }
            if (type == GL_FLOAT && gl4metalPackSwapBytes) {
                uint8_t *bytes = out;
                uint8_t temp = bytes[0]; bytes[0] = bytes[3]; bytes[3] = temp;
                temp = bytes[1]; bytes[1] = bytes[2]; bytes[2] = temp;
            }
        }
    }
    return GL_NO_ERROR;
}

static MTLSamplerAddressMode gl4metalSamplerAddressMode(NSInteger wrap) {
    switch (wrap) {
        case GL_CLAMP_TO_EDGE: return MTLSamplerAddressModeClampToEdge;
        case GL_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
        case GL_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
        case GL_REPEAT:
        default: return MTLSamplerAddressModeRepeat;
    }
}

static MTLCompareFunction gl4metalSamplerCompareFunction(NSInteger function) {
    switch (function) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        case GL_ALWAYS:
        default: return MTLCompareFunctionAlways;
    }
}

static id<MTLSamplerState> gl4metalBuildSampler(NSDictionary<NSNumber *, NSNumber *> *params,
                                                const GLfloat *borderColor) {
    if (!ctx || !ctx.device || !params) return nil;
    NSInteger minFilter = [params[@(GL_TEXTURE_MIN_FILTER)] integerValue];
    NSInteger magFilter = [params[@(GL_TEXTURE_MAG_FILTER)] integerValue];
    NSInteger wrapS = [params[@(GL_TEXTURE_WRAP_S)] integerValue];
    NSInteger wrapT = [params[@(GL_TEXTURE_WRAP_T)] integerValue];
    NSInteger wrapR = [params[@(GL_TEXTURE_WRAP_R)] integerValue];

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    switch (minFilter) {
        case GL_NEAREST:
        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
            sd.minFilter = MTLSamplerMinMagFilterNearest;
            break;
        default:
            sd.minFilter = MTLSamplerMinMagFilterLinear;
            break;
    }
    sd.magFilter = (magFilter == GL_NEAREST) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;

    BOOL mipmapped = (minFilter == GL_NEAREST_MIPMAP_NEAREST ||
                      minFilter == GL_LINEAR_MIPMAP_NEAREST ||
                      minFilter == GL_NEAREST_MIPMAP_LINEAR ||
                      minFilter == GL_LINEAR_MIPMAP_LINEAR);
    if (mipmapped) {
        switch (minFilter) {
            case GL_NEAREST_MIPMAP_NEAREST:
                sd.mipFilter = MTLSamplerMipFilterNearest;
                break;
            case GL_NEAREST_MIPMAP_LINEAR:
            case GL_LINEAR_MIPMAP_LINEAR:
            case GL_LINEAR_MIPMAP_NEAREST:
            default:
                sd.mipFilter = MTLSamplerMipFilterLinear;
                break;
        }
    } else {
        sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    }

    sd.sAddressMode = gl4metalSamplerAddressMode(wrapS);
    sd.tAddressMode = gl4metalSamplerAddressMode(wrapT);
    sd.rAddressMode = gl4metalSamplerAddressMode(wrapR);
    float minLOD = [params[@(GL_TEXTURE_MIN_LOD)] floatValue];
    float maxLOD = [params[@(GL_TEXTURE_MAX_LOD)] floatValue];
    NSNumber *baseLevel = params[@(GL_TEXTURE_BASE_LEVEL)];
    NSNumber *maxLevel = params[@(GL_TEXTURE_MAX_LEVEL)];
    if (baseLevel) minLOD = MAX(minLOD, baseLevel.floatValue);
    if (maxLevel) maxLOD = MIN(maxLOD, maxLevel.floatValue);
    sd.lodMinClamp = minLOD;
    sd.lodMaxClamp = MAX(minLOD, maxLOD);
    if ([params[@(GL_TEXTURE_COMPARE_MODE)] integerValue] == GL_COMPARE_REF_TO_TEXTURE)
        sd.compareFunction = gl4metalSamplerCompareFunction([params[@(GL_TEXTURE_COMPARE_FUNC)] integerValue]);

    // Metal exposes only three fixed border colors. Preserve the caller's full
    // GL border color for queries, and select the closest representable Metal
    // value for sampling.
    BOOL borderBlackOpaque = borderColor && borderColor[0] == 0.0f && borderColor[1] == 0.0f &&
        borderColor[2] == 0.0f && borderColor[3] == 1.0f;
    BOOL borderWhiteOpaque = borderColor && borderColor[0] == 1.0f && borderColor[1] == 1.0f &&
        borderColor[2] == 1.0f && borderColor[3] == 1.0f;
    sd.borderColor = borderWhiteOpaque ? MTLSamplerBorderColorOpaqueWhite :
        (borderBlackOpaque ? MTLSamplerBorderColorOpaqueBlack : MTLSamplerBorderColorTransparentBlack);
    return [ctx.device newSamplerStateWithDescriptor:sd];
}

static GLfloat *gl4metalDefaultBorderColor(void) {
    static GLfloat transparentBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    return transparentBlack;
}

static id<MTLSamplerState> gl4metalCreateSamplerForTexture(GLuint texture) {
    if (!ctx || !ctx.device || texture == 0) return nil;
    gl4metalEnsureTextureState();
    NSNumber *key = @(texture);
    id<MTLSamplerState> cached = gl4metalTextureSamplers[key];
    if (cached) return cached;
    const GLfloat *borderColor = (const GLfloat *)gl4metalGetTextureBorderData(texture).bytes;
    id<MTLSamplerState> sampler = gl4metalBuildSampler(gl4metalEnsureTextureParams(texture), borderColor);
    if (sampler) gl4metalTextureSamplers[key] = sampler;
    return sampler;
}

static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalDefaultSamplerParams(void) {
    return [@{
        @(GL_TEXTURE_MIN_FILTER): @(GL_NEAREST_MIPMAP_LINEAR),
        @(GL_TEXTURE_MAG_FILTER): @(GL_LINEAR),
        @(GL_TEXTURE_WRAP_S): @(GL_REPEAT),
        @(GL_TEXTURE_WRAP_T): @(GL_REPEAT),
        @(GL_TEXTURE_WRAP_R): @(GL_REPEAT),
        @(GL_TEXTURE_MIN_LOD): @(-1000.0f),
            @(GL_TEXTURE_MAX_LOD): @(1000.0f),
            @(GL_TEXTURE_LOD_BIAS): @(0.0f),
            @(GL_TEXTURE_BASE_LEVEL): @0,
            @(GL_TEXTURE_MAX_LEVEL): @1000,
            @(GL_TEXTURE_COMPARE_MODE): @(GL_NONE),
        @(GL_TEXTURE_COMPARE_FUNC): @(GL_LEQUAL)
    } mutableCopy];
}

static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalEnsureSamplerParams(GLuint sampler) {
    NSMutableDictionary *params = gl4metalSamplerParams[@(sampler)];
    if (!params) {
        params = gl4metalDefaultSamplerParams();
        gl4metalSamplerParams[@(sampler)] = params;
    }
    if (!gl4metalSamplerBorderColors[@(sampler)])
        gl4metalSamplerBorderColors[@(sampler)] = [NSData dataWithBytes:gl4metalDefaultBorderColor()
                                                                 length:sizeof(GLfloat) * 4];
    return params;
}

static BOOL gl4metalIsSamplerScalarParameter(GLenum pname) {
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER: case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_WRAP_S: case GL_TEXTURE_WRAP_T: case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_MIN_LOD: case GL_TEXTURE_MAX_LOD: case GL_TEXTURE_LOD_BIAS:
        case GL_TEXTURE_COMPARE_MODE: case GL_TEXTURE_COMPARE_FUNC:
            return YES;
        default: return NO;
    }
}

static BOOL gl4metalValidSamplerParameter(GLenum pname, GLfloat value) {
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
            return value == GL_NEAREST || value == GL_LINEAR ||
                value == GL_NEAREST_MIPMAP_NEAREST || value == GL_LINEAR_MIPMAP_NEAREST ||
                value == GL_NEAREST_MIPMAP_LINEAR || value == GL_LINEAR_MIPMAP_LINEAR;
        case GL_TEXTURE_MAG_FILTER:
            return value == GL_NEAREST || value == GL_LINEAR;
        case GL_TEXTURE_WRAP_S: case GL_TEXTURE_WRAP_T: case GL_TEXTURE_WRAP_R:
            return value == GL_REPEAT || value == GL_CLAMP_TO_EDGE ||
                value == GL_CLAMP_TO_BORDER || value == GL_MIRRORED_REPEAT;
        case GL_TEXTURE_COMPARE_MODE:
            return value == GL_NONE || value == GL_COMPARE_REF_TO_TEXTURE;
        case GL_TEXTURE_COMPARE_FUNC:
            return value == GL_NEVER || value == GL_LESS || value == GL_EQUAL ||
                value == GL_LEQUAL || value == GL_GREATER || value == GL_NOTEQUAL ||
                value == GL_GEQUAL || value == GL_ALWAYS;
        case GL_TEXTURE_MIN_LOD: case GL_TEXTURE_MAX_LOD: case GL_TEXTURE_LOD_BIAS:
            return isfinite(value);
        case GL_TEXTURE_BASE_LEVEL: case GL_TEXTURE_MAX_LEVEL:
            return isfinite(value) && value >= 0.0f && floorf(value) == value;
        default: return NO;
    }
}

static void gl4metalSetSamplerScalar(GLuint sampler, GLenum pname, GLfloat value) {
    if (!gl4metalSamplerObjectNames || ![gl4metalSamplerObjectNames containsObject:@(sampler)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!gl4metalIsSamplerScalarParameter(pname)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!gl4metalValidSamplerParameter(pname, value)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalEnsureSamplerParams(sampler)[@(pname)] = @(value);
    [gl4metalSamplerStates removeObjectForKey:@(sampler)];
    gl4metalSetError(GL_NO_ERROR);
}

static NSData *gl4metalSamplerBorderData(GLuint sampler) {
    gl4metalEnsureSamplerParams(sampler);
    return gl4metalSamplerBorderColors[@(sampler)];
}

void APIENTRY glGenSamplers(GLsizei count, GLuint *samplers) {
    if (count < 0 || (count > 0 && !samplers)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalEnsureTextureState();
    for (GLsizei i = 0; i < count; ++i) {
        GLuint name;
        do {
            name = gl4metalNextSamplerId++;
            if (name == 0) name = gl4metalNextSamplerId++;
        } while ([gl4metalSamplerGeneratedNames containsObject:@(name)] ||
                 [gl4metalSamplerObjectNames containsObject:@(name)]);
        samplers[i] = name;
        [gl4metalSamplerGeneratedNames addObject:@(name)];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteSamplers(GLsizei count, const GLuint *samplers) {
    if (count < 0 || (count > 0 && !samplers)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalEnsureTextureState();
    for (GLsizei i = 0; i < count; ++i) {
        GLuint name = samplers[i];
        if (name == 0) continue;
        [gl4metalSamplerGeneratedNames removeObject:@(name)];
        [gl4metalSamplerObjectNames removeObject:@(name)];
        [gl4metalSamplerParams removeObjectForKey:@(name)];
        [gl4metalSamplerBorderColors removeObjectForKey:@(name)];
        [gl4metalSamplerStates removeObjectForKey:@(name)];
        for (NSNumber *unit in [gl4metalSamplerUnitBindings allKeysForObject:@(name)])
            gl4metalSamplerUnitBindings[unit] = @0;
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsSampler(GLuint sampler) {
    gl4metalEnsureTextureState();
    return sampler != 0 && [gl4metalSamplerObjectNames containsObject:@(sampler)] ? GL_TRUE : GL_FALSE;
}

void APIENTRY glBindSampler(GLuint unit, GLuint sampler) {
    gl4metalEnsureTextureState();
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (sampler != 0 && ![gl4metalSamplerGeneratedNames containsObject:@(sampler)] &&
        ![gl4metalSamplerObjectNames containsObject:@(sampler)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (sampler != 0) {
        [gl4metalSamplerGeneratedNames removeObject:@(sampler)];
        [gl4metalSamplerObjectNames addObject:@(sampler)];
        gl4metalEnsureSamplerParams(sampler);
    }
    gl4metalSamplerUnitBindings[@(unit)] = @(sampler);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glSamplerParameteri(GLuint sampler, GLenum pname, GLint param) {
    gl4metalEnsureTextureState();
    gl4metalSetSamplerScalar(sampler, pname, (GLfloat)param);
}

void APIENTRY glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param) {
    gl4metalEnsureTextureState();
    gl4metalSetSamplerScalar(sampler, pname, param);
}

static void gl4metalSetSamplerBorder(GLuint sampler, const GLfloat color[4]) {
    if (!gl4metalSamplerObjectNames || ![gl4metalSamplerObjectNames containsObject:@(sampler)]) {
        gl4metalSetError(GL_INVALID_OPERATION); return;
    }
    gl4metalEnsureSamplerParams(sampler);
    gl4metalSamplerBorderColors[@(sampler)] = [NSData dataWithBytes:color length:sizeof(GLfloat) * 4];
    [gl4metalSamplerStates removeObjectForKey:@(sampler)];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat *param) {
    if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalEnsureTextureState();
    if (pname == GL_TEXTURE_BORDER_COLOR) { gl4metalSetSamplerBorder(sampler, param); return; }
    gl4metalSetSamplerScalar(sampler, pname, param[0]);
}

void APIENTRY glSamplerParameteriv(GLuint sampler, GLenum pname, const GLint *param) {
    if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        GLfloat color[4] = {(GLfloat)param[0], (GLfloat)param[1], (GLfloat)param[2], (GLfloat)param[3]};
        gl4metalEnsureTextureState(); gl4metalSetSamplerBorder(sampler, color); return;
    }
    glSamplerParameteri(sampler, pname, param[0]);
}

void APIENTRY glSamplerParameterIiv(GLuint sampler, GLenum pname, const GLint *param) {
    if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLfloat color[4] = {(GLfloat)param[0], (GLfloat)param[1], (GLfloat)param[2], (GLfloat)param[3]};
    gl4metalEnsureTextureState(); gl4metalSetSamplerBorder(sampler, color);
}

void APIENTRY glSamplerParameterIuiv(GLuint sampler, GLenum pname, const GLuint *param) {
    if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLfloat color[4] = {(GLfloat)param[0], (GLfloat)param[1], (GLfloat)param[2], (GLfloat)param[3]};
    gl4metalEnsureTextureState(); gl4metalSetSamplerBorder(sampler, color);
}

static BOOL gl4metalGetSamplerParameter(GLuint sampler, GLenum pname, GLfloat *floats) {
    if (!gl4metalSamplerObjectNames || ![gl4metalSamplerObjectNames containsObject:@(sampler)]) {
        gl4metalSetError(GL_INVALID_OPERATION); return NO;
    }
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        NSData *data = gl4metalSamplerBorderData(sampler);
        memcpy(floats, data.bytes, sizeof(GLfloat) * 4);
    } else if (gl4metalIsSamplerScalarParameter(pname)) {
        floats[0] = [gl4metalEnsureSamplerParams(sampler)[@(pname)] floatValue];
    } else {
        gl4metalSetError(GL_INVALID_ENUM); return NO;
    }
    gl4metalSetError(GL_NO_ERROR);
    return YES;
}

void APIENTRY glGetSamplerParameterfv(GLuint sampler, GLenum pname, GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalEnsureTextureState(); gl4metalGetSamplerParameter(sampler, pname, params);
}

void APIENTRY glGetSamplerParameteriv(GLuint sampler, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLfloat values[4] = {0, 0, 0, 0};
    gl4metalEnsureTextureState();
    if (!gl4metalGetSamplerParameter(sampler, pname, values)) return;
    NSUInteger count = pname == GL_TEXTURE_BORDER_COLOR ? 4 : 1;
    for (NSUInteger i = 0; i < count; ++i) params[i] = (GLint)values[i];
}

void APIENTRY glGetSamplerParameterIiv(GLuint sampler, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLfloat values[4]; gl4metalEnsureTextureState();
    if (!gl4metalGetSamplerParameter(sampler, pname, values)) return;
    for (NSUInteger i = 0; i < 4; ++i) params[i] = (GLint)values[i];
}

void APIENTRY glGetSamplerParameterIuiv(GLuint sampler, GLenum pname, GLuint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLfloat values[4]; gl4metalEnsureTextureState();
    if (!gl4metalGetSamplerParameter(sampler, pname, values)) return;
    for (NSUInteger i = 0; i < 4; ++i) params[i] = (GLuint)MAX(0.0f, values[i]);
}

// Exposed bridge functions. These are intentionally not GL entry points.
id<MTLTexture> gl4metalGetTextureForUnit(GLuint unit) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) return nil;
    GLuint texture = gl4metalTextureBoundToUnit(unit);
    if (!texture) return nil;
    return gl4metalTextureObjects[@(texture)];
}

id<MTLSamplerState> gl4metalGetSamplerForUnit(GLuint unit) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) return nil;
    gl4metalEnsureTextureState();
    GLuint samplerName = [gl4metalSamplerUnitBindings[@(unit)] unsignedIntValue];
    if (samplerName != 0 && [gl4metalSamplerObjectNames containsObject:@(samplerName)]) {
        id<MTLSamplerState> cached = gl4metalSamplerStates[@(samplerName)];
        if (cached) return cached;
        NSData *borderData = gl4metalSamplerBorderData(samplerName);
        const GLfloat *border = (const GLfloat *)borderData.bytes;
        id<MTLSamplerState> created = gl4metalBuildSampler(gl4metalEnsureSamplerParams(samplerName), border);
        if (created) gl4metalSamplerStates[@(samplerName)] = created;
        return created;
    }
    GLuint texture = gl4metalTextureBoundToUnit(unit);
    if (!texture) return nil;
    return gl4metalCreateSamplerForTexture(texture);
}

GLuint gl4metalGetTextureNameForUnit(GLuint unit) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) return 0;
    return gl4metalTextureBoundToUnit(unit);
}

GLuint gl4metalGetTextureNameForTarget(GLuint unit, GLenum target) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS ||
        (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D)) return 0;
    return gl4metalTextureBoundToUnitForTarget(unit, target);
}

GLuint gl4metalGetSamplerNameForUnit(GLuint unit) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) return 0;
    gl4metalEnsureTextureState();
    return [gl4metalSamplerUnitBindings[@(unit)] unsignedIntValue];
}

GLuint gl4metalGetActiveTextureUnit(void) {
    return gl4metalActiveTextureUnit;
}

void gl4metalInitTextureState(void) {
    gl4metalTextureObjects = [NSMutableDictionary dictionary];
    gl4metalTextureLevelInfo = [NSMutableDictionary dictionary];
    gl4metalTextureParams = [NSMutableDictionary dictionary];
    gl4metalTextureBorderColors = [NSMutableDictionary dictionary];
    gl4metalTextureSamplers = [NSMutableDictionary dictionary];
    gl4metalTextureUnitBindings = [NSMutableDictionary dictionary];
    gl4metalTexture1DUnitBindings = [NSMutableDictionary dictionary];
    gl4metalTextureTargets = [NSMutableDictionary dictionary];
    gl4metalSamplerUnitBindings = [NSMutableDictionary dictionary];
    gl4metalSamplerParams = [NSMutableDictionary dictionary];
    gl4metalSamplerBorderColors = [NSMutableDictionary dictionary];
    gl4metalSamplerStates = [NSMutableDictionary dictionary];
    gl4metalTextureObjectNames = [NSMutableSet set];
    gl4metalTextureGeneratedNames = [NSMutableSet set];
    gl4metalSamplerGeneratedNames = [NSMutableSet set];
    gl4metalSamplerObjectNames = [NSMutableSet set];
    gl4metalNextTextureId = 1;
    gl4metalNextSamplerId = 1;
    gl4metalActiveTextureUnit = 0;
    gl4metalUnpackAlignment = 4;
    gl4metalUnpackRowLength = 0;
    gl4metalUnpackSkipRows = 0;
    gl4metalUnpackSkipPixels = 0;
    gl4metalPackAlignment = 4;
    gl4metalPackRowLength = 0;
    gl4metalPackSkipRows = 0;
    gl4metalPackSkipPixels = 0;
    gl4metalUnpackSwapBytes = GL_FALSE;
    gl4metalUnpackLSBFirst = GL_FALSE;
    gl4metalPackSwapBytes = GL_FALSE;
    gl4metalPackLSBFirst = GL_FALSE;
}

GLint gl4metalGetPixelStoreValue(GLenum pname) {
    switch (pname) {
        case GL_UNPACK_ALIGNMENT: return gl4metalUnpackAlignment;
        case GL_UNPACK_ROW_LENGTH: return gl4metalUnpackRowLength;
        case GL_UNPACK_SKIP_ROWS: return gl4metalUnpackSkipRows;
        case GL_UNPACK_SKIP_PIXELS: return gl4metalUnpackSkipPixels;
        case GL_PACK_ALIGNMENT: return gl4metalPackAlignment;
        case GL_PACK_ROW_LENGTH: return gl4metalPackRowLength;
        case GL_PACK_SKIP_ROWS: return gl4metalPackSkipRows;
        case GL_PACK_SKIP_PIXELS: return gl4metalPackSkipPixels;
        case GL_UNPACK_SWAP_BYTES: return gl4metalUnpackSwapBytes;
        case GL_UNPACK_LSB_FIRST: return gl4metalUnpackLSBFirst;
        case GL_PACK_SWAP_BYTES: return gl4metalPackSwapBytes;
        case GL_PACK_LSB_FIRST: return gl4metalPackLSBFirst;
        default: return -1;
    }
}

void APIENTRY glPixelStorei(GLenum pname, GLint param) {
    switch (pname) {
        case GL_UNPACK_ALIGNMENT:
            if (param != 1 && param != 2 && param != 4 && param != 8) {
                gl4metalSetError(GL_INVALID_VALUE);
                return;
            }
            gl4metalUnpackAlignment = param;
            break;
        case GL_PACK_ALIGNMENT:
            if (param != 1 && param != 2 && param != 4 && param != 8) {
                gl4metalSetError(GL_INVALID_VALUE);
                return;
            }
            gl4metalPackAlignment = param;
            break;
        case GL_UNPACK_ROW_LENGTH:
        case GL_UNPACK_SKIP_ROWS:
        case GL_UNPACK_SKIP_PIXELS:
            if (param < 0) {
                gl4metalSetError(GL_INVALID_VALUE);
                return;
            }
            if (pname == GL_UNPACK_ROW_LENGTH) gl4metalUnpackRowLength = param;
            else if (pname == GL_UNPACK_SKIP_ROWS) gl4metalUnpackSkipRows = param;
            else gl4metalUnpackSkipPixels = param;
            break;
        case GL_PACK_ROW_LENGTH:
        case GL_PACK_SKIP_ROWS:
        case GL_PACK_SKIP_PIXELS:
            if (param < 0) {
                gl4metalSetError(GL_INVALID_VALUE);
                return;
            }
            if (pname == GL_PACK_ROW_LENGTH) gl4metalPackRowLength = param;
            else if (pname == GL_PACK_SKIP_ROWS) gl4metalPackSkipRows = param;
            else gl4metalPackSkipPixels = param;
            break;
        case GL_UNPACK_SWAP_BYTES:
        case GL_UNPACK_LSB_FIRST:
        case GL_PACK_SWAP_BYTES:
        case GL_PACK_LSB_FIRST:
            if (pname == GL_UNPACK_SWAP_BYTES) gl4metalUnpackSwapBytes = param != 0;
            else if (pname == GL_UNPACK_LSB_FIRST) gl4metalUnpackLSBFirst = param != 0;
            else if (pname == GL_PACK_SWAP_BYTES) gl4metalPackSwapBytes = param != 0;
            else gl4metalPackLSBFirst = param != 0;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPixelStoref(GLenum pname, GLfloat param) {
    glPixelStorei(pname, (GLint)param);
}

void APIENTRY glGenTextures(GLsizei n, GLuint *textures) {
    if (n < 0 || (n > 0 && !textures)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (n == 0) {
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    gl4metalEnsureTextureState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint texture = 0;
        do {
            texture = gl4metalNextTextureId++;
            if (texture == 0) texture = gl4metalNextTextureId++;
        } while ([gl4metalTextureObjectNames containsObject:@(texture)] ||
                 [gl4metalTextureGeneratedNames containsObject:@(texture)] ||
                 gl4metalTextureObjects[@(texture)] != nil);
        textures[i] = texture;
        [gl4metalTextureGeneratedNames addObject:@(texture)];
        gl4metalEnsureTextureParams(texture);
    }
    gl4metalSetError(GL_NO_ERROR);
}

static void gl4metalDetachTextureFromAllFramebuffers(GLuint texture);

void APIENTRY glDeleteTextures(GLsizei n, const GLuint *textures) {
    if (n < 0 || (n > 0 && !textures)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (n == 0) {
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    gl4metalEnsureTextureState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint texture = textures[i];
        if (texture == 0) continue;

        gl4metalDetachTextureFromAllFramebuffers(texture);
        if (ctx) ctx.pipelineState = nil;
        [gl4metalTextureObjects removeObjectForKey:@(texture)];
        [gl4metalTextureLevelInfo removeObjectForKey:@(texture)];
        [gl4metalTextureParams removeObjectForKey:@(texture)];
        [gl4metalTextureBorderColors removeObjectForKey:@(texture)];
        [gl4metalTextureSamplers removeObjectForKey:@(texture)];
        [gl4metalTextureObjectNames removeObject:@(texture)];
        [gl4metalTextureGeneratedNames removeObject:@(texture)];
        [gl4metalTextureTargets removeObjectForKey:@(texture)];

        NSArray<NSNumber *> *units = [gl4metalTextureUnitBindings allKeysForObject:@(texture)];
        for (NSNumber *unit in units) {
            [gl4metalTextureUnitBindings removeObjectForKey:unit];
        }
        units = [gl4metalTexture1DUnitBindings allKeysForObject:@(texture)];
        for (NSNumber *unit in units) [gl4metalTexture1DUnitBindings removeObjectForKey:unit];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindTexture(GLenum target, GLuint texture) {
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalEnsureTextureState();

    if (texture != 0) {
        GLenum existingTarget = [gl4metalTextureTargets[@(texture)] unsignedIntValue];
        if (existingTarget != 0 && existingTarget != target) {
            gl4metalSetError(GL_INVALID_OPERATION);
            return;
        }
        gl4metalTextureTargets[@(texture)] = @(target);
    }

    if (texture != 0 && !gl4metalTextureParams[@(texture)]) {
        gl4metalEnsureTextureParams(texture);
    }

    if (texture != 0) {
        [gl4metalTextureGeneratedNames removeObject:@(texture)];
        [gl4metalTextureObjectNames addObject:@(texture)];
    }

    NSMutableDictionary *bindings = target == GL_TEXTURE_1D
        ? gl4metalTexture1DUnitBindings : gl4metalTextureUnitBindings;
    bindings[@(gl4metalActiveTextureUnit)] = @(texture);
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsTexture(GLuint texture) {
    if (texture == 0) return GL_FALSE;
    gl4metalEnsureTextureState();
    return [gl4metalTextureObjectNames containsObject:@(texture)] ? GL_TRUE : GL_FALSE;
}

void APIENTRY glActiveTexture(GLenum texture) {
    if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + GL4METAL_MAX_TEXTURE_UNITS) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalActiveTextureUnit = (GLuint)(texture - GL_TEXTURE0);
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalSetBoundTextureScalar(GLenum target, GLenum pname, GLfloat value) {
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return NO; }
    if ((target == GL_TEXTURE_1D && pname == GL_TEXTURE_WRAP_T) || pname == GL_TEXTURE_WRAP_R) {
        gl4metalSetError(GL_INVALID_ENUM); return NO;
    }
    if (pname == GL_TEXTURE_BORDER_COLOR || !gl4metalIsSamplerScalarParameter(pname) &&
        pname != GL_TEXTURE_BASE_LEVEL && pname != GL_TEXTURE_MAX_LEVEL) {
        gl4metalSetError(GL_INVALID_ENUM); return NO;
    }
    if ((pname == GL_TEXTURE_BASE_LEVEL || pname == GL_TEXTURE_MAX_LEVEL) && value < 0.0f) {
        gl4metalSetError(GL_INVALID_VALUE); return NO;
    }
    if (!gl4metalValidSamplerParameter(pname, value)) { gl4metalSetError(GL_INVALID_ENUM); return NO; }
    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
    if (!texture || ![gl4metalTextureObjectNames containsObject:@(texture)]) {
        gl4metalSetError(GL_INVALID_OPERATION); return NO;
    }
    gl4metalEnsureTextureParams(texture)[@(pname)] = @(value);
    [gl4metalTextureSamplers removeObjectForKey:@(texture)];
    gl4metalSetError(GL_NO_ERROR);
    return YES;
}

static NSData *gl4metalGetTextureBorderData(GLuint texture) {
    NSData *data = gl4metalTextureBorderColors[@(texture)];
    if (!data) {
        const GLfloat defaultColor[4] = {0, 0, 0, 0};
        data = [NSData dataWithBytes:defaultColor length:sizeof(defaultColor)];
        gl4metalTextureBorderColors[@(texture)] = data;
    }
    return data;
}

static BOOL gl4metalSetBoundTextureBorder(GLenum target, const GLfloat color[4]) {
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return NO; }
    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
    if (!texture || ![gl4metalTextureObjectNames containsObject:@(texture)]) {
        gl4metalSetError(GL_INVALID_OPERATION); return NO;
    }
    gl4metalTextureBorderColors[@(texture)] = [NSData dataWithBytes:color length:sizeof(GLfloat) * 4];
    [gl4metalTextureSamplers removeObjectForKey:@(texture)];
    gl4metalSetError(GL_NO_ERROR);
    return YES;
}

void APIENTRY glTexParameteri(GLenum target, GLenum pname, GLint param) {
    (void)gl4metalSetBoundTextureScalar(target, pname, (GLfloat)param);
}

void APIENTRY glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
    (void)gl4metalSetBoundTextureScalar(target, pname, param);
}

void APIENTRY glTexParameteriv(GLenum target, GLenum pname, const GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        GLfloat color[4] = {(GLfloat)params[0], (GLfloat)params[1], (GLfloat)params[2], (GLfloat)params[3]};
        (void)gl4metalSetBoundTextureBorder(target, color);
    } else {
        (void)gl4metalSetBoundTextureScalar(target, pname, (GLfloat)params[0]);
    }
}

void APIENTRY glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_TEXTURE_BORDER_COLOR) (void)gl4metalSetBoundTextureBorder(target, params);
    else (void)gl4metalSetBoundTextureScalar(target, pname, params[0]);
}

void APIENTRY glTexParameterIiv(GLenum target, GLenum pname, const GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if ((target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) || pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalSetError(GL_INVALID_OPERATION); // No integer texture internal formats are allocated by this backend.
}

void APIENTRY glTexParameterIuiv(GLenum target, GLenum pname, const GLuint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if ((target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) || pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

static NSNumber *gl4metalGetBoundTextureParameter(GLenum target, GLenum pname, GLuint *textureOut) {
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return nil; }
    if ((target == GL_TEXTURE_1D && pname == GL_TEXTURE_WRAP_T) || pname == GL_TEXTURE_WRAP_R) {
        gl4metalSetError(GL_INVALID_ENUM); return nil;
    }
    if (pname == GL_TEXTURE_BORDER_COLOR || (!gl4metalIsSamplerScalarParameter(pname) &&
        pname != GL_TEXTURE_BASE_LEVEL && pname != GL_TEXTURE_MAX_LEVEL)) {
        gl4metalSetError(GL_INVALID_ENUM); return nil;
    }
    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
    if (texture == 0 || ![gl4metalTextureObjectNames containsObject:@(texture)]) {
        gl4metalSetError(GL_INVALID_OPERATION); return nil;
    }
    if (textureOut) *textureOut = texture;
    NSNumber *value = gl4metalEnsureTextureParams(texture)[@(pname)];
    gl4metalSetError(GL_NO_ERROR);
    return value;
}

void APIENTRY glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLuint texture = 0;
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
        gl4metalEnsureTextureState(); texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
        if (!texture || ![gl4metalTextureObjectNames containsObject:@(texture)]) { gl4metalSetError(GL_INVALID_OPERATION); return; }
        const GLfloat *color = (const GLfloat *)gl4metalGetTextureBorderData(texture).bytes;
        for (NSUInteger i = 0; i < 4; ++i) params[i] = (GLint)color[i];
        gl4metalSetError(GL_NO_ERROR); return;
    }
    NSNumber *value = gl4metalGetBoundTextureParameter(target, pname, &texture);
    if (value) params[0] = value.intValue;
}

void APIENTRY glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLuint texture = 0;
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
        gl4metalEnsureTextureState(); texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
        if (!texture || ![gl4metalTextureObjectNames containsObject:@(texture)]) { gl4metalSetError(GL_INVALID_OPERATION); return; }
        memcpy(params, gl4metalGetTextureBorderData(texture).bytes, sizeof(GLfloat) * 4);
        gl4metalSetError(GL_NO_ERROR); return;
    }
    NSNumber *value = gl4metalGetBoundTextureParameter(target, pname, &texture);
    if (value) params[0] = value.floatValue;
}

void APIENTRY glGetTexParameterIiv(GLenum target, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if ((target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) || pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if ((target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) || pname != GL_TEXTURE_BORDER_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLsizei height, GLint border,
                           GLenum format, GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || width < 0 || height < 0 || border != 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    switch ((GLenum)internalformat) {
        case GL_RED: case GL_R8: case GL_RG: case GL_RG8: case GL_ALPHA:
        case GL_RGB: case GL_RGB8: case GL_RGB565:
        case GL_RGBA: case GL_RGBA8: case GL_RGBA4: case GL_RGB5_A1:
        case GL_LUMINANCE: case GL_LUMINANCE_ALPHA:
        case GL_DEPTH_COMPONENT: case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32F: case GL_DEPTH24_STENCIL8: case GL_STENCIL_INDEX8:
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    BOOL depthInternalFormat = internalformat == GL_DEPTH_COMPONENT ||
        internalformat == GL_DEPTH_COMPONENT16 || internalformat == GL_DEPTH_COMPONENT24 ||
        internalformat == GL_DEPTH_COMPONENT32F;
    BOOL validPixelInput = depthInternalFormat
        ? (format == GL_DEPTH_COMPONENT && (type == GL_FLOAT || type == GL_UNSIGNED_SHORT || type == GL_UNSIGNED_INT))
        : gl4metalBytesPerPixelForFormatType(format, type) != 0;
    if (!validPixelInput) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx || !ctx.device) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    if (!textureName) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    id<MTLTexture> texture = gl4metalTextureObjects[@(textureName)];

    if (width == 0 || height == 0) {
        NSMutableDictionary *levelInfo = gl4metalTextureLevelInfo[@(textureName)];
        if (!levelInfo) {
            levelInfo = [NSMutableDictionary dictionary];
            gl4metalTextureLevelInfo[@(textureName)] = levelInfo;
        }
        levelInfo[@(level)] = @{@"internalFormat": @((GLenum)internalformat),
                               @"format": @(format), @"type": @(type),
                               @"width": @(width), @"height": @(height)};
        if (level == 0) [gl4metalTextureObjects removeObjectForKey:@(textureName)];
        [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
        if (ctx) ctx.pipelineState = nil;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    // Level 0 establishes the Metal allocation. Allocate the full mip chain so
    // later GL level uploads are legal even if the first mip-level parameter is
    // changed before the higher levels arrive.
    if (level == 0) {
        NSUInteger mipLevels = gl4metalMipLevelCount(width, height);
        MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;
        BOOL isDepth = NO;
        switch ((GLenum)internalformat) {
            case GL_DEPTH_COMPONENT:
            case GL_DEPTH_COMPONENT16:
            case GL_DEPTH_COMPONENT24:
            case GL_DEPTH_COMPONENT32F:
                pixelFormat = MTLPixelFormatDepth32Float;
                isDepth = YES;
                break;
            case GL_DEPTH24_STENCIL8:
                pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                isDepth = YES;
                break;
            case GL_STENCIL_INDEX8:
                pixelFormat = MTLPixelFormatStencil8;
                isDepth = YES;
                break;
            default:
                break;
        }
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                                        width:(NSUInteger)width
                                                                                       height:(NSUInteger)height
                                                                                    mipmapped:(isDepth ? NO : (mipLevels > 1))];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        // Ordinary GL textures stay GPU-private. CPU uploads are staged through
        // a shared MTLBuffer and copied with a Metal blit command.
        td.storageMode = MTLStorageModePrivate;
        texture = [ctx.device newTextureWithDescriptor:td];
        if (!texture) {
            gl4metalSetError(GL_OUT_OF_MEMORY);
            return;
        }
        gl4metalTextureObjects[@(textureName)] = texture;
    } else if (!texture) {
        // GL allows the level 0 allocation to precede higher levels. Without it
        // there is no size from which a Metal texture can be created.
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    BOOL uploadSucceeded = depthInternalFormat
        ? gl4metalUploadDepthPixels(texture, level, 0, 0, width, height, type, pixels)
        : gl4metalUploadPixelsToTexture(texture, level, 0, 0, width, height,
                                        (GLenum)internalformat, format, type, pixels);
    if (!uploadSucceeded) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSMutableDictionary *levelInfo = gl4metalTextureLevelInfo[@(textureName)];
    if (!levelInfo) {
        levelInfo = [NSMutableDictionary dictionary];
        gl4metalTextureLevelInfo[@(textureName)] = levelInfo;
    }
    levelInfo[@(level)] = @{@"internalFormat": @((GLenum)internalformat),
                            @"format": @(format), @"type": @(type),
                            @"width": @(width), @"height": @(height)};

    // Keep sampling changes deterministic when a texture is re-uploaded.
    [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
    ctx.pipelineState = nil;
    if (getenv("GL4METAL_TEXTURE_TRACE")) {
        printf("[gl4metal][TEXTURE] glTexImage2D GLtex=%u level=%d size=%dx%d format=0x%x type=0x%x pixels=%s\n",
               textureName, level, width, height, (unsigned)format, (unsigned)type,
               pixels ? "yes" : "null");
    }
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalBegin1DTextureOperation(NSNumber **previous2DBinding) {
    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, GL_TEXTURE_1D);
    if (texture == 0) { gl4metalSetError(GL_INVALID_OPERATION); return NO; }
    if (previous2DBinding) *previous2DBinding = gl4metalTextureUnitBindings[@(gl4metalActiveTextureUnit)];
    gl4metalTextureUnitBindings[@(gl4metalActiveTextureUnit)] = @(texture);
    return YES;
}

static void gl4metalEnd1DTextureOperation(NSNumber *previous2DBinding) {
    if (previous2DBinding) gl4metalTextureUnitBindings[@(gl4metalActiveTextureUnit)] = previous2DBinding;
    else [gl4metalTextureUnitBindings removeObjectForKey:@(gl4metalActiveTextureUnit)];
}

void APIENTRY glTexImage1D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLint border, GLenum format,
                           GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    NSNumber *previous = nil;
    if (!gl4metalBegin1DTextureOperation(&previous)) return;
    glTexImage2D(GL_TEXTURE_2D, level, internalformat, width, 1, border, format, type, pixels);
    gl4metalEnd1DTextureOperation(previous);
}

static GLint gl4metalTextureLevelParameter(NSDictionary<NSString *, NSNumber *> *info,
                                           GLenum pname) {
    if (!info) return 0;
    GLint width = [info[@"width"] intValue];
    GLint height = [info[@"height"] intValue];
    GLenum internalFormat = [info[@"internalFormat"] unsignedIntValue];
    switch (pname) {
        case GL_TEXTURE_WIDTH: return width;
        case GL_TEXTURE_HEIGHT: return height;
        case GL_TEXTURE_DEPTH: return 1;
        case GL_TEXTURE_INTERNAL_FORMAT: return (GLint)internalFormat;
        case GL_TEXTURE_COMPRESSED: return GL_FALSE;
        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE: return 0;
        case GL_TEXTURE_RED_SIZE:
            if (internalFormat == GL_RGBA4) return 4;
            if (internalFormat == GL_RGB5_A1 || internalFormat == GL_RGB565) return 5;
            if (internalFormat == GL_ALPHA || internalFormat == GL_DEPTH_COMPONENT || internalFormat == GL_DEPTH_COMPONENT16 ||
                internalFormat == GL_DEPTH_COMPONENT24 || internalFormat == GL_DEPTH_COMPONENT32F ||
                internalFormat == GL_DEPTH24_STENCIL8 || internalFormat == GL_STENCIL_INDEX8) return 0;
            return 8;
        case GL_TEXTURE_GREEN_SIZE:
            if (internalFormat == GL_RGBA4) return 4;
            if (internalFormat == GL_RGB5_A1) return 5;
            if (internalFormat == GL_RGB565) return 6;
            return (internalFormat == GL_RED || internalFormat == GL_R8 || internalFormat == GL_ALPHA ||
                    internalFormat == GL_DEPTH_COMPONENT || internalFormat == GL_DEPTH_COMPONENT16 ||
                    internalFormat == GL_DEPTH_COMPONENT24 || internalFormat == GL_DEPTH_COMPONENT32F ||
                    internalFormat == GL_DEPTH24_STENCIL8 || internalFormat == GL_STENCIL_INDEX8) ? 0 : 8;
        case GL_TEXTURE_BLUE_SIZE:
            if (internalFormat == GL_RGBA4) return 4;
            if (internalFormat == GL_RGB5_A1 || internalFormat == GL_RGB565) return 5;
            return (internalFormat == GL_RED || internalFormat == GL_R8 || internalFormat == GL_RG ||
                    internalFormat == GL_RG8 || internalFormat == GL_DEPTH_COMPONENT ||
                    internalFormat == GL_ALPHA ||
                    internalFormat == GL_DEPTH_COMPONENT16 || internalFormat == GL_DEPTH_COMPONENT24 ||
                    internalFormat == GL_DEPTH_COMPONENT32F || internalFormat == GL_DEPTH24_STENCIL8 ||
                    internalFormat == GL_STENCIL_INDEX8) ? 0 : 8;
        case GL_TEXTURE_ALPHA_SIZE:
            if (internalFormat == GL_RGBA4) return 4;
            if (internalFormat == GL_RGB5_A1) return 1;
            return (internalFormat == GL_RGBA || internalFormat == GL_RGBA8 ||
                    internalFormat == GL_LUMINANCE_ALPHA || internalFormat == GL_ALPHA) ? 8 : 0;
        case GL_TEXTURE_DEPTH_SIZE:
            return (internalFormat == GL_DEPTH_COMPONENT || internalFormat == GL_DEPTH_COMPONENT16) ? 16 :
                   (internalFormat == GL_DEPTH_COMPONENT24 || internalFormat == GL_DEPTH24_STENCIL8) ? 24 :
                   internalFormat == GL_DEPTH_COMPONENT32F ? 32 : 0;
        case GL_TEXTURE_STENCIL_SIZE:
            return internalFormat == GL_DEPTH24_STENCIL8 || internalFormat == GL_STENCIL_INDEX8 ? 8 : 0;
        default: return INT_MIN;
    }
}

void APIENTRY glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (level < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
    if (!textureName) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    NSDictionary *info = gl4metalTextureLevelInfo[@(textureName)][@(level)];
    GLint result = gl4metalTextureLevelParameter(info, pname);
    if (result == INT_MIN) { gl4metalSetError(GL_INVALID_ENUM); return; }
    params[0] = result;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLint value = 0;
    glGetTexLevelParameteriv(target, level, pname, &value);
    params[0] = (GLfloat)value;
}

void APIENTRY glTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                              GLint yoffset, GLsizei width, GLsizei height,
                              GLenum format, GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    id<MTLTexture> texture = textureName ? gl4metalTextureObjects[@(textureName)] : nil;
    if (!texture) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSDictionary *levelInfo = gl4metalTextureLevelInfo[@(textureName)][@(level)];
    if (!levelInfo) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    GLenum internalFormat = [levelInfo[@"internalFormat"] unsignedIntValue];
    BOOL depthInternalFormat = internalFormat == GL_DEPTH_COMPONENT ||
        internalFormat == GL_DEPTH_COMPONENT16 || internalFormat == GL_DEPTH_COMPONENT24 ||
        internalFormat == GL_DEPTH_COMPONENT32F;
    BOOL validPixelInput = depthInternalFormat
        ? (format == GL_DEPTH_COMPONENT && (type == GL_FLOAT || type == GL_UNSIGNED_SHORT || type == GL_UNSIGNED_INT))
        : gl4metalBytesPerPixelForFormatType(format, type) != 0;
    if (!validPixelInput) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (width == 0 || height == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    if (!pixels) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLint imageWidth = [levelInfo[@"width"] intValue];
    GLint imageHeight = [levelInfo[@"height"] intValue];
    if (xoffset > imageWidth || yoffset > imageHeight ||
        width > imageWidth - xoffset || height > imageHeight - yoffset) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    BOOL uploaded = depthInternalFormat
        ? gl4metalUploadDepthPixels(texture, level, xoffset, yoffset, width, height, type, pixels)
        : gl4metalUploadPixelsToTexture(texture, level, xoffset, yoffset, width, height,
                                        internalFormat, format, type, pixels);
    if (!uploaded) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
    if (getenv("GL4METAL_TEXTURE_TRACE")) {
        printf("[gl4metal][TEXTURE] glTexSubImage2D GLtex=%u level=%d rect=%d,%d %dx%d format=0x%x type=0x%x\n",
               textureName, level, xoffset, yoffset, width, height,
               (unsigned)format, (unsigned)type);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexSubImage1D(GLenum target, GLint level, GLint xoffset,
                              GLsizei width, GLenum format, GLenum type,
                              const void *pixels) {
    if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    NSNumber *previous = nil;
    if (!gl4metalBegin1DTextureOperation(&previous)) return;
    glTexSubImage2D(GL_TEXTURE_2D, level, xoffset, 0, width, 1, format, type, pixels);
    gl4metalEnd1DTextureOperation(previous);
}

void APIENTRY glGenerateMipmap(GLenum target) {
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx || !ctx.commandQueue) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnitForTarget(gl4metalActiveTextureUnit, target);
    id<MTLTexture> texture = textureName ? gl4metalTextureObjects[@(textureName)] : nil;
    if (!texture) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSDictionary *baseImage = gl4metalTextureLevelInfo[@(textureName)][@0];
    if (!baseImage || [baseImage[@"width"] intValue] <= 0 || [baseImage[@"height"] intValue] <= 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (texture.mipmapLevelCount <= 1) {
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    id<MTLCommandBuffer> cb = [ctx.commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!cb || !blit) {
        gl4metalSetError(GL_OUT_OF_MEMORY);
        return;
    }
    [blit generateMipmapsForTexture:texture];
    [blit endEncoding];
    cb.label = [NSString stringWithFormat:@"gl4metal mipmap texture %u", textureName];
    [cb commit];
    NSMutableDictionary *levels = gl4metalTextureLevelInfo[@(textureName)];
    GLsizei baseWidth = [baseImage[@"width"] intValue];
    GLsizei baseHeight = [baseImage[@"height"] intValue];
    for (NSUInteger level = 1; level < texture.mipmapLevelCount; ++level) {
        levels[@(level)] = @{
            @"internalFormat": baseImage[@"internalFormat"] ?: @(GL_RGBA8),
            @"format": baseImage[@"format"] ?: @(GL_RGBA),
            @"type": baseImage[@"type"] ?: @(GL_UNSIGNED_BYTE),
            @"width": @(MAX(1, baseWidth >> MIN(level, (NSUInteger)31))),
            @"height": @(MAX(1, baseHeight >> MIN(level, (NSUInteger)31)))
        };
    }
    [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
    gl4metalSetError(GL_NO_ERROR);
}

// -----------------------------------------------------------------------------
// Real framebuffer / renderbuffer implementation.
// -----------------------------------------------------------------------------

void gl4metalInitFramebufferState(void) {
    if (!gl4metalFramebufferColor) gl4metalFramebufferColor = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferDepth) gl4metalFramebufferDepth = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferStencil) gl4metalFramebufferStencil = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferColorRB) gl4metalFramebufferColorRB = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferDepthRB) gl4metalFramebufferDepthRB = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferStencilRB) gl4metalFramebufferStencilRB = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferColorLevel) gl4metalFramebufferColorLevel = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferDepthLevel) gl4metalFramebufferDepthLevel = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferStencilLevel) gl4metalFramebufferStencilLevel = [NSMutableDictionary dictionary];
    if (!gl4metalRenderbufferTextures) gl4metalRenderbufferTextures = [NSMutableDictionary dictionary];
    if (!gl4metalRenderbufferFormats) gl4metalRenderbufferFormats = [NSMutableDictionary dictionary];
    if (!gl4metalFramebufferGeneratedNames) gl4metalFramebufferGeneratedNames = [NSMutableSet set];
    if (!gl4metalRenderbufferGeneratedNames) gl4metalRenderbufferGeneratedNames = [NSMutableSet set];
}

void gl4metalResetFramebufferState(void) {
    gl4metalFramebufferColor = [NSMutableDictionary dictionary];
    gl4metalFramebufferDepth = [NSMutableDictionary dictionary];
    gl4metalFramebufferStencil = [NSMutableDictionary dictionary];
    gl4metalFramebufferColorRB = [NSMutableDictionary dictionary];
    gl4metalFramebufferDepthRB = [NSMutableDictionary dictionary];
    gl4metalFramebufferStencilRB = [NSMutableDictionary dictionary];
    gl4metalFramebufferColorLevel = [NSMutableDictionary dictionary];
    gl4metalFramebufferDepthLevel = [NSMutableDictionary dictionary];
    gl4metalFramebufferStencilLevel = [NSMutableDictionary dictionary];
    gl4metalRenderbufferTextures = [NSMutableDictionary dictionary];
    gl4metalRenderbufferFormats = [NSMutableDictionary dictionary];
    gl4metalFramebufferGeneratedNames = [NSMutableSet set];
    gl4metalRenderbufferGeneratedNames = [NSMutableSet set];
    gl4metalCurrentRenderbuffer = 0;
    gl4metalNextFramebufferId = 1;
    gl4metalNextRenderbufferId = 1;
}

static id<MTLTexture> gl4metalFramebufferAttachmentTexture(GLuint framebuffer, GLenum attachment) {
    if (framebuffer == 0) return nil;
    gl4metalInitFramebufferState();
    NSNumber *texName = nil;
    NSNumber *rbName = nil;
    if (attachment == GL_COLOR_ATTACHMENT0) {
        texName = gl4metalFramebufferColor[@(framebuffer)];
        rbName = gl4metalFramebufferColorRB[@(framebuffer)];
    } else if (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        texName = gl4metalFramebufferDepth[@(framebuffer)];
        rbName = gl4metalFramebufferDepthRB[@(framebuffer)];
    } else if (attachment == GL_STENCIL_ATTACHMENT) {
        texName = gl4metalFramebufferStencil[@(framebuffer)];
        rbName = gl4metalFramebufferStencilRB[@(framebuffer)];
    }
    if (texName) {
        return gl4metalTextureObjects[texName];
    }
    if (rbName) {
        return gl4metalRenderbufferTextures[rbName];
    }
    return nil;
}

static void gl4metalDetachTextureFromAllFramebuffers(GLuint texture) {
    gl4metalInitFramebufferState();
    NSArray<NSNumber *> *keys = [gl4metalFramebufferColor allKeysForObject:@(texture)];
    for (NSNumber *key in keys) {
        [gl4metalFramebufferColor removeObjectForKey:key];
        [gl4metalFramebufferColorLevel removeObjectForKey:key];
    }
    keys = [gl4metalFramebufferDepth allKeysForObject:@(texture)];
    for (NSNumber *key in keys) {
        [gl4metalFramebufferDepth removeObjectForKey:key];
        [gl4metalFramebufferDepthLevel removeObjectForKey:key];
    }
    keys = [gl4metalFramebufferStencil allKeysForObject:@(texture)];
    for (NSNumber *key in keys) {
        [gl4metalFramebufferStencil removeObjectForKey:key];
        [gl4metalFramebufferStencilLevel removeObjectForKey:key];
    }
}

static NSUInteger gl4metalFramebufferAttachmentLevel(GLuint framebuffer, GLenum attachment) {
    gl4metalInitFramebufferState();
    if (attachment == GL_COLOR_ATTACHMENT0)
        return [gl4metalFramebufferColorLevel[@(framebuffer)] unsignedIntegerValue];
    if (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
        return [gl4metalFramebufferDepthLevel[@(framebuffer)] unsignedIntegerValue];
    if (attachment == GL_STENCIL_ATTACHMENT)
        return [gl4metalFramebufferStencilLevel[@(framebuffer)] unsignedIntegerValue];
    return 0;
}

NSUInteger gl4metalGetCurrentFramebufferColorLevel(void) {
    return ctx ? gl4metalFramebufferAttachmentLevel(ctx.currentFramebuffer, GL_COLOR_ATTACHMENT0) : 0;
}

NSUInteger gl4metalGetCurrentFramebufferDepthLevel(void) {
    if (!ctx) return 0;
    return gl4metalFramebufferAttachmentLevel(ctx.currentFramebuffer, GL_DEPTH_ATTACHMENT);
}

NSUInteger gl4metalGetCurrentFramebufferStencilLevel(void) {
    if (!ctx) return 0;
    NSUInteger level = gl4metalFramebufferAttachmentLevel(ctx.currentFramebuffer, GL_STENCIL_ATTACHMENT);
    if (!gl4metalFramebufferStencil[@(ctx.currentFramebuffer)])
        level = gl4metalFramebufferAttachmentLevel(ctx.currentFramebuffer, GL_DEPTH_STENCIL_ATTACHMENT);
    return level;
}

id<MTLTexture> gl4metalGetCurrentFramebufferColorTexture(void) {
    if (!ctx || ctx.currentFramebuffer == 0) return nil;
    return gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_COLOR_ATTACHMENT0);
}

id<MTLTexture> gl4metalGetCurrentFramebufferDepthTexture(void) {
    if (!ctx || ctx.currentFramebuffer == 0) return nil;
    id<MTLTexture> depth = gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_DEPTH_ATTACHMENT);
    if (!depth) depth = gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_DEPTH_STENCIL_ATTACHMENT);
    return depth;
}

id<MTLTexture> gl4metalGetCurrentFramebufferStencilTexture(void) {
    if (!ctx || ctx.currentFramebuffer == 0) return nil;
    id<MTLTexture> stencil = gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_STENCIL_ATTACHMENT);
    if (!stencil) stencil = gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_DEPTH_STENCIL_ATTACHMENT);
    return stencil;
}

void APIENTRY glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height,
                           GLenum format, GLenum type, void *pixels) {
    if (width < 0 || height < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    BOOL depthRead = format == GL_DEPTH_COMPONENT;
    BOOL depthTypeValid = type == GL_FLOAT || type == GL_UNSIGNED_SHORT || type == GL_UNSIGNED_INT;
    if ((!depthRead && !gl4metalValidReadbackFormatType(format, type)) ||
        (depthRead && !depthTypeValid)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (!depthRead && ctx.readBuffer == GL_NONE) { gl4metalSetError(GL_INVALID_OPERATION); return; }

    id<MTLTexture> texture = nil;
    NSUInteger level = 0;
    if (ctx.currentReadFramebuffer == 0) {
        if (depthRead) texture = gl4metalGetDefaultDepthTexture();
        else {
            if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
            texture = ctx.currentDrawable.texture;
        }
    } else {
        GLenum status = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
            return;
        }
        GLenum attachment = depthRead ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0;
        texture = gl4metalFramebufferAttachmentTexture(ctx.currentReadFramebuffer, attachment);
        if (!texture && depthRead) texture = gl4metalFramebufferAttachmentTexture(ctx.currentReadFramebuffer, GL_DEPTH_STENCIL_ATTACHMENT);
        level = gl4metalFramebufferAttachmentLevel(ctx.currentReadFramebuffer, attachment);
    }
    if (!texture) { gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION); return; }
    GLenum error = depthRead
        ? gl4metalReadDepthTexture(texture, level, x, y, width, height, type, pixels, YES)
        : gl4metalReadColorTexture(texture, level, x, y, width, height, format, type, pixels, YES);
    gl4metalSetError(error);
}

void APIENTRY glGetTexImage(GLenum target, GLint level, GLenum format,
                            GLenum type, void *pixels) {
    if (target == GL_TEXTURE_1D) {
        NSNumber *previous = nil;
        if (!gl4metalBegin1DTextureOperation(&previous)) return;
        glGetTexImage(GL_TEXTURE_2D, level, format, type, pixels);
        gl4metalEnd1DTextureOperation(previous);
        return;
    }
    if (target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (level < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    BOOL depthRead = format == GL_DEPTH_COMPONENT;
    BOOL depthTypeValid = type == GL_FLOAT || type == GL_UNSIGNED_SHORT || type == GL_UNSIGNED_INT;
    if ((!depthRead && !gl4metalValidReadbackFormatType(format, type)) ||
        (depthRead && !depthTypeValid)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    if (textureName == 0) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    NSDictionary *info = gl4metalTextureLevelInfo[@(textureName)][@(level)];
    if (!info) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    GLsizei width = [info[@"width"] intValue];
    GLsizei height = [info[@"height"] intValue];
    if (width == 0 || height == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    id<MTLTexture> texture = gl4metalTextureObjects[@(textureName)];
    GLenum error = depthRead
        ? gl4metalReadDepthTexture(texture, (NSUInteger)level, 0, 0, width, height, type, pixels, NO)
        : gl4metalReadColorTexture(texture, (NSUInteger)level, 0, 0, width, height, format, type, pixels, NO);
    gl4metalSetError(error);
}

static GLenum gl4metalReadCurrentColor(GLint x, GLint y, GLsizei width, GLsizei height,
                                        void *pixels) {
    if (!ctx || ctx.readBuffer == GL_NONE) return GL_INVALID_OPERATION;
    id<MTLTexture> texture = nil;
    NSUInteger level = 0;
    if (ctx.currentReadFramebuffer == 0) {
        if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
        texture = ctx.currentDrawable.texture;
    } else {
        if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return GL_INVALID_FRAMEBUFFER_OPERATION;
        texture = gl4metalFramebufferAttachmentTexture(ctx.currentReadFramebuffer, GL_COLOR_ATTACHMENT0);
        level = gl4metalFramebufferAttachmentLevel(ctx.currentReadFramebuffer, GL_COLOR_ATTACHMENT0);
    }
    if (!texture) return GL_INVALID_FRAMEBUFFER_OPERATION;
    return gl4metalReadColorTexture(texture, level, x, y, width, height,
                                   GL_RGBA, GL_UNSIGNED_BYTE, pixels, YES);
}

static GLenum gl4metalCopyFramebufferRGBA(GLint x, GLint y, GLsizei width,
                                           GLsizei height, NSMutableData **outPixels) {
    if (width < 0 || height < 0) return GL_INVALID_VALUE;
    if ((NSUInteger)width > SIZE_MAX / 4u ||
        (NSUInteger)height > SIZE_MAX / MAX((NSUInteger)1, (NSUInteger)width * 4u))
        return GL_INVALID_VALUE;
    NSUInteger length = (NSUInteger)width * (NSUInteger)height * 4u;
    NSMutableData *pixels = [NSMutableData dataWithLength:MAX((NSUInteger)1, length)];
    if (!pixels) return GL_OUT_OF_MEMORY;

    GLint oldAlignment = gl4metalPackAlignment;
    GLint oldRowLength = gl4metalPackRowLength;
    GLint oldSkipRows = gl4metalPackSkipRows;
    GLint oldSkipPixels = gl4metalPackSkipPixels;
    gl4metalPackAlignment = 1;
    gl4metalPackRowLength = gl4metalPackSkipRows = gl4metalPackSkipPixels = 0;
    GLenum error = GL_NO_ERROR;
    if (width > 0 && height > 0)
        error = gl4metalReadCurrentColor(x, y, width, height, pixels.mutableBytes);
    gl4metalPackAlignment = oldAlignment;
    gl4metalPackRowLength = oldRowLength;
    gl4metalPackSkipRows = oldSkipRows;
    gl4metalPackSkipPixels = oldSkipPixels;
    if (error != GL_NO_ERROR) return error;
    *outPixels = pixels;
    return GL_NO_ERROR;
}

void APIENTRY glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat,
                               GLint x, GLint y, GLsizei width, GLsizei height,
                               GLint border) {
    if (target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (level < 0 || width < 0 || height < 0 || border != 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    NSMutableData *pixels = nil;
    GLenum error = gl4metalCopyFramebufferRGBA(x, y, width, height, &pixels);
    if (error != GL_NO_ERROR) { gl4metalSetError(error); return; }
    GLint oldAlignment = gl4metalUnpackAlignment;
    GLint oldRowLength = gl4metalUnpackRowLength;
    GLint oldSkipRows = gl4metalUnpackSkipRows;
    GLint oldSkipPixels = gl4metalUnpackSkipPixels;
    gl4metalUnpackAlignment = 1;
    gl4metalUnpackRowLength = gl4metalUnpackSkipRows = gl4metalUnpackSkipPixels = 0;
    glTexImage2D(target, level, (GLint)internalformat, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels.bytes);
    gl4metalUnpackAlignment = oldAlignment;
    gl4metalUnpackRowLength = oldRowLength;
    gl4metalUnpackSkipRows = oldSkipRows;
    gl4metalUnpackSkipPixels = oldSkipPixels;
}

void APIENTRY glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                                  GLint yoffset, GLint x, GLint y,
                                  GLsizei width, GLsizei height) {
    if (target != GL_TEXTURE_2D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    NSMutableData *pixels = nil;
    GLenum error = gl4metalCopyFramebufferRGBA(x, y, width, height, &pixels);
    if (error != GL_NO_ERROR) { gl4metalSetError(error); return; }
    GLint oldAlignment = gl4metalUnpackAlignment;
    GLint oldRowLength = gl4metalUnpackRowLength;
    GLint oldSkipRows = gl4metalUnpackSkipRows;
    GLint oldSkipPixels = gl4metalUnpackSkipPixels;
    gl4metalUnpackAlignment = 1;
    gl4metalUnpackRowLength = gl4metalUnpackSkipRows = gl4metalUnpackSkipPixels = 0;
    glTexSubImage2D(target, level, xoffset, yoffset, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, pixels.bytes);
    gl4metalUnpackAlignment = oldAlignment;
    gl4metalUnpackRowLength = oldRowLength;
    gl4metalUnpackSkipRows = oldSkipRows;
    gl4metalUnpackSkipPixels = oldSkipPixels;
}

void APIENTRY glCopyTexImage1D(GLenum target, GLint level, GLenum internalformat,
                               GLint x, GLint y, GLsizei width, GLint border) {
    if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    NSNumber *previous = nil;
    if (!gl4metalBegin1DTextureOperation(&previous)) return;
    glCopyTexImage2D(GL_TEXTURE_2D, level, internalformat, x, y, width, 1, border);
    gl4metalEnd1DTextureOperation(previous);
}

void APIENTRY glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset,
                                  GLint x, GLint y, GLsizei width) {
    if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
    NSNumber *previous = nil;
    if (!gl4metalBegin1DTextureOperation(&previous)) return;
    glCopyTexSubImage2D(GL_TEXTURE_2D, level, xoffset, 0, x, y, width, 1);
    gl4metalEnd1DTextureOperation(previous);
}

void APIENTRY glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                                GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                                GLbitfield mask, GLenum filter) {
    const GLbitfield validMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    if (mask & ~validMask) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (filter != GL_NEAREST && filter != GL_LINEAR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (mask == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    if (mask != GL_COLOR_BUFFER_BIT) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (ctx.readBuffer == GL_NONE || ctx.drawBuffer == GL_NONE) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if ((ctx.currentReadFramebuffer != 0 &&
         glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) ||
        (ctx.currentFramebuffer != 0 &&
         glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }

    id<MTLTexture> source = nil;
    id<MTLTexture> destination = nil;
    NSUInteger sourceLevel = 0, destinationLevel = 0;
    if (ctx.currentReadFramebuffer == 0) {
        if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
        source = ctx.currentDrawable.texture;
    } else {
        source = gl4metalFramebufferAttachmentTexture(ctx.currentReadFramebuffer, GL_COLOR_ATTACHMENT0);
        sourceLevel = gl4metalFramebufferAttachmentLevel(ctx.currentReadFramebuffer, GL_COLOR_ATTACHMENT0);
    }
    if (ctx.currentFramebuffer == 0) {
        if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
        destination = ctx.currentDrawable.texture;
    } else {
        destination = gl4metalFramebufferAttachmentTexture(ctx.currentFramebuffer, GL_COLOR_ATTACHMENT0);
        destinationLevel = gl4metalFramebufferAttachmentLevel(ctx.currentFramebuffer, GL_COLOR_ATTACHMENT0);
    }
    if (!source || !destination) { gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION); return; }
    NSUInteger sourceWidth = MAX((NSUInteger)1, source.width >> MIN(sourceLevel, (NSUInteger)31));
    NSUInteger sourceHeight = MAX((NSUInteger)1, source.height >> MIN(sourceLevel, (NSUInteger)31));
    NSUInteger destinationWidth = MAX((NSUInteger)1, destination.width >> MIN(destinationLevel, (NSUInteger)31));
    NSUInteger destinationHeight = MAX((NSUInteger)1, destination.height >> MIN(destinationLevel, (NSUInteger)31));
    if (srcX0 < 0 || srcY0 < 0 || srcX1 < srcX0 || srcY1 < srcY0 ||
        dstX0 < 0 || dstY0 < 0 || dstX1 < dstX0 || dstY1 < dstY0 ||
        (NSUInteger)srcX1 > sourceWidth || (NSUInteger)srcY1 > sourceHeight ||
        (NSUInteger)dstX1 > destinationWidth || (NSUInteger)dstY1 > destinationHeight) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLint copyWidth = srcX1 - srcX0;
    GLint copyHeight = srcY1 - srcY0;
    if (copyWidth != dstX1 - dstX0 || copyHeight != dstY1 - dstY0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (copyWidth == 0 || copyHeight == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    if (source.sampleCount != destination.sampleCount || source.sampleCount != 1 ||
        source.pixelFormat != destination.pixelFormat ||
        (source.pixelFormat != MTLPixelFormatRGBA8Unorm &&
         source.pixelFormat != MTLPixelFormatBGRA8Unorm)) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (source == destination && sourceLevel == destinationLevel &&
        srcX0 < dstX1 && dstX0 < srcX1 && srcY0 < dstY1 && dstY0 < srcY1) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    gl4metalFlushPendingClear();
    id<MTLCommandBuffer> commandBuffer = ctx.currentCommandBuffer;
    BOOL dedicated = NO;
    if (!commandBuffer) {
        commandBuffer = [ctx.commandQueue commandBuffer];
        dedicated = YES;
    }
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !blit) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
    [blit copyFromTexture:source
              sourceSlice:0
              sourceLevel:sourceLevel
             sourceOrigin:MTLOriginMake((NSUInteger)srcX0, sourceHeight - (NSUInteger)srcY1, 0)
               sourceSize:MTLSizeMake((NSUInteger)copyWidth, (NSUInteger)copyHeight, 1)
                 toTexture:destination
          destinationSlice:0
          destinationLevel:destinationLevel
         destinationOrigin:MTLOriginMake((NSUInteger)dstX0, destinationHeight - (NSUInteger)dstY1, 0)];
    [blit endEncoding];
    if (dedicated) [commandBuffer commit];
    gl4metalSetError(GL_NO_ERROR);
}

MTLPixelFormat gl4metalGetCurrentColorPixelFormat(void) {
    if (!ctx) return MTLPixelFormatInvalid;
    if (ctx.currentFramebuffer == 0) {
        if (ctx.currentDrawable) return ctx.currentDrawable.texture.pixelFormat;
        if (ctx.metalLayer) return ctx.metalLayer.pixelFormat;
        return MTLPixelFormatBGRA8Unorm;
    }
    id<MTLTexture> tex = gl4metalGetCurrentFramebufferColorTexture();
    return tex ? tex.pixelFormat : MTLPixelFormatInvalid;
}

MTLPixelFormat gl4metalGetCurrentDepthPixelFormat(void) {
    if (!ctx) return MTLPixelFormatInvalid;
    if (ctx.currentFramebuffer == 0) return MTLPixelFormatDepth32Float;
    id<MTLTexture> tex = gl4metalGetCurrentFramebufferDepthTexture();
    return tex ? tex.pixelFormat : MTLPixelFormatInvalid;
}

MTLPixelFormat gl4metalGetCurrentStencilPixelFormat(void) {
    if (!ctx || ctx.currentFramebuffer == 0) return MTLPixelFormatInvalid;
    id<MTLTexture> tex = gl4metalGetCurrentFramebufferStencilTexture();
    if (!tex) return MTLPixelFormatInvalid;
    MTLPixelFormat format = tex.pixelFormat;
    return (format == MTLPixelFormatStencil8 ||
            format == MTLPixelFormatDepth32Float_Stencil8)
        ? format : MTLPixelFormatInvalid;
}

void APIENTRY glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    if (n < 0 || (n > 0 && !framebuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = 0;
        do {
            framebuffer = gl4metalNextFramebufferId++;
            if (framebuffer == 0) framebuffer = gl4metalNextFramebufferId++;
        } while ([gl4metalFramebufferGeneratedNames containsObject:@(framebuffer)] ||
                 framebufferObjects[@(framebuffer)] != nil);
        framebuffers[i] = framebuffer;
        [gl4metalFramebufferGeneratedNames addObject:@(framebuffer)];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer) {
    if (!gl4metalIsDrawFramebufferTarget(target) && target != GL_READ_FRAMEBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    gl4metalInitFramebufferState();
    if (framebuffer != 0 && ![gl4metalFramebufferGeneratedNames containsObject:@(framebuffer)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (framebuffer != 0 && !framebufferObjects[@(framebuffer)]) {
        framebufferObjects[@(framebuffer)] = @(framebuffer);
    }
    if (gl4metalIsDrawFramebufferTarget(target)) {
        gl4metalFlushPendingClear();
        ctx.framebufferDrawBuffers[@(ctx.currentFramebuffer)] = @(ctx.drawBuffer);
        ctx.currentFramebuffer = framebuffer;
        NSNumber *drawBuffer = ctx.framebufferDrawBuffers[@(framebuffer)];
        if (!drawBuffer) {
            drawBuffer = @(framebuffer == 0 ? GL_BACK : GL_COLOR_ATTACHMENT0);
            ctx.framebufferDrawBuffers[@(framebuffer)] = drawBuffer;
        }
        ctx.drawBuffer = drawBuffer.unsignedIntValue;
        // Render target format and attachment set are part of the pipeline state.
        ctx.pipelineState = nil;
    }
    if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) {
        ctx.framebufferReadBuffers[@(ctx.currentReadFramebuffer)] = @(ctx.readBuffer);
        ctx.currentReadFramebuffer = framebuffer;
        NSNumber *readBuffer = ctx.framebufferReadBuffers[@(framebuffer)];
        if (!readBuffer) {
            readBuffer = @(framebuffer == 0 ? GL_BACK : GL_COLOR_ATTACHMENT0);
            ctx.framebufferReadBuffers[@(framebuffer)] = readBuffer;
        }
        ctx.readBuffer = readBuffer.unsignedIntValue;
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLenum APIENTRY glCheckFramebufferStatus(GLenum target) {
    if (!gl4metalIsDrawFramebufferTarget(target) && target != GL_READ_FRAMEBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }
    GLuint framebuffer = target == GL_READ_FRAMEBUFFER
        ? ctx.currentReadFramebuffer : ctx.currentFramebuffer;
    if (framebuffer == 0) {
        gl4metalSetError(GL_NO_ERROR);
        return GL_FRAMEBUFFER_COMPLETE;
    }

    gl4metalInitFramebufferState();
    GLuint fbo = framebuffer;
    id<MTLTexture> color = gl4metalFramebufferAttachmentTexture(fbo, GL_COLOR_ATTACHMENT0);
    id<MTLTexture> depth = gl4metalFramebufferAttachmentTexture(fbo, GL_DEPTH_ATTACHMENT);
    id<MTLTexture> ds = gl4metalFramebufferAttachmentTexture(fbo, GL_DEPTH_STENCIL_ATTACHMENT);
    id<MTLTexture> stencil = gl4metalFramebufferAttachmentTexture(fbo, GL_STENCIL_ATTACHMENT);
    id<MTLTexture> first = color ?: (depth ?: (ds ?: stencil));
    if (!first) {
        BOOL hasAttachment = gl4metalFramebufferColor[@(fbo)] || gl4metalFramebufferDepth[@(fbo)] ||
            gl4metalFramebufferStencil[@(fbo)] || gl4metalFramebufferColorRB[@(fbo)] ||
            gl4metalFramebufferDepthRB[@(fbo)] || gl4metalFramebufferStencilRB[@(fbo)];
        gl4metalSetError(GL_NO_ERROR);
        return hasAttachment ? GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
                             : GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }

    const GLenum points[] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT};
    NSUInteger firstWidth = 0, firstHeight = 0;
    for (NSUInteger i = 0; i < sizeof(points) / sizeof(points[0]); ++i) {
        GLenum point = points[i];
        id<MTLTexture> tex = gl4metalFramebufferAttachmentTexture(fbo, point);
        NSNumber *textureName = point == GL_COLOR_ATTACHMENT0 ? gl4metalFramebufferColor[@(fbo)] :
            (point == GL_DEPTH_ATTACHMENT ? gl4metalFramebufferDepth[@(fbo)] : gl4metalFramebufferStencil[@(fbo)]);
        NSNumber *renderbufferName = point == GL_COLOR_ATTACHMENT0 ? gl4metalFramebufferColorRB[@(fbo)] :
            (point == GL_DEPTH_ATTACHMENT ? gl4metalFramebufferDepthRB[@(fbo)] : gl4metalFramebufferStencilRB[@(fbo)]);
        if (!tex) {
            if (textureName || renderbufferName) {
                gl4metalSetError(GL_NO_ERROR);
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }
            continue;
        }
        NSUInteger level = gl4metalFramebufferAttachmentLevel(fbo, point);
        if (level >= tex.mipmapLevelCount) {
            gl4metalSetError(GL_NO_ERROR);
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
        MTLPixelFormat format = tex.pixelFormat;
        BOOL depthFormat = format == MTLPixelFormatDepth16Unorm ||
            format == MTLPixelFormatDepth32Float || format == MTLPixelFormatDepth32Float_Stencil8;
        BOOL stencilFormat = format == MTLPixelFormatStencil8 ||
            format == MTLPixelFormatDepth32Float_Stencil8;
        if ((point == GL_COLOR_ATTACHMENT0 && (depthFormat || format == MTLPixelFormatStencil8)) ||
            (point == GL_DEPTH_ATTACHMENT && !depthFormat) ||
            (point == GL_STENCIL_ATTACHMENT && !stencilFormat)) {
            gl4metalSetError(GL_NO_ERROR);
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
        NSUInteger width = MAX((NSUInteger)1, tex.width >> MIN(level, (NSUInteger)31));
        NSUInteger height = MAX((NSUInteger)1, tex.height >> MIN(level, (NSUInteger)31));
        if (textureName) {
            NSDictionary *image = gl4metalTextureLevelInfo[textureName][@(level)];
            if (!image || [image[@"width"] intValue] <= 0 || [image[@"height"] intValue] <= 0) {
                gl4metalSetError(GL_NO_ERROR);
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }
            width = (NSUInteger)[image[@"width"] unsignedIntValue];
            height = (NSUInteger)[image[@"height"] unsignedIntValue];
        }
        if (firstWidth == 0) {
            firstWidth = width;
            firstHeight = height;
        } else if (width != firstWidth || height != firstHeight) {
            gl4metalSetError(GL_NO_ERROR);
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
    }
    GLuint attachedColorTexture = [gl4metalFramebufferColor[@(fbo)] unsignedIntValue];
    GLuint attachedColorRenderbuffer = [gl4metalFramebufferColorRB[@(fbo)] unsignedIntValue];
    GLenum drawBuffer = [ctx.framebufferDrawBuffers[@(fbo)] unsignedIntValue];
    GLenum readBuffer = [ctx.framebufferReadBuffers[@(fbo)] unsignedIntValue];
    if (drawBuffer == GL_COLOR_ATTACHMENT0 &&
        attachedColorTexture == 0 && attachedColorRenderbuffer == 0) {
        gl4metalSetError(GL_NO_ERROR);
        return GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER;
    }
    if (readBuffer == GL_COLOR_ATTACHMENT0 &&
        attachedColorTexture == 0 && attachedColorRenderbuffer == 0) {
        gl4metalSetError(GL_NO_ERROR);
        return GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER;
    }
    gl4metalSetError(GL_NO_ERROR);
    return GL_FRAMEBUFFER_COMPLETE;
}

void APIENTRY glFramebufferTexture2D(GLenum target, GLenum attachment,
                                     GLenum textarget, GLuint texture, GLint level) {
    if (!gl4metalIsDrawFramebufferTarget(target) || textarget != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (level < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx || ctx.currentFramebuffer == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (attachment != GL_COLOR_ATTACHMENT0 && attachment != GL_DEPTH_ATTACHMENT &&
        attachment != GL_STENCIL_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalInitFramebufferState();
    GLuint fbo = ctx.currentFramebuffer;
    NSMutableDictionary *texMap = (attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColor :
                                    ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) ? gl4metalFramebufferDepth : gl4metalFramebufferStencil);
    NSMutableDictionary *rbMap = (attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColorRB :
                                   ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) ? gl4metalFramebufferDepthRB : gl4metalFramebufferStencilRB);
    if (texture == 0) {
        [texMap removeObjectForKey:@(fbo)];
    } else if ([gl4metalTextureObjectNames containsObject:@(texture)]) {
        id<MTLTexture> attachedTexture = gl4metalTextureObjects[@(texture)];
        if (attachedTexture && (NSUInteger)level >= attachedTexture.mipmapLevelCount) {
            gl4metalSetError(GL_INVALID_VALUE);
            return;
        }
        texMap[@(fbo)] = @(texture);
    } else {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    [rbMap removeObjectForKey:@(fbo)];
    NSMutableDictionary *levelMap = (attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColorLevel :
        ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
            ? gl4metalFramebufferDepthLevel : gl4metalFramebufferStencilLevel);
    if (texture == 0) [levelMap removeObjectForKey:@(fbo)];
    else levelMap[@(fbo)] = @(level);
    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        if (texture) {
            gl4metalFramebufferStencil[@(fbo)] = @(texture);
            gl4metalFramebufferStencilLevel[@(fbo)] = @(level);
        } else {
            [gl4metalFramebufferStencil removeObjectForKey:@(fbo)];
            [gl4metalFramebufferStencilLevel removeObjectForKey:@(fbo)];
        }
        [gl4metalFramebufferStencilRB removeObjectForKey:@(fbo)];
    }
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
    if (level < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalIsDrawFramebufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (texture == 0) {
        glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, 0, level);
        return;
    }
    GLenum textureTarget = [gl4metalTextureTargets[@(texture)] unsignedIntValue];
    if (textureTarget == GL_TEXTURE_1D) {
        glFramebufferTexture1D(target, attachment, GL_TEXTURE_1D, texture, level);
    } else if (textureTarget == GL_TEXTURE_2D) {
        glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
    } else {
        gl4metalSetError(GL_INVALID_OPERATION);
    }
}

void APIENTRY glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget,
                                     GLuint texture, GLint level) {
    if (!gl4metalIsDrawFramebufferTarget(target) || textarget != GL_TEXTURE_1D) {
        gl4metalSetError(GL_INVALID_ENUM); return;
    }
    if (level < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (texture != 0 && [gl4metalTextureTargets[@(texture)] unsignedIntValue] != GL_TEXTURE_1D) {
        gl4metalSetError(GL_INVALID_OPERATION); return;
    }
    glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
}

void APIENTRY glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget,
                                     GLuint texture, GLint level, GLint zoffset) {
    if (!gl4metalIsDrawFramebufferTarget(target) || textarget != GL_TEXTURE_3D) {
        gl4metalSetError(GL_INVALID_ENUM); return;
    }
    if (level < 0 || zoffset < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    (void)attachment; (void)texture;
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture,
                                        GLint level, GLint layer) {
    if (!gl4metalIsDrawFramebufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (level < 0 || layer < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (texture == 0) {
        glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, 0, level);
        return;
    }
    (void)attachment; (void)texture;
    gl4metalSetError(GL_INVALID_OPERATION);
}

static BOOL gl4metalValidFramebufferAttachment(GLenum attachment) {
    return attachment == GL_COLOR_ATTACHMENT0 || attachment == GL_DEPTH_ATTACHMENT ||
           attachment == GL_STENCIL_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT;
}

void APIENTRY glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment,
                                                     GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalIsDrawFramebufferTarget(target) && target != GL_READ_FRAMEBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM); return;
    }
    if (!gl4metalValidFramebufferAttachment(attachment)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    GLuint framebuffer = target == GL_READ_FRAMEBUFFER ? ctx.currentReadFramebuffer : ctx.currentFramebuffer;
    if (framebuffer == 0) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    gl4metalInitFramebufferState();
    NSMutableDictionary *textureMap = attachment == GL_COLOR_ATTACHMENT0 ? gl4metalFramebufferColor :
        (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
            ? gl4metalFramebufferDepth : gl4metalFramebufferStencil;
    NSMutableDictionary *renderbufferMap = attachment == GL_COLOR_ATTACHMENT0 ? gl4metalFramebufferColorRB :
        (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
            ? gl4metalFramebufferDepthRB : gl4metalFramebufferStencilRB;
    NSNumber *textureName = textureMap[@(framebuffer)];
    NSNumber *renderbufferName = renderbufferMap[@(framebuffer)];
    switch (pname) {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
            params[0] = textureName ? GL_TEXTURE : (renderbufferName ? GL_RENDERBUFFER : GL_NONE);
            break;
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
            params[0] = textureName ? textureName.intValue : renderbufferName.intValue;
            break;
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
            if (!textureName) { gl4metalSetError(GL_INVALID_OPERATION); return; }
            params[0] = [((attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColorLevel :
                ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
                    ? gl4metalFramebufferDepthLevel : gl4metalFramebufferStencilLevel))[@(framebuffer)] intValue];
            break;
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
            if (!textureName) { gl4metalSetError(GL_INVALID_OPERATION); return; }
            params[0] = 0;
            break;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            if (!textureName && !renderbufferName) { gl4metalSetError(GL_INVALID_OPERATION); return; }
            params[0] = GL_LINEAR;
            break;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            if (!textureName && !renderbufferName) { gl4metalSetError(GL_INVALID_OPERATION); return; }
            params[0] = GL_UNSIGNED_NORMALIZED;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glFramebufferRenderbuffer(GLenum target, GLenum attachment,
                                         GLenum renderbuffertarget, GLuint renderbuffer) {
    if (!gl4metalIsDrawFramebufferTarget(target) || renderbuffertarget != GL_RENDERBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx || ctx.currentFramebuffer == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (attachment != GL_COLOR_ATTACHMENT0 && attachment != GL_DEPTH_ATTACHMENT &&
        attachment != GL_STENCIL_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalInitFramebufferState();
    GLuint fbo = ctx.currentFramebuffer;
    NSMutableDictionary *rbMap = (attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColorRB :
                                   ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) ? gl4metalFramebufferDepthRB : gl4metalFramebufferStencilRB);
    NSMutableDictionary *texMap = (attachment == GL_COLOR_ATTACHMENT0) ? gl4metalFramebufferColor :
                                    ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) ? gl4metalFramebufferDepth : gl4metalFramebufferStencil);
    if (renderbuffer == 0) {
        [rbMap removeObjectForKey:@(fbo)];
    } else if (renderbufferObjects[@(renderbuffer)]) {
        rbMap[@(fbo)] = @(renderbuffer);
    } else {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    [texMap removeObjectForKey:@(fbo)];
    NSMutableDictionary *levelMap = attachment == GL_COLOR_ATTACHMENT0 ? gl4metalFramebufferColorLevel :
        ((attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT)
            ? gl4metalFramebufferDepthLevel : gl4metalFramebufferStencilLevel);
    [levelMap removeObjectForKey:@(fbo)];
    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        if (renderbuffer) gl4metalFramebufferStencilRB[@(fbo)] = @(renderbuffer);
        else [gl4metalFramebufferStencilRB removeObjectForKey:@(fbo)];
        [gl4metalFramebufferDepthLevel removeObjectForKey:@(fbo)];
        [gl4metalFramebufferStencilLevel removeObjectForKey:@(fbo)];
    }
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    if (n < 0 || (n > 0 && !framebuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = framebuffers[i];
        if (framebuffer == 0) continue;
        [gl4metalFramebufferGeneratedNames removeObject:@(framebuffer)];
        if (ctx.currentFramebuffer == framebuffer) gl4metalFlushPendingClear();
        [framebufferObjects removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferColor removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferDepth removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferStencil removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferColorRB removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferDepthRB removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferStencilRB removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferColorLevel removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferDepthLevel removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferStencilLevel removeObjectForKey:@(framebuffer)];
        [ctx.framebufferDrawBuffers removeObjectForKey:@(framebuffer)];
        [ctx.framebufferReadBuffers removeObjectForKey:@(framebuffer)];
        if (ctx.currentFramebuffer == framebuffer) {
            ctx.currentFramebuffer = 0;
            ctx.drawBuffer = [ctx.framebufferDrawBuffers[@0] unsignedIntValue];
            ctx.pipelineState = nil;
        }
        if (ctx.currentReadFramebuffer == framebuffer) {
            ctx.currentReadFramebuffer = 0;
            ctx.readBuffer = [ctx.framebufferReadBuffers[@0] unsignedIntValue];
        }
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) {
    if (n < 0 || (n > 0 && !renderbuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint renderbuffer = 0;
        do {
            renderbuffer = gl4metalNextRenderbufferId++;
            if (renderbuffer == 0) renderbuffer = gl4metalNextRenderbufferId++;
        } while ([gl4metalRenderbufferGeneratedNames containsObject:@(renderbuffer)] ||
                 renderbufferObjects[@(renderbuffer)] != nil);
        renderbuffers[i] = renderbuffer;
        [gl4metalRenderbufferGeneratedNames addObject:@(renderbuffer)];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    if (target != GL_RENDERBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalInitFramebufferState();
    if (renderbuffer != 0 && ![gl4metalRenderbufferGeneratedNames containsObject:@(renderbuffer)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    gl4metalCurrentRenderbuffer = renderbuffer;
    if (renderbuffer) renderbufferObjects[@(renderbuffer)] = @(renderbuffer);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorage(GLenum target, GLenum internalformat,
                                    GLsizei width, GLsizei height) {
    if (target != GL_RENDERBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 || height < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx || gl4metalCurrentRenderbuffer == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    MTLPixelFormat pf = MTLPixelFormatRGBA8Unorm;
    switch (internalformat) {
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGB565:
        case GL_RGB8:
        case GL_RGBA8:
            pf = MTLPixelFormatRGBA8Unorm;
            break;
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32F:
            pf = MTLPixelFormatDepth32Float;
            break;
        case GL_DEPTH24_STENCIL8:
            pf = MTLPixelFormatDepth32Float_Stencil8;
            break;
        case GL_STENCIL_INDEX8:
            pf = MTLPixelFormatStencil8;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    if (width == 0 || height == 0) {
        [gl4metalRenderbufferTextures removeObjectForKey:@(gl4metalCurrentRenderbuffer)];
        gl4metalRenderbufferFormats[@(gl4metalCurrentRenderbuffer)] = @(internalformat);
        renderbufferObjects[@(gl4metalCurrentRenderbuffer)] = @(gl4metalCurrentRenderbuffer);
        gl4metalSetError(GL_NO_ERROR);
        return;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                    width:(NSUInteger)width
                                                                                   height:(NSUInteger)height
                                                                                mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tex = [ctx.device newTextureWithDescriptor:td];
    if (!tex) {
        gl4metalSetError(GL_OUT_OF_MEMORY);
        return;
    }
    gl4metalInitFramebufferState();
    gl4metalRenderbufferTextures[@(gl4metalCurrentRenderbuffer)] = tex;
    gl4metalRenderbufferFormats[@(gl4metalCurrentRenderbuffer)] = @(internalformat);
    renderbufferObjects[@(gl4metalCurrentRenderbuffer)] = @(gl4metalCurrentRenderbuffer);
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorageMultisample(GLenum target, GLsizei samples,
                                                GLenum internalformat,
                                                GLsizei width, GLsizei height) {
    if (target != GL_RENDERBUFFER) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (samples < 0 || width < 0 || height < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (samples > 1) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    glRenderbufferStorage(target, internalformat, width, height);
}

void APIENTRY glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (target != GL_RENDERBUFFER) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!ctx || gl4metalCurrentRenderbuffer == 0) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    id<MTLTexture> texture = gl4metalRenderbufferTextures[@(gl4metalCurrentRenderbuffer)];
    GLenum format = [gl4metalRenderbufferFormats[@(gl4metalCurrentRenderbuffer)] unsignedIntValue];
    switch (pname) {
        case GL_RENDERBUFFER_WIDTH: params[0] = (GLint)texture.width; break;
        case GL_RENDERBUFFER_HEIGHT: params[0] = (GLint)texture.height; break;
        case GL_RENDERBUFFER_INTERNAL_FORMAT: params[0] = (GLint)format; break;
        case GL_RENDERBUFFER_RED_SIZE:
            params[0] = format == GL_RGBA4 ? 4 : (format == GL_RGB5_A1 || format == GL_RGB565) ? 5 :
                        (format == GL_RGB8 || format == GL_RGBA8) ? 8 : 0;
            break;
        case GL_RENDERBUFFER_GREEN_SIZE:
            params[0] = format == GL_RGBA4 ? 4 : format == GL_RGB5_A1 ? 5 :
                        format == GL_RGB565 ? 6 : (format == GL_RGB8 || format == GL_RGBA8) ? 8 : 0;
            break;
        case GL_RENDERBUFFER_BLUE_SIZE:
            params[0] = format == GL_RGBA4 ? 4 : (format == GL_RGB5_A1 || format == GL_RGB565) ? 5 :
                        (format == GL_RGB8 || format == GL_RGBA8) ? 8 : 0;
            break;
        case GL_RENDERBUFFER_ALPHA_SIZE:
            params[0] = format == GL_RGBA4 ? 4 : format == GL_RGB5_A1 ? 1 :
                        format == GL_RGBA8 ? 8 : 0;
            break;
        case GL_RENDERBUFFER_DEPTH_SIZE:
            params[0] = format == GL_DEPTH_COMPONENT16 ? 16 :
                        (format == GL_DEPTH_COMPONENT24 || format == GL_DEPTH24_STENCIL8) ? 24 :
                        format == GL_DEPTH_COMPONENT32F ? 32 : 0;
            break;
        case GL_RENDERBUFFER_STENCIL_SIZE:
            params[0] = format == GL_DEPTH24_STENCIL8 || format == GL_STENCIL_INDEX8 ? 8 : 0;
            break;
        case GL_RENDERBUFFER_SAMPLES: params[0] = 0; break;
        default: gl4metalSetError(GL_INVALID_ENUM); return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    if (n < 0 || (n > 0 && !renderbuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint renderbuffer = renderbuffers[i];
        if (!renderbuffer) continue;
        [gl4metalRenderbufferGeneratedNames removeObject:@(renderbuffer)];
        [renderbufferObjects removeObjectForKey:@(renderbuffer)];
        [gl4metalRenderbufferTextures removeObjectForKey:@(renderbuffer)];
        [gl4metalRenderbufferFormats removeObjectForKey:@(renderbuffer)];
        NSArray<NSNumber *> *rbKeys = [gl4metalFramebufferColorRB allKeysForObject:@(renderbuffer)];
        for (NSNumber *key in rbKeys) [gl4metalFramebufferColorRB removeObjectForKey:key];
        rbKeys = [gl4metalFramebufferDepthRB allKeysForObject:@(renderbuffer)];
        for (NSNumber *key in rbKeys) [gl4metalFramebufferDepthRB removeObjectForKey:key];
        rbKeys = [gl4metalFramebufferStencilRB allKeysForObject:@(renderbuffer)];
        for (NSNumber *key in rbKeys) [gl4metalFramebufferStencilRB removeObjectForKey:key];
        if (gl4metalCurrentRenderbuffer == renderbuffer) gl4metalCurrentRenderbuffer = 0;
    }
    if (ctx) ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsFramebuffer(GLuint framebuffer) {
    return framebuffer != 0 && framebufferObjects[@(framebuffer)] ? GL_TRUE : GL_FALSE;
}

GLboolean APIENTRY glIsRenderbuffer(GLuint renderbuffer) {
    return renderbuffer != 0 && renderbufferObjects[@(renderbuffer)] ? GL_TRUE : GL_FALSE;
}
