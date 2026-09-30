#import "gl4metal.h"
#import "gl4metal_runtime.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stddef.h>
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
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSNumber *> *> *gl4metalTextureParams = nil;
static NSMutableDictionary<NSNumber *, id<MTLSamplerState>> *gl4metalTextureSamplers = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalTextureUnitBindings = nil;
static NSMutableSet<NSNumber *> *gl4metalTextureObjectNames = nil;
static GLuint gl4metalNextTextureId = 1;
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

// Real framebuffer/renderbuffer state. GL texture names and renderbuffer names
// remain separate namespaces just like OpenGL; the Metal resources are stored
// in their corresponding dictionaries.
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferColor = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferDepth = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferStencil = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferColorRB = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferDepthRB = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalFramebufferStencilRB = nil;
static NSMutableDictionary<NSNumber *, id<MTLTexture>> *gl4metalRenderbufferTextures = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalRenderbufferFormats = nil;
static GLuint gl4metalCurrentRenderbuffer = 0;

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
            @(GL_TEXTURE_WRAP_T): @(GL_REPEAT)
        } mutableCopy];
        gl4metalTextureParams[@(texture)] = params;
    }
    return params;
}

static void gl4metalEnsureTextureState(void) {
    if (!gl4metalTextureObjects) gl4metalTextureObjects = [NSMutableDictionary dictionary];
    if (!gl4metalTextureParams) gl4metalTextureParams = [NSMutableDictionary dictionary];
    if (!gl4metalTextureSamplers) gl4metalTextureSamplers = [NSMutableDictionary dictionary];
    if (!gl4metalTextureUnitBindings) gl4metalTextureUnitBindings = [NSMutableDictionary dictionary];
    if (!gl4metalTextureObjectNames) gl4metalTextureObjectNames = [NSMutableSet set];
}

static GLuint gl4metalTextureBoundToUnit(GLuint unit) {
    gl4metalEnsureTextureState();
    NSNumber *value = gl4metalTextureUnitBindings[@(unit)];
    return value ? value.unsignedIntValue : 0;
}

static size_t gl4metalBytesPerPixelForFormatType(GLenum format, GLenum type) {
    if (type != GL_UNSIGNED_BYTE) return 0;
    switch (format) {
        case GL_RGBA:
        case GL_BGRA:
            return 4;
        case GL_RGB:
            return 3;
        case GL_RED:
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

    size_t pixelCount = (size_t)width * (size_t)height;
    size_t sourceRowPixels = gl4metalUnpackRowLength > 0
        ? (size_t)gl4metalUnpackRowLength : (size_t)width;
    size_t unalignedRowBytes = sourceRowPixels * sourceBpp;
    size_t alignment = (size_t)gl4metalUnpackAlignment;
    size_t sourceRowBytes = (unalignedRowBytes + alignment - 1) & ~(alignment - 1);
    size_t sourceStart = (size_t)gl4metalUnpackSkipRows * sourceRowBytes +
                         (size_t)gl4metalUnpackSkipPixels * sourceBpp;
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
            } else if (format == GL_RED && sourceBpp == 1) {
                q[0] = p[0]; q[1] = p[0]; q[2] = p[0]; q[3] = 255;
            } else if (format == GL_LUMINANCE && sourceBpp == 1) {
                q[0] = p[0]; q[1] = p[0]; q[2] = p[0]; q[3] = 255;
            } else if (format == GL_LUMINANCE_ALPHA && sourceBpp == 2) {
                q[0] = p[0]; q[1] = p[0]; q[2] = p[0]; q[3] = p[1];
            } else {
                return NO;
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

static NSUInteger gl4metalMipLevelCount(GLsizei width, GLsizei height) {
    NSUInteger largest = (NSUInteger)MAX(width, height);
    NSUInteger levels = 1;
    while (largest > 1) {
        largest >>= 1;
        ++levels;
    }
    return levels;
}

static id<MTLSamplerState> gl4metalCreateSamplerForTexture(GLuint texture) {
    if (!ctx || !ctx.device || texture == 0) return nil;
    gl4metalEnsureTextureState();

    NSNumber *key = @(texture);
    id<MTLSamplerState> cached = gl4metalTextureSamplers[key];
    if (cached) return cached;

    NSDictionary<NSNumber *, NSNumber *> *params = gl4metalEnsureTextureParams(texture);
    NSInteger minFilter = [params[@(GL_TEXTURE_MIN_FILTER)] integerValue];
    NSInteger magFilter = [params[@(GL_TEXTURE_MAG_FILTER)] integerValue];
    NSInteger wrapS = [params[@(GL_TEXTURE_WRAP_S)] integerValue];
    NSInteger wrapT = [params[@(GL_TEXTURE_WRAP_T)] integerValue];

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

    switch (wrapS) {
        case GL_CLAMP_TO_EDGE:
            sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
            break;
        case GL_MIRRORED_REPEAT:
            sd.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
            break;
        case GL_CLAMP_TO_BORDER:
            sd.sAddressMode = MTLSamplerAddressModeClampToBorderColor;
            sd.borderColor = MTLSamplerBorderColorTransparentBlack;
            break;
        case GL_REPEAT:
        default:
            sd.sAddressMode = MTLSamplerAddressModeRepeat;
            break;
    }
    switch (wrapT) {
        case GL_CLAMP_TO_EDGE:
            sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            break;
        case GL_MIRRORED_REPEAT:
            sd.tAddressMode = MTLSamplerAddressModeMirrorRepeat;
            break;
        case GL_CLAMP_TO_BORDER:
            sd.tAddressMode = MTLSamplerAddressModeClampToBorderColor;
            sd.borderColor = MTLSamplerBorderColorTransparentBlack;
            break;
        case GL_REPEAT:
        default:
            sd.tAddressMode = MTLSamplerAddressModeRepeat;
            break;
    }

    id<MTLSamplerState> sampler = [ctx.device newSamplerStateWithDescriptor:sd];
    if (sampler) gl4metalTextureSamplers[key] = sampler;
    return sampler;
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
    GLuint texture = gl4metalTextureBoundToUnit(unit);
    if (!texture) return nil;
    return gl4metalCreateSamplerForTexture(texture);
}

GLuint gl4metalGetTextureNameForUnit(GLuint unit) {
    if (unit >= GL4METAL_MAX_TEXTURE_UNITS) return 0;
    return gl4metalTextureBoundToUnit(unit);
}

void gl4metalInitTextureState(void) {
    gl4metalEnsureTextureState();
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
        GLuint texture = gl4metalNextTextureId++;
        if (texture == 0) texture = gl4metalNextTextureId++;
        textures[i] = texture;
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
        [gl4metalTextureObjects removeObjectForKey:@(texture)];
        [gl4metalTextureParams removeObjectForKey:@(texture)];
        [gl4metalTextureSamplers removeObjectForKey:@(texture)];
        [gl4metalTextureObjectNames removeObject:@(texture)];

        NSArray<NSNumber *> *units = [gl4metalTextureUnitBindings allKeysForObject:@(texture)];
        for (NSNumber *unit in units) {
            [gl4metalTextureUnitBindings removeObjectForKey:unit];
        }
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindTexture(GLenum target, GLuint texture) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalEnsureTextureState();

    if (texture != 0 && !gl4metalTextureParams[@(texture)]) {
        gl4metalEnsureTextureParams(texture);
    }

    if (texture != 0) [gl4metalTextureObjectNames addObject:@(texture)];

    gl4metalTextureUnitBindings[@(gl4metalActiveTextureUnit)] = @(texture);
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

void APIENTRY glTexParameteri(GLenum target, GLenum pname, GLint param) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    if (!texture) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
            if (param != GL_NEAREST && param != GL_LINEAR &&
                param != GL_NEAREST_MIPMAP_NEAREST && param != GL_LINEAR_MIPMAP_NEAREST &&
                param != GL_NEAREST_MIPMAP_LINEAR && param != GL_LINEAR_MIPMAP_LINEAR) {
                gl4metalSetError(GL_INVALID_ENUM);
                return;
            }
            break;
        case GL_TEXTURE_MAG_FILTER:
            if (pname == GL_TEXTURE_MAG_FILTER && param != GL_NEAREST && param != GL_LINEAR) {
                gl4metalSetError(GL_INVALID_ENUM);
                return;
            }
            break;
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
            if (param != GL_REPEAT && param != GL_CLAMP_TO_EDGE &&
                param != GL_CLAMP_TO_BORDER && param != GL_MIRRORED_REPEAT) {
                gl4metalSetError(GL_INVALID_ENUM);
                return;
            }
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
            gl4metalEnsureTextureParams(texture)[@(pname)] = @(param);
            [gl4metalTextureSamplers removeObjectForKey:@(texture)];
            gl4metalSetError(GL_NO_ERROR);
            return;
    }
}

void APIENTRY glTexParameteriv(GLenum target, GLenum pname, const GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glTexParameteri(target, pname, params[0]);
}

void APIENTRY glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
    glTexParameteri(target, pname, (GLint)param);
}

void APIENTRY glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glTexParameteri(target, pname, (GLint)params[0]);
}

static NSNumber *gl4metalGetBoundTextureParameter(GLenum target, GLenum pname) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return nil;
    }
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return nil;
    }

    gl4metalEnsureTextureState();
    GLuint texture = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    if (texture == 0 || ![gl4metalTextureObjectNames containsObject:@(texture)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return nil;
    }
    NSNumber *value = gl4metalEnsureTextureParams(texture)[@(pname)];
    gl4metalSetError(GL_NO_ERROR);
    return value;
}

void APIENTRY glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    NSNumber *value = gl4metalGetBoundTextureParameter(target, pname);
    if (value) params[0] = value.intValue;
}

void APIENTRY glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    NSNumber *value = gl4metalGetBoundTextureParameter(target, pname);
    if (value) params[0] = value.floatValue;
}

void APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLsizei height, GLint border,
                           GLenum format, GLenum type, const void *pixels) {
    (void)internalformat;
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || width <= 0 || height <= 0 || border != 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!gl4metalBytesPerPixelForFormatType(format, type)) {
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
        td.usage = isDepth ? MTLTextureUsageRenderTarget : (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
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

    if (!gl4metalUploadPixelsToTexture(texture, level, 0, 0, width, height, format, type, pixels)) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    // Keep sampling changes deterministic when a texture is re-uploaded.
    [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
    printf("[gl4metal][TEXTURE] glTexImage2D GLtex=%u level=%d size=%dx%d format=0x%x type=0x%x pixels=%s\n",
           textureName, level, width, height, (unsigned)format, (unsigned)type,
           pixels ? "yes" : "null");
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                              GLint yoffset, GLsizei width, GLsizei height,
                              GLenum format, GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || width <= 0 || height <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!gl4metalBytesPerPixelForFormatType(format, type)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!pixels) {
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

    if (!gl4metalUploadPixelsToTexture(texture, level, xoffset, yoffset, width, height, format, type, pixels)) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    [gl4metalTextureSamplers removeObjectForKey:@(textureName)];
    printf("[gl4metal][TEXTURE] glTexSubImage2D GLtex=%u level=%d rect=%d,%d %dx%d format=0x%x type=0x%x\n",
           textureName, level, xoffset, yoffset, width, height,
           (unsigned)format, (unsigned)type);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenerateMipmap(GLenum target) {
    if (target != GL_TEXTURE_2D) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx || !ctx.commandQueue) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    gl4metalEnsureTextureState();
    GLuint textureName = gl4metalTextureBoundToUnit(gl4metalActiveTextureUnit);
    id<MTLTexture> texture = textureName ? gl4metalTextureObjects[@(textureName)] : nil;
    if (!texture) {
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
    if (!gl4metalRenderbufferTextures) gl4metalRenderbufferTextures = [NSMutableDictionary dictionary];
    if (!gl4metalRenderbufferFormats) gl4metalRenderbufferFormats = [NSMutableDictionary dictionary];
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
    for (NSNumber *key in keys) { [gl4metalFramebufferColor removeObjectForKey:key]; }
    keys = [gl4metalFramebufferDepth allKeysForObject:@(texture)];
    for (NSNumber *key in keys) { [gl4metalFramebufferDepth removeObjectForKey:key]; }
    keys = [gl4metalFramebufferStencil allKeysForObject:@(texture)];
    for (NSNumber *key in keys) { [gl4metalFramebufferStencil removeObjectForKey:key]; }
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
    return tex ? tex.pixelFormat : MTLPixelFormatInvalid;
}

void APIENTRY glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    if (n < 0 || (n > 0 && !framebuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = ++nextBufferId;
        framebuffers[i] = framebuffer;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer) {
    if (!gl4metalIsDrawFramebufferTarget(target) && target != GL_READ_FRAMEBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalInitFramebufferState();
    if (framebuffer != 0 && !framebufferObjects[@(framebuffer)]) {
        framebufferObjects[@(framebuffer)] = @(framebuffer);
    }
    if (gl4metalIsDrawFramebufferTarget(target)) {
        ctx.currentFramebuffer = framebuffer;
        // Render target format and attachment set are part of the pipeline state.
        ctx.pipelineState = nil;
    }
    if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) {
        ctx.currentReadFramebuffer = framebuffer;
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
        gl4metalSetError(GL_NO_ERROR);
        return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }

    NSArray<id<MTLTexture>> *attachments = @[color ?: (id<MTLTexture>)[NSNull null],
                                              depth ?: (id<MTLTexture>)[NSNull null],
                                              ds ?: (id<MTLTexture>)[NSNull null],
                                              stencil ?: (id<MTLTexture>)[NSNull null]];
    for (id obj in attachments) {
        if (obj == (id)[NSNull null]) continue;
        id<MTLTexture> tex = (id<MTLTexture>)obj;
        if (tex.width != first.width || tex.height != first.height) {
            gl4metalSetError(GL_NO_ERROR);
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
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
    } else if (gl4metalTextureObjects[@(texture)]) {
        texMap[@(fbo)] = @(texture);
    } else {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    [rbMap removeObjectForKey:@(fbo)];
    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        if (texture) {
            gl4metalFramebufferStencil[@(fbo)] = @(texture);
        } else {
            [gl4metalFramebufferStencil removeObjectForKey:@(fbo)];
        }
        [gl4metalFramebufferStencilRB removeObjectForKey:@(fbo)];
    }
    ctx.pipelineState = nil;
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
    } else if (gl4metalRenderbufferTextures[@(renderbuffer)]) {
        rbMap[@(fbo)] = @(renderbuffer);
    } else {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    [texMap removeObjectForKey:@(fbo)];
    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        if (renderbuffer) gl4metalFramebufferStencilRB[@(fbo)] = @(renderbuffer);
        else [gl4metalFramebufferStencilRB removeObjectForKey:@(fbo)];
    }
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    if (n < 0 || (n > 0 && !framebuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = framebuffers[i];
        if (framebuffer == 0) continue;
        [framebufferObjects removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferColor removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferDepth removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferStencil removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferColorRB removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferDepthRB removeObjectForKey:@(framebuffer)];
        [gl4metalFramebufferStencilRB removeObjectForKey:@(framebuffer)];
        if (ctx.currentFramebuffer == framebuffer) {
            ctx.currentFramebuffer = 0;
            ctx.pipelineState = nil;
        }
        if (ctx.currentReadFramebuffer == framebuffer) ctx.currentReadFramebuffer = 0;
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
        GLuint renderbuffer = ++nextBufferId;
        renderbuffers[i] = renderbuffer;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    if (target != GL_RENDERBUFFER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalInitFramebufferState();
    gl4metalCurrentRenderbuffer = renderbuffer;
    if (renderbuffer) renderbufferObjects[@(renderbuffer)] = @(renderbuffer);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorage(GLenum target, GLenum internalformat,
                                    GLsizei width, GLsizei height) {
    if (target != GL_RENDERBUFFER || width <= 0 || height <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx || gl4metalCurrentRenderbuffer == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    MTLPixelFormat pf = MTLPixelFormatRGBA8Unorm;
    switch (internalformat) {
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
            break;
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

void APIENTRY glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    if (n < 0 || (n > 0 && !renderbuffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalInitFramebufferState();
    for (GLsizei i = 0; i < n; i++) {
        GLuint renderbuffer = renderbuffers[i];
        if (!renderbuffer) continue;
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
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsFramebuffer(GLuint framebuffer) {
    return framebuffer != 0 && framebufferObjects[@(framebuffer)] ? GL_TRUE : GL_FALSE;
}

GLboolean APIENTRY glIsRenderbuffer(GLuint renderbuffer) {
    return renderbuffer != 0 && renderbufferObjects[@(renderbuffer)] ? GL_TRUE : GL_FALSE;
}
