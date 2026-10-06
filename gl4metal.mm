//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import "gl4metal.h"
#import "gl4metal_runtime.h"
#import "gl4metal_shaders.h"

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#ifndef GL_CLAMP_VERTEX_COLOR
#define GL_CLAMP_VERTEX_COLOR 0x891A
#endif
#ifndef GL_CLAMP_FRAGMENT_COLOR
#define GL_CLAMP_FRAGMENT_COLOR 0x891B
#endif
#ifndef GL_POINT_SIZE_MIN
#define GL_POINT_SIZE_MIN 0x8126
#endif
#ifndef GL_POINT_SIZE_MAX
#define GL_POINT_SIZE_MAX 0x8127
#endif
#ifndef GL_POINT_FADE_THRESHOLD_SIZE
#define GL_POINT_FADE_THRESHOLD_SIZE 0x8128
#endif
#ifndef GL_POINT_DISTANCE_ATTENUATION
#define GL_POINT_DISTANCE_ATTENUATION 0x8129
#endif

// Track the actual MTLBuffer versions captured by each command buffer. A GL
// buffer can be copy-on-write replaced while an older version remains in use;
// tracking only its GL name would make every subsequent update clone again.
static NSMutableDictionary<NSValue *, NSNumber *> *gl4metalResourceInFlightCounts = nil;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *gl4metalCurrentCommandBufferResources = nil;
static NSMutableDictionary<NSString *, NSData *> *gl4metalIndexStatsCache = nil;
static NSUInteger gl4metalConvertedIndexBufferCacheBytes = 0;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *gl4metalTriangleFanBufferCache = nil;
static NSUInteger gl4metalTriangleFanBufferCacheBytes = 0;

static id<MTLCommandBuffer> gl4metalCreateCommandBuffer(void);

static void gl4metalInvalidateIndexStatsCache(void) {
    [gl4metalIndexStatsCache removeAllObjects];
}

static void gl4metalClearIndexCaches(void) {
    [convertedIndexBuffers removeAllObjects];
    gl4metalConvertedIndexBufferCacheBytes = 0;
    [gl4metalTriangleFanBufferCache removeAllObjects];
    gl4metalTriangleFanBufferCacheBytes = 0;
    gl4metalInvalidateIndexStatsCache();
}

static BOOL gl4metalSafeResourceUpdatesEnabled(void) {
    const char *v = getenv("GL4METAL_SAFE_RESOURCE_UPDATES");
    if (!v || !*v) return YES;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0);
}

static BOOL gl4metalBufferIsInFlight(GLuint bufferID) {
    id<MTLBuffer> buffer = bufferObjects[@(bufferID)];
    if (!buffer || !gl4metalResourceInFlightCounts) return NO;
    NSValue *resourceKey = [NSValue valueWithPointer:(__bridge const void *)buffer];
    @synchronized(gl4metalResourceInFlightCounts) {
        return [gl4metalResourceInFlightCounts[resourceKey] unsignedIntegerValue] > 0;
    }
}

static void gl4metalMarkBufferUsed(GLuint bufferID) {
    id<MTLBuffer> buffer = bufferObjects[@(bufferID)];
    if (!bufferID || !buffer || !gl4metalCurrentCommandBufferResources ||
        !gl4metalResourceInFlightCounts) return;
    NSValue *resourceKey = [NSValue valueWithPointer:(__bridge const void *)buffer];
    if (gl4metalCurrentCommandBufferResources[resourceKey]) return;
    gl4metalCurrentCommandBufferResources[resourceKey] = buffer;
    @synchronized(gl4metalResourceInFlightCounts) {
        NSUInteger count = [gl4metalResourceInFlightCounts[resourceKey] unsignedIntegerValue];
        gl4metalResourceInFlightCounts[resourceKey] = @(count + 1);
    }
}

static id<MTLBuffer> gl4metalCopyBufferForWrite(GLuint bufferID, id<MTLBuffer> oldBuffer,
                                                  GLintptr patchOffset, GLsizeiptr patchSize,
                                                  const void *patchData) {
    if (!ctx || !ctx.device || !oldBuffer || !patchData) return nil;
    if (patchOffset < 0 || patchSize < 0 ||
        (size_t)patchOffset > (size_t)[oldBuffer length] ||
        (size_t)patchSize > (size_t)[oldBuffer length] - (size_t)patchOffset) {
        return nil;
    }

    NSUInteger length = [oldBuffer length];
    id<MTLBuffer> newBuffer = [ctx.device newBufferWithLength:length
                                                        options:MTLResourceStorageModeShared];
    if (!newBuffer) return nil;

    memcpy(newBuffer.contents, oldBuffer.contents, length);
    BOOL patchAlreadyMatchesCopy = patchOffset == 0 && (NSUInteger)patchSize == length &&
        patchData == oldBuffer.contents;
    if (!patchAlreadyMatchesCopy)
        memcpy((uint8_t *)newBuffer.contents + patchOffset, patchData, (size_t)patchSize);

    if (getenv("GL4METAL_RESOURCE_UPDATE_TRACE")) {
        printf("[gl4metal][RESOURCE_UPDATE] COW buffer=%u length=%lu patch=%ld+%ld\n",
               bufferID, (unsigned long)length, (long)patchOffset, (long)patchSize);
    }
    return newBuffer;
}

gl4metalVertexArray *gl4metalGetCurrentVAO(void);

// V8 uniform/resource binding forward declarations.
// uniformLocationNames is defined in gl4metal_runtime.mm and declared extern by gl4metal_runtime.h.
static NSMutableDictionary<NSNumber *, NSNumber *> *uniformLocationPrograms = nil;
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSString *, NSNumber *> *> *programFragDataBindings = nil;
static NSMutableDictionary<NSNumber *, NSMutableDictionary<NSString *, NSNumber *> *> *programFragDataIndices = nil;
static NSMutableSet<NSNumber *> *gl4metalPendingProgramDeletes = nil;
static NSMutableSet<NSNumber *> *gl4metalPendingShaderDeletes = nil;
static NSMutableSet<NSNumber *> *bufferObjectNames = nil;
static NSMutableSet<NSNumber *> *bufferGeneratedNames = nil;
static NSMutableSet<NSNumber *> *vertexArrayGeneratedNames = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *mappedBufferPointers = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *mappedBufferOffsets = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *mappedBufferLengths = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *mappedBufferAccesses = nil;
static NSMutableDictionary<NSValue *, id<MTLCommandBuffer>> *gl4metalSyncCommands = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *bufferUsages = nil;

static void gl4metalDeleteShaderStorage(GLuint shader) {
    [shaderTypes removeObjectForKey:@(shader)];
    shaderSources.erase(shader);
    [shaderCompileStatus removeObjectForKey:@(shader)];
    [shaderInfoLog removeObjectForKey:@(shader)];
    [gl4metalPendingShaderDeletes removeObject:@(shader)];
}

static void gl4metalDeleteShaderIfUnattached(GLuint shader) {
    for (NSNumber *programId in programShaders) {
        if ([programShaders[programId] containsObject:@(shader)]) return;
    }
    gl4metalDeleteShaderStorage(shader);
}

static void gl4metalDestroyProgramStorage(GLuint program) {
    NSNumber *programKey = @(program);
    NSArray<NSNumber *> *attachedShaders = [programShaders[programKey] copy];

    NSString *cachePrefix = [NSString stringWithFormat:@"p%u:", program];
    for (NSString *cacheKey in [pipelineLayoutCache.allKeys copy]) {
        if ([cacheKey hasPrefix:cachePrefix]) [pipelineLayoutCache removeObjectForKey:cacheKey];
    }
    [ctx.programPipelines removeObjectForKey:programKey];
    [ctx.programResourceInfo removeObjectForKey:programKey];
    [programAttribBindings removeObjectForKey:programKey];
    [programFragDataBindings removeObjectForKey:programKey];
    [programFragDataIndices removeObjectForKey:programKey];

    for (NSNumber *location in [uniformLocationPrograms.allKeys copy]) {
        if ([uniformLocationPrograms[location] unsignedIntValue] != program) continue;
        [uniformLocationPrograms removeObjectForKey:location];
        [uniformLocationNames removeObjectForKey:location];
        [uniform1fValues removeObjectForKey:location];
        [uniform2fValues removeObjectForKey:location];
        [uniform3fValues removeObjectForKey:location];
        [uniform4fValues removeObjectForKey:location];
        [uniform1iValues removeObjectForKey:location];
        [uniform2iValues removeObjectForKey:location];
        [uniform3iValues removeObjectForKey:location];
        [uniform4iValues removeObjectForKey:location];
        [uniformMatrix4fvValues removeObjectForKey:location];
        [uniformRawValues removeObjectForKey:location];
    }

    [programShaders removeObjectForKey:programKey];
    [programLinkStatus removeObjectForKey:programKey];
    [programInfoLog removeObjectForKey:programKey];
    [gl4metalPendingProgramDeletes removeObject:programKey];
    for (NSNumber *shader in attachedShaders) {
        if ([gl4metalPendingShaderDeletes containsObject:shader])
            gl4metalDeleteShaderIfUnattached(shader.unsignedIntValue);
    }
}

static void gl4metalFinalizePendingProgramDelete(GLuint program) {
    if (program != 0 && [gl4metalPendingProgramDeletes containsObject:@(program)])
        gl4metalDestroyProgramStorage(program);
}

static GLenum gl4metalLogicOperation = GL_COPY;
static GLboolean gl4metalColorLogicOperationEnabled = GL_FALSE;
static GLenum gl4metalClampVertexColor = GL_FIXED_ONLY;
static GLenum gl4metalClampFragmentColor = GL_FIXED_ONLY;
static GLenum gl4metalClampReadColor = GL_TRUE;
static GLfloat gl4metalPointSizeMin = 0.0f;
static GLfloat gl4metalPointSizeMax = 1.0f;
static GLfloat gl4metalPointFadeThreshold = 1.0f;
static GLfloat gl4metalPointDistanceAttenuation[3] = {1.0f, 0.0f, 0.0f};
static GLenum gl4metalPointSpriteCoordOrigin = GL_UPPER_LEFT;
static void gl4metalBindProgramResources(id<MTLRenderCommandEncoder> encoder, GLuint program);
static MTLBlendFactor gl4metalBlendFactor(GLenum factor);
static MTLBlendOperation gl4metalBlendOperation(GLenum equation);
static NSData *gl4metalUniformDataForName(NSString *name, GLuint program, NSUInteger length);
static GLint gl4metalFindUniformLocationByName(NSString *name, GLuint program);
static void gl4metalEncodePendingClear(void);
static NSArray<NSDictionary<NSString *, id> *> *gl4metalProgramVariables(GLuint program, BOOL uniforms);
static NSArray<NSDictionary<NSString *, id> *> *gl4metalProgramFragmentOutputs(GLuint program);

static GLintptr gl4metalResolvePointerOffset(const void *pointer, id<MTLBuffer> buffer) {
    if (pointer == NULL || !buffer) return 0;

    uintptr_t raw = (uintptr_t)pointer;
    size_t bufferLength = (size_t)[buffer length];
    if (raw > bufferLength) {
        printf("[gl4metal] WARNING: pointer offset 0x%zx exceeds bound buffer length 0x%zx; clamping to 0\n", (size_t)raw, bufferLength);
        return 0;
    }
    return (GLintptr)raw;
}

// Uniform locations are program-local in OpenGL. The previous implementation
// used name-only locations, which caused ProjMat/ModelViewMat/Sampler0 values
// from one program to overwrite the state of another program. Keep the public
// location an opaque integer, but derive it from both program and name.
static GLint gl4metalResolveLocationFromName(GLuint program, const char *name) {
    if (name == NULL || program == 0) return -1;

    const uint8_t *bytes = (const uint8_t *)name;
    uint32_t hash = 2166136261u; // FNV-1a
    hash ^= (uint8_t)(program & 0xffu);       hash *= 16777619u;
    hash ^= (uint8_t)((program >> 8) & 0xffu);  hash *= 16777619u;
    hash ^= (uint8_t)((program >> 16) & 0xffu); hash *= 16777619u;
    hash ^= (uint8_t)((program >> 24) & 0xffu); hash *= 16777619u;
    while (*bytes) {
        hash ^= *bytes++;
        hash *= 16777619u;
    }

    GLint location = (GLint)(hash & 0x7fffffffU);
    if (location <= 0) location = 1;
    return location;
}

void gl4metalSetError(GLenum error) {
    // OpenGL keeps one pending flag per error code. Repeated instances of an
    // already-pending error collapse, while a distinct error remains queued.
    if (error == GL_NO_ERROR) return;
    if (!gl4metalPendingErrors) gl4metalPendingErrors = [NSMutableArray array];
    NSNumber *key = @(error);
    if (![gl4metalPendingErrors containsObject:key]) [gl4metalPendingErrors addObject:key];
    lastGL4MetalError = gl4metalPendingErrors.firstObject.unsignedIntValue;
}

static void gl4metalTraceUniformUpdate(GLint location, const char *kind, const float *values, NSUInteger count) {
    if (!getenv("GL4METAL_UNIFORM_UPDATE_TRACE") || !uniformLocationNames || !uniformLocationPrograms) return;
    NSString *name = uniformLocationNames[@(location)];
    NSNumber *owner = uniformLocationPrograms[@(location)];
    if (!name || !owner) return;
    GLuint program = owner.unsignedIntValue;
    if (!(program == 7 || program == 16 || program == 17 || program == 53)) return;
    printf("[gl4metal][UNIFORM_UPDATE] program=%u name=%s kind=%s", program, name.UTF8String, kind);
    NSUInteger n = MIN(count, (NSUInteger)4);
    for (NSUInteger i = 0; i < n; ++i) printf(" %.6g", values[i]);
    printf("\n");
}

static void gl4metalTraceUniformUpdateInt(GLint location, const char *kind, const GLint *values, NSUInteger count) {
    if (!getenv("GL4METAL_UNIFORM_UPDATE_TRACE") || !uniformLocationNames || !uniformLocationPrograms) return;
    NSString *name = uniformLocationNames[@(location)];
    NSNumber *owner = uniformLocationPrograms[@(location)];
    if (!name || !owner) return;
    GLuint program = owner.unsignedIntValue;
    if (!(program == 7 || program == 16 || program == 17 || program == 53)) return;
    printf("[gl4metal][UNIFORM_UPDATE] program=%u name=%s kind=%s", program, name.UTF8String, kind);
    NSUInteger n = MIN(count, (NSUInteger)4);
    for (NSUInteger i = 0; i < n; ++i) printf(" %d", values[i]);
    printf("\n");
}

static BOOL gl4metalResourceTraceEnabled(void) {
    return getenv("GL4METAL_RESOURCE_TRACE") || getenv("GL4METAL_UNIFORM_UPDATE_TRACE");
}

static uint64_t gl4metalResourceTraceHashMix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
    h ^= (h >> 29);
    return h;
}

static void gl4metalTraceResourceSnapshot(GLuint program) {
    if (!gl4metalResourceTraceEnabled() || !ctx || program == 0) return;
    if (!(program == 1 || program == 7 || program == 16 || program == 17 || program == 53)) return;
    NSString *info = ctx.programResourceInfo[@(program)];
    if (!info) return;

    uint64_t hash = 1469598103934665603ULL;
    for (GLuint unit = 0; unit < 4; ++unit) {
        GLuint texName = gl4metalGetTextureNameForUnit(unit);
        hash = gl4metalResourceTraceHashMix(hash, ((uint64_t)unit << 32) | texName);
    }

    NSArray<NSString *> *lines = [info componentsSeparatedByString:@"\n"];
    for (NSString *line in lines) {
        if (line.length == 0 || [line hasPrefix:@"vertexFunction="]) continue;
        const char *raw = line.UTF8String;
        char stage[16] = {0}, name[256] = {0}, type[32] = {0};
        unsigned long index = 0, dataSize = 0;
        if (sscanf(raw, "%15s name=%255s type=%31s index=%lu", stage, name, type, &index) != 4) continue;
        NSString *resourceName = [NSString stringWithUTF8String:name];
        GLint location = gl4metalFindUniformLocationByName(resourceName, program);
        if (strcmp(type, "texture") == 0 || strcmp(type, "sampler") == 0) {
            NSString *lookupName = resourceName;
            if (strcmp(type, "sampler") == 0 && [lookupName hasSuffix:@"Smplr"]) {
                lookupName = [lookupName substringToIndex:lookupName.length - 5];
            }
            GLint unit = 0;
            NSNumber *unitValue = uniform1iValues[@(gl4metalFindUniformLocationByName(lookupName, program))];
            if (unitValue) unit = unitValue.intValue;
            if (unit < 0) unit = 0;
            GLuint texName = gl4metalGetTextureNameForUnit((GLuint)unit);
            hash = gl4metalResourceTraceHashMix(hash, ((uint64_t)(uint32_t)location << 32) | (uint32_t)unit);
            hash = gl4metalResourceTraceHashMix(hash, texName);
        } else if (strcmp(type, "buffer") == 0 &&
                   ([resourceName isEqualToString:@"ModelViewMat"] ||
                    [resourceName isEqualToString:@"ProjMat"] ||
                    [resourceName isEqualToString:@"IViewRotMat"] ||
                    [resourceName isEqualToString:@"ColorModulator"] ||
                    [resourceName isEqualToString:@"FogStart"] ||
                    [resourceName isEqualToString:@"FogEnd"] ||
                    [resourceName isEqualToString:@"FogColor"] ||
                    [resourceName isEqualToString:@"FogShape"])) {
            const char *sizeMarker = strstr(raw, "dataSize=");
            if (sizeMarker) dataSize = strtoul(sizeMarker + 9, NULL, 10);
            if (dataSize == 0) dataSize = 16;
            NSData *data = gl4metalUniformDataForName(resourceName, program, (NSUInteger)dataSize);
            hash = gl4metalResourceTraceHashMix(hash, ((uint64_t)(uint32_t)location << 32) | data.length);
            if (data.length) {
                const uint8_t *b = (const uint8_t *)data.bytes;
                NSUInteger n = MIN(data.length, (NSUInteger)16);
                for (NSUInteger i = 0; i < n; ++i) hash = gl4metalResourceTraceHashMix(hash, b[i]);
            }
        }
    }

    static uint64_t lastHash[54] = {0};
    static uint32_t changeCount[54] = {0};
    if (program >= 54) return;
    if (lastHash[program] == hash) return;
    lastHash[program] = hash;
    if (changeCount[program]++ >= 32) return;

    printf("[gl4metal][RESOURCE_SNAPSHOT] program=%u\n", program);
    printf("[gl4metal][RESOURCE_SNAPSHOT] textures:");
    for (GLuint unit = 0; unit < 4; ++unit) {
        GLuint texName = gl4metalGetTextureNameForUnit(unit);
        id<MTLTexture> tex = gl4metalGetTextureForUnit(unit);
        if (tex) {
            printf(" u%u=GLtex%u(%lux%lu)", unit, texName, (unsigned long)tex.width, (unsigned long)tex.height);
        } else {
            printf(" u%u=GLtex%u(nil)", unit, texName);
        }
    }
    printf("\n");

    for (NSString *line in lines) {
        if (line.length == 0 || [line hasPrefix:@"vertexFunction="]) continue;
        const char *raw = line.UTF8String;
        char stage[16] = {0}, name[256] = {0}, type[32] = {0};
        unsigned long index = 0, dataSize = 0;
        if (sscanf(raw, "%15s name=%255s type=%31s index=%lu", stage, name, type, &index) != 4) continue;
        NSString *resourceName = [NSString stringWithUTF8String:name];
        if (strncmp(name, "vertexBuffer.", 13) == 0) continue;

        if (strcmp(type, "texture") == 0 || strcmp(type, "sampler") == 0) {
            NSString *lookupName = resourceName;
            if (strcmp(type, "sampler") == 0 && [lookupName hasSuffix:@"Smplr"]) {
                lookupName = [lookupName substringToIndex:lookupName.length - 5];
            }
            GLint location = gl4metalFindUniformLocationByName(lookupName, program);
            NSNumber *unitValue = uniform1iValues[@(location)];
            GLint unit = unitValue ? unitValue.intValue : 0;
            if (unit < 0) unit = 0;
            GLuint texName = gl4metalGetTextureNameForUnit((GLuint)unit);
            printf("[gl4metal][RESOURCE_SNAPSHOT] %s %s name=%s index=%lu loc=%d unit=%d GLtex=%u\n",
                   stage, type, name, index, location, unit, texName);
            continue;
        }

        if (strcmp(type, "buffer") != 0) continue;
        if (strstr(name, "vertexBuffer.") == name) continue;
        const char *sizeMarker = strstr(raw, "dataSize=");
        if (sizeMarker) dataSize = strtoul(sizeMarker + 9, NULL, 10);
        if (dataSize == 0) dataSize = 16;
        NSData *data = gl4metalUniformDataForName(resourceName, program, (NSUInteger)dataSize);
        printf("[gl4metal][RESOURCE_SNAPSHOT] %s buffer name=%s index=%lu bytes=%lu",
               stage, name, index, (unsigned long)data.length);
        if ([resourceName isEqualToString:@"ModelViewMat"] || [resourceName isEqualToString:@"ProjMat"]) {
            const float *v = (const float *)data.bytes;
            if (data.length >= 64) {
                printf(" diag=[%.6g %.6g %.6g %.6g] pos=[%.6g %.6g %.6g]",
                       v[0], v[5], v[10], v[15], v[12], v[13], v[14]);
            } else {
                printf(" first=[%.6g %.6g %.6g %.6g]", v[0], v[1], v[2], v[3]);
            }
        } else if ([resourceName isEqualToString:@"IViewRotMat"]) {
            const float *v = (const float *)data.bytes;
            if (data.length >= 48) printf(" col0=[%.6g %.6g %.6g] col1=[%.6g %.6g %.6g] col2=[%.6g %.6g %.6g]", v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
        } else if ([resourceName isEqualToString:@"ColorModulator"] || [resourceName isEqualToString:@"FogColor"]) {
            const float *v = (const float *)data.bytes;
            if (data.length >= 16) printf(" value=[%.6g %.6g %.6g %.6g]", v[0], v[1], v[2], v[3]);
        } else if ([resourceName isEqualToString:@"FogStart"] || [resourceName isEqualToString:@"FogEnd"] || [resourceName isEqualToString:@"FogShape"]) {
            if (data.length >= 4) {
                const uint32_t *u = (const uint32_t *)data.bytes;
                const float *f = (const float *)data.bytes;
                if ([resourceName isEqualToString:@"FogShape"]) printf(" value=%d", (int)u[0]);
                else printf(" value=%.6g", f[0]);
            }
        }
        printf("\n");
    }
}

static void gl4metalUpdateFallbackMVP(const GLfloat *value, GLsizei count) {
    if (!ctx || !value || count <= 0) return;

    size_t copyCount = MIN((size_t)count, 16u);
    memcpy(cachedFallbackMVP, value, copyCount * sizeof(GLfloat));
    if (copyCount < 16) {
        memset(cachedFallbackMVP + copyCount, 0, (16 - copyCount) * sizeof(GLfloat));
    }
    cachedFallbackMVPIsValid = YES;
}

static NSMutableSet<NSNumber *> *inferredConstantVertexAttributes = nil;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *constantVertexAttributeBuffers = nil;

static void gl4metalPackCurrentVertexAttribute(NSUInteger index, MTLVertexFormat format, void *dst, size_t dstSize) {
    if (!dst || dstSize < 16 || index >= 16) return;
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    uint8_t bytes[16] = {0};
    if (vao) {
        const gl4metalVertexAttrib *a = &vao->attribs[index];
        if (a->currentValueIsInteger) {
            if (a->currentValueIsUnsigned || format == MTLVertexFormatUInt || format == MTLVertexFormatUInt2 ||
                format == MTLVertexFormatUInt3 || format == MTLVertexFormatUInt4) {
                memcpy(bytes, a->currentUIntValue, sizeof(a->currentUIntValue));
            } else {
                memcpy(bytes, a->currentIntValue, sizeof(a->currentIntValue));
            }
        } else {
            memcpy(bytes, a->currentValue, sizeof(a->currentValue));
        }
    } else {
        GLfloat defaults[4] = {0.f, 0.f, 0.f, 1.f};
        memcpy(bytes, defaults, sizeof(defaults));
    }
    memcpy(dst, bytes, MIN(dstSize, sizeof(bytes)));
}

static void gl4metalEnsureConstantVertexAttributeBuffer(NSUInteger index, MTLVertexFormat format) {
    if (!ctx || index >= 16 || format == MTLVertexFormatInvalid) return;
    if (!inferredConstantVertexAttributes) inferredConstantVertexAttributes = [NSMutableSet set];
    if (!constantVertexAttributeBuffers) constantVertexAttributeBuffers = [NSMutableDictionary dictionary];

    id<MTLBuffer> buffer = constantVertexAttributeBuffers[@(index)];
    if (!buffer) {
        buffer = [ctx.device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (!buffer) return;
        constantVertexAttributeBuffers[@(index)] = buffer;
    }

    gl4metalPackCurrentVertexAttribute(index, format, [buffer contents], 16);
    [inferredConstantVertexAttributes addObject:@(index)];
}

void gl4metalRegisterConstantVertexAttribute(NSUInteger index, MTLVertexFormat format) {
    gl4metalEnsureConstantVertexAttributeBuffer(index, format);
    printf("[gl4metal] Constant GL vertex attribute[%lu] -> Metal format %lu\n",
           (unsigned long)index, (unsigned long)format);
}

void gl4metalUpdateConstantVertexAttribute(NSUInteger index) {
    if (!ctx || index >= 16) return;
    id<MTLBuffer> oldBuffer = constantVertexAttributeBuffers[@(index)];
    if (!oldBuffer || ![inferredConstantVertexAttributes containsObject:@(index)]) return;
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) return;

    uint8_t bytes[16] = {0};
    if (vao->attribs[index].currentValueIsInteger) {
        if (vao->attribs[index].currentValueIsUnsigned) {
            memcpy(bytes, vao->attribs[index].currentUIntValue, 16);
        } else {
            memcpy(bytes, vao->attribs[index].currentIntValue, 16);
        }
    } else {
        memcpy(bytes, vao->attribs[index].currentValue, 16);
    }

    // Constant GL attributes are also vertex resources. Replace the Metal
    // buffer instead of mutating an object that a submitted draw may still use.
    id<MTLBuffer> replacement = oldBuffer;
    if (gl4metalSafeResourceUpdatesEnabled()) {
        replacement = [ctx.device newBufferWithBytes:bytes length:sizeof(bytes) options:MTLResourceStorageModeShared];
        if (!replacement) return;
        constantVertexAttributeBuffers[@(index)] = replacement;
    } else {
        memcpy(oldBuffer.contents, bytes, sizeof(bytes));
    }
}


static void gl4metalGetFallbackMVP(float matrix[16]) {
    static const float identity[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    if (cachedFallbackMVPIsValid) {
        memcpy(matrix, cachedFallbackMVP, sizeof(cachedFallbackMVP));
        return;
    }

    memcpy(matrix, identity, sizeof(identity));
}

@implementation gl4metalVertexArray
@end

@implementation gl4metalContext
@end

const char *gl4metalGetDeviceName(void) {
    NSString *device = [NSString stringWithFormat:@"gl4metal (%@)", ctx.device ? ctx.device.name : @"Unknown"];
    return device.UTF8String;
}

NSUInteger gl4metalGetGPUFamily(void) {
    return ctx ? ctx.gpuFamily : 0;
}

GLboolean APIENTRY gl4metalInit(void) {
    if (ctx) return GL_TRUE;

    ctx = [[gl4metalContext alloc] init];
    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) return GL_FALSE;

    ctx.commandQueue = [ctx.device newCommandQueue];
    ctx.boundBuffers = [[NSMutableDictionary alloc] init];
    ctx.indexedBufferBindings = [[NSMutableDictionary alloc] init];
    ctx.framebufferDrawBuffers = [@{@0: @(GL_BACK)} mutableCopy];
    ctx.framebufferReadBuffers = [@{@0: @(GL_BACK)} mutableCopy];
    ctx.programPipelines = [[NSMutableDictionary alloc] init];
    ctx.programResourceInfo = [[NSMutableDictionary alloc] init];
    inferredConstantVertexAttributes = [NSMutableSet set];
    constantVertexAttributeBuffers = [NSMutableDictionary dictionary];
    ctx.viewport = (gl4metalRect){0, 0, 800, 600};
    ctx.scissorRect = (gl4metalRect){0, 0, 800, 600};
    ctx.depthTestEnabled = GL_FALSE;
    ctx.stencilTestEnabled = GL_FALSE;
    ctx.stencilFunc = GL_ALWAYS;
    ctx.stencilRef = 0;
    ctx.stencilReadMask = 0xffffffffu;
    ctx.stencilFail = GL_KEEP;
    ctx.stencilDepthFail = GL_KEEP;
    ctx.stencilDepthPass = GL_KEEP;
    ctx.stencilBackFunc = GL_ALWAYS;
    ctx.stencilBackRef = 0;
    ctx.stencilBackReadMask = 0xffffffffu;
    ctx.stencilBackWriteMask = 0xffffffffu;
    ctx.stencilBackFail = GL_KEEP;
    ctx.stencilBackDepthFail = GL_KEEP;
    ctx.stencilBackDepthPass = GL_KEEP;
    ctx.depthWriteEnabled = GL_TRUE;
    ctx.blendEnabled = GL_FALSE;
    ctx.blendSrcRGB = GL_ONE;
    ctx.blendDstRGB = GL_ZERO;
    ctx.blendSrcAlpha = GL_ONE;
    ctx.blendDstAlpha = GL_ZERO;
    ctx.blendEquationRGB = GL_FUNC_ADD;
    ctx.blendEquationAlpha = GL_FUNC_ADD;
    ctx.blendConstantRed = 0.0f;
    ctx.blendConstantGreen = 0.0f;
    ctx.blendConstantBlue = 0.0f;
    ctx.blendConstantAlpha = 0.0f;
    ctx.depthFunc = GL_LESS;
    ctx.cullEnabled = GL_FALSE;
    ctx.cullFace = GL_BACK;
    ctx.frontFace = GL_CCW;
    ctx.polygonMode = GL_FILL;
    ctx.lineWidth = 1.0f;
    ctx.pointSize = 1.0f;
    ctx.polygonOffsetFactor = 0.0f;
    ctx.polygonOffsetUnits = 0.0f;
    ctx.polygonOffsetFillEnabled = GL_FALSE;
    ctx.polygonOffsetLineEnabled = GL_FALSE;
    ctx.polygonOffsetPointEnabled = GL_FALSE;
    ctx.sampleCoverageEnabled = GL_FALSE;
    ctx.sampleCoverageValue = 1.0f;
    ctx.sampleCoverageInvert = GL_FALSE;
    ctx.primitiveRestartEnabled = GL_FALSE;
    ctx.primitiveRestartIndex = 0xffffffffu;
    ctx.depthRangeNear = 0.0;
    ctx.depthRangeFar = 1.0;
    ctx.clearStencil = 0;
    ctx.stencilWriteMask = 0xffffffffu;
    ctx.colorMaskRed = GL_TRUE;
    ctx.colorMaskGreen = GL_TRUE;
    ctx.colorMaskBlue = GL_TRUE;
    ctx.colorMaskAlpha = GL_TRUE;
    ctx.drawBuffer = GL_BACK;
    ctx.readBuffer = GL_BACK;
    ctx.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    ctx.clearDepth = 1.0;
    ctx.pendingClearFlags = 0;
    ctx.pendingClearFramebuffer = 0;
    ctx.pendingClearColor = ctx.clearColor;
    ctx.pendingClearDepth = ctx.clearDepth;
    ctx.pendingClearStencil = (GLuint)ctx.clearStencil;

    gl4metalResourceInFlightCounts = [NSMutableDictionary dictionary];
    gl4metalCurrentCommandBufferResources = nil;
    gl4metalIndexStatsCache = [NSMutableDictionary dictionary];

    bufferObjects = [NSMutableDictionary dictionary];
    bufferSizes = [NSMutableDictionary dictionary];
    bufferObjectNames = [NSMutableSet set];
    bufferGeneratedNames = [NSMutableSet set];
    vertexArrayGeneratedNames = [NSMutableSet set];
    mappedBufferPointers = [NSMutableDictionary dictionary];
    mappedBufferOffsets = [NSMutableDictionary dictionary];
    mappedBufferLengths = [NSMutableDictionary dictionary];
    mappedBufferAccesses = [NSMutableDictionary dictionary];
    gl4metalSyncCommands = [NSMutableDictionary dictionary];
    bufferUsages = [NSMutableDictionary dictionary];
    convertedIndexBuffers = [NSMutableDictionary dictionary];
    vaoObjects = [NSMutableDictionary dictionary];
    framebufferObjects = [NSMutableDictionary dictionary];
    renderbufferObjects = [NSMutableDictionary dictionary];
    shaderTypes = [NSMutableDictionary dictionary];
    // shaderSources = [NSMutableDictionary dictionary];
    shaderCompileStatus = [NSMutableDictionary dictionary];
    shaderInfoLog = [NSMutableDictionary dictionary];
    programShaders = [NSMutableDictionary dictionary];
    programLinkStatus = [NSMutableDictionary dictionary];
    programInfoLog = [NSMutableDictionary dictionary];
    uniform1fValues = [NSMutableDictionary dictionary];
    uniform2fValues = [NSMutableDictionary dictionary];
    uniform3fValues = [NSMutableDictionary dictionary];
    uniform4fValues = [NSMutableDictionary dictionary];
    uniform1iValues = [NSMutableDictionary dictionary];
    uniform2iValues = [NSMutableDictionary dictionary];
    uniform3iValues = [NSMutableDictionary dictionary];
    uniform4iValues = [NSMutableDictionary dictionary];
    uniformMatrix4fvValues = [NSMutableDictionary dictionary];
    uniformRawValues = [NSMutableDictionary dictionary];
    uniformLocationNames = [NSMutableDictionary dictionary];
    uniformLocationPrograms = [NSMutableDictionary dictionary];
    programAttribBindings = [NSMutableDictionary dictionary];
    programFragDataBindings = [NSMutableDictionary dictionary];
    programFragDataIndices = [NSMutableDictionary dictionary];
    pipelineLayoutCache = [NSMutableDictionary dictionary];
    gl4metalLogicOperation = GL_COPY;
    gl4metalColorLogicOperationEnabled = GL_FALSE;
    gl4metalClampVertexColor = GL_FIXED_ONLY;
    gl4metalClampFragmentColor = GL_FIXED_ONLY;
    gl4metalClampReadColor = GL_TRUE;
    gl4metalPointSizeMin = 0.0f;
    gl4metalPointSizeMax = 1.0f;
    gl4metalPointFadeThreshold = 1.0f;
    gl4metalPointDistanceAttenuation[0] = 1.0f;
    gl4metalPointDistanceAttenuation[1] = 0.0f;
    gl4metalPointDistanceAttenuation[2] = 0.0f;
    gl4metalPointSpriteCoordOrigin = GL_UPPER_LEFT;
    currentProgram = 0;
    gl4metalPendingErrors = [NSMutableArray array];
    lastGL4MetalError = GL_NO_ERROR;
    gl4metalInitTextureState();
    gl4metalResetFramebufferState();

    glslang::InitializeProcess();
    updateDepthStencilState();

    printf("[gl4metal] Initialized with Metal device: %s\n", ctx.device.name.UTF8String);
    return GL_TRUE;
}

void APIENTRY glMakeCurrent(void *metalLayerPtr) {
    if (!ctx || !metalLayerPtr) return;
    CAMetalLayer *layer = (__bridge CAMetalLayer *)metalLayerPtr;
    layer.device = ctx.device;
    layer.framebufferOnly = YES;
    ctx.metalLayer = layer;
}

static NSUInteger gl4metalDrawSerial = 0;
static BOOL gl4metalGPUIsolationEnabled = NO;
static BOOL gl4metalGPUIsolationFailed = NO;

// Completed command buffers return their uniform arenas here. Reusing the
// buffers avoids allocating a fresh arena for every frame. Pool size is capped
// to prevent high-water workloads from retaining unbounded GPU memory.
static NSMutableArray<id<MTLBuffer>> *gl4metalUniformArenaPool = nil;
static NSUInteger gl4metalUniformArenaPoolBytes = 0;
static const NSUInteger gl4metalUniformArenaPoolByteLimit = 8u * 1024u * 1024u;

static id<MTLCommandBuffer> gl4metalCreateCommandBuffer(void) {
    if (!ctx || !ctx.commandQueue) return nil;
    id<MTLCommandBuffer> cb = [ctx.commandQueue commandBuffer];
    if (!cb) return nil;

    // Uniform storage is owned by the command-buffer generation. Arenas are
    // acquired lazily as draws need them and held until this command buffer completes.
    if (!gl4metalUniformArenaPool) {
        gl4metalUniformArenaPool = [NSMutableArray array];
    }
    NSMutableArray<id<MTLBuffer>> *uniformArenas = [NSMutableArray array];
    NSMutableDictionary<NSValue *, id<MTLBuffer>> *usedBufferResources = [NSMutableDictionary dictionary];
    gl4metalCurrentCommandBufferResources = usedBufferResources;
    ctx.uniformArenaBuffers = uniformArenas;
    ctx.uniformArenaOffset = 0;

    cb.label = [NSString stringWithFormat:@"gl4metal frame"];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        // The command buffer is finished, so its uniform buffers are safe to
        // recycle for a future frame.
        @synchronized(gl4metalResourceInFlightCounts) {
            for (NSValue *resourceKey in usedBufferResources) {
                NSUInteger count = [gl4metalResourceInFlightCounts[resourceKey] unsignedIntegerValue];
                if (count > 1) gl4metalResourceInFlightCounts[resourceKey] = @(count - 1);
                else [gl4metalResourceInFlightCounts removeObjectForKey:resourceKey];
            }
        }
        @synchronized(gl4metalUniformArenaPool) {
            for (id<MTLBuffer> arena in uniformArenas) {
                NSUInteger arenaLength = arena.length;
                if (arenaLength <= gl4metalUniformArenaPoolByteLimit &&
                    gl4metalUniformArenaPoolBytes <= gl4metalUniformArenaPoolByteLimit - arenaLength) {
                    [gl4metalUniformArenaPool addObject:arena];
                    gl4metalUniformArenaPoolBytes += arenaLength;
                }
            }
        }
        MTLCommandBufferStatus status = completed.status;
        NSError *error = completed.error;
        if (status == MTLCommandBufferStatusError || error) {
            printf("[gl4metal][GPU] Command buffer FAILED: status=%ld error=%s\n",
                   (long)status, error ? error.localizedDescription.UTF8String : "<none>");
            if (error) {
                printf("[gl4metal][GPU] domain=%s code=%ld userInfo=%s\n",
                       error.domain.UTF8String, (long)error.code,
                       error.userInfo.description.UTF8String);
            }
        } else {
            // Successful command buffers are intentionally silent in normal runtime;
            // printing once per frame/draw is a measurable CPU-side cost.
        }
    }];
    return cb;
}

static BOOL gl4metalFinishIsolatedDraw(id<MTLCommandBuffer> commandBuffer, NSUInteger serial, const char *drawKind) {
    if (!commandBuffer) return NO;
    commandBuffer.label = [NSString stringWithFormat:@"gl4metal GPU-isolated draw #%lu (%s)",
                           (unsigned long)serial, drawKind ? drawKind : "unknown"];
    printf("[gl4metal][GPU-DEBUG] submit draw #%lu kind=%s and wait\n",
           (unsigned long)serial, drawKind ? drawKind : "unknown");
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    MTLCommandBufferStatus status = commandBuffer.status;
    NSError *error = commandBuffer.error;
    if (status == MTLCommandBufferStatusError || error) {
        printf("[gl4metal][GPU-DEBUG] FIRST FAILURE CANDIDATE draw #%lu kind=%s status=%ld\n",
               (unsigned long)serial, drawKind ? drawKind : "unknown", (long)status);
        if (error) {
            printf("[gl4metal][GPU-DEBUG] error domain=%s code=%ld desc=%s\n",
                   error.domain.UTF8String, (long)error.code, error.localizedDescription.UTF8String);
            printf("[gl4metal][GPU-DEBUG] userInfo=%s\n", error.userInfo.description.UTF8String);
        }
        gl4metalGPUIsolationFailed = YES;
        return NO;
    }
    printf("[gl4metal][GPU-DEBUG] draw #%lu completed OK\n", (unsigned long)serial);
    return YES;
}

void APIENTRY glSwapBuffers(void) {
    gl4metalEncodePendingClear();
    if (!ctx || ctx.currentFramebuffer != 0 || !ctx.currentDrawable) {
        if (getenv("GL4METAL_PRESENT_TRACE")) {
            printf("[gl4metal][PRESENT_TRACE] swap skipped ctx=%p fbo=%u drawable=%p cb=%p\\n",
                   ctx, ctx ? ctx.currentFramebuffer : 0,
                   ctx ? ctx.currentDrawable : nil,
                   ctx ? ctx.currentCommandBuffer : nil);
        }
        return;
    }

    if (getenv("GL4METAL_PRESENT_TRACE")) {
        id<MTLTexture> drawableTexture = ctx.currentDrawable.texture;
        printf("[gl4metal][PRESENT_TRACE] swap drawable=%p texture=%p %lux%lu cb=%p\\n",
               ctx.currentDrawable, drawableTexture,
               (unsigned long)drawableTexture.width,
               (unsigned long)drawableTexture.height,
               ctx.currentCommandBuffer);
    }

    if (gl4metalGPUIsolationEnabled) {
        // In isolation mode every render draw has already been committed and
        // waited on. Present the fully rendered drawable with a tiny command buffer.
        if (ctx.currentCommandBuffer) {
            printf("[gl4metal][GPU-DEBUG] warning: residual command buffer at swap; committing it normally\n");
            [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
            [ctx.currentCommandBuffer commit];
            [ctx.currentCommandBuffer waitUntilCompleted];
            ctx.currentCommandBuffer = nil;
        } else {
            id<MTLCommandBuffer> presentCB = gl4metalCreateCommandBuffer();
            if (presentCB) {
                presentCB.label = @"gl4metal GPU-isolated present";
                [presentCB presentDrawable:ctx.currentDrawable];
                printf("[gl4metal][GPU-DEBUG] present drawable\n");
                [presentCB commit];
                [presentCB waitUntilCompleted];
            }
        }
        ctx.currentDrawable = nil;
        return;
    }

    id<MTLCommandBuffer> presentCommandBuffer = ctx.currentCommandBuffer;
    if (!presentCommandBuffer) presentCommandBuffer = gl4metalCreateCommandBuffer();
    if (!presentCommandBuffer) return;
    [presentCommandBuffer presentDrawable:ctx.currentDrawable];
    [presentCommandBuffer commit];
    if (getenv("GL4METAL_PRESENT_TRACE")) {
        printf("[gl4metal][PRESENT_TRACE] present submitted\\n");
    }
    ctx.currentCommandBuffer = nil;
    ctx.currentDrawable = nil;
}

void APIENTRY glSwapInterval(GLint interval) {
    if (!ctx) return;
    ctx.swapInterval = interval;
    // Metal presentation is already asynchronous. Reducing drawable queue
    // depth for interval 0 adds back-pressure without disabling display sync.
    if (ctx.metalLayer) ctx.metalLayer.maximumDrawableCount = 3;
}

void APIENTRY glClear(GLbitfield mask) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    const GLbitfield validMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    if (mask & ~validMask) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (ctx.currentFramebuffer != 0 &&
        glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }
    if (ctx.drawBuffer == GL_NONE ||
        (!ctx.colorMaskRed && !ctx.colorMaskGreen && !ctx.colorMaskBlue && !ctx.colorMaskAlpha))
        mask &= ~GL_COLOR_BUFFER_BIT;
    if (!ctx.depthWriteEnabled) mask &= ~GL_DEPTH_BUFFER_BIT;
    if (ctx.stencilWriteMask == 0 && ctx.stencilBackWriteMask == 0)
        mask &= ~GL_STENCIL_BUFFER_BIT;
    if (mask == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    // Preserve the state snapshot and ordering of consecutive clears.
    gl4metalEncodePendingClear();
    ctx.pendingClearFlags = mask;
    ctx.pendingClearFramebuffer = ctx.currentFramebuffer;
    ctx.pendingClearColor = ctx.clearColor;
    ctx.pendingClearDepth = ctx.clearDepth;
    ctx.pendingClearStencil = (GLuint)ctx.clearStencil;
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalValidateClearBuffer(GLenum buffer, GLint drawbuffer,
                                        BOOL allowDepthStencil) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return NO; }
    if (buffer != GL_COLOR && buffer != GL_DEPTH && buffer != GL_STENCIL &&
        !(allowDepthStencil && buffer == GL_DEPTH_STENCIL)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return NO;
    }
    if (drawbuffer < 0) { gl4metalSetError(GL_INVALID_VALUE); return NO; }
    if (ctx.currentFramebuffer != 0 &&
        glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return NO;
    }
    if (buffer == GL_COLOR) {
        if (drawbuffer >= 1) { gl4metalSetError(GL_INVALID_VALUE); return NO; }
        return YES;
    }
    if (drawbuffer != 0) { gl4metalSetError(GL_INVALID_VALUE); return NO; }
    return YES;
}

static void gl4metalQueueClearBuffer(GLbitfield flags) {
    gl4metalEncodePendingClear();
    ctx.pendingClearFlags = flags;
    ctx.pendingClearFramebuffer = ctx.currentFramebuffer;
    ctx.pendingClearColor = ctx.clearColor;
    ctx.pendingClearDepth = ctx.clearDepth;
    ctx.pendingClearStencil = (GLuint)ctx.clearStencil;
}

void APIENTRY glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value) {
    if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateClearBuffer(buffer, drawbuffer, NO)) return;
    if (buffer == GL_COLOR) {
        if (ctx.drawBuffer == GL_NONE) { gl4metalSetError(GL_NO_ERROR); return; }
        gl4metalQueueClearBuffer(GL_COLOR_BUFFER_BIT);
        ctx.pendingClearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
    } else if (buffer == GL_DEPTH) {
        if (!ctx.depthWriteEnabled) { gl4metalSetError(GL_NO_ERROR); return; }
        gl4metalQueueClearBuffer(GL_DEPTH_BUFFER_BIT);
        ctx.pendingClearDepth = fmin(1.0, fmax(0.0, value[0]));
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint *value) {
    if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateClearBuffer(buffer, drawbuffer, NO)) return;
    if (buffer == GL_COLOR) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    } else if (buffer == GL_STENCIL) {
        if (ctx.stencilWriteMask == 0 && ctx.stencilBackWriteMask == 0) {
            gl4metalSetError(GL_NO_ERROR);
            return;
        }
        gl4metalQueueClearBuffer(GL_STENCIL_BUFFER_BIT);
        ctx.pendingClearStencil = (GLuint)value[0];
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value) {
    if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateClearBuffer(buffer, drawbuffer, NO)) return;
    if (buffer != GL_COLOR) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glClearBufferfi(GLenum buffer, GLint drawbuffer,
                              GLfloat depth, GLint stencil) {
    if (!gl4metalValidateClearBuffer(buffer, drawbuffer, YES)) return;
    if (buffer != GL_DEPTH_STENCIL) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLbitfield flags = 0;
    if (ctx.depthWriteEnabled) flags |= GL_DEPTH_BUFFER_BIT;
    if (ctx.stencilWriteMask != 0 || ctx.stencilBackWriteMask != 0)
        flags |= GL_STENCIL_BUFFER_BIT;
    if (flags == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    gl4metalQueueClearBuffer(flags);
    ctx.pendingClearDepth = fmin(1.0, fmax(0.0, depth));
    ctx.pendingClearStencil = (GLuint)stencil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.clearColor = MTLClearColorMake(fminf(1.0f, fmaxf(0.0f, red)),
                                       fminf(1.0f, fmaxf(0.0f, green)),
                                       fminf(1.0f, fmaxf(0.0f, blue)),
                                       fminf(1.0f, fmaxf(0.0f, alpha)));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClearDepth(GLclampd depth) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.clearDepth = fmin(1.0, fmax(0.0, depth));
    gl4metalSetError(GL_NO_ERROR);
}

static MTLRenderPassDescriptor *createRenderPassDescriptor(void) {
    if (!ctx) return nil;

    id<MTLTexture> colorTexture = nil;
    id<MTLTexture> depthTexture = nil;
    id<MTLTexture> stencilTexture = nil;
    NSUInteger targetWidth = 0;
    NSUInteger targetHeight = 0;

    if (ctx.currentFramebuffer != 0) {
        colorTexture = gl4metalGetCurrentFramebufferColorTexture();
        depthTexture = gl4metalGetCurrentFramebufferDepthTexture();
        stencilTexture = gl4metalGetCurrentFramebufferStencilTexture();
        id<MTLTexture> targetTexture = colorTexture ?: (depthTexture ?: stencilTexture);
        if (!targetTexture) {
            printf("[gl4metal] ERROR: FBO %u has no renderable attachment\n", ctx.currentFramebuffer);
            return nil;
        }
        NSUInteger level = colorTexture ? gl4metalGetCurrentFramebufferColorLevel() :
            (depthTexture ? gl4metalGetCurrentFramebufferDepthLevel() : gl4metalGetCurrentFramebufferStencilLevel());
        targetWidth = MAX((NSUInteger)1, targetTexture.width >> MIN(level, (NSUInteger)31));
        targetHeight = MAX((NSUInteger)1, targetTexture.height >> MIN(level, (NSUInteger)31));
    } else {
        if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
        if (!ctx.currentDrawable) return nil;
        colorTexture = ctx.currentDrawable.texture;
        targetWidth = colorTexture.width;
        targetHeight = colorTexture.height;
        ensureDepthTexture(CGSizeMake(targetWidth, targetHeight));
        depthTexture = ctx.depthTexture;
    }

    if (!colorTexture || targetWidth == 0 || targetHeight == 0) return nil;

    MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    GLbitfield effectiveClearFlags =
        (ctx.pendingClearFramebuffer == ctx.currentFramebuffer)
            ? ctx.pendingClearFlags : 0;
    if (!colorTexture || ctx.drawBuffer == GL_NONE) effectiveClearFlags &= ~GL_COLOR_BUFFER_BIT;
    if (!depthTexture) effectiveClearFlags &= ~GL_DEPTH_BUFFER_BIT;
    if (!stencilTexture) effectiveClearFlags &= ~GL_STENCIL_BUFFER_BIT;
    if (colorTexture) {
        passDesc.colorAttachments[0].texture = colorTexture;
        if (ctx.currentFramebuffer != 0)
            passDesc.colorAttachments[0].level = gl4metalGetCurrentFramebufferColorLevel();
        passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
        passDesc.colorAttachments[0].loadAction = (effectiveClearFlags & GL_COLOR_BUFFER_BIT)
            ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.colorAttachments[0].clearColor = ctx.pendingClearColor;
    }

    if (depthTexture) {
        passDesc.depthAttachment.texture = depthTexture;
        if (ctx.currentFramebuffer != 0)
            passDesc.depthAttachment.level = gl4metalGetCurrentFramebufferDepthLevel();
        passDesc.depthAttachment.storeAction = MTLStoreActionStore;
        passDesc.depthAttachment.loadAction = (effectiveClearFlags & GL_DEPTH_BUFFER_BIT)
            ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.depthAttachment.clearDepth = ctx.pendingClearDepth;
    }

    if (stencilTexture &&
        (stencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ||
         stencilTexture.pixelFormat == MTLPixelFormatStencil8)) {
        passDesc.stencilAttachment.texture = stencilTexture;
        if (ctx.currentFramebuffer != 0)
            passDesc.stencilAttachment.level = gl4metalGetCurrentFramebufferStencilLevel();
        passDesc.stencilAttachment.storeAction = MTLStoreActionStore;
        passDesc.stencilAttachment.loadAction = (effectiveClearFlags & GL_STENCIL_BUFFER_BIT)
            ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.stencilAttachment.clearStencil = ctx.pendingClearStencil;
    }

    if (ctx.pendingClearFramebuffer == ctx.currentFramebuffer) ctx.pendingClearFlags = 0;
    return passDesc;
}

static void gl4metalEncodePendingClear(void) {
    if (!ctx || ctx.pendingClearFlags == 0 ||
        ctx.pendingClearFramebuffer != ctx.currentFramebuffer) return;
    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = gl4metalCreateCommandBuffer();
    if (ctx.currentFramebuffer == 0 && !ctx.currentDrawable && ctx.metalLayer) {
        ctx.currentDrawable = [ctx.metalLayer nextDrawable];
    }
    if (!ctx.currentCommandBuffer) return;
    MTLRenderPassDescriptor *pass = createRenderPassDescriptor();
    if (!pass) return;
    id<MTLRenderCommandEncoder> encoder =
        [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:pass];
    [encoder endEncoding];
}

void gl4metalFlushPendingClear(void) {
    gl4metalEncodePendingClear();
}

void APIENTRY glFlush(void) {
    if (!ctx) return;
    gl4metalEncodePendingClear();
    if (!ctx.currentCommandBuffer) return;
    [ctx.currentCommandBuffer commit];
    ctx.currentCommandBuffer = nil;
}

void APIENTRY glFinish(void) {
    if (!ctx) return;
    gl4metalEncodePendingClear();
    if (!ctx.currentCommandBuffer) return;
    [ctx.currentCommandBuffer commit];
    [ctx.currentCommandBuffer waitUntilCompleted];
    ctx.currentCommandBuffer = nil;
}

void APIENTRY glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (width < 0 || height < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.viewport = (gl4metalRect){x, y, width, height};
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDepthRange(GLdouble nearValue, GLdouble farValue) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    // OpenGL clamps depth range endpoints to [0, 1]. Metal uses the same
    // normalized depth interval, so the values can be forwarded at draw time.
    ctx.depthRangeNear = fmin(1.0, fmax(0.0, nearValue));
    ctx.depthRangeFar = fmin(1.0, fmax(0.0, farValue));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (ctx) ctx.scissorRect = (gl4metalRect){x, y, width, height};
}

static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight) {
    if (!ctx || !encoder || targetWidth == 0 || targetHeight == 0) return;
    gl4metalRect viewport = ctx.viewport;
    [encoder setViewport:(MTLViewport){
        (double)viewport.x,
        (double)targetHeight - viewport.y - viewport.height,
        (double)viewport.width,
        (double)viewport.height,
        ctx.depthRangeNear,
        ctx.depthRangeFar
    }];
    gl4metalRect scissor = ctx.scissorTestEnabled ? ctx.scissorRect : (gl4metalRect){0, 0, (GLsizei)targetWidth, (GLsizei)targetHeight};
    [encoder setScissorRect:(MTLScissorRect){
        static_cast<NSUInteger>(MAX(0, scissor.x)),
        static_cast<NSUInteger>(MAX(0, (GLint)targetHeight - scissor.y - scissor.height)),
        static_cast<NSUInteger>(MIN((NSUInteger)MAX(0, scissor.width), targetWidth)),
        static_cast<NSUInteger>(MIN((NSUInteger)MAX(0, scissor.height), targetHeight))
    }];
}

void APIENTRY glCullFace(GLenum mode) {
    if (mode != GL_FRONT && mode != GL_BACK && mode != GL_FRONT_AND_BACK) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.cullFace = mode;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glFrontFace(GLenum mode) {
    if (mode != GL_CW && mode != GL_CCW) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.frontFace = mode;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPolygonMode(GLenum face, GLenum mode) {
    if (face != GL_FRONT_AND_BACK || (mode != GL_POINT && mode != GL_LINE && mode != GL_FILL)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.polygonMode = mode;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPolygonOffset(GLfloat factor, GLfloat units) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.polygonOffsetFactor = factor;
    ctx.polygonOffsetUnits = units;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glLineWidth(GLfloat width) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!isfinite(width) || width <= 0.0f) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    // Metal line primitives and wireframe rasterization are fixed to one
    // pixel. OpenGL requires out-of-range widths to be clamped to the
    // implementation's supported range; this backend reports [1, 1].
    ctx.lineWidth = 1.0f;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPointSize(GLfloat size) {
    if (size <= 0.0f) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    ctx.pointSize = size;
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalValidLogicOperation(GLenum operation) {
    switch (operation) {
        case GL_CLEAR: case GL_AND: case GL_AND_REVERSE: case GL_COPY:
        case GL_AND_INVERTED: case GL_NOOP: case GL_XOR: case GL_OR:
        case GL_NOR: case GL_EQUIV: case GL_INVERT: case GL_OR_REVERSE:
        case GL_COPY_INVERTED: case GL_OR_INVERTED: case GL_NAND: case GL_SET:
            return YES;
        default:
            return NO;
    }
}

static void gl4metalPointParameterScalar(GLenum pname, GLfloat value) {
    switch (pname) {
        case GL_POINT_SIZE_MIN:
            if (value < 0.0f) { gl4metalSetError(GL_INVALID_VALUE); return; }
            gl4metalPointSizeMin = value;
            break;
        case GL_POINT_SIZE_MAX:
            if (value < 0.0f) { gl4metalSetError(GL_INVALID_VALUE); return; }
            gl4metalPointSizeMax = value;
            break;
        case GL_POINT_FADE_THRESHOLD_SIZE:
            if (value < 0.0f) { gl4metalSetError(GL_INVALID_VALUE); return; }
            gl4metalPointFadeThreshold = value;
            break;
        case GL_POINT_SPRITE_COORD_ORIGIN:
            if (value != (GLfloat)GL_UPPER_LEFT && value != (GLfloat)GL_LOWER_LEFT) {
                gl4metalSetError(GL_INVALID_ENUM); return;
            }
            gl4metalPointSpriteCoordOrigin = (GLenum)value;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

static void gl4metalPointParameterAttenuation(const GLfloat value[3]) {
    if (value[0] < 0.0f || value[1] < 0.0f || value[2] < 0.0f ||
        (value[0] == 0.0f && value[1] == 0.0f && value[2] == 0.0f)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    memcpy(gl4metalPointDistanceAttenuation, value, sizeof(gl4metalPointDistanceAttenuation));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPointParameterf(GLenum pname, GLfloat param) {
    gl4metalPointParameterScalar(pname, param);
}

void APIENTRY glPointParameteri(GLenum pname, GLint param) {
    gl4metalPointParameterScalar(pname, (GLfloat)param);
}

void APIENTRY glPointParameterfv(GLenum pname, const GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_POINT_DISTANCE_ATTENUATION) gl4metalPointParameterAttenuation(params);
    else gl4metalPointParameterScalar(pname, params[0]);
}

void APIENTRY glPointParameteriv(GLenum pname, const GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_POINT_DISTANCE_ATTENUATION) {
        GLfloat value[3] = {(GLfloat)params[0], (GLfloat)params[1], (GLfloat)params[2]};
        gl4metalPointParameterAttenuation(value);
    } else {
        gl4metalPointParameterScalar(pname, (GLfloat)params[0]);
    }
}

void APIENTRY glLogicOp(GLenum operation) {
    if (!gl4metalValidLogicOperation(operation)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    gl4metalLogicOperation = operation;
    if (ctx) {
        ctx.pipelineState = nil;
        [pipelineLayoutCache removeAllObjects];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClampColor(GLenum target, GLenum clamp) {
    if (target != GL_CLAMP_VERTEX_COLOR && target != GL_CLAMP_FRAGMENT_COLOR &&
        target != GL_CLAMP_READ_COLOR) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if ((target == GL_CLAMP_READ_COLOR && clamp != GL_TRUE && clamp != GL_FALSE) ||
        (target != GL_CLAMP_READ_COLOR && clamp != GL_TRUE && clamp != GL_FALSE && clamp != GL_FIXED_ONLY)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (target == GL_CLAMP_VERTEX_COLOR) gl4metalClampVertexColor = clamp;
    else if (target == GL_CLAMP_FRAGMENT_COLOR) gl4metalClampFragmentColor = clamp;
    else gl4metalClampReadColor = clamp;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glClearStencil(GLint s) {
    ctx.clearStencil = s;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glStencilMask(GLuint mask) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.stencilWriteMask = mask;
    ctx.stencilBackWriteMask = mask;
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glStencilFunc(GLenum func, GLint ref, GLuint mask) {
    if (func != GL_NEVER && func != GL_LESS && func != GL_EQUAL && func != GL_LEQUAL &&
        func != GL_GREATER && func != GL_NOTEQUAL && func != GL_GEQUAL && func != GL_ALWAYS) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    ctx.stencilFunc = func;
    ctx.stencilRef = ref;
    ctx.stencilReadMask = mask;
    ctx.stencilBackFunc = func;
    ctx.stencilBackRef = ref;
    ctx.stencilBackReadMask = mask;
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalValidStencilOperation(GLenum operation) {
    return operation == GL_KEEP || operation == GL_ZERO || operation == GL_REPLACE ||
           operation == GL_INCR || operation == GL_DECR || operation == GL_INVERT ||
           operation == GL_INCR_WRAP || operation == GL_DECR_WRAP;
}

void APIENTRY glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
    if (!gl4metalValidStencilOperation(fail) || !gl4metalValidStencilOperation(zfail) ||
        !gl4metalValidStencilOperation(zpass)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    ctx.stencilFail = fail;
    ctx.stencilDepthFail = zfail;
    ctx.stencilDepthPass = zpass;
    ctx.stencilBackFail = fail;
    ctx.stencilBackDepthFail = zfail;
    ctx.stencilBackDepthPass = zpass;
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalValidStencilFace(GLenum face) {
    return face == GL_FRONT || face == GL_BACK || face == GL_FRONT_AND_BACK;
}

void APIENTRY glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
    if (!gl4metalValidStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (func != GL_NEVER && func != GL_LESS && func != GL_EQUAL && func != GL_LEQUAL &&
        func != GL_GREATER && func != GL_NOTEQUAL && func != GL_GEQUAL && func != GL_ALWAYS) {
        gl4metalSetError(GL_INVALID_ENUM); return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (face != GL_BACK) {
        ctx.stencilFunc = func; ctx.stencilRef = ref; ctx.stencilReadMask = mask;
    }
    if (face != GL_FRONT) {
        ctx.stencilBackFunc = func; ctx.stencilBackRef = ref; ctx.stencilBackReadMask = mask;
    }
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glStencilMaskSeparate(GLenum face, GLuint mask) {
    if (!gl4metalValidStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (face != GL_BACK) ctx.stencilWriteMask = mask;
    if (face != GL_FRONT) ctx.stencilBackWriteMask = mask;
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glStencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass) {
    if (!gl4metalValidStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!gl4metalValidStencilOperation(fail) || !gl4metalValidStencilOperation(zfail) ||
        !gl4metalValidStencilOperation(zpass)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (face != GL_BACK) {
        ctx.stencilFail = fail; ctx.stencilDepthFail = zfail; ctx.stencilDepthPass = zpass;
    }
    if (face != GL_FRONT) {
        ctx.stencilBackFail = fail; ctx.stencilBackDepthFail = zfail; ctx.stencilBackDepthPass = zpass;
    }
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.colorMaskRed = red;
    ctx.colorMaskGreen = green;
    ctx.colorMaskBlue = blue;
    ctx.colorMaskAlpha = alpha;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDrawBuffer(GLenum buf) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    BOOL valid = ctx.currentFramebuffer == 0
        ? (buf == GL_BACK || buf == GL_NONE)
        : (buf == GL_COLOR_ATTACHMENT0 || buf == GL_NONE);
    if (!valid) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.drawBuffer = buf;
    ctx.framebufferDrawBuffers[@(ctx.currentFramebuffer)] = @(buf);
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDrawBuffers(GLsizei n, const GLenum *bufs) {
    if (n < 0 || (n > 0 && !bufs)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (n > 1) {
        // This backend exposes a single color attachment/draw target.
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (n == 0) {
        ctx.drawBuffer = GL_NONE;
        ctx.framebufferDrawBuffers[@(ctx.currentFramebuffer)] = @(GL_NONE);
        ctx.pipelineState = nil;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }
    glDrawBuffer(bufs[0]);
}

void APIENTRY glReadBuffer(GLenum src) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    BOOL valid = NO;
    if (ctx.currentReadFramebuffer == 0) {
        valid = src == GL_BACK || src == GL_NONE;
    } else {
        valid = src == GL_COLOR_ATTACHMENT0 || src == GL_NONE;
    }
    if (!valid) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.readBuffer = src;
    ctx.framebufferReadBuffers[@(ctx.currentReadFramebuffer)] = @(src);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glHint(GLenum target, GLenum mode) {
    if (target != GL_LINE_SMOOTH_HINT && target != GL_POLYGON_SMOOTH_HINT &&
        target != GL_TEXTURE_COMPRESSION_HINT && target != GL_FRAGMENT_SHADER_DERIVATIVE_HINT) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (mode != GL_DONT_CARE && mode != GL_FASTEST && mode != GL_NICEST) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

static void applyRasterState(id<MTLRenderCommandEncoder> encoder) {
    if (!ctx || !encoder) return;

    MTLCullMode cullMode = MTLCullModeNone;
    if (ctx.cullEnabled) {
        cullMode = ctx.cullFace == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    }
    [encoder setCullMode:cullMode];
    BOOL polygonOffsetEnabled = ctx.polygonMode == GL_FILL ? ctx.polygonOffsetFillEnabled :
                                ctx.polygonMode == GL_LINE ? ctx.polygonOffsetLineEnabled :
                                ctx.polygonOffsetPointEnabled;
    [encoder setDepthBias:polygonOffsetEnabled ? ctx.polygonOffsetUnits : 0.0f
              slopeScale:polygonOffsetEnabled ? ctx.polygonOffsetFactor : 0.0f
                   clamp:0.0f];
    // Keep the original OpenGL -> Metal front-face mapping. The GL viewport
    // Y conversion is a viewport-coordinate conversion, not a reason to
    // globally invert the rasterizer winding. A global inversion makes
    // Minecraft programs that enable back-face culling disappear.
    [encoder setFrontFacingWinding:ctx.frontFace == GL_CW ? MTLWindingClockwise : MTLWindingCounterClockwise];

    // Optional diagnostics for the text renderer (program 53). These are
    // intentionally opt-in because Program 53 can be reused by different
    // render paths depending on the Minecraft build.
    if (currentProgram == 53) {
        if (getenv("GL4METAL_PROGRAM53_NO_CULL")) {
            [encoder setCullMode:MTLCullModeNone];
        }
    }
    [encoder setTriangleFillMode:ctx.polygonMode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
    if (ctx.blendEnabled) {
        [encoder setBlendColorRed:ctx.blendConstantRed green:ctx.blendConstantGreen blue:ctx.blendConstantBlue alpha:ctx.blendConstantAlpha];
    }
}

void APIENTRY glGenBuffers(GLsizei n, GLuint *buffers) {
    if (n < 0 || (n > 0 && !buffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    for (GLsizei i = 0; i < n; i++) {
        GLuint name = 0;
        do {
            name = nextBufferId++;
            if (name == 0) name = nextBufferId++;
        } while ([bufferObjectNames containsObject:@(name)] ||
                 [bufferGeneratedNames containsObject:@(name)]);
        buffers[i] = name;
        [bufferGeneratedNames addObject:@(name)];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays) {
    if (n < 0 || (n > 0 && !arrays)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    for (GLsizei i = 0; i < n; i++) {
        GLuint id = 0;
        do {
            id = nextVAOId++;
            if (id == 0) id = nextVAOId++;
        } while ([vertexArrayGeneratedNames containsObject:@(id)] || vaoObjects[@(id)] != nil);
        arrays[i] = id;
        [vertexArrayGeneratedNames addObject:@(id)];
    }
    gl4metalSetError(GL_NO_ERROR);
}

gl4metalVertexArray *gl4metalGetCurrentVAO(void) {
    if (!ctx || ctx.currentVAO == 0) return nil;
    return vaoObjects[@(ctx.currentVAO)];
}

static GLuint gl4metalGetBoundBufferForTarget(GLenum target) {
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (target == GL_ELEMENT_ARRAY_BUFFER && vao) return vao->elementArrayBufferID;

    NSNumber *bound = ctx.boundBuffers[@(target)];
    return bound ? [bound unsignedIntValue] : 0;
}

static BOOL gl4metalValidBufferTarget(GLenum target) {
    switch (target) {
        case GL_ARRAY_BUFFER:
        case GL_ELEMENT_ARRAY_BUFFER:
        case GL_PIXEL_PACK_BUFFER:
        case GL_PIXEL_UNPACK_BUFFER:
        case GL_COPY_READ_BUFFER:
        case GL_COPY_WRITE_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_UNIFORM_BUFFER:
            return YES;
        default:
            return NO;
    }
}

void APIENTRY glBindBuffer(GLenum target, GLuint buffer) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!gl4metalValidBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (buffer != 0) {
        // Buffer names need not come from glGenBuffers; first binding creates
        // the object in desktop OpenGL. Keep the generated-name set only for
        // distinguishing names reserved by glGenBuffers before that bind.
        [bufferGeneratedNames removeObject:@(buffer)];
        [bufferObjectNames addObject:@(buffer)];
    }
    ctx.boundBuffers[@(target)] = @(buffer);

    if (ctx.currentVAO != 0) {
        gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
        if (vao) {
            if (target == GL_ELEMENT_ARRAY_BUFFER) {
                vao->elementArrayBufferID = buffer;
            }
        }
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    NSUInteger maxBindings = target == GL_UNIFORM_BUFFER ? 24 :
                             target == GL_TRANSFORM_FEEDBACK_BUFFER ? 4 : 0;
    if (maxBindings == 0) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (index >= maxBindings || offset < 0 || size <= 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    id<MTLBuffer> mtlBuffer = bufferObjects[@(buffer)];
    if (buffer == 0 || ![bufferObjectNames containsObject:@(buffer)] || !mtlBuffer) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if ((size_t)offset > mtlBuffer.length || (size_t)size > mtlBuffer.length - (size_t)offset) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_UNIFORM_BUFFER && ((uint64_t)offset % 256u) != 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glBindBuffer(target, buffer);
    NSMutableDictionary *bindings = ctx.indexedBufferBindings[@(target)];
    if (!bindings) {
        bindings = [NSMutableDictionary dictionary];
        ctx.indexedBufferBindings[@(target)] = bindings;
    }
    bindings[@(index)] = @{@"buffer": @(buffer), @"offset": @((GLint64)offset), @"size": @((GLint64)size)};
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
    NSUInteger maxBindings = target == GL_UNIFORM_BUFFER ? 24 :
                             target == GL_TRANSFORM_FEEDBACK_BUFFER ? 4 : 0;
    if (maxBindings == 0) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (index >= maxBindings) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (buffer != 0 && ![bufferObjectNames containsObject:@(buffer)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    glBindBuffer(target, buffer);
    NSMutableDictionary *bindings = ctx.indexedBufferBindings[@(target)];
    if (!bindings) {
        bindings = [NSMutableDictionary dictionary];
        ctx.indexedBufferBindings[@(target)] = bindings;
    }
    id<MTLBuffer> mtlBuffer = bufferObjects[@(buffer)];
    bindings[@(index)] = @{@"buffer": @(buffer), @"offset": @0,
                           @"size": @((GLint64)(mtlBuffer ? mtlBuffer.length : 0))};
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    if (!gl4metalValidBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);

    if (currentBound == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    if (size < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (usage != GL_STREAM_DRAW && usage != GL_STREAM_READ && usage != GL_STREAM_COPY &&
        usage != GL_STATIC_DRAW && usage != GL_STATIC_READ && usage != GL_STATIC_COPY &&
        usage != GL_DYNAMIC_DRAW && usage != GL_DYNAMIC_READ && usage != GL_DYNAMIC_COPY) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    if (size == 0) {
        [bufferObjects removeObjectForKey:@(currentBound)];
        bufferSizes[@(currentBound)] = @0;
        bufferUsages[@(currentBound)] = @(usage);
        gl4metalClearIndexCaches();
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    id<MTLBuffer> oldBuffer = bufferObjects[@(currentBound)];
    if (oldBuffer) {
        gl4metalClearIndexCaches();
    }

    id<MTLBuffer> mtlBuffer = nil;
    if (data != NULL) {
        mtlBuffer = [ctx.device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    } else {
        mtlBuffer = [ctx.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    }

    if (mtlBuffer) {
        bufferObjects[@(currentBound)] = mtlBuffer;
        bufferSizes[@(currentBound)] = @(size);
        bufferUsages[@(currentBound)] = @(usage);
        gl4metalSetError(GL_NO_ERROR);
    } else {
        gl4metalSetError(GL_OUT_OF_MEMORY);
    }
}

void APIENTRY glDeleteBuffers(GLsizei n, const GLuint *buffers) {
    if (n < 0 || (n > 0 && !buffers)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        GLuint buffer = buffers[i];
        if (buffer == 0) continue;
        [bufferGeneratedNames removeObject:@(buffer)];
        [bufferObjectNames removeObject:@(buffer)];
        [bufferObjects removeObjectForKey:@(buffer)];
        [bufferSizes removeObjectForKey:@(buffer)];
        [bufferUsages removeObjectForKey:@(buffer)];
        [mappedBufferPointers removeObjectForKey:@(buffer)];
        [mappedBufferOffsets removeObjectForKey:@(buffer)];
        [mappedBufferLengths removeObjectForKey:@(buffer)];
        [mappedBufferAccesses removeObjectForKey:@(buffer)];
        gl4metalClearIndexCaches();
        for (NSNumber *target in [ctx.boundBuffers.allKeys copy]) {
            if ([ctx.boundBuffers[target] unsignedIntValue] == buffer) ctx.boundBuffers[target] = @(0);
        }
        for (NSNumber *target in ctx.indexedBufferBindings) {
            NSMutableDictionary *bindings = ctx.indexedBufferBindings[target];
            for (NSNumber *index in [bindings.allKeys copy]) {
                if ([bindings[index][@"buffer"] unsignedIntValue] == buffer)
                    bindings[index] = @{@"buffer": @0, @"offset": @0, @"size": @0};
            }
        }
        for (gl4metalVertexArray *vao in vaoObjects.allValues) {
            if (vao->arrayBufferID == buffer) vao->arrayBufferID = 0;
            if (vao->elementArrayBufferID == buffer) vao->elementArrayBufferID = 0;
            for (NSUInteger index = 0; index < 16; ++index) {
                if (vao->attribs[index].boundVBO == buffer) vao->attribs[index].boundVBO = 0;
            }
        }
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    if (!gl4metalValidBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!buffer || (size > 0 && !data)) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (offset < 0 || size < 0 || (size_t)offset > (size_t)[buffer length] || (size_t)size > (size_t)[buffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glBufferSubData OOB: buffer=%u offset=%ld size=%ld length=%lu\n", currentBound, (long)offset, (long)size, (unsigned long)[buffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    if (size == 0) { gl4metalSetError(GL_NO_ERROR); return; }

    if (gl4metalSafeResourceUpdatesEnabled() && gl4metalBufferIsInFlight(currentBound)) {
        id<MTLBuffer> replacement = gl4metalCopyBufferForWrite(currentBound, buffer, offset, size, data);
        if (!replacement) {
            gl4metalSetError(GL_OUT_OF_MEMORY);
            return;
        }
        bufferObjects[@(currentBound)] = replacement;
    } else {
        memcpy((char *)[buffer contents] + offset, data, (size_t)size);
    }
    gl4metalClearIndexCaches();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void *data) {
    if (!gl4metalValidBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (!buffer || (size > 0 && !data)) {
        gl4metalSetError(size > 0 && !data ? GL_INVALID_VALUE : GL_INVALID_OPERATION);
        return;
    }
    if (offset < 0 || size < 0 || (size_t)offset > (size_t)buffer.length ||
        (size_t)size > (size_t)buffer.length - (size_t)offset) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (size > 0) memcpy(data, (const uint8_t *)buffer.contents + offset, (size_t)size);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glCopyBufferSubData(GLenum readTarget, GLenum writeTarget,
                                  GLintptr readOffset, GLintptr writeOffset,
                                  GLsizeiptr size) {
    if (!gl4metalValidBufferTarget(readTarget) || !gl4metalValidBufferTarget(writeTarget)) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    GLuint readBufferName = gl4metalGetBoundBufferForTarget(readTarget);
    GLuint writeBufferName = gl4metalGetBoundBufferForTarget(writeTarget);
    id<MTLBuffer> source = bufferObjects[@(readBufferName)];
    id<MTLBuffer> destination = bufferObjects[@(writeBufferName)];
    if (!source || !destination) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (readOffset < 0 || writeOffset < 0 || size < 0 ||
        (size_t)readOffset > (size_t)source.length ||
        (size_t)size > (size_t)source.length - (size_t)readOffset ||
        (size_t)writeOffset > (size_t)destination.length ||
        (size_t)size > (size_t)destination.length - (size_t)writeOffset) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (readBufferName == writeBufferName && size > 0 &&
        (size_t)readOffset < (size_t)writeOffset + (size_t)size &&
        (size_t)writeOffset < (size_t)readOffset + (size_t)size) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (size > 0 && gl4metalSafeResourceUpdatesEnabled() && gl4metalBufferIsInFlight(writeBufferName)) {
        id<MTLBuffer> replacement = gl4metalCopyBufferForWrite(
            writeBufferName, destination, writeOffset, size,
            (const uint8_t *)source.contents + readOffset);
        if (!replacement) {
            gl4metalSetError(GL_OUT_OF_MEMORY);
            return;
        }
        bufferObjects[@(writeBufferName)] = replacement;
    } else if (size > 0) {
        memcpy((uint8_t *)destination.contents + writeOffset,
               (const uint8_t *)source.contents + readOffset, (size_t)size);
    }
    gl4metalClearIndexCaches();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBufferParameteriv(GLenum target, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (!currentBound || !buffer) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_BUFFER_SIZE:
            params[0] = (GLint)[buffer length];
            break;
        case GL_BUFFER_USAGE:
            params[0] = [bufferUsages[@(currentBound)] intValue];
            break;
        case GL_BUFFER_MAPPED:
            params[0] = mappedBufferPointers[@(currentBound)] ? GL_TRUE : GL_FALSE;
            break;
        case GL_BUFFER_ACCESS_FLAGS:
            params[0] = [mappedBufferAccesses[@(currentBound)] intValue];
            break;
        case GL_BUFFER_MAP_OFFSET:
            params[0] = (GLint)[mappedBufferOffsets[@(currentBound)] longLongValue];
            break;
        case GL_BUFFER_MAP_LENGTH:
            params[0] = (GLint)[mappedBufferLengths[@(currentBound)] longLongValue];
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64 *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (!currentBound || !buffer) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_BUFFER_SIZE:
            params[0] = (GLint64)[buffer length];
            break;
        case GL_BUFFER_MAP_OFFSET:
            params[0] = [mappedBufferOffsets[@(currentBound)] longLongValue];
            break;
        case GL_BUFFER_MAP_LENGTH:
            params[0] = [mappedBufferLengths[@(currentBound)] longLongValue];
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetIntegeri_v(GLenum target, GLuint index, GLint *data) {
    if (!data) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }

    switch (target) {
        case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: {
            if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
            gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
            data[0] = (GLint)(vao && index < 16 ? vao->attribs[index].boundVBO : 0);
            break;
        }
        case GL_UNIFORM_BUFFER_BINDING:
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: {
            NSUInteger maxBindings = target == GL_UNIFORM_BUFFER_BINDING ? 24 : 4;
            if (index >= maxBindings) { gl4metalSetError(GL_INVALID_VALUE); return; }
            NSDictionary *binding = ctx.indexedBufferBindings[@(target == GL_UNIFORM_BUFFER_BINDING
                ? GL_UNIFORM_BUFFER : GL_TRANSFORM_FEEDBACK_BUFFER)][@(index)];
            data[0] = [binding[@"buffer"] intValue];
            break;
        }
        case GL_SAMPLER_BINDING:
            if (index >= 32) { gl4metalSetError(GL_INVALID_VALUE); return; }
            data[0] = (GLint)gl4metalGetSamplerNameForUnit(index);
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetInteger64i_v(GLenum target, GLuint index, GLint64 *data) {
    if (!data) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    GLenum bindingTarget = 0;
    NSUInteger maxBindings = 0;
    BOOL wantOffset = NO;
    if (target == GL_UNIFORM_BUFFER_START || target == GL_UNIFORM_BUFFER_SIZE) {
        bindingTarget = GL_UNIFORM_BUFFER; maxBindings = 24;
        wantOffset = target == GL_UNIFORM_BUFFER_START;
    } else if (target == GL_TRANSFORM_FEEDBACK_BUFFER_START || target == GL_TRANSFORM_FEEDBACK_BUFFER_SIZE) {
        bindingTarget = GL_TRANSFORM_FEEDBACK_BUFFER; maxBindings = 4;
        wantOffset = target == GL_TRANSFORM_FEEDBACK_BUFFER_START;
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (index >= maxBindings) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSDictionary *binding = ctx.indexedBufferBindings[@(bindingTarget)][@(index)];
    data[0] = [binding[wantOffset ? @"offset" : @"size"] longLongValue];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetInteger64v(GLenum pname, GLint64 *data) {
    if (!data) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (pname == GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT) {
        data[0] = 256;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }
    GLint value = 0;
    glGetIntegerv(pname, &value);
    data[0] = value;
    gl4metalSetError(GL_NO_ERROR);
}

void* APIENTRY glMapBuffer(GLenum target, GLenum access) {
    if (access != GL_READ_ONLY && access != GL_WRITE_ONLY && access != GL_READ_WRITE) {
        gl4metalSetError(GL_INVALID_ENUM);
        return NULL;
    }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }

    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) {
        printf("[gl4metal] ERROR: glMapbuffer: No MTLBuffer found for ID %u at target 0x%X\n", currentBound, target);
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }

    if (gl4metalSafeResourceUpdatesEnabled() && gl4metalBufferIsInFlight(currentBound)) {
        id<MTLBuffer> replacement = gl4metalCopyBufferForWrite(currentBound, mtlBuffer, 0, (GLsizeiptr)[mtlBuffer length], mtlBuffer.contents);
        if (!replacement) {
            gl4metalSetError(GL_OUT_OF_MEMORY);
            return NULL;
        }
        bufferObjects[@(currentBound)] = replacement;
        mtlBuffer = replacement;
    }

    void *mapped = [mtlBuffer contents];
    mappedBufferPointers[@(currentBound)] = [NSValue valueWithPointer:mapped];
    mappedBufferOffsets[@(currentBound)] = @0;
    mappedBufferLengths[@(currentBound)] = @([mtlBuffer length]);
    GLbitfield mapFlags = access == GL_READ_ONLY ? GL_MAP_READ_BIT :
                          access == GL_WRITE_ONLY ? GL_MAP_WRITE_BIT :
                          (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    mappedBufferAccesses[@(currentBound)] = @(mapFlags);
    gl4metalSetError(GL_NO_ERROR);
    return mapped;
}

void *APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    const GLbitfield validFlags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
        GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
        GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_UNSYNCHRONIZED_BIT;
    if ((access & ~validFlags) != 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return NULL;
    }
    if ((access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)) == 0 ||
        ((access & GL_MAP_INVALIDATE_RANGE_BIT) && (access & GL_MAP_INVALIDATE_BUFFER_BIT)) ||
        ((access & GL_MAP_READ_BIT) && (access & (GL_MAP_INVALIDATE_RANGE_BIT |
                                                  GL_MAP_INVALIDATE_BUFFER_BIT |
                                                  GL_MAP_UNSYNCHRONIZED_BIT))) ||
        ((access & GL_MAP_FLUSH_EXPLICIT_BIT) && !(access & GL_MAP_WRITE_BIT))) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }
    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) { gl4metalSetError(GL_INVALID_OPERATION); return NULL; }

    if (offset < 0 || length <= 0 || (size_t)offset > (size_t)[mtlBuffer length] || (size_t)length > (size_t)[mtlBuffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glMapBufferRange OOB: buffer=%u offset=%ld length=%ld bufferLength=%lu\n", currentBound, (long)offset, (long)length, (unsigned long)[mtlBuffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
        return NULL;
    }
    if (gl4metalSafeResourceUpdatesEnabled() && gl4metalBufferIsInFlight(currentBound)) {
        // INVALIDATE_BUFFER explicitly discards the old contents. Avoid copying
        // the full in-flight allocation only to zero it immediately below.
        id<MTLBuffer> replacement = (access & GL_MAP_INVALIDATE_BUFFER_BIT)
            ? [ctx.device newBufferWithLength:mtlBuffer.length options:MTLResourceStorageModeShared]
            : gl4metalCopyBufferForWrite(currentBound, mtlBuffer, 0,
                                         (GLsizeiptr)mtlBuffer.length, mtlBuffer.contents);
        if (!replacement) {
            gl4metalSetError(GL_OUT_OF_MEMORY);
            return NULL;
        }
        bufferObjects[@(currentBound)] = replacement;
        mtlBuffer = replacement;
    }
    if (access & GL_MAP_INVALIDATE_BUFFER_BIT) {
        memset([mtlBuffer contents], 0, [mtlBuffer length]);
    } else if (access & GL_MAP_INVALIDATE_RANGE_BIT) {
        memset((uint8_t *)[mtlBuffer contents] + offset, 0, (size_t)length);
    }
    gl4metalSetError(GL_NO_ERROR);
    void *mapped = (void *)((char *)[mtlBuffer contents] + offset);
    mappedBufferPointers[@(currentBound)] = [NSValue valueWithPointer:mapped];
    mappedBufferOffsets[@(currentBound)] = @(offset);
    mappedBufferLengths[@(currentBound)] = @(length);
    mappedBufferAccesses[@(currentBound)] = @(access);
    return mapped;
}

void APIENTRY glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    NSNumber *flags = mappedBufferAccesses[@(currentBound)];
    if (currentBound == 0 || !mappedBufferPointers[@(currentBound)] || !flags ||
        !([flags unsignedIntValue] & GL_MAP_FLUSH_EXPLICIT_BIT)) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLintptr mappedLength = (GLintptr)[mappedBufferLengths[@(currentBound)] longLongValue];
    if (offset < 0 || length < 0 || offset > mappedLength || length > mappedLength - offset) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    // MTLBuffer uses shared storage for GL buffer objects, so writes are
    // coherent already; validating the explicit flush range is sufficient.
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glUnmapBuffer(GLenum target) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || !mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    [mappedBufferPointers removeObjectForKey:@(currentBound)];
    [mappedBufferOffsets removeObjectForKey:@(currentBound)];
    [mappedBufferLengths removeObjectForKey:@(currentBound)];
    [mappedBufferAccesses removeObjectForKey:@(currentBound)];
    gl4metalClearIndexCaches();
    gl4metalSetError(GL_NO_ERROR);
    return GL_TRUE;
}

void APIENTRY glGetBufferPointerv(GLenum target, GLenum pname, void **params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (pname != GL_BUFFER_MAP_POINTER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || !bufferObjects[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    *params = [mappedBufferPointers[@(currentBound)] pointerValue];
    gl4metalSetError(GL_NO_ERROR);
}

// VAO State
void APIENTRY glBindVertexArray(GLuint array) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (array != 0 && ![vertexArrayGeneratedNames containsObject:@(array)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    // VAO changes alter the vertex descriptor selected by the current GL
    // program. The pipeline cache key includes VAO state, but a previously
    // resolved ctx.pipelineState must never survive a VAO switch.
    if (ctx) ctx.pipelineState = nil;
    ctx.currentVAO = array;
    if (array == 0) {
        ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] = @(0);
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    gl4metalVertexArray *vao = vaoObjects[@(array)];
    if (!vao) {
        vao = [[gl4metalVertexArray alloc] init];
        vao.id = array;
        for (NSUInteger a = 0; a < 16; a++) {
            vao->attribs[a].size = 4;
            vao->attribs[a].type = GL_FLOAT;
            vao->attribs[a].currentValue[0] = 0.0f;
            vao->attribs[a].currentValue[1] = 0.0f;
            vao->attribs[a].currentValue[2] = 0.0f;
            vao->attribs[a].currentValue[3] = 1.0f;
            vao->attribs[a].currentIntValue[0] = 0;
            vao->attribs[a].currentIntValue[1] = 0;
            vao->attribs[a].currentIntValue[2] = 0;
            vao->attribs[a].currentIntValue[3] = 1;
            vao->attribs[a].currentUIntValue[0] = 0;
            vao->attribs[a].currentUIntValue[1] = 0;
            vao->attribs[a].currentUIntValue[2] = 0;
            vao->attribs[a].currentUIntValue[3] = 1;
        }
        vaoObjects[@(array)] = vao;
    }

    ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] = @(vao->elementArrayBufferID);
    gl4metalSetError(GL_NO_ERROR);
}

static void gl4metalInvalidateVertexLayoutPipeline(void) {
    if (ctx) ctx.pipelineState = nil;
}

void APIENTRY glEnableVertexAttribArray(GLuint index) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (!vao) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    vao->attribs[index].enabled = GL_TRUE;
    gl4metalInvalidateVertexLayoutPipeline();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDisableVertexAttribArray(GLuint index) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (!vao) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    vao->attribs[index].enabled = GL_FALSE;
    gl4metalInvalidateVertexLayoutPipeline();
    gl4metalSetError(GL_NO_ERROR);
}

static gl4metalVertexAttrib *gl4metalGetVertexAttribState(GLuint index) {
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return NULL;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (vao) return &vao->attribs[index];

    static gl4metalVertexAttrib defaultAttribs[16];
    static BOOL initialized = NO;
    if (!initialized) {
        for (NSUInteger i = 0; i < 16; ++i) {
            defaultAttribs[i].size = 4;
            defaultAttribs[i].type = GL_FLOAT;
            defaultAttribs[i].currentValue[3] = 1.0f;
            defaultAttribs[i].currentIntValue[3] = 1;
            defaultAttribs[i].currentUIntValue[3] = 1;
        }
        initialized = YES;
    }
    return &defaultAttribs[index];
}

static GLint gl4metalVertexAttribIntValue(const gl4metalVertexAttrib *attrib, GLenum pname) {
    switch (pname) {
        case GL_VERTEX_ATTRIB_ARRAY_ENABLED: return attrib->enabled;
        case GL_VERTEX_ATTRIB_ARRAY_SIZE: return attrib->size;
        case GL_VERTEX_ATTRIB_ARRAY_STRIDE: return attrib->stride;
        case GL_VERTEX_ATTRIB_ARRAY_TYPE: return (GLint)attrib->type;
        case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED: return attrib->normalized;
        case GL_VERTEX_ATTRIB_ARRAY_INTEGER: return attrib->integer;
        case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: return (GLint)attrib->boundVBO;
        case GL_VERTEX_ATTRIB_ARRAY_DIVISOR: return (GLint)attrib->divisor;
        default: return 0;
    }
}

void APIENTRY glGetVertexAttribiv(GLuint index, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        for (NSUInteger i = 0; i < 4; ++i) {
            params[i] = attrib->currentValueIsInteger
                ? (attrib->currentValueIsUnsigned ? (GLint)attrib->currentUIntValue[i] : attrib->currentIntValue[i])
                : (GLint)attrib->currentValue[i];
        }
    } else if (pname == GL_VERTEX_ATTRIB_ARRAY_ENABLED || pname == GL_VERTEX_ATTRIB_ARRAY_SIZE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_STRIDE || pname == GL_VERTEX_ATTRIB_ARRAY_TYPE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_NORMALIZED || pname == GL_VERTEX_ATTRIB_ARRAY_INTEGER ||
               pname == GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING || pname == GL_VERTEX_ATTRIB_ARRAY_DIVISOR) {
        params[0] = gl4metalVertexAttribIntValue(attrib, pname);
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        for (NSUInteger i = 0; i < 4; ++i) {
            if (attrib->currentValueIsInteger) {
                params[i] = attrib->currentValueIsUnsigned
                    ? (GLfloat)attrib->currentUIntValue[i] : (GLfloat)attrib->currentIntValue[i];
            } else {
                params[i] = attrib->currentValue[i];
            }
        }
    } else if (pname == GL_VERTEX_ATTRIB_ARRAY_ENABLED || pname == GL_VERTEX_ATTRIB_ARRAY_SIZE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_STRIDE || pname == GL_VERTEX_ATTRIB_ARRAY_TYPE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_NORMALIZED || pname == GL_VERTEX_ATTRIB_ARRAY_INTEGER ||
               pname == GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING || pname == GL_VERTEX_ATTRIB_ARRAY_DIVISOR) {
        params[0] = (GLfloat)gl4metalVertexAttribIntValue(attrib, pname);
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetVertexAttribdv(GLuint index, GLenum pname, GLdouble *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        for (NSUInteger i = 0; i < 4; ++i) {
            if (attrib->currentValueIsInteger) {
                params[i] = attrib->currentValueIsUnsigned
                    ? (GLdouble)attrib->currentUIntValue[i] : (GLdouble)attrib->currentIntValue[i];
            } else {
                params[i] = attrib->currentValue[i];
            }
        }
    } else if (pname == GL_VERTEX_ATTRIB_ARRAY_ENABLED || pname == GL_VERTEX_ATTRIB_ARRAY_SIZE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_STRIDE || pname == GL_VERTEX_ATTRIB_ARRAY_TYPE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_NORMALIZED || pname == GL_VERTEX_ATTRIB_ARRAY_INTEGER ||
               pname == GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING || pname == GL_VERTEX_ATTRIB_ARRAY_DIVISOR) {
        params[0] = (GLdouble)gl4metalVertexAttribIntValue(attrib, pname);
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetVertexAttribPointerv(GLuint index, GLenum pname, void **pointer) {
    if (!pointer) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname != GL_VERTEX_ATTRIB_ARRAY_POINTER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    *pointer = (void *)(uintptr_t)attrib->pointerOffset;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetVertexAttribIiv(GLuint index, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        for (NSUInteger i = 0; i < 4; ++i) {
            params[i] = attrib->currentValueIsUnsigned
                ? (GLint)attrib->currentUIntValue[i] : attrib->currentIntValue[i];
        }
    } else if (pname == GL_VERTEX_ATTRIB_ARRAY_ENABLED || pname == GL_VERTEX_ATTRIB_ARRAY_SIZE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_STRIDE || pname == GL_VERTEX_ATTRIB_ARRAY_TYPE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_NORMALIZED || pname == GL_VERTEX_ATTRIB_ARRAY_INTEGER ||
               pname == GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING || pname == GL_VERTEX_ATTRIB_ARRAY_DIVISOR) {
        params[0] = gl4metalVertexAttribIntValue(attrib, pname);
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexAttrib *attrib = gl4metalGetVertexAttribState(index);
    if (!attrib) return;
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        for (NSUInteger i = 0; i < 4; ++i) {
            params[i] = attrib->currentValueIsUnsigned
                ? attrib->currentUIntValue[i] : (GLuint)attrib->currentIntValue[i];
        }
    } else if (pname == GL_VERTEX_ATTRIB_ARRAY_ENABLED || pname == GL_VERTEX_ATTRIB_ARRAY_SIZE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_STRIDE || pname == GL_VERTEX_ATTRIB_ARRAY_TYPE ||
               pname == GL_VERTEX_ATTRIB_ARRAY_NORMALIZED || pname == GL_VERTEX_ATTRIB_ARRAY_INTEGER ||
               pname == GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING || pname == GL_VERTEX_ATTRIB_ARRAY_DIVISOR) {
        params[0] = (GLuint)gl4metalVertexAttribIntValue(attrib, pname);
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

// Recover tightly-packed interleaved vertex offsets only when the GL state is
// internally consistent with a packed layout. This is intentionally generic
// for Minecraft-style VertexFormat layouts: attribute locations are ordered,
// all attributes share one VBO/stride, every recorded offset is zero, and the
// sum of element byte sizes exactly matches the declared stride. In that case
// there is no padding/unknown gap to guess, so cumulative byte offsets are
// deterministic. Explicit non-zero GL offsets are never overwritten.
static size_t gl4metalAttribPackedElementSize(const gl4metalVertexAttrib *a) {
    if (!a || a->size <= 0) return 0;
    size_t scalar = 0;
    switch (a->type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            scalar = 1;
            break;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
            scalar = 2;
            break;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
            scalar = 4;
            break;
        default:
            return 0;
    }
    return scalar * (size_t)a->size;
}

static void gl4metalRecoverPackedInterleavedOffsets(gl4metalVertexArray *vao) {
    if (!vao) return;

    GLuint seenVBOs[16] = {0};
    NSUInteger seenCount = 0;

    for (NSUInteger ai = 0; ai < 16; ++ai) {
        gl4metalVertexAttrib *first = &vao->attribs[ai];
        if (!first->enabled || first->boundVBO == 0 || first->stride <= 0 || first->divisor != 0)
            continue;

        GLuint vbo = first->boundVBO;
        BOOL alreadySeen = NO;
        for (NSUInteger si = 0; si < seenCount; ++si) {
            if (seenVBOs[si] == vbo) {
                alreadySeen = YES;
                break;
            }
        }
        if (alreadySeen) continue;
        if (seenCount < 16) seenVBOs[seenCount++] = vbo;

        GLsizei stride = first->stride;
        BOOL allZero = YES;
        BOOL sameStride = YES;
        NSUInteger attrCount = 0;
        size_t packedSize = 0;

        for (NSUInteger aj = 0; aj < 16; ++aj) {
            gl4metalVertexAttrib *a = &vao->attribs[aj];
            if (!a->enabled || a->boundVBO != vbo) continue;

            attrCount++;
            if (a->stride != stride || a->divisor != 0) sameStride = NO;
            if (a->pointerOffset != 0) allZero = NO;

            size_t elemSize = gl4metalAttribPackedElementSize(a);
            if (elemSize == 0 || packedSize > SIZE_MAX - elemSize) {
                packedSize = 0;
                sameStride = NO;
                break;
            }
            packedSize += elemSize;
        }

        // Require a real interleaved layout (2+ attributes) whose element
        // sizes exactly fill the stride. This avoids guessing padding or
        // separate-array layouts that merely happen to have zero offsets.
        if (attrCount < 2 || !allZero || !sameStride || packedSize != (size_t)stride)
            continue;

        // Minecraft's POSITION_COLOR_TEX vertex format stores Color before
        // UV in the interleaved VBO even when the linked shader binds UV and
        // Color to locations 1 and 2 respectively. Program 17 is the vanilla
        // text/UI path in 1.20.1 and its reflected formats are Float3, Float2,
        // UChar4Normalized. Recover the real VBO byte order explicitly instead
        // of assuming attribute-location order equals struct-member order.
        if (currentProgram == 17 && stride == 24 && attrCount == 3) {
            NSUInteger posIndex = NSNotFound;
            NSUInteger uvIndex = NSNotFound;
            NSUInteger colorIndex = NSNotFound;
            for (NSUInteger aj = 0; aj < 16; ++aj) {
                gl4metalVertexAttrib *a = &vao->attribs[aj];
                if (!a->enabled || a->boundVBO != vbo) continue;
                if (a->type == GL_FLOAT && a->size == 3) posIndex = aj;
                else if (a->type == GL_FLOAT && a->size == 2) uvIndex = aj;
                else if (a->type == GL_UNSIGNED_BYTE && a->size == 4 && a->normalized) colorIndex = aj;
            }
            if (posIndex != NSNotFound && uvIndex != NSNotFound && colorIndex != NSNotFound) {
                vao->attribs[posIndex].pointerOffset = 0;
                vao->attribs[colorIndex].pointerOffset = 12;
                vao->attribs[uvIndex].pointerOffset = 16;
                printf("[gl4metal][ATTR_RECOVER] program=17 POSITION_COLOR_TEX VBO=%u stride=24 offsets: a%lu=0 a%lu=12 a%lu=16\n",
                       vbo, (unsigned long)posIndex, (unsigned long)colorIndex, (unsigned long)uvIndex);
                continue;
            }
        }

        size_t runningOffset = 0;
        NSUInteger changed = 0;
        BOOL valid = YES;
        for (NSUInteger aj = 0; aj < 16; ++aj) {
            gl4metalVertexAttrib *a = &vao->attribs[aj];
            if (!a->enabled || a->boundVBO != vbo) continue;

            size_t elemSize = gl4metalAttribPackedElementSize(a);
            if (elemSize == 0 || runningOffset > (size_t)stride - elemSize) {
                valid = NO;
                break;
            }
            a->pointerOffset = (GLintptr)runningOffset;
            runningOffset += elemSize;
            changed++;
        }

        if (!valid || runningOffset != (size_t)stride) {
            // Should not be reachable after the packedSize check, but keep the
            // operation fail-safe if the state changes during iteration.
            for (NSUInteger aj = 0; aj < 16; ++aj) {
                gl4metalVertexAttrib *a = &vao->attribs[aj];
                if (!a->enabled || a->boundVBO != vbo) continue;
                a->pointerOffset = 0;
            }
            continue;
        }

        if (changed > 0) {
            printf("[gl4metal][ATTR_RECOVER] VBO=%u stride=%d packed=%zu offsets:",
                   vbo, stride, packedSize);
            for (NSUInteger aj = 0; aj < 16; ++aj) {
                gl4metalVertexAttrib *a = &vao->attribs[aj];
                if (!a->enabled || a->boundVBO != vbo) continue;
                printf(" a%lu=%ld", (unsigned long)aj, (long)a->pointerOffset);
            }
            printf("\n");
        }
    }
}

static BOOL gl4metalValidateBufferOffset(id<MTLBuffer> buffer, GLintptr offsetBytes, NSUInteger *validatedOffset) {
    if (!buffer || !validatedOffset) return NO;

    size_t bufferLength = (size_t)[buffer length];
    uintptr_t offset = (uintptr_t)offsetBytes;

    if (offsetBytes == 0) {
        *validatedOffset = 0;
        return YES;
    }

    if (offset >= bufferLength) {
        printf("[gl4metal] WARNING: buffer offset 0x%zx exceeds buffer length 0x%zx; clamping to 0\n", (size_t)offset, bufferLength);
        *validatedOffset = 0;
        return NO;
    }

    *validatedOffset = (NSUInteger)offset;
    return YES;
}

void APIENTRY glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (size < 1 || size > 4 || stride < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) {
        printf("[gl4metal] ERROR: glVertexAttribPointer called without bound VAO!\n");
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    GLuint currentVBO = gl4metalGetBoundBufferForTarget(GL_ARRAY_BUFFER);
    if (currentVBO == 0) {
        printf("[gl4metal] ERROR: glVertexAttribPointer called without bound GL_ARRAY_BUFFER\n");
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    size_t elemSize = 0;
    switch (type) {
        case GL_FLOAT: elemSize = sizeof(GLfloat); break;
        case GL_UNSIGNED_BYTE: elemSize = sizeof(GLubyte); break;
        case GL_UNSIGNED_SHORT: elemSize = sizeof(GLushort); break;
        case GL_UNSIGNED_INT: elemSize = sizeof(GLuint); break;
        case GL_BYTE: elemSize = sizeof(GLbyte); break;
        case GL_SHORT: elemSize = sizeof(GLshort); break;
        case GL_INT: elemSize = sizeof(GLint); break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    if (stride == 0) {
        stride = (GLsizei)(size * (GLint)elemSize);
    }

    id<MTLBuffer> boundBuffer = bufferObjects[@(currentVBO)];
    GLintptr pointerOffset = gl4metalResolvePointerOffset(pointer, boundBuffer);

    vao->attribs[index].size = size;
    vao->attribs[index].type = type;
    vao->attribs[index].normalized = normalized;
    vao->attribs[index].integer = GL_FALSE;
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointerOffset = pointerOffset;
    vao->attribs[index].boundVBO = currentVBO;
    gl4metalInvalidateVertexLayoutPipeline();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (size < 1 || size > 4 || stride < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    GLuint currentVBO = gl4metalGetBoundBufferForTarget(GL_ARRAY_BUFFER);
    if (currentVBO == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    size_t elemSize = 0;
    switch (type) {
        case GL_BYTE:           elemSize = sizeof(GLbyte); break;
        case GL_UNSIGNED_BYTE:  elemSize = sizeof(GLubyte); break;
        case GL_SHORT:          elemSize = sizeof(GLshort); break;
        case GL_UNSIGNED_SHORT: elemSize = sizeof(GLushort); break;
        case GL_INT:            elemSize = sizeof(GLint); break;
        case GL_UNSIGNED_INT:   elemSize = sizeof(GLuint); break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    if (stride == 0) stride = (GLsizei)(size * (GLint)elemSize);

    id<MTLBuffer> boundBuffer = bufferObjects[@(currentVBO)];
    GLintptr pointerOffset = gl4metalResolvePointerOffset(pointer, boundBuffer);

    vao->attribs[index].size = size;
    vao->attribs[index].type = type;
    vao->attribs[index].normalized = GL_FALSE;
    vao->attribs[index].integer = GL_TRUE;
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointerOffset = pointerOffset;
    vao->attribs[index].boundVBO = currentVBO;
    gl4metalInvalidateVertexLayoutPipeline();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribDivisor(GLuint index, GLuint divisor) {
    if (index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    vao->attribs[index].divisor = divisor;
    gl4metalInvalidateVertexLayoutPipeline();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteVertexArrays(GLsizei n, const GLuint *arrays) {
    if (n < 0 || (n > 0 && !arrays)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        if (arrays[i] == 0) continue;
        if (ctx.currentVAO == arrays[i]) glBindVertexArray(0);
        [vertexArrayGeneratedNames removeObject:@(arrays[i])];
        [vaoObjects removeObjectForKey:@(arrays[i])];
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsBuffer(GLuint buffer) {
    return buffer != 0 && [bufferObjectNames containsObject:@(buffer)] ? GL_TRUE : GL_FALSE;
}

GLboolean APIENTRY glIsVertexArray(GLuint array) {
    return array != 0 && vaoObjects[@(array)] != nil ? GL_TRUE : GL_FALSE;
}

GLuint APIENTRY glCreateShader(GLenum type) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return 0;
    }
    if (type != GL_VERTEX_SHADER && type != GL_FRAGMENT_SHADER &&
        type != GL_GEOMETRY_SHADER) {
        gl4metalSetError(GL_INVALID_ENUM);
        return 0;
    }

    GLuint shaderId = nextShaderId++;
    GLShader data;
    data.type = type;
    shaderTypes[@(shaderId)] = @(type);
    shaderSources[shaderId] = std::move(data);
    shaderCompileStatus[@(shaderId)] = @(GL_FALSE);
    shaderInfoLog[@(shaderId)] = @"";
    gl4metalSetError(GL_NO_ERROR);
    return shaderId;
}

void APIENTRY glShaderSource(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length) {
    if (!ctx || shader == 0 || count <= 0 || string == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (shaderSources.find(shader) == shaderSources.end()) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    std::string source;
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;

        if (length != nullptr && length[i] >= 0) {
            source.append(string[i], static_cast<size_t>(length[i]));
        } else {
            source.append(string[i]);
        }
    }

    shaderSources[shader].glslSource = source;
    shaderCompileStatus[@(shader)] = @(GL_FALSE);
    shaderInfoLog[@(shader)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glCompileShader(GLuint shader) {
    auto it = shaderSources.find(shader);
    if (it == shaderSources.end()) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    GLShader &shaderData = it->second;
    shaderData.compiled = false;
    shaderData.mslSource.clear();
    shaderData.infoLog.clear();
    shaderData.mtlLibrary = nil;
    shaderData.mtlFunction = nil;
    shaderCompileStatus[@(shader)] = @(GL_FALSE);

    std::string transpileDiagnostics;
    std::string mslSource = transpileGLSLToMSL(
        shaderData.type, shaderData.glslSource, &transpileDiagnostics);
    shaderData.mslSource = mslSource;
    if (mslSource.empty()) {
        shaderData.compiled = false;
        shaderCompileStatus[@(shader)] = @(GL_FALSE);
        shaderData.infoLog = transpileDiagnostics.empty()
            ? "Failed to transpile GLSL to MSL" : transpileDiagnostics;
        NSString *message = [NSString stringWithUTF8String:shaderData.infoLog.c_str()];
        shaderInfoLog[@(shader)] = message ?: @"Failed to transpile GLSL to MSL";
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    NSString *sourceCode = [NSString stringWithUTF8String:mslSource.c_str()];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_1;

    NSError *error = nil;
    shaderData.mtlLibrary = [ctx.device newLibraryWithSource:sourceCode options:options error:&error];

    if (error || !shaderData.mtlLibrary) {
        NSString *message = error.localizedDescription ?: @"Unknown Metal shader compilation error";
        printf("[gl4metal] Shader compilation error: %s\n", message.UTF8String);
        shaderData.compiled = false;
        shaderCompileStatus[@(shader)] = @(GL_FALSE);
        shaderData.infoLog = message.UTF8String ? message.UTF8String : "Unknown Metal shader compilation error";
        shaderInfoLog[@(shader)] = message;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    shaderData.mtlFunction = [shaderData.mtlLibrary newFunctionWithName:@"main0"];
    if (!shaderData.mtlFunction) {
        printf("[gl4metal] Shader function 'main0' not found in compiled library\n");
        shaderData.compiled = false;
    }
    shaderData.compiled = (shaderData.mtlFunction != nil);
    shaderCompileStatus[@(shader)] = @(shaderData.compiled ? GL_TRUE : GL_FALSE);

    if (!shaderData.compiled) {
        shaderInfoLog[@(shader)] = @"Metal library compiled, but shader entry point 'main0' was not found";
        shaderData.infoLog = "Metal library compiled, but shader entry point 'main0' was not found";
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    shaderData.infoLog.clear();
    shaderInfoLog[@(shader)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

GLuint APIENTRY glCreateProgram(void) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return 0;
    }

    GLuint programId = nextProgramId++;
    programShaders[@(programId)] = [[NSMutableArray alloc] init];
    programLinkStatus[@(programId)] = @(GL_FALSE);
    programInfoLog[@(programId)] = @"";
    gl4metalSetError(GL_NO_ERROR);
    return programId;
}

void APIENTRY glAttachShader(GLuint program, GLuint shader) {
    if (!ctx || program == 0 || shader == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (!attached) {
        attached = [[NSMutableArray alloc] init];
        programShaders[@(program)] = attached;
    }

    if (![attached containsObject:@(shader)]) {
        [attached addObject:@(shader)];
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDetachShader(GLuint program, GLuint shader) {
    if (!programShaders[@(program)] || shader == 0 || shaderSources.find(shader) == shaderSources.end()) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    NSUInteger index = [attached indexOfObject:@(shader)];
    if (index == NSNotFound) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    [attached removeObjectAtIndex:index];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetAttachedShaders(GLuint program, GLsizei maxCount,
                                   GLsizei *count, GLuint *shaders) {
    if (maxCount < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (!attached) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLsizei copied = (GLsizei)MIN((NSUInteger)maxCount, attached.count);
    if (copied > 0 && !shaders) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < copied; ++i) shaders[i] = attached[(NSUInteger)i].unsignedIntValue;
    if (count) *count = copied;
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsShader(GLuint shader) {
    return shader != 0 && shaderSources.find(shader) != shaderSources.end() ? GL_TRUE : GL_FALSE;
}

GLboolean APIENTRY glIsProgram(GLuint program) {
    return program != 0 && programShaders[@(program)] != nil ? GL_TRUE : GL_FALSE;
}

void APIENTRY glLinkProgram(GLuint program) {
    if (!ctx || program == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (attached == nil || [attached count] == 0) {
        programLinkStatus[@(program)] = @(GL_FALSE);
        programInfoLog[@(program)] = @"No shaders attached";
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    for (NSNumber *shaderId in attached) {
        if ([shaderCompileStatus[shaderId] boolValue] == GL_FALSE) {
            programLinkStatus[@(program)] = @(GL_FALSE);
            programInfoLog[@(program)] = @"Attached shader not compiled";
            gl4metalSetError(GL_INVALID_OPERATION);
            return;
        }
    }

    programLinkStatus[@(program)] = @(GL_TRUE);
    programInfoLog[@(program)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glValidateProgram(GLuint program) {
    if (!programShaders[@(program)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    // Pipeline resources are checked when the Metal render pipeline is
    // assembled. At validation time, the GL-visible prerequisite is that the
    // executable linked successfully.
    gl4metalSetError(GL_NO_ERROR);
}

static MTLVertexFormat gl4metalVertexFormatForAttrib(const gl4metalVertexAttrib *attrib) {
    if (!attrib) return MTLVertexFormatInvalid;

    // glVertexAttribIPointer means the shader input is an integer vector.
    // Metal must receive an integer vertex format; feeding Float4 here makes
    // pipeline creation fail with: "Cannot convert attribute from
    // MTLAttributeFormatFloat4 to int2 or uint2".
    if (attrib->integer) {
        switch (attrib->type) {
            case GL_BYTE:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatChar;
                    case 2: return MTLVertexFormatChar2;
                    case 3: return MTLVertexFormatChar3;
                    case 4: return MTLVertexFormatChar4;
                }
                break;
            case GL_UNSIGNED_BYTE:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatUChar;
                    case 2: return MTLVertexFormatUChar2;
                    case 3: return MTLVertexFormatUChar3;
                    case 4: return MTLVertexFormatUChar4;
                }
                break;
            case GL_SHORT:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatShort;
                    case 2: return MTLVertexFormatShort2;
                    case 3: return MTLVertexFormatShort3;
                    case 4: return MTLVertexFormatShort4;
                }
                break;
            case GL_UNSIGNED_SHORT:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatUShort;
                    case 2: return MTLVertexFormatUShort2;
                    case 3: return MTLVertexFormatUShort3;
                    case 4: return MTLVertexFormatUShort4;
                }
                break;
            case GL_INT:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatInt;
                    case 2: return MTLVertexFormatInt2;
                    case 3: return MTLVertexFormatInt3;
                    case 4: return MTLVertexFormatInt4;
                }
                break;
            case GL_UNSIGNED_INT:
                switch (attrib->size) {
                    case 1: return MTLVertexFormatUInt;
                    case 2: return MTLVertexFormatUInt2;
                    case 3: return MTLVertexFormatUInt3;
                    case 4: return MTLVertexFormatUInt4;
                }
                break;
            default:
                break;
        }
        return MTLVertexFormatInvalid;
    }

    switch (attrib->type) {
        case GL_FLOAT:
            switch (attrib->size) {
                case 1: return MTLVertexFormatFloat;
                case 2: return MTLVertexFormatFloat2;
                case 3: return MTLVertexFormatFloat3;
                case 4: return MTLVertexFormatFloat4;
            }
            break;
        case GL_UNSIGNED_BYTE:
            switch (attrib->size) {
                case 1: return attrib->normalized ? MTLVertexFormatUCharNormalized : MTLVertexFormatUChar;
                case 2: return attrib->normalized ? MTLVertexFormatUChar2Normalized : MTLVertexFormatUChar2;
                case 4: return attrib->normalized ? MTLVertexFormatUChar4Normalized : MTLVertexFormatUChar4;
            }
            break;
        case GL_UNSIGNED_SHORT:
            switch (attrib->size) {
                case 1: return attrib->normalized ? MTLVertexFormatUShortNormalized : MTLVertexFormatUShort;
                case 2: return attrib->normalized ? MTLVertexFormatUShort2Normalized : MTLVertexFormatUShort2;
                case 4: return attrib->normalized ? MTLVertexFormatUShort4Normalized : MTLVertexFormatUShort4;
            }
            break;
        case GL_INT:
            // Non-integer glVertexAttribPointer with GL_INT is converted to
            // floating point by OpenGL.
            switch (attrib->size) {
                case 1: return MTLVertexFormatFloat;
                case 2: return MTLVertexFormatFloat2;
                case 3: return MTLVertexFormatFloat3;
                case 4: return MTLVertexFormatFloat4;
            }
            break;
        case GL_UNSIGNED_INT:
            switch (attrib->size) {
                case 1: return MTLVertexFormatFloat;
                case 2: return MTLVertexFormatFloat2;
                case 3: return MTLVertexFormatFloat3;
                case 4: return MTLVertexFormatFloat4;
            }
            break;
        default:
            break;
    }
    return MTLVertexFormatInvalid;
}

MTLVertexDescriptor *gl4metalCreateVertexDescriptorForCurrentVAO(void) {
    MTLVertexDescriptor *descriptor = [MTLVertexDescriptor vertexDescriptor];
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();

    for (NSUInteger index = 0; index < 16; index++) {
        gl4metalVertexAttrib *attrib = vao ? &vao->attribs[index] : NULL;
        if (!attrib || !attrib->enabled) continue;

        MTLVertexFormat format = gl4metalVertexFormatForAttrib(attrib);
        descriptor.attributes[index].format = format;

        // A Metal vertex-buffer layout must only exist when at least one
        // valid vertex attribute actually references that bufferIndex.
        // Leaving an orphan layout (for example layout[5] with no attribute
        // using bufferIndex 5) makes MTLVertexDescriptor reject the whole
        // descriptor during serialization.
        if (format == MTLVertexFormatInvalid) {
            descriptor.attributes[index].offset = 0;
            descriptor.attributes[index].bufferIndex = 0;
            descriptor.layouts[index].stride = 0;
            descriptor.layouts[index].stepRate = 1;
            descriptor.layouts[index].stepFunction = MTLVertexStepFunctionPerVertex;
            continue;
        }

        // pointerOffset is applied when the buffer is bound in the draw call.
        descriptor.attributes[index].offset = 0;
        descriptor.attributes[index].bufferIndex = index;
        descriptor.layouts[index].stride = attrib->stride > 0
            ? (NSUInteger)attrib->stride
            : (NSUInteger)(MAX(attrib->size, 1) * (GLsizei)sizeof(GLfloat));
        descriptor.layouts[index].stepRate = attrib->divisor > 0 ? attrib->divisor : 1;
        descriptor.layouts[index].stepFunction = attrib->divisor > 0
            ? MTLVertexStepFunctionPerInstance
            : MTLVertexStepFunctionPerVertex;
    }

    // Do not invent Float4 attributes for a real GL program. If an attribute
    // is disabled, leave the Metal descriptor entry invalid. The previous
    // Float4 defaults could conflict with a real shader declaring int2/uint2.
    return descriptor;
}


static id<MTLRenderPipelineState> gl4metalResolveCurrentPipeline(void) {
    if (!ctx) return nil;
    if (gl4metalColorLogicOperationEnabled &&
        (gl4metalLogicOperation != GL_COPY || currentProgram == 0)) {
        // Metal has no fixed-function bitwise color logic operation. Reject
        // draws we cannot reproduce instead of silently rendering the wrong
        // pixels. GL_COPY is handled in the real-program pipeline below.
        gl4metalSetError(GL_INVALID_OPERATION);
        return nil;
    }
    if (currentProgram != 0) {
        NSDictionary *indices = programFragDataIndices[@(currentProgram)];
        for (NSDictionary *output in gl4metalProgramFragmentOutputs(currentProgram)) {
            if ([indices[output[@"name"]] unsignedIntValue] == 1 &&
                indices[output[@"name"]] != nil) {
                // Metal does not expose the dual-source fragment output pair
                // required by GL index 1.
                gl4metalSetError(GL_INVALID_OPERATION);
                return nil;
            }
        }
    }

    // A GL program's pipeline must be resolved from the current program and
    // current vertex-array layout. Never let the generic fallback pipeline
    // short-circuit a real program.
    if (currentProgram != 0) {
        NSString *cacheKey = gl4metalBuildPipelineCacheKey(currentProgram, gl4metalGetCurrentVAO());
        id<MTLRenderPipelineState> layoutPipeline = pipelineLayoutCache[cacheKey];
        if (layoutPipeline) {
            ctx.pipelineState = layoutPipeline;
            return layoutPipeline;
        }

        if (!gl4metalCreateProgramPipelineForProgram(currentProgram)) {
            ctx.pipelineState = nil;
            return nil;
        }

        id<MTLRenderPipelineState> programPipeline = ctx.pipelineState;
        if (programPipeline) {
            ctx.programPipelines[@(currentProgram)] = programPipeline;
            pipelineLayoutCache[cacheKey] = programPipeline;
            return programPipeline;
        }

        ctx.pipelineState = nil;
        return nil;
    }

    // Program 0 is the only path allowed to use the compatibility fallback.
    ctx.pipelineState = defaultFallbackPipeline;
    return defaultFallbackPipeline;
}

void APIENTRY glUseProgram(GLuint program) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    GLuint previousProgram = currentProgram;
    if (program == 0) {
        currentProgram = 0;
        ctx.pipelineState = defaultFallbackPipeline;
        gl4metalFinalizePendingProgramDelete(previousProgram);
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    if (programLinkStatus[@(program)] == nil ||
        [programLinkStatus[@(program)] boolValue] == GL_FALSE) {
        gl4metalSetError(GL_INVALID_OPERATION);
        printf("[gl4metal] glUseProgram called with unlinked or invalid program: %u\n", program);
        return;
    }

    currentProgram = program;
    if (previousProgram != program)
        gl4metalFinalizePendingProgramDelete(previousProgram);

    // Do not keep the previous program/fallback pipeline merely because the
    // program ID is unchanged. Resolve against the current VAO/layout.
    ctx.pipelineState = nil;
    id<MTLRenderPipelineState> programPipeline = gl4metalResolveCurrentPipeline();
    if (!programPipeline) {
        printf("[gl4metal] ERROR: Program %u is linked, but its real Metal pipeline could not be created\n", program);
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    ctx.pipelineState = programPipeline;
    gl4metalSetError(GL_NO_ERROR);
}

GLint APIENTRY glGetUniformLocation(GLuint program, const GLchar *name) {
    if (!ctx || name == nil) {
        gl4metalSetError(GL_INVALID_VALUE);
        return -1;
    }
    if (!programShaders[@(program)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return -1;
    }
    if (![programLinkStatus[@(program)] boolValue]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    NSString *requestedName = [NSString stringWithUTF8String:name];
    BOOL active = NO;
    for (NSDictionary *variable in gl4metalProgramVariables(program, YES)) {
        NSString *activeName = variable[@"name"];
        NSString *baseName = [activeName hasSuffix:@"[0]"]
            ? [activeName substringToIndex:activeName.length - 3] : activeName;
        if ([requestedName isEqualToString:activeName] || [requestedName isEqualToString:baseName]) {
            active = YES;
            break;
        }
    }
    if (!active) {
        gl4metalSetError(GL_NO_ERROR);
        return -1;
    }

    NSString *canonicalName = requestedName;
    if ([requestedName hasSuffix:@"[0]"]) {
        canonicalName = [requestedName substringToIndex:requestedName.length - 3];
    }
    GLint location = gl4metalResolveLocationFromName(program, canonicalName.UTF8String);
    while (uniformLocationNames[@(location)] &&
           (![uniformLocationPrograms[@(location)] isEqual:@(program)] ||
            ![uniformLocationNames[@(location)] isEqualToString:canonicalName])) {
        location = location == INT_MAX ? 1 : location + 1;
    }
    uniformLocationNames[@(location)] = canonicalName;
    uniformLocationPrograms[@(location)] = @(program);
    gl4metalSetError(GL_NO_ERROR);
    return location;
}

static BOOL gl4metalUniformUpdateUnavailable(GLint location) {
    if (location == -1) return YES;
    NSNumber *owner = uniformLocationPrograms[@(location)];
    if (!ctx || currentProgram == 0 || !owner || owner.unsignedIntValue != currentProgram) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return YES;
    }
    return NO;
}

static void gl4metalStoreUniformRaw(GLint location, const void *bytes, NSUInteger length) {
    if (location < 0 || !bytes || length == 0) return;
    uniformRawValues[@(location)] = [NSData dataWithBytes:bytes length:length];
}

void APIENTRY glUniform1f(GLint location, GLfloat v0) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1fValues[@(location)] = [NSValue valueWithBytes:&v0 objCType:@encode(GLfloat)];
    gl4metalStoreUniformRaw(location, &v0, sizeof(v0));
    gl4metalTraceUniformUpdate(location, "1f", &v0, 1);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1i(GLint location, GLint v0) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, &v0, sizeof(v0));
    if (gl4metalResourceTraceEnabled()) {
        gl4metalTraceUniformUpdateInt(location, "1i", &v0, 1);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2f(GLint location, GLfloat v0, GLfloat v1) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[2] = { v0, v1 };
    uniform2fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[2])];
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2i(GLint location, GLint v0, GLint v1) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLint values[2] = { v0, v1 };
    uniform2iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[3] = { v0, v1, v2 };
    uniform3fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[3])];
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3i(GLint location, GLint v0, GLint v1, GLint v2) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLint values[3] = { v0, v1, v2 };
    uniform3iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[4] = { v0, v1, v2, v3 };
    uniform4fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[4])];
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalTraceUniformUpdate(location, "4f", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

// The LWJGL/Minecraft path may use the vector upload entry points instead of
// the scalar convenience calls. Keep the first vector in the same uniform
// store consumed by the Metal resource bridge.
void APIENTRY glUniform2fv(GLint location, GLsizei count, const GLfloat *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    GLfloat values[2] = { value[0], value[1] };
    uniform2fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[2])];
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 2 * sizeof(GLfloat));
    gl4metalTraceUniformUpdate(location, "2fv", values, 2);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3fv(GLint location, GLsizei count, const GLfloat *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    GLfloat values[3] = { value[0], value[1], value[2] };
    uniform3fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[3])];
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 3 * sizeof(GLfloat));
    gl4metalTraceUniformUpdate(location, "3fv", values, 3);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4fv(GLint location, GLsizei count, const GLfloat *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    GLfloat values[4] = { value[0], value[1], value[2], value[3] };
    uniform4fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[4])];
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 4 * sizeof(GLfloat));
    gl4metalTraceUniformUpdate(location, "4fv", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLint values[4] = { v0, v1, v2, v3 };
    uniform4iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values));
    gl4metalTraceUniformUpdateInt(location, "4i", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2iv(GLint location, GLsizei count, const GLint *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform2iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 2 * sizeof(GLint));
    gl4metalTraceUniformUpdateInt(location, "2iv", value, 2);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3iv(GLint location, GLsizei count, const GLint *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform3iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 3 * sizeof(GLint));
    gl4metalTraceUniformUpdateInt(location, "3iv", value, 3);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4iv(GLint location, GLsizei count, const GLint *value) {
    if (count < 0 || (count > 0 && !value)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform4iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 4 * sizeof(GLint));
    gl4metalTraceUniformUpdateInt(location, "4iv", value, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1ui(GLint location, GLuint v0) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    uniform1iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, &v0, sizeof(v0));
    gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform2ui(GLint location, GLuint v0, GLuint v1) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    GLuint values[2] = {v0, v1}; uniform2iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values)); gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform3ui(GLint location, GLuint v0, GLuint v1, GLuint v2) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    GLuint values[3] = {v0, v1, v2}; uniform3iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values)); gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform4ui(GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3) {
    if (gl4metalUniformUpdateUnavailable(location)) return;
    GLuint values[4] = {v0, v1, v2, v3}; uniform4iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, values, sizeof(values)); gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform1uiv(GLint location, GLsizei count, const GLuint *value) {
    if (count < 0 || (count > 0 && !value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform1iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * sizeof(GLuint));
    gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform2uiv(GLint location, GLsizei count, const GLuint *value) {
    if (count < 0 || (count > 0 && !value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform2iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 2 * sizeof(GLuint));
    gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform3uiv(GLint location, GLsizei count, const GLuint *value) {
    if (count < 0 || (count > 0 && !value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform3iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 3 * sizeof(GLuint));
    gl4metalSetError(GL_NO_ERROR);
}
void APIENTRY glUniform4uiv(GLint location, GLsizei count, const GLuint *value) {
    if (count < 0 || (count > 0 && !value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    uniform4iValues[@(location)] = @(value[0]);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * 4 * sizeof(GLuint));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1fv(GLint location, GLsizei count, const GLfloat *value) {
    if (count < 0 || (count > 0 && value == NULL)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    // A scalar uniform uploaded with glUniform1fv(count=1) must retain a
    // scalar Objective-C type. Storing it as GLfloat[4] makes a later
    // getValue:size: with a 4-byte destination throw NSInvalidArgumentException.
    if (count == 1) {
        GLfloat scalar = value[0];
        uniform1fValues[@(location)] = [NSValue valueWithBytes:&scalar objCType:@encode(GLfloat)];
    } else {
        // The current runtime only materializes scalar uniform buffers. Keep
        // the first element rather than fabricating a scalar value with an
        // incompatible NSValue type; array uniforms can be handled separately.
        GLfloat scalar = value[0];
        uniform1fValues[@(location)] = [NSValue valueWithBytes:&scalar objCType:@encode(GLfloat)];
    }
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * sizeof(GLfloat));
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1iv(GLint location, GLsizei count, const GLint *value) {
    if (count < 0 || (count > 0 && value == NULL)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    GLint v0 = count > 0 ? value[0] : 0;
    uniform1iValues[@(location)] = @(v0);
    gl4metalStoreUniformRaw(location, value, (NSUInteger)count * sizeof(GLint));
    if (gl4metalResourceTraceEnabled()) {
        gl4metalTraceUniformUpdateInt(location, "1iv", &v0, 1);
    }
    gl4metalSetError(GL_NO_ERROR);
}

static void gl4metalSetUniformMatrix(GLint location, GLsizei count, GLboolean transpose,
                                     const GLfloat *value, NSUInteger columns, NSUInteger rows) {
    if (count < 0 || (count > 0 && !value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (count == 0 || gl4metalUniformUpdateUnavailable(location)) return;
    NSUInteger matrixElements = columns * rows;
    NSMutableData *canonical = [NSMutableData dataWithLength:(NSUInteger)count * matrixElements * sizeof(GLfloat)];
    GLfloat *destination = (GLfloat *)canonical.mutableBytes;
    for (GLsizei matrixIndex = 0; matrixIndex < count; ++matrixIndex) {
        const GLfloat *source = value + (NSUInteger)matrixIndex * matrixElements;
        GLfloat *matrix = destination + (NSUInteger)matrixIndex * matrixElements;
        if (transpose) {
            for (NSUInteger c = 0; c < columns; ++c)
                for (NSUInteger r = 0; r < rows; ++r)
                    matrix[c * rows + r] = source[r * columns + c];
        } else {
            memcpy(matrix, source, matrixElements * sizeof(GLfloat));
        }
    }
    uniformRawValues[@(location)] = canonical;
    NSString *currentName = uniformLocationNames[@(location)];
    if (columns == 4 && rows == 4 && count > 0 &&
        ([currentName isEqualToString:@"uMVP"] ||
         [currentName isEqualToString:@"uModelViewProjectionMatrix"] ||
         [currentName isEqualToString:@"uProjection"])) {
        gl4metalUpdateFallbackMVP(destination, 16);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 2, 2); }
void APIENTRY glUniformMatrix3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 3, 3); }
void APIENTRY glUniformMatrix4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 4, 4); }
void APIENTRY glUniformMatrix2x3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 2, 3); }
void APIENTRY glUniformMatrix3x2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 3, 2); }
void APIENTRY glUniformMatrix2x4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 2, 4); }
void APIENTRY glUniformMatrix4x2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 4, 2); }
void APIENTRY glUniformMatrix3x4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 3, 4); }
void APIENTRY glUniformMatrix4x3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { gl4metalSetUniformMatrix(l, n, t, v, 4, 3); }

void APIENTRY glDeleteProgram(GLuint program) {
    if (program == 0 || !programShaders[@(program)]) return;
    if (program == currentProgram) {
        if (!gl4metalPendingProgramDeletes)
            gl4metalPendingProgramDeletes = [NSMutableSet set];
        [gl4metalPendingProgramDeletes addObject:@(program)];
    } else {
        gl4metalDestroyProgramStorage(program);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteShader(GLuint shader) {
    if (shader == 0) return;

    // In OpenGL, deleting a shader that is attached to a program only marks
    // it for deletion. The shader object remains alive until it is detached
    // from every program. Pojav/Minecraft commonly deletes shaders shortly
    // after linking, so erasing shaderSources here caused: \"Program N
    // references missing shader M\" when the real Metal pipeline was created
    // later. Keep the compiled shader alive while any program references it.
    for (NSNumber *programId in programShaders) {
        NSArray<NSNumber *> *attached = programShaders[programId];
        if ([attached containsObject:@(shader)]) {
            if (!gl4metalPendingShaderDeletes)
                gl4metalPendingShaderDeletes = [NSMutableSet set];
            [gl4metalPendingShaderDeletes addObject:@(shader)];
            gl4metalSetError(GL_NO_ERROR);
            return;
        }
    }

    gl4metalDeleteShaderStorage(shader);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    auto it = shaderSources.find(shader);
    if (it == shaderSources.end()) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_SHADER_TYPE:
            *params = (GLint)it->second.type;
            break;
        case GL_COMPILE_STATUS:
            *params = it->second.compiled ? GL_TRUE : GL_FALSE;
            break;
        case GL_INFO_LOG_LENGTH: {
            NSString *message = shaderInfoLog[@(shader)] ?: @"";
            *params = message.length ? (GLint)strlen(message.UTF8String) + 1 : 0;
            break;
        }
        case GL_SHADER_SOURCE_LENGTH:
            *params = it->second.glslSource.empty()
                ? 0 : (GLint)it->second.glslSource.size() + 1;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

static GLenum gl4metalGLSLTypeToGLenum(NSString *type) {
    static NSDictionary<NSString *, NSNumber *> *types = nil;
    if (!types) {
        types = @{
            @"float": @(GL_FLOAT), @"vec2": @(GL_FLOAT_VEC2), @"vec3": @(GL_FLOAT_VEC3), @"vec4": @(GL_FLOAT_VEC4),
            @"int": @(GL_INT), @"ivec2": @(GL_INT_VEC2), @"ivec3": @(GL_INT_VEC3), @"ivec4": @(GL_INT_VEC4),
            @"uint": @(GL_UNSIGNED_INT), @"uvec2": @(GL_UNSIGNED_INT_VEC2), @"uvec3": @(GL_UNSIGNED_INT_VEC3), @"uvec4": @(GL_UNSIGNED_INT_VEC4),
            @"bool": @(GL_BOOL), @"bvec2": @(GL_BOOL_VEC2), @"bvec3": @(GL_BOOL_VEC3), @"bvec4": @(GL_BOOL_VEC4),
            @"mat2": @(GL_FLOAT_MAT2), @"mat3": @(GL_FLOAT_MAT3), @"mat4": @(GL_FLOAT_MAT4),
            @"mat2x3": @(GL_FLOAT_MAT2x3), @"mat2x4": @(GL_FLOAT_MAT2x4),
            @"mat3x2": @(GL_FLOAT_MAT3x2), @"mat3x4": @(GL_FLOAT_MAT3x4),
            @"mat4x2": @(GL_FLOAT_MAT4x2), @"mat4x3": @(GL_FLOAT_MAT4x3),
            @"sampler1D": @(GL_SAMPLER_1D), @"sampler2D": @(GL_SAMPLER_2D), @"sampler3D": @(GL_SAMPLER_3D),
            @"samplerCube": @(GL_SAMPLER_CUBE), @"sampler1DShadow": @(GL_SAMPLER_1D_SHADOW),
            @"sampler2DShadow": @(GL_SAMPLER_2D_SHADOW), @"samplerCubeShadow": @(GL_SAMPLER_CUBE_SHADOW),
            @"sampler1DArray": @(GL_SAMPLER_1D_ARRAY), @"sampler2DArray": @(GL_SAMPLER_2D_ARRAY),
            @"sampler1DArrayShadow": @(GL_SAMPLER_1D_ARRAY_SHADOW), @"sampler2DArrayShadow": @(GL_SAMPLER_2D_ARRAY_SHADOW),
            @"isampler2D": @(GL_INT_SAMPLER_2D), @"isampler3D": @(GL_INT_SAMPLER_3D), @"isamplerCube": @(GL_INT_SAMPLER_CUBE),
            @"usampler2D": @(GL_UNSIGNED_INT_SAMPLER_2D), @"usampler3D": @(GL_UNSIGNED_INT_SAMPLER_3D), @"usamplerCube": @(GL_UNSIGNED_INT_SAMPLER_CUBE)
        };
    }
    return types[type].unsignedIntValue;
}

static NSArray<NSDictionary<NSString *, id> *> *gl4metalProgramVariables(GLuint program, BOOL uniforms) {
    NSMutableString *allSources = [NSMutableString string];
    NSArray<NSNumber *> *attached = programShaders[@(program)];
    for (NSNumber *shaderID in attached) {
        auto shader = shaderSources.find(shaderID.unsignedIntValue);
        if (shader == shaderSources.end()) continue;
        if (!uniforms && shader->second.type != GL_VERTEX_SHADER) continue;
        [allSources appendFormat:@"\n%s\n", shader->second.glslSource.c_str()];
    }
    if (allSources.length == 0) return @[];

    NSRegularExpression *comments = [NSRegularExpression regularExpressionWithPattern:
        @"/\\*[\\s\\S]*?\\*/|//[^\\r\\n]*" options:0 error:nil];
    NSString *source = [comments stringByReplacingMatchesInString:allSources options:0
        range:NSMakeRange(0, allSources.length) withTemplate:@" "];
    NSString *storage = uniforms ? @"uniform" : @"(?:in|attribute)";
    NSString *pattern = [NSString stringWithFormat:
        @"(?:layout\\s*\\([^)]*\\)\\s*)?(?:flat\\s+|smooth\\s+|noperspective\\s+|centroid\\s+|sample\\s+)*"
         @"%@\\s+(?:(?:lowp|mediump|highp|readonly|writeonly|coherent|volatile|restrict)\\s+)*"
         @"([A-Za-z_][A-Za-z0-9_]*)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\[\\s*([0-9]+)\\s*\\])?\\s*;", storage];
    NSRegularExpression *declarations = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
    NSMutableArray<NSDictionary<NSString *, id> *> *variables = [NSMutableArray array];
    NSMutableSet<NSString *> *seenNames = [NSMutableSet set];
    NSArray<NSTextCheckingResult *> *matches = [declarations matchesInString:source options:0 range:NSMakeRange(0, source.length)];
    for (NSTextCheckingResult *match in matches) {
        NSString *typeName = [source substringWithRange:[match rangeAtIndex:1]];
        NSString *name = [source substringWithRange:[match rangeAtIndex:2]];
        if ([seenNames containsObject:name]) continue;
        NSString *identifierPattern = [NSString stringWithFormat:@"\\b%@\\b", [NSRegularExpression escapedPatternForString:name]];
        NSRegularExpression *identifier = [NSRegularExpression regularExpressionWithPattern:identifierPattern options:0 error:nil];
        NSUInteger occurrences = [identifier numberOfMatchesInString:source options:0 range:NSMakeRange(0, source.length)];
        NSUInteger declarationsOfName = 0;
        for (NSTextCheckingResult *candidate in matches) {
            if ([[source substringWithRange:[candidate rangeAtIndex:2]] isEqualToString:name]) declarationsOfName++;
        }
        if (occurrences <= declarationsOfName) continue;

        GLint size = 1;
        if ([match rangeAtIndex:3].location != NSNotFound) {
            size = MAX(1, [[source substringWithRange:[match rangeAtIndex:3]] intValue]);
        }
        NSString *activeName = size > 1 ? [name stringByAppendingString:@"[0]"] : name;
        [variables addObject:@{@"name": activeName, @"type": @(gl4metalGLSLTypeToGLenum(typeName)), @"size": @(size)}];
        [seenNames addObject:name];
    }
    return variables;
}

static NSArray<NSDictionary<NSString *, id> *> *gl4metalProgramFragmentOutputs(GLuint program) {
    NSMutableString *allSources = [NSMutableString string];
    for (NSNumber *shaderID in programShaders[@(program)]) {
        auto shader = shaderSources.find(shaderID.unsignedIntValue);
        if (shader != shaderSources.end() && shader->second.type == GL_FRAGMENT_SHADER)
            [allSources appendFormat:@"\n%s\n", shader->second.glslSource.c_str()];
    }
    if (allSources.length == 0) return @[];
    NSRegularExpression *comments = [NSRegularExpression regularExpressionWithPattern:
        @"/\\*[\\s\\S]*?\\*/|//[^\\r\\n]*" options:0 error:nil];
    NSString *source = [comments stringByReplacingMatchesInString:allSources options:0
        range:NSMakeRange(0, allSources.length) withTemplate:@" "];
    NSString *pattern = @"(?:layout\\s*\\(([^)]*)\\)\\s*)?(?:flat\\s+|smooth\\s+|noperspective\\s+|centroid\\s+|sample\\s+)*"
        @"out\\s+(?:(?:lowp|mediump|highp)\\s+)*[A-Za-z_][A-Za-z0-9_]*\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\[\\s*([0-9]+)\\s*\\])?\\s*;";
    NSRegularExpression *declarations = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
    NSArray<NSTextCheckingResult *> *matches = [declarations matchesInString:source options:0 range:NSMakeRange(0, source.length)];
    NSMutableArray *outputs = [NSMutableArray array];
    NSMutableSet *seen = [NSMutableSet set];
    for (NSTextCheckingResult *match in matches) {
        NSString *name = [source substringWithRange:[match rangeAtIndex:2]];
        if ([seen containsObject:name]) continue;
        NSString *escaped = [NSRegularExpression escapedPatternForString:name];
        NSRegularExpression *identifier = [NSRegularExpression regularExpressionWithPattern:
            [NSString stringWithFormat:@"\\b%@\\b", escaped] options:0 error:nil];
        if ([identifier numberOfMatchesInString:source options:0 range:NSMakeRange(0, source.length)] < 2) continue;
        GLint location = -1;
        if ([match rangeAtIndex:1].location != NSNotFound) {
            NSString *layout = [source substringWithRange:[match rangeAtIndex:1]];
            NSRegularExpression *locationPattern = [NSRegularExpression regularExpressionWithPattern:
                @"location\\s*=\\s*([0-9]+)" options:NSRegularExpressionCaseInsensitive error:nil];
            NSTextCheckingResult *locationMatch = [locationPattern firstMatchInString:layout options:0 range:NSMakeRange(0, layout.length)];
            if (locationMatch) location = [[layout substringWithRange:[locationMatch rangeAtIndex:1]] intValue];
        }
        [outputs addObject:@{@"name": name, @"location": @(location)}];
        [seen addObject:name];
    }
    return outputs;
}

void APIENTRY glBindFragDataLocation(GLuint program, GLuint colorNumber, const GLchar *name) {
    if (!name || !programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (colorNumber >= 1) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSString *outputName = [NSString stringWithUTF8String:name];
    if (!outputName) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSMutableDictionary *bindings = programFragDataBindings[@(program)];
    if (!bindings) {
        bindings = [NSMutableDictionary dictionary];
        programFragDataBindings[@(program)] = bindings;
    }
    bindings[outputName] = @(colorNumber);
    gl4metalSetError(GL_NO_ERROR);
}

GLint APIENTRY glGetFragDataLocation(GLuint program, const GLchar *name) {
    if (!name || !programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return -1; }
    if (![programLinkStatus[@(program)] boolValue]) { gl4metalSetError(GL_INVALID_OPERATION); return -1; }
    NSString *outputName = [NSString stringWithUTF8String:name];
    if (!outputName) { gl4metalSetError(GL_INVALID_VALUE); return -1; }
    NSArray *outputs = gl4metalProgramFragmentOutputs(program);
    for (NSUInteger i = 0; i < outputs.count; ++i) {
        NSDictionary *output = outputs[i];
        if ([output[@"name"] isEqualToString:outputName]) {
            NSNumber *bound = programFragDataBindings[@(program)][outputName];
            GLint location = bound ? bound.intValue : [output[@"location"] intValue];
            if (location < 0 && i == 0) location = 0;
            gl4metalSetError(GL_NO_ERROR);
            return location < 1 ? location : -1;
        }
    }
    gl4metalSetError(GL_NO_ERROR);
    return -1;
}

void APIENTRY glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber,
                                            GLuint index, const GLchar *name) {
    if (!name || !programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (colorNumber >= 1 || index > 1) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSString *outputName = [NSString stringWithUTF8String:name];
    if (!outputName) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSMutableDictionary *locations = programFragDataBindings[@(program)];
    if (!locations) { locations = [NSMutableDictionary dictionary]; programFragDataBindings[@(program)] = locations; }
    NSMutableDictionary *indices = programFragDataIndices[@(program)];
    if (!indices) { indices = [NSMutableDictionary dictionary]; programFragDataIndices[@(program)] = indices; }
    locations[outputName] = @(colorNumber);
    indices[outputName] = @(index);
    gl4metalSetError(GL_NO_ERROR);
}

GLint APIENTRY glGetFragDataIndex(GLuint program, const GLchar *name) {
    if (!name || !programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return -1; }
    if (![programLinkStatus[@(program)] boolValue]) { gl4metalSetError(GL_INVALID_OPERATION); return -1; }
    NSString *outputName = [NSString stringWithUTF8String:name];
    if (!outputName) { gl4metalSetError(GL_INVALID_VALUE); return -1; }
    for (NSDictionary *output in gl4metalProgramFragmentOutputs(program)) {
        if ([output[@"name"] isEqualToString:outputName]) {
            NSNumber *index = programFragDataIndices[@(program)][outputName];
            gl4metalSetError(GL_NO_ERROR);
            return index ? index.intValue : 0;
        }
    }
    gl4metalSetError(GL_NO_ERROR);
    return -1;
}

static void gl4metalCopyActiveVariableName(NSString *name, GLsizei bufSize, GLsizei *length, GLchar *outName) {
    if (length) *length = 0;
    if (bufSize <= 0) return;
    size_t copyLength = MIN([name lengthOfBytesUsingEncoding:NSUTF8StringEncoding], (NSUInteger)(bufSize - 1));
    if (!outName) return;
    NSData *utf8 = [name dataUsingEncoding:NSUTF8StringEncoding];
    if (copyLength) memcpy(outName, utf8.bytes, copyLength);
    outName[copyLength] = '\0';
    if (length) *length = (GLsizei)copyLength;
}

void APIENTRY glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    if (bufSize < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (![programLinkStatus[@(program)] boolValue]) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    NSArray *variables = gl4metalProgramVariables(program, NO);
    if (index >= variables.count) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSDictionary *variable = variables[index];
    if (size) *size = [variable[@"size"] intValue];
    if (type) *type = [variable[@"type"] unsignedIntValue];
    gl4metalCopyActiveVariableName(variable[@"name"], bufSize, length, name);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    if (bufSize < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (![programLinkStatus[@(program)] boolValue]) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    NSArray *variables = gl4metalProgramVariables(program, YES);
    if (index >= variables.count) { gl4metalSetError(GL_INVALID_VALUE); return; }
    NSDictionary *variable = variables[index];
    if (size) *size = [variable[@"size"] intValue];
    if (type) *type = [variable[@"type"] unsignedIntValue];
    gl4metalCopyActiveVariableName(variable[@"name"], bufSize, length, name);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    if (!programShaders[@(program)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_LINK_STATUS:
            params[0] = [programLinkStatus[@(program)] boolValue] ? GL_TRUE : GL_FALSE;
            break;
        case GL_VALIDATE_STATUS:
            params[0] = [programLinkStatus[@(program)] boolValue] ? GL_TRUE : GL_FALSE;
            break;
        case GL_INFO_LOG_LENGTH: {
            NSString *message = programInfoLog[@(program)] ?: @"";
            params[0] = message.length ? (GLint)strlen(message.UTF8String) + 1 : 0;
            break;
        }
        case GL_ATTACHED_SHADERS:
            params[0] = (GLint)[programShaders[@(program)] count];
            break;
        case GL_ACTIVE_UNIFORMS:
            params[0] = [programLinkStatus[@(program)] boolValue]
                ? (GLint)gl4metalProgramVariables(program, YES).count : 0;
            break;
        case GL_ACTIVE_ATTRIBUTES:
            params[0] = [programLinkStatus[@(program)] boolValue]
                ? (GLint)gl4metalProgramVariables(program, NO).count : 0;
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH: {
            NSUInteger maxLength = 0;
            for (NSDictionary *variable in ([programLinkStatus[@(program)] boolValue] ? gl4metalProgramVariables(program, YES) : @[]))
                maxLength = MAX(maxLength, [variable[@"name"] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
            params[0] = (GLint)maxLength;
            break;
        }
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: {
            NSUInteger maxLength = 0;
            for (NSDictionary *variable in ([programLinkStatus[@(program)] boolValue] ? gl4metalProgramVariables(program, NO) : @[]))
                maxLength = MAX(maxLength, [variable[@"name"] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
            params[0] = (GLint)maxLength;
            break;
        }
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (length) *length = 0;
    if (bufSize <= 0 || !infoLog) {
        if (bufSize < 0) gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSString *message = shaderInfoLog[@(shader)];
    if (!message) message = @"";

    const char *utf8 = [message UTF8String];
    size_t messageLength = utf8 ? strlen(utf8) : 0;
    size_t copyLength = MIN(messageLength, static_cast<size_t>(bufSize - 1));

    if (copyLength > 0) memcpy(infoLog, utf8, copyLength);
    infoLog[copyLength] = '\0';
    if (length) *length = static_cast<GLsizei>(copyLength);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (!infoLog) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSString *message = programInfoLog[@(program)];
    if (!message) message = @"";

    const char *utf8 = [message UTF8String];
    GLsizei copyLength = (GLsizei)MIN((size_t)bufSize - 1, strlen(utf8));
    if (copyLength > 0) {
        memcpy(infoLog, utf8, copyLength);
        infoLog[copyLength] = '\0';
    } else {
        infoLog[0] = '\0';
        copyLength = 0;
    }

    if (length) *length = copyLength;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetIntegerv(GLenum pname, GLint *data) {
    if (!data) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_VIEWPORT: {
            data[0] = ctx.viewport.x; data[1] = ctx.viewport.y;
            data[2] = ctx.viewport.width; data[3] = ctx.viewport.height;
            break;
        }
        case GL_SCISSOR_BOX: {
            data[0] = ctx.scissorRect.x; data[1] = ctx.scissorRect.y;
            data[2] = ctx.scissorRect.width; data[3] = ctx.scissorRect.height;
            break;
        }
        case GL_DEPTH_RANGE:
            data[0] = (GLint)ctx.depthRangeNear;
            data[1] = (GLint)ctx.depthRangeFar;
            break;
        case GL_CURRENT_PROGRAM:
            data[0] = (GLint)currentProgram;
            break;
        case GL_ARRAY_BUFFER_BINDING:
            data[0] = (GLint)gl4metalGetBoundBufferForTarget(GL_ARRAY_BUFFER);
            break;
        case GL_ELEMENT_ARRAY_BUFFER_BINDING:
            data[0] = (GLint)gl4metalGetBoundBufferForTarget(GL_ELEMENT_ARRAY_BUFFER);
            break;
        case GL_VERTEX_ARRAY_BINDING:
            data[0] = (GLint)ctx.currentVAO;
            break;
        case GL_ACTIVE_TEXTURE:
            data[0] = (GLint)(GL_TEXTURE0 + gl4metalGetActiveTextureUnit());
            break;
        case GL_TEXTURE_BINDING_2D:
            data[0] = (GLint)gl4metalGetTextureNameForUnit(gl4metalGetActiveTextureUnit());
            break;
        case GL_TEXTURE_BINDING_1D:
            data[0] = (GLint)gl4metalGetTextureNameForTarget(gl4metalGetActiveTextureUnit(), GL_TEXTURE_1D);
            break;
        case GL_MAX_VERTEX_ATTRIBS:
            data[0] = 16;
            break;
        case GL_MAX_TEXTURE_SIZE:
            data[0] = 16384;
            break;
        case GL_MAX_DRAW_BUFFERS:
            data[0] = 1;
            break;
        case GL_MAX_TEXTURE_IMAGE_UNITS:
            data[0] = 32;
            break;
        case GL_NUM_EXTENSIONS:
            data[0] = gl4metalGetExtensionCount();
            break;
        case GL_PACK_ALIGNMENT:
        case GL_PACK_ROW_LENGTH:
        case GL_PACK_SKIP_ROWS:
        case GL_PACK_SKIP_PIXELS:
        case GL_PACK_SWAP_BYTES:
        case GL_PACK_LSB_FIRST:
        case GL_UNPACK_ALIGNMENT:
        case GL_UNPACK_ROW_LENGTH:
        case GL_UNPACK_SKIP_ROWS:
        case GL_UNPACK_SKIP_PIXELS:
        case GL_UNPACK_SWAP_BYTES:
        case GL_UNPACK_LSB_FIRST:
            data[0] = gl4metalGetPixelStoreValue(pname);
            break;
        case GL_MAX_COLOR_ATTACHMENTS:
            data[0] = 1;
            break;
        case GL_FRAMEBUFFER_BINDING:
            data[0] = (GLint)ctx.currentFramebuffer;
            break;
        case GL_READ_FRAMEBUFFER_BINDING:
            data[0] = (GLint)ctx.currentReadFramebuffer;
            break;
        case GL_DRAW_BUFFER:
            data[0] = (GLint)ctx.drawBuffer;
            break;
        case GL_READ_BUFFER:
            data[0] = (GLint)ctx.readBuffer;
            break;
        case GL_DEPTH_FUNC:
            data[0] = (GLint)ctx.depthFunc;
            break;
        case GL_LOGIC_OP_MODE:
            data[0] = (GLint)gl4metalLogicOperation;
            break;
        case GL_CLAMP_VERTEX_COLOR:
            data[0] = (GLint)gl4metalClampVertexColor;
            break;
        case GL_CLAMP_FRAGMENT_COLOR:
            data[0] = (GLint)gl4metalClampFragmentColor;
            break;
        case GL_CLAMP_READ_COLOR:
            data[0] = (GLint)gl4metalClampReadColor;
            break;
        case GL_POINT_SIZE_MIN:
            data[0] = (GLint)gl4metalPointSizeMin;
            break;
        case GL_POINT_SIZE_MAX:
            data[0] = (GLint)gl4metalPointSizeMax;
            break;
        case GL_POINT_FADE_THRESHOLD_SIZE:
            data[0] = (GLint)gl4metalPointFadeThreshold;
            break;
        case GL_POINT_DISTANCE_ATTENUATION:
            data[0] = (GLint)gl4metalPointDistanceAttenuation[0];
            data[1] = (GLint)gl4metalPointDistanceAttenuation[1];
            data[2] = (GLint)gl4metalPointDistanceAttenuation[2];
            break;
        case GL_POINT_SPRITE_COORD_ORIGIN:
            data[0] = (GLint)gl4metalPointSpriteCoordOrigin;
            break;
        case GL_CULL_FACE_MODE:
            data[0] = (GLint)ctx.cullFace;
            break;
        case GL_FRONT_FACE:
            data[0] = (GLint)ctx.frontFace;
            break;
        case GL_STENCIL_CLEAR_VALUE:
            data[0] = ctx.clearStencil;
            break;
        case GL_STENCIL_FUNC:
            data[0] = (GLint)ctx.stencilFunc;
            break;
        case GL_STENCIL_REF:
            data[0] = ctx.stencilRef;
            break;
        case GL_STENCIL_VALUE_MASK:
            data[0] = (GLint)ctx.stencilReadMask;
            break;
        case GL_STENCIL_WRITEMASK:
            data[0] = (GLint)ctx.stencilWriteMask;
            break;
        case GL_STENCIL_FAIL:
            data[0] = (GLint)ctx.stencilFail;
            break;
        case GL_STENCIL_PASS_DEPTH_FAIL:
            data[0] = (GLint)ctx.stencilDepthFail;
            break;
        case GL_STENCIL_PASS_DEPTH_PASS:
            data[0] = (GLint)ctx.stencilDepthPass;
            break;
        case GL_STENCIL_BACK_FUNC:
            data[0] = (GLint)ctx.stencilBackFunc;
            break;
        case GL_STENCIL_BACK_FAIL:
            data[0] = (GLint)ctx.stencilBackFail;
            break;
        case GL_STENCIL_BACK_PASS_DEPTH_FAIL:
            data[0] = (GLint)ctx.stencilBackDepthFail;
            break;
        case GL_STENCIL_BACK_PASS_DEPTH_PASS:
            data[0] = (GLint)ctx.stencilBackDepthPass;
            break;
        case GL_STENCIL_BACK_REF:
            data[0] = ctx.stencilBackRef;
            break;
        case GL_STENCIL_BACK_VALUE_MASK:
            data[0] = (GLint)ctx.stencilBackReadMask;
            break;
        case GL_STENCIL_BACK_WRITEMASK:
            data[0] = (GLint)ctx.stencilBackWriteMask;
            break;
        case GL_BLEND_SRC:
        case GL_BLEND_SRC_RGB:
            data[0] = (GLint)ctx.blendSrcRGB;
            break;
        case GL_BLEND_DST:
        case GL_BLEND_DST_RGB:
            data[0] = (GLint)ctx.blendDstRGB;
            break;
        case GL_BLEND_SRC_ALPHA:
            data[0] = (GLint)ctx.blendSrcAlpha;
            break;
        case GL_BLEND_DST_ALPHA:
            data[0] = (GLint)ctx.blendDstAlpha;
            break;
        case GL_SAMPLE_COVERAGE_VALUE:
            data[0] = (GLint)ctx.sampleCoverageValue;
            break;
        case GL_SAMPLE_COVERAGE_INVERT:
            data[0] = (GLint)ctx.sampleCoverageInvert;
            break;
        case GL_PRIMITIVE_RESTART_INDEX:
            data[0] = (GLint)ctx.primitiveRestartIndex;
            break;
        case GL_MAJOR_VERSION:
            data[0] = 3;
            break;
        case GL_MINOR_VERSION:
            data[0] = 3;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetFloatv(GLenum pname, GLfloat *data) {
    if (!data) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_COLOR_CLEAR_VALUE:
            data[0] = ctx.clearColor.red; data[1] = ctx.clearColor.green;
            data[2] = ctx.clearColor.blue; data[3] = ctx.clearColor.alpha;
            break;
        case GL_DEPTH_CLEAR_VALUE:
            data[0] = (GLfloat)ctx.clearDepth;
            break;
        case GL_LINE_WIDTH:
            data[0] = ctx.lineWidth;
            break;
        case GL_ALIASED_LINE_WIDTH_RANGE:
        case GL_SMOOTH_LINE_WIDTH_RANGE:
            data[0] = 1.0f;
            data[1] = 1.0f;
            break;
        case GL_SMOOTH_LINE_WIDTH_GRANULARITY:
            data[0] = 1.0f;
            break;
        case GL_POINT_SIZE:
            data[0] = ctx.pointSize;
            break;
        case GL_POINT_SIZE_MIN:
            data[0] = gl4metalPointSizeMin;
            break;
        case GL_POINT_SIZE_MAX:
            data[0] = gl4metalPointSizeMax;
            break;
        case GL_POINT_FADE_THRESHOLD_SIZE:
            data[0] = gl4metalPointFadeThreshold;
            break;
        case GL_POINT_DISTANCE_ATTENUATION:
            data[0] = gl4metalPointDistanceAttenuation[0];
            data[1] = gl4metalPointDistanceAttenuation[1];
            data[2] = gl4metalPointDistanceAttenuation[2];
            break;
        case GL_POINT_SPRITE_COORD_ORIGIN:
            data[0] = (GLfloat)gl4metalPointSpriteCoordOrigin;
            break;
        case GL_BLEND_COLOR:
            data[0] = ctx.blendConstantRed; data[1] = ctx.blendConstantGreen;
            data[2] = ctx.blendConstantBlue; data[3] = ctx.blendConstantAlpha;
            break;
        case GL_DEPTH_RANGE:
            data[0] = (GLfloat)ctx.depthRangeNear;
            data[1] = (GLfloat)ctx.depthRangeFar;
            break;
        case GL_SAMPLE_COVERAGE_VALUE:
            data[0] = ctx.sampleCoverageValue;
            break;
        case GL_SAMPLE_COVERAGE_INVERT:
            data[0] = (GLfloat)ctx.sampleCoverageInvert;
            break;
        case GL_PRIMITIVE_RESTART_INDEX:
            data[0] = (GLfloat)ctx.primitiveRestartIndex;
            break;
        case GL_VIEWPORT:
            data[0] = (GLfloat)ctx.viewport.x; data[1] = (GLfloat)ctx.viewport.y;
            data[2] = (GLfloat)ctx.viewport.width; data[3] = (GLfloat)ctx.viewport.height;
            break;
        case GL_SCISSOR_BOX:
            data[0] = (GLfloat)ctx.scissorRect.x; data[1] = (GLfloat)ctx.scissorRect.y;
            data[2] = (GLfloat)ctx.scissorRect.width; data[3] = (GLfloat)ctx.scissorRect.height;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBooleanv(GLenum pname, GLboolean *data) {
    if (!data) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_DEPTH_TEST:
            data[0] = ctx.depthTestEnabled;
            break;
        case GL_STENCIL_TEST:
            data[0] = ctx.stencilTestEnabled;
            break;
        case GL_BLEND:
            data[0] = ctx.blendEnabled;
            break;
        case GL_CULL_FACE:
            data[0] = ctx.cullEnabled;
            break;
        case GL_SCISSOR_TEST:
            data[0] = ctx.scissorTestEnabled;
            break;
        case GL_DEPTH_WRITEMASK:
            data[0] = ctx.depthWriteEnabled;
            break;
        case GL_SAMPLE_COVERAGE_VALUE:
            data[0] = ctx.sampleCoverageValue != 0.0f ? GL_TRUE : GL_FALSE;
            break;
        case GL_SAMPLE_COVERAGE_INVERT:
            data[0] = ctx.sampleCoverageInvert;
            break;
        case GL_PRIMITIVE_RESTART:
            data[0] = ctx.primitiveRestartEnabled;
            break;
        case GL_COLOR_LOGIC_OP:
            data[0] = gl4metalColorLogicOperationEnabled;
            break;
        case GL_COLOR_WRITEMASK:
            data[0] = ctx.colorMaskRed; data[1] = ctx.colorMaskGreen;
            data[2] = ctx.colorMaskBlue; data[3] = ctx.colorMaskAlpha;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetDoublev(GLenum pname, GLdouble *data) {
    if (!data) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    switch (pname) {
        case GL_COLOR_CLEAR_VALUE:
            data[0] = ctx.clearColor.red; data[1] = ctx.clearColor.green;
            data[2] = ctx.clearColor.blue; data[3] = ctx.clearColor.alpha;
            break;
        case GL_DEPTH_CLEAR_VALUE:
            data[0] = ctx.clearDepth;
            break;
        case GL_LINE_WIDTH:
            data[0] = ctx.lineWidth;
            break;
        case GL_ALIASED_LINE_WIDTH_RANGE:
        case GL_SMOOTH_LINE_WIDTH_RANGE:
            data[0] = 1.0;
            data[1] = 1.0;
            break;
        case GL_SMOOTH_LINE_WIDTH_GRANULARITY:
            data[0] = 1.0;
            break;
        case GL_POINT_SIZE:
            data[0] = ctx.pointSize;
            break;
        case GL_POINT_SIZE_MIN:
            data[0] = gl4metalPointSizeMin;
            break;
        case GL_POINT_SIZE_MAX:
            data[0] = gl4metalPointSizeMax;
            break;
        case GL_POINT_FADE_THRESHOLD_SIZE:
            data[0] = gl4metalPointFadeThreshold;
            break;
        case GL_POINT_DISTANCE_ATTENUATION:
            data[0] = gl4metalPointDistanceAttenuation[0];
            data[1] = gl4metalPointDistanceAttenuation[1];
            data[2] = gl4metalPointDistanceAttenuation[2];
            break;
        case GL_POINT_SPRITE_COORD_ORIGIN:
            data[0] = gl4metalPointSpriteCoordOrigin;
            break;
        case GL_BLEND_COLOR:
            data[0] = ctx.blendConstantRed; data[1] = ctx.blendConstantGreen;
            data[2] = ctx.blendConstantBlue; data[3] = ctx.blendConstantAlpha;
            break;
        case GL_DEPTH_RANGE:
            data[0] = ctx.depthRangeNear;
            data[1] = ctx.depthRangeFar;
            break;
        case GL_SAMPLE_COVERAGE_VALUE:
            data[0] = ctx.sampleCoverageValue;
            break;
        case GL_SAMPLE_COVERAGE_INVERT:
            data[0] = ctx.sampleCoverageInvert;
            break;
        case GL_PRIMITIVE_RESTART_INDEX:
            data[0] = ctx.primitiveRestartIndex;
            break;
        case GL_VIEWPORT:
            data[0] = ctx.viewport.x; data[1] = ctx.viewport.y;
            data[2] = ctx.viewport.width; data[3] = ctx.viewport.height;
            break;
        case GL_SCISSOR_BOX:
            data[0] = ctx.scissorRect.x; data[1] = ctx.scissorRect.y;
            data[2] = ctx.scissorRect.width; data[3] = ctx.scissorRect.height;
            break;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
    if (name == NULL || index >= 16) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!programShaders[@(program)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSMutableDictionary<NSNumber *, NSString *> *bindings = programAttribBindings[@(program)];
    if (!bindings) {
        bindings = [[NSMutableDictionary alloc] init];
        programAttribBindings[@(program)] = bindings;
    }

    bindings[@(index)] = [NSString stringWithUTF8String:name];
    gl4metalSetError(GL_NO_ERROR);
}

GLint APIENTRY glGetAttribLocation(GLuint program, const GLchar *name) {
    if (!ctx || name == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return -1;
    }
    if (!programShaders[@(program)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return -1;
    }
    if (![programLinkStatus[@(program)] boolValue]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    GLShader *vertexShader = nullptr;
    for (NSNumber *shaderID in programShaders[@(program)]) {
        auto shader = shaderSources.find(shaderID.unsignedIntValue);
        if (shader != shaderSources.end() && shader->second.type == GL_VERTEX_SHADER) {
            vertexShader = &shader->second;
            break;
        }
    }
    if (!vertexShader) { gl4metalSetError(GL_NO_ERROR); return -1; }

    NSString *source = [NSString stringWithUTF8String:vertexShader->glslSource.c_str()];
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:
        @"(?:layout\\s*\\(\\s*location\\s*=\\s*([0-9]+)[^)]*\\)\\s*)?"
         @"(?:flat\\s+|smooth\\s+|noperspective\\s+|centroid\\s+|sample\\s+)*"
         @"(?:in|attribute)\\s+(?:(?:lowp|mediump|highp)\\s+)?"
         @"([A-Za-z_][A-Za-z0-9_]*)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\[[^]]+\\])?\\s*;"
        options:0 error:nil];
    NSString *requestedName = [NSString stringWithUTF8String:name];
    NSMutableDictionary<NSNumber *, NSString *> *bindings = programAttribBindings[@(program)];
    NSMutableSet<NSNumber *> *usedLocations = [NSMutableSet set];
    NSUInteger nextImplicitLocation = 0;
    GLint result = -1;
    NSArray<NSTextCheckingResult *> *matches = [regex matchesInString:source options:0 range:NSMakeRange(0, source.length)];
    for (NSTextCheckingResult *match in matches) {
        NSString *type = [source substringWithRange:[match rangeAtIndex:2]];
        NSString *attributeName = [source substringWithRange:[match rangeAtIndex:3]];
        NSString *locationText = ([match rangeAtIndex:1].location == NSNotFound)
            ? nil : [source substringWithRange:[match rangeAtIndex:1]];
        NSUInteger location = NSNotFound;
        if (locationText) {
            location = (NSUInteger)locationText.integerValue;
        } else {
            for (NSNumber *index in bindings) {
                if ([bindings[index] isEqualToString:attributeName]) {
                    location = index.unsignedIntegerValue;
                    break;
                }
            }
        }
        if (location == NSNotFound) {
            while ([usedLocations containsObject:@(nextImplicitLocation)]) nextImplicitLocation++;
            location = nextImplicitLocation;
        }
        NSUInteger advance = [type isEqualToString:@"mat2"] ? 2 :
                             [type isEqualToString:@"mat3"] ? 3 :
                             [type isEqualToString:@"mat4"] ? 4 : 1;
        nextImplicitLocation = MAX(nextImplicitLocation, location + advance);
        for (NSUInteger slot = 0; slot < advance; ++slot) [usedLocations addObject:@(location + slot)];

        if (![attributeName isEqualToString:requestedName]) continue;
        // An input declaration optimized out of the linked program is not an
        // active attribute and therefore has no location.
        NSRegularExpression *identifier = [NSRegularExpression regularExpressionWithPattern:
            [NSString stringWithFormat:@"\\b%@\\b", [NSRegularExpression escapedPatternForString:attributeName]]
            options:0 error:nil];
        if ([identifier numberOfMatchesInString:source options:0 range:NSMakeRange(0, source.length)] > 1 && location < 16) {
            result = (GLint)location;
        }
        break;
    }
    gl4metalSetError(GL_NO_ERROR);
    return result;
}

void APIENTRY glVertexAttrib1f(GLuint index, GLfloat x) {
    if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentValue[0] = x;
    vao->attribs[index].currentValue[1] = 0.0f;
    vao->attribs[index].currentValue[2] = 0.0f;
    vao->attribs[index].currentValue[3] = 1.0f;
    vao->attribs[index].currentValueIsInteger = GL_FALSE;
    vao->attribs[index].currentValueIsUnsigned = GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttrib1fv(GLuint index, const GLfloat *v) {
    if (v == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glVertexAttrib1f(index, v[0]);
}

void APIENTRY glVertexAttrib2f(GLuint index, GLfloat x, GLfloat y) {
    if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentValue[0] = x;
    vao->attribs[index].currentValue[1] = y;
    vao->attribs[index].currentValue[2] = 0.0f;
    vao->attribs[index].currentValue[3] = 1.0f;
    vao->attribs[index].currentValueIsInteger = GL_FALSE;
    vao->attribs[index].currentValueIsUnsigned = GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttrib2fv(GLuint index, const GLfloat *v) {
    if (v == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glVertexAttrib2f(index, v[0], v[1]);
}

void APIENTRY glVertexAttrib3f(GLuint index, GLfloat x, GLfloat y, GLfloat z) {
    if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentValue[0] = x;
    vao->attribs[index].currentValue[1] = y;
    vao->attribs[index].currentValue[2] = z;
    vao->attribs[index].currentValue[3] = 1.0f;
    vao->attribs[index].currentValueIsInteger = GL_FALSE;
    vao->attribs[index].currentValueIsUnsigned = GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttrib3fv(GLuint index, const GLfloat *v) {
    if (v == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glVertexAttrib3f(index, v[0], v[1], v[2]);
}

void APIENTRY glVertexAttrib4f(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentValue[0] = x;
    vao->attribs[index].currentValue[1] = y;
    vao->attribs[index].currentValue[2] = z;
    vao->attribs[index].currentValue[3] = w;
    vao->attribs[index].currentValueIsInteger = GL_FALSE;
    vao->attribs[index].currentValueIsUnsigned = GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttrib4fv(GLuint index, const GLfloat *v) {
    if (v == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glVertexAttrib4f(index, v[0], v[1], v[2], v[3]);
}

// OpenGL exposes double, short, and integer overloads for generic current
// attributes. The current GL value is stored as floating point unless the
// explicit glVertexAttribI* entrypoint is used.
void APIENTRY glVertexAttrib1d(GLuint i, GLdouble x) { glVertexAttrib1f(i, (GLfloat)x); }
void APIENTRY glVertexAttrib1dv(GLuint i, const GLdouble *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib1d(i, v[0]); }
void APIENTRY glVertexAttrib1s(GLuint i, GLshort x) { glVertexAttrib1f(i, (GLfloat)x); }
void APIENTRY glVertexAttrib1sv(GLuint i, const GLshort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib1s(i, v[0]); }
void APIENTRY glVertexAttrib2d(GLuint i, GLdouble x, GLdouble y) { glVertexAttrib2f(i, (GLfloat)x, (GLfloat)y); }
void APIENTRY glVertexAttrib2dv(GLuint i, const GLdouble *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib2d(i, v[0], v[1]); }
void APIENTRY glVertexAttrib2s(GLuint i, GLshort x, GLshort y) { glVertexAttrib2f(i, (GLfloat)x, (GLfloat)y); }
void APIENTRY glVertexAttrib2sv(GLuint i, const GLshort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib2s(i, v[0], v[1]); }
void APIENTRY glVertexAttrib3d(GLuint i, GLdouble x, GLdouble y, GLdouble z) { glVertexAttrib3f(i, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void APIENTRY glVertexAttrib3dv(GLuint i, const GLdouble *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib3d(i, v[0], v[1], v[2]); }
void APIENTRY glVertexAttrib3s(GLuint i, GLshort x, GLshort y, GLshort z) { glVertexAttrib3f(i, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void APIENTRY glVertexAttrib3sv(GLuint i, const GLshort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib3s(i, v[0], v[1], v[2]); }
void APIENTRY glVertexAttrib4d(GLuint i, GLdouble x, GLdouble y, GLdouble z, GLdouble w) { glVertexAttrib4f(i, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
void APIENTRY glVertexAttrib4dv(GLuint i, const GLdouble *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4d(i, v[0], v[1], v[2], v[3]); }
void APIENTRY glVertexAttrib4s(GLuint i, GLshort x, GLshort y, GLshort z, GLshort w) { glVertexAttrib4f(i, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
void APIENTRY glVertexAttrib4sv(GLuint i, const GLshort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4s(i, v[0], v[1], v[2], v[3]); }
void APIENTRY glVertexAttrib4bv(GLuint i, const GLbyte *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4f(i, v[0], v[1], v[2], v[3]); }
void APIENTRY glVertexAttrib4iv(GLuint i, const GLint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4f(i, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
void APIENTRY glVertexAttrib4ubv(GLuint i, const GLubyte *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4f(i, v[0], v[1], v[2], v[3]); }
void APIENTRY glVertexAttrib4uiv(GLuint i, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4f(i, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
void APIENTRY glVertexAttrib4usv(GLuint i, const GLushort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttrib4f(i, v[0], v[1], v[2], v[3]); }

static GLfloat gl4metalNormalizeSigned(GLint value, GLdouble maximum) {
    return (GLfloat)fmax(-1.0, (GLdouble)value / maximum);
}
void APIENTRY glVertexAttrib4Nbv(GLuint i, const GLbyte *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttrib4f(i, gl4metalNormalizeSigned(v[0], SCHAR_MAX), gl4metalNormalizeSigned(v[1], SCHAR_MAX), gl4metalNormalizeSigned(v[2], SCHAR_MAX), gl4metalNormalizeSigned(v[3], SCHAR_MAX));
}
void APIENTRY glVertexAttrib4Nsv(GLuint i, const GLshort *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttrib4f(i, gl4metalNormalizeSigned(v[0], SHRT_MAX), gl4metalNormalizeSigned(v[1], SHRT_MAX), gl4metalNormalizeSigned(v[2], SHRT_MAX), gl4metalNormalizeSigned(v[3], SHRT_MAX));
}
void APIENTRY glVertexAttrib4Niv(GLuint i, const GLint *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttrib4f(i, gl4metalNormalizeSigned(v[0], INT_MAX), gl4metalNormalizeSigned(v[1], INT_MAX), gl4metalNormalizeSigned(v[2], INT_MAX), gl4metalNormalizeSigned(v[3], INT_MAX));
}
void APIENTRY glVertexAttrib4Nub(GLuint i, GLubyte x, GLubyte y, GLubyte z, GLubyte w) {
    glVertexAttrib4f(i, x / 255.0f, y / 255.0f, z / 255.0f, w / 255.0f);
}
void APIENTRY glVertexAttrib4Nubv(GLuint i, const GLubyte *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttrib4Nub(i, v[0], v[1], v[2], v[3]);
}
void APIENTRY glVertexAttrib4Nusv(GLuint i, const GLushort *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttrib4f(i, v[0] / 65535.0f, v[1] / 65535.0f, v[2] / 65535.0f, v[3] / 65535.0f);
}
void APIENTRY glVertexAttrib4Nuiv(GLuint i, const GLuint *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    const GLdouble maximum = 4294967295.0;
    glVertexAttrib4f(i, (GLfloat)(v[0] / maximum), (GLfloat)(v[1] / maximum), (GLfloat)(v[2] / maximum), (GLfloat)(v[3] / maximum));
}

static void gl4metalSetIntegerCurrentAttrib(GLuint index, GLint size,
                                             const GLint *signedValues,
                                             const GLuint *unsignedValues) {
    if (index >= 16) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    gl4metalVertexAttrib *attrib = &vao->attribs[index];
    for (GLint component = 0; component < 4; ++component) {
        GLint defaultValue = component == 3 ? 1 : 0;
        GLuint defaultUnsignedValue = component == 3 ? 1u : 0u;
        if (unsignedValues) {
            attrib->currentUIntValue[component] = component < size ? unsignedValues[component] : defaultUnsignedValue;
        } else {
            attrib->currentUIntValue[component] = (GLuint)(component < size ? signedValues[component] : defaultValue);
        }
        if (signedValues) {
            attrib->currentIntValue[component] = component < size ? signedValues[component] : defaultValue;
        } else {
            attrib->currentIntValue[component] = (GLint)(component < size ? unsignedValues[component] : defaultUnsignedValue);
        }
    }
    attrib->currentValueIsInteger = GL_TRUE;
    attrib->currentValueIsUnsigned = unsignedValues ? GL_TRUE : GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribI1i(GLuint i, GLint x) { gl4metalSetIntegerCurrentAttrib(i, 1, &x, NULL); }
void APIENTRY glVertexAttribI1ui(GLuint i, GLuint x) { gl4metalSetIntegerCurrentAttrib(i, 1, NULL, &x); }
void APIENTRY glVertexAttribI2i(GLuint i, GLint x, GLint y) { GLint v[2] = {x,y}; gl4metalSetIntegerCurrentAttrib(i, 2, v, NULL); }
void APIENTRY glVertexAttribI2ui(GLuint i, GLuint x, GLuint y) { GLuint v[2] = {x,y}; gl4metalSetIntegerCurrentAttrib(i, 2, NULL, v); }
void APIENTRY glVertexAttribI3i(GLuint i, GLint x, GLint y, GLint z) { GLint v[3] = {x,y,z}; gl4metalSetIntegerCurrentAttrib(i, 3, v, NULL); }
void APIENTRY glVertexAttribI3ui(GLuint i, GLuint x, GLuint y, GLuint z) { GLuint v[3] = {x,y,z}; gl4metalSetIntegerCurrentAttrib(i, 3, NULL, v); }
void APIENTRY glVertexAttribI1iv(GLuint i, const GLint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 1, v, NULL); }
void APIENTRY glVertexAttribI1uiv(GLuint i, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 1, NULL, v); }
void APIENTRY glVertexAttribI2iv(GLuint i, const GLint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 2, v, NULL); }
void APIENTRY glVertexAttribI2uiv(GLuint i, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 2, NULL, v); }
void APIENTRY glVertexAttribI3iv(GLuint i, const GLint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 3, v, NULL); }
void APIENTRY glVertexAttribI3uiv(GLuint i, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } gl4metalSetIntegerCurrentAttrib(i, 3, NULL, v); }
void APIENTRY glVertexAttribI4bv(GLuint i, const GLbyte *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } GLint wide[4] = {v[0],v[1],v[2],v[3]}; gl4metalSetIntegerCurrentAttrib(i, 4, wide, NULL); }
void APIENTRY glVertexAttribI4sv(GLuint i, const GLshort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } GLint wide[4] = {v[0],v[1],v[2],v[3]}; gl4metalSetIntegerCurrentAttrib(i, 4, wide, NULL); }
void APIENTRY glVertexAttribI4ubv(GLuint i, const GLubyte *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } GLuint wide[4] = {v[0],v[1],v[2],v[3]}; gl4metalSetIntegerCurrentAttrib(i, 4, NULL, wide); }
void APIENTRY glVertexAttribI4usv(GLuint i, const GLushort *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } GLuint wide[4] = {v[0],v[1],v[2],v[3]}; gl4metalSetIntegerCurrentAttrib(i, 4, NULL, wide); }

void APIENTRY glVertexAttribI4i(GLuint index, GLint x, GLint y, GLint z, GLint w) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentIntValue[0] = x;
    vao->attribs[index].currentIntValue[1] = y;
    vao->attribs[index].currentIntValue[2] = z;
    vao->attribs[index].currentIntValue[3] = w;
    vao->attribs[index].currentValueIsInteger = GL_TRUE;
    vao->attribs[index].currentValueIsUnsigned = GL_FALSE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribI4ui(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    vao->attribs[index].currentUIntValue[0] = x;
    vao->attribs[index].currentUIntValue[1] = y;
    vao->attribs[index].currentUIntValue[2] = z;
    vao->attribs[index].currentUIntValue[3] = w;
    vao->attribs[index].currentValueIsInteger = GL_TRUE;
    vao->attribs[index].currentValueIsUnsigned = GL_TRUE;
    gl4metalUpdateConstantVertexAttribute(index);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribI4iv(GLuint index, const GLint *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttribI4i(index, v[0], v[1], v[2], v[3]);
}

void APIENTRY glVertexAttribI4uiv(GLuint index, const GLuint *v) {
    if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glVertexAttribI4ui(index, v[0], v[1], v[2], v[3]);
}

static GLint gl4metalSignExtendPacked(GLuint value, GLuint width) {
    GLuint shift = 32u - width;
    return (GLint)(value << shift) >> shift;
}

static void gl4metalVertexAttribPacked(GLuint index, GLenum type, GLboolean normalized,
                                       GLuint packed, NSUInteger components) {
    if (type != GL_INT_2_10_10_10_REV && type != GL_UNSIGNED_INT_2_10_10_10_REV) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    BOOL isSigned = type == GL_INT_2_10_10_10_REV;
    const GLuint widths[4] = {10, 10, 10, 2};
    const GLuint shifts[4] = {0, 10, 20, 30};
    GLfloat value[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (NSUInteger component = 0; component < components; ++component) {
        GLuint width = widths[component];
        GLuint mask = (1u << width) - 1u;
        GLuint raw = (packed >> shifts[component]) & mask;
        if (isSigned) {
            GLint signedValue = gl4metalSignExtendPacked(raw, width);
            value[component] = normalized
                ? (GLfloat)fmax(-1.0, (GLdouble)signedValue / (GLdouble)((1u << (width - 1)) - 1u))
                : (GLfloat)signedValue;
        } else {
            value[component] = normalized
                ? (GLfloat)((GLdouble)raw / (GLdouble)mask)
                : (GLfloat)raw;
        }
    }
    glVertexAttrib4f(index, value[0], value[1], value[2], value[3]);
}

void APIENTRY glVertexAttribP1ui(GLuint i, GLenum t, GLboolean n, GLuint v) { gl4metalVertexAttribPacked(i,t,n,v,1); }
void APIENTRY glVertexAttribP2ui(GLuint i, GLenum t, GLboolean n, GLuint v) { gl4metalVertexAttribPacked(i,t,n,v,2); }
void APIENTRY glVertexAttribP3ui(GLuint i, GLenum t, GLboolean n, GLuint v) { gl4metalVertexAttribPacked(i,t,n,v,3); }
void APIENTRY glVertexAttribP4ui(GLuint i, GLenum t, GLboolean n, GLuint v) { gl4metalVertexAttribPacked(i,t,n,v,4); }
void APIENTRY glVertexAttribP1uiv(GLuint i, GLenum t, GLboolean n, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttribP1ui(i,t,n,v[0]); }
void APIENTRY glVertexAttribP2uiv(GLuint i, GLenum t, GLboolean n, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttribP2ui(i,t,n,v[0]); }
void APIENTRY glVertexAttribP3uiv(GLuint i, GLenum t, GLboolean n, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttribP3ui(i,t,n,v[0]); }
void APIENTRY glVertexAttribP4uiv(GLuint i, GLenum t, GLboolean n, const GLuint *v) { if (!v) { gl4metalSetError(GL_INVALID_VALUE); return; } glVertexAttribP4ui(i,t,n,v[0]); }

static BOOL gl4metalValidateUniformQuery(GLuint program, GLint location) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return NO; }
    if (!programShaders[@(program)]) { gl4metalSetError(GL_INVALID_VALUE); return NO; }
    if (![programLinkStatus[@(program)] boolValue]) { gl4metalSetError(GL_INVALID_OPERATION); return NO; }
    NSNumber *owner = uniformLocationPrograms[@(location)];
    if (location < 0 || !owner || owner.unsignedIntValue != program) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NO;
    }
    return YES;
}

static GLenum gl4metalUniformTypeForLocation(GLuint program, GLint location) {
    NSString *name = uniformLocationNames[@(location)];
    for (NSDictionary *variable in gl4metalProgramVariables(program, YES)) {
        NSString *activeName = variable[@"name"];
        if ([activeName isEqualToString:name] ||
            ([activeName hasSuffix:@"[0]"] && [[activeName substringToIndex:activeName.length - 3] isEqualToString:name])) {
            return [variable[@"type"] unsignedIntValue];
        }
    }
    return 0;
}

static BOOL gl4metalTypeIsUnsignedInteger(GLenum type) {
    return type == GL_UNSIGNED_INT || type == GL_UNSIGNED_INT_VEC2 ||
           type == GL_UNSIGNED_INT_VEC3 || type == GL_UNSIGNED_INT_VEC4;
}
static BOOL gl4metalTypeIsInteger(GLenum type) {
    return type == GL_INT || type == GL_INT_VEC2 || type == GL_INT_VEC3 || type == GL_INT_VEC4 ||
           type == GL_BOOL || type == GL_BOOL_VEC2 || type == GL_BOOL_VEC3 || type == GL_BOOL_VEC4;
}

static void gl4metalGetUniformValues(GLuint program, GLint location, void *params, GLenum requestedType) {
    NSData *raw = uniformRawValues[@(location)];
    if (raw.length > 0) {
        GLenum storedType = gl4metalUniformTypeForLocation(program, location);
        NSUInteger count = raw.length / sizeof(uint32_t);
        if (gl4metalTypeIsInteger(storedType) || gl4metalTypeIsUnsignedInteger(storedType)) {
            if (requestedType == GL_FLOAT) {
                GLfloat *out = (GLfloat *)params;
                if (gl4metalTypeIsUnsignedInteger(storedType)) {
                    const GLuint *in = (const GLuint *)raw.bytes;
                    for (NSUInteger i = 0; i < count; ++i) out[i] = (GLfloat)in[i];
                } else {
                    const GLint *in = (const GLint *)raw.bytes;
                    for (NSUInteger i = 0; i < count; ++i) out[i] = (GLfloat)in[i];
                }
            } else if (requestedType == GL_UNSIGNED_INT) {
                GLuint *out = (GLuint *)params;
                if (gl4metalTypeIsUnsignedInteger(storedType)) memcpy(out, raw.bytes, raw.length);
                else { const GLint *in = (const GLint *)raw.bytes; for (NSUInteger i = 0; i < count; ++i) out[i] = (GLuint)in[i]; }
            } else {
                GLint *out = (GLint *)params;
                if (gl4metalTypeIsUnsignedInteger(storedType)) { const GLuint *in = (const GLuint *)raw.bytes; for (NSUInteger i = 0; i < count; ++i) out[i] = (GLint)in[i]; }
                else memcpy(out, raw.bytes, raw.length);
            }
        } else if (requestedType == GL_FLOAT) {
            memcpy(params, raw.bytes, raw.length);
        } else if (requestedType == GL_UNSIGNED_INT) {
            const GLfloat *in = (const GLfloat *)raw.bytes; GLuint *out = (GLuint *)params;
            for (NSUInteger i = 0; i < count; ++i) out[i] = (GLuint)in[i];
        } else {
            const GLfloat *in = (const GLfloat *)raw.bytes; GLint *out = (GLint *)params;
            for (NSUInteger i = 0; i < count; ++i) out[i] = (GLint)in[i];
        }
        return;
    }
    NSValue *matrix = uniformMatrix4fvValues[@(location)];
    if (matrix) {
        NSUInteger encodedSize = 0;
        NSGetSizeAndAlignment(matrix.objCType, &encodedSize, NULL);
        NSUInteger count = encodedSize / sizeof(GLfloat);
        GLfloat *values = (GLfloat *)malloc(encodedSize);
        if (!values) return;
        [matrix getValue:values size:encodedSize];
        if (requestedType == GL_FLOAT) memcpy(params, values, encodedSize);
        else if (requestedType == GL_UNSIGNED_INT) {
            GLuint *out = (GLuint *)params;
            for (NSUInteger i = 0; i < count; ++i) out[i] = (GLuint)values[i];
        } else {
            GLint *out = (GLint *)params;
            for (NSUInteger i = 0; i < count; ++i) out[i] = (GLint)values[i];
        }
        free(values);
        return;
    }
    if (requestedType == GL_FLOAT) {
        GLfloat value = 0.0f;
        NSValue *scalar = uniform1fValues[@(location)];
        NSNumber *integer = uniform1iValues[@(location)];
        if (scalar) [scalar getValue:&value]; else if (integer) value = (GLfloat)integer.doubleValue;
        *(GLfloat *)params = value;
    } else if (requestedType == GL_UNSIGNED_INT) {
        *(GLuint *)params = [uniform1iValues[@(location)] unsignedIntValue];
    } else {
        *(GLint *)params = [uniform1iValues[@(location)] intValue];
    }
}

void APIENTRY glGetUniformfv(GLuint program, GLint location, GLfloat *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateUniformQuery(program, location)) return;
    gl4metalGetUniformValues(program, location, params, GL_FLOAT);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetUniformiv(GLuint program, GLint location, GLint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateUniformQuery(program, location)) return;
    gl4metalGetUniformValues(program, location, params, GL_INT);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetUniformuiv(GLuint program, GLint location, GLuint *params) {
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalValidateUniformQuery(program, location)) return;
    gl4metalGetUniformValues(program, location, params, GL_UNSIGNED_INT);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderSource(GLuint shader,
                                GLsizei bufSize,
                                GLsizei *length,
                                GLchar *source) {
    if (!source || bufSize < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    auto it = shaderSources.find(shader);
    const std::string *text = nullptr;
    if (it != shaderSources.end()) {
        text = &it->second.glslSource;
    }

    if (bufSize == 0) {
        if (length) {
            *length = 0;
        }
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    const char *utf8 = text ? text->c_str() : "";
    size_t textLength = strlen(utf8);

    size_t maxCopy = static_cast<size_t>(bufSize - 1);
    size_t copyLength = std::min(maxCopy, textLength);

    if (copyLength > 0) {
        memcpy(source, utf8, copyLength);
    }

    source[copyLength] = '\0';

    if (length) {
        *length = static_cast<GLsizei>(copyLength);
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary) {
    (void)program;
    if (bufSize < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (length) *length = 0;
    if (binaryFormat) *binaryFormat = 0;
    if (bufSize > 0 && !binary) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glProgramBinary(GLuint program, GLenum binaryFormat, const void *binary, GLsizei length) {
    (void)program; (void)binaryFormat; (void)binary;
    if (length < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glProgramParameteri(GLuint program, GLenum pname, GLint value) {
    (void)program; (void)pname; (void)value;
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glGetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint *params) {
    (void)pipeline; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glBindProgramPipeline(GLuint pipeline) {
    (void)pipeline;
    gl4metalSetError(GL_INVALID_OPERATION);
}

void APIENTRY glDeleteProgramPipelines(GLsizei n, const GLuint *pipelines) {
    (void)pipelines;
    if (n < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalSetError(GL_INVALID_OPERATION);
}

static NSValue *gl4metalSyncKey(GLsync sync) {
    return sync ? [NSValue valueWithPointer:(void *)sync] : nil;
}

static BOOL gl4metalSyncIsSignaled(id<MTLCommandBuffer> commandBuffer) {
    return commandBuffer.status == MTLCommandBufferStatusCompleted ||
           commandBuffer.status == MTLCommandBufferStatusError;
}

GLsync APIENTRY glFenceSync(GLenum condition, GLbitfield flags) {
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE || flags != 0) {
        gl4metalSetError(condition != GL_SYNC_GPU_COMMANDS_COMPLETE ? GL_INVALID_ENUM : GL_INVALID_VALUE);
        return NULL;
    }
    if (!ctx || !ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return NULL; }
    gl4metalEncodePendingClear();
    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = gl4metalCreateCommandBuffer();
    id<MTLCommandBuffer> commandBuffer = ctx.currentCommandBuffer;
    if (!commandBuffer) { gl4metalSetError(GL_OUT_OF_MEMORY); return NULL; }

    GLsync sync = (GLsync)malloc(1);
    if (!sync) { gl4metalSetError(GL_OUT_OF_MEMORY); return NULL; }
    [commandBuffer commit];
    ctx.currentCommandBuffer = nil;
    gl4metalSyncCommands[gl4metalSyncKey(sync)] = commandBuffer;
    gl4metalSetError(GL_NO_ERROR);
    return sync;
}

GLboolean APIENTRY glIsSync(GLsync sync) {
    if (!sync) return GL_FALSE;
    return gl4metalSyncCommands[gl4metalSyncKey(sync)] ? GL_TRUE : GL_FALSE;
}

GLenum APIENTRY glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    id<MTLCommandBuffer> commandBuffer = gl4metalSyncCommands[gl4metalSyncKey(sync)];
    if (!sync || !commandBuffer) { gl4metalSetError(GL_INVALID_VALUE); return GL_WAIT_FAILED; }
    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) { gl4metalSetError(GL_INVALID_VALUE); return GL_WAIT_FAILED; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return GL_WAIT_FAILED; }
    if (flags & GL_SYNC_FLUSH_COMMANDS_BIT) glFlush();
    if (gl4metalSyncIsSignaled(commandBuffer)) {
        gl4metalSetError(GL_NO_ERROR);
        return GL_CONDITION_SATISFIED;
    }
    if (timeout == 0) { gl4metalSetError(GL_NO_ERROR); return GL_TIMEOUT_EXPIRED; }
    if (timeout == GL_TIMEOUT_IGNORED) {
        [commandBuffer waitUntilCompleted];
    } else {
        struct timespec start = {0}, now = {0}, pause = {0, 100000};
        clock_gettime(CLOCK_MONOTONIC, &start);
        uint64_t endNS = (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec + timeout;
        for (;;) {
            if (gl4metalSyncIsSignaled(commandBuffer)) break;
            clock_gettime(CLOCK_MONOTONIC, &now);
            uint64_t nowNS = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
            if (nowNS >= endNS) { gl4metalSetError(GL_NO_ERROR); return GL_TIMEOUT_EXPIRED; }
            nanosleep(&pause, NULL);
        }
    }
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_WAIT_FAILED;
    }
    gl4metalSetError(GL_NO_ERROR);
    return GL_CONDITION_SATISFIED;
}

void APIENTRY glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    if (!sync || !gl4metalSyncCommands[gl4metalSyncKey(sync)]) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (flags != 0 || timeout != GL_TIMEOUT_IGNORED) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    // All work is submitted to one serial Metal command queue. The fence
    // command buffer was committed before later GL command buffers, so queue
    // order provides the server-side wait without blocking the CPU.
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetSynciv(GLsync sync, GLenum pname, GLsizei bufSize, GLsizei *length, GLint *values) {
    if (length) *length = 0;
    if (bufSize < 0 || (bufSize > 0 && !values)) { gl4metalSetError(GL_INVALID_VALUE); return; }
    id<MTLCommandBuffer> commandBuffer = gl4metalSyncCommands[gl4metalSyncKey(sync)];
    if (!sync || !commandBuffer) { gl4metalSetError(GL_INVALID_VALUE); return; }
    GLint value = 0;
    switch (pname) {
        case GL_OBJECT_TYPE: value = GL_SYNC_FENCE; break;
        case GL_SYNC_CONDITION: value = GL_SYNC_GPU_COMMANDS_COMPLETE; break;
        case GL_SYNC_FLAGS: value = 0; break;
        case GL_SYNC_STATUS: value = gl4metalSyncIsSignaled(commandBuffer) ? GL_SIGNALED : GL_UNSIGNALED; break;
        default: gl4metalSetError(GL_INVALID_ENUM); return;
    }
    if (bufSize > 0) {
        values[0] = value;
        if (length) *length = 1;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteSync(GLsync sync) {
    NSValue *key = gl4metalSyncKey(sync);
    if (!sync || !gl4metalSyncCommands[key]) { gl4metalSetError(GL_INVALID_VALUE); return; }
    [gl4metalSyncCommands removeObjectForKey:key];
    free((void *)sync);
    gl4metalSetError(GL_NO_ERROR);
}

static MTLPrimitiveType getMetalPrimitiveType(GLenum mode) {
    switch(mode) {
        case GL_TRIANGLES: return MTLPrimitiveTypeTriangle;
        case GL_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
        case GL_LINES: return MTLPrimitiveTypeLine;
        case GL_LINE_STRIP: return MTLPrimitiveTypeLineStrip;
        case GL_POINTS: return MTLPrimitiveTypePoint;
        default:
            printf("[gl4metal] WARNING: Unsupported GL mode 0x%X, defaulting to Triangles\n", mode);
            return MTLPrimitiveTypeTriangle;
    }
}

static size_t gl4metalVertexAttribElementSize(const gl4metalVertexAttrib *a) {
    if (!a || a->size <= 0) return 0;
    size_t scalar = 0;
    switch (a->type) {
        case GL_BYTE: case GL_UNSIGNED_BYTE: scalar = 1; break;
        case GL_SHORT: case GL_UNSIGNED_SHORT: scalar = 2; break;
        case GL_INT: case GL_UNSIGNED_INT: case GL_FLOAT: scalar = 4; break;
        default: return 0;
    }
    return scalar * (size_t)a->size;
}

static BOOL gl4metalValidateVertexAttribRange(const gl4metalVertexAttrib *a,
                                               id<MTLBuffer> buffer,
                                               NSUInteger firstVertex,
                                               NSUInteger vertexCount,
                                               NSUInteger instanceCount,
                                               NSUInteger attribIndex) {
    if (!a || !buffer || !a->enabled || a->boundVBO == 0) return YES;
    size_t elemSize = gl4metalVertexAttribElementSize(a);
    size_t stride = a->stride > 0 ? (size_t)a->stride : elemSize;
    size_t base = (size_t)a->pointerOffset;
    if (elemSize == 0 || stride == 0) {
        printf("[gl4metal][BOUNDS] attr[%lu] invalid element/stride (elem=%zu stride=%zu)\n", (unsigned long)attribIndex, elemSize, stride);
        return NO;
    }
    size_t logicalIndex = 0;
    if (a->divisor == 0) {
        if (vertexCount == 0) return YES;
        if (firstVertex > SIZE_MAX - (vertexCount - 1)) {
            printf("[gl4metal][BOUNDS] attr[%lu] vertex index overflow\n", (unsigned long)attribIndex);
            return NO;
        }
        logicalIndex = firstVertex + vertexCount - 1;
    } else {
        if (instanceCount == 0) return YES;
        logicalIndex = (instanceCount - 1) / a->divisor;
    }
    if (logicalIndex > (SIZE_MAX - base - elemSize) / stride) {
        printf("[gl4metal][BOUNDS] attr[%lu] arithmetic overflow\n", (unsigned long)attribIndex);
        return NO;
    }
    size_t end = base + logicalIndex * stride + elemSize;
    size_t length = (size_t)[buffer length];
    if (getenv("GL4METAL_BOUNDS_TRACE")) {
        printf("[gl4metal][BOUNDS] attr[%lu] check: VBO=%u base=%zu stride=%zu elem=%zu lastIndex=%zu end=%zu buffer=%zu divisor=%u\n",
               (unsigned long)attribIndex, a->boundVBO, base, stride, elemSize, logicalIndex, end, length, a->divisor);
    }
    if (end > length) {
        printf("[gl4metal][BOUNDS] attr[%lu] OOB: VBO=%u base=%zu stride=%zu elem=%zu lastIndex=%zu end=%zu buffer=%zu divisor=%u\n",
               (unsigned long)attribIndex, a->boundVBO, base, stride, elemSize,
               logicalIndex, end, length, a->divisor);
        return NO;
    }
    return YES;
}

static BOOL gl4metalReadIndexValue(id<MTLBuffer> buffer, GLenum type, size_t offset, NSUInteger i, NSUInteger *outValue) {
    if (!buffer || !outValue) return NO;
    size_t width = (type == GL_UNSIGNED_BYTE) ? 1 : (type == GL_UNSIGNED_SHORT ? 2 : (type == GL_UNSIGNED_INT ? 4 : 0));
    if (!width || i > (SIZE_MAX - offset) / width || offset + i * width + width > (size_t)[buffer length]) return NO;
    const uint8_t *p = (const uint8_t *)[buffer contents] + offset + i * width;
    if (type == GL_UNSIGNED_BYTE) *outValue = p[0];
    else if (type == GL_UNSIGNED_SHORT) { uint16_t v; memcpy(&v,p,2); *outValue=v; }
    else { uint32_t v; memcpy(&v,p,4); *outValue=v; }
    return YES;
}

// Collect index bounds in one pass. Samples and the diagnostic hash are only
// computed when index tracing is enabled; normal draws need only min/max.
static BOOL gl4metalFindIndexStats(id<MTLBuffer> indexBuffer, GLenum indexType, NSUInteger indexOffset, NSUInteger count,
                                   NSUInteger *outMinIndex, NSUInteger *outMaxIndex,
                                   NSUInteger sampleValues[8], NSUInteger *outSampleCount,
                                   uint64_t *outHash) {
    if (!indexBuffer || !outMinIndex || !outMaxIndex) return NO;
    NSUInteger minIndex = SIZE_MAX;
    NSUInteger maxIndex = 0;
    NSUInteger sampleCount = 0;
    BOOL collectDiagnostics = sampleValues || outSampleCount || outHash;
    uint64_t hash = 1469598103934665603ULL; // FNV-1a offset basis
    for (NSUInteger i = 0; i < count; ++i) {
        NSUInteger value = 0;
        if (!gl4metalReadIndexValue(indexBuffer, indexType, indexOffset, i, &value)) {
            printf("[gl4metal][BOUNDS] cannot inspect index %lu\n", (unsigned long)i);
            return NO;
        }
        if (value < minIndex) minIndex = value;
        if (value > maxIndex) maxIndex = value;
        if (collectDiagnostics && sampleValues && sampleCount < 8) sampleValues[sampleCount++] = value;
        if (collectDiagnostics) {
            hash ^= (uint64_t)value;
            hash *= 1099511628211ULL; // FNV-1a prime
        }
    }
    if (count == 0) minIndex = 0;
    *outMinIndex = minIndex;
    *outMaxIndex = maxIndex;
    if (outSampleCount) *outSampleCount = sampleCount;
    if (outHash) *outHash = hash;
    return YES;
}

static BOOL gl4metalFindMaxIndex(id<MTLBuffer> indexBuffer, GLenum indexType, NSUInteger indexOffset, NSUInteger count, NSUInteger *outMaxIndex) {
    return gl4metalFindIndexStats(indexBuffer, indexType, indexOffset, count, NULL, outMaxIndex, NULL, NULL, NULL);
}

static BOOL gl4metalValidateIndexedVertexAttribRange(const gl4metalVertexAttrib *a,
                                                      id<MTLBuffer> vertexBuffer,
                                                      NSUInteger maxIndex,
                                                      NSUInteger instanceCount,
                                                      NSUInteger attribIndex) {
    if (!a || !a->enabled || !vertexBuffer || a->boundVBO == 0) return YES;
    if (a->divisor != 0) return gl4metalValidateVertexAttribRange(a, vertexBuffer, 0, 0, instanceCount, attribIndex);
    return gl4metalValidateVertexAttribRange(a, vertexBuffer, maxIndex, 1, instanceCount, attribIndex);
}

static BOOL gl4metalValidateDrawArraysBuffers(gl4metalVertexArray *vao, NSUInteger first, NSUInteger count, NSUInteger instanceCount) {
    if (!vao) return YES;
    BOOL ok = YES;
    for (NSUInteger i=0;i<16;i++) {
        gl4metalVertexAttrib *a=&vao->attribs[i];
        if (!a->enabled || a->boundVBO==0) continue;
        id<MTLBuffer> b=bufferObjects[@(a->boundVBO)];
        if (!b || !gl4metalValidateVertexAttribRange(a,b,first,count,instanceCount,i)) ok=NO;
    }
    return ok;
}

static MTLIndexType gl4metalIndexTypeForElementType(GLenum type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:  return MTLIndexTypeUInt16;
        case GL_UNSIGNED_SHORT: return MTLIndexTypeUInt16;
        case GL_UNSIGNED_INT:   return MTLIndexTypeUInt32;
        default:                return MTLIndexTypeUInt16;
    }
}

static id<MTLBuffer> gl4metalFindCachedTriangleFanBuffer(NSString *key) {
    return key ? gl4metalTriangleFanBufferCache[key] : nil;
}

static void gl4metalCacheTriangleFanBuffer(NSString *key, id<MTLBuffer> buffer, NSUInteger length) {
    const NSUInteger cacheByteLimit = 16u * 1024u * 1024u;
    const NSUInteger cacheEntryLimit = 256;
    if (!key || !buffer || length > cacheByteLimit) return;
    if (!gl4metalTriangleFanBufferCache) {
        gl4metalTriangleFanBufferCache = [NSMutableDictionary dictionary];
    }
    if (gl4metalTriangleFanBufferCache.count >= cacheEntryLimit ||
        gl4metalTriangleFanBufferCacheBytes > cacheByteLimit - length) {
        [gl4metalTriangleFanBufferCache removeAllObjects];
        gl4metalTriangleFanBufferCacheBytes = 0;
    }
    gl4metalTriangleFanBufferCache[key] = buffer;
    gl4metalTriangleFanBufferCacheBytes += length;
}

static BOOL gl4metalMakeTriangleFanIndexBuffer(id<MTLBuffer> source,
                                                NSUInteger sourceOffset,
                                                GLsizei count,
                                                GLenum type,
                                                id<MTLBuffer> *outBuffer,
                                                MTLIndexType *outType) {
    if (!ctx || !ctx.device || !source || !outBuffer || !outType || count < 3) return NO;
    NSUInteger srcSize = (type == GL_UNSIGNED_BYTE) ? 1 :
                          (type == GL_UNSIGNED_SHORT) ? 2 :
                          (type == GL_UNSIGNED_INT) ? 4 : 0;
    if (srcSize == 0 || sourceOffset > [source length] ||
        (size_t)count > ((size_t)[source length] - sourceOffset) / srcSize) return NO;

    NSUInteger fanTriangles = (NSUInteger)count - 2;
    NSUInteger outCount = fanTriangles * 3;
    NSUInteger outIndexSize = (type == GL_UNSIGNED_INT) ? 4 : 2;
    if (fanTriangles > NSUIntegerMax / (3 * outIndexSize)) return NO;
    NSUInteger outBytes = outCount * outIndexSize;
    NSString *cacheKey = [NSString stringWithFormat:@"indexed:%p:%lu:%d:%u",
                          (void *)source, (unsigned long)sourceOffset, count, type];
    id<MTLBuffer> cached = gl4metalFindCachedTriangleFanBuffer(cacheKey);
    if (cached) {
        *outBuffer = cached;
        *outType = (type == GL_UNSIGNED_INT) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
        return YES;
    }
    id<MTLBuffer> out = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
    if (!out) return NO;

    const uint8_t *src = (const uint8_t *)source.contents + sourceOffset;
    if (outIndexSize == 2) {
        uint16_t *dst = (uint16_t *)out.contents;
        for (NSUInteger i = 0; i < fanTriangles; ++i) {
            NSUInteger i1 = i + 1, i2 = i + 2;
            uint32_t a = (type == GL_UNSIGNED_BYTE) ? src[0] : ((const uint16_t *)src)[0];
            uint32_t b = (type == GL_UNSIGNED_BYTE) ? src[i1] : ((const uint16_t *)src)[i1];
            uint32_t c = (type == GL_UNSIGNED_BYTE) ? src[i2] : ((const uint16_t *)src)[i2];
            dst[i * 3 + 0] = (uint16_t)a;
            dst[i * 3 + 1] = (uint16_t)b;
            dst[i * 3 + 2] = (uint16_t)c;
        }
        *outType = MTLIndexTypeUInt16;
    } else {
        const uint32_t *src32 = (const uint32_t *)src;
        uint32_t *dst = (uint32_t *)out.contents;
        for (NSUInteger i = 0; i < fanTriangles; ++i) {
            dst[i * 3 + 0] = src32[0];
            dst[i * 3 + 1] = src32[i + 1];
            dst[i * 3 + 2] = src32[i + 2];
        }
        *outType = MTLIndexTypeUInt32;
    }
    *outBuffer = out;
    gl4metalCacheTriangleFanBuffer(cacheKey, out, outBytes);
    return YES;
}

static BOOL gl4metalDrawTraceEnabledForProgram(GLuint program) {
    return program == 53 && getenv("GL4METAL_DRAW_TRACE");
}

static BOOL gl4metalIsValidDrawMode(GLenum mode) {
    switch (mode) {
        case GL_POINTS:
        case GL_LINES:
        case GL_LINE_LOOP:
        case GL_LINE_STRIP:
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY:
            return YES;
        default:
            return NO;
    }
}

static BOOL gl4metalModeIsImplemented(GLenum mode, BOOL indexed) {
    return mode == GL_POINTS || mode == GL_LINES || mode == GL_LINE_STRIP ||
           mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP ||
           (indexed && mode == GL_TRIANGLE_FAN);
}

static void gl4metalDrawArraysInternal(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    if (!gl4metalIsValidDrawMode(mode)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (first < 0 || count < 0 || instancecount < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!gl4metalModeIsImplemented(mode, NO)) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (ctx.currentFramebuffer != 0 &&
        glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }
    if (count == 0 || instancecount == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_BEGIN] kind=arrays program=%u vao=%u fbo=%u mode=0x%X first=%d count=%d instances=%d\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, first, count, instancecount);
        fflush(stdout);
    }
    if (gl4metalGPUIsolationEnabled && gl4metalGPUIsolationFailed) {
        printf("[gl4metal][GPU-DEBUG] draw skipped after previous GPU failure\n");
        return;
    }
    gl4metalVertexArray *preVAO = gl4metalGetCurrentVAO();
    if (preVAO) gl4metalRecoverPackedInterleavedOffsets(preVAO);
    if (!gl4metalValidateDrawArraysBuffers(preVAO, (NSUInteger)first, (NSUInteger)count, (NSUInteger)instancecount)) {
        printf("[gl4metal][BOUNDS] Aborting glDrawArrays: vertex buffer range invalid\n");
        return;
    }
    if (getenv("GL4METAL_DEBUG")) {
        printf("[gl4metal][DRAW] glDrawArrays mode=0x%X first=%d count=%d instances=%d program=%u vao=%u\n", mode, first, count, instancecount, currentProgram, ctx.currentVAO);
    }
    gl4metalDumpProgramResourceInfo(currentProgram);

    // Always resolve against the current GL state. The resolver is backed by
    // pipelineLayoutCache, so this is a cheap cache lookup on the hot path,
    // while preventing stale pipelines after program/VAO/FBO changes.
    id<MTLRenderPipelineState> activePipeline = gl4metalResolveCurrentPipeline();
    if (!activePipeline) {
        printf("[gl4metal] WARNING: No valid Metal pipeline for glDrawArrays\n");
        return;
    }
    ctx.pipelineState = activePipeline;

    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = gl4metalCreateCommandBuffer();
    if (ctx.currentFramebuffer == 0 && !ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
    if (ctx.currentFramebuffer == 0 && !ctx.currentDrawable) {
        printf("[gl4metal] ERROR: No drawable available for glDrawArrays\n");
        return;
    }

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();
    if (!renderPassDesc) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }

    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSUInteger targetWidth = 0;
    NSUInteger targetHeight = 0;
    if (ctx.currentFramebuffer != 0) {
        id<MTLTexture> target = gl4metalGetCurrentFramebufferColorTexture();
        targetWidth = target ? target.width : 0;
        targetHeight = target ? target.height : 0;
    } else {
        targetWidth = ctx.currentDrawable.texture.width;
        targetHeight = ctx.currentDrawable.texture.height;
    }
    applyViewportAndScissor(encoder, targetWidth, targetHeight);

    if (ctx.depthStencilState) {
        [encoder setDepthStencilState:ctx.depthStencilState];
    }
    if (currentProgram == 53 && getenv("GL4METAL_PROGRAM53_NO_DEPTH")) {
        id<MTLDepthStencilState> noDepthState = nil;
        MTLDepthStencilDescriptor *dds = [[MTLDepthStencilDescriptor alloc] init];
        dds.depthCompareFunction = MTLCompareFunctionAlways;
        dds.depthWriteEnabled = NO;
        noDepthState = [ctx.device newDepthStencilStateWithDescriptor:dds];
        if (noDepthState) [encoder setDepthStencilState:noDepthState];
    }
    [encoder setStencilFrontReferenceValue:(uint32_t)MAX(ctx.stencilRef, 0)
                      backReferenceValue:(uint32_t)MAX(ctx.stencilBackRef, 0)];

    applyRasterState(encoder);
    [encoder setRenderPipelineState:ctx.pipelineState];

    gl4metalVertexArray *currentVAO = vaoObjects[@(ctx.currentVAO)];
    if (currentVAO) {
        for (NSUInteger index = 0; index < 16; index++) {
            gl4metalVertexAttrib *attrib = &currentVAO->attribs[index];
            if (attrib->enabled && attrib->boundVBO != 0) {
                id<MTLBuffer> buffer = bufferObjects[@(attrib->boundVBO)];
                if (buffer) {
                    NSUInteger offset = 0;
                    if (!gl4metalValidateBufferOffset(buffer, attrib->pointerOffset, &offset)) {
                        continue;
                    }
                    gl4metalMarkBufferUsed(attrib->boundVBO);
                    [encoder setVertexBuffer:buffer offset:offset atIndex:index];
                    if (getenv("GL4METAL_VERTEX_BIND_TRACE")) {
                        printf("[gl4metal][VERTEX_BIND_TRACE] program=%u vao=%u attr=%lu vbo=%u offset=%lu stride=%d size=%d type=0x%X\n",
                               currentProgram, ctx.currentVAO, (unsigned long)index, attrib->boundVBO,
                               (unsigned long)offset, attrib->stride, attrib->size, attrib->type);
                    }
                }
            } else if ((!attrib->enabled || attrib->boundVBO == 0) &&
                       [inferredConstantVertexAttributes containsObject:@(index)]) {
                id<MTLBuffer> constantBuffer = constantVertexAttributeBuffers[@(index)];
                if (constantBuffer) {
                    [encoder setVertexBuffer:constantBuffer offset:0 atIndex:index];
                }
            }
        }
    }

    gl4metalBindProgramResources(encoder, currentProgram);

    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_READY] kind=arrays program=%u vao=%u fbo=%u mode=0x%X count=%d pipeline=%p encoder=%p target=%lux%lu\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, count,
               (void *)ctx.pipelineState, (void *)encoder,
               (unsigned long)targetWidth, (unsigned long)targetHeight);
        fflush(stdout);
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        NSUInteger triCount = (NSUInteger)count - 2;
        NSUInteger indexCount = triCount * 3;
        MTLIndexType indexType = ((NSUInteger)first + (NSUInteger)count <= 65535)
            ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        NSUInteger indexSize = (indexType == MTLIndexTypeUInt16) ? 2 : 4;
        NSUInteger indexBytes = indexCount * indexSize;
        NSString *cacheKey = [NSString stringWithFormat:@"arrays:%d:%d:%u", first, count, (unsigned)indexType];
        id<MTLBuffer> fan = gl4metalFindCachedTriangleFanBuffer(cacheKey);
        BOOL wasCached = fan != nil;
        if (!wasCached) fan = [ctx.device newBufferWithLength:indexBytes options:MTLResourceStorageModeShared];
        if (fan) {
            if (!wasCached && indexType == MTLIndexTypeUInt16) {
                uint16_t *dst = (uint16_t *)fan.contents;
                for (NSUInteger i = 0; i < triCount; ++i) {
                    dst[i * 3 + 0] = (uint16_t)first;
                    dst[i * 3 + 1] = (uint16_t)(first + (GLint)i + 1);
                    dst[i * 3 + 2] = (uint16_t)(first + (GLint)i + 2);
                }
            } else if (!wasCached) {
                uint32_t *dst = (uint32_t *)fan.contents;
                for (NSUInteger i = 0; i < triCount; ++i) {
                    dst[i * 3 + 0] = (uint32_t)first;
                    dst[i * 3 + 1] = (uint32_t)(first + (GLint)i + 1);
                    dst[i * 3 + 2] = (uint32_t)(first + (GLint)i + 2);
                }
            }
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                 indexCount:indexCount
                                  indexType:indexType
                                indexBuffer:fan
                          indexBufferOffset:0
                              instanceCount:(NSUInteger)instancecount];
            if (!wasCached) {
                gl4metalCacheTriangleFanBuffer(cacheKey, fan, indexBytes);
            }
        }
    } else {
        MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
        [encoder drawPrimitives:primitiveType vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count instanceCount:(NSUInteger)instancecount];
    }

    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_AFTER] kind=arrays program=%u vao=%u fbo=%u mode=0x%X count=%d pipeline=%p encoder=%p\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, count,
               (void *)ctx.pipelineState, (void *)encoder);
        fflush(stdout);
    }

    [encoder endEncoding];
    if (gl4metalGPUIsolationEnabled) {
        NSUInteger serial = ++gl4metalDrawSerial;
        if (!gl4metalFinishIsolatedDraw(ctx.currentCommandBuffer, serial, "glDrawArrays")) {
            ctx.currentCommandBuffer = nil;
            return;
        }
        ctx.currentCommandBuffer = nil;
    }
}


// -----------------------------------------------------------------------------
// Program resource binding
// Metal buffer(0..15) is reserved for vertex-input resources. GLSL uniforms
// are remapped by the shader translator to buffer(16+). Keep the data binding
// deterministic here as well. Texture/sampler slots remain 0-based.
// -----------------------------------------------------------------------------
static id<MTLTexture> gl4metalDummyTexture = nil;
static id<MTLSamplerState> gl4metalDummySampler = nil;

static GLint gl4metalFindUniformLocationByName(NSString *name, GLuint program) {
    if (!name || !uniformLocationNames || !uniformLocationPrograms || program == 0) return -1;
    for (NSNumber *key in uniformLocationNames) {
        NSNumber *owner = uniformLocationPrograms[key];
        NSString *candidate = uniformLocationNames[key];
        if (owner && owner.unsignedIntValue == program && [candidate isEqualToString:name]) {
            return key.intValue;
        }
    }
    return -1;
}

static BOOL gl4metalCopyNSValueBytes(NSValue *value, void *destination, NSUInteger destinationSize) {
    if (!value || !destination || destinationSize == 0) return NO;
    NSUInteger encodedSize = 0;
    NSGetSizeAndAlignment(value.objCType, &encodedSize, NULL);
    if (encodedSize == 0 || encodedSize > destinationSize) {
        return NO;
    }
    [value getValue:destination size:encodedSize];
    return YES;
}

static NSUInteger gl4metalAlignUniformOffset(NSUInteger value) {
    const NSUInteger alignment = 256;
    return (value + (alignment - 1)) & ~(alignment - 1);
}

static id<MTLBuffer> gl4metalUniformArenaAlloc(NSUInteger length, NSUInteger *outOffset) {
    if (!ctx || !ctx.device || length == 0 || !outOffset) return nil;

    const NSUInteger chunkSize = 64 * 1024;
    NSUInteger alignedLength = gl4metalAlignUniformOffset(length);

    if (!ctx.uniformArenaBuffers) {
        ctx.uniformArenaBuffers = [NSMutableArray array];
        ctx.uniformArenaOffset = 0;
    }

    NSUInteger offset = gl4metalAlignUniformOffset(ctx.uniformArenaOffset);
    id<MTLBuffer> arena = ctx.uniformArenaBuffers.lastObject;

    if (!arena || offset + alignedLength > arena.length) {
        NSUInteger capacity = MAX(chunkSize, alignedLength);
        @synchronized(gl4metalUniformArenaPool) {
            for (NSUInteger poolIndex = gl4metalUniformArenaPool.count; poolIndex > 0; --poolIndex) {
                id<MTLBuffer> candidate = gl4metalUniformArenaPool[poolIndex - 1];
                if (candidate.length >= capacity) {
                    arena = candidate;
                    gl4metalUniformArenaPoolBytes -= candidate.length;
                    [gl4metalUniformArenaPool removeObjectAtIndex:poolIndex - 1];
                    break;
                }
            }
        }
        if (!arena) {
            arena = [ctx.device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
        }
        if (!arena) return nil;
        [ctx.uniformArenaBuffers addObject:arena];
        offset = 0;

        if (getenv("GL4METAL_UNIFORM_ARENA_TRACE")) {
            printf("[gl4metal][UNIFORM_ARENA] new chunk=%lu capacity=%lu\n",
                   (unsigned long)ctx.uniformArenaBuffers.count - 1,
                   (unsigned long)capacity);
        }
    }

    ctx.uniformArenaOffset = offset + alignedLength;
    *outOffset = offset;
    return arena;
}

// Materialize uniform contents for a single encoded draw. The returned
// NSData is CPU-side storage only; the draw path copies it into a per-command
// buffer arena. The arena is held until command-buffer completion, so earlier
// draws can never observe later uniform writes.
static BOOL gl4metalMatrixDimensions(GLenum type, NSUInteger *columns, NSUInteger *rows) {
    NSUInteger c = 0, r = 0;
    switch (type) {
        case GL_FLOAT_MAT2: c = 2; r = 2; break;
        case GL_FLOAT_MAT3: c = 3; r = 3; break;
        case GL_FLOAT_MAT4: c = 4; r = 4; break;
        case GL_FLOAT_MAT2x3: c = 2; r = 3; break;
        case GL_FLOAT_MAT2x4: c = 2; r = 4; break;
        case GL_FLOAT_MAT3x2: c = 3; r = 2; break;
        case GL_FLOAT_MAT3x4: c = 3; r = 4; break;
        case GL_FLOAT_MAT4x2: c = 4; r = 2; break;
        case GL_FLOAT_MAT4x3: c = 4; r = 3; break;
        default: return NO;
    }
    if (columns) *columns = c;
    if (rows) *rows = r;
    return YES;
}

static void gl4metalPackMatrixUniform(NSMutableData *destination, NSData *source, GLenum type) {
    NSUInteger columns = 0, rows = 0;
    if (!gl4metalMatrixDimensions(type, &columns, &rows) || source.length == 0) return;
    NSUInteger sourceMatrixBytes = columns * rows * sizeof(GLfloat);
    NSUInteger columnStride = rows == 2 ? 2 * sizeof(GLfloat) : 4 * sizeof(GLfloat);
    NSUInteger matrixStride = (columns * columnStride + 15u) & ~15u;
    const GLfloat *src = (const GLfloat *)source.bytes;
    for (NSUInteger m = 0; m < source.length / sourceMatrixBytes; ++m) {
        for (NSUInteger c = 0; c < columns; ++c) {
            NSUInteger dstOffset = m * matrixStride + c * columnStride;
            NSUInteger srcOffset = m * columns * rows + c * rows;
            if (dstOffset + rows * sizeof(GLfloat) > destination.length) return;
            memcpy((uint8_t *)destination.mutableBytes + dstOffset, src + srcOffset,
                   rows * sizeof(GLfloat));
        }
    }
}

static NSData *gl4metalUniformDataForName(NSString *name, GLuint program, NSUInteger length) {
    if (!ctx || !ctx.device || !name || length == 0) return nil;

    NSMutableData *data = [NSMutableData dataWithLength:length];
    if (!data) return nil;
    memset(data.mutableBytes, 0, length);

    GLint location = gl4metalFindUniformLocationByName(name, program);
    if (getenv("GL4METAL_UNIFORM_TRACE")) {
        printf("[gl4metal][UNIFORM_TRACE] program=%u resource=%s length=%lu location=%d %s\n",
               program, name.UTF8String, (unsigned long)length, location,
               location >= 0 ? "FOUND" : "MISSING");
    }
    if (location >= 0) {
        NSData *rawValue = uniformRawValues[@(location)];
        if (rawValue.length > 0) {
            GLenum type = gl4metalUniformTypeForLocation(program, location);
            if (gl4metalMatrixDimensions(type, NULL, NULL)) {
                gl4metalPackMatrixUniform(data, rawValue, type);
            } else {
                memcpy(data.mutableBytes, rawValue.bytes, MIN(length, rawValue.length));
            }
            return data;
        }

        NSValue *vec4 = uniform4fValues[@(location)];
        if (vec4 && length >= 16) {
            float values[4] = {0};
            if (!gl4metalCopyNSValueBytes(vec4, values, sizeof(values))) return data;
            memcpy(data.mutableBytes, values, sizeof(values));
            return data;
        }
        NSValue *vec3 = uniform3fValues[@(location)];
        if (vec3 && length >= 12) {
            float values[3] = {0};
            if (!gl4metalCopyNSValueBytes(vec3, values, sizeof(values))) return data;
            memcpy(data.mutableBytes, values, sizeof(values));
            return data;
        }
        NSValue *vec2 = uniform2fValues[@(location)];
        if (vec2 && length >= 8) {
            float values[2] = {0};
            if (!gl4metalCopyNSValueBytes(vec2, values, sizeof(values))) return data;
            memcpy(data.mutableBytes, values, sizeof(values));
            return data;
        }
        NSValue *scalar = uniform1fValues[@(location)];
        if (scalar && length >= 4) {
            float value = 0.0f;
            if (!gl4metalCopyNSValueBytes(scalar, &value, sizeof(value))) return data;
            memcpy(data.mutableBytes, &value, sizeof(value));
            return data;
        }
        NSNumber *ivalue = uniform1iValues[@(location)];
        if (ivalue && length >= 4) {
            int value = ivalue.intValue;
            memcpy(data.mutableBytes, &value, sizeof(value));
            return data;
        }
    }

    // Minimal internal shader fallback.
    if ([name isEqualToString:@"uniforms"] && length >= 64) {
        float identity[16] = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
        };
        memcpy(data.mutableBytes, identity, sizeof(identity));
    } else if ([name isEqualToString:@"ColorModulator"] && length >= 16) {
        float white[4] = {1,1,1,1};
        memcpy(data.mutableBytes, white, sizeof(white));
    }
    return data;
}

static void gl4metalEnsureDummyResources(void) {
    if (!ctx || !ctx.device) return;

    if (!gl4metalDummyTexture) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                        width:1 height:1 mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        gl4metalDummyTexture = [ctx.device newTextureWithDescriptor:td];
        if (gl4metalDummyTexture) {
            uint32_t white = 0xffffffffu;
            [gl4metalDummyTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                    mipmapLevel:0
                                      withBytes:&white
                                    bytesPerRow:sizeof(white)];
        }
    }

    if (!gl4metalDummySampler) {
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        gl4metalDummySampler = [ctx.device newSamplerStateWithDescriptor:sd];
    }
}

// Bind every active resource reported by Metal reflection. This deliberately
// uses the reflected stage/index instead of guessing from GLSL uniform names.
// Vertex attribute buffers are already bound by the draw path at 0..15 and are
// therefore excluded here.
static void gl4metalBindProgramResources(id<MTLRenderCommandEncoder> encoder, GLuint program) {
    if (!encoder || !ctx || program == 0) return;

    NSString *info = ctx.programResourceInfo[@(program)];
    if (!info) {
        printf("[gl4metal][RESOURCES] program=%u has no reflection metadata\n", program);
        return;
    }

    gl4metalEnsureDummyResources();
    gl4metalTraceResourceSnapshot(program);

    NSArray<NSString *> *lines = [info componentsSeparatedByString:@"\n"];
    NSUInteger boundBuffers = 0, boundTextures = 0, boundSamplers = 0;

    for (NSString *line in lines) {
        if (line.length == 0 || [line hasPrefix:@"vertexFunction="]) continue;

        const char *raw = line.UTF8String;
        char stage[16] = {0};
        char name[256] = {0};
        char type[32] = {0};
        unsigned long index = 0;
        unsigned long dataSize = 0;

        int parsed = sscanf(raw, "%15s name=%255s type=%31s index=%lu", stage, name, type, &index);
        if (parsed != 4) continue;

        // The vertex input resources are bound separately from VAO state.
        if (strncmp(name, "vertexBuffer.", 13) == 0) continue;

        if (strcmp(type, "buffer") == 0) {
            const char *sizeMarker = strstr(raw, "dataSize=");
            if (sizeMarker) dataSize = strtoul(sizeMarker + 9, NULL, 10);
            if (dataSize == 0) dataSize = 16;

            NSString *resourceName = [NSString stringWithUTF8String:name];
            NSData *uniformData = gl4metalUniformDataForName(resourceName, program, (NSUInteger)dataSize);
            if (!uniformData) continue;

            @autoreleasepool {
                NSUInteger bufferOffset = 0;
                id<MTLBuffer> buffer = gl4metalUniformArenaAlloc(uniformData.length, &bufferOffset);
                if (!buffer) continue;
                memcpy((uint8_t *)buffer.contents + bufferOffset,
                       uniformData.bytes,
                       uniformData.length);

                if (strcmp(stage, "VERTEX") == 0) {
                    [encoder setVertexBuffer:buffer offset:bufferOffset atIndex:(NSUInteger)index];
                } else if (strcmp(stage, "FRAGMENT") == 0) {
                    [encoder setFragmentBuffer:buffer offset:bufferOffset atIndex:(NSUInteger)index];
                } else {
                    continue;
                }
                ++boundBuffers;
            }
            if (getenv("GL4METAL_VERBOSE_RESOURCES")) {
                printf("[gl4metal][RESOURCES] auto-bound %s buffer(%lu) %s size=%lu\n",
                       stage, index, name, dataSize);
            }
        } else if (strcmp(type, "texture") == 0) {
            NSString *resourceName = [NSString stringWithUTF8String:name];
            GLint location = gl4metalFindUniformLocationByName(resourceName, program);
            GLint unit = 0;
            NSNumber *unitValue = uniform1iValues[@(location)];
            if (unitValue) unit = unitValue.intValue;
            if (unit < 0) unit = 0;
            id<MTLTexture> texture = gl4metalGetTextureForUnit((GLuint)unit);
            if (!texture) texture = gl4metalDummyTexture;
            if (!texture) continue;
            if (strcmp(stage, "VERTEX") == 0) {
                [encoder setVertexTexture:texture atIndex:(NSUInteger)index];
            } else if (strcmp(stage, "FRAGMENT") == 0) {
                [encoder setFragmentTexture:texture atIndex:(NSUInteger)index];
            } else {
                continue;
            }
            ++boundTextures;
            GLuint glTexture = gl4metalGetTextureNameForUnit((GLuint)unit);
            if (getenv("GL4METAL_TEXTURE_BIND_TRACE") &&
                (program == 1 || program == 17 || program == 53)) {
                printf("[gl4metal][TEXTURE_BIND_TRACE] program=%u stage=%s name=%s resourceIndex=%lu unit=%d GLtex=%u metal=%p\n",
                       program, stage, name, index, unit, glTexture, texture);
            }
            if (getenv("GL4METAL_VERBOSE_RESOURCES")) {
                printf("[gl4metal][RESOURCES] auto-bound %s texture(%lu) %s %s unit=%d GLtex=%u\n",
                       stage, index, name, glTexture ? "[real]" : "[dummy]", unit, glTexture);
            }
        } else if (strcmp(type, "sampler") == 0) {
            NSString *resourceName = [NSString stringWithUTF8String:name];
            if ([resourceName hasSuffix:@"Smplr"]) {
                resourceName = [resourceName substringToIndex:resourceName.length - 5];
            }
            GLint location = gl4metalFindUniformLocationByName(resourceName, program);
            GLint unit = 0;
            NSNumber *unitValue = uniform1iValues[@(location)];
            if (unitValue) unit = unitValue.intValue;
            if (unit < 0) unit = 0;
            id<MTLSamplerState> sampler = gl4metalGetSamplerForUnit((GLuint)unit);
            if (!sampler) sampler = gl4metalDummySampler;
            if (!sampler) continue;
            if (strcmp(stage, "VERTEX") == 0) {
                [encoder setVertexSamplerState:sampler atIndex:(NSUInteger)index];
            } else if (strcmp(stage, "FRAGMENT") == 0) {
                [encoder setFragmentSamplerState:sampler atIndex:(NSUInteger)index];
            } else {
                continue;
            }
            ++boundSamplers;
            if (getenv("GL4METAL_VERBOSE_RESOURCES")) {
                printf("[gl4metal][RESOURCES] auto-bound %s sampler(%lu) %s %s unit=%d\n",
                       stage, index, name, gl4metalGetTextureNameForUnit((GLuint)unit) ? "[real]" : "[dummy]", unit);
            }
        }
    }

    if (getenv("GL4METAL_VERBOSE_RESOURCES")) {
        printf("[gl4metal][RESOURCES] automatic binding complete program=%u buffers=%lu textures=%lu samplers=%lu\n",
               program, (unsigned long)boundBuffers, (unsigned long)boundTextures, (unsigned long)boundSamplers);
    }
}

static MTLIndexType getMetalIndexType(GLenum type) {
    switch(type) {
        case GL_UNSIGNED_BYTE: return MTLIndexTypeUInt16; // Metal does not support 8-bit indices, defaulting to 16-bit
        case GL_UNSIGNED_SHORT: return MTLIndexTypeUInt16;
        case GL_UNSIGNED_INT: return MTLIndexTypeUInt32;
        default:
            printf("[gl4metal] WARNING: Unsupported GL index type 0x%X, defaulting to UInt16\n", type);
            return MTLIndexTypeUInt16;
    }
}

void APIENTRY glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    gl4metalDrawArraysInternal(mode, first, count, 1);
}

void APIENTRY glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    gl4metalDrawArraysInternal(mode, first, count, instancecount);
}

void APIENTRY glMultiDrawArrays(GLenum mode, const GLint *first,
                                const GLsizei *count, GLsizei drawcount) {
    if (drawcount < 0 || (drawcount > 0 && (!first || !count))) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw) {
        if (first[draw] < 0 || count[draw] < 0) {
            gl4metalSetError(GL_INVALID_VALUE);
            return;
        }
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw) {
        glDrawArrays(mode, first[draw], count[draw]);
    }
}

static void gl4metalDrawElementsInternal(GLenum mode, GLsizei count, GLenum type,
                                         const void *indices, GLsizei instancecount,
                                         GLint baseVertex) {
    if (!gl4metalIsValidDrawMode(mode)) { gl4metalSetError(GL_INVALID_ENUM); return; }
    if (count < 0 || instancecount < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (!gl4metalModeIsImplemented(mode, YES)) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (ctx.currentFramebuffer != 0 &&
        glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }
    if (count == 0 || instancecount == 0) { gl4metalSetError(GL_NO_ERROR); return; }
    if (ctx.primitiveRestartEnabled) {
        gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
        GLuint elementBuffer = vao ? vao->elementArrayBufferID
            : [ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] unsignedIntValue];
        id<MTLBuffer> indicesBuffer = bufferObjects[@(elementBuffer)];
        size_t indexWidth = type == GL_UNSIGNED_BYTE ? 1u :
                            type == GL_UNSIGNED_SHORT ? 2u : 4u;
        uintptr_t offset = (uintptr_t)indices;
        if (!elementBuffer || !indicesBuffer) {
            gl4metalSetError(GL_INVALID_OPERATION);
            return;
        }
        if (offset > indicesBuffer.length ||
            (size_t)count > (indicesBuffer.length - (size_t)offset) / indexWidth) {
            gl4metalSetError(GL_INVALID_OPERATION);
            return;
        }
        uint64_t maxIndexValue = type == GL_UNSIGNED_BYTE ? UINT8_MAX :
                                 type == GL_UNSIGNED_SHORT ? UINT16_MAX : UINT32_MAX;
        uint64_t restartValue = ctx.primitiveRestartIndex;
        if (restartValue > maxIndexValue) restartValue = UINT64_MAX;
        GLsizei segmentStart = 0;
        ctx.primitiveRestartEnabled = GL_FALSE;
        for (GLsizei index = 0; index < count; ++index) {
            NSUInteger value = 0;
            if (!gl4metalReadIndexValue(indicesBuffer, type, (NSUInteger)offset,
                                        (NSUInteger)index, &value)) {
                ctx.primitiveRestartEnabled = GL_TRUE;
                gl4metalSetError(GL_INVALID_OPERATION);
                return;
            }
            if ((uint64_t)value == restartValue) {
                GLsizei segmentCount = index - segmentStart;
                if (segmentCount > 0) {
                    uintptr_t segmentOffset = offset + (uintptr_t)segmentStart * indexWidth;
                    gl4metalDrawElementsInternal(mode, segmentCount, type,
                        (const void *)segmentOffset, instancecount, baseVertex);
                }
                segmentStart = index + 1;
            }
        }
        GLsizei finalSegmentCount = count - segmentStart;
        if (finalSegmentCount > 0) {
            uintptr_t segmentOffset = offset + (uintptr_t)segmentStart * indexWidth;
            gl4metalDrawElementsInternal(mode, finalSegmentCount, type,
                (const void *)segmentOffset, instancecount, baseVertex);
        }
        ctx.primitiveRestartEnabled = GL_TRUE;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }
    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_BEGIN] kind=elements program=%u vao=%u fbo=%u mode=0x%X count=%d type=0x%X instances=%d indices=%p\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, count, type, instancecount, indices);
        fflush(stdout);
    }
    if (gl4metalGPUIsolationEnabled && gl4metalGPUIsolationFailed) {
        printf("[gl4metal][GPU-DEBUG] draw skipped after previous GPU failure\n");
        return;
    }

    // Always resolve against the current GL state. The resolver is backed by
    // pipelineLayoutCache, so this is a cheap cache lookup on the hot path,
    // while preventing stale pipelines after program/VAO/FBO changes.
    id<MTLRenderPipelineState> activePipeline = gl4metalResolveCurrentPipeline();
    if (!activePipeline) {
        printf("[gl4metal] WARNING: No valid Metal pipeline for glDrawElements\n");
        return;
    }
    ctx.pipelineState = activePipeline;

    gl4metalVertexArray *currentVAO = gl4metalGetCurrentVAO();
    if (currentVAO) gl4metalRecoverPackedInterleavedOffsets(currentVAO);
    GLuint elementVBO = currentVAO ? currentVAO->elementArrayBufferID : [ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] unsignedIntValue];

    if (elementVBO == 0) {
        printf("[gl4metal] ERROR: glDrawElements called without bound GL_ELEMENT_ARRAY_BUFFER\n");
        return;
    }

    id<MTLBuffer> indexBuffer = bufferObjects[@(elementVBO)];
    if (!indexBuffer) {
        printf("[gl4metal] ERROR: Index MTLBuffer not found for VBO ID %u\n", elementVBO);
        return;
    }
    size_t indexWidth = (type == GL_UNSIGNED_BYTE) ? 1 : (type == GL_UNSIGNED_SHORT ? 2 : (type == GL_UNSIGNED_INT ? 4 : 0));
    if (!indexWidth) { printf("[gl4metal][BOUNDS] Unsupported index type 0x%X\n", type); return; }
    GLintptr rawIndexOffset = (indices == NULL) ? 0 : (GLintptr)(uintptr_t)indices;
    if (rawIndexOffset < 0 || (size_t)rawIndexOffset > (size_t)[indexBuffer length] || (size_t)count > ((size_t)[indexBuffer length] - (size_t)rawIndexOffset) / indexWidth) {
        printf("[gl4metal][BOUNDS] Aborting glDrawElements: index range OOB type=0x%X count=%d offset=%ld buffer=%lu\n", type, count, (long)rawIndexOffset, (unsigned long)[indexBuffer length]);
        return;
    }
    NSUInteger minIndex = 0;
    NSUInteger maxIndex = 0;
    NSUInteger sampleValues[8] = {0};
    NSUInteger sampleCount = 0;
    uint64_t indexHash = 0;
    BOOL collectIndexTrace = getenv("GL4METAL_INDEX_TRACE") != NULL;
    NSString *statsKey = [NSString stringWithFormat:@"%u:%u:%lu:%lu",
                          elementVBO, type, (unsigned long)rawIndexOffset, (unsigned long)count];
    NSData *cachedStats = collectIndexTrace ? nil : gl4metalIndexStatsCache[statsKey];
    if (cachedStats.length == sizeof(NSUInteger) * 2) {
        NSUInteger values[2] = {0};
        memcpy(values, cachedStats.bytes, sizeof(values));
        minIndex = values[0];
        maxIndex = values[1];
    } else {
        BOOL statsOK = gl4metalFindIndexStats(indexBuffer, type, (NSUInteger)rawIndexOffset,
            (NSUInteger)count, &minIndex, &maxIndex,
            collectIndexTrace ? sampleValues : NULL,
            collectIndexTrace ? &sampleCount : NULL,
            collectIndexTrace ? &indexHash : NULL);
        if (!statsOK) {
            printf("[gl4metal][BOUNDS] Aborting glDrawElements: cannot inspect index buffer\n");
            return;
        }
        if (!collectIndexTrace && gl4metalIndexStatsCache) {
            if (gl4metalIndexStatsCache.count >= 4096) [gl4metalIndexStatsCache removeAllObjects];
            NSUInteger values[2] = {minIndex, maxIndex};
            gl4metalIndexStatsCache[statsKey] = [NSData dataWithBytes:values length:sizeof(values)];
        }
    }
    int64_t effectiveMinIndex = (int64_t)minIndex + (int64_t)baseVertex;
    int64_t effectiveMaxIndex = (int64_t)maxIndex + (int64_t)baseVertex;
    if (effectiveMinIndex < 0 || effectiveMaxIndex < effectiveMinIndex ||
        (uint64_t)effectiveMaxIndex > (uint64_t)SIZE_MAX) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSUInteger effectiveMaxVertex = (NSUInteger)effectiveMaxIndex;
    if (getenv("GL4METAL_DEBUG")) {
        printf("[gl4metal][BOUNDS] index check: EBO=%u offset=%ld count=%d type=0x%X width=%zu buffer=%lu minIndex=%lu maxIndex=%lu\n",
               elementVBO, (long)rawIndexOffset, count, type, indexWidth,
               (unsigned long)[indexBuffer length], (unsigned long)minIndex, (unsigned long)maxIndex);
    }

    // Opt-in indexed-draw trace used to diagnose stretched triangles / bad VAO-EBO
    // state without permanently flooding the normal render log. Each unique
    // draw signature+index-data hash is printed once per process.
    if (collectIndexTrace) {
        static NSMutableSet<NSString *> *seenIndexTraces = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{ seenIndexTraces = [NSMutableSet set]; });

        NSString *traceKey = [NSString stringWithFormat:@"%u:%u:%u:%u:%ld:%d:%llu:%lu",
                              ctx.currentFramebuffer, ctx.currentVAO, currentProgram, elementVBO,
                              (long)rawIndexOffset, count, (unsigned long long)indexHash,
                              (unsigned long)[indexBuffer length]];
        if (![seenIndexTraces containsObject:traceKey] && seenIndexTraces.count < 512) {
            [seenIndexTraces addObject:traceKey];
            printf("[gl4metal][INDEX_TRACE] program=%u vao=%u fbo=%u EBO=%u offset=%ld count=%d type=0x%X eboSize=%lu min=%lu max=%lu hash=%016llx\n",
                   currentProgram, ctx.currentVAO, ctx.currentFramebuffer, elementVBO,
                   (long)rawIndexOffset, count, type, (unsigned long)[indexBuffer length],
                   (unsigned long)minIndex, (unsigned long)maxIndex, (unsigned long long)indexHash);
            printf("[gl4metal][INDEX_TRACE] first=%lu", (unsigned long)sampleCount);
            for (NSUInteger si = 0; si < sampleCount; ++si) {
                printf(" %lu", (unsigned long)sampleValues[si]);
            }
            printf("\n");
            gl4metalVertexArray *traceVAO = currentVAO;
            if (traceVAO) {
                printf("[gl4metal][INDEX_TRACE] attributes:");
                for (NSUInteger ai = 0; ai < 16; ++ai) {
                    gl4metalVertexAttrib *a = &traceVAO->attribs[ai];
                    if (!a->enabled && a->boundVBO == 0) continue;
                    printf(" a%lu{en=%d,vbo=%u,off=%ld,stride=%d,size=%d,type=0x%X,norm=%d,int=%d,div=%u}",
                           (unsigned long)ai, a->enabled, a->boundVBO, (long)a->pointerOffset,
                           a->stride, a->size, a->type, a->normalized, a->integer, a->divisor);
                }
                printf("\n");
            }
            printf("[gl4metal][INDEX_TRACE] state: viewport=%d,%d %dx%d scissor=%d,%d %dx%d scissorTest=%d depthTest=%d depthWrite=%d depthFunc=0x%X blend=%d cull=%d frontFace=0x%X\n",
                   ctx.viewport.x, ctx.viewport.y, ctx.viewport.width, ctx.viewport.height,
                   ctx.scissorRect.x, ctx.scissorRect.y, ctx.scissorRect.width, ctx.scissorRect.height,
                   ctx.scissorTestEnabled, ctx.depthTestEnabled, ctx.depthWriteEnabled, ctx.depthFunc,
                   ctx.blendEnabled, ctx.cullEnabled, ctx.frontFace);
        }
    }
    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = gl4metalCreateCommandBuffer();
    if (ctx.currentFramebuffer == 0 && !ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];

    if (ctx.currentFramebuffer == 0 && !ctx.currentDrawable) {
        printf("[gl4metal] ERROR: No drawable available for rendering\n");
        return;
    }

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();
    if (!renderPassDesc) {
        gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return;
    }
    if (!ctx.currentCommandBuffer) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    // The active command-buffer resource set must be installed before marking
    // the EBO; otherwise the first indexed draw after a swap is charged to the
    // previous frame's resource set and its hazards go untracked.
    gl4metalMarkBufferUsed(elementVBO);
    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSUInteger targetWidth = 0;
    NSUInteger targetHeight = 0;
    if (ctx.currentFramebuffer != 0) {
        id<MTLTexture> target = gl4metalGetCurrentFramebufferColorTexture();
        targetWidth = target ? target.width : 0;
        targetHeight = target ? target.height : 0;
    } else if (ctx.currentDrawable) {
        targetWidth = ctx.currentDrawable.texture.width;
        targetHeight = ctx.currentDrawable.texture.height;
    }
    applyViewportAndScissor(encoder, targetWidth, targetHeight);
    if (ctx.depthStencilState) {
        [encoder setDepthStencilState:ctx.depthStencilState];
    }
    if (currentProgram == 53 && getenv("GL4METAL_PROGRAM53_NO_DEPTH")) {
        id<MTLDepthStencilState> noDepthState = nil;
        MTLDepthStencilDescriptor *dds = [[MTLDepthStencilDescriptor alloc] init];
        dds.depthCompareFunction = MTLCompareFunctionAlways;
        dds.depthWriteEnabled = NO;
        noDepthState = [ctx.device newDepthStencilStateWithDescriptor:dds];
        if (noDepthState) [encoder setDepthStencilState:noDepthState];
    }
        [encoder setStencilFrontReferenceValue:(uint32_t)MAX(ctx.stencilRef, 0)
                          backReferenceValue:(uint32_t)MAX(ctx.stencilBackRef, 0)];
    applyRasterState(encoder);
    [encoder setRenderPipelineState:ctx.pipelineState];

    if (currentVAO) {
        for (NSUInteger index = 0; index < 16; index++) {
            gl4metalVertexAttrib *attrib = &currentVAO->attribs[index];
            if (attrib->enabled && attrib->boundVBO != 0) {
                id<MTLBuffer> buffer = bufferObjects[@(attrib->boundVBO)];
                if (buffer) {
                    NSUInteger offset = 0;
                    if (!gl4metalValidateBufferOffset(buffer, attrib->pointerOffset, &offset)) {
                        continue;
                    }
                    gl4metalMarkBufferUsed(attrib->boundVBO);
                    [encoder setVertexBuffer:buffer offset:offset atIndex:index];
                    if (getenv("GL4METAL_VERTEX_BIND_TRACE")) {
                        printf("[gl4metal][VERTEX_BIND_TRACE] program=%u vao=%u attr=%lu vbo=%u offset=%lu stride=%d size=%d type=0x%X\n",
                               currentProgram, ctx.currentVAO, (unsigned long)index, attrib->boundVBO,
                               (unsigned long)offset, attrib->stride, attrib->size, attrib->type);
                    }
                }
            } else if ((!attrib->enabled || attrib->boundVBO == 0) &&
                       [inferredConstantVertexAttributes containsObject:@(index)]) {
                id<MTLBuffer> constantBuffer = constantVertexAttributeBuffers[@(index)];
                if (constantBuffer) {
                    [encoder setVertexBuffer:constantBuffer offset:0 atIndex:index];
                }
            }
        }
    }

    gl4metalBindProgramResources(encoder, currentProgram);

    MTLIndexType indexType = getMetalIndexType(type);
    GLintptr indexOffsetValue = gl4metalResolvePointerOffset(indices, indexBuffer);
    NSUInteger indexOffset = 0;
    if (!gl4metalValidateBufferOffset(indexBuffer, indexOffsetValue, &indexOffset)) {
        printf("[gl4metal][BOUNDS] Aborting glDrawElements: invalid index offset\n");
        [encoder endEncoding];
        return;
    }
    if (currentVAO) {
        BOOL vertexOK = YES;
        for (NSUInteger ai=0; ai<16; ++ai) {
            gl4metalVertexAttrib *a=&currentVAO->attribs[ai];
            if (!a->enabled || a->boundVBO==0) continue;
            id<MTLBuffer> vb=bufferObjects[@(a->boundVBO)];
            if (!vb || !gl4metalValidateIndexedVertexAttribRange(a, vb, effectiveMaxVertex, (NSUInteger)instancecount, ai)) vertexOK=NO;
        }
        if (!vertexOK) {
            printf("[gl4metal][BOUNDS] Aborting glDrawElements: vertex buffer range invalid\n");
            [encoder endEncoding];
            return;
        }
    }

    if (getenv("GL4METAL_DEBUG")) {
        printf("[gl4metal][DRAW] glDrawElements mode=0x%X count=%d type=0x%X instances=%d program=%u vao=%u\n", mode, count, type, instancecount, currentProgram, ctx.currentVAO);
    }
    gl4metalDumpProgramResourceInfo(currentProgram);

    id<MTLBuffer> drawIndexBuffer = indexBuffer;
    NSUInteger drawIndexOffset = indexOffset;
    MTLIndexType drawIndexType = indexType;

    if (type == GL_UNSIGNED_BYTE) {
        NSString *cacheKey = [NSString stringWithFormat:@"%u:%lu:%d", elementVBO, (unsigned long)indexOffset, (int)count];
        id<MTLBuffer> cachedBuffer = convertedIndexBuffers[cacheKey];
        if (cachedBuffer) {
            drawIndexBuffer = cachedBuffer;
            drawIndexOffset = 0;
            drawIndexType = MTLIndexTypeUInt16;
        }

        const GLubyte *src = (const GLubyte *)((const char *)[indexBuffer contents] + indexOffset);
        if (indices != NULL && elementVBO == 0) {
            src = (const GLubyte *)indices;
        }

        if (!cachedBuffer) {
            size_t tmpByteLength = (size_t)count * sizeof(uint16_t);
            id<MTLBuffer> tmpBuffer = [ctx.device newBufferWithLength:tmpByteLength options:MTLResourceStorageModeShared];
            if (tmpBuffer) {
                uint16_t *dst = (uint16_t *)[tmpBuffer contents];
                for (GLsizei i = 0; i < count; i++) {
                    dst[i] = (uint16_t)src[i];
                }
                // Bound the conversion cache: a game can issue many unique
                // offset/count pairs against static byte-index buffers.
                const NSUInteger cacheByteLimit = 64u * 1024u * 1024u;
                const NSUInteger cacheEntryLimit = 512;
                if (tmpByteLength <= cacheByteLimit) {
                    if (convertedIndexBuffers.count >= cacheEntryLimit ||
                        gl4metalConvertedIndexBufferCacheBytes > cacheByteLimit - tmpByteLength) {
                        [convertedIndexBuffers removeAllObjects];
                        gl4metalConvertedIndexBufferCacheBytes = 0;
                    }
                    convertedIndexBuffers[cacheKey] = tmpBuffer;
                    gl4metalConvertedIndexBufferCacheBytes += tmpByteLength;
                }
                drawIndexBuffer = tmpBuffer;
                drawIndexOffset = 0;
                drawIndexType = MTLIndexTypeUInt16;
            }
        }
    }

    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_READY] kind=elements program=%u vao=%u fbo=%u mode=0x%X count=%d type=0x%X indexType=%u indexOffset=%lu pipeline=%p encoder=%p indexBuffer=%p\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, count, type,
               (unsigned)drawIndexType, (unsigned long)drawIndexOffset,
               (void *)ctx.pipelineState, (void *)encoder, (void *)drawIndexBuffer);
        fflush(stdout);
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        id<MTLBuffer> fanBuffer = nil;
        MTLIndexType fanType = MTLIndexTypeUInt16;
        if (gl4metalMakeTriangleFanIndexBuffer(drawIndexBuffer, drawIndexOffset, count,
                                               (drawIndexType == MTLIndexTypeUInt32) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT,
                                               &fanBuffer, &fanType)) {
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                 indexCount:((NSUInteger)count - 2) * 3
                                  indexType:fanType
                                indexBuffer:fanBuffer
                          indexBufferOffset:0
                              instanceCount:(NSUInteger)instancecount
                                 baseVertex:baseVertex
                               baseInstance:0];
        }
    } else {
        MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
        [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:drawIndexType
                           indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset
                         instanceCount:(NSUInteger)instancecount baseVertex:baseVertex baseInstance:0];
    }

    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_AFTER] kind=elements program=%u vao=%u fbo=%u mode=0x%X count=%d type=0x%X indexType=%u indexOffset=%lu pipeline=%p encoder=%p indexBuffer=%p\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, count, type,
               (unsigned)drawIndexType, (unsigned long)drawIndexOffset,
               (void *)ctx.pipelineState, (void *)encoder, (void *)drawIndexBuffer);
        fflush(stdout);
    }

    [encoder endEncoding];
    if (gl4metalGPUIsolationEnabled) {
        NSUInteger serial = ++gl4metalDrawSerial;
        if (!gl4metalFinishIsolatedDraw(ctx.currentCommandBuffer, serial, "glDrawElements")) {
            ctx.currentCommandBuffer = nil;
            return;
        }
        ctx.currentCommandBuffer = nil;
    }
}

void APIENTRY glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
    gl4metalDrawElementsInternal(mode, count, type, indices, 1, 0);
}

void APIENTRY glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    gl4metalDrawElementsInternal(mode, count, type, indices, instancecount, 0);
}

void APIENTRY glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type,
                                       const void *indices, GLint basevertex) {
    gl4metalDrawElementsInternal(mode, count, type, indices, 1, basevertex);
}

void APIENTRY glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type,
                                                const void *indices, GLsizei instancecount,
                                                GLint basevertex) {
    gl4metalDrawElementsInternal(mode, count, type, indices, instancecount, basevertex);
}

void APIENTRY glDrawRangeElements(GLenum mode, GLuint start, GLuint end,
                                  GLsizei count, GLenum type, const void *indices) {
    if (start > end || count < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glDrawElements(mode, count, type, indices);
}

void APIENTRY glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end,
                                            GLsizei count, GLenum type,
                                            const void *indices, GLint basevertex) {
    if (start > end || count < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    gl4metalDrawElementsInternal(mode, count, type, indices, 1, basevertex);
}

void APIENTRY glMultiDrawElements(GLenum mode, const GLsizei *count, GLenum type,
                                  const void *const *indices, GLsizei drawcount) {
    if (drawcount < 0 || (drawcount > 0 && (!count || !indices))) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw) {
        if (count[draw] < 0) {
            gl4metalSetError(GL_INVALID_VALUE);
            return;
        }
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw) {
        glDrawElements(mode, count[draw], type, indices[draw]);
    }
}

void APIENTRY glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei *count,
                                            GLenum type, const void *const *indices,
                                            GLsizei drawcount, const GLint *basevertex) {
    if (drawcount < 0 || (drawcount > 0 && (!count || !indices || !basevertex))) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw) {
        if (count[draw] < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    }
    for (GLsizei draw = 0; draw < drawcount; ++draw)
        gl4metalDrawElementsInternal(mode, count[draw], type, indices[draw], 1, basevertex[draw]);
}

static MTLCompareFunction getMetalCompareFunction(GLenum func) {
    switch(func) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        case GL_ALWAYS: return MTLCompareFunctionAlways;
        default:
            printf("[gl4metal] WARNING: Unsupported GL depth func 0x%X, defaulting to Less\n", func);
            return MTLCompareFunctionLess;
    }
}

static MTLStencilOperation getMetalStencilOperation(GLenum operation) {
    switch (operation) {
        case GL_KEEP: return MTLStencilOperationKeep;
        case GL_ZERO: return MTLStencilOperationZero;
        case GL_REPLACE: return MTLStencilOperationReplace;
        case GL_INCR: return MTLStencilOperationIncrementClamp;
        case GL_DECR: return MTLStencilOperationDecrementClamp;
        case GL_INVERT: return MTLStencilOperationInvert;
        case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        default: return MTLStencilOperationKeep;
    }
}

static void updateDepthStencilState(void) {
    if (!ctx || !ctx.device) return;

    MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];

    if (ctx.depthTestEnabled) {
        desc.depthCompareFunction = getMetalCompareFunction(ctx.depthFunc);
        desc.depthWriteEnabled = ctx.depthWriteEnabled;
    } else {
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = NO;
    }

    if (ctx.stencilTestEnabled) {
        MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
        front.stencilCompareFunction = getMetalCompareFunction(ctx.stencilFunc);
        front.stencilFailureOperation = getMetalStencilOperation(ctx.stencilFail);
        front.depthFailureOperation = getMetalStencilOperation(ctx.stencilDepthFail);
        front.depthStencilPassOperation = getMetalStencilOperation(ctx.stencilDepthPass);
        front.readMask = ctx.stencilReadMask;
        front.writeMask = ctx.stencilWriteMask;
        MTLStencilDescriptor *back = [[MTLStencilDescriptor alloc] init];
        back.stencilCompareFunction = getMetalCompareFunction(ctx.stencilBackFunc);
        back.stencilFailureOperation = getMetalStencilOperation(ctx.stencilBackFail);
        back.depthFailureOperation = getMetalStencilOperation(ctx.stencilBackDepthFail);
        back.depthStencilPassOperation = getMetalStencilOperation(ctx.stencilBackDepthPass);
        back.readMask = ctx.stencilBackReadMask;
        back.writeMask = ctx.stencilBackWriteMask;
        desc.frontFaceStencil = front;
        desc.backFaceStencil = back;
    }

    ctx.depthStencilState = [ctx.device newDepthStencilStateWithDescriptor:desc];
}

static void ensureDepthTexture(CGSize size) {
    if (!ctx.depthTexture || ctx.depthTexture.width != size.width || ctx.depthTexture.height != size.height) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:size.width height:size.height mipmapped:NO];
        desc.usage = MTLTextureUsageRenderTarget;
        desc.storageMode = MTLStorageModePrivate; // only gpu can access Depth Buffer
        ctx.depthTexture = [ctx.device newTextureWithDescriptor:desc];
        printf("[gl4metal] Created new depth texture of size: %.0fx%.0f\n", size.width, size.height);
    }
}

id<MTLTexture> gl4metalGetDefaultDepthTexture(void) {
    if (!ctx) return nil;
    if (!ctx.currentDrawable && ctx.metalLayer) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
    if (ctx.currentDrawable) {
        ensureDepthTexture(CGSizeMake(ctx.currentDrawable.texture.width,
                                      ctx.currentDrawable.texture.height));
    }
    return ctx.depthTexture;
}

static const char *gl4metalArgumentTypeName(MTLArgumentType type) {
    switch (type) {
        case MTLArgumentTypeBuffer: return "buffer";
        case MTLArgumentTypeThreadgroupMemory: return "threadgroup_memory";
        case MTLArgumentTypeTexture: return "texture";
        case MTLArgumentTypeSampler: return "sampler";
        default: return "unknown";
    }
}

static const char *gl4metalAccessName(MTLArgumentAccess access) {
    switch (access) {
        case MTLArgumentAccessReadOnly: return "read";
        case MTLArgumentAccessReadWrite: return "readwrite";
        case MTLArgumentAccessWriteOnly: return "write";
        default: return "unknown";
    }
}

static void gl4metalAppendReflectionArguments(NSMutableString *info,
                                               NSArray<MTLArgument *> *arguments,
                                               const char *stage) {
    if (!arguments) return;
    for (MTLArgument *arg in arguments) {
        if (!arg.active) continue;
        [info appendFormat:@"%s name=%@ type=%s index=%lu access=%s",
         stage, arg.name ?: @"<unnamed>", gl4metalArgumentTypeName(arg.type),
         (unsigned long)arg.index, gl4metalAccessName(arg.access)];
        if (arg.type == MTLArgumentTypeBuffer) {
            [info appendFormat:@" dataType=%lu dataSize=%lu",
             (unsigned long)arg.bufferDataType, (unsigned long)arg.bufferDataSize];
        } else if (arg.type == MTLArgumentTypeTexture) {
            [info appendFormat:@" textureType=%lu arrayLength=%lu",
             (unsigned long)arg.textureType, (unsigned long)arg.arrayLength];
        } else if (arg.type == MTLArgumentTypeSampler) {
            [info appendFormat:@" arrayLength=%lu", (unsigned long)arg.arrayLength];
        }
        [info appendString:@"\n"];
    }
}

static BOOL gl4metalCreatePipelineStateInternal(GLuint program,
                                                  id<MTLFunction> vertexFunction,
                                                  id<MTLFunction> fragmentFunction,
                                                  MTLVertexDescriptor *vertexDescriptor) {
    if (!ctx || !ctx.device || !vertexFunction || !fragmentFunction) return NO;

    MTLVertexDescriptor *vertexDesc = vertexDescriptor ? [vertexDescriptor copy] : [MTLVertexDescriptor vertexDescriptor];

    // Some real OpenGL shaders (notably Minecraft compositor/blit passes)
    // generate vertices procedurally with gl_VertexID and therefore have no
    // vertex input attributes at all. Metal does not accept an empty
    // MTLVertexDescriptor on this path; omit the descriptor entirely instead.
    // For shaders with at least one valid attribute, keep the real descriptor.
    BOOL hasValidVertexAttributes = NO;
    for (NSUInteger i = 0; i < 16; i++) {
        if (vertexDesc.attributes[i].format != MTLVertexFormatInvalid) {
            hasValidVertexAttributes = YES;
            break;
        }
    }

    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunction;
    pipelineDesc.fragmentFunction = fragmentFunction;
    pipelineDesc.vertexDescriptor = hasValidVertexAttributes ? vertexDesc : nil;
    if (!hasValidVertexAttributes) {
        printf("[gl4metal][ATTR] Shader uses no vertex input attributes; using nil MTLVertexDescriptor (procedural/gl_VertexID path).\n");
    }
    MTLPixelFormat targetColorFormat = gl4metalGetCurrentColorPixelFormat();
    MTLPixelFormat targetDepthFormat = gl4metalGetCurrentDepthPixelFormat();
    MTLPixelFormat targetStencilFormat = gl4metalGetCurrentStencilPixelFormat();
    // Metal validates this pipeline field against stencil-capable pixel
    // formats. A depth-only attachment must never be advertised as stencil.
    if (targetStencilFormat != MTLPixelFormatStencil8 &&
        targetStencilFormat != MTLPixelFormatDepth32Float_Stencil8) {
        targetStencilFormat = MTLPixelFormatInvalid;
    }
    if (targetColorFormat == MTLPixelFormatInvalid) {
        printf("[gl4metal] ERROR: No valid current color render-target format for program %u\n", program);
        return NO;
    }
    pipelineDesc.colorAttachments[0].pixelFormat = targetColorFormat;
    pipelineDesc.colorAttachments[0].writeMask = ctx.drawBuffer == GL_NONE ? MTLColorWriteMaskNone :
        ((ctx.colorMaskRed ? MTLColorWriteMaskRed : 0) |
         (ctx.colorMaskGreen ? MTLColorWriteMaskGreen : 0) |
         (ctx.colorMaskBlue ? MTLColorWriteMaskBlue : 0) |
         (ctx.colorMaskAlpha ? MTLColorWriteMaskAlpha : 0));
    // Blend state is taken directly from the emulated OpenGL state. Do not
    // force a blend mode by program number: Program 53 is the important
    // alpha-text candidate in vanilla 1.20.1, while Program 17 is used by
    // other position/texture/color passes as well.
    GLenum effectiveSrcRGB = ctx.blendSrcRGB;
    GLenum effectiveDstRGB = ctx.blendDstRGB;
    GLenum effectiveSrcAlpha = ctx.blendSrcAlpha;
    GLenum effectiveDstAlpha = ctx.blendDstAlpha;
    GLenum effectiveEqRGB = ctx.blendEquationRGB;
    GLenum effectiveEqAlpha = ctx.blendEquationAlpha;
    BOOL effectiveBlendEnabled = ctx.blendEnabled && !gl4metalColorLogicOperationEnabled;

    // Optional canonical alpha-blend A/B test for the vanilla 1.20.1 text
    // program. No default behavior changes unless the env var is present.
    if (program == 53 && getenv("GL4METAL_PROGRAM53_FORCE_ALPHA_BLEND")) {
        effectiveBlendEnabled = GL_TRUE;
        effectiveSrcRGB = GL_SRC_ALPHA;
        effectiveDstRGB = GL_ONE_MINUS_SRC_ALPHA;
        effectiveSrcAlpha = GL_ONE;
        effectiveDstAlpha = GL_ONE_MINUS_SRC_ALPHA;
        effectiveEqRGB = GL_FUNC_ADD;
        effectiveEqAlpha = GL_FUNC_ADD;
    }

    pipelineDesc.colorAttachments[0].blendingEnabled = effectiveBlendEnabled;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = gl4metalBlendFactor(effectiveSrcRGB);
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = gl4metalBlendFactor(effectiveDstRGB);
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = gl4metalBlendFactor(effectiveSrcAlpha);
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = gl4metalBlendFactor(effectiveDstAlpha);
    pipelineDesc.colorAttachments[0].rgbBlendOperation = gl4metalBlendOperation(effectiveEqRGB);
    pipelineDesc.colorAttachments[0].alphaBlendOperation = gl4metalBlendOperation(effectiveEqAlpha);
    pipelineDesc.depthAttachmentPixelFormat = targetDepthFormat;
    pipelineDesc.stencilAttachmentPixelFormat = targetStencilFormat;

    NSError *error = nil;
    MTLRenderPipelineReflection *reflection = nil;
    id<MTLRenderPipelineState> pipeline =
        [ctx.device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                   options:MTLPipelineOptionArgumentInfo
                                                reflection:&reflection
                                                     error:&error];
    if (!pipeline) {
        NSString *errorText = error.localizedDescription ?: @"Unknown Metal pipeline error";
        printf("[gl4metal] ERROR: Failed to create strict real pipeline: %s\n", errorText.UTF8String);
        ctx.pipelineState = nil;
        return NO;
    }

    NSMutableString *resourceInfo = [NSMutableString string];
    [resourceInfo appendFormat:@"vertexFunction=%@ fragmentFunction=%@\n",
     vertexFunction.name ?: @"<unnamed>", fragmentFunction.name ?: @"<unnamed>"];
    gl4metalAppendReflectionArguments(resourceInfo, reflection.vertexArguments, "VERTEX");
    gl4metalAppendReflectionArguments(resourceInfo, reflection.fragmentArguments, "FRAGMENT");
    if (resourceInfo.length == 0) [resourceInfo appendString:@"<no reflected resources>\n"];
    if (program != 0) {
        ctx.programResourceInfo[@(program)] = [resourceInfo copy];
    }

    ctx.pipelineState = pipeline;
    printf("[gl4metal] Real Metal pipeline created without attribute-format probing.\n");
    if (program == 17 || program == 20 || program == 21 || program == 27 || program == 53) {
        printf("[gl4metal][STATE] program=%u blend=%d rgb=0x%X/0x%X alpha=0x%X/0x%X eq=0x%X/0x%X cull=%d frontFace=0x%X depth=%d depthWrite=%d\n",
               program, (int)effectiveBlendEnabled, (unsigned)effectiveSrcRGB, (unsigned)effectiveDstRGB,
               (unsigned)effectiveSrcAlpha, (unsigned)effectiveDstAlpha,
               (unsigned)effectiveEqRGB, (unsigned)effectiveEqAlpha,
               (int)ctx.cullEnabled, (unsigned)ctx.frontFace,
               (int)ctx.depthTestEnabled, (int)ctx.depthWriteEnabled);
    }
    if (program != 0) {
        printf("[gl4metal][RESOURCES] program=%u reflected Metal resources:\n", program);
        printf("%s", resourceInfo.UTF8String);
    }
    if (getenv("GL4METAL_VERBOSE_RESOURCES") || getenv("GL4METAL_PIPELINE_TRACE")) {
        printf("[gl4metal] Pipeline vertex attributes:\n");
        for (NSUInteger i = 0; i < 16; i++) {
            MTLVertexFormat f = vertexDesc.attributes[i].format;
            if (f != MTLVertexFormatInvalid) {
                printf("[gl4metal]   attr[%lu] format=%lu buffer=%lu stride=%lu step=%lu rate=%lu\n",
                       (unsigned long)i, (unsigned long)f,
                       (unsigned long)vertexDesc.attributes[i].bufferIndex,
                       (unsigned long)vertexDesc.layouts[i].stride,
                       (unsigned long)vertexDesc.layouts[i].stepFunction,
                       (unsigned long)vertexDesc.layouts[i].stepRate);
            }
        }
    }
    return YES;
}

BOOL gl4metalCreatePipelineStateForProgram(GLuint program,
                                           id<MTLFunction> vertexFunction,
                                           id<MTLFunction> fragmentFunction,
                                           MTLVertexDescriptor *vertexDescriptor) {
    return gl4metalCreatePipelineStateInternal(program, vertexFunction, fragmentFunction, vertexDescriptor);
}

BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction,
                                 id<MTLFunction> fragmentFunction,
                                 MTLVertexDescriptor *vertexDescriptor) {
    return gl4metalCreatePipelineStateInternal(0, vertexFunction, fragmentFunction, vertexDescriptor);
}

void gl4metalDumpProgramResourceInfo(GLuint program) {
    if (!ctx || program == 0 || getenv("GL4METAL_VERBOSE_RESOURCES") == NULL) return;
    NSString *info = ctx.programResourceInfo[@(program)];
    if (!info) {
        printf("[gl4metal][RESOURCES] program=%u no reflection info cached\n", program);
        return;
    }
    printf("[gl4metal][RESOURCES] draw program=%u expected Metal resources:\n%s", program, info.UTF8String);
    printf("[gl4metal][RESOURCES] NOTE: draw path uses automatic reflection-based resource binding.\n");
}

static MTLBlendFactor gl4metalBlendFactor(GLenum factor) {
    switch (factor) {
        case GL_ZERO: return MTLBlendFactorZero;
        case GL_ONE: return MTLBlendFactorOne;
        case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
        case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
        case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case GL_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
        case GL_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
        case GL_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
        case GL_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
        case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
        default: return MTLBlendFactorOne;
    }
}

static MTLBlendOperation gl4metalBlendOperation(GLenum equation) {
    switch (equation) {
        case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
        case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case GL_MIN: return MTLBlendOperationMin;
        case GL_MAX: return MTLBlendOperationMax;
        case GL_FUNC_ADD:
        default: return MTLBlendOperationAdd;
    }
}

void APIENTRY glBlendFunc(GLenum sfactor, GLenum dfactor) {
    if (!ctx) return;
    ctx.blendSrcRGB = sfactor;
    ctx.blendDstRGB = dfactor;
    ctx.blendSrcAlpha = sfactor;
    ctx.blendDstAlpha = dfactor;
    ctx.pipelineState = nil;
    if (getenv("GL4METAL_BLEND_TRACE")) {
        printf("[gl4metal][BLEND_TRACE] glBlendFunc rgb=%s/0x%X alpha=0x%X/0x%X\n",
               "set", (unsigned)sfactor, (unsigned)sfactor, (unsigned)dfactor);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
    if (!ctx) return;
    ctx.blendSrcRGB = srcRGB;
    ctx.blendDstRGB = dstRGB;
    ctx.blendSrcAlpha = srcAlpha;
    ctx.blendDstAlpha = dstAlpha;
    ctx.pipelineState = nil;
    if (getenv("GL4METAL_BLEND_TRACE")) {
        printf("[gl4metal][BLEND_TRACE] glBlendFuncSeparate rgb=0x%X/0x%X alpha=0x%X/0x%X\n",
               (unsigned)srcRGB, (unsigned)dstRGB, (unsigned)srcAlpha, (unsigned)dstAlpha);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendEquation(GLenum mode) {
    if (!ctx) return;
    ctx.blendEquationRGB = mode;
    ctx.blendEquationAlpha = mode;
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
    if (!ctx) return;
    ctx.blendEquationRGB = modeRGB;
    ctx.blendEquationAlpha = modeAlpha;
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    if (!ctx) return;
    ctx.blendConstantRed = red;
    ctx.blendConstantGreen = green;
    ctx.blendConstantBlue = blue;
    ctx.blendConstantAlpha = alpha;
    ctx.pipelineState = nil;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glSampleCoverage(GLclampf value, GLboolean invert) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (value < 0.0f || value > 1.0f) { gl4metalSetError(GL_INVALID_VALUE); return; }
    ctx.sampleCoverageValue = value;
    ctx.sampleCoverageInvert = invert ? GL_TRUE : GL_FALSE;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glPrimitiveRestartIndex(GLuint index) {
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    ctx.primitiveRestartIndex = index;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glEnable(GLenum cap) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    if (cap == GL_BLEND) {
        if (!ctx.blendEnabled) { ctx.blendEnabled = GL_TRUE; ctx.pipelineState = nil; }
    } else if (cap == GL_DEPTH_TEST) {
        if (!ctx.depthTestEnabled) {
            ctx.depthTestEnabled = GL_TRUE;
            updateDepthStencilState();
        }
    } else if (cap == GL_STENCIL_TEST) {
        if (!ctx.stencilTestEnabled) {
            ctx.stencilTestEnabled = GL_TRUE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (!ctx.scissorTestEnabled) {
            ctx.scissorTestEnabled = GL_TRUE;
        }
    } else if (cap == GL_CULL_FACE) {
        ctx.cullEnabled = GL_TRUE;
    } else if (cap == GL_POLYGON_OFFSET_FILL) {
        ctx.polygonOffsetFillEnabled = GL_TRUE;
    } else if (cap == GL_POLYGON_OFFSET_LINE) {
        ctx.polygonOffsetLineEnabled = GL_TRUE;
    } else if (cap == GL_POLYGON_OFFSET_POINT) {
        ctx.polygonOffsetPointEnabled = GL_TRUE;
    } else if (cap == GL_SAMPLE_COVERAGE) {
        ctx.sampleCoverageEnabled = GL_TRUE;
    } else if (cap == GL_PRIMITIVE_RESTART) {
        ctx.primitiveRestartEnabled = GL_TRUE;
    } else if (cap == GL_COLOR_LOGIC_OP) {
        gl4metalColorLogicOperationEnabled = GL_TRUE;
        ctx.pipelineState = nil;
        [pipelineLayoutCache removeAllObjects];
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDisable(GLenum cap) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (cap == GL_BLEND) {
        if (ctx.blendEnabled) { ctx.blendEnabled = GL_FALSE; ctx.pipelineState = nil; }
    } else if (cap == GL_DEPTH_TEST) {
        if (ctx.depthTestEnabled) {
            ctx.depthTestEnabled = GL_FALSE;
            updateDepthStencilState();
        }
    } else if (cap == GL_STENCIL_TEST) {
        if (ctx.stencilTestEnabled) {
            ctx.stencilTestEnabled = GL_FALSE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (ctx.scissorTestEnabled) {
            ctx.scissorTestEnabled = GL_FALSE;
        }
    } else if (cap == GL_CULL_FACE) {
        ctx.cullEnabled = GL_FALSE;
    } else if (cap == GL_POLYGON_OFFSET_FILL) {
        ctx.polygonOffsetFillEnabled = GL_FALSE;
    } else if (cap == GL_POLYGON_OFFSET_LINE) {
        ctx.polygonOffsetLineEnabled = GL_FALSE;
    } else if (cap == GL_POLYGON_OFFSET_POINT) {
        ctx.polygonOffsetPointEnabled = GL_FALSE;
    } else if (cap == GL_SAMPLE_COVERAGE) {
        ctx.sampleCoverageEnabled = GL_FALSE;
    } else if (cap == GL_PRIMITIVE_RESTART) {
        ctx.primitiveRestartEnabled = GL_FALSE;
    } else if (cap == GL_COLOR_LOGIC_OP) {
        gl4metalColorLogicOperationEnabled = GL_FALSE;
        ctx.pipelineState = nil;
        [pipelineLayoutCache removeAllObjects];
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsEnabled(GLenum cap) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    switch (cap) {
        case GL_BLEND: return ctx.blendEnabled;
        case GL_DEPTH_TEST: return ctx.depthTestEnabled;
        case GL_STENCIL_TEST: return ctx.stencilTestEnabled;
        case GL_SCISSOR_TEST: return ctx.scissorTestEnabled;
        case GL_CULL_FACE: return ctx.cullEnabled;
        case GL_POLYGON_OFFSET_FILL: return ctx.polygonOffsetFillEnabled;
        case GL_POLYGON_OFFSET_LINE: return ctx.polygonOffsetLineEnabled;
        case GL_POLYGON_OFFSET_POINT: return ctx.polygonOffsetPointEnabled;
        case GL_SAMPLE_COVERAGE: return ctx.sampleCoverageEnabled;
        case GL_PRIMITIVE_RESTART: return ctx.primitiveRestartEnabled;
        case GL_COLOR_LOGIC_OP: return gl4metalColorLogicOperationEnabled;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return GL_FALSE;
    }
}

void APIENTRY glEnablei(GLenum cap, GLuint index) {
    if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (cap != GL_BLEND && cap != GL_SCISSOR_TEST) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    glEnable(cap);
}

void APIENTRY glDisablei(GLenum cap, GLuint index) {
    if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (cap != GL_BLEND && cap != GL_SCISSOR_TEST) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    glDisable(cap);
}

GLboolean APIENTRY glIsEnabledi(GLenum cap, GLuint index) {
    if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return GL_FALSE; }
    if (cap != GL_BLEND && cap != GL_SCISSOR_TEST) {
        gl4metalSetError(GL_INVALID_ENUM);
        return GL_FALSE;
    }
    return glIsEnabled(cap);
}

void APIENTRY glColorMaski(GLuint index, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    glColorMask(red, green, blue, alpha);
}

void APIENTRY glGetBooleani_v(GLenum target, GLuint index, GLboolean *data) {
    if (!data) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
    if (!ctx) { gl4metalSetError(GL_INVALID_OPERATION); return; }
    if (target == GL_COLOR_WRITEMASK) {
        data[0] = ctx.colorMaskRed; data[1] = ctx.colorMaskGreen;
        data[2] = ctx.colorMaskBlue; data[3] = ctx.colorMaskAlpha;
    } else if (target == GL_BLEND) {
        data[0] = ctx.blendEnabled;
    } else {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDepthFunc(GLenum func) {
    if (func != GL_NEVER && func != GL_LESS && func != GL_EQUAL && func != GL_LEQUAL &&
        func != GL_GREATER && func != GL_NOTEQUAL && func != GL_GEQUAL && func != GL_ALWAYS) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    if (ctx.depthFunc != func) {
        ctx.depthFunc = func;
        updateDepthStencilState();
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDepthMask(GLboolean flag) {
    if (ctx.depthWriteEnabled != flag) {
        ctx.depthWriteEnabled = flag;
        updateDepthStencilState();
    }
}
