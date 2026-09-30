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
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

// Resource hazards: OpenGL lets the CPU rewrite a buffer while the previous
// frame is still using that GL object. Metal cannot safely observe arbitrary
// CPU mutation of an in-flight MTLBuffer. Track GL buffer IDs used by each
// command buffer so glBufferSubData/glMapBuffer can copy-on-write when needed.
static NSMutableDictionary<NSNumber *, NSNumber *> *gl4metalBufferInFlightCounts = nil;
static NSMutableSet<NSNumber *> *gl4metalCurrentCommandBufferBufferUses = nil;

static BOOL gl4metalSafeResourceUpdatesEnabled(void) {
    const char *v = getenv("GL4METAL_SAFE_RESOURCE_UPDATES");
    if (!v || !*v) return YES;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0);
}

static BOOL gl4metalBufferIsInFlight(GLuint bufferID) {
    if (!bufferID || !gl4metalBufferInFlightCounts) return NO;
    @synchronized(gl4metalBufferInFlightCounts) {
        return [gl4metalBufferInFlightCounts[@(bufferID)] unsignedIntegerValue] > 0;
    }
}

static void gl4metalMarkBufferUsed(GLuint bufferID) {
    if (!bufferID || !gl4metalCurrentCommandBufferBufferUses || !gl4metalBufferInFlightCounts) return;
    NSNumber *key = @(bufferID);
    if ([gl4metalCurrentCommandBufferBufferUses containsObject:key]) return;
    [gl4metalCurrentCommandBufferBufferUses addObject:key];
    @synchronized(gl4metalBufferInFlightCounts) {
        NSUInteger count = [gl4metalBufferInFlightCounts[key] unsignedIntegerValue];
        gl4metalBufferInFlightCounts[key] = @(count + 1);
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
static NSMutableSet<NSNumber *> *bufferObjectNames = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *mappedBufferPointers = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *bufferUsages = nil;
static void gl4metalBindProgramResources(id<MTLRenderCommandEncoder> encoder, GLuint program);
static MTLBlendFactor gl4metalBlendFactor(GLenum factor);
static MTLBlendOperation gl4metalBlendOperation(GLenum equation);
static NSData *gl4metalUniformDataForName(NSString *name, GLuint program, NSUInteger length);
static GLint gl4metalFindUniformLocationByName(NSString *name, GLuint program);
static void gl4metalEncodePendingClear(void);

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
    // OpenGL errors remain pending until glGetError consumes them. A later
    // successful call must not erase an error the application has not read.
    if (error == GL_NO_ERROR || lastGL4MetalError != GL_NO_ERROR) return;
    lastGL4MetalError = error;
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
    ctx.clearStencil = 0;
    ctx.stencilWriteMask = 0xffffffffu;
    ctx.colorMaskRed = GL_TRUE;
    ctx.colorMaskGreen = GL_TRUE;
    ctx.colorMaskBlue = GL_TRUE;
    ctx.colorMaskAlpha = GL_TRUE;
    ctx.drawBuffer = GL_BACK;
    ctx.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    ctx.clearDepth = 1.0;
    ctx.pendingClearFlags = 0;
    ctx.pendingClearFramebuffer = 0;

    gl4metalBufferInFlightCounts = [NSMutableDictionary dictionary];
    gl4metalCurrentCommandBufferBufferUses = nil;

    bufferObjects = [NSMutableDictionary dictionary];
    bufferSizes = [NSMutableDictionary dictionary];
    bufferObjectNames = [NSMutableSet set];
    mappedBufferPointers = [NSMutableDictionary dictionary];
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
    uniformLocationNames = [NSMutableDictionary dictionary];
    uniformLocationPrograms = [NSMutableDictionary dictionary];
    programAttribBindings = [NSMutableDictionary dictionary];
    pipelineLayoutCache = [NSMutableDictionary dictionary];
    currentProgram = 0;
    lastGL4MetalError = GL_NO_ERROR;
    gl4metalInitTextureState();
    gl4metalInitFramebufferState();

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
// buffers avoids allocating a fresh 256 KiB MTLBuffer for every frame while
// preserving GPU lifetime safety: a buffer is returned only after its command
// buffer has completed.
static NSMutableArray<id<MTLBuffer>> *gl4metalUniformArenaPool = nil;

static id<MTLCommandBuffer> gl4metalCreateCommandBuffer(void) {
    if (!ctx || !ctx.commandQueue) return nil;
    id<MTLCommandBuffer> cb = [ctx.commandQueue commandBuffer];
    if (!cb) return nil;

    // Uniform storage is owned by the command-buffer generation. Reuse a
    // completed arena when possible; every in-flight command buffer still owns
    // its arena strongly through the completion-handler capture.
    if (!gl4metalUniformArenaPool) {
        gl4metalUniformArenaPool = [NSMutableArray array];
    }
    NSMutableArray<id<MTLBuffer>> *uniformArenas = [NSMutableArray array];
    NSMutableSet<NSNumber *> *usedGLBuffers = [NSMutableSet set];
    gl4metalCurrentCommandBufferBufferUses = usedGLBuffers;
    @synchronized(gl4metalUniformArenaPool) {
        if (gl4metalUniformArenaPool.count > 0) {
            id<MTLBuffer> recycled = gl4metalUniformArenaPool.lastObject;
            [gl4metalUniformArenaPool removeLastObject];
            if (recycled) [uniformArenas addObject:recycled];
        }
    }
    ctx.uniformArenaBuffers = uniformArenas;
    ctx.uniformArenaOffset = 0;

    cb.label = [NSString stringWithFormat:@"gl4metal frame"];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        // The command buffer is finished, so its uniform buffers are safe to
        // recycle for a future frame.
        @synchronized(gl4metalBufferInFlightCounts) {
            for (NSNumber *bufferKey in usedGLBuffers) {
                NSUInteger count = [gl4metalBufferInFlightCounts[bufferKey] unsignedIntegerValue];
                if (count > 1) gl4metalBufferInFlightCounts[bufferKey] = @(count - 1);
                else [gl4metalBufferInFlightCounts removeObjectForKey:bufferKey];
            }
        }
        @synchronized(gl4metalUniformArenaPool) {
            [gl4metalUniformArenaPool addObjectsFromArray:uniformArenas];
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
    printf("[gl4metal] Committing Metal command buffer\n");
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
    if (ctx.metalLayer) ctx.metalLayer.maximumDrawableCount = interval == 0 ? 2 : 3;
}

void APIENTRY glClear(GLbitfield mask) {
    if (!ctx) return;
    ctx.pendingClearFlags |= mask;
    ctx.pendingClearFramebuffer = ctx.currentFramebuffer;
}

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
    if (ctx) ctx.clearColor = MTLClearColorMake(red, green, blue, alpha);
}

void APIENTRY glClearDepth(GLclampd depth) {
    if (ctx) ctx.clearDepth = depth;
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
        if (!colorTexture) {
            printf("[gl4metal] ERROR: FBO %u has no color attachment for draw path\n", ctx.currentFramebuffer);
            return nil;
        }
        targetWidth = colorTexture.width;
        targetHeight = colorTexture.height;
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
    passDesc.colorAttachments[0].texture = colorTexture;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    GLbitfield effectiveClearFlags =
        (ctx.pendingClearFramebuffer == ctx.currentFramebuffer)
            ? ctx.pendingClearFlags : 0;
    passDesc.colorAttachments[0].loadAction = (effectiveClearFlags & GL_COLOR_BUFFER_BIT)
        ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.colorAttachments[0].clearColor = ctx.clearColor;

    if (depthTexture) {
        passDesc.depthAttachment.texture = depthTexture;
        passDesc.depthAttachment.storeAction = MTLStoreActionStore;
        passDesc.depthAttachment.loadAction = (effectiveClearFlags & GL_DEPTH_BUFFER_BIT)
            ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.depthAttachment.clearDepth = ctx.clearDepth;
    }

    if (stencilTexture &&
        (stencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ||
         stencilTexture.pixelFormat == MTLPixelFormatStencil8)) {
        passDesc.stencilAttachment.texture = stencilTexture;
        passDesc.stencilAttachment.storeAction = MTLStoreActionStore;
        passDesc.stencilAttachment.loadAction = (effectiveClearFlags & GL_STENCIL_BUFFER_BIT)
            ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.stencilAttachment.clearStencil = (NSUInteger)ctx.clearStencil;
    }

    if (effectiveClearFlags != 0) {
        ctx.pendingClearFlags &= ~effectiveClearFlags;
        if (ctx.pendingClearFlags == 0) ctx.pendingClearFramebuffer = ctx.currentFramebuffer;
    }
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
    if (ctx) ctx.viewport = (gl4metalRect){x, y, width, height};
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
        0.0,
        1.0
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

void APIENTRY glLineWidth(GLfloat width) {
    if (width <= 0.0f) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    ctx.lineWidth = width;
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

void APIENTRY glClearStencil(GLint s) {
    ctx.clearStencil = s;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glStencilMask(GLuint mask) {
    ctx.stencilWriteMask = mask;
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
    updateDepthStencilState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    ctx.colorMaskRed = red;
    ctx.colorMaskGreen = green;
    ctx.colorMaskBlue = blue;
    ctx.colorMaskAlpha = alpha;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDrawBuffer(GLenum buf) {
    if (buf != GL_BACK && buf != GL_NONE) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.drawBuffer = buf;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glHint(GLenum target, GLenum mode) {
    (void)target;
    (void)mode;
    gl4metalSetError(GL_NO_ERROR);
}

static void applyRasterState(id<MTLRenderCommandEncoder> encoder) {
    if (!ctx || !encoder) return;

    MTLCullMode cullMode = MTLCullModeNone;
    if (ctx.cullEnabled) {
        cullMode = ctx.cullFace == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    }
    [encoder setCullMode:cullMode];
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
    for (GLsizei i = 0; i < n; i++) buffers[i] = nextBufferId++;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays) {
    if (n < 0 || (n > 0 && !arrays)) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        GLuint id = nextVAOId++;
        arrays[i] = id;
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

void APIENTRY glBindBuffer(GLenum target, GLuint buffer) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (buffer != 0) [bufferObjectNames addObject:@(buffer)];
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
    (void)index; (void)offset; (void)size;
    glBindBuffer(target, buffer);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
    (void)index;
    glBindBuffer(target, buffer);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);

    if (currentBound == 0) {
        printf("[gl4metal] ERROR: glBufferData called with no bound buffer for target 0x%X\n", target);
        return;
    }

    if (size <= 0) {
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

    id<MTLBuffer> oldBuffer = bufferObjects[@(currentBound)];
    if (oldBuffer) {
        [convertedIndexBuffers removeAllObjects];
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
        [bufferObjectNames removeObject:@(buffer)];
        [bufferObjects removeObjectForKey:@(buffer)];
        [bufferSizes removeObjectForKey:@(buffer)];
        [bufferUsages removeObjectForKey:@(buffer)];
        [mappedBufferPointers removeObjectForKey:@(buffer)];
        [convertedIndexBuffers removeAllObjects];
        for (NSNumber *target in [ctx.boundBuffers.allKeys copy]) {
            if ([ctx.boundBuffers[target] unsignedIntValue] == buffer) ctx.boundBuffers[target] = @(0);
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
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (!buffer || !data) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (offset < 0 || size < 0 || (size_t)offset > (size_t)[buffer length] || (size_t)size > (size_t)[buffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glBufferSubData OOB: buffer=%u offset=%ld size=%ld length=%lu\n", currentBound, (long)offset, (long)size, (unsigned long)[buffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

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
    [convertedIndexBuffers removeAllObjects];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void *data) {
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
    [convertedIndexBuffers removeAllObjects];
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

    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    switch (target) {
        case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING:
            data[0] = (GLint)(vao && index < 16 ? vao->attribs[index].boundVBO : 0);
            break;
        default:
            data[0] = 0;
            break;
    }

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
    gl4metalSetError(GL_NO_ERROR);
    return mapped;
}

void *APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    (void)access;
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return NULL;
    }
    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) { gl4metalSetError(GL_INVALID_OPERATION); return NULL; }

    if (offset < 0 || length < 0 || (size_t)offset > (size_t)[mtlBuffer length] || (size_t)length > (size_t)[mtlBuffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glMapBufferRange OOB: buffer=%u offset=%ld length=%ld bufferLength=%lu\n", currentBound, (long)offset, (long)length, (unsigned long)[mtlBuffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
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
    gl4metalSetError(GL_NO_ERROR);
    void *mapped = (void *)((char *)[mtlBuffer contents] + offset);
    mappedBufferPointers[@(currentBound)] = [NSValue valueWithPointer:mapped];
    return mapped;
}

GLboolean APIENTRY glUnmapBuffer(GLenum target) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    if (currentBound == 0 || !mappedBufferPointers[@(currentBound)]) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    [mappedBufferPointers removeObjectForKey:@(currentBound)];
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
    for (GLsizei i = 0; i < n; i++) {
        if (ctx.currentVAO == arrays[i]) glBindVertexArray(0);
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

    std::string source;
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;

        if (length != nullptr && length[i] >= 0) {
            source.append(string[i], static_cast<size_t>(length[i]));
        } else {
            source.append(string[i]);
        }

        // GLSL line comments run until a newline. glShaderSource() may be
        // called with a source string that does not end in one, so always
        // terminate each input string before concatenating the next one.
        source.push_back('\n');
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

    std::string mslSource = transpileGLSLToMSL(shaderData.type, shaderData.glslSource);
    if (mslSource.empty()) {
        shaderData.compiled = false;
        shaderCompileStatus[@(shader)] = @(GL_FALSE);
        shaderInfoLog[@(shader)] = @"Failed to transpile GLSL to MSL";
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSString *sourceCode = [NSString stringWithUTF8String:mslSource.c_str()];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_1;

    NSError *error = nil;
    shaderData.mtlLibrary = [ctx.device newLibraryWithSource:sourceCode options:options error:&error];

    if (error || !shaderData.mtlLibrary) {
        printf("[gl4metal] Shader compilation error: %s\n", error.localizedDescription.UTF8String);
        shaderData.compiled = false;
        shaderCompileStatus[@(shader)] = @(GL_FALSE);
        shaderInfoLog[@(shader)] = error.localizedDescription ?: @"Unknown error";
        gl4metalSetError(GL_INVALID_OPERATION);
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
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

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

    if (program == 0) {
        currentProgram = 0;
        ctx.pipelineState = defaultFallbackPipeline;
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
    if (!ctx || program == 0 || name == nil) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    GLint location = gl4metalResolveLocationFromName(program, name);
    uniformLocationNames[@(location)] = [NSString stringWithUTF8String:name];
    uniformLocationPrograms[@(location)] = @(program);
    gl4metalSetError(GL_NO_ERROR);
    return location;
}

void APIENTRY glUniform1f(GLint location, GLfloat v0) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1fValues[@(location)] = [NSValue valueWithBytes:&v0 objCType:@encode(GLfloat)];
    gl4metalTraceUniformUpdate(location, "1f", &v0, 1);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1i(GLint location, GLint v0) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1iValues[@(location)] = @(v0);
    if (gl4metalResourceTraceEnabled()) {
        gl4metalTraceUniformUpdateInt(location, "1i", &v0, 1);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2f(GLint location, GLfloat v0, GLfloat v1) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[2] = { v0, v1 };
    uniform2fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[2])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2i(GLint location, GLint v0, GLint v1) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform2iValues[@(location)] = [NSNumber numberWithLongLong:(long long)((v0 << 16) | (v1 & 0xffff))];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[3] = { v0, v1, v2 };
    uniform3fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[3])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3i(GLint location, GLint v0, GLint v1, GLint v2) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLint values[3] = { v0, v1, v2 };
    uniform3iValues[@(location)] = [NSNumber numberWithLongLong:(long long)((uint32_t)v0 | ((uint32_t)v1 << 8) | ((uint32_t)v2 << 16))];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    GLfloat values[4] = { v0, v1, v2, v3 };
    uniform4fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[4])];
    gl4metalTraceUniformUpdate(location, "4f", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

// The LWJGL/Minecraft path may use the vector upload entry points instead of
// the scalar convenience calls. Keep the first vector in the same uniform
// store consumed by the Metal resource bridge.
void APIENTRY glUniform2fv(GLint location, GLsizei count, const GLfloat *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat values[2] = { value[0], value[1] };
    uniform2fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[2])];
    gl4metalTraceUniformUpdate(location, "2fv", values, 2);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3fv(GLint location, GLsizei count, const GLfloat *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat values[3] = { value[0], value[1], value[2] };
    uniform3fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[3])];
    gl4metalTraceUniformUpdate(location, "3fv", values, 3);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4fv(GLint location, GLsizei count, const GLfloat *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat values[4] = { value[0], value[1], value[2], value[3] };
    uniform4fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[4])];
    gl4metalTraceUniformUpdate(location, "4fv", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform4iValues[@(location)] = @(v0);
    GLint values[4] = { v0, v1, v2, v3 };
    gl4metalTraceUniformUpdateInt(location, "4i", values, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform2iv(GLint location, GLsizei count, const GLint *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    uniform2iValues[@(location)] = @(value[0]);
    gl4metalTraceUniformUpdateInt(location, "2iv", value, 2);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform3iv(GLint location, GLsizei count, const GLint *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    uniform3iValues[@(location)] = @(value[0]);
    gl4metalTraceUniformUpdateInt(location, "3iv", value, 3);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4iv(GLint location, GLsizei count, const GLint *value) {
    if (!ctx || count <= 0 || !value) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    uniform4iValues[@(location)] = @(value[0]);
    gl4metalTraceUniformUpdateInt(location, "4iv", value, 4);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1fv(GLint location, GLsizei count, const GLfloat *value) {
    if (!ctx || count <= 0 || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
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
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1iv(GLint location, GLsizei count, const GLint *value) {
    if (!ctx || count <= 0 || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLint v0 = count > 0 ? value[0] : 0;
    uniform1iValues[@(location)] = @(v0);
    if (gl4metalResourceTraceEnabled()) {
        gl4metalTraceUniformUpdateInt(location, "1iv", &v0, 1);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[4] = { value[0], value[1], value[2], value[3] };
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[4])];
    NSString *currentName = uniformLocationNames[@(location)];
    if ([currentName isEqualToString:@"uMVP"] || [currentName isEqualToString:@"uModelViewProjectionMatrix"] || [currentName isEqualToString:@"uProjection"]) {
        gl4metalUpdateFallbackMVP(value, count * 2 * 2);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[9] = { 0 };
    for (GLsizei i = 0; i < MIN(count * 3 * 3, 9); ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[9])];
    NSString *currentName = uniformLocationNames[@(location)];
    if ([currentName isEqualToString:@"uMVP"] || [currentName isEqualToString:@"uModelViewProjectionMatrix"] || [currentName isEqualToString:@"uProjection"]) {
        gl4metalUpdateFallbackMVP(value, count * 3 * 3);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[16] = { 0 };
    for (GLsizei i = 0; i < 16; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[16])];
    NSString *currentName = uniformLocationNames[@(location)];
    if ([currentName isEqualToString:@"uMVP"] || [currentName isEqualToString:@"uModelViewProjectionMatrix"] || [currentName isEqualToString:@"uProjection"]) {
        gl4metalUpdateFallbackMVP(value, 16);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[6] = { 0 };
    for (GLsizei i = 0; i < 6; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[6])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[6] = { 0 };
    for (GLsizei i = 0; i < 6; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[6])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[8] = { 0 };
    for (GLsizei i = 0; i < 8; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[8])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[8] = { 0 };
    for (GLsizei i = 0; i < 8; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[8])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[12] = { 0 };
    for (GLsizei i = 0; i < 12; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[12])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    (void)count; (void)transpose;
    if (!ctx || value == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    GLfloat matrix[12] = { 0 };
    for (GLsizei i = 0; i < 12; ++i) matrix[i] = value[i];
    uniformMatrix4fvValues[@(location)] = [NSValue valueWithBytes:matrix objCType:@encode(GLfloat[12])];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteProgram(GLuint program) {
    if (program == 0) return;
    [programShaders removeObjectForKey:@(program)];
    [programLinkStatus removeObjectForKey:@(program)];
    [programInfoLog removeObjectForKey:@(program)];
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
            printf("[gl4metal] glDeleteShader(%u): shader remains alive because program %u still references it\n", shader, programId.unsignedIntValue);
            gl4metalSetError(GL_NO_ERROR);
            return;
        }
    }

    [shaderTypes removeObjectForKey:@(shader)];
    shaderSources.erase(shader);
    [shaderCompileStatus removeObjectForKey:@(shader)];
    [shaderInfoLog removeObjectForKey:@(shader)];
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
        case GL_INFO_LOG_LENGTH: {
            NSString *message = programInfoLog[@(program)] ?: @"";
            params[0] = message.length ? (GLint)strlen(message.UTF8String) + 1 : 0;
            break;
        }
        case GL_ATTACHED_SHADERS:
            params[0] = (GLint)[programShaders[@(program)] count];
            break;
        case GL_ACTIVE_UNIFORMS:
            params[0] = 4;
            break;
        case GL_ACTIVE_ATTRIBUTES:
            params[0] = 8;
            break;
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
        case GL_DEPTH_FUNC:
            data[0] = (GLint)ctx.depthFunc;
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
        case GL_POINT_SIZE:
            data[0] = ctx.pointSize;
            break;
        case GL_BLEND_COLOR:
            data[0] = ctx.blendConstantRed; data[1] = ctx.blendConstantGreen;
            data[2] = ctx.blendConstantBlue; data[3] = ctx.blendConstantAlpha;
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
        case GL_POINT_SIZE:
            data[0] = ctx.pointSize;
            break;
        case GL_BLEND_COLOR:
            data[0] = ctx.blendConstantRed; data[1] = ctx.blendConstantGreen;
            data[2] = ctx.blendConstantBlue; data[3] = ctx.blendConstantAlpha;
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
    if (!ctx || program == 0 || name == NULL) {
        gl4metalSetError(GL_INVALID_OPERATION);
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
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    if (program != 0) {
        NSMutableDictionary<NSNumber *, NSString *> *bindings = programAttribBindings[@(program)];
        if (bindings) {
            NSString *attributeName = [NSString stringWithUTF8String:name];
            for (NSNumber *index in bindings) {
                if ([bindings[index] isEqualToString:attributeName]) {
                    gl4metalSetError(GL_NO_ERROR);
                    return (GLint)[index intValue];
                }
            }
        }
    }

    NSString *attributeName = [NSString stringWithUTF8String:name];
    NSArray *knownNames = @[@"aPosition", @"aPos", @"position", @"vertex",
                            @"aNormal", @"normal", @"aTexCoord", @"aUV",
                            @"aColor", @"color"];
    NSInteger index = [knownNames indexOfObject:attributeName];
    gl4metalSetError(GL_NO_ERROR);
    return (index == NSNotFound) ? -1 : (GLint)index;
}

void APIENTRY glVertexAttrib1f(GLuint index, GLfloat x) {
    if (index >= 16) return;
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
    if (index >= 16) return;
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
    if (index >= 16) return;
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
    if (index >= 16) return;
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

void APIENTRY glGetUniformfv(GLuint program, GLint location, GLfloat *params) {
    (void)program;
    if (!ctx || !params) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSValue *value = uniform1fValues[@(location)];
    if (value) {
        GLfloat v = 0.0f;
        [value getValue:&v];
        params[0] = v;
    } else {
        params[0] = 0.0f;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetUniformiv(GLuint program, GLint location, GLint *params) {
    (void)program;
    if (!ctx || !params) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSNumber *value = uniform1iValues[@(location)];
    params[0] = value ? [value intValue] : 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetUniformuiv(GLuint program, GLint location, GLuint *params) {
    (void)program;
    if (!ctx || !params) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSNumber *value = uniform1iValues[@(location)];
    params[0] = value ? (GLuint)[value unsignedIntValue] : 0;
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
    (void)bufSize;
    (void)length;
    (void)binaryFormat;
    (void)binary;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glProgramBinary(GLuint program, GLenum binaryFormat, const void *binary, GLsizei length) {
    (void)program;
    (void)binaryFormat;
    (void)binary;
    (void)length;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glProgramParameteri(GLuint program, GLenum pname, GLint value) {
    (void)program;
    (void)pname;
    (void)value;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint *params) {
    (void)pipeline;
    (void)pname;
    if (params) params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindProgramPipeline(GLuint pipeline) {
    (void)pipeline;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteProgramPipelines(GLsizei n, const GLuint *pipelines) {
    (void)n;
    (void)pipelines;
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

// Collect index diagnostics in the same pass already required for max-index bounds
// validation. This keeps the trace effectively free when disabled and avoids a
// second walk over large Minecraft index buffers.
static BOOL gl4metalFindIndexStats(id<MTLBuffer> indexBuffer, GLenum indexType, NSUInteger indexOffset, NSUInteger count,
                                   NSUInteger *outMinIndex, NSUInteger *outMaxIndex,
                                   NSUInteger sampleValues[8], NSUInteger *outSampleCount,
                                   uint64_t *outHash) {
    if (!indexBuffer || !outMinIndex || !outMaxIndex) return NO;
    NSUInteger minIndex = SIZE_MAX;
    NSUInteger maxIndex = 0;
    NSUInteger sampleCount = 0;
    uint64_t hash = 1469598103934665603ULL; // FNV-1a offset basis
    for (NSUInteger i = 0; i < count; ++i) {
        NSUInteger value = 0;
        if (!gl4metalReadIndexValue(indexBuffer, indexType, indexOffset, i, &value)) {
            printf("[gl4metal][BOUNDS] cannot inspect index %lu\n", (unsigned long)i);
            return NO;
        }
        if (value < minIndex) minIndex = value;
        if (value > maxIndex) maxIndex = value;
        if (sampleValues && sampleCount < 8) sampleValues[sampleCount++] = value;
        hash ^= (uint64_t)value;
        hash *= 1099511628211ULL; // FNV-1a prime
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
    NSUInteger outBytes = outCount * outIndexSize;
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
    return YES;
}

static BOOL gl4metalDrawTraceEnabledForProgram(GLuint program) {
    return program == 53 && getenv("GL4METAL_DRAW_TRACE");
}

static void gl4metalDrawArraysInternal(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    if (!ctx || count <= 0 || instancecount <= 0) return;
    if (gl4metalDrawTraceEnabledForProgram(currentProgram)) {
        printf("[gl4metal][DRAW_TRACE_BEGIN] kind=arrays program=%u vao=%u fbo=%u mode=0x%X first=%d count=%d instances=%d\n",
               currentProgram, ctx.currentVAO, ctx.currentFramebuffer, mode, first, count, instancecount);
        fflush(stdout);
    }
    if (gl4metalGPUIsolationEnabled && gl4metalGPUIsolationFailed) {
        printf("[gl4metal][GPU-DEBUG] draw skipped after previous GPU failure\n");
        return;
    }
    if (first < 0) { printf("[gl4metal][BOUNDS] glDrawArrays negative first=%d\n", first); return; }
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
        printf("[gl4metal] ERROR: Failed to create render pass descriptor for glDrawArrays\n");
        return;
    }

    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        printf("[gl4metal] ERROR: Failed to create MTLRenderCommandEncoder\n");
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
    [encoder setStencilReferenceValue:(uint32_t)MAX(ctx.stencilRef, 0)];

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
        id<MTLBuffer> fan = [ctx.device newBufferWithLength:indexCount * indexSize options:MTLResourceStorageModeShared];
        if (fan) {
            if (indexType == MTLIndexTypeUInt16) {
                uint16_t *dst = (uint16_t *)fan.contents;
                for (NSUInteger i = 0; i < triCount; ++i) {
                    dst[i * 3 + 0] = (uint16_t)first;
                    dst[i * 3 + 1] = (uint16_t)(first + (GLint)i + 1);
                    dst[i * 3 + 2] = (uint16_t)(first + (GLint)i + 2);
                }
            } else {
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
        arena = [ctx.device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
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
        // Matrix uniforms: the reflection dataSize is the Metal ABI size.
        // mat4 is 64 bytes; mat3 is 48 bytes because each column occupies a
        // 16-byte slot in Metal's constant-buffer ABI.
        NSValue *matrix = uniformMatrix4fvValues[@(location)];
        if (matrix) {
            if (length >= 64) {
                float values[16] = {0};
                if (!gl4metalCopyNSValueBytes(matrix, values, sizeof(values))) return data;
                memcpy(data.mutableBytes, values, MIN(length, sizeof(values)));
                return data;
            }
            if (length >= 48) {
                float values[9] = {0};
                if (!gl4metalCopyNSValueBytes(matrix, values, sizeof(values))) return data;
                float *dst = (float *)data.mutableBytes;
                for (int c = 0; c < 3; ++c) {
                    dst[c * 4 + 0] = values[c * 3 + 0];
                    dst[c * 4 + 1] = values[c * 3 + 1];
                    dst[c * 4 + 2] = values[c * 3 + 2];
                    dst[c * 4 + 3] = 0.0f;
                }
                return data;
            }
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

static void gl4metalDrawElementsInternal(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    if (!ctx || count <= 0 || instancecount <= 0) return;
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
    gl4metalMarkBufferUsed(elementVBO);
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
    if (!gl4metalFindIndexStats(indexBuffer, type, (NSUInteger)rawIndexOffset, (NSUInteger)count,
                                &minIndex, &maxIndex, sampleValues, &sampleCount, &indexHash)) {
        printf("[gl4metal][BOUNDS] Aborting glDrawElements: cannot inspect index buffer\n");
        return;
    }
    if (getenv("GL4METAL_DEBUG")) {
        printf("[gl4metal][BOUNDS] index check: EBO=%u offset=%ld count=%d type=0x%X width=%zu buffer=%lu minIndex=%lu maxIndex=%lu\n",
               elementVBO, (long)rawIndexOffset, count, type, indexWidth,
               (unsigned long)[indexBuffer length], (unsigned long)minIndex, (unsigned long)maxIndex);
    }

    // Opt-in indexed-draw trace used to diagnose stretched triangles / bad VAO-EBO
    // state without permanently flooding the normal render log. Each unique
    // draw signature+index-data hash is printed once per process.
    if (getenv("GL4METAL_INDEX_TRACE")) {
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

    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        printf("[gl4metal] ERROR: Failed to create MTLRenderCommandEncoder\n");
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
    [encoder setStencilReferenceValue:(uint32_t)MAX(ctx.stencilRef, 0)];
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
            if (!vb || !gl4metalValidateIndexedVertexAttribRange(a, vb, maxIndex, (NSUInteger)instancecount, ai)) vertexOK=NO;
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
                convertedIndexBuffers[cacheKey] = tmpBuffer;
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
                              instanceCount:(NSUInteger)instancecount];
        }
    } else {
        MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
        [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset instanceCount:(NSUInteger)instancecount];
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
    gl4metalDrawElementsInternal(mode, count, type, indices, 1);
}

void APIENTRY glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    gl4metalDrawElementsInternal(mode, count, type, indices, instancecount);
}

void APIENTRY glDrawRangeElements(GLenum mode, GLuint start, GLuint end,
                                  GLsizei count, GLenum type, const void *indices) {
    if (start > end || count < 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glDrawElements(mode, count, type, indices);
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
        MTLStencilDescriptor *stencil = [[MTLStencilDescriptor alloc] init];
        stencil.stencilCompareFunction = getMetalCompareFunction(ctx.stencilFunc);
        stencil.stencilFailureOperation = getMetalStencilOperation(ctx.stencilFail);
        stencil.depthFailureOperation = getMetalStencilOperation(ctx.stencilDepthFail);
        stencil.depthStencilPassOperation = getMetalStencilOperation(ctx.stencilDepthPass);
        stencil.readMask = ctx.stencilReadMask;
        stencil.writeMask = ctx.stencilWriteMask;
        desc.frontFaceStencil = stencil;
        desc.backFaceStencil = stencil;
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
    if (targetColorFormat == MTLPixelFormatInvalid) {
        printf("[gl4metal] ERROR: No valid current color render-target format for program %u\n", program);
        return NO;
    }
    pipelineDesc.colorAttachments[0].pixelFormat = targetColorFormat;
    pipelineDesc.colorAttachments[0].writeMask =
        (ctx.colorMaskRed ? MTLColorWriteMaskRed : 0) |
        (ctx.colorMaskGreen ? MTLColorWriteMaskGreen : 0) |
        (ctx.colorMaskBlue ? MTLColorWriteMaskBlue : 0) |
        (ctx.colorMaskAlpha ? MTLColorWriteMaskAlpha : 0);
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
    BOOL effectiveBlendEnabled = ctx.blendEnabled;

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
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return GL_FALSE;
    }
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
