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

gl4metalVertexArray *gl4metalGetCurrentVAO(void);

// V8 uniform/resource binding forward declarations.
// uniformLocationNames is defined in gl4metal_runtime.mm and declared extern by gl4metal_runtime.h.
static NSMutableDictionary<NSNumber *, NSNumber *> *uniformLocationPrograms = nil;
static void gl4metalBindProgramResources(id<MTLRenderCommandEncoder> encoder, GLuint program);
static MTLBlendFactor gl4metalBlendFactor(GLenum factor);
static MTLBlendOperation gl4metalBlendOperation(GLenum equation);

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
    lastGL4MetalError = error;
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
    id<MTLBuffer> buffer = constantVertexAttributeBuffers[@(index)];
    if (!buffer || ![inferredConstantVertexAttributes containsObject:@(index)]) return;
    // The exact format was fixed when the real shader pipeline was built.
    // Repack using the existing 16-byte storage; integer-vs-float is tracked by VAO state.
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) return;
    // Preserve the raw bytes for the current value. For float values this is float32;
    // for integer values it is signed/unsigned 32-bit as required by the shader input.
    if (vao->attribs[index].currentValueIsInteger) {
        if (vao->attribs[index].currentValueIsUnsigned) {
            memcpy([buffer contents], vao->attribs[index].currentUIntValue, 16);
        } else {
            memcpy([buffer contents], vao->attribs[index].currentIntValue, 16);
        }
    } else {
        memcpy([buffer contents], vao->attribs[index].currentValue, 16);
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

    bufferObjects = [NSMutableDictionary dictionary];
    bufferSizes = [NSMutableDictionary dictionary];
    convertedIndexBuffers = [NSMutableDictionary dictionary];
    vaoObjects = [NSMutableDictionary dictionary];
    framebufferObjects = [NSMutableDictionary dictionary];
    renderbufferObjects = [NSMutableDictionary dictionary];
    shaderTypes = [NSMutableDictionary dictionary];
    // shaderSources = [NSMutableDictionary dictionary];
    shaderCompileStatus = [NSMutableDictionary dictionary];
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

    if (!ctx.currentCommandBuffer) return;
    [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
    printf("[gl4metal] Committing Metal command buffer\n");
    [ctx.currentCommandBuffer commit];
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
    [encoder setFrontFacingWinding:ctx.frontFace == GL_CW ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [encoder setTriangleFillMode:ctx.polygonMode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
    if (ctx.blendEnabled) {
        [encoder setBlendColorRed:ctx.blendConstantRed green:ctx.blendConstantGreen blue:ctx.blendConstantBlue alpha:ctx.blendConstantAlpha];
    }
}

void APIENTRY glGenBuffers(GLsizei n, GLuint *buffers) {
    if (!buffers || n < 0) return;
    for (GLsizei i = 0; i < n; i++) buffers[i] = nextBufferId++;
}

void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays) {
    if (!arrays || n < 0) return;
    for (GLsizei i = 0; i < n; i++) {
        GLuint id = nextVAOId++;
        arrays[i] = id;
        gl4metalVertexArray *vao = [[gl4metalVertexArray alloc] init];
        vao.id = id;
        for (NSUInteger a = 0; a < 16; a++) {
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
        vaoObjects[@(id)] = vao;
    }
}

gl4metalVertexArray *gl4metalGetCurrentVAO(void) {
    if (!ctx || ctx.currentVAO == 0) return nil;
    return vaoObjects[@(ctx.currentVAO)];
}

static GLuint gl4metalGetBoundBufferForTarget(GLenum target) {
    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (vao) {
        if (target == GL_ARRAY_BUFFER) return vao->arrayBufferID;
        if (target == GL_ELEMENT_ARRAY_BUFFER) return vao->elementArrayBufferID;
    }

    NSNumber *bound = ctx.boundBuffers[@(target)];
    return bound ? [bound unsignedIntValue] : 0;
}

void APIENTRY glBindBuffer(GLenum target, GLuint buffer) {
    ctx.boundBuffers[@(target)] = @(buffer);

    if (ctx.currentVAO != 0) {
        gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
        if (vao) {
            if (target == GL_ARRAY_BUFFER) {
                vao->arrayBufferID = buffer;
            } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
                vao->elementArrayBufferID = buffer;
            }
        }
    }
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
    (void)usage;
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);

    if (currentBound == 0) {
        printf("[gl4metal] ERROR: glBufferData called with no bound buffer for target 0x%X\n", target);
        return;
    }

    if (size <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
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
        [bufferObjects removeObjectForKey:@(buffer)];
        [bufferSizes removeObjectForKey:@(buffer)];
        [convertedIndexBuffers removeAllObjects];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (!buffer || !data) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    if (offset < 0 || size < 0 || (size_t)offset > (size_t)[buffer length] || (size_t)size > (size_t)[buffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glBufferSubData OOB: buffer=%u offset=%ld size=%ld length=%lu\n", currentBound, (long)offset, (long)size, (unsigned long)[buffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    memcpy((char *)[buffer contents] + offset, data, (size_t)size);
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

    switch (pname) {
        case GL_BUFFER_SIZE:
            params[0] = (GLint)(buffer ? [buffer length] : 0);
            break;
        case GL_BUFFER_USAGE:
            params[0] = GL_STATIC_DRAW;
            break;
        default:
            params[0] = 0;
            break;
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

    switch (pname) {
        case GL_BUFFER_SIZE:
            params[0] = (GLint64)(buffer ? [buffer length] : 0);
            break;
        default:
            params[0] = 0;
            break;
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
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);

    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) {
        printf("[gl4metal] ERROR: glMapbuffer: No MTLBuffer found for ID %u at target 0x%X\n", currentBound, target);
        return NULL;
    }

    return [mtlBuffer contents];
}

void *APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    (void)access;
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) { gl4metalSetError(GL_INVALID_OPERATION); return NULL; }

    if (offset < 0 || length < 0 || (size_t)offset > (size_t)[mtlBuffer length] || (size_t)length > (size_t)[mtlBuffer length] - (size_t)offset) {
        printf("[gl4metal][BOUNDS] glMapBufferRange OOB: buffer=%u offset=%ld length=%ld bufferLength=%lu\n", currentBound, (long)offset, (long)length, (unsigned long)[mtlBuffer length]);
        gl4metalSetError(GL_INVALID_VALUE);
        return NULL;
    }
    gl4metalSetError(GL_NO_ERROR);
    return (void *)((char *)[mtlBuffer contents] + offset);
}

GLboolean APIENTRY glUnmapBuffer(GLenum target) {
    return GL_TRUE;
}

// VAO State
void APIENTRY glBindVertexArray(GLuint array) {
    ctx.currentVAO = array;
    if (array == 0) {
        ctx.boundBuffers[@(GL_ARRAY_BUFFER)] = @(0);
        ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] = @(0);
        return;
    }

    gl4metalVertexArray *vao = vaoObjects[@(array)];
    if (!vao) {
        vao = [[gl4metalVertexArray alloc] init];
        vao.id = array;
        for (NSUInteger a = 0; a < 16; a++) {
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

    ctx.boundBuffers[@(GL_ARRAY_BUFFER)] = @(vao->arrayBufferID);
    ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] = @(vao->elementArrayBufferID);
}

static void gl4metalInvalidateVertexLayoutPipeline(void) {
    if (ctx) ctx.pipelineState = nil;
}

void APIENTRY glEnableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_TRUE;
        gl4metalInvalidateVertexLayoutPipeline();
    }
}

void APIENTRY glDisableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_FALSE;
        gl4metalInvalidateVertexLayoutPipeline();
    }
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
    if (index >= 16) return;

    gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
    if (!vao) {
        printf("[gl4metal] ERROR: glVertexAttribPointer called without bound VAO!\n");
        return;
    }

    GLuint currentVBO = gl4metalGetBoundBufferForTarget(GL_ARRAY_BUFFER);
    if (currentVBO == 0) {
        printf("[gl4metal] ERROR: glVertexAttribPointer called without bound GL_ARRAY_BUFFER\n");
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
        default: elemSize = sizeof(GLfloat); break;
    }

    if (stride <= 0) {
        stride = (GLsizei)(size * (GLint)elemSize);
    }

    id<MTLBuffer> boundBuffer = bufferObjects[@(currentVBO)];
    GLintptr pointerOffset = gl4metalResolvePointerOffset(pointer, boundBuffer);

    vao->attribs[index].enabled = GL_TRUE;
    vao->attribs[index].size = size;
    vao->attribs[index].type = type;
    vao->attribs[index].normalized = normalized;
    vao->attribs[index].integer = GL_FALSE;
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointerOffset = pointerOffset;
    vao->attribs[index].boundVBO = currentVBO;
    gl4metalInvalidateVertexLayoutPipeline();
}

void APIENTRY glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
    if (index >= 16) return;

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

    if (stride <= 0) stride = (GLsizei)(size * (GLint)elemSize);

    id<MTLBuffer> boundBuffer = bufferObjects[@(currentVBO)];
    GLintptr pointerOffset = gl4metalResolvePointerOffset(pointer, boundBuffer);

    vao->attribs[index].enabled = GL_TRUE;
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
    if (!arrays) return;
    for (GLsizei i = 0; i < n; i++) {
        [vaoObjects removeObjectForKey:@(arrays[i])];
    }
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
    programInfoLog[@(shaderId)] = @"";
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
    programInfoLog[@(shader)] = @"";
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
        programInfoLog[@(shader)] = @"Failed to transpile GLSL to MSL";
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
        programInfoLog[@(shader)] = error.localizedDescription ?: @"Unknown error";
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
        programInfoLog[@(shader)] = @"Metal library compiled, but shader entry point 'main0' was not found";
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    programInfoLog[@(shader)] = @"";
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

        descriptor.attributes[index].format = gl4metalVertexFormatForAttrib(attrib);
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
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform1i(GLint location, GLint v0) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1iValues[@(location)] = @(v0);
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
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform4iValues[@(location)] = @(v0);
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
    [programInfoLog removeObjectForKey:@(shader)];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    auto it = shaderSources.find(shader);
    if (it == shaderSources.end()) {
        *params = 0;
        return;
    }

    if (pname == GL_COMPILE_STATUS) {
        *params = it->second.compiled ? GL_TRUE : GL_FALSE;
    } else if (pname == GL_INFO_LOG_LENGTH) {
        *params = it->second.compiled ? 0 : 256;
    } else {
        *params = 0;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_LINK_STATUS:
            params[0] = [programLinkStatus[@(program)] boolValue] ? GL_TRUE : GL_FALSE;
            break;
        case GL_CURRENT_PROGRAM:
            params[0] = (GLint)currentProgram;
            break;
        case GL_ACTIVE_UNIFORMS:
            params[0] = 4;
            break;
        case GL_ACTIVE_ATTRIBUTES:
            params[0] = 8;
            break;
        default:
            params[0] = 0;
            break;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (length) *length = 0;
    if (bufSize <= 0 || !infoLog) {
        if (bufSize < 0) gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSString *message = programInfoLog[@(shader)];
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
    if (!ctx || !data) {
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
            data[0] = 8;
            break;
        case GL_MAX_TEXTURE_IMAGE_UNITS:
            data[0] = 32;
            break;
        case GL_MAX_COLOR_ATTACHMENTS:
            data[0] = 8;
            break;
        case GL_FRAMEBUFFER_BINDING:
            data[0] = (GLint)ctx.currentFramebuffer;
            break;
        case GL_MAJOR_VERSION:
            data[0] = 3;
            break;
        case GL_MINOR_VERSION:
            data[0] = 3;
            break;
        default:
            data[0] = 0;
            break;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetFloatv(GLenum pname, GLfloat *data) {
    if (!ctx || !data) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_COLOR_CLEAR_VALUE:
            data[0] = ctx.clearColor.red; data[1] = ctx.clearColor.green;
            data[2] = ctx.clearColor.blue; data[3] = ctx.clearColor.alpha;
            break;
        default:
            data[0] = 0.0f; data[1] = 0.0f; data[2] = 0.0f; data[3] = 0.0f;
            break;
    }

    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetBooleanv(GLenum pname, GLboolean *data) {
    if (!ctx || !data) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    switch (pname) {
        case GL_DEPTH_TEST:
            data[0] = ctx.depthTestEnabled;
            break;
        case GL_SCISSOR_TEST:
            data[0] = ctx.scissorTestEnabled;
            break;
        default:
            data[0] = GL_FALSE;
            break;
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
    printf("[gl4metal][BOUNDS] attr[%lu] check: VBO=%u base=%zu stride=%zu elem=%zu lastIndex=%zu end=%zu buffer=%zu divisor=%u\n",
           (unsigned long)attribIndex, a->boundVBO, base, stride, elemSize, logicalIndex, end, length, a->divisor);
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

static void gl4metalDrawArraysInternal(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    if (!ctx || count <= 0 || instancecount <= 0) return;
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

    id<MTLRenderPipelineState> activePipeline = ctx.pipelineState ?: gl4metalResolveCurrentPipeline();
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

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
    [encoder drawPrimitives:primitiveType vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count instanceCount:(NSUInteger)instancecount];

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

static void gl4metalDrawElementsInternal(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    if (!ctx || count <= 0 || instancecount <= 0) return;
    if (gl4metalGPUIsolationEnabled && gl4metalGPUIsolationFailed) {
        printf("[gl4metal][GPU-DEBUG] draw skipped after previous GPU failure\n");
        return;
    }

    id<MTLRenderPipelineState> activePipeline = ctx.pipelineState ?: gl4metalResolveCurrentPipeline();
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

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
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

    [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset instanceCount:(NSUInteger)instancecount];

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
    pipelineDesc.colorAttachments[0].blendingEnabled = ctx.blendEnabled;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = gl4metalBlendFactor(ctx.blendSrcRGB);
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = gl4metalBlendFactor(ctx.blendDstRGB);
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = gl4metalBlendFactor(ctx.blendSrcAlpha);
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = gl4metalBlendFactor(ctx.blendDstAlpha);
    pipelineDesc.colorAttachments[0].rgbBlendOperation = gl4metalBlendOperation(ctx.blendEquationRGB);
    pipelineDesc.colorAttachments[0].alphaBlendOperation = gl4metalBlendOperation(ctx.blendEquationAlpha);
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = gl4metalBlendFactor(ctx.blendSrcRGB);
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = gl4metalBlendFactor(ctx.blendDstRGB);
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
    if (program != 0) {
        printf("[gl4metal][RESOURCES] program=%u reflected Metal resources:\n", program);
        printf("%s", resourceInfo.UTF8String);
    }
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
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
    if (!ctx) return;
    ctx.blendSrcRGB = srcRGB;
    ctx.blendDstRGB = dstRGB;
    ctx.blendSrcAlpha = srcAlpha;
    ctx.blendDstAlpha = dstAlpha;
    ctx.pipelineState = nil;
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
    if (!ctx) return;

    if (cap == GL_BLEND) {
        if (!ctx.blendEnabled) { ctx.blendEnabled = GL_TRUE; ctx.pipelineState = nil; }
    } else if (cap == GL_DEPTH_TEST) {
        if (!ctx.depthTestEnabled) {
            ctx.depthTestEnabled = GL_TRUE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (!ctx.scissorTestEnabled) {
            ctx.scissorTestEnabled = GL_TRUE;
        }
    } else if (cap == GL_CULL_FACE) {
        ctx.cullEnabled = GL_TRUE;
    }
}

void APIENTRY glDisable(GLenum cap) {
    if (cap == GL_BLEND) {
        if (ctx.blendEnabled) { ctx.blendEnabled = GL_FALSE; ctx.pipelineState = nil; }
    } else if (cap == GL_DEPTH_TEST) {
        if (ctx.depthTestEnabled) {
            ctx.depthTestEnabled = GL_FALSE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (ctx.scissorTestEnabled) {
            ctx.scissorTestEnabled = GL_FALSE;
        }
    } else if (cap == GL_CULL_FACE) {
        ctx.cullEnabled = GL_FALSE;
    }
}

void APIENTRY glDepthFunc(GLenum func) {
    if (ctx.depthFunc != func) {
        ctx.depthFunc = func;
        updateDepthStencilState();
    }
}

void APIENTRY glDepthMask(GLboolean flag) {
    if (ctx.depthWriteEnabled != flag) {
        ctx.depthWriteEnabled = flag;
        updateDepthStencilState();
    }
}

