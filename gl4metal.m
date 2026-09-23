//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import "gl4metal.h"
#import "gl4metal_runtime.h"
#import "gl4metal_shaders.h"

gl4metalVertexArray *gl4metalGetCurrentVAO(void);
static void invalidatePipelineState(void);

static GLintptr gl4metalResolvePointerOffset(const void *pointer, id<MTLBuffer> buffer) {
    if (pointer == NULL || !buffer) return 0;

    uintptr_t raw = (uintptr_t)pointer;
    size_t bufferLength = (size_t)[buffer length];
    if (raw > bufferLength) {
        NSLog(@"[gl4metal] WARNING: pointer offset 0x%zx exceeds bound buffer length 0x%zx; clamping to 0", (size_t)raw, bufferLength);
        return 0;
    }
    return (GLintptr)raw;
}

static GLint gl4metalResolveLocationFromName(const char *name) {
    if (name == NULL) return -1;
    NSString *symbolName = [NSString stringWithUTF8String:name];
    NSArray *knownNames = @[
        @"uProjection", @"uView", @"uModel", @"uMVP",
        @"uModelViewProjectionMatrix", @"uViewProj", @"uTime",
        @"uResolution", @"uTex", @"uTexture0", @"uTexture",
        @"aPosition", @"aPos", @"position", @"vertex",
        @"aNormal", @"normal", @"aTexCoord", @"aUV",
        @"aColor", @"color"
    ];

    NSInteger index = [knownNames indexOfObject:symbolName];
    if (index != NSNotFound) {
        return (GLint)(index + 1);
    }

    NSUInteger hash = [symbolName hash];
    return (GLint)(hash & 0x7fffffff);
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
    static NSString *device = nil;
    device = [NSString stringWithFormat:@"gl4metal (%@)", ctx.device ? ctx.device.name : @"Unknown"];
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
    ctx.viewport = (gl4metalRect){0, 0, 800, 600};
    ctx.scissorRect = (gl4metalRect){0, 0, 800, 600};
    ctx.depthTestEnabled = GL_FALSE;
    ctx.depthWriteEnabled = GL_TRUE;
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
    ctx.readBuffer = GL_BACK;
    ctx.blendEnabled = GL_FALSE;
    ctx.blendSourceRGB = GL_ONE;
    ctx.blendDestinationRGB = GL_ZERO;
    ctx.blendSourceAlpha = GL_ONE;
    ctx.blendDestinationAlpha = GL_ZERO;
    ctx.blendEquationRGB = GL_FUNC_ADD;
    ctx.blendEquationAlpha = GL_FUNC_ADD;
    ctx.depthRangeNear = 0.0;
    ctx.depthRangeFar = 1.0;
    ctx.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    ctx.clearDepth = 1.0;
    MTLSamplerDescriptor *samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeRepeat;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeRepeat;
    ctx.defaultSamplerState = [ctx.device newSamplerStateWithDescriptor:samplerDescriptor];

    bufferObjects = [NSMutableDictionary dictionary];
    bufferSizes = [NSMutableDictionary dictionary];
    convertedIndexBuffers = [NSMutableDictionary dictionary];
    vaoObjects = [NSMutableDictionary dictionary];
    framebufferObjects = [NSMutableDictionary dictionary];
    renderbufferObjects = [NSMutableDictionary dictionary];
    shaderTypes = [NSMutableDictionary dictionary];
    shaderSources = [NSMutableDictionary dictionary];
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
    programAttribBindings = [NSMutableDictionary dictionary];
    pipelineLayoutCache = [NSMutableDictionary dictionary];
    textureObjects = [NSMutableDictionary dictionary];
    boundTextureUnits = [NSMutableDictionary dictionary];
    uniformLayouts = [NSMutableDictionary dictionary];
    programMetalLibraries = [NSMutableDictionary dictionary];
    currentProgram = 0;
    lastGL4MetalError = GL_NO_ERROR;
    updateDepthStencilState();
    return GL_TRUE;
}

void APIENTRY glMakeCurrent(void *metalLayerPtr) {
    if (!ctx || !metalLayerPtr) return;
    CAMetalLayer *layer = (__bridge CAMetalLayer *)metalLayerPtr;
    layer.device = ctx.device;
    layer.framebufferOnly = YES;
    ctx.metalLayer = layer;
}

void APIENTRY glSwapBuffers(void) {
    if (!ctx || !ctx.currentCommandBuffer || !ctx.currentDrawable) return;
    [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
    [ctx.currentCommandBuffer commit];
    ctx.currentCommandBuffer = nil;
    ctx.currentDrawable = nil;
}

void APIENTRY glFlush(void) {
    if (!ctx || !ctx.currentCommandBuffer) return;
    [ctx.currentCommandBuffer commit];
    ctx.currentCommandBuffer = nil;
    ctx.currentDrawable = nil;
}

void APIENTRY glFinish(void) {
    if (!ctx || !ctx.currentCommandBuffer) return;
    id<MTLCommandBuffer> commandBuffer = ctx.currentCommandBuffer;
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    ctx.currentCommandBuffer = nil;
    ctx.currentDrawable = nil;
}

void APIENTRY glSwapInterval(GLint interval) {
    if (!ctx) return;
    ctx.swapInterval = interval;
    if (ctx.metalLayer) ctx.metalLayer.maximumDrawableCount = interval == 0 ? 2 : 3;
}

void APIENTRY glClear(GLbitfield mask) {
    if (ctx) ctx.pendingClearFlags |= mask;
}

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
    if (ctx) ctx.clearColor = MTLClearColorMake(red, green, blue, alpha);
}

void APIENTRY glClearDepth(GLclampd depth) {
    if (ctx) ctx.clearDepth = depth;
}

static MTLRenderPassDescriptor *createRenderPassDescriptor(void) {
    if (!ctx || !ctx.currentDrawable) return nil;
    CGSize size = CGSizeMake(ctx.currentDrawable.texture.width, ctx.currentDrawable.texture.height);
    ensureDepthTexture(size);

    MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = ctx.currentDrawable.texture;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    passDesc.colorAttachments[0].loadAction = (ctx.pendingClearFlags & GL_COLOR_BUFFER_BIT)
        ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.colorAttachments[0].clearColor = ctx.clearColor;
    passDesc.depthAttachment.texture = ctx.depthTexture;
    passDesc.depthAttachment.storeAction = MTLStoreActionDontCare;
    passDesc.depthAttachment.loadAction = (ctx.pendingClearFlags & GL_DEPTH_BUFFER_BIT)
        ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.depthAttachment.clearDepth = ctx.clearDepth;
    ctx.pendingClearFlags = 0;
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
        ctx.depthRangeNear,
        ctx.depthRangeFar
    }];
    gl4metalRect scissor = ctx.scissorTestEnabled ? ctx.scissorRect : (gl4metalRect){0, 0, (GLsizei)targetWidth, (GLsizei)targetHeight};
    [encoder setScissorRect:(MTLScissorRect){
        MAX(0, scissor.x),
        MAX(0, (GLint)targetHeight - scissor.y - scissor.height),
        MIN((NSUInteger)MAX(0, scissor.width), targetWidth),
        MIN((NSUInteger)MAX(0, scissor.height), targetHeight)
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

void APIENTRY glReadBuffer(GLenum src) {
    if (src != GL_BACK && src != GL_NONE) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    ctx.readBuffer = src;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glHint(GLenum target, GLenum mode) {
    (void)target;
    (void)mode;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDepthRange(GLdouble nearValue, GLdouble farValue) {
    if (nearValue < 0.0 || nearValue > 1.0 || farValue < 0.0 || farValue > 1.0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    ctx.depthRangeNear = nearValue;
    ctx.depthRangeFar = farValue;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDepthRangef(GLfloat nearValue, GLfloat farValue) {
    glDepthRange((GLdouble)nearValue, (GLdouble)farValue);
}

void APIENTRY glBlendFunc(GLenum source, GLenum destination) {
    glBlendFuncSeparate(source, destination, source, destination);
}

void APIENTRY glBlendFuncSeparate(GLenum sourceRGB, GLenum destinationRGB, GLenum sourceAlpha, GLenum destinationAlpha) {
    ctx.blendSourceRGB = sourceRGB;
    ctx.blendDestinationRGB = destinationRGB;
    ctx.blendSourceAlpha = sourceAlpha;
    ctx.blendDestinationAlpha = destinationAlpha;
    invalidatePipelineState();
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBlendEquation(GLenum mode) {
    glBlendEquationSeparate(mode, mode);
}

void APIENTRY glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
    ctx.blendEquationRGB = modeRGB;
    ctx.blendEquationAlpha = modeAlpha;
    invalidatePipelineState();
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
}

static void invalidatePipelineState(void) {
    if (!ctx) return;
    ctx.pipelineState = nil;
    [ctx.programPipelines removeAllObjects];
    [pipelineLayoutCache removeAllObjects];
}

static MTLBlendFactor getMetalBlendFactor(GLenum factor) {
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

static MTLBlendOperation getMetalBlendOperation(GLenum operation) {
    switch (operation) {
        case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
        case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case GL_MIN: return MTLBlendOperationMin;
        case GL_MAX: return MTLBlendOperationMax;
        case GL_FUNC_ADD:
        default: return MTLBlendOperationAdd;
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
        NSLog(@"[gl4metal] ERROR: glBufferData called with no bound buffer for target 0x%X", target);
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
    if (!buffer || !data) return;

    size_t bufferLength = [buffer length];
    if ((size_t)offset + (size_t)size > bufferLength) return;

    memcpy((char *)[buffer contents] + offset, data, size);
    [convertedIndexBuffers removeAllObjects];
    bufferSizes[@(currentBound)] = @(MAX((NSInteger)[bufferSizes[@(currentBound)] integerValue], (NSInteger)(offset + size)));
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
        NSLog(@"[gl4metal] ERROR: glMapbuffer: No MTLBuffer found for ID %u at target 0x%X", currentBound, target);
        return NULL;
    }

    return [mtlBuffer contents];
}

void *APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    (void)access;
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> mtlBuffer = bufferObjects[@(currentBound)];
    if (!mtlBuffer) return NULL;

    size_t bufferLength = [mtlBuffer length];
    if ((size_t)offset + (size_t)length > bufferLength) return NULL;
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
        vaoObjects[@(array)] = vao;
    }

    ctx.boundBuffers[@(GL_ARRAY_BUFFER)] = @(vao->arrayBufferID);
    ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] = @(vao->elementArrayBufferID);
}

void APIENTRY glEnableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_TRUE;
    }
}

void APIENTRY glDisableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = vaoObjects[@(ctx.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_FALSE;
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
        NSLog(@"[gl4metal] WARNING: buffer offset 0x%zx exceeds buffer length 0x%zx; clamping to 0", (size_t)offset, bufferLength);
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
        NSLog(@"[gl4metal] ERROR: glVertexAttribPointer called without bound VAO!");
        return;
    }

    GLuint currentVBO = gl4metalGetBoundBufferForTarget(GL_ARRAY_BUFFER);
    if (currentVBO == 0) {
        NSLog(@"[gl4metal] ERROR: glVertexAttribPointer called without bound GL_ARRAY_BUFFER");
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
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointerOffset = pointerOffset;
    vao->attribs[index].boundVBO = currentVBO;
}

void APIENTRY glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
    glVertexAttribPointer(index, size, type, GL_FALSE, stride, pointer);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttribDivisor(GLuint index, GLuint divisor) {
    (void)index; (void)divisor;
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
    shaderTypes[@(shaderId)] = @(type);
    shaderSources[@(shaderId)] = @"";
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

    NSMutableString *source = [[NSMutableString alloc] init];
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;
        if (length && length[i] > 0) {
            NSString *part = [[NSString alloc] initWithBytes:string[i] length:(NSUInteger)length[i] encoding:NSUTF8StringEncoding];
            if (part) {
                [source appendString:part];
            }
        } else {
            [source appendString:@(string[i])];
        }
    }

    shaderSources[@(shader)] = source;
    shaderCompileStatus[@(shader)] = @(GL_FALSE);
    programInfoLog[@(shader)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glCompileShader(GLuint shader) {
    if (!ctx || shader == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    NSString *source = shaderSources[@(shader)];
    if (source == nil || [source length] == 0) {
        gl4metalSetError(GL_INVALID_OPERATION);
        programInfoLog[@(shader)] = @"Shader source is empty";
        return;
    }

    shaderCompileStatus[@(shader)] = @(GL_TRUE);
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

    [programMetalLibraries removeObjectForKey:@(program)];
    [uniformLayouts removeObjectForKey:@(program)];
    [ctx.programPipelines removeObjectForKey:@(program)];
    [pipelineLayoutCache removeAllObjects];
    programLinkStatus[@(program)] = @(GL_TRUE);
    programInfoLog[@(program)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

static MTLVertexFormat gl4metalVertexFormatForAttrib(const gl4metalVertexAttrib *attrib) {
    if (!attrib) return MTLVertexFormatInvalid;

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
        default:
            break;
    }
    return MTLVertexFormatFloat4;
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
        descriptor.layouts[index].stepRate = 1;
        descriptor.layouts[index].stepFunction = MTLVertexStepFunctionPerVertex;
    }

    if (!vao || !vao->attribs[0].enabled) {
        descriptor.attributes[0].format = MTLVertexFormatFloat4;
        descriptor.attributes[0].bufferIndex = 0;
        descriptor.layouts[0].stride = sizeof(float) * 4;
        descriptor.layouts[0].stepRate = 1;
        descriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    if (!vao || !vao->attribs[1].enabled) {
        descriptor.attributes[1].format = MTLVertexFormatFloat4;
        descriptor.attributes[1].bufferIndex = 1;
        descriptor.layouts[1].stride = sizeof(float) * 4;
        descriptor.layouts[1].stepRate = 1;
        descriptor.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    return descriptor;
}


static id<MTLRenderPipelineState> gl4metalResolveCurrentPipeline(void) {
    if (!ctx) return nil;

    if (ctx.pipelineState) {
        return ctx.pipelineState;
    }

    if (currentProgram != 0) {
        NSString *cacheKey = gl4metalBuildPipelineCacheKey(currentProgram, gl4metalGetCurrentVAO());
        id<MTLRenderPipelineState> programPipeline = pipelineLayoutCache[cacheKey];
        if (!programPipeline) {
            if (!gl4metalCreateProgramPipelineForProgram(currentProgram)) {
                return nil;
            }
            programPipeline = ctx.pipelineState;
            if (programPipeline) {
                pipelineLayoutCache[cacheKey] = programPipeline;
            }
        }
        if (programPipeline) {
            ctx.pipelineState = programPipeline;
            return programPipeline;
        }
    }

    return ctx.pipelineState;
}

void APIENTRY glUseProgram(GLuint program) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    if (program == 0) {
        currentProgram = 0;
        ctx.pipelineState = nil;
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    if ([programLinkStatus[@(program)] boolValue] == GL_FALSE) {
        gl4metalSetError(GL_INVALID_OPERATION);
        NSLog(@"[gl4metal] glUseProgram called with unlinked or invalid program: %u", program);
        return;
    }

    if (program == currentProgram && ctx.pipelineState) {
        gl4metalSetError(GL_NO_ERROR);
        return;
    }

    currentProgram = program;
    id<MTLRenderPipelineState> programPipeline = gl4metalResolveCurrentPipeline();
    if (!programPipeline) {
        NSLog(@"[gl4metal] WARNING: Program %u linked but custom Metal pipeline creation failed", program);
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

    GLint location = gl4metalResolveLocationFromName(name);
    uniformLocationNames[@(location)] = [NSString stringWithUTF8String:name];
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
    GLfloat values[4] = {0,0,0,0};
    for (GLsizei i = 0; i < MIN(count, 4); ++i) values[i] = value[i];
    uniform1fValues[@(location)] = [NSValue valueWithBytes:values objCType:@encode(GLfloat[4])];
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
    [programMetalLibraries removeObjectForKey:@(program)];
    [uniformLayouts removeObjectForKey:@(program)];
    [ctx.programPipelines removeObjectForKey:@(program)];
    [pipelineLayoutCache removeAllObjects];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteShader(GLuint shader) {
    if (shader == 0) return;
    [shaderTypes removeObjectForKey:@(shader)];
    [shaderSources removeObjectForKey:@(shader)];
    [shaderCompileStatus removeObjectForKey:@(shader)];
    [programInfoLog removeObjectForKey:@(shader)];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_SHADER_TYPE:
            params[0] = [shaderTypes[@(shader)] intValue];
            break;
        case GL_COMPILE_STATUS:
            params[0] = [shaderCompileStatus[@(shader)] intValue];
            break;
        default:
            params[0] = 0;
            break;
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
            params[0] = [programLinkStatus[@(program)] intValue];
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
    if (!infoLog || bufSize <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSString *message = programInfoLog[@(shader)];
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

void APIENTRY glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (!infoLog || bufSize <= 0) {
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
    (void)index;
    (void)x;
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
    (void)index; (void)x; (void)y;
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
    (void)index; (void)x; (void)y; (void)z;
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
    (void)index; (void)x; (void)y; (void)z; (void)w;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glVertexAttrib4fv(GLuint index, const GLfloat *v) {
    if (v == NULL) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    glVertexAttrib4f(index, v[0], v[1], v[2], v[3]);
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

void APIENTRY glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source) {
    if (!source || bufSize <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    NSString *text = shaderSources[@(shader)];
    if (!text) text = @"";

    const char *utf8 = [text UTF8String];
    GLsizei copyLength = (GLsizei)MIN((size_t)bufSize - 1, strlen(utf8));
    if (copyLength > 0) {
        memcpy(source, utf8, copyLength);
        source[copyLength] = '\0';
    } else {
        source[0] = '\0';
        copyLength = 0;
    }

    if (length) *length = copyLength;
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
            NSLog(@"[gl4metal] WARNING: Unsupported GL mode 0x%X, defaulting to Triangles", mode);
            return MTLPrimitiveTypeTriangle;
    }
}

void APIENTRY glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    if (!ctx || count <= 0) return;

    id<MTLRenderPipelineState> activePipeline = ctx.pipelineState ?: gl4metalResolveCurrentPipeline();
    if (!activePipeline) {
        NSLog(@"[gl4metal] WARNING: No valid Metal pipeline for glDrawArrays");
        return;
    }
    ctx.pipelineState = activePipeline;

    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
    if (!ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
    if (!ctx.currentDrawable) {
        NSLog(@"[gl4metal] ERROR: No drawable available for glDrawArrays");
        return;
    }

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();
    if (!renderPassDesc) {
        NSLog(@"[gl4metal] ERROR: Failed to create render pass descriptor for glDrawArrays");
        return;
    }

    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        NSLog(@"[gl4metal] ERROR: Failed to create MTLRenderCommandEncoder");
        return;
    }

    NSUInteger targetWidth = ctx.currentDrawable.texture.width;
    NSUInteger targetHeight = ctx.currentDrawable.texture.height;
    applyViewportAndScissor(encoder, targetWidth, targetHeight);

    if (ctx.depthStencilState) {
        [encoder setDepthStencilState:ctx.depthStencilState];
    }

    applyRasterState(encoder);
    [encoder setRenderPipelineState:ctx.pipelineState];

    float fallbackMVP[16];
    gl4metalGetFallbackMVP(fallbackMVP);
    NSData *uniformData = gl4metalBuildUniformDataForProgram(currentProgram);
    [encoder setVertexBytes:uniformData.bytes length:uniformData.length atIndex:16];
    [encoder setFragmentBytes:uniformData.bytes length:uniformData.length atIndex:16];
    gl4metalBindFragmentResources(encoder, currentProgram);

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
                }
            }
        }
    }

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
    [encoder drawPrimitives:primitiveType vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count];

    [encoder endEncoding];
}

static MTLIndexType getMetalIndexType(GLenum type) {
    switch(type) {
        case GL_UNSIGNED_BYTE: return MTLIndexTypeUInt16; // Metal does not support 8-bit indices, defaulting to 16-bit
        case GL_UNSIGNED_SHORT: return MTLIndexTypeUInt16;
        case GL_UNSIGNED_INT: return MTLIndexTypeUInt32;
        default:
            NSLog(@"[gl4metal] WARNING: Unsupported GL index type 0x%X, defaulting to UInt16", type);
            return MTLIndexTypeUInt16;
    }
}

void APIENTRY glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    if (instancecount <= 0) return;
    glDrawArrays(mode, first, count);
}

void APIENTRY glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
    if (!ctx || count <= 0) return;

    id<MTLRenderPipelineState> activePipeline = ctx.pipelineState ?: gl4metalResolveCurrentPipeline();
    if (!activePipeline) {
        NSLog(@"[gl4metal] WARNING: No valid Metal pipeline for glDrawElements");
        return;
    }
    ctx.pipelineState = activePipeline;

    gl4metalVertexArray *currentVAO = gl4metalGetCurrentVAO();
    GLuint elementVBO = currentVAO ? currentVAO->elementArrayBufferID : [ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] unsignedIntValue];

    if (elementVBO == 0) {
        NSLog(@"[gl4metal] ERROR: glDrawElements called without bound GL_ELEMENT_ARRAY_BUFFER");
        return;
    }

    id<MTLBuffer> indexBuffer = bufferObjects[@(elementVBO)];
    if (!indexBuffer) {
        NSLog(@"[gl4metal] ERROR: Index MTLBuffer not found for VBO ID %u", elementVBO);
        return;
    }

    if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
    if (!ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];

    if (!ctx.currentDrawable) {
        NSLog(@"[gl4metal] ERROR: No drawable available for rendering");
        return;
    }

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();

    id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        NSLog(@"[gl4metal] ERROR: Failed to create MTLRenderCommandEncoder");
        return;
    }

    applyRasterState(encoder);
    [encoder setRenderPipelineState:ctx.pipelineState];

    float fallbackMVP[16];
    gl4metalGetFallbackMVP(fallbackMVP);
    NSData *uniformData = gl4metalBuildUniformDataForProgram(currentProgram);
    [encoder setVertexBytes:uniformData.bytes length:uniformData.length atIndex:16];
    [encoder setFragmentBytes:uniformData.bytes length:uniformData.length atIndex:16];
    gl4metalBindFragmentResources(encoder, currentProgram);

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
                }
            }
        }
    }

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
    MTLIndexType indexType = getMetalIndexType(type);
    GLintptr indexOffsetValue = gl4metalResolvePointerOffset(indices, indexBuffer);
    NSUInteger indexOffset = 0;
    if (!gl4metalValidateBufferOffset(indexBuffer, indexOffsetValue, &indexOffset)) {
        NSLog(@"[gl4metal] WARNING: Index buffer offset invalid for glDrawElements, defaulting to 0");
        indexOffset = 0;
    }

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

    [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset];

    [encoder endEncoding];
}

void APIENTRY glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    if (instancecount <= 0) return;
    glDrawElements(mode, count, type, indices);
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
            NSLog(@"[gl4metal] WARNING: Unsupported GL depth func 0x%X, defaulting to Less", func);
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
        NSLog(@"[gl4metal] Created new depth texture of size: %.0fx%.0f", size.width, size.height);
    }
}

BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor) {
    if (!ctx || !ctx.device) return NO;

    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunction;
    pipelineDesc.fragmentFunction = fragmentFunction;
    pipelineDesc.vertexDescriptor = vertexDescriptor;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.colorAttachments[0].writeMask =
        (ctx.colorMaskRed ? MTLColorWriteMaskRed : 0) |
        (ctx.colorMaskGreen ? MTLColorWriteMaskGreen : 0) |
        (ctx.colorMaskBlue ? MTLColorWriteMaskBlue : 0) |
        (ctx.colorMaskAlpha ? MTLColorWriteMaskAlpha : 0);
    pipelineDesc.colorAttachments[0].blendingEnabled = ctx.blendEnabled;
    if (ctx.blendEnabled) {
        pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = getMetalBlendFactor(ctx.blendSourceRGB);
        pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = getMetalBlendFactor(ctx.blendDestinationRGB);
        pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = getMetalBlendFactor(ctx.blendSourceAlpha);
        pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = getMetalBlendFactor(ctx.blendDestinationAlpha);
        pipelineDesc.colorAttachments[0].rgbBlendOperation = getMetalBlendOperation(ctx.blendEquationRGB);
        pipelineDesc.colorAttachments[0].alphaBlendOperation = getMetalBlendOperation(ctx.blendEquationAlpha);
    }
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    NSError *error = nil;
    ctx.pipelineState = [ctx.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (error || !ctx.pipelineState) {
        NSLog(@"[gl4metal] ERROR: Failed to create pipeline state: %@", error.localizedDescription);
        return NO;
    }
    return YES;
}

void APIENTRY glEnable(GLenum cap) {
    if (!ctx) return;

    if (cap == GL_DEPTH_TEST) {
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
    } else if (cap == GL_BLEND) {
        if (!ctx.blendEnabled) {
            ctx.blendEnabled = GL_TRUE;
            invalidatePipelineState();
        }
    }
}

void APIENTRY glDisable(GLenum cap) {
    if (cap == GL_DEPTH_TEST) {
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
    } else if (cap == GL_BLEND) {
        if (ctx.blendEnabled) {
            ctx.blendEnabled = GL_FALSE;
            invalidatePipelineState();
        }
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

