//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import "gl4metal.h"

static gl4metalContext *ctx = nil;

static GLuint nextBufferId = 1;
static GLuint nextVAOId = 1;

static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *bufferObjects = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *bufferSizes = nil;
static NSMutableDictionary<NSNumber *, gl4metalVertexArray *> *vaoObjects = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *framebufferObjects = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *renderbufferObjects = nil;

GLenum lastGL4MetalError = GL_NO_ERROR;

static GLuint nextShaderId = 1;
static GLuint nextProgramId = 1;
static GLuint currentProgram = 0;

static NSMutableDictionary<NSNumber *, NSNumber *> *shaderTypes = nil;
static NSMutableDictionary<NSNumber *, NSString *> *shaderSources = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *shaderCompileStatus = nil;
static NSMutableDictionary<NSNumber *, NSMutableArray<NSNumber *> *> *programShaders = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *programLinkStatus = nil;
static NSMutableDictionary<NSNumber *, NSString *> *programInfoLog = nil;

static NSMutableDictionary<NSNumber *, NSValue *> *uniform1fValues = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *uniform2fValues = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *uniform3fValues = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *uniform4fValues = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *uniform1iValues = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *uniform2iValues = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *uniform3iValues = nil;
static NSMutableDictionary<NSNumber *, NSNumber *> *uniform4iValues = nil;
static NSMutableDictionary<NSNumber *, NSValue *> *uniformMatrix4fvValues = nil;

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

static void gl4metalSetError(GLenum error) {
    lastGL4MetalError = error;
}

@implementation gl4metalVertexArray
@end

@implementation gl4metalContext
@end

#pragma mark - OpenGL implementation

GLboolean APIENTRY gl4metalInit(void) {
    if (ctx != nil) return GL_TRUE;

    ctx = [[gl4metalContext alloc] init];
    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) return GL_FALSE;

    ctx.commandQueue = [ctx.device newCommandQueue];
    ctx.boundBuffers = [[NSMutableDictionary alloc] init];
    ctx.currentVAO = 0;
    ctx.currentFramebuffer = 0;
    ctx.currentReadFramebuffer = 0;
    // ctx.swapInterval = 1;

    bufferObjects = [[NSMutableDictionary alloc] init];
    bufferSizes = [[NSMutableDictionary alloc] init];
    vaoObjects = [[NSMutableDictionary alloc] init];
    framebufferObjects = [[NSMutableDictionary alloc] init];
    renderbufferObjects = [[NSMutableDictionary alloc] init];
    shaderTypes = [[NSMutableDictionary alloc] init];
    shaderSources = [[NSMutableDictionary alloc] init];
    shaderCompileStatus = [[NSMutableDictionary alloc] init];
    programShaders = [[NSMutableDictionary alloc] init];
    programLinkStatus = [[NSMutableDictionary alloc] init];
    programInfoLog = [[NSMutableDictionary alloc] init];
    uniform1fValues = [[NSMutableDictionary alloc] init];
    uniform2fValues = [[NSMutableDictionary alloc] init];
    uniform3fValues = [[NSMutableDictionary alloc] init];
    uniform4fValues = [[NSMutableDictionary alloc] init];
    uniform1iValues = [[NSMutableDictionary alloc] init];
    uniform2iValues = [[NSMutableDictionary alloc] init];
    uniform3iValues = [[NSMutableDictionary alloc] init];
    uniform4iValues = [[NSMutableDictionary alloc] init];
    uniformMatrix4fvValues = [[NSMutableDictionary alloc] init];
    currentProgram = 0;
    lastGL4MetalError = GL_NO_ERROR;

    NSLog(@"[gl4metal] Context created on device: %@", ctx.device.name);

    ctx.depthTestEnabled = GL_FALSE;
    ctx.depthWriteEnabled = GL_TRUE;
    ctx.depthFunc = GL_LESS;
    updateDepthStencilState();

    ctx.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    ctx.clearDepth = 1.0;
    ctx.pendingClearFlags = 0;

    // Initialize viewport and scissor rect to default values
    ctx.viewport = (gl4metalRect){0, 0, 800, 600};
    ctx.scissorRect = (gl4metalRect){0, 0, 800, 600};
    ctx.scissorTestEnabled = GL_FALSE;

    return GL_TRUE;
}

void APIENTRY glMakeCurrent(void* metalLayerPtr) {
    if (ctx == nil) {
	    return;
    }

    CAMetalLayer *layer = (__bridge CAMetalLayer *)metalLayerPtr;
    layer.device = ctx.device;
    layer.framebufferOnly = YES;

    ctx.metalLayer = layer;
    NSLog(@"[gl4metal] Context made current with CAMetalLayer");
}

void APIENTRY glSwapBuffers(void) {
    if (!ctx) return;

    if (ctx.pendingClearFlags != 0 && ctx.metalLayer) {
        if (!ctx.currentCommandBuffer) ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
        if (!ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
        if (ctx.currentDrawable) {
            MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();
            if (renderPassDesc) {
                id<MTLRenderCommandEncoder> encoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
                [encoder endEncoding];
            }
        }
    }

    if (!ctx.currentCommandBuffer || !ctx.currentDrawable) {
        NSLog(@"[gl4metal] ERROR: No command buffer or drawable available for swap");
        return;
    }

    if (ctx.swapInterval == 0) {
        [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable atTime:0];
    } else {
        [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
    }

    [ctx.currentCommandBuffer commit];

    ctx.currentCommandBuffer = nil;
    ctx.currentDrawable = nil;
}

void APIENTRY glSwapInterval(GLint interval) {
    if (!ctx) return;

    ctx.swapInterval = interval;
    if (ctx.metalLayer) {
        ctx.metalLayer.maximumDrawableCount = (interval == 0) ? 2 : 3;
    }

    NSLog(@"[gl4metal] glSwapInterval set to: %d", interval);
}

void APIENTRY glClear(GLbitfield mask) {
    if (!ctx) return;

    ctx.pendingClearFlags |= mask;
}

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
	if (!ctx) return;
    ctx.clearColor = MTLClearColorMake(red, green, blue, alpha);
}

void APIENTRY glClearDepth(GLclampd depth) {
    if (!ctx) return;
    ctx.clearDepth = depth;
}

static MTLRenderPassDescriptor* createRenderPassDescriptor() {
    if (!ctx || !ctx.currentDrawable) return nil;

    CGSize drawableSize = CGSizeMake(ctx.currentDrawable.texture.width, ctx.currentDrawable.texture.height);
    ensureDepthTexture(drawableSize);

    MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];

    passDesc.colorAttachments[0].texture = ctx.currentDrawable.texture;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    if (ctx.pendingClearFlags & GL_COLOR_BUFFER_BIT) {
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].clearColor = ctx.clearColor;
        ctx.pendingClearFlags &= ~GL_COLOR_BUFFER_BIT;
    } else {
        passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }

    passDesc.depthAttachment.texture = ctx.depthTexture;
    passDesc.depthAttachment.storeAction = MTLStoreActionDontCare;

    if (ctx.pendingClearFlags & GL_DEPTH_BUFFER_BIT) {
        passDesc.depthAttachment.loadAction = MTLLoadActionClear;
        passDesc.depthAttachment.clearDepth = ctx.clearDepth;
        ctx.pendingClearFlags &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        passDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    }
    
    return passDesc;
}

void APIENTRY glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!ctx) return;
    ctx.viewport = (gl4metalRect){x, y, width, height};
}

void APIENTRY glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!ctx) return;
    ctx.scissorRect = (gl4metalRect){x, y, width, height};
}

static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight) {
    if (!encoder || targetWidth == 0 || targetHeight == 0) return;

    gl4metalRect viewport = ctx.viewport;
    double metalViewportY = (double)targetHeight - ((double)viewport.y + (double)viewport.height);

    MTLViewport mtlViewport = {
        .originX = (double)viewport.x,
        .originY = metalViewportY,
        .width = (double)viewport.width,
        .height = (double)viewport.height,
        .znear = 0.0,
        .zfar = 1.0
    };
    [encoder setViewport:mtlViewport];

    MTLScissorRect mtlScissor;

    if (ctx.scissorTestEnabled) {
        gl4metalRect scissor = ctx.scissorRect;
        NSInteger flippedY = (NSInteger)targetHeight - ((NSInteger)scissor.y + (NSInteger)scissor.height);
        
        NSInteger clampedX = MAX(0, MIN((NSInteger)targetWidth, (NSInteger)scissor.x));
        NSInteger clampedY = MAX(0, MIN((NSInteger)targetHeight, flippedY));
        NSUInteger clampedWidth = MAX(0, MIN((NSUInteger)targetWidth - clampedX, (NSUInteger)scissor.width));
        NSUInteger clampedHeight = MAX(0, MIN((NSUInteger)targetHeight - clampedY, (NSUInteger)scissor.height));

        mtlScissor = (MTLScissorRect){
            .x = (NSUInteger)clampedX,
            .y = (NSUInteger)clampedY,
            .width = (NSUInteger)clampedWidth,
            .height = (NSUInteger)clampedHeight
        };
    } else {
        mtlScissor = (MTLScissorRect){
            .x = 0,
            .y = 0,
            .width = targetWidth,
            .height = targetHeight
        };
    }

    [encoder setScissorRect:mtlScissor];
}

void APIENTRY glGenBuffers(GLsizei n, GLuint *buffers) {
    if (!buffers) return;
    for (GLsizei i = 0; i < n; i++) {
        buffers[i] = nextBufferId++;
    }
}

static gl4metalVertexArray *gl4metalGetCurrentVAO(void) {
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

void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);

    if (currentBound == 0) {
        NSLog(@"[gl4metal] ERROR: glBufferData called with no bound buffer for target 0x%X", target);
        return;
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
    }
}

void APIENTRY glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    GLuint currentBound = gl4metalGetBoundBufferForTarget(target);
    id<MTLBuffer> buffer = bufferObjects[@(currentBound)];
    if (!buffer || !data) return;

    size_t bufferLength = [buffer length];
    if ((size_t)offset + (size_t)size > bufferLength) return;

    memcpy((char *)[buffer contents] + offset, data, size);
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
void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays) {
    if (!arrays) return;

    for (GLsizei i = 0; i < n; i++) {
        GLuint vaoId = nextVAOId++;
        arrays[i] = vaoId;
        gl4metalVertexArray *vao = [[gl4metalVertexArray alloc] init];
        vao.id = vaoId;
        vaoObjects[@(vaoId)] = vao;
    }
}

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

    GLintptr pointerOffset = (pointer != NULL) ? (GLintptr)(uintptr_t)pointer : 0;

    vao->attribs[index].enabled = GL_TRUE;
    vao->attribs[index].size = size;
    vao->attribs[index].type = type;
    vao->attribs[index].normalized = normalized;
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointerOffset = pointerOffset;
    vao->attribs[index].boundVBO = currentVBO;
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

    programLinkStatus[@(program)] = @(GL_TRUE);
    programInfoLog[@(program)] = @"";
    gl4metalSetError(GL_NO_ERROR);
}

static BOOL gl4metalCreateDefaultProgramPipelineForProgram(GLuint program) {
    if (!ctx || !ctx.device) return NO;

    NSString *source = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct VertexIn {\n"
        "    float4 position [[attribute(0)]];\n"
        "    float4 color [[attribute(1)]];\n"
        "};\n"
        "struct VertexOut {\n"
        "    float4 position [[position]];\n"
        "    float4 color;\n"
        "};\n"
        "vertex VertexOut gl4metal_default_vertex(VertexIn in [[stage_in]]) {\n"
        "    VertexOut out;\n"
        "    out.position = in.position;\n"
        "    out.color = in.color;\n"
        "    return out;\n"
        "}\n"
        "fragment half4 gl4metal_default_fragment(VertexOut in [[stage_in]]) {\n"
        "    return half4(in.color);\n"
        "}\n";

    NSError *error = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        NSLog(@"[gl4metal] ERROR: Failed to compile default Metal library for program %u: %@", program, error.localizedDescription);
        return NO;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"gl4metal_default_vertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"gl4metal_default_fragment"];
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"[gl4metal] ERROR: Default Metal functions missing for program %u", program);
        return NO;
    }

    MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat4;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
    vertexDescriptor.attributes[1].offset = 0;
    vertexDescriptor.attributes[1].bufferIndex = 1;
    vertexDescriptor.layouts[0].stride = sizeof(float) * 4;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[1].stride = sizeof(float) * 4;
    vertexDescriptor.layouts[1].stepRate = 1;
    vertexDescriptor.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;

    return gl4metalCreatePipelineState(vertexFunction, fragmentFunction, vertexDescriptor);
}

void APIENTRY glUseProgram(GLuint program) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    if (program == 0 || [programLinkStatus[@(program)] boolValue] == GL_FALSE) {
        gl4metalSetError(GL_INVALID_OPERATION);
        NSLog(@"[gl4metal] glUseProgram called with unlinked or invalid program: %u", program);
        return;
    }

    currentProgram = program;
    if (!ctx.pipelineState) {
        if (!gl4metalCreateDefaultProgramPipelineForProgram(program)) {
            NSLog(@"[gl4metal] WARNING: Program %u linked but default Metal pipeline creation failed", program);
        }
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLint APIENTRY glGetUniformLocation(GLuint program, const GLchar *name) {
    if (!ctx || program == 0 || name == nil) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    gl4metalSetError(GL_NO_ERROR);
    return gl4metalResolveLocationFromName(name);
}

void APIENTRY glUniform1f(GLint location, GLfloat v0) {
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    uniform1fValues[@(location)] = [NSValue valueWithPointer:&v0];
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

void APIENTRY glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    if (!framebuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = ++nextBufferId;
        framebuffers[i] = framebuffer;
        framebufferObjects[@(framebuffer)] = @(framebuffer);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer) {
    (void)target;
    if (framebuffer == 0) {
        ctx.currentFramebuffer = 0;
        ctx.currentReadFramebuffer = 0;
    } else {
        ctx.currentFramebuffer = framebuffer;
        ctx.currentReadFramebuffer = framebuffer;
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLenum APIENTRY glCheckFramebufferStatus(GLenum target) {
    (void)target;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }

    gl4metalSetError(GL_NO_ERROR);
    return GL_FRAMEBUFFER_COMPLETE;
}

void APIENTRY glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    (void)target; (void)attachment; (void)textarget; (void)texture; (void)level;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    if (!framebuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        [framebufferObjects removeObjectForKey:@(framebuffers[i])];
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) {
    if (!renderbuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        GLuint renderbuffer = ++nextBufferId;
        renderbuffers[i] = renderbuffer;
        renderbufferObjects[@(renderbuffer)] = @(renderbuffer);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    (void)target; (void)renderbuffer;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
    (void)target; (void)internalformat; (void)width; (void)height;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    if (!renderbuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        [renderbufferObjects removeObjectForKey:@(renderbuffers[i])];
    }
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
    if (!infoLog) {
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
            data[0] = 4096;
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

GLint APIENTRY glGetAttribLocation(GLuint program, const GLchar *name) {
    (void)program;
    if (!ctx || name == NULL) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return -1;
    }

    NSString *attributeName = [NSString stringWithUTF8String:name];
    NSArray *knownNames = @[@"aPosition", @"aPos", @"position", @"vertex",
                            @"aNormal", @"normal", @"aTexCoord", @"aUV",
                            @"aColor", @"color"];
    NSInteger index = [knownNames indexOfObject:attributeName];
    gl4metalSetError(GL_NO_ERROR);
    return (index == NSNotFound) ? -1 : (GLint)index;
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
    if (!source) {
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

    if (!ctx.pipelineState && currentProgram != 0) {
        gl4metalCreateDefaultProgramPipelineForProgram(currentProgram);
    }

    if (!ctx.pipelineState) {
        NSLog(@"[gl4metal] WARNING: No valid Metal pipeline for glDrawArrays");
        return;
    }

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

void APIENTRY glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
    if (!ctx || count <= 0) return;

    if (!ctx.pipelineState && currentProgram != 0) {
        gl4metalCreateDefaultProgramPipelineForProgram(currentProgram);
    }

    if (!ctx.pipelineState) {
        NSLog(@"[gl4metal] WARNING: No valid Metal pipeline for glDrawElements");
        return;
    }

    gl4metalVertexArray *currentVAO = gl4metalGetCurrentVAO();
    GLuint elementVBO = currentVAO ? currentVAO->elementArrayBufferID : [ctx.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] unsignedIntValue];

    if (elementVBO == 0) {
        NSLog(@"[gl4metal] ERROR: glDrawElements called without bound GL_ELEMENT_ARRAY_BUFFER");
        return;
    }

    if (type == GL_UNSIGNED_BYTE) {
        NSLog(@"[gl4metal] WARNING: Metal does not support GL_UNSIGNED_BYTE index buffers directly; skipping draw for safety");
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
                }
            }
        }
    }

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
    MTLIndexType indexType = getMetalIndexType(type);
    GLintptr indexOffsetValue = (indices != NULL) ? (GLintptr)(uintptr_t)indices : 0;
    NSUInteger indexOffset = 0;
    if (!gl4metalValidateBufferOffset(indexBuffer, indexOffsetValue, &indexOffset)) {
        NSLog(@"[gl4metal] WARNING: Index buffer offset invalid for glDrawElements, defaulting to 0");
        indexOffset = 0;
    }

    id<MTLBuffer> drawIndexBuffer = indexBuffer;
    NSUInteger drawIndexOffset = indexOffset;
    MTLIndexType drawIndexType = indexType;

    if (type == GL_UNSIGNED_BYTE) {
        const GLubyte *src = (const GLubyte *)((const char *)[indexBuffer contents] + indexOffset);
        if (indices != NULL && elementVBO == 0) {
            src = (const GLubyte *)indices;
        }

        size_t tmpByteLength = (size_t)count * sizeof(uint16_t);
        id<MTLBuffer> tmpBuffer = [ctx.device newBufferWithLength:tmpByteLength options:MTLResourceStorageModeShared];
        if (tmpBuffer) {
            uint16_t *dst = (uint16_t *)[tmpBuffer contents];
            for (GLsizei i = 0; i < count; i++) {
                dst[i] = (uint16_t)src[i];
            }
            drawIndexBuffer = tmpBuffer;
            drawIndexOffset = 0;
            drawIndexType = MTLIndexTypeUInt16;
        }
    }

    [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset];

    [encoder endEncoding];
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

static void updateDepthStencilState() {
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

