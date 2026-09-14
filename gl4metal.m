//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import "gl4metal.h"

static gl4metalContext *g_context = nil;

static GLuint g_nextBufferId = 1;
static GLuint g_nextVAOId = 1;

static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_bufferObjects = nil;
static NSMutableDictionary<NSNumber *, gl4metalVertexArray *> *g_vaoObjects = nil;

@implementation gl4metalVertexArray
@end

@implementation gl4metalContext
@end

#pragma mark - OpenGL 3.3 implementation

GLboolean glInit() {
    if (g_context != nil) return GL_TRUE;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return GL_FALSE;

    g_context = [[gl4metalContext alloc] init];
    g_context.device = device;
    g_context.commandQueue = [g_context.device newCommandQueue];
    g_context.boundBuffers = [[NSMutableDictionary alloc] init];
    g_context.currentVAO = 0;
    // g_context.swapInterval = 1;

    g_bufferObjects = [[NSMutableDictionary alloc] init];
    g_vaoObjects = [[NSMutableDictionary alloc] init];

    NSLog(@"[gl4metal] Context created on device: %@", device.name);

    g_context.depthTestEnabled = GL_FALSE;
    g_context.depthWriteEnabled = GL_TRUE;
    g_context.depthFunc = GL_LESS;
    updateDepthStencilState();

    g_context.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    g_context.clearDepth = 1.0;
    g_context.pendingClearFlags = 0;

    // Initialize viewport and scissor rect to default values
    g_context.viewport = (gl4metalRect){0, 0, 800, 600};
    g_context.scissorRect = (gl4metalRect){0, 0, 800, 600};
    g_context.scissorTestEnabled = GL_FALSE;

    return GL_TRUE;
}

void APIENTRY glMakeCurrent(void* metalLayerPtr) {
    if (g_context == nil) {
	    return;
    }

    CAMetalLayer *layer = (__bridge CAMetalLayer *)metalLayerPtr;
    layer.device = g_context.device;
    layer.framebufferOnly = YES;

    g_context.metalLayer = layer;
    NSLog(@"[gl4metal] Context made current with CAMetalLayer");
}

void APIENTRY glSwapBuffers(void) {
    if (!g_context) return;

    if (g_context.pendingClearFlags != 0 && g_context.metalLayer) {
        if (!g_context.currentCommandBuffer) g_context.currentCommandBuffer = [g_context.commandQueue commandBuffer];
        if (!g_context.currentDrawable) g_context.currentDrawable = [g_context.metalLayer nextDrawable];
        if (g_context.currentDrawable) {
            MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();
            if (renderPassDesc) {
                id<MTLRenderCommandEncoder> encoder = [g_context.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
                [encoder endEncoding];
            }
        }
    }

    if (!g_context.currentCommandBuffer || !g_context.currentDrawable) {
        NSLog(@"[gl4metal] ERROR: No command buffer or drawable available for swap");
        return;
    }

    if (g_context.swapInterval == 0) {
        [g_context.currentCommandBuffer presentDrawable:g_context.currentDrawable atTime:0];
    } else {
        [g_context.currentCommandBuffer presentDrawable:g_context.currentDrawable];
    }

    [g_context.currentCommandBuffer commit];

    g_context.currentCommandBuffer = nil;
    g_context.currentDrawable = nil;
}

void APIENTRY glSwapInterval(GLint interval) {
    if (!g_context) return;

    g_context.swapInterval = interval;
    if (g_context.metalLayer) {
        g_context.metalLayer.maximumDrawableCount = (interval == 0) ? 2 : 3;
    }

    NSLog(@"[gl4metal] glSwapInterval set to: %d", interval);
}

const GLubyte* APIENTRY glGetString(GLenum name) {
    switch(name) {
        case GL_VENDOR:
            return (const GLubyte *)"congcq";
        case GL_RENDERER:
            return (const GLubyte *)"gl4metal 0.1-beta";
        case GL_VERSION:
            return (const GLubyte *)"3.3 Core Profile";
        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)"3.30";
        case GL_EXTENSIONS:
            return (const GLubyte *)"GL_ARB_multitexture GL_ARB_texture_compression";
        default:
            return NULL;
    }
}

const GLubyte *APIENTRY glGetStringi (GLenum name, GLuint index) {
    switch(name) {
        case GL_EXTENSIONS:
            // Return a list of supported extensions (simplified for this example)
            switch (index) {
                case 0:
                    return (const GLubyte *)"GL_ARB_multitexture";
                case 1:
                    return (const GLubyte *)"GL_ARB_texture_compression";
                default:
                    return NULL;
            }
        default:
            return NULL;
    }
}

void APIENTRY glClear(GLbitfield mask) {
    if (!g_context) return;

    g_context.pendingClearFlags |= mask;
}

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
	if (!g_context) return;
    g_context.clearColor = MTLClearColorMake(red, green, blue, alpha);
}

void APIENTRY glClearDepth(GLclampd depth) {
    if (!g_context) return;
    g_context.clearDepth = depth;
}

static MTLRenderPassDescriptor* createRenderPassDescriptor() {
    if (!g_context || !g_context.currentDrawable) return nil;

    CGSize drawableSize = CGSizeMake(g_context.currentDrawable.texture.width, g_context.currentDrawable.texture.height);
    ensureDepthTexture(drawableSize);

    MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];

    passDesc.colorAttachments[0].texture = g_context.currentDrawable.texture;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    if (g_context.pendingClearFlags & GL_COLOR_BUFFER_BIT) {
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].clearColor = g_context.clearColor;
        g_context.pendingClearFlags &= ~GL_COLOR_BUFFER_BIT;
    } else {
        passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }

    passDesc.depthAttachment.texture = g_context.depthTexture;
    passDesc.depthAttachment.storeAction = MTLStoreActionDontCare;

    if (g_context.pendingClearFlags & GL_DEPTH_BUFFER_BIT) {
        passDesc.depthAttachment.loadAction = MTLLoadActionClear;
        passDesc.depthAttachment.clearDepth = g_context.clearDepth;
        g_context.pendingClearFlags &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        passDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    }
    
    return passDesc;
}

void APIENTRY glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!g_context) return;
    g_context.viewport = (gl4metalRect){x, y, width, height};
}

void APIENTRY glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!g_context) return;
    g_context.scissorRect = (gl4metalRect){x, y, width, height};
}

static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight) {
    if (!encoder || targetWidth == 0 || targetHeight == 0) return;

    gl4metalRect viewport = g_context.viewport;
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

    if (g_context.scissorTestEnabled) {
        gl4metalRect scissor = g_context.scissorRect;
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
        buffers[i] = g_nextBufferId++;
    }
}

void APIENTRY glBindBuffer(GLenum target, GLuint buffer) {
    g_context.boundBuffers[@(target)] = @(buffer);

    if (target == GL_ELEMENT_ARRAY_BUFFER && g_context.currentVAO != 0) {
        gl4metalVertexArray *vao = g_vaoObjects[@(g_context.currentVAO)];
        if (vao) {
            vao.elementArrayBufferID = buffer;
        }

    }
}

void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    GLuint currentBound = [g_context.boundBuffers[@(target)] unsignedIntValue];

    if (currentBound == 0) {
        NSLog(@"[gl4metal] ERROR: glBufferData called with no bound buffer for target 0x%X", target);
        return;
    }

    id<MTLBuffer> mtlBuffer = nil;
    if (data != NULL) {
        mtlBuffer = [g_context.device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    } else {
        mtlBuffer = [g_context.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    }

    if (mtlBuffer) {
        g_bufferObjects[@(currentBound)] = mtlBuffer;
    }
}

void* APIENTRY glMapBuffer(GLenum target, GLenum access) {
    GLuint currentBound = [g_context.boundBuffers[@(target)] unsignedIntValue];

    id<MTLBuffer> mtlBuffer = g_bufferObjects[@(currentBound)];
    if (!mtlBuffer) {
        NSLog(@"[gl4metal] ERROR: glMapbuffer: No MTLBuffer found for ID %u at target 0x%X", currentBound, target);
        return NULL;
    }

    return [mtlBuffer contents];
}

GLboolean APIENTRY glUnmapBuffer(GLenum target) {
    return GL_TRUE;
}

// VAO State
void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays) {
    if (!arrays) return;

    for (GLsizei i = 0; i < n; i++) {
        GLuint vaoId = g_nextVAOId++;
        arrays[i] = vaoId;
        gl4metalVertexArray *vao = [[gl4metalVertexArray alloc] init];
        vao.id = vaoId;
        g_vaoObjects[@(vaoId)] = vao;
    }
}

void APIENTRY glBindVertexArray(GLuint array) {
    g_context.currentVAO = array;
}

void APIENTRY glEnableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = g_vaoObjects[@(g_context.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_TRUE;
    }
}

void APIENTRY glDisableVertexAttribArray(GLuint index) {
    if (index >= 16) return;
    gl4metalVertexArray *vao = g_vaoObjects[@(g_context.currentVAO)];
    if (vao) {
        vao->attribs[index].enabled = GL_FALSE;
    }
}

void APIENTRY glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= 16) return;

    gl4metalVertexArray *vao = g_vaoObjects[@(g_context.currentVAO)];

    if (!vao) {
        NSLog(@"[gl4metal] ERROR: glVertexAttribPointer called without bound VAO!");
        return;
    }

    // get VBO binding in GL_ARRAY_BUFFER
    GLuint currentVBO = [g_context.boundBuffers[@(GL_ARRAY_BUFFER)] unsignedIntValue];

    vao->attribs[index].enabled = GL_TRUE;
    vao->attribs[index].size = size;
    vao->attribs[index].type = type;
    vao->attribs[index].normalized = normalized;
    vao->attribs[index].stride = stride;
    vao->attribs[index].pointer = pointer;
    vao->attribs[index].boundVBO = currentVBO;
}

void APIENTRY glDeleteVertexArrays(GLsizei n, const GLuint *arrays) {
    if (!arrays) return;
    for (GLsizei i = 0; i < n; i++) {
        [g_vaoObjects removeObjectForKey:@(arrays[i])];
    }
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
    if (!g_context || count <= 0) return;

    if (!g_context.currentCommandBuffer) g_context.currentCommandBuffer = [g_context.commandQueue commandBuffer];
    if (!g_context.currentDrawable) g_context.currentDrawable = [g_context.metalLayer nextDrawable];

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();

    id<MTLRenderCommandEncoder> encoder = [g_context.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) NSLog(@"Failed to create MTLRenderCommandEncoder"); return;

    // Set viewport and scissor rect based on current context settings
    NSUInteger targetWidth = g_context.currentDrawable.texture.width;
    NSUInteger targetHeight = g_context.currentDrawable.texture.height;
    applyViewportAndScissor(encoder, targetWidth, targetHeight);

    if (g_context.depthStencilState) {
        [encoder setDepthStencilState:g_context.depthStencilState];
    }

    [encoder setRenderPipelineState:g_context.pipelineState];

    gl4metalVertexArray *currentVAO = g_vaoObjects[@(g_context.currentVAO)];
    if (currentVAO) {
        for (NSUInteger index = 0; index < 16; index++) {
            gl4metalVertexAttrib *attrib = &currentVAO->attribs[index];
            if (attrib->enabled && attrib->boundVBO != 0) {
                id<MTLBuffer> buffer = g_bufferObjects[@(attrib->boundVBO)];
                if (buffer) {
                    NSUInteger offset = (NSUInteger)attrib->pointer;
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
    if (!g_context || !g_context.pipelineState || count <= 0) return;

    gl4metalVertexArray *currentVAO = g_vaoObjects[@(g_context.currentVAO)];
    GLuint elementVBO = currentVAO ? currentVAO.elementArrayBufferID : [g_context.boundBuffers[@(GL_ELEMENT_ARRAY_BUFFER)] unsignedIntValue];

    if (elementVBO == 0) {
        NSLog(@"[gl4metal] ERROR: glDrawElements called without bound GL_ELEMENT_ARRAY_BUFFER");
        return;
    }

    id<MTLBuffer> indexBuffer = g_bufferObjects[@(elementVBO)];
    if (!indexBuffer) {
        NSLog(@"[gl4metal] ERROR: Index MTLBuffer not found for VBO ID %u", elementVBO);
        return;
    }

    if (!g_context.currentCommandBuffer) g_context.currentCommandBuffer = [g_context.commandQueue commandBuffer];
    if (!g_context.currentDrawable) g_context.currentDrawable = [g_context.metalLayer nextDrawable];

    if (!g_context.currentDrawable) {
        NSLog(@"[gl4metal] ERROR: No drawable available for rendering");
        return;
    }

    MTLRenderPassDescriptor *renderPassDesc = createRenderPassDescriptor();

    id<MTLRenderCommandEncoder> encoder = [g_context.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        NSLog(@"[gl4metal] ERROR: Failed to create MTLRenderCommandEncoder");
        return;
    }

    [encoder setRenderPipelineState:g_context.pipelineState];

    if (currentVAO) {
        for (NSUInteger index = 0; index < 16; index++) {
            gl4metalVertexAttrib *attrib = &currentVAO->attribs[index];
            if (attrib->enabled && attrib->boundVBO != 0) {
                id<MTLBuffer> buffer = g_bufferObjects[@(attrib->boundVBO)];
                if (buffer) {
                    NSUInteger offset = (NSUInteger)attrib->pointer;
                    [encoder setVertexBuffer:buffer offset:offset atIndex:index];
                }
            }
        }
    }

    MTLPrimitiveType primitiveType = getMetalPrimitiveType(mode);
    MTLIndexType indexType = getMetalIndexType(type);
    NSUInteger indexOffset = (NSUInteger)indices;

    [encoder drawIndexedPrimitives:primitiveType indexCount:(NSUInteger)count indexType:indexType indexBuffer:indexBuffer indexBufferOffset:indexOffset];

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
    if (!g_context || !g_context.device) return;

    MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];

    if (g_context.depthTestEnabled) {
        desc.depthCompareFunction = getMetalCompareFunction(g_context.depthFunc);
        desc.depthWriteEnabled = g_context.depthWriteEnabled;
    } else {
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = NO;
    }

    g_context.depthStencilState = [g_context.device newDepthStencilStateWithDescriptor:desc];
}

static void ensureDepthTexture(CGSize size) {
    if (!g_context.depthTexture || g_context.depthTexture.width != size.width || g_context.depthTexture.height != size.height) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:size.width height:size.height mipmapped:NO];
        desc.usage = MTLTextureUsageRenderTarget;
        desc.storageMode = MTLStorageModePrivate; // only gpu can access Depth Buffer
        g_context.depthTexture = [g_context.device newTextureWithDescriptor:desc];
        NSLog(@"[gl4metal] Created new depth texture of size: %.0fx%.0f", size.width, size.height);
    }
}

BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor) {
    if (!g_context || !g_context.device) return NO;

    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunction;
    pipelineDesc.fragmentFunction = fragmentFunction;
    pipelineDesc.vertexDescriptor = vertexDescriptor;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    NSError *error = nil;
    g_context.pipelineState = [g_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (error || !g_context.pipelineState) {
        NSLog(@"[gl4metal] ERROR: Failed to create pipeline state: %@", error.localizedDescription);
        return NO;
    }
    return YES;
}

void APIENTRY glEnable(GLenum cap) {
    if (!g_context) return;

    if (cap == GL_DEPTH_TEST) {
        if (!g_context.depthTestEnabled) {
            g_context.depthTestEnabled = GL_TRUE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (!g_context.scissorTestEnabled) {
            g_context.scissorTestEnabled = GL_TRUE;
        }
    }
}

void APIENTRY glDisable(GLenum cap) {
    if (cap == GL_DEPTH_TEST) {
        if (g_context.depthTestEnabled) {
            g_context.depthTestEnabled = GL_FALSE;
            updateDepthStencilState();
        }
    } else if (cap == GL_SCISSOR_TEST) {
        if (g_context.scissorTestEnabled) {
            g_context.scissorTestEnabled = GL_FALSE;
        }
    }
}

void APIENTRY glDepthFunc(GLenum func) {
    if (g_context.depthFunc != func) {
        g_context.depthFunc = func;
        updateDepthStencilState();
    }
}

void APIENTRY glDepthMask(GLboolean flag) {
    if (g_context.depthWriteEnabled != flag) {
        g_context.depthWriteEnabled = flag;
        updateDepthStencilState();
    }
}

void APIENTRY glUseProgram(GLuint program) {
    if (!g_context) return;
    // g_context.currentProgram = program;
    NSLog(@"[gl4metal] glUseProgram called with program ID: %u (not implemented)", program);
}

GLenum APIENTRY glCheckFramebufferStatus(GLenum target) {
    if (!g_context) return GL_FRAMEBUFFER_UNSUPPORTED;
    // Unimplemented: In a full implementation, you would check the framebuffer completeness here.
    NSLog(@"[gl4metal] glCheckFramebufferStatus called for target 0x%X (not fully implemented)", target);
    return GL_FRAMEBUFFER_COMPLETE;
}

void APIENTRY glGetIntegerv(GLenum pname, GLint *data) {
    if (!g_context || !data) return;
    NSLog(@"[gl4metal] glGetIntegerv called for pname 0x%X (not implemented)", pname);
}

void APIENTRY glGetFloatv(GLenum pname, GLfloat *data) {
    if (!g_context || !data) return;
    NSLog(@"[gl4metal] glGetFloatv called for pname 0x%X (not implemented)", pname);
}

void APIENTRY glGetBooleanv(GLenum pname, GLboolean *data) {
    if (!g_context || !data) return;
    NSLog(@"[gl4metal] glGetBooleanv called for pname 0x%X (not implemented)", pname);
}

void APIENTRY glGetVertexAttribiv(GLuint index, GLenum pname, GLint *params) {
    if (!g_context || !params) return;
    NSLog(@"[gl4metal] glGetVertexAttribiv called for index %u and pname 0x%X (not implemented)", index, pname);
}

void APIENTRY glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat *params) {
    if (!g_context || !params) return;
    NSLog(@"[gl4metal] glGetVertexAttribfv called for index %u and pname 0x%X (not implemented)", index, pname);
}

void APIENTRY glGetVertexAttribPointerv(GLuint index, GLenum pname, void **pointer) {
    if (!g_context || !pointer) return;
    NSLog(@"[gl4metal] glGetVertexAttribPointerv called for index %u and pname 0x%X (not implemented)", index, pname);
}

void APIENTRY glDeleteBuffers(GLsizei n, const GLuint *buffers) {
    if (!buffers) return;
    for (GLsizei i = 0; i < n; i++) {
        [g_bufferObjects removeObjectForKey:@(buffers[i])];
    }
}

void APIENTRY glDeleteProgram(GLuint program) {
    NSLog(@"[gl4metal] glDeleteProgram called for program ID: %u (not implemented)", program);
}

void APIENTRY glDeleteShader(GLuint shader) {
    NSLog(@"[gl4metal] glDeleteShader called for shader ID: %u (not implemented)", shader);
}

void APIENTRY glDetachShader(GLuint program, GLuint shader) {
    NSLog(@"[gl4metal] glDetachShader called for program ID: %u and shader ID: %u (not implemented)", program, shader);
}

void APIENTRY glLinkProgram(GLuint program) {
    NSLog(@"[gl4metal] glLinkProgram called for program ID: %u (not implemented)", program);
}

void APIENTRY glCompileShader(GLuint shader) {
    NSLog(@"[gl4metal] glCompileShader called for shader ID: %u (not implemented)", shader);
}

void APIENTRY glShaderSource(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length) {
    NSLog(@"[gl4metal] glShaderSource called for shader ID: %u (not implemented)", shader);
}

void APIENTRY glAttachShader(GLuint program, GLuint shader) {
    NSLog(@"[gl4metal] glAttachShader called for program ID: %u and shader ID: %u (not implemented)", program, shader);
}

void APIENTRY glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetProgramiv called for program ID: %u and pname 0x%X (not implemented)", program, pname);
}

void APIENTRY glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (!infoLog) return;
    NSLog(@"[gl4metal] glGetProgramInfoLog called for program ID: %u (not implemented)", program);
}

void APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetShaderiv called for shader ID: %u and pname 0x%X (not implemented)", shader, pname);
}

void APIENTRY glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    if (!infoLog) return;
    NSLog(@"[gl4metal] glGetShaderInfoLog called for shader ID: %u (not implemented)", shader);
}

void APIENTRY glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders) {
    if (!shaders) return;
    NSLog(@"[gl4metal] glGetAttachedShaders called for program ID: %u (not implemented)", program);
}

void APIENTRY glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    if (!name) return;
    NSLog(@"[gl4metal] glGetActiveUniform called for program ID: %u and index %u (not implemented)", program, index);
}

void APIENTRY glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
    if (!name) return;
    NSLog(@"[gl4metal] glGetActiveAttrib called for program ID: %u and index %u (not implemented)", program, index);
}

GLint APIENTRY glGetUniformLocation(GLuint program, const GLchar *name) {
    NSLog(@"[gl4metal] glGetUniformLocation called for program ID: %u and name: %s (not implemented)", program, name);
    return program;
}

void APIENTRY glUniform1f(GLint location, GLfloat v0) {
    NSLog(@"[gl4metal] glUniform1f called for location: %d and value: %f (not implemented)", location, v0);
}

void APIENTRY glUniform1i(GLint location, GLint v0) {
    NSLog(@"[gl4metal] glUniform1i called for location: %d and value: %d (not implemented)", location, v0);
}

void APIENTRY glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix4fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    NSLog(@"[gl4metal] glUniform4f called for location: %d and values: (%f, %f, %f, %f) (not implemented)", location, v0, v1, v2, v3);
}

void APIENTRY glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    NSLog(@"[gl4metal] glUniform4i called for location: %d and values: (%d, %d, %d, %d) (not implemented)", location, v0, v1, v2, v3);
}

void APIENTRY glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    NSLog(@"[gl4metal] glUniform3f called for location: %d and values: (%f, %f, %f) (not implemented)", location, v0, v1, v2);
}

void APIENTRY glUniform3i(GLint location, GLint v0, GLint v1, GLint v2) {
    NSLog(@"[gl4metal] glUniform3i called for location: %d and values: (%d, %d, %d) (not implemented)", location, v0, v1, v2);
}

void APIENTRY glUniform2f(GLint location, GLfloat v0, GLfloat v1) {
    NSLog(@"[gl4metal] glUniform2f called for location: %d and values: (%f, %f) (not implemented)", location, v0, v1);
}

void APIENTRY glUniform2i(GLint location, GLint v0, GLint v1) {
    NSLog(@"[gl4metal] glUniform2i called for location: %d and values: (%d, %d) (not implemented)", location, v0, v1);
}

void APIENTRY glUniform1fv(GLint location, GLsizei count, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniform1fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniform1iv(GLint location, GLsizei count, const GLint *value) {
    NSLog(@"[gl4metal] glUniform1iv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix3fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix2fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix2x3fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix3x2fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix2x4fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix4x2fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix3x4fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    NSLog(@"[gl4metal] glUniformMatrix4x3fv called for location: %d and count: %d (not implemented)", location, count);
}

void APIENTRY glGetUniformfv(GLuint program, GLint location, GLfloat *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetUniformfv called for program ID: %u and location: %d (not implemented)", program, location);
}

void APIENTRY glGetUniformiv(GLuint program, GLint location, GLint *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetUniformiv called for program ID: %u and location: %d (not implemented)", program, location);
}

void APIENTRY glGetUniformuiv(GLuint program, GLint location, GLuint *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetUniformuiv called for program ID: %u and location: %d (not implemented)", program, location);
}

GLint APIENTRY glGetAttribLocation(GLuint program, const GLchar *name) {
    NSLog(@"[gl4metal] glGetAttribLocation called for program ID: %u and name: %s (not implemented)", program, name);
    return program;
}

void APIENTRY glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source) {
    if (!source) return;
    NSLog(@"[gl4metal] glGetShaderSource called for shader ID: %u (not implemented)", shader);
}

void APIENTRY glGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary) {
    if (!binary) return;
    NSLog(@"[gl4metal] glGetProgramBinary called for program ID: %u (not implemented)", program);
}

void APIENTRY glProgramBinary(GLuint program, GLenum binaryFormat, const void *binary, GLsizei length) {
    NSLog(@"[gl4metal] glProgramBinary called for program ID: %u (not implemented)", program);
}

void APIENTRY glProgramParameteri(GLuint program, GLenum pname, GLint value) {
    NSLog(@"[gl4metal] glProgramParameteri called for program ID: %u and pname: 0x%X (not implemented)", program, pname);
}

void APIENTRY glGetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint *params) {
    if (!params) return;
    NSLog(@"[gl4metal] glGetProgramPipelineiv called for pipeline ID: %u and pname: 0x%X (not implemented)", pipeline, pname);
}

void APIENTRY glBindProgramPipeline(GLuint pipeline) {
    NSLog(@"[gl4metal] glBindProgramPipeline called for pipeline ID: %u (not implemented)", pipeline);
}

void APIENTRY glDeleteProgramPipelines(GLsizei n, const GLuint *pipelines) {
    if (!pipelines) return;
    NSLog(@"[gl4metal] glDeleteProgramPipelines called for %d pipelines (not implemented)", n);
}