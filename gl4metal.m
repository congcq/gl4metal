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

#pragma mark - OpenGL implementation

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
    if (!g_context || !g_context.currentCommandBuffer || !g_context.currentDrawable) {
    // avoid crash if the game call glSwapBuffers before draw anything
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

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
	if (!g_context.metalLayer) return;
	
	if (!g_context.currentCommandBuffer) {
		g_context.currentCommandBuffer = [g_context.commandQueue commandBuffer];
	}
	
	if (!g_context.currentDrawable) {
		g_context.currentDrawable = [g_context.metalLayer nextDrawable];
	}
	
    // NSLog(@"[gl4metal] glClearColor: (%.2f, %.2f, %.2f, %.2f)", red, green, blue, alpha);
    NSLog(@"[gl4metal] glClearColor: New frame, draw ready");
}

void APIENTRY glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    NSLog(@"[gl4metal] glViewport: x=%d, y=%d, width=%d, height=%d", x, y, width, height);
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

    CGSize drawableSize = CGSizeMake(g_context.currentDrawable.texture.width, g_context.currentDrawable.texture.height);
    ensureDepthTexture(drawableSize);

    MTLRenderPassDescriptor *renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPassDesc.colorAttachments[0].texture = g_context.currentDrawable.texture;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    renderPassDesc.depthAttachment.texture = g_context.depthTexture;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionDontCare;

    id<MTLRenderCommandEncoder> encoder = [g_context.currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) NSLog(@"Failed to create MTLRenderCommandEncoder"); return;

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

    MTLRenderPassDescriptor *renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPassDesc.colorAttachments[0].texture = g_context.currentDrawable.texture;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad; // Use existing content
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

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
    if (cap == GL_DEPTH_TEST) {
        if (!g_context.depthTestEnabled) {
            g_context.depthTestEnabled = GL_TRUE;
            updateDepthStencilState();
        }
    }
}

void APIENTRY glDisable(GLenum cap) {
    if (cap == GL_DEPTH_TEST) {
        if (g_context.depthTestEnabled) {
            g_context.depthTestEnabled = GL_FALSE;
            updateDepthStencilState();
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
    // For simplicity, we assume the framebuffer is always complete in this example
    return GL_FRAMEBUFFER_COMPLETE;
}

void APIENTRY glGetIntegerv(GLenum pname, GLint *data) {
    if (!g_context || !data) return;
}
