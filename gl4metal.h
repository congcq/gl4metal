//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "glcorearb.h"

typedef struct {
    GLboolean enabled;
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLsizei stride;
    GLintptr pointerOffset;
    GLuint boundVBO;
} gl4metalVertexAttrib;

typedef struct {
    GLint x;
    GLint y;
    GLsizei width;
    GLsizei height;
} gl4metalRect;

@interface gl4metalVertexArray : NSObject {
    @public gl4metalVertexAttrib attribs[16];
    @public GLuint arrayBufferID;
    @public GLuint elementArrayBufferID;
}

@property (nonatomic, assign) GLuint id;

@end

@interface gl4metalContext : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSNumber *> *boundBuffers;
@property (nonatomic, assign) GLuint currentVAO;
@property (nonatomic, assign) GLuint currentFramebuffer;
@property (nonatomic, assign) GLuint currentReadFramebuffer;
@property (nonatomic, assign) GLint swapInterval;

@property (nonatomic, strong) CAMetalLayer *metalLayer;
@property (nonatomic, strong) id<CAMetalDrawable> currentDrawable;
@property (nonatomic, strong) id<MTLCommandBuffer> currentCommandBuffer;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;

@property (nonatomic, assign) GLboolean depthTestEnabled;
@property (nonatomic, assign) GLboolean depthWriteEnabled;
@property (nonatomic, assign) GLenum depthFunc;

@property (nonatomic, strong) id<MTLDepthStencilState> depthStencilState;
@property (nonatomic, strong) id<MTLTexture> depthTexture;

@property (nonatomic, assign) MTLClearColor clearColor;
@property (nonatomic, assign) double clearDepth;
@property (nonatomic, assign) GLbitfield pendingClearFlags;

@property (nonatomic, assign) gl4metalRect viewport;
@property (nonatomic, assign) gl4metalRect scissorRect;
@property (nonatomic, assign) GLboolean scissorTestEnabled;

@end

#ifdef __cplusplus
extern "C" {
#endif

GLboolean gl4metalInit(void);
const GLubyte* APIENTRY glGetString(GLenum name);
void APIENTRY glMakeCurrent(void *metalLayerPtr);
void APIENTRY glSwapBuffers(void);
void APIENTRY glSwapInterval(GLint interval);

static void updateDepthStencilState(void);
static void ensureDepthTexture(CGSize size);
static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight);
static MTLRenderPassDescriptor* createRenderPassDescriptor();
BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor);

#ifdef __cplusplus
}
#endif