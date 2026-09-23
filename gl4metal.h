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
@property (nonatomic, assign) NSUInteger gpuFamily;
@property (nonatomic, copy) NSString *gpuFamilyName;
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
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *programPipelines;

@property (nonatomic, assign) GLboolean depthTestEnabled;
@property (nonatomic, assign) GLboolean depthWriteEnabled;
@property (nonatomic, assign) GLenum depthFunc;
@property (nonatomic, assign) GLboolean cullEnabled;
@property (nonatomic, assign) GLenum cullFace;
@property (nonatomic, assign) GLenum frontFace;
@property (nonatomic, assign) GLenum polygonMode;
@property (nonatomic, assign) GLfloat lineWidth;
@property (nonatomic, assign) GLfloat pointSize;
@property (nonatomic, assign) GLint clearStencil;
@property (nonatomic, assign) GLuint stencilWriteMask;
@property (nonatomic, assign) GLboolean colorMaskRed;
@property (nonatomic, assign) GLboolean colorMaskGreen;
@property (nonatomic, assign) GLboolean colorMaskBlue;
@property (nonatomic, assign) GLboolean colorMaskAlpha;
@property (nonatomic, assign) GLenum drawBuffer;
@property (nonatomic, assign) GLenum readBuffer;
@property (nonatomic, assign) GLboolean blendEnabled;
@property (nonatomic, assign) GLenum blendSourceRGB;
@property (nonatomic, assign) GLenum blendDestinationRGB;
@property (nonatomic, assign) GLenum blendSourceAlpha;
@property (nonatomic, assign) GLenum blendDestinationAlpha;
@property (nonatomic, assign) GLenum blendEquationRGB;
@property (nonatomic, assign) GLenum blendEquationAlpha;
@property (nonatomic, assign) GLdouble depthRangeNear;
@property (nonatomic, assign) GLdouble depthRangeFar;
@property (nonatomic, assign) GLenum stencilFunc;
@property (nonatomic, assign) GLint stencilRef;
@property (nonatomic, assign) GLuint stencilValueMask;
@property (nonatomic, assign) GLenum stencilFail;
@property (nonatomic, assign) GLenum stencilDepthFail;
@property (nonatomic, assign) GLenum stencilDepthPass;
@property (nonatomic, assign) GLfloat polygonOffsetFactor;
@property (nonatomic, assign) GLfloat polygonOffsetUnits;
@property (nonatomic, assign) MTLClearColor blendColor;
@property (nonatomic, assign) GLint unpackAlignment;
@property (nonatomic, assign) GLint packAlignment;

@property (nonatomic, strong) id<MTLDepthStencilState> depthStencilState;
@property (nonatomic, strong) id<MTLSamplerState> defaultSamplerState;
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
const char *gl4metalGetDeviceName(void);
NSUInteger gl4metalGetGPUFamily(void);
const GLubyte* APIENTRY glGetString(GLenum name);
void APIENTRY glMakeCurrent(void *metalLayerPtr);
void APIENTRY glSwapBuffers(void);
void APIENTRY glSwapInterval(GLint interval);

static void updateDepthStencilState(void);
static void ensureDepthTexture(CGSize size);
static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight);
static MTLRenderPassDescriptor* createRenderPassDescriptor();
gl4metalVertexArray *gl4metalGetCurrentVAO(void);
MTLVertexDescriptor *gl4metalCreateVertexDescriptorForCurrentVAO(void);
NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao);
BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor);
void gl4metalSetError(GLenum error);

#ifdef __cplusplus
}
#endif