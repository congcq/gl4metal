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
    GLboolean integer;
    GLsizei stride;
    GLintptr pointerOffset;
    GLuint boundVBO;
    GLfloat currentValue[4];
    GLint currentIntValue[4];
    GLuint currentUIntValue[4];
    GLboolean currentValueIsInteger;
    GLboolean currentValueIsUnsigned;
    GLuint divisor;
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
@property (nonatomic, strong) NSMutableArray<id<MTLBuffer>> *uniformArenaBuffers;
@property (nonatomic, assign) NSUInteger uniformArenaOffset;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *programPipelines;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSString *> *programResourceInfo;

@property (nonatomic, assign) GLboolean depthTestEnabled;
@property (nonatomic, assign) GLboolean blendEnabled;
@property (nonatomic, assign) GLenum blendSrcRGB;
@property (nonatomic, assign) GLenum blendDstRGB;
@property (nonatomic, assign) GLenum blendSrcAlpha;
@property (nonatomic, assign) GLenum blendDstAlpha;
@property (nonatomic, assign) GLenum blendEquationRGB;
@property (nonatomic, assign) GLenum blendEquationAlpha;
@property (nonatomic, assign) GLfloat blendConstantRed;
@property (nonatomic, assign) GLfloat blendConstantGreen;
@property (nonatomic, assign) GLfloat blendConstantBlue;
@property (nonatomic, assign) GLfloat blendConstantAlpha;
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

@property (nonatomic, strong) id<MTLDepthStencilState> depthStencilState;
@property (nonatomic, strong) id<MTLTexture> depthTexture;

@property (nonatomic, assign) MTLClearColor clearColor;
@property (nonatomic, assign) double clearDepth;
@property (nonatomic, assign) GLbitfield pendingClearFlags;
@property (nonatomic, assign) GLuint pendingClearFramebuffer;

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
void APIENTRY glUniform2fv(GLint location, GLsizei count, const GLfloat *value);
void APIENTRY glUniform3fv(GLint location, GLsizei count, const GLfloat *value);
void APIENTRY glUniform4fv(GLint location, GLsizei count, const GLfloat *value);
void APIENTRY glUniform2iv(GLint location, GLsizei count, const GLint *value);
void APIENTRY glUniform3iv(GLint location, GLsizei count, const GLint *value);
void APIENTRY glUniform4iv(GLint location, GLsizei count, const GLint *value);

static void updateDepthStencilState(void);
static void ensureDepthTexture(CGSize size);
static void applyViewportAndScissor(id<MTLRenderCommandEncoder> encoder, NSUInteger targetWidth, NSUInteger targetHeight);
static MTLRenderPassDescriptor* createRenderPassDescriptor();
gl4metalVertexArray *gl4metalGetCurrentVAO(void);
MTLVertexDescriptor *gl4metalCreateVertexDescriptorForCurrentVAO(void);
NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao);
BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor);
BOOL gl4metalCreatePipelineStateForProgram(GLuint program, id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor);
void gl4metalDumpProgramResourceInfo(GLuint program);
void gl4metalSetError(GLenum error);
void gl4metalRegisterConstantVertexAttribute(NSUInteger index, MTLVertexFormat format);
void gl4metalUpdateConstantVertexAttribute(NSUInteger index);
void gl4metalInitTextureState(void);
void gl4metalInitFramebufferState(void);

// Current GL framebuffer -> Metal render-target bridge.
id<MTLTexture> gl4metalGetCurrentFramebufferColorTexture(void);
id<MTLTexture> gl4metalGetCurrentFramebufferDepthTexture(void);
id<MTLTexture> gl4metalGetCurrentFramebufferStencilTexture(void);
MTLPixelFormat gl4metalGetCurrentColorPixelFormat(void);
MTLPixelFormat gl4metalGetCurrentDepthPixelFormat(void);
MTLPixelFormat gl4metalGetCurrentStencilPixelFormat(void);
id<MTLTexture> gl4metalGetTextureForUnit(GLuint unit);
id<MTLSamplerState> gl4metalGetSamplerForUnit(GLuint unit);
GLuint gl4metalGetTextureNameForUnit(GLuint unit);

#ifdef __cplusplus
}
#endif