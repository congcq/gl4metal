//
// gl4metal.h
// gl4metal
//
// Created by congcq on 03.09.26.
//

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <glcorearb.h>

typedef struct {
    GLboolean enabled;
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLsizei stride;
    const void *pointer;
    GLuint boundVBO;
} gl4metalVertexAttrib;

@interface gl4metalVertexArray : NSObject {
    @public gl4metalVertexAttrib attribs[16];
}

@property (nonatomic, assign) GLuint id;
@property (nonatomic, assign) GLuint elementArrayBufferID; // for GL_ELEMENT_ARRAY_BUFFER

@end

@interface gl4metalContext : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSNumber *> *boundBuffers;
@property (nonatomic, assign) GLuint currentVAO;
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

@end

#ifdef __cplusplus
extern "C" {
#endif

GLboolean glInit(void);
const GLubyte* APIENTRY glGetString(GLenum name);
void APIENTRY glMakeCurrent(void *metalLayerPtr);
void APIENTRY glSwapBuffers(void);
void APIENTRY glSwapInterval(GLint interval);

void APIENTRY glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void APIENTRY glViewport(GLint x, GLint y, GLsizei width, GLsizei height);

void APIENTRY glGenBuffers(GLsizei n, GLuint *buffers);
void APIENTRY glBindBuffer(GLenum target, GLuint buffer);
void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
void* APIENTRY glMapBuffer(GLenum target, GLenum access);
GLboolean APIENTRY glUnmapBuffer(GLenum target);

// VAO Functions
void APIENTRY glGenVertexArrays(GLsizei n, GLuint *arrays);
void APIENTRY glBindVertexArray(GLuint array);
void APIENTRY glEnableVertexAttribArray(GLuint index);
void APIENTRY glDisableVertexAttribArray(GLuint index);
void APIENTRY glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
void APIENTRY glDeleteVertexArrays(GLsizei n, const GLuint *arrays);

void APIENTRY glDrawArrays(GLenum mode, GLint first, GLsizei count);
void APIENTRY glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices);
void APIENTRY glUseProgram(GLuint program);

void APIENTRY glEnable(GLenum cap);
void APIENTRY glDisable(GLenum cap);
void APIENTRY glDepthFunc(GLenum func);
void APIENTRY glDepthMask(GLboolean flag);
void APIENTRY glClear(GLbitfield mask);
void APIENTRY glClearDepth(GLclampd depth);

static void updateDepthStencilState(void);
static void ensureDepthTexture(CGSize size);
BOOL gl4metalCreatePipelineState(id<MTLFunction> vertexFunction, id<MTLFunction> fragmentFunction, MTLVertexDescriptor *vertexDescriptor);

#ifdef __cplusplus
}
#endif