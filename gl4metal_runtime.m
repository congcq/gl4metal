#import "gl4metal_runtime.h"

gl4metalContext *ctx = nil;
GLuint nextBufferId = 1;
GLuint nextVAOId = 1;
GLuint nextTextureId = 1;
NSMutableDictionary<NSNumber *, id<MTLBuffer>> *bufferObjects = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *bufferSizes = nil;
NSMutableDictionary<NSString *, id<MTLBuffer>> *convertedIndexBuffers = nil;
NSMutableDictionary<NSNumber *, gl4metalVertexArray *> *vaoObjects = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *framebufferObjects = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *renderbufferObjects = nil;
GLenum lastGL4MetalError = GL_NO_ERROR;
GLuint nextShaderId = 1;
GLuint nextProgramId = 1;
GLuint currentProgram = 0;
float cachedFallbackMVP[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};
BOOL cachedFallbackMVPIsValid = NO;
NSMutableDictionary<NSNumber *, NSNumber *> *shaderTypes = nil;
NSMutableDictionary<NSNumber *, NSString *> *shaderSources = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *shaderCompileStatus = nil;
NSMutableDictionary<NSNumber *, NSMutableArray<NSNumber *> *> *programShaders = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *programLinkStatus = nil;
NSMutableDictionary<NSNumber *, NSString *> *programInfoLog = nil;
NSMutableDictionary<NSNumber *, NSValue *> *uniform1fValues = nil;
NSMutableDictionary<NSNumber *, NSValue *> *uniform2fValues = nil;
NSMutableDictionary<NSNumber *, NSValue *> *uniform3fValues = nil;
NSMutableDictionary<NSNumber *, NSValue *> *uniform4fValues = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *uniform1iValues = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *uniform2iValues = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *uniform3iValues = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *uniform4iValues = nil;
NSMutableDictionary<NSNumber *, NSValue *> *uniformMatrix4fvValues = nil;
NSMutableDictionary<NSNumber *, NSString *> *uniformLocationNames = nil;
NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSString *> *> *programAttribBindings = nil;
NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *pipelineLayoutCache = nil;
id<MTLRenderPipelineState> defaultFallbackPipeline = nil;
NSMutableDictionary<NSNumber *, id<MTLTexture>> *textureObjects = nil;
NSMutableDictionary<NSNumber *, NSNumber *> *boundTextureUnits = nil;
GLenum activeTextureUnit = GL_TEXTURE0;
NSMutableDictionary<NSNumber *, NSArray<NSDictionary *> *> *uniformLayouts = nil;
NSMutableDictionary<NSNumber *, id<MTLLibrary>> *programMetalLibraries = nil;
