#import "gl4metal.h"

#ifndef GL4METAL_RUNTIME_H
#define GL4METAL_RUNTIME_H

extern gl4metalContext *ctx;
extern GLuint nextBufferId;
extern GLuint nextVAOId;
extern NSMutableDictionary<NSNumber *, id<MTLBuffer>> *bufferObjects;
extern NSMutableDictionary<NSNumber *, NSNumber *> *bufferSizes;
extern NSMutableDictionary<NSString *, id<MTLBuffer>> *convertedIndexBuffers;
extern NSMutableDictionary<NSNumber *, gl4metalVertexArray *> *vaoObjects;
extern NSMutableDictionary<NSNumber *, NSNumber *> *framebufferObjects;
extern NSMutableDictionary<NSNumber *, NSNumber *> *renderbufferObjects;
extern GLenum lastGL4MetalError;
extern GLuint nextShaderId;
extern GLuint nextProgramId;
extern GLuint currentProgram;
extern float cachedFallbackMVP[16];
extern BOOL cachedFallbackMVPIsValid;
extern NSMutableDictionary<NSNumber *, NSNumber *> *shaderTypes;
extern NSMutableDictionary<NSNumber *, NSString *> *shaderSources;
extern NSMutableDictionary<NSNumber *, NSNumber *> *shaderCompileStatus;
extern NSMutableDictionary<NSNumber *, NSMutableArray<NSNumber *> *> *programShaders;
extern NSMutableDictionary<NSNumber *, NSNumber *> *programLinkStatus;
extern NSMutableDictionary<NSNumber *, NSString *> *programInfoLog;
extern NSMutableDictionary<NSNumber *, NSValue *> *uniform1fValues;
extern NSMutableDictionary<NSNumber *, NSValue *> *uniform2fValues;
extern NSMutableDictionary<NSNumber *, NSValue *> *uniform3fValues;
extern NSMutableDictionary<NSNumber *, NSValue *> *uniform4fValues;
extern NSMutableDictionary<NSNumber *, NSNumber *> *uniform1iValues;
extern NSMutableDictionary<NSNumber *, NSNumber *> *uniform2iValues;
extern NSMutableDictionary<NSNumber *, NSNumber *> *uniform3iValues;
extern NSMutableDictionary<NSNumber *, NSNumber *> *uniform4iValues;
extern NSMutableDictionary<NSNumber *, NSValue *> *uniformMatrix4fvValues;
extern NSMutableDictionary<NSNumber *, NSString *> *uniformLocationNames;
extern NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, NSString *> *> *programAttribBindings;
extern NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *pipelineLayoutCache;
extern id<MTLRenderPipelineState> defaultFallbackPipeline;

void gl4metalSetError(GLenum error);

#endif
