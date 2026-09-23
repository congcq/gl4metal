#import "gl4metal.h"

#ifndef GL4METAL_SHADERS_H
#define GL4METAL_SHADERS_H

BOOL gl4metalCreateDefaultProgramPipelineForProgram(GLuint program);
BOOL gl4metalCreateProgramPipelineForProgram(GLuint program);
void gl4metalCachePipelineForCurrentLayout(GLuint program);
NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao);
NSString *gl4metalBuildTranslatedMetalSource(NSString *vertexSource, NSString *fragmentSource, BOOL *hasColor, BOOL *hasTexCoord);
BOOL gl4metalShaderUsesAttribute(NSString *source, NSString *name);
void gl4metalRegisterUniformLayout(GLuint program, NSString *vertexSource, NSString *fragmentSource);
NSData *gl4metalBuildUniformDataForProgram(GLuint program);
void gl4metalBindFragmentResources(id<MTLRenderCommandEncoder> encoder, GLuint program);

#endif
