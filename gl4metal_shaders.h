#import "gl4metal.h"

#ifndef GL4METAL_SHADERS_H
#define GL4METAL_SHADERS_H

#include <iostream>
#include <string>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <spirv_msl.hpp>

BOOL gl4metalCreateDefaultProgramPipelineForProgram(GLuint program);
BOOL gl4metalCreateProgramPipelineForProgram(GLuint program);
void gl4metalCachePipelineForCurrentLayout(GLuint program);
NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao);
NSString *gl4metalBuildTranslatedMetalSource(NSString *vertexSource, NSString *fragmentSource, BOOL *hasColor, BOOL *hasTexCoord);
BOOL gl4metalShaderUsesAttribute(NSString *source, NSString *name);
BOOL gl4metalApplyVertexShaderInputFormats(NSString *vertexSource, GLuint program, MTLVertexDescriptor *descriptor);
std::string transpileGLSLToMSL(GLenum shaderType, const std::string &glslSource,
                               std::string *diagnostics);
#endif
