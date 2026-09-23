#ifndef GL4METAL_SHADER_COMPILER_H
#define GL4METAL_SHADER_COMPILER_H

#ifdef __cplusplus
extern "C" {
#endif

char *gl4metalCompileGLSLToMSL(const char *vertexSource, const char *fragmentSource);
void gl4metalFreeCompiledMSL(char *source);

#ifdef __cplusplus
}
#endif

#endif
