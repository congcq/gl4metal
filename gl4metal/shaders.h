#ifndef SHADERS_H
#define SHADERS_H

#import "gl4metal.h"

#include <string>

void releaseShaderIfUnused(GLuint shader); // erases a deleted shader once no program uses it
void copyGLString(const std::string &src, GLsizei bufSize, GLsizei *length, GLchar *dst);

#endif // SHADERS_H
