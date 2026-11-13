#ifndef TEXTURE_H
#define TEXTURE_H

#import "gl4metal.h"
#import "framebuffer.h"

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

MTLPixelFormat metalPixelFormat(GLenum internalformat);
void resetTextureUploadArena(id<MTLCommandBuffer> commandBuffer);
void refreshTextureBufferViews(GLuint bufferObject);
GLint maxTextureBufferTexels(void);

// Internal implementation hooks used by GL_ARB_clear_texture entry points.
void gl4metalClearTexImageImpl(GLuint texture, GLint level, GLenum format, GLenum type, const void *data);
void gl4metalClearTexSubImageImpl(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
	GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data);

#endif /* TEXTURE_H */
