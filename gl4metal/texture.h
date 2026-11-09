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

#endif /* TEXTURE_H */
