#import "texture.h"

// GL_ARB_clear_texture entry points. The format conversion and Metal staging
// implementation remain in texture.mm, alongside the texture upload helpers.
void glClearTexImage(GLuint texture, GLint level, GLenum format, GLenum type, const void *data) {
	gl4metalClearTexImageImpl(texture, level, format, type, data);
}

void glClearTexSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
	GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data) {
	gl4metalClearTexSubImageImpl(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data);
}
