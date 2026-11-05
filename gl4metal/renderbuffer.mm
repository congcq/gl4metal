#import "renderbuffer.h"

void glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) {
	GL4METAL_LOG("glGenRenderbuffers: Unimplemented");
}

void glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
	GL4METAL_LOG("glDeleteRenderbuffers: Unimplemented");
}

GLboolean glIsRenderbuffer(GLuint renderbuffer) {
	GL4METAL_LOG("glIsRenderbuffer: Unimplemented, defaulting return GL_TRUE");
	return GL_TRUE;
}

void glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
	GL4METAL_LOG("glBindRenderbuffer: Unimplemented");
}

void glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
	GL4METAL_LOG("glRenderbufferStorage: Unimplemented");
}

void glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) {
	GL4METAL_LOG("glRenderbufferStorageMultisample: Unimplemented");
}

void glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetRenderbufferParameteriv: Unimplemented");
}
