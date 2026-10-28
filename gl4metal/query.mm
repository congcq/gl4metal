#import "query.h"

void glGenQueries(GLsizei n, GLuint *ids) {
	GL4METAL_LOG("glGenQueries: Unimplemented");
}

void glDeleteQueries(GLsizei n, const GLuint *ids) {
	GL4METAL_LOG("glDeleteQueries: Unimplemented");
}

GLboolean glIsQuery(GLuint id) {
	GL4METAL_LOG("glIsQuery: Unimplemented, defaulting return GL_TRUE");
	return GL_TRUE;
}

void glBeginQuery(GLenum target, GLuint id) {
	GL4METAL_LOG("glBeginQuery: Unimplemented");
}

void glEndQuery(GLenum target) {
	GL4METAL_LOG("glEndQuery: Unimplemented");
}

void glQueryCounter(GLuint id, GLenum target) {
	GL4METAL_LOG("glQueryCounter: Unimplemented");
}

void glGetQueryiv(GLenum target, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetQueryiv: Unimplemented");
}

void glGetQueryObjectiv(GLuint id, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetQueryObjectiv: Unimplemented");
}

void glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params) {
	GL4METAL_LOG("glGetQueryObjectuiv: Unimplemented");
}

void glGetQueryObjecti64v(GLuint id, GLenum pname, GLint64 *params) {
	GL4METAL_LOG("glGetQueryObjecti64v: Unimplemented");
}

void glGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *params) {
	GL4METAL_LOG("glGetQueryObjectui64v: Unimplemented");
}

// MARK: - Conditional rendering

void glBeginConditionalRender(GLuint id, GLenum mode) {
	GL4METAL_LOG("glBeginConditionalRender: Unimplemented");
}

void glEndConditionalRender(void) {
	GL4METAL_LOG("glEndConditionalRender: Unimplemented");
}
