#import "vertexarray.h"

// MARK: - Vertex array objects

void glGenVertexArrays(GLsizei n, GLuint *arrays) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = ++ctx.next_vao_id;
		ctx.vao_objects[id] = VertexArray(); // zero initialized: all attributes disabled
		arrays[i] = id;
	}
}

void glDeleteVertexArrays(GLsizei n, const GLuint *arrays) {
	for (GLsizei i = 0; i < n; i++) {
		if (arrays[i] == 0) {
			continue;
		}
		if (ctx.current_vao == arrays[i]) {
			ctx.current_vao = 0;
		}
		ctx.vao_objects.erase(arrays[i]);
	}
}

GLboolean glIsVertexArray(GLuint array) {
	return array != 0 && ctx.vao_objects.find(array) != ctx.vao_objects.end();
}

void glBindVertexArray(GLuint array) {
	if (array != 0 && ctx.vao_objects.find(array) == ctx.vao_objects.end()) {
		ctx.vao_objects[array] = VertexArray();
	}
	ctx.current_vao = array;
}

// MARK: - Vertex attribute arrays

void glEnableVertexAttribArray(GLuint index) {
	if (index >= 16) {
		GL4METAL_LOG("glEnableVertexAttribArray: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].enabled = true;
}

void glDisableVertexAttribArray(GLuint index) {
	if (index >= 16) {
		GL4METAL_LOG("glDisableVertexAttribArray: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].enabled = false;
}

void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
	if (index >= 16) {
		GL4METAL_LOG("glVertexAttribPointer: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	VertexAttrib &a = ctx.vao_objects[ctx.current_vao].attribs[index];
	a.bound_vbo = boundBuffer(GL_ARRAY_BUFFER); // the array buffer binding at call time is captured into the VAO
	a.size = size;
	a.type = type;
	a.normalized = normalized;
	a.stride = stride;
	a.pointer = pointer; // byte offset into the VBO (core profile has no client-side arrays)
	a.integer = false;
}

void glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
	if (index >= 16) {
		GL4METAL_LOG("glVertexAttribIPointer: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	VertexAttrib &a = ctx.vao_objects[ctx.current_vao].attribs[index];
	a.bound_vbo = boundBuffer(GL_ARRAY_BUFFER);
	a.size = size;
	a.type = type;
	a.normalized = GL_FALSE;
	a.stride = stride;
	a.pointer = pointer;
	a.integer = true;
}

void glVertexAttribDivisor(GLuint index, GLuint divisor) {
	if (index >= 16) {
		GL4METAL_LOG("glVertexAttribDivisor: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].divisor = divisor;
}

// MARK: - Vertex attribute queries

void glGetVertexAttribdv(GLuint index, GLenum pname, GLdouble *params) {
	GL4METAL_LOG("glGetVertexAttribdv: Unimplemented");
}

void glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat *params) {
	GL4METAL_LOG("glGetVertexAttribfv: Unimplemented");
}

void glGetVertexAttribiv(GLuint index, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetVertexAttribiv: Unimplemented");
}

void glGetVertexAttribIiv(GLuint index, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetVertexAttribIiv: Unimplemented");
}

void glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint *params) {
	GL4METAL_LOG("glGetVertexAttribIuiv: Unimplemented");
}

void glGetVertexAttribPointerv(GLuint index, GLenum pname, void **pointer) {
	GL4METAL_LOG("glGetVertexAttribPointerv: Unimplemented");
}

// MARK: - Generic (constant) vertex attributes

void glVertexAttrib1d(GLuint index, GLdouble x) {
	GL4METAL_LOG("glVertexAttrib1d: Unimplemented");
}

void glVertexAttrib1dv(GLuint index, const GLdouble *v) {
	GL4METAL_LOG("glVertexAttrib1dv: Unimplemented");
}

void glVertexAttrib1f(GLuint index, GLfloat x) {
	GL4METAL_LOG("glVertexAttrib1f: Unimplemented");
}

void glVertexAttrib1fv(GLuint index, const GLfloat *v) {
	GL4METAL_LOG("glVertexAttrib1fv: Unimplemented");
}

void glVertexAttrib1s(GLuint index, GLshort x) {
	GL4METAL_LOG("glVertexAttrib1s: Unimplemented");
}

void glVertexAttrib1sv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttrib1sv: Unimplemented");
}

void glVertexAttrib2d(GLuint index, GLdouble x, GLdouble y) {
	GL4METAL_LOG("glVertexAttrib2d: Unimplemented");
}

void glVertexAttrib2dv(GLuint index, const GLdouble *v) {
	GL4METAL_LOG("glVertexAttrib2dv: Unimplemented");
}

void glVertexAttrib2f(GLuint index, GLfloat x, GLfloat y) {
	GL4METAL_LOG("glVertexAttrib2f: Unimplemented");
}

void glVertexAttrib2fv(GLuint index, const GLfloat *v) {
	GL4METAL_LOG("glVertexAttrib2fv: Unimplemented");
}

void glVertexAttrib2s(GLuint index, GLshort x, GLshort y) {
	GL4METAL_LOG("glVertexAttrib2s: Unimplemented");
}

void glVertexAttrib2sv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttrib2sv: Unimplemented");
}

void glVertexAttrib3d(GLuint index, GLdouble x, GLdouble y, GLdouble z) {
	GL4METAL_LOG("glVertexAttrib3d: Unimplemented");
}

void glVertexAttrib3dv(GLuint index, const GLdouble *v) {
	GL4METAL_LOG("glVertexAttrib3dv: Unimplemented");
}

void glVertexAttrib3f(GLuint index, GLfloat x, GLfloat y, GLfloat z) {
	GL4METAL_LOG("glVertexAttrib3f: Unimplemented");
}

void glVertexAttrib3fv(GLuint index, const GLfloat *v) {
	GL4METAL_LOG("glVertexAttrib3fv: Unimplemented");
}

void glVertexAttrib3s(GLuint index, GLshort x, GLshort y, GLshort z) {
	GL4METAL_LOG("glVertexAttrib3s: Unimplemented");
}

void glVertexAttrib3sv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttrib3sv: Unimplemented");
}

void glVertexAttrib4d(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
	GL4METAL_LOG("glVertexAttrib4d: Unimplemented");
}

void glVertexAttrib4dv(GLuint index, const GLdouble *v) {
	GL4METAL_LOG("glVertexAttrib4dv: Unimplemented");
}

void glVertexAttrib4f(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
	GL4METAL_LOG("glVertexAttrib4f: Unimplemented");
}

void glVertexAttrib4fv(GLuint index, const GLfloat *v) {
	GL4METAL_LOG("glVertexAttrib4fv: Unimplemented");
}

void glVertexAttrib4s(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w) {
	GL4METAL_LOG("glVertexAttrib4s: Unimplemented");
}

void glVertexAttrib4sv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttrib4sv: Unimplemented");
}

void glVertexAttrib4Nbv(GLuint index, const GLbyte *v) {
	GL4METAL_LOG("glVertexAttrib4Nbv: Unimplemented");
}

void glVertexAttrib4Niv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttrib4Niv: Unimplemented");
}

void glVertexAttrib4Nsv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttrib4Nsv: Unimplemented");
}

void glVertexAttrib4Nub(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w) {
	GL4METAL_LOG("glVertexAttrib4Nub: Unimplemented");
}

void glVertexAttrib4Nubv(GLuint index, const GLubyte *v) {
	GL4METAL_LOG("glVertexAttrib4Nubv: Unimplemented");
}

void glVertexAttrib4Nuiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttrib4Nuiv: Unimplemented");
}

void glVertexAttrib4Nusv(GLuint index, const GLushort *v) {
	GL4METAL_LOG("glVertexAttrib4Nusv: Unimplemented");
}

void glVertexAttrib4bv(GLuint index, const GLbyte *v) {
	GL4METAL_LOG("glVertexAttrib4bv: Unimplemented");
}

void glVertexAttrib4iv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttrib4iv: Unimplemented");
}

void glVertexAttrib4ubv(GLuint index, const GLubyte *v) {
	GL4METAL_LOG("glVertexAttrib4ubv: Unimplemented");
}

void glVertexAttrib4uiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttrib4uiv: Unimplemented");
}

void glVertexAttrib4usv(GLuint index, const GLushort *v) {
	GL4METAL_LOG("glVertexAttrib4usv: Unimplemented");
}

// MARK: - Integer vertex attributes

void glVertexAttribI1i(GLuint index, GLint x) {
	GL4METAL_LOG("glVertexAttribI1i: Unimplemented");
}

void glVertexAttribI1iv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttribI1iv: Unimplemented");
}

void glVertexAttribI2i(GLuint index, GLint x, GLint y) {
	GL4METAL_LOG("glVertexAttribI2i: Unimplemented");
}

void glVertexAttribI2iv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttribI2iv: Unimplemented");
}

void glVertexAttribI3i(GLuint index, GLint x, GLint y, GLint z) {
	GL4METAL_LOG("glVertexAttribI3i: Unimplemented");
}

void glVertexAttribI3iv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttribI3iv: Unimplemented");
}

void glVertexAttribI4i(GLuint index, GLint x, GLint y, GLint z, GLint w) {
	GL4METAL_LOG("glVertexAttribI4i: Unimplemented");
}

void glVertexAttribI4iv(GLuint index, const GLint *v) {
	GL4METAL_LOG("glVertexAttribI4iv: Unimplemented");
}

void glVertexAttribI1ui(GLuint index, GLuint x) {
	GL4METAL_LOG("glVertexAttribI1ui: Unimplemented");
}

void glVertexAttribI1uiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttribI1uiv: Unimplemented");
}

void glVertexAttribI2ui(GLuint index, GLuint x, GLuint y) {
	GL4METAL_LOG("glVertexAttribI2ui: Unimplemented");
}

void glVertexAttribI2uiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttribI2uiv: Unimplemented");
}

void glVertexAttribI3ui(GLuint index, GLuint x, GLuint y, GLuint z) {
	GL4METAL_LOG("glVertexAttribI3ui: Unimplemented");
}

void glVertexAttribI3uiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttribI3uiv: Unimplemented");
}

void glVertexAttribI4ui(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w) {
	GL4METAL_LOG("glVertexAttribI4ui: Unimplemented");
}

void glVertexAttribI4uiv(GLuint index, const GLuint *v) {
	GL4METAL_LOG("glVertexAttribI4uiv: Unimplemented");
}

void glVertexAttribI4bv(GLuint index, const GLbyte *v) {
	GL4METAL_LOG("glVertexAttribI4bv: Unimplemented");
}

void glVertexAttribI4sv(GLuint index, const GLshort *v) {
	GL4METAL_LOG("glVertexAttribI4sv: Unimplemented");
}

void glVertexAttribI4ubv(GLuint index, const GLubyte *v) {
	GL4METAL_LOG("glVertexAttribI4ubv: Unimplemented");
}

void glVertexAttribI4usv(GLuint index, const GLushort *v) {
	GL4METAL_LOG("glVertexAttribI4usv: Unimplemented");
}

// MARK: - Packed (2_10_10_10) vertex attributes

void glVertexAttribP1ui(GLuint index, GLenum type, GLboolean normalized, GLuint value) {
	GL4METAL_LOG("glVertexAttribP1ui: Unimplemented");
}

void glVertexAttribP1uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint *value) {
	GL4METAL_LOG("glVertexAttribP1uiv: Unimplemented");
}

void glVertexAttribP2ui(GLuint index, GLenum type, GLboolean normalized, GLuint value) {
	GL4METAL_LOG("glVertexAttribP2ui: Unimplemented");
}

void glVertexAttribP2uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint *value) {
	GL4METAL_LOG("glVertexAttribP2uiv: Unimplemented");
}

void glVertexAttribP3ui(GLuint index, GLenum type, GLboolean normalized, GLuint value) {
	GL4METAL_LOG("glVertexAttribP3ui: Unimplemented");
}

void glVertexAttribP3uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint *value) {
	GL4METAL_LOG("glVertexAttribP3uiv: Unimplemented");
}

void glVertexAttribP4ui(GLuint index, GLenum type, GLboolean normalized, GLuint value) {
	GL4METAL_LOG("glVertexAttribP4ui: Unimplemented");
}

void glVertexAttribP4uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint *value) {
	GL4METAL_LOG("glVertexAttribP4uiv: Unimplemented");
}
