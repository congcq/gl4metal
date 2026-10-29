#import "vertexarray.h"

// MARK: - Generic vertex attributes (the "current value" used when an attribute has no array enabled)

struct GenericAttrib {
	float f[4];
	int32_t i[4];
};

static GenericAttrib &genericAt(GLuint index) {
	static GenericAttrib attribs[16];
	static bool initialized = false;
	if (!initialized) {
		initialized = true;
		for (GenericAttrib &a : attribs) { // GL's initial value is (0, 0, 0, 1)
			a.f[0] = a.f[1] = a.f[2] = 0.0f; a.f[3] = 1.0f;
			a.i[0] = a.i[1] = a.i[2] = 0; a.i[3] = 1;
		}
	}
	return attribs[index & 15];
}

void genericAttribValue(GLuint index, bool integer, void *out16) {
	GenericAttrib &a = genericAt(index);
	memcpy(out16, integer ? (const void *)a.i : (const void *)a.f, 16);
}

static void setGenericFloat(GLuint index, int n, const float *values) {
	if (index >= 16) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glVertexAttrib*: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	GenericAttrib &a = genericAt(index);
	for (int c = 0; c < 4; c++) {
		a.f[c] = c < n ? values[c] : (c == 3 ? 1.0f : 0.0f); // missing components: 0, 0, 0, 1
	}
}

static void setGenericInt(GLuint index, int n, const int32_t *values) {
	if (index >= 16) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glVertexAttribI*: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	GenericAttrib &a = genericAt(index);
	for (int c = 0; c < 4; c++) {
		a.i[c] = c < n ? values[c] : (c == 3 ? 1 : 0);
	}
}

static float normalizeSigned(double v, double max) { return (float)std::max(v / max, -1.0); }

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
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glEnableVertexAttribArray: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].enabled = true;
}

void glDisableVertexAttribArray(GLuint index) {
	if (index >= 16) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDisableVertexAttribArray: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].enabled = false;
}

void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
	if (index >= 16) {
		gl4metalSetError(GL_INVALID_VALUE);
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
		gl4metalSetError(GL_INVALID_VALUE);
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
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glVertexAttribDivisor: index %u >= GL_MAX_VERTEX_ATTRIBS", index);
		return;
	}
	ctx.vao_objects[ctx.current_vao].attribs[index].divisor = divisor;
}

// MARK: - Vertex attribute queries

void glGetVertexAttribdv(GLuint index, GLenum pname, GLdouble *params) {
	GLfloat values[4] = { 0, 0, 0, 0 };
	glGetVertexAttribfv(index, pname, values);
	int count = pname == GL_CURRENT_VERTEX_ATTRIB ? 4 : 1;
	for (int c = 0; c < count; c++) {
		params[c] = values[c];
	}
}

void glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat *params) {
	if (pname == GL_CURRENT_VERTEX_ATTRIB && index < 16) {
		memcpy(params, genericAt(index).f, 16);
		return;
	}
	GLint value = 0;
	glGetVertexAttribiv(index, pname, &value);
	*params = (GLfloat)value;
}

void glGetVertexAttribiv(GLuint index, GLenum pname, GLint *params) {
	if (index >= 16) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetVertexAttribiv: index %u >= GL_MAX_VERTEX_ATTRIBS (GL_INVALID_VALUE)", index);
		return;
	}
	const VertexAttrib &a = ctx.vao_objects[ctx.current_vao].attribs[index];
	switch (pname) {
		case GL_VERTEX_ATTRIB_ARRAY_ENABLED:     *params = a.enabled; break;
		case GL_VERTEX_ATTRIB_ARRAY_SIZE:        *params = a.size ? a.size : 4; break;
		case GL_VERTEX_ATTRIB_ARRAY_TYPE:        *params = (GLint)(a.type ? a.type : GL_FLOAT); break;
		case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED:  *params = a.normalized; break;
		case GL_VERTEX_ATTRIB_ARRAY_STRIDE:      *params = a.stride; break;
		case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: *params = (GLint)a.bound_vbo; break;
		case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:     *params = (GLint)a.divisor; break;
		case GL_VERTEX_ATTRIB_ARRAY_INTEGER:     *params = a.integer; break;
		case GL_CURRENT_VERTEX_ATTRIB: {
			const GenericAttrib &g = genericAt(index);
			for (int c = 0; c < 4; c++) {
				params[c] = (GLint)g.f[c];
			}
			break;
		}
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetVertexAttribiv: unhandled pname 0x%X", pname);
			break;
	}
}

void glGetVertexAttribIiv(GLuint index, GLenum pname, GLint *params) {
	if (pname == GL_CURRENT_VERTEX_ATTRIB && index < 16) {
		memcpy(params, genericAt(index).i, 16);
		return;
	}
	glGetVertexAttribiv(index, pname, params);
}

void glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint *params) {
	glGetVertexAttribIiv(index, pname, (GLint *)params);
}

void glGetVertexAttribPointerv(GLuint index, GLenum pname, void **pointer) {
	*pointer = index < 16 ? (void *)ctx.vao_objects[ctx.current_vao].attribs[index].pointer : NULL;
}

// MARK: - Generic (constant) vertex attributes

void glVertexAttrib1d(GLuint index, GLdouble x) {
	const float values[1] = { (float)x };
	setGenericFloat(index, 1, values);
}

void glVertexAttrib1dv(GLuint index, const GLdouble *v) {
	float values[1];
	for (int i = 0; i < 1; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 1, values);
}

void glVertexAttrib1f(GLuint index, GLfloat x) {
	const float values[1] = { (float)x };
	setGenericFloat(index, 1, values);
}

void glVertexAttrib1fv(GLuint index, const GLfloat *v) {
	float values[1];
	for (int i = 0; i < 1; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 1, values);
}

void glVertexAttrib1s(GLuint index, GLshort x) {
	const float values[1] = { (float)x };
	setGenericFloat(index, 1, values);
}

void glVertexAttrib1sv(GLuint index, const GLshort *v) {
	float values[1];
	for (int i = 0; i < 1; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 1, values);
}

void glVertexAttrib2d(GLuint index, GLdouble x, GLdouble y) {
	const float values[2] = { (float)x, (float)y };
	setGenericFloat(index, 2, values);
}

void glVertexAttrib2dv(GLuint index, const GLdouble *v) {
	float values[2];
	for (int i = 0; i < 2; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 2, values);
}

void glVertexAttrib2f(GLuint index, GLfloat x, GLfloat y) {
	const float values[2] = { (float)x, (float)y };
	setGenericFloat(index, 2, values);
}

void glVertexAttrib2fv(GLuint index, const GLfloat *v) {
	float values[2];
	for (int i = 0; i < 2; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 2, values);
}

void glVertexAttrib2s(GLuint index, GLshort x, GLshort y) {
	const float values[2] = { (float)x, (float)y };
	setGenericFloat(index, 2, values);
}

void glVertexAttrib2sv(GLuint index, const GLshort *v) {
	float values[2];
	for (int i = 0; i < 2; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 2, values);
}

void glVertexAttrib3d(GLuint index, GLdouble x, GLdouble y, GLdouble z) {
	const float values[3] = { (float)x, (float)y, (float)z };
	setGenericFloat(index, 3, values);
}

void glVertexAttrib3dv(GLuint index, const GLdouble *v) {
	float values[3];
	for (int i = 0; i < 3; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 3, values);
}

void glVertexAttrib3f(GLuint index, GLfloat x, GLfloat y, GLfloat z) {
	const float values[3] = { (float)x, (float)y, (float)z };
	setGenericFloat(index, 3, values);
}

void glVertexAttrib3fv(GLuint index, const GLfloat *v) {
	float values[3];
	for (int i = 0; i < 3; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 3, values);
}

void glVertexAttrib3s(GLuint index, GLshort x, GLshort y, GLshort z) {
	const float values[3] = { (float)x, (float)y, (float)z };
	setGenericFloat(index, 3, values);
}

void glVertexAttrib3sv(GLuint index, const GLshort *v) {
	float values[3];
	for (int i = 0; i < 3; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 3, values);
}

void glVertexAttrib4d(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
	const float values[4] = { (float)x, (float)y, (float)z, (float)w };
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4dv(GLuint index, const GLdouble *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4f(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
	const float values[4] = { (float)x, (float)y, (float)z, (float)w };
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4fv(GLuint index, const GLfloat *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4s(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w) {
	const float values[4] = { (float)x, (float)y, (float)z, (float)w };
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4sv(GLuint index, const GLshort *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nbv(GLuint index, const GLbyte *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = normalizeSigned(v[i], 127.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Niv(GLuint index, const GLint *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = normalizeSigned(v[i], 2147483647.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nsv(GLuint index, const GLshort *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = normalizeSigned(v[i], 32767.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nub(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w) {
	const float values[4] = { x / 255.0f, y / 255.0f, z / 255.0f, w / 255.0f };
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nubv(GLuint index, const GLubyte *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)(v[i] / 255.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nuiv(GLuint index, const GLuint *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)(v[i] / 4294967295.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4Nusv(GLuint index, const GLushort *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)(v[i] / 65535.0);
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4bv(GLuint index, const GLbyte *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4iv(GLuint index, const GLint *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4ubv(GLuint index, const GLubyte *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4uiv(GLuint index, const GLuint *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

void glVertexAttrib4usv(GLuint index, const GLushort *v) {
	float values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (float)v[i];
	}
	setGenericFloat(index, 4, values);
}

// MARK: - Integer vertex attributes

void glVertexAttribI1i(GLuint index, GLint x) {
	const int32_t values[1] = { (int32_t)x };
	setGenericInt(index, 1, values);
}

void glVertexAttribI1iv(GLuint index, const GLint *v) {
	int32_t values[1];
	for (int i = 0; i < 1; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 1, values);
}

void glVertexAttribI2i(GLuint index, GLint x, GLint y) {
	const int32_t values[2] = { (int32_t)x, (int32_t)y };
	setGenericInt(index, 2, values);
}

void glVertexAttribI2iv(GLuint index, const GLint *v) {
	int32_t values[2];
	for (int i = 0; i < 2; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 2, values);
}

void glVertexAttribI3i(GLuint index, GLint x, GLint y, GLint z) {
	const int32_t values[3] = { (int32_t)x, (int32_t)y, (int32_t)z };
	setGenericInt(index, 3, values);
}

void glVertexAttribI3iv(GLuint index, const GLint *v) {
	int32_t values[3];
	for (int i = 0; i < 3; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 3, values);
}

void glVertexAttribI4i(GLuint index, GLint x, GLint y, GLint z, GLint w) {
	const int32_t values[4] = { (int32_t)x, (int32_t)y, (int32_t)z, (int32_t)w };
	setGenericInt(index, 4, values);
}

void glVertexAttribI4iv(GLuint index, const GLint *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
}

void glVertexAttribI1ui(GLuint index, GLuint x) {
	const int32_t values[1] = { (int32_t)x };
	setGenericInt(index, 1, values);
}

void glVertexAttribI1uiv(GLuint index, const GLuint *v) {
	int32_t values[1];
	for (int i = 0; i < 1; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 1, values);
}

void glVertexAttribI2ui(GLuint index, GLuint x, GLuint y) {
	const int32_t values[2] = { (int32_t)x, (int32_t)y };
	setGenericInt(index, 2, values);
}

void glVertexAttribI2uiv(GLuint index, const GLuint *v) {
	int32_t values[2];
	for (int i = 0; i < 2; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 2, values);
}

void glVertexAttribI3ui(GLuint index, GLuint x, GLuint y, GLuint z) {
	const int32_t values[3] = { (int32_t)x, (int32_t)y, (int32_t)z };
	setGenericInt(index, 3, values);
}

void glVertexAttribI3uiv(GLuint index, const GLuint *v) {
	int32_t values[3];
	for (int i = 0; i < 3; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 3, values);
}

void glVertexAttribI4ui(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w) {
	const int32_t values[4] = { (int32_t)x, (int32_t)y, (int32_t)z, (int32_t)w };
	setGenericInt(index, 4, values);
}

void glVertexAttribI4uiv(GLuint index, const GLuint *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
}

void glVertexAttribI4bv(GLuint index, const GLbyte *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
}

void glVertexAttribI4sv(GLuint index, const GLshort *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
}

void glVertexAttribI4ubv(GLuint index, const GLubyte *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
}

void glVertexAttribI4usv(GLuint index, const GLushort *v) {
	int32_t values[4];
	for (int i = 0; i < 4; i++) {
		values[i] = (int32_t)v[i];
	}
	setGenericInt(index, 4, values);
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
