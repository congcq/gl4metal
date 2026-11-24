#import "transformfeedback.h"

void glBeginTransformFeedback(GLenum primitiveMode) {
	if (primitiveMode != GL_POINTS && primitiveMode != GL_LINES && primitiveMode != GL_TRIANGLES) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	// Metal has no GL transform-feedback equivalent in the current translator.
	// Do not report success: draws must not proceed under a false capture state.
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glBeginTransformFeedback: vertex-output capture is not implemented on the iOS Metal backend");
}

void glEndTransformFeedback(void) {
	// glBeginTransformFeedback cannot enter the active state until capture exists.
	gl4metalSetError(GL_INVALID_OPERATION);
}

void glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar *const *varyings, GLenum bufferMode) {
	if (count < 0 || (count > 0 && !varyings)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (bufferMode != GL_INTERLEAVED_ATTRIBS && bufferMode != GL_SEPARATE_ATTRIBS) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (ctx.program_objects.find(program) == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	std::unordered_set<std::string> uniqueVaryings;
	for (GLsizei varyingIndex = 0; varyingIndex < count; ++varyingIndex) {
		if (!varyings[varyingIndex] || !varyings[varyingIndex][0]) {
			gl4metalSetError(GL_INVALID_VALUE);
			return;
		}
		if (!uniqueVaryings.insert(varyings[varyingIndex]).second) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
	}
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glTransformFeedbackVaryings: capture layout validated, but Metal vertex-output instrumentation is not implemented");
}

void glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length,
	GLsizei *size, GLenum *type, GLchar *name) {
	if (bufSize < 0 || (bufSize > 0 && !name)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (ctx.program_objects.find(program) == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		if (size) *size = 0;
		if (type) *type = 0;
		return;
	}
	// No varying can be queried until glTransformFeedbackVaryings can be linked
	// into the Metal shader's capture ABI.
	gl4metalSetError(GL_INVALID_VALUE);
	if (length) *length = 0;
	if (size) *size = 0;
	if (type) *type = 0;
	if (name && bufSize > 0) name[0] = '\0';
	(void)index;
}
