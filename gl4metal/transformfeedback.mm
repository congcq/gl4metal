#import "transformfeedback.h"

void glBeginTransformFeedback(GLenum primitiveMode) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glBeginTransformFeedback: Unimplemented");
}

void glEndTransformFeedback(void) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glEndTransformFeedback: Unimplemented");
}

void glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glTransformFeedbackVaryings: Unimplemented");
}

void glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name) {
	gl4metalSetError(GL_INVALID_OPERATION);
	if (length) *length = 0;
	if (size) *size = 0;
	if (type) *type = 0;
	if (name && bufSize > 0) name[0] = '\0';
	GL4METAL_LOG("glGetTransformFeedbackVarying: Unimplemented");
}
