#import "transformfeedback.h"

void glBeginTransformFeedback(GLenum primitiveMode) {
	GL4METAL_LOG("glBeginTransformFeedback: Unimplemented");
}

void glEndTransformFeedback(void) {
	GL4METAL_LOG("glEndTransformFeedback: Unimplemented");
}

void glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode) {
	GL4METAL_LOG("glTransformFeedbackVaryings: Unimplemented");
}

void glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name) {
	GL4METAL_LOG("glGetTransformFeedbackVarying: Unimplemented");
}
