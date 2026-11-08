#import "exports.h"

// ARB_draw_buffers_blend uses the ARB-suffixed names in a GL 3.x context.
void glBlendFunciARB(GLuint bufferIndex, GLenum src, GLenum dst) {
	glBlendFunci(bufferIndex, src, dst);
}

void glBlendFuncSeparateiARB(GLuint bufferIndex, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
	glBlendFuncSeparatei(bufferIndex, srcRGB, dstRGB, srcAlpha, dstAlpha);
}

void glBlendEquationiARB(GLuint bufferIndex, GLenum mode) {
	glBlendEquationi(bufferIndex, mode);
}

void glBlendEquationSeparateiARB(GLuint bufferIndex, GLenum modeRGB, GLenum modeAlpha) {
	glBlendEquationSeparatei(bufferIndex, modeRGB, modeAlpha);
}