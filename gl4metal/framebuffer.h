//
// framebuffer.h
// gl4metal
//
// Created by congcq on 16.10.26.
//

#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#import "gl4metal.h"

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

id<MTLRenderCommandEncoder> beginRenderPass(GLbitfield clearMask, bool preservePassSerial = false);  // reuses a compatible open pass or creates one for the current draw FBO
// The wrapper records the exact call site that closes a render encoder when
// GL4METAL_TRACE_RENDER_PASSES=1. Keep all call sites source-compatible.
void endRenderEncoderAt(const char *callerFile, int callerLine);
#define endRenderEncoder() endRenderEncoderAt(__FILE__, __LINE__)
void applyViewportAndScissor(void);
void clearDepthStencil(GLbitfield mask, GLfloat depth, GLint stencil);
void clearOneColorBuffer(GLint drawbuffer, const GLfloat *rgba);

// Internal hooks used by GL_ARB_invalidate_subdata entry points.
void gl4metalInvalidateFramebufferImpl(GLenum target, GLsizei numAttachments, const GLenum *attachments);
void gl4metalInvalidateSubFramebufferImpl(GLenum target, GLsizei numAttachments, const GLenum *attachments,
	GLint x, GLint y, GLsizei width, GLsizei height);

#endif /* FRAMEBUFFER_H */
