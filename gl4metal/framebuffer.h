//
// framebuffer.h
// gl4metal
//
// Created by congcq on 16.10.26.
//

#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#import <Metal/Metal.h>

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

#include "config.h"

struct FBOAttachment {
	GLuint texture = 0;       // GL_TEXTURE object, 0 when not a texture attachment
	GLuint renderbuffer = 0;  // GL_RENDERBUFFER object, 0 when not a renderbuffer attachment
	GLint level = 0;
	GLint layer = -1;         // selected 2D-array slice; -1 for ordinary 2D attachments
	mutable __strong id<MTLTexture> view = nil;        // cached single-slice 2D view
	mutable __strong id<MTLTexture> view_source = nil; // storage identity used to build view
};

struct RenderbufferObject {
	id<MTLTexture> texture = nil;
	GLenum internalformat = 0;
	GLsizei width = 0;
	GLsizei height = 0;
	GLsizei samples = 0;
	bool initialized = false;
};

struct FBO {
	FBOAttachment color[GL4METAL_MAX_COLOR_ATTACHMENTS];
	FBOAttachment depth, stencil;
	GLenum draw_buffers[GL4METAL_MAX_COLOR_ATTACHMENTS] = { GL_COLOR_ATTACHMENT0 };
	GLenum read_buffer = GL_COLOR_ATTACHMENT0;
	// Pending GL invalidation hints. They become MTLLoadActionDontCare at the next pass.
	uint32_t invalidated_color_mask = 0;
	bool invalidated_depth = false;
	bool invalidated_stencil = false;
};

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
