#import "gl4metal.h"
#import "texture.h"
#import "framebuffer.h"

#include <algorithm>

// GL_ARB_invalidate_subdata entry points. Buffer invalidation renames storage
// when useful; texture and FBO invalidations are conservative discard hints.
void glInvalidateBufferData(GLuint buffer) {
	auto it = ctx.buffer_objects.find(buffer);
	if (buffer == 0 || it == ctx.buffer_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const NSUInteger logicalSize = glBufferLogicalSize(buffer);
	if (logicalSize == 0) {
		invalidateConvertedVertexBuffers(buffer);
		it->second = nil;
		ctx.buffer_last_use.erase(buffer);
		refreshTextureBufferViews(buffer);
		return;
	}
	if (!ctx.device || logicalSize > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	// Invalidation makes the old contents undefined. Rename to fresh shared storage
	// rather than synchronizing with a command buffer that may still read the old one.
	// Derived vertex conversions and texture-buffer views must also stop aliasing the old bytes.
	invalidateConvertedVertexBuffers(buffer);
	id<MTLBuffer> replacement = [ctx.device newBufferWithLength:logicalSize options:MTLResourceStorageModeShared];
	if (!replacement) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	it->second = replacement;
	ctx.buffer_last_use.erase(buffer);
	refreshTextureBufferViews(buffer);
}

void glInvalidateBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr length) {
	auto it = ctx.buffer_objects.find(buffer);
	if (buffer == 0 || it == ctx.buffer_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (offset < 0 || length < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const NSUInteger logicalSize = glBufferLogicalSize(buffer);
	const NSUInteger begin = (NSUInteger)offset;
	const NSUInteger count = (NSUInteger)length;
	if (begin > logicalSize || count > logicalSize - begin) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	// The call is a hint. Retaining the old bytes is legal and avoids an expensive
	// read-modify-write copy for a range that the application promises not to use.
}

void glInvalidateTexImage(GLuint texture, GLint level) {
	auto it = ctx.texture_objects.find(texture);
	if (texture == 0 || it == ctx.texture_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (level < 0 || !it->second.texture || (NSUInteger)level >= it->second.texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	// This is a discard hint. GL4Metal currently retains texture storage here; render
	// attachment invalidation has a direct Metal store-action fast path in framebuffer.mm.
}

void glInvalidateTexSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
	GLsizei width, GLsizei height, GLsizei depth) {
	auto it = ctx.texture_objects.find(texture);
	if (texture == 0 || it == ctx.texture_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!it->second.texture || level < 0 || (NSUInteger)level >= it->second.texture.mipmapLevelCount ||
		xoffset < 0 || yoffset < 0 || zoffset < 0 || width < 0 || height < 0 || depth < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const NSUInteger levelWidth = std::max<NSUInteger>(1, (NSUInteger)it->second.width >> level);
	const NSUInteger levelHeight = std::max<NSUInteger>(1, (NSUInteger)it->second.height >> level);
	const NSUInteger levelDepth = std::max<NSUInteger>(1, (NSUInteger)it->second.depth >> level);
	if ((NSUInteger)xoffset > levelWidth || (NSUInteger)width > levelWidth - (NSUInteger)xoffset ||
		(NSUInteger)yoffset > levelHeight || (NSUInteger)height > levelHeight - (NSUInteger)yoffset ||
		(NSUInteger)zoffset > levelDepth || (NSUInteger)depth > levelDepth - (NSUInteger)zoffset) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	// Partial texture invalidation is conservatively retained until a subresource-level
	// Metal residency/discard path is available; preserving data is legal for this hint.
}

void glInvalidateFramebuffer(GLenum target, GLsizei numAttachments, const GLenum *attachments) {
	gl4metalInvalidateFramebufferImpl(target, numAttachments, attachments);
}

void glInvalidateSubFramebuffer(GLenum target, GLsizei numAttachments, const GLenum *attachments,
	GLint x, GLint y, GLsizei width, GLsizei height) {
	gl4metalInvalidateSubFramebufferImpl(target, numAttachments, attachments, x, y, width, height);
}
