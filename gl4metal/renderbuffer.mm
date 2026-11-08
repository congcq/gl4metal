#import "renderbuffer.h"

#import "texture.h"
#import "sync.h"

#include <algorithm>
#include <new>

static bool validRenderbufferTarget(GLenum target) {
	return target == GL_RENDERBUFFER;
}

static bool renderbufferFormatBits(GLenum internalformat, GLint pname, GLint &value) {
	value = 0;
	switch (pname) {
		case GL_RENDERBUFFER_RED_SIZE:
			value = (internalformat == GL_R8 || internalformat == GL_R16 || internalformat == GL_R16F || internalformat == GL_R32F) ?
				(internalformat == GL_R8 ? 8 : internalformat == GL_R16 ? 16 : internalformat == GL_R16F ? 16 : 32) :
				(internalformat == GL_RG8 || internalformat == GL_RG16 || internalformat == GL_RG16F || internalformat == GL_RG32F) ?
				(internalformat == GL_RG8 ? 8 : internalformat == GL_RG16 ? 16 : internalformat == GL_RG16F ? 16 : 32) :
				(internalformat == GL_RGB8 || internalformat == GL_RGBA8 || internalformat == GL_SRGB8 || internalformat == GL_SRGB8_ALPHA8) ? 8 :
				(internalformat == GL_RGB16 || internalformat == GL_RGBA16 || internalformat == GL_RGB16F || internalformat == GL_RGBA16F) ? 16 :
				(internalformat == GL_RGB32F || internalformat == GL_RGBA32F) ? 32 :
				internalformat == GL_RGB10_A2 ? 10 : internalformat == GL_R11F_G11F_B10F ? 11 : 0;
			return true;
		case GL_RENDERBUFFER_GREEN_SIZE:
			if (internalformat == GL_R8 || internalformat == GL_R16 || internalformat == GL_R16F || internalformat == GL_R32F) value = 0;
			else if (internalformat == GL_RG8 || internalformat == GL_RG16 || internalformat == GL_RG16F || internalformat == GL_RG32F) value = (internalformat == GL_RG8 ? 8 : internalformat == GL_RG16 ? 16 : internalformat == GL_RG16F ? 16 : 32);
			else value = renderbufferFormatBits(internalformat, GL_RENDERBUFFER_RED_SIZE, value) ? value : 0;
			return true;
		case GL_RENDERBUFFER_BLUE_SIZE:
			value = (internalformat == GL_R8 || internalformat == GL_R16 || internalformat == GL_R16F || internalformat == GL_R32F ||
				internalformat == GL_RG8 || internalformat == GL_RG16 || internalformat == GL_RG16F || internalformat == GL_RG32F) ? 0 :
				(internalformat == GL_RGB8 || internalformat == GL_RGBA8 || internalformat == GL_SRGB8 || internalformat == GL_SRGB8_ALPHA8) ? 8 :
				(internalformat == GL_RGB16 || internalformat == GL_RGBA16 || internalformat == GL_RGB16F || internalformat == GL_RGBA16F) ? 16 :
				(internalformat == GL_RGB32F || internalformat == GL_RGBA32F) ? 32 : internalformat == GL_RGB10_A2 ? 10 : internalformat == GL_R11F_G11F_B10F ? 10 : 0;
			return true;
		case GL_RENDERBUFFER_ALPHA_SIZE:
			value = (internalformat == GL_RGBA8 || internalformat == GL_SRGB8_ALPHA8) ? 8 :
				(internalformat == GL_RGBA16 || internalformat == GL_RGBA16F) ? 16 :
				internalformat == GL_RGBA32F ? 32 : internalformat == GL_RGB10_A2 ? 2 : 0;
			return true;
		case GL_RENDERBUFFER_DEPTH_SIZE:
			value = internalformat == GL_DEPTH_COMPONENT16 ? 16 :
				(internalformat == GL_DEPTH_COMPONENT24 || internalformat == GL_DEPTH24_STENCIL8 ||
				 internalformat == GL_DEPTH_COMPONENT32 || internalformat == GL_DEPTH_COMPONENT32F || internalformat == GL_DEPTH32F_STENCIL8) ? 32 : 0;
			return true;
		case GL_RENDERBUFFER_STENCIL_SIZE:
			value = (internalformat == GL_STENCIL_INDEX8 || internalformat == GL_DEPTH24_STENCIL8 || internalformat == GL_DEPTH32F_STENCIL8) ? 8 : 0;
			return true;
		default:
			return false;
	}
}

static void renderbufferStorage(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) {
	if (!validRenderbufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (samples < 0 || width <= 0 || height <= 0 || width > GL4METAL_MAX_TEXTURE_SIZE || height > GL4METAL_MAX_TEXTURE_SIZE) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (ctx.bound_renderbuffer == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const MTLPixelFormat pixelFormat = metalPixelFormat(internalformat);
	if (pixelFormat == MTLPixelFormatInvalid) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("renderbufferStorage: unsupported internal format 0x%X", internalformat);
		return;
	}
	const NSUInteger sampleCount = std::max<NSUInteger>(1, (NSUInteger)samples);
	if (![ctx.device supportsTextureSampleCount:sampleCount]) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("renderbufferStorage: sample count %lu is not supported by this iOS Metal device", (unsigned long)sampleCount);
		return;
	}
	MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
		width:(NSUInteger)width height:(NSUInteger)height mipmapped:NO];
	descriptor.textureType = sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
	descriptor.sampleCount = sampleCount;
	descriptor.mipmapLevelCount = 1;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	// Color/depth renderbuffers can also be sampled by the GL blit path; stencil-only is render-target-only.
	if (sampleCount == 1 && pixelFormat != MTLPixelFormatStencil8) {
		descriptor.usage = descriptor.usage | MTLTextureUsageShaderRead;
	}
	id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
	if (!newTexture) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("renderbufferStorage: Metal allocation failed for %dx%d format=0x%X samples=%lu", width, height, internalformat, (unsigned long)sampleCount);
		return;
	}
	endRenderEncoder();
	// Already encoded command buffers retain the previous MTLTexture. Replacing the
	// GL renderbuffer storage does not require stalling the whole GPU queue.
	RenderbufferObject &object = ctx.renderbuffer_objects[ctx.bound_renderbuffer];
	object.texture = newTexture;
	object.internalformat = internalformat;
	object.width = width;
	object.height = height;
	object.samples = samples;
	object.initialized = true;
}

void glGenRenderbuffers(GLsizei count, GLuint *renderbuffers) {
	if (count < 0 || (count > 0 && !renderbuffers)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei i = 0; i < count; ++i) {
		GLuint name = ++ctx.next_renderbuffer_id;
		ctx.renderbuffer_objects.emplace(name, RenderbufferObject());
		renderbuffers[i] = name;
	}
}

void glDeleteRenderbuffers(GLsizei count, const GLuint *renderbuffers) {
	if (count < 0 || (count > 0 && !renderbuffers)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	bool detached = false;
	for (GLsizei i = 0; i < count; ++i) {
		const GLuint name = renderbuffers[i];
		if (name == 0) continue;
		if (ctx.bound_renderbuffer == name) ctx.bound_renderbuffer = 0;
		for (auto &entry : ctx.fbo_objects) {
			auto clear = [&](FBOAttachment &attachment) {
				if (attachment.renderbuffer == name) {
					attachment.renderbuffer = 0;
					detached = true;
				}
			};
			for (FBOAttachment &attachment : entry.second.color) clear(attachment);
			clear(entry.second.depth);
			clear(entry.second.stencil);
		}
		ctx.renderbuffer_objects.erase(name);
	}
	if (detached) endRenderEncoder();
}

GLboolean glIsRenderbuffer(GLuint renderbuffer) {
	auto it = ctx.renderbuffer_objects.find(renderbuffer);
	return renderbuffer != 0 && it != ctx.renderbuffer_objects.end() && it->second.initialized;
}

void glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
	if (!validRenderbufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (renderbuffer != 0) {
		RenderbufferObject &object = ctx.renderbuffer_objects[renderbuffer];
		object.initialized = true;
	}
	ctx.bound_renderbuffer = renderbuffer;
}

void glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
	renderbufferStorage(target, 0, internalformat, width, height);
}

void glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) {
	renderbufferStorage(target, samples, internalformat, width, height);
}

void glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint *params) {
	if (!params) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!validRenderbufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (ctx.bound_renderbuffer == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	auto it = ctx.renderbuffer_objects.find(ctx.bound_renderbuffer);
	if (it == ctx.renderbuffer_objects.end() || !it->second.initialized) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const RenderbufferObject &object = it->second;
	switch (pname) {
		case GL_RENDERBUFFER_WIDTH: *params = object.width; return;
		case GL_RENDERBUFFER_HEIGHT: *params = object.height; return;
		case GL_RENDERBUFFER_INTERNAL_FORMAT: *params = (GLint)object.internalformat; return;
		case GL_RENDERBUFFER_SAMPLES: *params = object.samples; return;
		case GL_RENDERBUFFER_RED_SIZE:
		case GL_RENDERBUFFER_GREEN_SIZE:
		case GL_RENDERBUFFER_BLUE_SIZE:
		case GL_RENDERBUFFER_ALPHA_SIZE:
		case GL_RENDERBUFFER_DEPTH_SIZE:
		case GL_RENDERBUFFER_STENCIL_SIZE:
			if (renderbufferFormatBits(object.internalformat, pname, *params)) return;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
}
