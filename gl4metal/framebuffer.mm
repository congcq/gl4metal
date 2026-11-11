//
// framebuffer.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//

#import "framebuffer.h"

#include <mutex>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>

// MARK: - Render passes

static bool drawBufferTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_DRAW_BUFFERS");
		return value && value[0] == '1';
	}();
	return enabled;
}

// Metal records work into encoders, GL into global state: a render pass is opened by glClear (or the first draw)
// for the current draw framebuffer and closed when the target changes or the frame is presented.

void endRenderEncoder(void) {
	if (ctx.currentEncoder) {
		if (ctx.pass_uses_default_framebuffer) {
			const MTLStoreAction depthStencilStore = ctx.discard_default_depth_stencil_on_end
				? MTLStoreActionDontCare : MTLStoreActionStore;
			// Default-FB descriptors use Unknown so these encoder methods can choose the
			// final action. Mid-frame switches must preserve data; present can discard it.
			[ctx.currentEncoder setDepthStoreAction:depthStencilStore];
			[ctx.currentEncoder setStencilStoreAction:depthStencilStore];
		}
		[ctx.currentEncoder endEncoding];
		ctx.currentEncoder = nil;
	}
	ctx.fragcoord_params_bound = false;
	ctx.discard_default_depth_stencil_on_end = false;
	ctx.pass_uses_default_framebuffer = false;
}

static bool isDepthFormat(MTLPixelFormat format) {
	return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float ||
		format == MTLPixelFormatDepth32Float_Stencil8;
}

static bool isStencilFormat(MTLPixelFormat format) {
	return format == MTLPixelFormatStencil8 || format == MTLPixelFormatDepth32Float_Stencil8;
}

static bool isDepthStencilFormatForFBO(MTLPixelFormat format) {
	return isDepthFormat(format) || isStencilFormat(format);
}

static bool hasAttachment(const FBOAttachment &a) {
	return a.texture != 0 || a.renderbuffer != 0;
}

static id<MTLTexture> attachmentTexture(const FBOAttachment &a) {
	if (!hasAttachment(a) || a.level < 0 || (a.texture != 0 && a.renderbuffer != 0)) return nil;
	id<MTLTexture> texture = nil;
	if (a.renderbuffer != 0) {
		auto object = ctx.renderbuffer_objects.find(a.renderbuffer);
		if (object == ctx.renderbuffer_objects.end() || !object->second.initialized || !object->second.texture || a.level != 0) return nil;
		texture = object->second.texture;
	} else {
		auto object = ctx.texture_objects.find(a.texture);
		if (object == ctx.texture_objects.end() || !object->second.texture) return nil;
		texture = object->second.texture;
		if ((NSUInteger)a.level >= texture.mipmapLevelCount) return nil;
	}
	if ((texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DMultisample) ||
		(texture.usage & MTLTextureUsageRenderTarget) == 0) return nil;
	if (texture.textureType == MTLTextureType2DMultisample && a.level != 0) return nil;
	return texture;
}

static bool checkFBOAttachment(const FBOAttachment &a, int role, NSUInteger &width, NSUInteger &height,
	NSUInteger &sampleCount, bool &haveExtent, bool &haveSamples, bool &sampleMismatch) {
	// role: 0=color, 1=depth, 2=stencil. An unattached point is valid.
	if (!hasAttachment(a)) return true;
	id<MTLTexture> texture = attachmentTexture(a);
	if (!texture) return false;
	const MTLPixelFormat format = texture.pixelFormat;
	if ((role == 0 && isDepthStencilFormatForFBO(format)) ||
		(role == 1 && !isDepthFormat(format)) ||
		(role == 2 && !isStencilFormat(format))) return false;
	const NSUInteger levelWidth = std::max<NSUInteger>(1, texture.width >> (NSUInteger)a.level);
	const NSUInteger levelHeight = std::max<NSUInteger>(1, texture.height >> (NSUInteger)a.level);
	if (!haveExtent) {
		width = levelWidth;
		height = levelHeight;
		haveExtent = true;
	} else if (width != levelWidth || height != levelHeight) {
		return false;
	}
	if (!haveSamples) {
		sampleCount = texture.sampleCount;
		haveSamples = true;
	} else if (sampleCount != texture.sampleCount) {
		sampleMismatch = true;
		return false;
	}
	return true;
}

static id<MTLTexture> defaultDepthTexture(NSUInteger width, NSUInteger height) {
	if (ctx.depthTexture && ctx.depthTexture.width == width && ctx.depthTexture.height == height) {
		return ctx.depthTexture;
	}
	// iOS has no 24-bit depth formats (Depth24Unorm_Stencil8 is macOS only), so use 32-bit float depth + 8-bit stencil.
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
																				  width:width
																				 height:height
																			  mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget;
	d.storageMode = MTLStorageModePrivate;
	ctx.default_depth_has_contents = false;
	ctx.default_stencil_has_contents = false;
	ctx.depthTexture = [ctx.device newTextureWithDescriptor:d];
	if (!ctx.depthTexture || (ctx.depthTexture.usage & MTLTextureUsageRenderTarget) == 0) {
		GL4METAL_LOG("beginRenderPass: failed to allocate default depth/stencil texture (%lux%lu)", (unsigned long)width, (unsigned long)height);
		ctx.depthTexture = nil;
	}
	return ctx.depthTexture;
}

void applyViewportAndScissor(void) {
	if (!ctx.currentEncoder || ctx.render_target_width == 0 || ctx.render_target_height == 0) {
		return;
	}
	double W = (double)ctx.render_target_width;
	double H = (double)ctx.render_target_height;

	// GL's origin is bottom-left. Metal rows start at the top, so the default framebuffer (drawable) mirrors the rows.
	// Textures are drawn with Y flipped in the vertex shader instead, so their rows already follow GL.
	bool mirror = !ctx.pass_flipped;

	MTLViewport vp;
	if (ctx.viewport.w > 0 && ctx.viewport.h > 0) {
		vp.originX = ctx.viewport.x;
		vp.originY = mirror ? H - (double)(ctx.viewport.y + ctx.viewport.h) : (double)ctx.viewport.y;
		vp.width = ctx.viewport.w;
		vp.height = ctx.viewport.h;
	} else {
		vp.originX = 0; vp.originY = 0; vp.width = W; vp.height = H; // glViewport not called yet
	}
	vp.znear = ctx.depth_range_near;
	vp.zfar = ctx.depth_range_far;
	[ctx.currentEncoder setViewport:vp];

	// Metal requires the scissor rect to lie inside the render target.
	NSInteger x0 = 0, y0 = 0, x1 = (NSInteger)W, y1 = (NSInteger)H;
	if (ctx.scissor_test_enabled) {
		NSInteger sy0 = ctx.scissor.y, sy1 = (NSInteger)ctx.scissor.y + ctx.scissor.h;
		const NSInteger maxX = (NSInteger)W, maxY = (NSInteger)H;
		x0 = std::min<NSInteger>(std::max<NSInteger>(ctx.scissor.x, 0), maxX);
		x1 = std::min<NSInteger>(std::max<NSInteger>((NSInteger)ctx.scissor.x + ctx.scissor.w, 0), maxX);
		const NSInteger mappedY0 = mirror ? maxY - sy1 : sy0;
		const NSInteger mappedY1 = mirror ? maxY - sy0 : sy1;
		y0 = std::min<NSInteger>(std::max<NSInteger>(mappedY0, 0), maxY);
		y1 = std::min<NSInteger>(std::max<NSInteger>(mappedY1, 0), maxY);
	}
	MTLScissorRect sr;
	sr.x = (NSUInteger)x0;
	sr.y = (NSUInteger)y0;
	sr.width = (NSUInteger)std::max<NSInteger>(x1 - x0, 0);
	sr.height = (NSUInteger)std::max<NSInteger>(y1 - y0, 0);
	[ctx.currentEncoder setScissorRect:sr];
}

id<MTLRenderCommandEncoder> beginRenderPass(GLbitfield clearMask, bool preservePassSerial) {
	endRenderEncoder();
	if (!preservePassSerial) {
		++ctx.render_pass_serial;
		if (ctx.render_pass_serial == 0) ++ctx.render_pass_serial; // reserve zero for never-opened state
	}

	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	MTLLoadAction colorLoad = (clearMask & GL_COLOR_BUFFER_BIT) ? MTLLoadActionClear : MTLLoadActionLoad;
	MTLLoadAction depthLoad = (clearMask & GL_DEPTH_BUFFER_BIT) ? MTLLoadActionClear : MTLLoadActionLoad;
	MTLLoadAction stencilLoad = (clearMask & GL_STENCIL_BUFFER_BIT) ? MTLLoadActionClear : MTLLoadActionLoad;
	NSUInteger rtWidth = 0, rtHeight = 0;

	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		ctx.pass_color_format[i] = MTLPixelFormatInvalid;
	}
	ctx.pass_depth_format = MTLPixelFormatInvalid;
	ctx.pass_stencil_format = MTLPixelFormatInvalid;
	ctx.pass_sample_count = 1;
	ctx.pass_flipped = (ctx.draw_fbo != 0);

	if (ctx.draw_fbo == 0) {
		// default framebuffer = the CAMetalLayer drawable (+ our own depth/stencil)
		if (!ctx.metalLayer) {
			static bool warned = false;
			if (!warned) {
				warned = true;
				GL4METAL_LOG("beginRenderPass: no CAMetalLayer, glMakeCurrent has not been called with one (further reports suppressed)");
			}
			return nil;
		}
		if (!ctx.currentDrawable) {
			ctx.currentDrawable = [ctx.metalLayer nextDrawable];
		}
		if (!ctx.currentDrawable) {
			return nil; // e.g. app is in the background
		}
		id<MTLTexture> color = ctx.currentDrawable.texture;
		if (!color || (color.usage & MTLTextureUsageRenderTarget) == 0) {
			GL4METAL_LOG("beginRenderPass: CAMetalLayer returned a drawable without a valid render-target texture");
			return nil;
		}
		rtWidth = color.width;
		rtHeight = color.height;

		// Newly acquired drawables have undefined previous contents. Don't load them
		// on the first pass unless GL has already rendered to this drawable this frame.
		if (!(clearMask & GL_COLOR_BUFFER_BIT) && !ctx.default_color_has_contents) {
			colorLoad = MTLLoadActionDontCare;
		}
		MTLRenderPassColorAttachmentDescriptor *ca = rp.colorAttachments[0];
		ca.texture = color;
		ca.loadAction = colorLoad;
		ca.storeAction = MTLStoreActionStore;
		ca.clearColor = ctx.clear_color;
		ctx.pass_color_format[0] = color.pixelFormat;

		id<MTLTexture> depth = defaultDepthTexture(rtWidth, rtHeight);
		if (!depth) return nil;
		// The drawable's depth/stencil is not preserved across presented frames. Avoid
		// loading undefined contents on the first pass that does not explicitly clear.
		if (!(clearMask & GL_DEPTH_BUFFER_BIT) && !ctx.default_depth_has_contents) {
			depthLoad = MTLLoadActionDontCare;
		}
		if (!(clearMask & GL_STENCIL_BUFFER_BIT) && !ctx.default_stencil_has_contents) {
			stencilLoad = MTLLoadActionDontCare;
		}
		rp.depthAttachment.texture = depth;
		rp.depthAttachment.loadAction = depthLoad;
		// Store is deferred until endRenderEncoder(): Store for an intermediate pass,
		// DontCare after present. Metal requires Unknown for the encoder setter APIs.
		rp.depthAttachment.storeAction = MTLStoreActionUnknown;
		rp.depthAttachment.clearDepth = ctx.clear_depth;
		rp.stencilAttachment.texture = depth;
		rp.stencilAttachment.loadAction = stencilLoad;
		rp.stencilAttachment.storeAction = MTLStoreActionUnknown;
		rp.stencilAttachment.clearStencil = (uint32_t)ctx.clear_stencil;
		ctx.pass_depth_format = depth.pixelFormat;
		ctx.pass_stencil_format = depth.pixelFormat;
	} else {
		const GLenum fboStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
		if (fboStatus != GL_FRAMEBUFFER_COMPLETE) {
			gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
			GL4METAL_LOG("beginRenderPass: draw framebuffer %u is incomplete (status=0x%X); pass not encoded",
				ctx.draw_fbo, fboStatus);
			return nil;
		}
		auto it = ctx.fbo_objects.find(ctx.draw_fbo);
		if (it == ctx.fbo_objects.end()) {
			return nil;
		}
		const FBO &fbo = it->second;
		bool any = false;

		for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
			GLenum buf = fbo.draw_buffers[i]; // draw buffer i writes to GL_COLOR_ATTACHMENT(buf - GL_COLOR_ATTACHMENT0)
			if (buf < GL_COLOR_ATTACHMENT0 || buf >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
				continue;
			}
			const FBOAttachment &a = fbo.color[buf - GL_COLOR_ATTACHMENT0];
			id<MTLTexture> tex = attachmentTexture(a);
			if (!tex) {
				continue;
			}
			MTLRenderPassColorAttachmentDescriptor *ca = rp.colorAttachments[i];
			ca.texture = tex;
			ca.level = (NSUInteger)a.level;
			ca.loadAction = colorLoad;
			ca.storeAction = MTLStoreActionStore;
			ca.clearColor = ctx.clear_color;
			ctx.pass_color_format[i] = tex.pixelFormat;
			if (!any) {
				rtWidth = std::max<NSUInteger>(tex.width >> a.level, 1);
				rtHeight = std::max<NSUInteger>(tex.height >> a.level, 1);
				ctx.pass_sample_count = tex.sampleCount;
			}
			any = true;
		}

		id<MTLTexture> depth = attachmentTexture(fbo.depth);
		if (depth) {
			rp.depthAttachment.texture = depth;
			rp.depthAttachment.level = (NSUInteger)fbo.depth.level;
			rp.depthAttachment.loadAction = depthLoad;
			rp.depthAttachment.storeAction = MTLStoreActionStore;
			rp.depthAttachment.clearDepth = ctx.clear_depth;
			ctx.pass_depth_format = depth.pixelFormat;
			if (!any) {
				rtWidth = std::max<NSUInteger>(depth.width >> fbo.depth.level, 1);
				rtHeight = std::max<NSUInteger>(depth.height >> fbo.depth.level, 1);
				ctx.pass_sample_count = depth.sampleCount;
			}
			any = true;
		}

		id<MTLTexture> stencil = attachmentTexture(fbo.stencil);
		if (stencil && (stencil.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 || stencil.pixelFormat == MTLPixelFormatStencil8)) {
			rp.stencilAttachment.texture = stencil;
			rp.stencilAttachment.level = (NSUInteger)fbo.stencil.level;
			rp.stencilAttachment.loadAction = stencilLoad;
			rp.stencilAttachment.storeAction = MTLStoreActionStore;
			rp.stencilAttachment.clearStencil = (uint32_t)ctx.clear_stencil;
			ctx.pass_stencil_format = stencil.pixelFormat;
			any = true;
		}

		if (!any) {
			GL4METAL_LOG("beginRenderPass: framebuffer %u has no attachment with storage", ctx.draw_fbo);
			return nil;
		}
	}

	if (!ctx.commandQueue) {
		GL4METAL_LOG("beginRenderPass: Metal command queue is unavailable");
		return nil;
	}
	if (!ctx.currentCommandBuffer) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	if (!ctx.currentCommandBuffer) return nil;
	ctx.currentEncoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
	if (!ctx.currentEncoder) {
		GL4METAL_LOG("beginRenderPass: Metal could not create render encoder (FBO=%u, size=%lux%lu)",
			ctx.draw_fbo, (unsigned long)rtWidth, (unsigned long)rtHeight);
		return nil;
	}
	ctx.render_target_width = rtWidth;
	ctx.render_target_height = rtHeight;
	ctx.pass_uses_default_framebuffer = (ctx.draw_fbo == 0);
	if (ctx.pass_uses_default_framebuffer) {
		if (clearMask & GL_COLOR_BUFFER_BIT) ctx.default_color_has_contents = true;
		if (clearMask & GL_DEPTH_BUFFER_BIT) ctx.default_depth_has_contents = true;
		if (clearMask & GL_STENCIL_BUFFER_BIT) ctx.default_stencil_has_contents = true;
	}
	applyViewportAndScissor();
	return ctx.currentEncoder;
}

// MARK: - Framebuffer objects

void glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint buf_id = ++ctx.next_fbo_id;
		ctx.fbo_objects[buf_id] = FBO();
		framebuffers[i] = buf_id;
	}
}

void glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint buf_id = framebuffers[i];
		if (buf_id == 0) {
			continue;
		}
		if (ctx.draw_fbo == buf_id) {
			endRenderEncoder();
			ctx.draw_fbo = 0; // GL falls back to the default framebuffer
		}
		if (ctx.read_fbo == buf_id) {
			ctx.read_fbo = 0;
		}
		ctx.fbo_objects.erase(buf_id);
	}
}

GLboolean glIsFramebuffer(GLuint framebuffer) {
	return framebuffer != 0 && ctx.fbo_objects.find(framebuffer) != ctx.fbo_objects.end();
}

void glBindFramebuffer(GLenum target, GLuint framebuffer) {
	if (framebuffer != 0 && ctx.fbo_objects.find(framebuffer) == ctx.fbo_objects.end()) {
		ctx.fbo_objects[framebuffer] = FBO(); // GL allows binding a name that was never generated
	}
	switch (target) {
		case GL_FRAMEBUFFER:
			if (ctx.draw_fbo != framebuffer) {
				endRenderEncoder();
			}
			ctx.draw_fbo = framebuffer;
			ctx.read_fbo = framebuffer;
			break;
		case GL_DRAW_FRAMEBUFFER:
			if (ctx.draw_fbo != framebuffer) {
				endRenderEncoder();
			}
			ctx.draw_fbo = framebuffer;
			break;
		case GL_READ_FRAMEBUFFER:
			ctx.read_fbo = framebuffer;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glBindFramebuffer: invalid target 0x%X", target);
			break;
	}
}

GLenum glCheckFramebufferStatus(GLenum target) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glCheckFramebufferStatus: invalid target 0x%X", target);
		return 0;
	}
	GLuint buf_id = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (buf_id == 0) {
		return GL_FRAMEBUFFER_COMPLETE; // the layer drawable is always complete
	}
	auto it = ctx.fbo_objects.find(buf_id);
	if (it == ctx.fbo_objects.end()) return GL_FRAMEBUFFER_UNDEFINED;
	const FBO &fbo = it->second;
	bool any = false;
	bool haveExtent = false;
	bool haveSamples = false;
	bool sampleMismatch = false;
	NSUInteger width = 0, height = 0, sampleCount = 1;
	auto check = [&](const FBOAttachment &a, int role) -> bool {
		if (!hasAttachment(a)) return true;
		any = true;
		return checkFBOAttachment(a, role, width, height, sampleCount, haveExtent, haveSamples, sampleMismatch);
	};
	if (!check(fbo.depth, 1) || !check(fbo.stencil, 2)) {
		return sampleMismatch ? GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE : GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
	}
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		if (!check(fbo.color[i], 0)) {
			return sampleMismatch ? GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE : GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
		}
	}
	if (!any) return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
	// Metal cannot render/read into a missing attachment selected by GL's draw/read buffer state.
	if (target != GL_READ_FRAMEBUFFER) {
		for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
			const GLenum drawBuffer = fbo.draw_buffers[i];
			if (drawBuffer == GL_NONE) continue;
			if (drawBuffer < GL_COLOR_ATTACHMENT0 || drawBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS ||
				!hasAttachment(fbo.color[drawBuffer - GL_COLOR_ATTACHMENT0])) {
				return GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER;
			}
		}
	}
	if (target == GL_READ_FRAMEBUFFER && fbo.read_buffer != GL_NONE) {
		if (fbo.read_buffer < GL_COLOR_ATTACHMENT0 || fbo.read_buffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS ||
			!hasAttachment(fbo.color[fbo.read_buffer - GL_COLOR_ATTACHMENT0])) {
			return GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER;
		}
	}
	return GL_FRAMEBUFFER_COMPLETE;
}

// MARK: - Framebuffer attachments

void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
	glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
}

void glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glFramebufferTexture1D: Unimplemented");
}

void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glFramebufferTexture2D: invalid target 0x%X", target);
		return;
	}
	if (level < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glFramebufferTexture2D: negative mip level %d", level);
		return;
	}
	GLuint buf_id = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (buf_id == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glFramebufferTexture2D: the default framebuffer is bound (GL_INVALID_OPERATION)");
		return;
	}
	if (texture != 0 && textarget != GL_TEXTURE_2D) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glFramebufferTexture2D: unsupported textarget 0x%X", textarget);
		return;
	}
	if (texture != 0) {
		auto object = ctx.texture_objects.find(texture);
		if (object == ctx.texture_objects.end() || object->second.target != GL_TEXTURE_2D) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glFramebufferTexture2D: texture %u is not an initialized GL_TEXTURE_2D object", texture);
			return;
		}
	}
	FBO &fbo = ctx.fbo_objects[buf_id];
	FBOAttachment a;
	a.texture = texture;
	a.renderbuffer = 0;
	a.level = level;

	if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		fbo.color[attachment - GL_COLOR_ATTACHMENT0] = a;
	} else if (attachment == GL_DEPTH_ATTACHMENT) {
		fbo.depth = a;
	} else if (attachment == GL_STENCIL_ATTACHMENT) {
		fbo.stencil = a;
	} else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		fbo.depth = a;
		fbo.stencil = a;
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glFramebufferTexture2D: invalid attachment 0x%X", attachment);
		return;
	}
	if (buf_id == ctx.draw_fbo) {
		endRenderEncoder(); // the open render pass was built from the old attachments
	}
}

void glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glFramebufferTexture3D: Unimplemented");
}

void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glFramebufferTextureLayer: Unimplemented");
}

void glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (renderbuffertarget != GL_RENDERBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	GLuint framebufferName = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (framebufferName == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (renderbuffer != 0) {
		auto object = ctx.renderbuffer_objects.find(renderbuffer);
		if (object == ctx.renderbuffer_objects.end() || !object->second.initialized) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
	}
	FBOAttachment newAttachment;
	newAttachment.renderbuffer = renderbuffer;
	FBO &fbo = ctx.fbo_objects[framebufferName];
	if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		fbo.color[attachment - GL_COLOR_ATTACHMENT0] = newAttachment;
	} else if (attachment == GL_DEPTH_ATTACHMENT) {
		fbo.depth = newAttachment;
	} else if (attachment == GL_STENCIL_ATTACHMENT) {
		fbo.stencil = newAttachment;
	} else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		fbo.depth = newAttachment;
		fbo.stencil = newAttachment;
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (framebufferName == ctx.draw_fbo) endRenderEncoder();
}

void glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint *params) {
	if (!params || (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER)) {
		gl4metalSetError(!params ? GL_INVALID_VALUE : GL_INVALID_ENUM);
		return;
	}
	GLuint buf_id = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (buf_id == 0) { // default framebuffer
		*params = (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) ? GL_FRAMEBUFFER_DEFAULT : 0;
		return;
	}
	auto it = ctx.fbo_objects.find(buf_id);
	if (it == ctx.fbo_objects.end()) {
		*params = 0;
		return;
	}
	const FBO &fbo = it->second;
	const FBOAttachment *a = nullptr;
	if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		a = &fbo.color[attachment - GL_COLOR_ATTACHMENT0];
	} else if (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		a = &fbo.depth;
	} else if (attachment == GL_STENCIL_ATTACHMENT) {
		a = &fbo.stencil;
	}
	switch (pname) {
		case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
			*params = !a ? GL_NONE : a->texture ? GL_TEXTURE : a->renderbuffer ? GL_RENDERBUFFER : GL_NONE;
			break;
		case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
			*params = !a ? 0 : (GLint)(a->texture ? a->texture : a->renderbuffer);
			break;
		case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
			*params = (a && a->texture) ? a->level : 0;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetFramebufferAttachmentParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

// MARK: - Draw / read buffers

void glDrawBuffers(GLsizei n, const GLenum *bufs) {
	if (n < 0 || n > GL4METAL_MAX_COLOR_ATTACHMENTS || (n > 0 && !bufs)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDrawBuffers: invalid count %d or null buffer list", n);
		return;
	}
	for (GLsizei i = 0; i < n; i++) {
		if (bufs[i] != GL_NONE && (bufs[i] < GL_COLOR_ATTACHMENT0 || bufs[i] >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS)) {
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glDrawBuffers: invalid draw buffer enum 0x%X", bufs[i]);
			return;
		}
	}
	if (ctx.draw_fbo == 0) {
		return; // the default framebuffer has one colour buffer
	}
	auto it = ctx.fbo_objects.find(ctx.draw_fbo);
	if (it == ctx.fbo_objects.end()) {
		return;
	}
	// Draw buffer i writes to the colour attachment named by bufs[i] (GL_NONE: not written).
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		it->second.draw_buffers[i] = (i < n) ? bufs[i] : GL_NONE;
	}
	// Shaderpacks encode target routing in comments such as DRAWBUFFERS:41.
	// Trace the actual GL state and the texture currently attached at each target.
	// This is deduplicated by FBO + complete draw-buffer array to avoid per-frame spam.
	if (drawBufferTraceEnabled()) {
		std::string stateKey = std::to_string(ctx.draw_fbo);
		std::string stateList;
		for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; ++i) {
			const GLenum drawBuffer = it->second.draw_buffers[i];
			stateKey += ":" + std::to_string(drawBuffer);
			if (i >= n) continue;
			if (!stateList.empty()) stateList += ", ";
			if (drawBuffer == GL_NONE) {
				stateList += "GL_NONE";
			} else {
				const GLuint textureName = it->second.color[drawBuffer - GL_COLOR_ATTACHMENT0].texture;
				char enumText[16];
				std::snprintf(enumText, sizeof(enumText), "0x%04X", (unsigned int)drawBuffer);
				stateList += enumText;
				stateList += "(texture=" + std::to_string(textureName) + ")";
			}
		}
		static std::mutex traceMutex;
		static std::unordered_set<std::string> reportedStates;
		bool firstReport = false;
		{
			std::lock_guard<std::mutex> lock(traceMutex);
			firstReport = reportedStates.insert(stateKey).second;
		}
		if (firstReport) {
			GL4METAL_LOG("glDrawBuffers: framebuffer=%u count=%d selected=[%s]", ctx.draw_fbo, n, stateList.c_str());
		}
	}
	endRenderEncoder(); // the open pass was built from the old draw buffers
}

void glDrawBuffer(GLenum buf) {
	glDrawBuffers(1, &buf);
}

void glReadBuffer(GLenum src) {
	if (src != GL_NONE && (src < GL_COLOR_ATTACHMENT0 || src >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glReadBuffer: invalid source 0x%X", src);
		return;
	}
	if (ctx.read_fbo == 0) {
		return;
	}
	auto it = ctx.fbo_objects.find(ctx.read_fbo);
	if (it != ctx.fbo_objects.end()) {
		it->second.read_buffer = src;
	}
}

// MARK: - Clear

static inline double clamp01(double v) {
	return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

void glClear(GLbitfield mask) {
	mask &= (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	if (mask == 0) {
		return;
	}
	if (ctx.scissor_test_enabled) {
		GL4METAL_LOG("glClear: scissor test is ignored (the clear is a render pass load action)");
	}
	beginRenderPass(mask); // a pass whose load action is "clear" performs the clear even without draws
}

void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
	ctx.clear_color = MTLClearColorMake(clamp01(red), clamp01(green), clamp01(blue), clamp01(alpha));
}

void glClearDepth(GLdouble depth) {
	ctx.clear_depth = clamp01(depth);
}

void glClearStencil(GLint s) {
	ctx.clear_stencil = s;
}

void glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint *value) {
	if (buffer == GL_STENCIL) {
		clearDepthStencil(GL_STENCIL_BUFFER_BIT, (GLfloat)ctx.clear_depth, value[0]);
	} else {
		GL4METAL_LOG("glClearBufferiv: integer colour buffers are not supported");
	}
}

void glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value) {
	GL4METAL_LOG("glClearBufferuiv: integer colour buffers are not supported");
}

void glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value) {
	if (buffer == GL_COLOR) {
		clearOneColorBuffer(drawbuffer, value);
	} else if (buffer == GL_DEPTH) {
		clearDepthStencil(GL_DEPTH_BUFFER_BIT, value[0], ctx.clear_stencil);
	} else {
		GL4METAL_LOG("glClearBufferfv: unsupported buffer 0x%X", buffer);
	}
}

void glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
	clearDepthStencil(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, depth, stencil);
}

// MARK: - Blit / read back


// glClearBuffer*: one draw buffer of the current framebuffer.
void clearOneColorBuffer(GLint drawbuffer, const GLfloat *rgba) {
	if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glClearBuffer*: draw buffer %d is out of range (GL_INVALID_VALUE)", drawbuffer);
		return;
	}
	MTLClearColor saved = ctx.clear_color;
	ctx.clear_color = MTLClearColorMake(rgba[0], rgba[1], rgba[2], rgba[3]);
	if (ctx.draw_fbo == 0) {
		beginRenderPass(GL_COLOR_BUFFER_BIT);
	} else {
		auto it = ctx.fbo_objects.find(ctx.draw_fbo);
		if (it != ctx.fbo_objects.end()) {
			FBO &fbo = it->second;
			GLenum savedBuffers[GL4METAL_MAX_COLOR_ATTACHMENTS];
			memcpy(savedBuffers, fbo.draw_buffers, sizeof(savedBuffers));
			for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
				fbo.draw_buffers[i] = GL_NONE;
			}
			fbo.draw_buffers[0] = savedBuffers[drawbuffer]; // only this buffer takes part in the clearing pass
			beginRenderPass(GL_COLOR_BUFFER_BIT);
			endRenderEncoder(); // its attachments differ from the ones draws use
			memcpy(fbo.draw_buffers, savedBuffers, sizeof(savedBuffers));
		}
	}
	ctx.clear_color = saved;
}

void clearDepthStencil(GLbitfield mask, GLfloat depth, GLint stencil) {
	double savedDepth = ctx.clear_depth;
	GLint savedStencil = ctx.clear_stencil;
	ctx.clear_depth = clamp01(depth);
	ctx.clear_stencil = stencil;
	beginRenderPass(mask);
	ctx.clear_depth = savedDepth;
	ctx.clear_stencil = savedStencil;
}

// MARK: - Blit helpers

// A framebuffer blit is a small render pass: a full-screen triangle samples the source texture and the fragment shader
// maps every destination pixel back to GL source coordinates, which gives scaling, flipped rectangles and the row order
// difference between the drawable (top-first) and textures (bottom-first) in one place.
struct BlitParams {
	float src_rect[4];    // GL coordinates (x0, y0, x1, y1)
	float dst_rect[4];
	float src_size[2];
	float dst_size[2];
	uint32_t src_default; // 1: framebuffer is the drawable, its rows start at the top
	uint32_t dst_default;
	uint32_t pad[2];
};

static NSString *blitSource() {
	return [NSString stringWithUTF8String:R"MSL(
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
	float4 src_rect;
	float4 dst_rect;
	float2 src_size;
	float2 dst_size;
	uint src_default;
	uint dst_default;
};

struct VOut { float4 position [[position]]; };

vertex VOut blit_vs(uint vid [[vertex_id]]) {
	float2 p = float2((vid << 1) & 2, vid & 2);
	VOut o;
	o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
	return o;
}

fragment float4 blit_fs(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], sampler smp [[sampler(0)]],
						constant BlitParams &u [[buffer(0)]]) {
	float2 pos = in.position.xy; // centre of the destination pixel, rows counted from the top of the texture
	float dstGLy = u.dst_default ? (u.dst_size.y - pos.y) : pos.y;
	float2 t = (float2(pos.x, dstGLy) - u.dst_rect.xy) / (u.dst_rect.zw - u.dst_rect.xy);
	float2 s = u.src_rect.xy + t * (u.src_rect.zw - u.src_rect.xy);
	float srcRow = u.src_default ? (u.src_size.y - s.y) : s.y;
	return src.sample(smp, float2(s.x, srcRow) / u.src_size);
}
)MSL"];
}

static id<MTLRenderPipelineState> blitPipeline(MTLPixelFormat format) {
	static std::mutex cacheMutex;
	std::lock_guard<std::mutex> lock(cacheMutex);
	static id<MTLLibrary> library = nil;
	static std::unordered_map<NSUInteger, id<MTLRenderPipelineState>> pipelines;

	auto it = pipelines.find((NSUInteger)format);
	if (it != pipelines.end()) {
		return it->second;
	}
	NSError *error = nil;
	if (!library) {
		library = [ctx.device newLibraryWithSource:blitSource() options:nil error:&error];
		if (!library) {
			GL4METAL_LOG("glBlitFramebuffer: blit shader failed to compile: %s",
				error.localizedDescription.UTF8String ?: "unknown Metal error");
			return nil;
		}
	}
	MTLRenderPipelineDescriptor *d = [[MTLRenderPipelineDescriptor alloc] init];
	d.vertexFunction = [library newFunctionWithName:@"blit_vs"];
	d.fragmentFunction = [library newFunctionWithName:@"blit_fs"];
	if (!d.vertexFunction || !d.fragmentFunction) {
		GL4METAL_LOG("glBlitFramebuffer: failed to resolve blit shader entry points");
		return nil;
	}
	d.colorAttachments[0].pixelFormat = format;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:d error:&error];
	if (!pipeline) {
		GL4METAL_LOG("glBlitFramebuffer: blit pipeline for pixel format %d failed: %s", (int)format,
			error.localizedDescription.UTF8String ?: "unknown Metal error");
		return nil; // Do not permanently cache a transient nil allocation result.
	}
	pipelines[(NSUInteger)format] = pipeline;
	return pipeline;
}

static bool ensureDefaultFramebufferDepthTexture() {
	if (!ctx.metalLayer) return false;
	if (!ctx.currentDrawable) ctx.currentDrawable = [ctx.metalLayer nextDrawable];
	if (!ctx.currentDrawable || !ctx.currentDrawable.texture) return false;
	return defaultDepthTexture(ctx.currentDrawable.texture.width, ctx.currentDrawable.texture.height) != nil;
}

static id<MTLTexture> framebufferDepthStencilTexture(GLuint framebuffer, bool depth) {
	if (framebuffer == 0) {
		if (!ensureDefaultFramebufferDepthTexture()) return nil;
		return ctx.depthTexture;
	}
	auto framebufferIt = ctx.fbo_objects.find(framebuffer);
	if (framebufferIt == ctx.fbo_objects.end()) return nil;
	return attachmentTexture(depth ? framebufferIt->second.depth : framebufferIt->second.stencil);
}

static NSUInteger blitMetalY(GLuint framebuffer, id<MTLTexture> texture, NSUInteger level, GLint glY, NSUInteger height) {
	const NSUInteger textureHeight = std::max<NSUInteger>(1, texture.height >> level);
	if (framebuffer == 0) return textureHeight - ((NSUInteger)glY + height);
	return (NSUInteger)glY;
}

static bool encodeDepthStencilPlaneCopy(id<MTLTexture> source, NSUInteger sourceLevel,
	id<MTLTexture> destination, NSUInteger destinationLevel,
	NSUInteger sourceX, NSUInteger sourceY, NSUInteger destinationX, NSUInteger destinationY,
	NSUInteger width, NSUInteger height, bool depthPlane) {
	if (!source || !destination || !ctx.device || source.pixelFormat != destination.pixelFormat ||
		source.sampleCount != 1 || destination.sampleCount != 1 ||
		source.textureType != MTLTextureType2D || destination.textureType != MTLTextureType2D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: depth/stencil source and destination must be single-sample 2D textures with identical formats");
		return false;
	}
	const NSUInteger sourceWidth = std::max<NSUInteger>(1, source.width >> sourceLevel);
	const NSUInteger sourceHeight = std::max<NSUInteger>(1, source.height >> sourceLevel);
	const NSUInteger destinationWidth = std::max<NSUInteger>(1, destination.width >> destinationLevel);
	const NSUInteger destinationHeight = std::max<NSUInteger>(1, destination.height >> destinationLevel);
	if (sourceX > sourceWidth || width > sourceWidth - sourceX || sourceY > sourceHeight || height > sourceHeight - sourceY ||
		destinationX > destinationWidth || width > destinationWidth - destinationX ||
		destinationY > destinationHeight || height > destinationHeight - destinationY) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBlitFramebuffer: depth/stencil rectangle is outside the source or destination attachment");
		return false;
	}

	const bool packedDepthStencil = source.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
	if (!packedDepthStencil) {
		if (depthPlane && !isDepthFormat(source.pixelFormat)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
		if (!depthPlane && !isStencilFormat(source.pixelFormat)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
		if (!ctx.currentCommandBuffer) {
			if (!ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
			ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
		}
		id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
		if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
		[copyEncoder copyFromTexture:source sourceSlice:0 sourceLevel:sourceLevel
			sourceOrigin:MTLOriginMake(sourceX, sourceY, 0)
			sourceSize:MTLSizeMake(width, height, 1)
			toTexture:destination destinationSlice:0 destinationLevel:destinationLevel
			destinationOrigin:MTLOriginMake(destinationX, destinationY, 0)];
		[copyEncoder endEncoding];
		return true;
	}

	// A depth-only or stencil-only blit from a packed Depth32Float_Stencil8 texture
	// uses Metal's explicit aspect extraction options through a temporary linear buffer.
	const NSUInteger bytesPerPixel = depthPlane ? 4 : 1;
	const NSUInteger rawRowBytes = width * bytesPerPixel;
	const NSUInteger rowBytes = (rawRowBytes + 255u) & ~((NSUInteger)255u);
	if (height > NSUIntegerMax / rowBytes) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
	const NSUInteger scratchLength = rowBytes * height;
	if (scratchLength > ctx.device.maxBufferLength) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
	id<MTLBuffer> scratch = [ctx.device newBufferWithLength:scratchLength options:MTLResourceStorageModeShared];
	if (!scratch) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
	if (!ctx.currentCommandBuffer) {
		if (!ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
	const MTLBlitOption aspectOption = depthPlane ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;
	[copyEncoder copyFromTexture:source sourceSlice:0 sourceLevel:sourceLevel
		sourceOrigin:MTLOriginMake(sourceX, sourceY, 0)
		sourceSize:MTLSizeMake(width, height, 1)
		toBuffer:scratch destinationOffset:0 destinationBytesPerRow:rowBytes
		destinationBytesPerImage:0 options:aspectOption];
	[copyEncoder copyFromBuffer:scratch sourceOffset:0 sourceBytesPerRow:rowBytes
		sourceBytesPerImage:0 sourceSize:MTLSizeMake(width, height, 1)
		toTexture:destination destinationSlice:0 destinationLevel:destinationLevel
		destinationOrigin:MTLOriginMake(destinationX, destinationY, 0) options:aspectOption];
	[copyEncoder endEncoding];
	return true;
}

static bool performDepthStencilBlit(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
	GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask) {
	if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) == 0) return true;
	auto markDefaultDepthStencilContents = [&](GLbitfield copiedAspects) {
		if (ctx.draw_fbo != 0) return;
		if (copiedAspects & GL_DEPTH_BUFFER_BIT) ctx.default_depth_has_contents = true;
		if (copiedAspects & GL_STENCIL_BUFFER_BIT) ctx.default_stencil_has_contents = true;
	};
	if ((ctx.read_fbo != 0 && glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) ||
		(ctx.draw_fbo != 0 && glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)) {
		gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
		return false;
	}
	if (srcX0 < 0 || srcY0 < 0 || dstX0 < 0 || dstY0 < 0 ||
		srcX1 <= srcX0 || srcY1 <= srcY0 || dstX1 <= dstX0 || dstY1 <= dstY0 ||
		srcX1 - srcX0 != dstX1 - dstX0 || srcY1 - srcY0 != dstY1 - dstY0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: depth/stencil blits require positive, equal-sized source and destination rectangles");
		return false;
	}
	const NSUInteger width = (NSUInteger)(srcX1 - srcX0);
	const NSUInteger height = (NSUInteger)(srcY1 - srcY0);
	id<MTLTexture> sourceDepth = nil, destinationDepth = nil;
	id<MTLTexture> sourceStencil = nil, destinationStencil = nil;
	NSUInteger sourceDepthLevel = 0, destinationDepthLevel = 0;
	NSUInteger sourceStencilLevel = 0, destinationStencilLevel = 0;

	if (mask & GL_DEPTH_BUFFER_BIT) {
		if (ctx.read_fbo != 0) {
			auto sourceFbo = ctx.fbo_objects.find(ctx.read_fbo);
			if (sourceFbo != ctx.fbo_objects.end()) sourceDepthLevel = (NSUInteger)std::max(sourceFbo->second.depth.level, 0);
		}
		if (ctx.draw_fbo != 0) {
			auto destinationFbo = ctx.fbo_objects.find(ctx.draw_fbo);
			if (destinationFbo != ctx.fbo_objects.end()) destinationDepthLevel = (NSUInteger)std::max(destinationFbo->second.depth.level, 0);
		}
		sourceDepth = framebufferDepthStencilTexture(ctx.read_fbo, true);
		destinationDepth = framebufferDepthStencilTexture(ctx.draw_fbo, true);
		if (!sourceDepth || !destinationDepth) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: depth bit requested but the read/draw framebuffer lacks a valid depth attachment");
			return false;
		}
	}
	if (mask & GL_STENCIL_BUFFER_BIT) {
		if (ctx.read_fbo != 0) {
			auto sourceFbo = ctx.fbo_objects.find(ctx.read_fbo);
			if (sourceFbo != ctx.fbo_objects.end()) sourceStencilLevel = (NSUInteger)std::max(sourceFbo->second.stencil.level, 0);
		}
		if (ctx.draw_fbo != 0) {
			auto destinationFbo = ctx.fbo_objects.find(ctx.draw_fbo);
			if (destinationFbo != ctx.fbo_objects.end()) destinationStencilLevel = (NSUInteger)std::max(destinationFbo->second.stencil.level, 0);
		}
		sourceStencil = framebufferDepthStencilTexture(ctx.read_fbo, false);
		destinationStencil = framebufferDepthStencilTexture(ctx.draw_fbo, false);
		if (!sourceStencil || !destinationStencil) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: stencil bit requested but the read/draw framebuffer lacks a valid stencil attachment");
			return false;
		}
	}

	// Copy both aspects in one operation when both framebuffers use the same packed format.
	if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) == (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) &&
		sourceDepth && destinationDepth && sourceStencil == sourceDepth && destinationStencil == destinationDepth &&
		sourceDepth.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && destinationDepth.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
		if (sourceDepth.sampleCount != 1 || destinationDepth.sampleCount != 1 || sourceDepth.textureType != MTLTextureType2D || destinationDepth.textureType != MTLTextureType2D) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
		const NSUInteger sourceWidth = std::max<NSUInteger>(1, sourceDepth.width >> sourceDepthLevel);
		const NSUInteger sourceHeight = std::max<NSUInteger>(1, sourceDepth.height >> sourceDepthLevel);
		const NSUInteger destinationWidth = std::max<NSUInteger>(1, destinationDepth.width >> destinationDepthLevel);
		const NSUInteger destinationHeight = std::max<NSUInteger>(1, destinationDepth.height >> destinationDepthLevel);
		if ((NSUInteger)srcX0 + width > sourceWidth || (NSUInteger)srcY0 + height > sourceHeight ||
			(NSUInteger)dstX0 + width > destinationWidth || (NSUInteger)dstY0 + height > destinationHeight) {
			gl4metalSetError(GL_INVALID_VALUE);
			return false;
		}
		if (sourceDepth == destinationDepth) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
		endRenderEncoder();
		if (!ctx.currentCommandBuffer) {
			if (!ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
			ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
		}
		id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
		if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
		const NSUInteger sourceMetalY = blitMetalY(ctx.read_fbo, sourceDepth, sourceDepthLevel, srcY0, height);
		const NSUInteger destinationMetalY = blitMetalY(ctx.draw_fbo, destinationDepth, destinationDepthLevel, dstY0, height);
		[copyEncoder copyFromTexture:sourceDepth sourceSlice:0 sourceLevel:sourceDepthLevel
			sourceOrigin:MTLOriginMake((NSUInteger)srcX0, sourceMetalY, 0)
			sourceSize:MTLSizeMake(width, height, 1)
			toTexture:destinationDepth destinationSlice:0 destinationLevel:destinationDepthLevel
			destinationOrigin:MTLOriginMake((NSUInteger)dstX0, destinationMetalY, 0)];
		[copyEncoder endEncoding];
		markDefaultDepthStencilContents(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		return true;
	}

	endRenderEncoder();
	if (mask & GL_DEPTH_BUFFER_BIT) {
		if (sourceDepth == destinationDepth) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
		const NSUInteger sourceMetalY = blitMetalY(ctx.read_fbo, sourceDepth, sourceDepthLevel, srcY0, height);
		const NSUInteger destinationMetalY = blitMetalY(ctx.draw_fbo, destinationDepth, destinationDepthLevel, dstY0, height);
		if (!encodeDepthStencilPlaneCopy(sourceDepth, sourceDepthLevel, destinationDepth, destinationDepthLevel,
			(NSUInteger)srcX0, sourceMetalY, (NSUInteger)dstX0, destinationMetalY, width, height, true)) return false;
		markDefaultDepthStencilContents(GL_DEPTH_BUFFER_BIT);
	}
	if (mask & GL_STENCIL_BUFFER_BIT) {
		if (sourceStencil == destinationStencil) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
		const NSUInteger sourceMetalY = blitMetalY(ctx.read_fbo, sourceStencil, sourceStencilLevel, srcY0, height);
		const NSUInteger destinationMetalY = blitMetalY(ctx.draw_fbo, destinationStencil, destinationStencilLevel, dstY0, height);
		if (!encodeDepthStencilPlaneCopy(sourceStencil, sourceStencilLevel, destinationStencil, destinationStencilLevel,
			(NSUInteger)srcX0, sourceMetalY, (NSUInteger)dstX0, destinationMetalY, width, height, false)) return false;
		markDefaultDepthStencilContents(GL_STENCIL_BUFFER_BIT);
	}
	return true;
}

static id<MTLSamplerState> blitSampler(bool linear) {
	static std::mutex samplerMutex;
	std::lock_guard<std::mutex> lock(samplerMutex);
	static id<MTLSamplerState> samplers[2] = { nil, nil };
	__strong id<MTLSamplerState> &s = samplers[linear ? 1 : 0];
	if (!s) {
		MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
		d.minFilter = d.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
		d.sAddressMode = d.tAddressMode = MTLSamplerAddressModeClampToEdge;
		s = [ctx.device newSamplerStateWithDescriptor:d];
	}
	return s;
}

void glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) {
	if (filter != GL_NEAREST && filter != GL_LINEAR) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBlitFramebuffer: invalid filter 0x%X", filter);
		return;
	}
	if ((mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (srcX0 == srcX1 || srcY0 == srcY1 || dstX0 == dstX1 || dstY0 == dstY1) return;
	if (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) {
		if (filter != GL_NEAREST) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: depth/stencil blits require GL_NEAREST");
			return;
		}
		if (!performDepthStencilBlit(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask)) return;
	}
	if (!(mask & GL_COLOR_BUFFER_BIT)) return;
	if (srcX0 == srcX1 || srcY0 == srcY1 || dstX0 == dstX1 || dstY0 == dstY1) return;

	// Source: colour attachment of the read framebuffer.
	if (ctx.read_fbo == 0) {
		GL4METAL_LOG("glBlitFramebuffer: reading the default framebuffer is not supported (the drawable is framebuffer-only)");
		return;
	}
	auto rit = ctx.fbo_objects.find(ctx.read_fbo);
	if (rit == ctx.fbo_objects.end() || rit->second.read_buffer < GL_COLOR_ATTACHMENT0 ||
		rit->second.read_buffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		return;
	}
	const FBOAttachment &sa = rit->second.color[rit->second.read_buffer - GL_COLOR_ATTACHMENT0];
	id<MTLTexture> src = attachmentTexture(sa);
	if (!src || (src.usage & MTLTextureUsageShaderRead) == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: source attachment is not a ShaderRead texture");
		return;
	}
	float srcW = (float)std::max<NSUInteger>(src.width >> sa.level, 1);
	float srcH = (float)std::max<NSUInteger>(src.height >> sa.level, 1);

	// Destination: first draw buffer of the draw framebuffer.
	endRenderEncoder(); // the blit is its own pass; the next draw opens a new one that loads the attachments
	bool dstDefault = (ctx.draw_fbo == 0);
	id<MTLTexture> dst = nil;
	NSUInteger dstLevel = 0;
	if (dstDefault) {
		if (!ctx.metalLayer) {
			return;
		}
		if (!ctx.currentDrawable) {
			ctx.currentDrawable = [ctx.metalLayer nextDrawable];
		}
		dst = ctx.currentDrawable.texture;
	} else {
		auto dit = ctx.fbo_objects.find(ctx.draw_fbo);
		if (dit == ctx.fbo_objects.end()) {
			return;
		}
		GLenum buf = dit->second.draw_buffers[0];
		if (buf < GL_COLOR_ATTACHMENT0 || buf >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
			return;
		}
		const FBOAttachment &da = dit->second.color[buf - GL_COLOR_ATTACHMENT0];
		dst = attachmentTexture(da);
		dstLevel = (NSUInteger)da.level;
	}
	if (!dst || (dst.usage & MTLTextureUsageRenderTarget) == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: destination attachment is not a RenderTarget texture");
		return;
	}
	if (src == dst) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: source and destination alias the same Metal texture; pass skipped to avoid read/write feedback");
		return;
	}
	float dstW = (float)std::max<NSUInteger>(dst.width >> dstLevel, 1);
	float dstH = (float)std::max<NSUInteger>(dst.height >> dstLevel, 1);

	id<MTLRenderPipelineState> pipeline = blitPipeline(dst.pixelFormat);
	if (!pipeline) {
		return;
	}

	// Scissor = destination rectangle, converted to rows and clipped to the texture.
	NSInteger gx0 = std::max<NSInteger>(std::min(dstX0, dstX1), 0), gx1 = std::min<NSInteger>(std::max(dstX0, dstX1), (NSInteger)dstW);
	NSInteger gy0 = std::max<NSInteger>(std::min(dstY0, dstY1), 0), gy1 = std::min<NSInteger>(std::max(dstY0, dstY1), (NSInteger)dstH);
	if (gx1 <= gx0 || gy1 <= gy0) {
		return;
	}
	NSInteger row0 = dstDefault ? (NSInteger)dstH - gy1 : gy0;
	NSInteger row1 = dstDefault ? (NSInteger)dstH - gy0 : gy1;

	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = dst;
	rp.colorAttachments[0].level = dstLevel;
	// A newly acquired drawable has undefined contents. Loading it before the
	// first write is both unnecessary and potentially undefined on tile GPUs.
	rp.colorAttachments[0].loadAction =
		(dstDefault && !ctx.default_color_has_contents) ? MTLLoadActionDontCare : MTLLoadActionLoad;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;

	if (!ctx.currentCommandBuffer) {
		if (!ctx.commandQueue) return;
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	if (!ctx.currentCommandBuffer) return;
	id<MTLRenderCommandEncoder> enc = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
	if (!enc) {
		GL4METAL_LOG("glBlitFramebuffer: failed to create Metal render encoder");
		return;
	}
	id<MTLSamplerState> sampler = blitSampler(filter == GL_LINEAR);
	if (!sampler) {
		[enc endEncoding];
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glBlitFramebuffer: failed to allocate Metal sampler");
		return;
	}

	BlitParams params;
	memset(&params, 0, sizeof(params));
	params.src_rect[0] = (float)srcX0; params.src_rect[1] = (float)srcY0; params.src_rect[2] = (float)srcX1; params.src_rect[3] = (float)srcY1;
	params.dst_rect[0] = (float)dstX0; params.dst_rect[1] = (float)dstY0; params.dst_rect[2] = (float)dstX1; params.dst_rect[3] = (float)dstY1;
	params.src_size[0] = srcW; params.src_size[1] = srcH;
	params.dst_size[0] = dstW; params.dst_size[1] = dstH;
	params.src_default = 0;
	params.dst_default = dstDefault ? 1 : 0;

	MTLViewport vp;
	vp.originX = 0; vp.originY = 0; vp.width = dstW; vp.height = dstH; vp.znear = 0.0; vp.zfar = 1.0;
	MTLScissorRect sr;
	sr.x = (NSUInteger)gx0; sr.y = (NSUInteger)row0; sr.width = (NSUInteger)(gx1 - gx0); sr.height = (NSUInteger)(row1 - row0);

	[enc setRenderPipelineState:pipeline];
	[enc setViewport:vp];
	[enc setScissorRect:sr];
	[enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
	[enc setFragmentTexture:src atIndex:0];
	[enc setFragmentSamplerState:sampler atIndex:0];
	[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[enc endEncoding];
	if (dstDefault) ctx.default_color_has_contents = true;
}

static bool checkedReadSizeAdd(size_t a, size_t b, size_t &out) {
	if (b > SIZE_MAX - a) return false;
	out = a + b;
	return true;
}

static bool checkedReadSizeMul(size_t a, size_t b, size_t &out) {
	if (a != 0 && b > SIZE_MAX / a) return false;
	out = a * b;
	return true;
}

static bool checkedReadAlign(size_t value, size_t alignment, size_t &out) {
	size_t sum = 0;
	if (alignment == 0 || !checkedReadSizeAdd(value, alignment - 1, sum)) return false;
	out = (sum / alignment) * alignment;
	return true;
}

void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels) {
	if (width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (width == 0 || height == 0) return;
	if (ctx.read_fbo == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glReadPixels: reading the default framebuffer is not supported (the drawable is framebuffer-only)");
		return;
	}
	if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
		GL4METAL_LOG("glReadPixels: read framebuffer %u is incomplete", ctx.read_fbo);
		return;
	}
	auto it = ctx.fbo_objects.find(ctx.read_fbo);
	if (it == ctx.fbo_objects.end() || it->second.read_buffer < GL_COLOR_ATTACHMENT0 ||
		it->second.read_buffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const FBOAttachment &a = it->second.color[it->second.read_buffer - GL_COLOR_ATTACHMENT0];
	id<MTLTexture> tex = attachmentTexture(a);
	if (!tex || a.level < 0 || (NSUInteger)a.level >= tex.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glReadPixels: selected read attachment has no valid texture mip level");
		return;
	}

	// Source: 8-bit colour textures, with the index of R, G, B, A inside one pixel (-1: absent).
	int srcComps = 0, srcIdx[4] = { -1, -1, -1, -1 };
	switch (tex.pixelFormat) {
		case MTLPixelFormatR8Unorm:         srcComps = 1; srcIdx[0] = 0; break;
		case MTLPixelFormatRG8Unorm:        srcComps = 2; srcIdx[0] = 0; srcIdx[1] = 1; break;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB: srcComps = 4; srcIdx[0] = 0; srcIdx[1] = 1; srcIdx[2] = 2; srcIdx[3] = 3; break;
		case MTLPixelFormatBGRA8Unorm:      srcComps = 4; srcIdx[2] = 0; srcIdx[1] = 1; srcIdx[0] = 2; srcIdx[3] = 3; break;
		default:
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glReadPixels: unsupported source MTLPixelFormat %d", (int)tex.pixelFormat);
			return;
	}
	// Destination layout: which source channel (0 R, 1 G, 2 B, 3 A) feeds each component.
	int dstComps = 0, dstChannel[4] = { 0, 0, 0, 0 };
	switch (format) {
		case GL_RED:  dstComps = 1; dstChannel[0] = 0; break;
		case GL_RG:   dstComps = 2; dstChannel[0] = 0; dstChannel[1] = 1; break;
		case GL_RGB:  dstComps = 3; dstChannel[0] = 0; dstChannel[1] = 1; dstChannel[2] = 2; break;
		case GL_BGR:  dstComps = 3; dstChannel[0] = 2; dstChannel[1] = 1; dstChannel[2] = 0; break;
		case GL_RGBA: dstComps = 4; dstChannel[0] = 0; dstChannel[1] = 1; dstChannel[2] = 2; dstChannel[3] = 3; break;
		case GL_BGRA: dstComps = 4; dstChannel[0] = 2; dstChannel[1] = 1; dstChannel[2] = 0; dstChannel[3] = 3; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glReadPixels: unsupported format 0x%X", format);
			return;
	}
	if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_INT_8_8_8_8_REV) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glReadPixels: unsupported type 0x%X", type);
		return;
	}
	const NSUInteger levelW = std::max<NSUInteger>(tex.width >> a.level, 1);
	const NSUInteger levelH = std::max<NSUInteger>(tex.height >> a.level, 1);
	if (x < 0 || y < 0 || (NSUInteger)x > levelW || (NSUInteger)y > levelH ||
		(NSUInteger)width > levelW - (NSUInteger)x || (NSUInteger)height > levelH - (NSUInteger)y ||
		tex.storageMode == MTLStorageModePrivate) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glReadPixels: region %d,%d %dx%d is outside the %lux%lu attachment or its storage is private", x, y, width, height, (unsigned long)levelW, (unsigned long)levelH);
		return;
	}

	size_t rawRowBytes = 0, rawBytes = 0;
	if (!checkedReadSizeMul((size_t)width, (size_t)srcComps, rawRowBytes) ||
		!checkedReadSizeMul(rawRowBytes, (size_t)height, rawBytes) || rawBytes > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glReadPixels: staging read size is too large");
		return;
	}
	size_t rowLength = ctx.pack_row_length > 0 ? (size_t)ctx.pack_row_length : (size_t)width;
	size_t rowPixelsBytes = 0, rowBytes = 0, skipRowsBytes = 0, skipPixelsBytes = 0;
	size_t lastRowBytes = 0, copyRowBytes = 0, baseOffset = (uintptr_t)pixels, requiredSpan = 0;
	if (!checkedReadSizeMul(rowLength, (size_t)dstComps, rowPixelsBytes) ||
		!checkedReadAlign(rowPixelsBytes, (size_t)std::max(ctx.pack_alignment, 1), rowBytes) ||
		!checkedReadSizeMul((size_t)ctx.pack_skip_rows, rowBytes, skipRowsBytes) ||
		!checkedReadSizeMul((size_t)ctx.pack_skip_pixels, (size_t)dstComps, skipPixelsBytes) ||
		!checkedReadSizeMul((size_t)(height - 1), rowBytes, lastRowBytes) ||
		!checkedReadSizeMul((size_t)width, (size_t)dstComps, copyRowBytes) ||
		!checkedReadSizeAdd(skipRowsBytes, skipPixelsBytes, requiredSpan) ||
		!checkedReadSizeAdd(requiredSpan, lastRowBytes, requiredSpan) ||
		!checkedReadSizeAdd(requiredSpan, copyRowBytes, requiredSpan)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glReadPixels: pixel-pack layout overflows addressable memory");
		return;
	}

	uint8_t *dst = nullptr;
	const GLuint pbo = boundBuffer(GL_PIXEL_PACK_BUFFER);
	if (pbo != 0) {
		auto buffer = ctx.buffer_objects.find(pbo);
		if (buffer == ctx.buffer_objects.end() || !buffer->second || !buffer->second.contents) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glReadPixels: GL_PIXEL_PACK_BUFFER %u has no CPU-visible storage", pbo);
			return;
		}
		const NSUInteger logicalLength = glBufferLogicalSize(pbo);
		if (baseOffset > logicalLength || requiredSpan > logicalLength - baseOffset) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glReadPixels: pixel-pack range offset=%llu span=%llu exceeds buffer %u length %llu",
				(unsigned long long)baseOffset, (unsigned long long)requiredSpan, pbo, (unsigned long long)logicalLength);
			return;
		}
		dst = (uint8_t *)buffer->second.contents + baseOffset;
	} else {
		if (!pixels) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glReadPixels: null destination pointer without GL_PIXEL_PACK_BUFFER");
			return;
		}
		dst = (uint8_t *)pixels;
	}
	dst += skipRowsBytes + skipPixelsBytes;

	flushCommandBuffer(true); // the rendering has to be finished before the CPU reads it
	std::vector<uint8_t> raw;
	try {
		raw.resize(rawBytes);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glReadPixels: CPU readback staging allocation failed (%lu bytes)", (unsigned long)rawBytes);
		return;
	}
	[tex getBytes:raw.data() bytesPerRow:rawRowBytes fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height) mipmapLevel:(NSUInteger)a.level];
	for (GLsizei row = 0; row < height; row++) {
		const uint8_t *src = raw.data() + (size_t)row * rawRowBytes;
		uint8_t *rowDst = dst + (size_t)row * rowBytes;
		for (GLsizei col = 0; col < width; col++, src += srcComps, rowDst += dstComps) {
			for (int c = 0; c < dstComps; c++) {
				const int idx = srcIdx[dstChannel[c]];
				rowDst[c] = idx >= 0 ? src[idx] : (dstChannel[c] == 3 ? 255 : 0);
			}
		}
	}
}
