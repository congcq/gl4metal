//
// framebuffer.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//


#import "framebuffer.h"

extern void invalidateDrawEncoderStateCache(void);

#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <functional>
#include <unordered_map>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <sstream>


// MARK: - Render passes

static bool framebufferDiagnosticsEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_DIAGNOSTICS");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool drawBufferTraceEnabled() {
	static const bool enabled = [] {
		if (!framebufferDiagnosticsEnabled()) return false;
		const char *value = std::getenv("GL4METAL_TRACE_DRAW_BUFFERS");
		return value && value[0] == '1';
	}();
	return enabled;
}

// Metal records work into encoders, GL into global state: a render pass is opened by glClear (or the first draw)
// for the current draw framebuffer and closed when the target changes or the frame is presented.

void endRenderEncoder(void) {
	// Metal encoders on one command buffer are sequential. Close a cached compute
	// encoder before any render-pass, blit, or resource-update path that uses this
	// common transition helper. Consecutive compute dispatches bypass this helper
	// when no render encoder is active and can therefore share one compute encoder.
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
	}
	if (ctx.currentEncoder) {
		// On Apple TBDR GPUs, StoreActionDontCare avoids writing transient attachment
		// contents from tile memory back to device memory. Default and user FBO passes
		// use Unknown descriptors and select a valid action here before endEncoding.
		for (NSUInteger colorIndex = 0; colorIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++colorIndex) {
			if ((ctx.pass_color_attachment_mask & (1u << colorIndex)) == 0) continue;
			const MTLStoreAction action = (ctx.pass_discard_color_store_mask & (1u << colorIndex))
				? MTLStoreActionDontCare : MTLStoreActionStore;
			[ctx.currentEncoder setColorStoreAction:action atIndex:colorIndex];
		}
		if (ctx.pass_depth_format != MTLPixelFormatInvalid) {
			const bool discard = ctx.pass_discard_depth_store ||
				(ctx.pass_uses_default_framebuffer && ctx.discard_default_depth_stencil_on_end);
			[ctx.currentEncoder setDepthStoreAction:discard ? MTLStoreActionDontCare : MTLStoreActionStore];
		}
		if (ctx.pass_stencil_format != MTLPixelFormatInvalid) {
			const bool discard = ctx.pass_discard_stencil_store ||
				(ctx.pass_uses_default_framebuffer && ctx.discard_default_depth_stencil_on_end);
			[ctx.currentEncoder setStencilStoreAction:discard ? MTLStoreActionDontCare : MTLStoreActionStore];
		}
		[ctx.currentEncoder endEncoding];
		ctx.currentEncoder = nil;
	}
	invalidateDrawEncoderStateCache();
	ctx.fragcoord_params_bound = false;
	ctx.discard_default_depth_stencil_on_end = false;
	ctx.pass_uses_default_framebuffer = false;
	ctx.pass_color_attachment_mask = 0;
	ctx.pass_discard_color_store_mask = 0;
	ctx.pass_discard_depth_store = false;
	ctx.pass_discard_stencil_store = false;
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

static void logFBOAttachmentState(GLuint framebufferName, const char *attachmentName, const FBOAttachment &attachment) {
	if (!hasAttachment(attachment)) return;
	if (attachment.texture != 0) {
		auto object = ctx.texture_objects.find(attachment.texture);
		if (object == ctx.texture_objects.end()) {
			GL4METAL_LOG("FBO %u %s: missing GL texture object name=%u level=%d",
				framebufferName, attachmentName, attachment.texture, attachment.level);
			return;
		}
		const TextureObject &textureObject = object->second;
		id<MTLTexture> metalTexture = textureObject.texture;
		if (!metalTexture) {
			GL4METAL_LOG("FBO %u %s: GL texture=%u has no Metal storage (tracked internalformat=0x%X, imageSize=%dx%dx%d, target=0x%X, level=%d)",
				framebufferName, attachmentName, attachment.texture, textureObject.internalformat,
				textureObject.width, textureObject.height, textureObject.depth, textureObject.target, attachment.level);
			return;
		}
		GL4METAL_LOG("FBO %u %s: texture=%u level=%d MetalFormat=%lu size=%lux%lu type=%lu usage=0x%lx sampleCount=%lu renderTarget=%d",
			framebufferName, attachmentName, attachment.texture, attachment.level,
			(unsigned long)metalTexture.pixelFormat, (unsigned long)metalTexture.width, (unsigned long)metalTexture.height,
			(unsigned long)metalTexture.textureType, (unsigned long)metalTexture.usage,
			(unsigned long)metalTexture.sampleCount,
			(metalTexture.usage & MTLTextureUsageRenderTarget) != 0 ? 1 : 0);
		return;
	}
	auto object = ctx.renderbuffer_objects.find(attachment.renderbuffer);
	if (object == ctx.renderbuffer_objects.end() || !object->second.initialized || !object->second.texture) {
		GL4METAL_LOG("FBO %u %s: renderbuffer=%u has no initialized Metal storage",
			framebufferName, attachmentName, attachment.renderbuffer);
		return;
	}
	id<MTLTexture> metalTexture = object->second.texture;
	GL4METAL_LOG("FBO %u %s: renderbuffer=%u MetalFormat=%lu size=%lux%lu samples=%lu renderTarget=%d",
		framebufferName, attachmentName, attachment.renderbuffer,
		(unsigned long)metalTexture.pixelFormat, (unsigned long)metalTexture.width, (unsigned long)metalTexture.height,
		(unsigned long)metalTexture.sampleCount,
		(metalTexture.usage & MTLTextureUsageRenderTarget) != 0 ? 1 : 0);
}

static void logFBOAttachmentStates(GLuint framebufferName, const FBO &fbo) {
	logFBOAttachmentState(framebufferName, "DEPTH_ATTACHMENT", fbo.depth);
	logFBOAttachmentState(framebufferName, "STENCIL_ATTACHMENT", fbo.stencil);
	for (int attachmentIndex = 0; attachmentIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachmentIndex) {
		char attachmentName[32];
		snprintf(attachmentName, sizeof(attachmentName), "COLOR_ATTACHMENT%d", attachmentIndex);
		logFBOAttachmentState(framebufferName, attachmentName, fbo.color[attachmentIndex]);
	}
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
	// Keep this Private rather than Memoryless: OpenGL may split one logical frame into multiple
	// Metal render encoders (FBO switches, blits, feedback snapshots, or readback), and a later
	// pass may need the previous depth/stencil values. End-of-frame invalidation below safely
	// discards the final store without making intermediate passes lose their contents.
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
	uint32_t consumedInvalidatedColors = 0;
	bool consumedInvalidatedDepth = false;
	bool consumedInvalidatedStencil = false;

	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		ctx.pass_color_format[i] = MTLPixelFormatInvalid;
	}
	ctx.pass_depth_format = MTLPixelFormatInvalid;
	ctx.pass_stencil_format = MTLPixelFormatInvalid;
	ctx.pass_sample_count = 1;
	ctx.pass_color_attachment_mask = 0;
	ctx.pass_discard_color_store_mask = 0;
	ctx.pass_discard_depth_store = false;
	ctx.pass_discard_stencil_store = false;
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
		ca.storeAction = MTLStoreActionUnknown;
		ca.clearColor = ctx.clear_color;
		ctx.pass_color_format[0] = color.pixelFormat;
		ctx.pass_color_attachment_mask |= 1u;

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
			auto diagnosticFBO = GL4METAL_FBO_FIND(ctx.draw_fbo);
			if (diagnosticFBO != ctx.fbo_objects.end()) {
				logFBOAttachmentStates(ctx.draw_fbo, diagnosticFBO->second);
			}
			return nil;
		}
		auto it = GL4METAL_FBO_FIND(ctx.draw_fbo);
		if (it == ctx.fbo_objects.end()) {
			return nil;
		}
		FBO &fbo = it->second;
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
			const uint32_t attachmentBit = 1u << (buf - GL_COLOR_ATTACHMENT0);
			const bool invalidated = (fbo.invalidated_color_mask & attachmentBit) != 0;
			ca.loadAction = (clearMask & GL_COLOR_BUFFER_BIT) ? MTLLoadActionClear
				: (invalidated ? MTLLoadActionDontCare : MTLLoadActionLoad);
			ca.storeAction = MTLStoreActionUnknown;
			ca.clearColor = ctx.clear_color;
			ctx.pass_color_format[i] = tex.pixelFormat;
			ctx.pass_color_attachment_mask |= (1u << i);
			if (invalidated) consumedInvalidatedColors |= attachmentBit;
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
			const bool depthInvalidated = fbo.invalidated_depth;
			rp.depthAttachment.loadAction = (clearMask & GL_DEPTH_BUFFER_BIT) ? MTLLoadActionClear
				: (depthInvalidated ? MTLLoadActionDontCare : MTLLoadActionLoad);
			rp.depthAttachment.storeAction = MTLStoreActionUnknown;
			rp.depthAttachment.clearDepth = ctx.clear_depth;
			consumedInvalidatedDepth = depthInvalidated;
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
			const bool stencilInvalidated = fbo.invalidated_stencil;
			rp.stencilAttachment.loadAction = (clearMask & GL_STENCIL_BUFFER_BIT) ? MTLLoadActionClear
				: (stencilInvalidated ? MTLLoadActionDontCare : MTLLoadActionLoad);
			rp.stencilAttachment.storeAction = MTLStoreActionUnknown;
			rp.stencilAttachment.clearStencil = (uint32_t)ctx.clear_stencil;
			consumedInvalidatedStencil = stencilInvalidated;
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
		// A load-only pass doesn't create valid contents by itself. Mark only aspects
		// explicitly cleared here; draw paths mark depth/color/stencil after a real write.
		if (clearMask & GL_COLOR_BUFFER_BIT) ctx.default_color_has_contents = true;
		if (clearMask & GL_DEPTH_BUFFER_BIT) ctx.default_depth_has_contents = true;
		if (clearMask & GL_STENCIL_BUFFER_BIT) ctx.default_stencil_has_contents = true;
	} else {
		auto fboIt = GL4METAL_FBO_FIND(ctx.draw_fbo);
		if (fboIt != ctx.fbo_objects.end()) {
			fboIt->second.invalidated_color_mask &= ~consumedInvalidatedColors;
			if (consumedInvalidatedDepth) fboIt->second.invalidated_depth = false;
			if (consumedInvalidatedStencil) fboIt->second.invalidated_stencil = false;
		}
	}
	applyViewportAndScissor();
	return ctx.currentEncoder;
}

// MARK: - Explicit attachment invalidation (TBDR load/store optimization)

static bool isFramebufferTarget(GLenum target) {
	return target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER;
}

static GLuint framebufferNameForTarget(GLenum target) {
	return target == GL_READ_FRAMEBUFFER ? ctx.read_fbo : ctx.draw_fbo;
}

static bool isValidInvalidateAttachment(GLuint framebufferName, GLenum attachment) {
	if (framebufferName == 0) {
		return attachment == GL_COLOR || attachment == GL_DEPTH || attachment == GL_STENCIL;
	}
	return (attachment >= GL_COLOR_ATTACHMENT0 &&
		attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) ||
		attachment == GL_DEPTH_ATTACHMENT || attachment == GL_STENCIL_ATTACHMENT ||
		attachment == GL_DEPTH_STENCIL_ATTACHMENT;
}

static bool framebufferExtent(GLuint framebufferName, NSUInteger &width, NSUInteger &height) {
	width = height = 0;
	if (framebufferName == 0) {
		id<MTLTexture> color = ctx.currentDrawable.texture;
		if (!color && ctx.metalLayer) {
			const CGSize drawableSize = ctx.metalLayer.drawableSize;
			width = (NSUInteger)std::max<CGFloat>(drawableSize.width, 0);
			height = (NSUInteger)std::max<CGFloat>(drawableSize.height, 0);
			return width != 0 && height != 0;
		}
		if (!color) return false;
		width = color.width;
		height = color.height;
		return true;
	}
	auto fboIt = GL4METAL_FBO_FIND(framebufferName);
	if (fboIt == ctx.fbo_objects.end()) return false;
	const FBO &fbo = fboIt->second;
	auto useAttachment = [&](const FBOAttachment &attachment) -> bool {
		id<MTLTexture> texture = attachmentTexture(attachment);
		if (!texture) return false;
		width = std::max<NSUInteger>(1, texture.width >> (NSUInteger)attachment.level);
		height = std::max<NSUInteger>(1, texture.height >> (NSUInteger)attachment.level);
		return true;
	};
	if (useAttachment(fbo.depth) || useAttachment(fbo.stencil)) return true;
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; ++i) {
		if (useAttachment(fbo.color[i])) return true;
	}
	return false;
}

static void invalidateFramebufferInternal(GLenum target, GLsizei numAttachments,
	const GLenum *attachments, bool fullAttachment) {
	if (!isFramebufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (numAttachments < 0 || (numAttachments > 0 && !attachments)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const GLuint framebufferName = framebufferNameForTarget(target);
	if (framebufferName != 0 && GL4METAL_FBO_FIND(framebufferName) == ctx.fbo_objects.end()) {
		gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
		return;
	}
	for (GLsizei i = 0; i < numAttachments; ++i) {
		if (!isValidInvalidateAttachment(framebufferName, attachments[i])) {
			gl4metalSetError(GL_INVALID_ENUM);
			return;
		}
	}
	// Partial invalidation cannot be represented by a Metal attachment store action.
	// Conservatively retain those pixels rather than incorrectly discarding the whole image.
	if (!fullAttachment || numAttachments == 0) return;

	bool activePassTouched = false;
	if (framebufferName == 0) {
		for (GLsizei i = 0; i < numAttachments; ++i) {
			switch (attachments[i]) {
				case GL_COLOR:
					ctx.default_color_has_contents = false;
					if (ctx.currentEncoder && ctx.draw_fbo == 0 && (ctx.pass_color_attachment_mask & 1u)) {
						ctx.pass_discard_color_store_mask |= 1u;
						activePassTouched = true;
					}
					break;
				case GL_DEPTH:
					ctx.default_depth_has_contents = false;
					if (ctx.currentEncoder && ctx.draw_fbo == 0 && ctx.pass_depth_format != MTLPixelFormatInvalid) {
						ctx.pass_discard_depth_store = true;
						activePassTouched = true;
					}
					break;
				case GL_STENCIL:
					ctx.default_stencil_has_contents = false;
					if (ctx.currentEncoder && ctx.draw_fbo == 0 && ctx.pass_stencil_format != MTLPixelFormatInvalid) {
						ctx.pass_discard_stencil_store = true;
						activePassTouched = true;
					}
					break;
			}
		}
	} else {
		FBO &fbo = ctx.fbo_objects[framebufferName];
		for (GLsizei i = 0; i < numAttachments; ++i) {
			const GLenum attachment = attachments[i];
			if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
				const uint32_t bit = 1u << (attachment - GL_COLOR_ATTACHMENT0);
				fbo.invalidated_color_mask |= bit;
				if (ctx.currentEncoder && ctx.draw_fbo == framebufferName) {
					for (NSUInteger colorIndex = 0; colorIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++colorIndex) {
						if ((ctx.pass_color_attachment_mask & (1u << colorIndex)) &&
							fbo.draw_buffers[colorIndex] == attachment) {
							ctx.pass_discard_color_store_mask |= (1u << colorIndex);
							activePassTouched = true;
						}
					}
				}
			} else if (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
				fbo.invalidated_depth = true;
				if (ctx.currentEncoder && ctx.draw_fbo == framebufferName && ctx.pass_depth_format != MTLPixelFormatInvalid) {
					ctx.pass_discard_depth_store = true;
					activePassTouched = true;
				}
			}
			if (attachment == GL_STENCIL_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
				fbo.invalidated_stencil = true;
				if (ctx.currentEncoder && ctx.draw_fbo == framebufferName && ctx.pass_stencil_format != MTLPixelFormatInvalid) {
					ctx.pass_discard_stencil_store = true;
					activePassTouched = true;
				}
			}
		}
	}
	if (activePassTouched) endRenderEncoder();
}

void gl4metalInvalidateFramebufferImpl(GLenum target, GLsizei numAttachments, const GLenum *attachments) {
	invalidateFramebufferInternal(target, numAttachments, attachments, true);
}

void gl4metalInvalidateSubFramebufferImpl(GLenum target, GLsizei numAttachments, const GLenum *attachments,
	GLint x, GLint y, GLsizei width, GLsizei height) {
	if (width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!isFramebufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (numAttachments < 0 || (numAttachments > 0 && !attachments)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const GLuint framebufferName = framebufferNameForTarget(target);
	if (framebufferName != 0 && GL4METAL_FBO_FIND(framebufferName) == ctx.fbo_objects.end()) {
		gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
		return;
	}
	for (GLsizei i = 0; i < numAttachments; ++i) {
		if (!isValidInvalidateAttachment(framebufferName, attachments[i])) {
			gl4metalSetError(GL_INVALID_ENUM);
			return;
		}
	}
	if (width == 0 || height == 0 || numAttachments == 0) return;
	NSUInteger fbWidth = 0, fbHeight = 0;
	if (!framebufferExtent(framebufferName, fbWidth, fbHeight)) return;
	const int64_t right = (int64_t)x + (int64_t)width;
	const int64_t top = (int64_t)y + (int64_t)height;
	const bool coversWholeAttachment = x <= 0 && y <= 0 &&
		right >= (int64_t)fbWidth && top >= (int64_t)fbHeight;
	invalidateFramebufferInternal(target, numAttachments, attachments, coversWholeAttachment);
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
	return framebuffer != 0 && GL4METAL_FBO_FIND(framebuffer) != ctx.fbo_objects.end();
}

void glBindFramebuffer(GLenum target, GLuint framebuffer) {
	if (framebuffer != 0 && GL4METAL_FBO_FIND(framebuffer) == ctx.fbo_objects.end()) {
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
	auto it = GL4METAL_FBO_FIND(buf_id);
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
		fbo.invalidated_color_mask &= ~(1u << (attachment - GL_COLOR_ATTACHMENT0));
	} else if (attachment == GL_DEPTH_ATTACHMENT) {
		fbo.depth = a;
		fbo.invalidated_depth = false;
	} else if (attachment == GL_STENCIL_ATTACHMENT) {
		fbo.stencil = a;
		fbo.invalidated_stencil = false;
	} else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		fbo.depth = a;
		fbo.stencil = a;
		fbo.invalidated_depth = false;
		fbo.invalidated_stencil = false;
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
		fbo.invalidated_color_mask &= ~(1u << (attachment - GL_COLOR_ATTACHMENT0));
	} else if (attachment == GL_DEPTH_ATTACHMENT) {
		fbo.depth = newAttachment;
		fbo.invalidated_depth = false;
	} else if (attachment == GL_STENCIL_ATTACHMENT) {
		fbo.stencil = newAttachment;
		fbo.invalidated_stencil = false;
	} else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		fbo.depth = newAttachment;
		fbo.stencil = newAttachment;
		fbo.invalidated_depth = false;
		fbo.invalidated_stencil = false;
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
	auto it = GL4METAL_FBO_FIND(buf_id);
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
	auto it = GL4METAL_FBO_FIND(ctx.draw_fbo);
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
		static std::mutex *traceMutex = new std::mutex();
		static std::unordered_set<std::string> reportedStates;
		bool firstReport = false;
		{
			std::lock_guard<std::mutex> lock(*traceMutex);
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
	auto it = GL4METAL_FBO_FIND(ctx.read_fbo);
	if (it != ctx.fbo_objects.end()) {
		it->second.read_buffer = src;
	}
}

// MARK: - Clear

static inline double clamp01(double v) {
	return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

enum class ClearValueKind : uint8_t {
	Float,
	SignedInteger,
	UnsignedInteger
};

enum class ClearOutputKind : uint8_t {
	Float,
	SignedInteger,
	UnsignedInteger
};

struct ClearColorValue {
	ClearValueKind kind = ClearValueKind::Float;
	float floatValues[4] = {};
	int32_t signedValues[4] = {};
	uint32_t unsignedValues[4] = {};
};

// MSL uses 16-byte vector alignment. This layout intentionally mirrors ClearParams below.
struct alignas(16) ClearFragmentParameters {
	float floatColors[GL4METAL_MAX_COLOR_ATTACHMENTS][4] = {};
	int32_t signedColors[GL4METAL_MAX_COLOR_ATTACHMENTS][4] = {};
	uint32_t unsignedColors[GL4METAL_MAX_COLOR_ATTACHMENTS][4] = {};
	float depth = 1.0f;
};
static_assert(GL4METAL_MAX_COLOR_ATTACHMENTS == 8, "Clear MSL arrays must match the GL color attachment limit");
static_assert(offsetof(ClearFragmentParameters, signedColors) == 128, "Clear parameter signed-color offset mismatch");
static_assert(offsetof(ClearFragmentParameters, unsignedColors) == 256, "Clear parameter unsigned-color offset mismatch");
static_assert(offsetof(ClearFragmentParameters, depth) == 384, "Clear parameter depth offset mismatch");
static_assert(sizeof(ClearFragmentParameters) == 400, "Clear parameter size mismatch");

static ClearOutputKind clearOutputKindForPixelFormat(MTLPixelFormat format) {
	switch (format) {
		case MTLPixelFormatR8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatR32Sint:
		case MTLPixelFormatRG8Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG32Sint:
		case MTLPixelFormatRGBA8Sint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Sint:
			return ClearOutputKind::SignedInteger;
		case MTLPixelFormatR8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatR32Uint:
		case MTLPixelFormatRG8Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG32Uint:
		case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA32Uint:
		case MTLPixelFormatRGB10A2Uint:
			return ClearOutputKind::UnsignedInteger;
		default:
			return ClearOutputKind::Float;
	}
}

static bool clearMaskHasAnyComponent(const GLboolean mask[4]) {
	return mask[0] || mask[1] || mask[2] || mask[3];
}

static MTLColorWriteMask metalClearWriteMask(const GLboolean mask[4]) {
	MTLColorWriteMask result = MTLColorWriteMaskNone;
	if (mask[0]) result |= MTLColorWriteMaskRed;
	if (mask[1]) result |= MTLColorWriteMaskGreen;
	if (mask[2]) result |= MTLColorWriteMaskBlue;
	if (mask[3]) result |= MTLColorWriteMaskAlpha;
	return result;
}

static bool colorMaskIsFull(const GLboolean mask[4]) {
	return mask[0] && mask[1] && mask[2] && mask[3];
}

static bool getEffectiveClearScissor(MTLScissorRect &rect) {
	const NSInteger width = (NSInteger)ctx.render_target_width;
	const NSInteger height = (NSInteger)ctx.render_target_height;
	if (width <= 0 || height <= 0) return false;

	NSInteger x0 = 0, y0 = 0, x1 = width, y1 = height;
	if (ctx.scissor_test_enabled) {
		const NSInteger sx0 = (NSInteger)ctx.scissor.x;
		const NSInteger sy0 = (NSInteger)ctx.scissor.y;
		const NSInteger sx1 = sx0 + (NSInteger)ctx.scissor.w;
		const NSInteger sy1 = sy0 + (NSInteger)ctx.scissor.h;
		const bool mirror = !ctx.pass_flipped;
		const NSInteger mappedY0 = mirror ? height - sy1 : sy0;
		const NSInteger mappedY1 = mirror ? height - sy0 : sy1;
		x0 = std::min<NSInteger>(std::max<NSInteger>(sx0, 0), width);
		x1 = std::min<NSInteger>(std::max<NSInteger>(sx1, 0), width);
		y0 = std::min<NSInteger>(std::max<NSInteger>(mappedY0, 0), height);
		y1 = std::min<NSInteger>(std::max<NSInteger>(mappedY1, 0), height);
	}
	if (x1 <= x0 || y1 <= y0) return false;
	rect.x = (NSUInteger)x0;
	rect.y = (NSUInteger)y0;
	rect.width = (NSUInteger)(x1 - x0);
	rect.height = (NSUInteger)(y1 - y0);
	return true;
}

static void appendClearKeyValue(std::string &key, uint64_t value) {
	key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

static std::string clearShaderSource(uint32_t colorOutputMask, bool writeDepth,
	const ClearOutputKind outputKinds[GL4METAL_MAX_COLOR_ATTACHMENTS]) {
	std::ostringstream source;
	source << R"MSL(#include <metal_stdlib>
using namespace metal;

struct ClearParams {
    float4 floatColors[8];
    int4 signedColors[8];
    uint4 unsignedColors[8];
    float depth;
};
)MSL";

	if (colorOutputMask != 0 || writeDepth) {
		source << "struct ClearOutput {\n";
		for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
			if ((colorOutputMask & (1u << attachment)) == 0) continue;
			const char *typeName = "float4";
			if (outputKinds[attachment] == ClearOutputKind::SignedInteger) typeName = "int4";
			else if (outputKinds[attachment] == ClearOutputKind::UnsignedInteger) typeName = "uint4";
			source << "    " << typeName << " color" << attachment << " [[color(" << attachment << ")]];\n";
		}
		if (writeDepth) source << "    float depth [[depth(any)]];\n";
		source << "};\n";
	}

	source << R"MSL(
vertex float4 clearVertex(uint vertexIndex [[vertex_id]]) {
    float2 position = float2(-1.0, -1.0);
    if (vertexIndex == 1) position = float2(3.0, -1.0);
    else if (vertexIndex == 2) position = float2(-1.0, 3.0);
    return float4(position, 0.0, 1.0);
}
)MSL";

	if (colorOutputMask != 0 || writeDepth) {
		source << "fragment ClearOutput clearFragment(constant ClearParams& params [[buffer(0)]]) {\n";
		source << "    ClearOutput output;\n";
		for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
			if ((colorOutputMask & (1u << attachment)) == 0) continue;
			const char *fieldName = "floatColors";
			if (outputKinds[attachment] == ClearOutputKind::SignedInteger) fieldName = "signedColors";
			else if (outputKinds[attachment] == ClearOutputKind::UnsignedInteger) fieldName = "unsignedColors";
			source << "    output.color" << attachment << " = params." << fieldName << "[" << attachment << "];\n";
		}
		if (writeDepth) source << "    output.depth = params.depth;\n";
		source << "    return output;\n}\n";
	}
	return source.str();
}

static id<MTLRenderPipelineState> clearPipeline(uint32_t colorOutputMask, bool writeDepth,
	const ClearOutputKind outputKinds[GL4METAL_MAX_COLOR_ATTACHMENTS]) {
	using ClearPipelineCache = std::unordered_map<std::string, id<MTLRenderPipelineState>>;
	static thread_local ClearPipelineCache *cache = new ClearPipelineCache();
	if (!ctx.device) return nil;

	std::string key;
	appendClearKeyValue(key, (uint64_t)(uintptr_t)(__bridge void *)ctx.device);
	appendClearKeyValue(key, colorOutputMask);
	appendClearKeyValue(key, writeDepth ? 1u : 0u);
	appendClearKeyValue(key, (uint64_t)ctx.pass_depth_format);
	appendClearKeyValue(key, (uint64_t)ctx.pass_stencil_format);
	appendClearKeyValue(key, (uint64_t)ctx.pass_sample_count);
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		appendClearKeyValue(key, (uint64_t)ctx.pass_color_format[attachment]);
		appendClearKeyValue(key, (uint64_t)outputKinds[attachment]);
		appendClearKeyValue(key, (uint64_t)metalClearWriteMask(ctx.color_masks[attachment]));
	}
	auto cached = cache->find(key);
	if (cached != cache->end()) return cached->second;

	const std::string source = clearShaderSource(colorOutputMask, writeDepth, outputKinds);
	NSString *sourceString = [NSString stringWithUTF8String:source.c_str()];
	NSError *libraryError = nil;
	id<MTLLibrary> library = sourceString ? [ctx.device newLibraryWithSource:sourceString options:nil error:&libraryError] : nil;
	if (!library) {
		GL4METAL_LOG("glClear: failed to compile Metal clear shader: %s",
			libraryError ? libraryError.localizedDescription.UTF8String : "invalid MSL source");
		cache->emplace(std::move(key), nil);
		return nil;
	}
	id<MTLFunction> vertexFunction = [library newFunctionWithName:@"clearVertex"];
	id<MTLFunction> fragmentFunction = (colorOutputMask != 0 || writeDepth)
		? [library newFunctionWithName:@"clearFragment"] : nil;
	if (!vertexFunction || ((colorOutputMask != 0 || writeDepth) && !fragmentFunction)) {
		GL4METAL_LOG("glClear: Metal clear shader is missing an entry point");
		cache->emplace(std::move(key), nil);
		return nil;
	}

	MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
	descriptor.vertexFunction = vertexFunction;
	// A nil fragment function is the Metal depth/stencil-only path. Rasterization
	// still updates depth/stencil, while no color fragment outputs are generated.
	descriptor.fragmentFunction = fragmentFunction;
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		if (ctx.pass_color_format[attachment] == MTLPixelFormatInvalid) continue;
		MTLRenderPipelineColorAttachmentDescriptor *color = descriptor.colorAttachments[attachment];
		color.pixelFormat = ctx.pass_color_format[attachment];
		color.writeMask = (colorOutputMask & (1u << attachment))
			? metalClearWriteMask(ctx.color_masks[attachment]) : MTLColorWriteMaskNone;
	}
	descriptor.depthAttachmentPixelFormat = ctx.pass_depth_format;
	descriptor.stencilAttachmentPixelFormat = ctx.pass_stencil_format;
	descriptor.rasterSampleCount = std::max<NSUInteger>(1, ctx.pass_sample_count);

	NSError *pipelineError = nil;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:descriptor error:&pipelineError];
	if (!pipeline) {
		GL4METAL_LOG("glClear: failed to create Metal clear pipeline: %s",
			pipelineError ? pipelineError.localizedDescription.UTF8String : "unknown Metal pipeline error");
	}
	cache->emplace(std::move(key), pipeline);
	return pipeline;
}

static id<MTLDepthStencilState> clearDepthStencilState(bool writeDepth, bool writeStencil,
	uint32_t frontStencilMask, uint32_t backStencilMask) {
	using ClearDepthCache = std::unordered_map<std::string, id<MTLDepthStencilState>>;
	static thread_local ClearDepthCache *cache = new ClearDepthCache();
	std::string key;
	appendClearKeyValue(key, (uint64_t)(uintptr_t)(__bridge void *)ctx.device);
	appendClearKeyValue(key, (uint64_t)ctx.pass_depth_format);
	appendClearKeyValue(key, (uint64_t)ctx.pass_stencil_format);
	appendClearKeyValue(key, writeDepth ? 1u : 0u);
	appendClearKeyValue(key, writeStencil ? 1u : 0u);
	appendClearKeyValue(key, frontStencilMask);
	appendClearKeyValue(key, backStencilMask);
	auto cached = cache->find(key);
	if (cached != cache->end()) return cached->second;

	MTLDepthStencilDescriptor *descriptor = [[MTLDepthStencilDescriptor alloc] init];
	descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	descriptor.depthWriteEnabled = writeDepth ? YES : NO;
	auto makeClearStencil = [](uint32_t writeMask) {
		MTLStencilDescriptor *stencil = [[MTLStencilDescriptor alloc] init];
		stencil.stencilCompareFunction = MTLCompareFunctionAlways;
		stencil.stencilFailureOperation = MTLStencilOperationKeep;
		stencil.depthFailureOperation = MTLStencilOperationKeep;
		stencil.depthStencilPassOperation = writeMask ? MTLStencilOperationReplace : MTLStencilOperationKeep;
		stencil.readMask = UINT32_MAX;
		stencil.writeMask = writeMask;
		return stencil;
	};
	descriptor.frontFaceStencil = makeClearStencil(writeStencil ? frontStencilMask : 0);
	descriptor.backFaceStencil = makeClearStencil(writeStencil ? backStencilMask : 0);
	id<MTLDepthStencilState> state = [ctx.device newDepthStencilStateWithDescriptor:descriptor];
	cache->emplace(std::move(key), state);
	return state;
}

static bool drawBufferHasColorTarget(GLint drawbuffer) {
	if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) return false;
	if (ctx.draw_fbo == 0) return drawbuffer == 0;
	auto fbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
	return fbo != ctx.fbo_objects.end() && fbo->second.draw_buffers[drawbuffer] != GL_NONE;
}

static bool anyWritableColorTarget(GLint drawbuffer) {
	for (int index = 0; index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++index) {
		if (drawbuffer >= 0 && index != drawbuffer) continue;
		if (drawBufferHasColorTarget(index) && clearMaskHasAnyComponent(ctx.color_masks[index])) return true;
	}
	return false;
}

static bool allColorTargetsHaveFullMask() {
	bool foundTarget = false;
	const int limit = ctx.draw_fbo == 0 ? 1 : GL4METAL_MAX_COLOR_ATTACHMENTS;
	for (int index = 0; index < limit; ++index) {
		if (!drawBufferHasColorTarget(index)) continue;
		foundTarget = true;
		if (!colorMaskIsFull(ctx.color_masks[index])) return false;
	}
	return foundTarget;
}

static bool framebufferColorLoadClearIsFloatCompatible() {
	if (ctx.draw_fbo == 0) return true;
	auto fbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
	if (fbo == ctx.fbo_objects.end()) return false;
	for (int index = 0; index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++index) {
		if (fbo->second.draw_buffers[index] == GL_NONE) continue;
		const GLenum attachmentEnum = fbo->second.draw_buffers[index];
		if (attachmentEnum < GL_COLOR_ATTACHMENT0 || attachmentEnum >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) continue;
		id<MTLTexture> texture = attachmentTexture(fbo->second.color[attachmentEnum - GL_COLOR_ATTACHMENT0]);
		if (texture && clearOutputKindForPixelFormat(texture.pixelFormat) != ClearOutputKind::Float) return false;
	}
	return true;
}

static void populateClearColor(ClearFragmentParameters &parameters, int attachment,
	MTLPixelFormat pixelFormat, const ClearColorValue &value, bool allowFloatToInteger) {
	const ClearOutputKind outputKind = clearOutputKindForPixelFormat(pixelFormat);
	for (int component = 0; component < 4; ++component) {
		if (outputKind == ClearOutputKind::Float) {
			if (value.kind == ClearValueKind::Float) parameters.floatColors[attachment][component] = value.floatValues[component];
			else if (value.kind == ClearValueKind::SignedInteger) parameters.floatColors[attachment][component] = (float)value.signedValues[component];
			else parameters.floatColors[attachment][component] = (float)value.unsignedValues[component];
		} else if (outputKind == ClearOutputKind::SignedInteger) {
			if (value.kind == ClearValueKind::SignedInteger) parameters.signedColors[attachment][component] = value.signedValues[component];
			else if (value.kind == ClearValueKind::UnsignedInteger) parameters.signedColors[attachment][component] = value.unsignedValues[component] > (uint32_t)INT32_MAX ? INT32_MAX : (int32_t)value.unsignedValues[component];
			else if (allowFloatToInteger) {
				const double v = value.floatValues[component];
				parameters.signedColors[attachment][component] = v >= (double)INT32_MAX ? INT32_MAX : (v <= (double)INT32_MIN ? INT32_MIN : (int32_t)v);
			}
		} else {
			if (value.kind == ClearValueKind::UnsignedInteger) parameters.unsignedColors[attachment][component] = value.unsignedValues[component];
			else if (value.kind == ClearValueKind::SignedInteger) parameters.unsignedColors[attachment][component] = value.signedValues[component] < 0 ? 0u : (uint32_t)value.signedValues[component];
			else if (allowFloatToInteger) {
				const double v = value.floatValues[component];
				parameters.unsignedColors[attachment][component] = v <= 0.0 ? 0u : (v >= (double)UINT32_MAX ? UINT32_MAX : (uint32_t)v);
			}
		}
	}
}

static GLbitfield fastClearLoadMask(GLbitfield mask, bool allowFastColor) {
	if (ctx.scissor_test_enabled) return 0;
	GLbitfield fastMask = 0;
	if ((mask & GL_COLOR_BUFFER_BIT) && allowFastColor && allColorTargetsHaveFullMask() && framebufferColorLoadClearIsFloatCompatible())
		fastMask |= GL_COLOR_BUFFER_BIT;
	if ((mask & GL_DEPTH_BUFFER_BIT) && ctx.depth_write_enabled)
		fastMask |= GL_DEPTH_BUFFER_BIT;
	// OpenGL glClear uses the front stencil write mask even when the back mask differs.
	if ((mask & GL_STENCIL_BUFFER_BIT) && ctx.stencil_front.write_mask == UINT32_MAX)
		fastMask |= GL_STENCIL_BUFFER_BIT;
	return fastMask;
}

static void performFramebufferClear(GLbitfield mask, GLint drawbuffer,
	const ClearColorValue *colorValue, GLfloat depthValue, GLint stencilValue, bool allowFastLoad) {
	mask &= GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
	if (mask == 0) return;
	if ((mask & GL_COLOR_BUFFER_BIT) && !anyWritableColorTarget(drawbuffer)) mask &= ~GL_COLOR_BUFFER_BIT;
	if ((mask & GL_DEPTH_BUFFER_BIT) && !ctx.depth_write_enabled) mask &= ~GL_DEPTH_BUFFER_BIT;
	const bool hasStencilWriteMask = ctx.stencil_front.write_mask != 0;
	if ((mask & GL_STENCIL_BUFFER_BIT) && !hasStencilWriteMask) mask &= ~GL_STENCIL_BUFFER_BIT;
	if (mask == 0) return;

	const GLbitfield fastMask = allowFastLoad && drawbuffer < 0
		? fastClearLoadMask(mask, true) : 0;
	id<MTLRenderCommandEncoder> encoder = beginRenderPass(fastMask);
	if (!encoder) return;
	const GLbitfield drawMask = mask & ~fastMask;
	if (drawMask == 0) return; // Metal performed the full-attachment clears as load actions.

	MTLScissorRect scissorRect;
	if (!getEffectiveClearScissor(scissorRect)) return;

	uint32_t colorOutputMask = 0;
	ClearOutputKind outputKinds[GL4METAL_MAX_COLOR_ATTACHMENTS];
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		outputKinds[attachment] = ClearOutputKind::Float;
		if ((drawMask & GL_COLOR_BUFFER_BIT) == 0 || (drawbuffer >= 0 && drawbuffer != attachment)) continue;
		if (ctx.pass_color_format[attachment] == MTLPixelFormatInvalid || !clearMaskHasAnyComponent(ctx.color_masks[attachment])) continue;
		outputKinds[attachment] = clearOutputKindForPixelFormat(ctx.pass_color_format[attachment]);
		if (colorValue && colorValue->kind == ClearValueKind::SignedInteger && outputKinds[attachment] != ClearOutputKind::SignedInteger) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		if (colorValue && colorValue->kind == ClearValueKind::UnsignedInteger && outputKinds[attachment] != ClearOutputKind::UnsignedInteger) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		if (colorValue && colorValue->kind == ClearValueKind::Float && outputKinds[attachment] != ClearOutputKind::Float && !allowFastLoad) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		colorOutputMask |= (1u << attachment);
	}

	const bool writeDepth = (drawMask & GL_DEPTH_BUFFER_BIT) != 0 && ctx.pass_depth_format != MTLPixelFormatInvalid;
	// Per OpenGL semantics, clears always use GL_STENCIL_WRITEMASK (front face),
	// not GL_STENCIL_BACK_WRITEMASK. Apply that same mask to both Metal faces so
	// the fullscreen primitive's winding cannot change the clear result.
	const uint32_t frontStencilMask = ctx.stencil_front.write_mask;
	const bool writeStencil = (drawMask & GL_STENCIL_BUFFER_BIT) != 0 &&
		ctx.pass_stencil_format != MTLPixelFormatInvalid && frontStencilMask != 0;
	if (colorOutputMask == 0 && !writeDepth && !writeStencil) return;

	id<MTLRenderPipelineState> pipeline = clearPipeline(colorOutputMask, writeDepth, outputKinds);
	id<MTLDepthStencilState> depthState = clearDepthStencilState(writeDepth, writeStencil, frontStencilMask, frontStencilMask);
	if (!pipeline || !depthState) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}

	ClearFragmentParameters parameters;
	if (colorOutputMask != 0 && colorValue) {
		for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
			if (colorOutputMask & (1u << attachment))
				populateClearColor(parameters, attachment, ctx.pass_color_format[attachment], *colorValue, allowFastLoad);
		}
	}
	parameters.depth = (float)clamp01(depthValue);

	MTLViewport viewport;
	viewport.originX = 0.0;
	viewport.originY = 0.0;
	viewport.width = (double)ctx.render_target_width;
	viewport.height = (double)ctx.render_target_height;
	viewport.znear = 0.0;
	viewport.zfar = 1.0;

	[encoder setRenderPipelineState:pipeline];
	[encoder setDepthStencilState:depthState];
	if (writeStencil) {
		const uint32_t reference = (uint32_t)stencilValue;
		[encoder setStencilFrontReferenceValue:reference backReferenceValue:reference];
	}
	[encoder setFrontFacingWinding:MTLWindingCounterClockwise];
	[encoder setCullMode:MTLCullModeNone];
	[encoder setTriangleFillMode:MTLTriangleFillModeFill];
	[encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
	[encoder setViewport:viewport];
	[encoder setScissorRect:scissorRect];
	if (colorOutputMask != 0 || writeDepth) {
		[encoder setFragmentBytes:&parameters length:sizeof(parameters) atIndex:0];
	}
	[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	// The clear covers the attachment rather than GL's viewport. Restore the GL
	// viewport/scissor immediately so following draws on this still-open encoder
	// keep their original rasterization rectangle.
	applyViewportAndScissor();

	// The clear draw bypasses draw.mm's encoder-state cache. Invalidate its shadow
	// state without ending the render pass; the next GL draw will rebind its pipeline,
	// depth state, viewport, buffers and resources on this same encoder.
	invalidateDrawEncoderStateCache();
	ctx.fragcoord_params_bound = false;
	if (ctx.pass_uses_default_framebuffer) {
		if (colorOutputMask != 0) ctx.default_color_has_contents = true;
		if (writeDepth) ctx.default_depth_has_contents = true;
		if (writeStencil) ctx.default_stencil_has_contents = true;
	}
}

void glClear(GLbitfield mask) {
	const GLbitfield validMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
	if (mask & ~validMask) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (mask == 0) return;
	ClearColorValue color;
	color.kind = ClearValueKind::Float;
	color.floatValues[0] = (float)ctx.clear_color.red;
	color.floatValues[1] = (float)ctx.clear_color.green;
	color.floatValues[2] = (float)ctx.clear_color.blue;
	color.floatValues[3] = (float)ctx.clear_color.alpha;
	performFramebufferClear(mask, -1, &color, (GLfloat)ctx.clear_depth, ctx.clear_stencil, true);
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
		performFramebufferClear(GL_STENCIL_BUFFER_BIT, -1, nullptr, (GLfloat)ctx.clear_depth, value[0], false);
	} else if (buffer == GL_COLOR) {
		if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
			gl4metalSetError(GL_INVALID_VALUE);
			return;
		}
		ClearColorValue color;
		color.kind = ClearValueKind::SignedInteger;
		for (int component = 0; component < 4; ++component) color.signedValues[component] = value[component];
		performFramebufferClear(GL_COLOR_BUFFER_BIT, drawbuffer, &color, (GLfloat)ctx.clear_depth, ctx.clear_stencil, false);
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
	}
}

void glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value) {
	if (buffer != GL_COLOR) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ClearColorValue color;
	color.kind = ClearValueKind::UnsignedInteger;
	for (int component = 0; component < 4; ++component) color.unsignedValues[component] = value[component];
	performFramebufferClear(GL_COLOR_BUFFER_BIT, drawbuffer, &color, (GLfloat)ctx.clear_depth, ctx.clear_stencil, false);
}

void glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value) {
	if (buffer == GL_COLOR) {
		if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
			gl4metalSetError(GL_INVALID_VALUE);
			return;
		}
		ClearColorValue color;
		color.kind = ClearValueKind::Float;
		for (int component = 0; component < 4; ++component) color.floatValues[component] = value[component];
		performFramebufferClear(GL_COLOR_BUFFER_BIT, drawbuffer, &color, (GLfloat)ctx.clear_depth, ctx.clear_stencil, false);
	} else if (buffer == GL_DEPTH) {
		performFramebufferClear(GL_DEPTH_BUFFER_BIT, -1, nullptr, value[0], ctx.clear_stencil, false);
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
	}
}

void glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
	(void)drawbuffer; // ignored for GL_DEPTH_STENCIL by OpenGL
	if (buffer != GL_DEPTH_STENCIL) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	performFramebufferClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, -1, nullptr, depth, stencil, false);
}

// glClearBuffer*: one draw buffer of the current framebuffer.
void clearOneColorBuffer(GLint drawbuffer, const GLfloat *rgba) {
	if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ClearColorValue color;
	color.kind = ClearValueKind::Float;
	for (int component = 0; component < 4; ++component) color.floatValues[component] = rgba[component];
	performFramebufferClear(GL_COLOR_BUFFER_BIT, drawbuffer, &color, (GLfloat)ctx.clear_depth, ctx.clear_stencil, false);
}

void clearDepthStencil(GLbitfield mask, GLfloat depth, GLint stencil) {
	performFramebufferClear(mask, -1, nullptr, depth, stencil, false);
}

// MARK: - Blit / read back


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

// Process-lifetime cache: the launcher may tear down JVM/GL threads while another thread is
// still unwinding a blit. A function-local static unordered_map is destroyed during C++ exit and
// can race that late call. Keep the registry allocated for process lifetime, matching the shader
// stage caches in program.mm. Key by device as well as format: MTL objects are device-specific.
struct BlitPipelineCacheKey {
	uintptr_t device_key = 0;
	NSUInteger pixel_format = 0;
	bool operator==(const BlitPipelineCacheKey &other) const {
		return device_key == other.device_key && pixel_format == other.pixel_format;
	}
};

struct BlitPipelineCacheKeyHash {
	size_t operator()(const BlitPipelineCacheKey &key) const {
		const size_t a = std::hash<uintptr_t>{}(key.device_key);
		const size_t b = std::hash<NSUInteger>{}(key.pixel_format);
		return a ^ (b + (size_t)0x9e3779b9u + (a << 6) + (a >> 2));
	}
};

struct BlitPipelineCache {
	std::mutex mutex;
	std::unordered_map<uintptr_t, id<MTLLibrary>> libraries;
	std::unordered_map<BlitPipelineCacheKey, id<MTLRenderPipelineState>, BlitPipelineCacheKeyHash> pipelines;
};

static BlitPipelineCache &blitPipelineCache() {
	static BlitPipelineCache *cache = new BlitPipelineCache();
	return *cache;
}

static id<MTLRenderPipelineState> blitPipeline(MTLPixelFormat format) {
	if (!ctx.device) return nil;
	BlitPipelineCache &cache = blitPipelineCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	const uintptr_t deviceKey = reinterpret_cast<uintptr_t>((__bridge void *)ctx.device);
	const BlitPipelineCacheKey key{deviceKey, (NSUInteger)format};

	auto it = cache.pipelines.find(key);
	if (it != cache.pipelines.end()) return it->second;

	id<MTLLibrary> library = nil;
	auto libraryIt = cache.libraries.find(deviceKey);
	if (libraryIt != cache.libraries.end()) {
		library = libraryIt->second;
	} else {
		NSError *libraryError = nil;
		library = [ctx.device newLibraryWithSource:blitSource() options:nil error:&libraryError];
		if (!library) {
			GL4METAL_LOG("glBlitFramebuffer: blit shader failed to compile: %s",
				libraryError.localizedDescription.UTF8String ?: "unknown Metal error");
			return nil;
		}
		cache.libraries.emplace(deviceKey, library);
	}

	NSError *error = nil;
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
	cache.pipelines.emplace(key, pipeline);
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
	auto framebufferIt = GL4METAL_FBO_FIND(framebuffer);
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
			auto sourceFbo = GL4METAL_FBO_FIND(ctx.read_fbo);
			if (sourceFbo != ctx.fbo_objects.end()) sourceDepthLevel = (NSUInteger)std::max(sourceFbo->second.depth.level, 0);
		}
		if (ctx.draw_fbo != 0) {
			auto destinationFbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
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
			auto sourceFbo = GL4METAL_FBO_FIND(ctx.read_fbo);
			if (sourceFbo != ctx.fbo_objects.end()) sourceStencilLevel = (NSUInteger)std::max(sourceFbo->second.stencil.level, 0);
		}
		if (ctx.draw_fbo != 0) {
			auto destinationFbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
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
	static std::mutex *samplerMutex = new std::mutex();
	std::lock_guard<std::mutex> lock(*samplerMutex);
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
	auto rit = GL4METAL_FBO_FIND(ctx.read_fbo);
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
		auto dit = GL4METAL_FBO_FIND(ctx.draw_fbo);
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
	auto it = GL4METAL_FBO_FIND(ctx.read_fbo);
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
	const bool srcA1BGR5 = tex.pixelFormat == MTLPixelFormatA1BGR5Unorm;
	switch (tex.pixelFormat) {
		case MTLPixelFormatR8Unorm:         srcComps = 1; srcIdx[0] = 0; break;
		case MTLPixelFormatRG8Unorm:        srcComps = 2; srcIdx[0] = 0; srcIdx[1] = 1; break;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB: srcComps = 4; srcIdx[0] = 0; srcIdx[1] = 1; srcIdx[2] = 2; srcIdx[3] = 3; break;
		case MTLPixelFormatBGRA8Unorm:      srcComps = 4; srcIdx[2] = 0; srcIdx[1] = 1; srcIdx[0] = 2; srcIdx[3] = 3; break;
		case MTLPixelFormatA1BGR5Unorm:    srcComps = 2; break;
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
			uint8_t packedRGBA[4] = { 0, 0, 0, 255 };
			if (srcA1BGR5) {
				uint16_t packed = 0;
				memcpy(&packed, src, sizeof(packed));
				packedRGBA[0] = (uint8_t)(((packed & 0x1fu) * 255u + 15u) / 31u);
				packedRGBA[1] = (uint8_t)((((packed >> 5) & 0x1fu) * 255u + 15u) / 31u);
				packedRGBA[2] = (uint8_t)((((packed >> 10) & 0x1fu) * 255u + 15u) / 31u);
				packedRGBA[3] = (packed & 0x8000u) ? 255u : 0u;
			}
			for (int c = 0; c < dstComps; c++) {
				const int channel = dstChannel[c];
				const int idx = srcIdx[channel];
				rowDst[c] = srcA1BGR5 ? packedRGBA[channel] :
					(idx >= 0 ? src[idx] : (channel == 3 ? 255 : 0));
			}
		}
	}
}
