//
// framebuffer.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//


#import "framebuffer.h"
#import "gl4metal.h"

extern void invalidateDrawEncoderStateCache(void);
extern void invalidateComputeEncoderStateCache(void);

#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
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

static bool textureCopyTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_TEXTURE_COPIES");
		return value && value[0] == '1';
	}();
	return enabled;
}

// Metal records work into encoders, GL into global state: a render pass is opened by glClear (or the first draw)
// for the current draw framebuffer and closed when the target changes or the frame is presented.

static bool renderPassTraceEnabled(void) {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_RENDER_PASSES");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool renderPassDetailTraceEnabled(void) {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_RENDER_PASS_DETAILS");
		return value && value[0] == '1';
	}();
	return enabled;
}

static void traceRenderPassAttachmentSet(const char *phase, const char *callerFile, int callerLine);

// The render encoder is reused across many Minecraft chunk draws. Re-encoding
// unchanged viewport/scissor state on every draw adds avoidable CPU encoder cost.
// Keep this cache per GL thread and tie it to the exact encoder; a new encoder
// always starts with an invalid cache, even if Metal reuses an object address.
struct ViewportScissorStateCache {
	const void *encoder_key = nullptr;
	bool viewport_valid = false;
	bool scissor_valid = false;
	MTLViewport viewport{};
	MTLScissorRect scissor{};
};

static thread_local ViewportScissorStateCache viewportScissorStateCache;

static void invalidateViewportAndScissorCache(void) {
	viewportScissorStateCache.encoder_key = nullptr;
	viewportScissorStateCache.viewport_valid = false;
	viewportScissorStateCache.scissor_valid = false;
}

static bool sameViewport(const MTLViewport &lhs, const MTLViewport &rhs) {
	return lhs.originX == rhs.originX && lhs.originY == rhs.originY &&
		lhs.width == rhs.width && lhs.height == rhs.height &&
		lhs.znear == rhs.znear && lhs.zfar == rhs.zfar;
}

static bool sameScissorRect(const MTLScissorRect &lhs, const MTLScissorRect &rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y &&
		lhs.width == rhs.width && lhs.height == rhs.height;
}

void endRenderEncoderAt(const char *callerFile, int callerLine) {
	// Metal encoders on one command buffer are sequential. Close a cached compute
	// encoder before any render-pass, blit, or resource-update path that uses this
	// common transition helper. Consecutive compute dispatches bypass this helper
	// when no render encoder is active and can therefore share one compute encoder.
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
		invalidateComputeEncoderStateCache();
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
		if (renderPassDetailTraceEnabled()) {
			traceRenderPassAttachmentSet("END", callerFile, callerLine);
		}
		[ctx.currentEncoder endEncoding];
		ctx.currentEncoder = nil;
	}
	invalidateViewportAndScissorCache();
	invalidateDrawEncoderStateCache();
	ctx.fragcoord_params_bound = false;
	ctx.discard_default_depth_stencil_on_end = false;
	ctx.pass_uses_default_framebuffer = false;
	ctx.pass_color_attachment_mask = 0;
	ctx.pass_discard_color_store_mask = 0;
	ctx.pass_discard_depth_store = false;
	ctx.pass_discard_stencil_store = false;
	ctx.pass_draw_fbo = UINT32_MAX;
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
		id<MTLTexture> baseTexture = object->second.texture;
		if ((NSUInteger)a.level >= baseTexture.mipmapLevelCount) return nil;
		if (a.layer >= 0) {
			if (baseTexture.textureType != MTLTextureType2DArray ||
				(NSUInteger)a.layer >= baseTexture.arrayLength) return nil;
			// The GL attachment references the texture object, not a particular
			// Metal allocation. Rebuild the single-layer view after glTexImage*
			// replaces the texture object's backing storage.
			if (!a.view || a.view_source != baseTexture) {
				a.view = [baseTexture newTextureViewWithPixelFormat:baseTexture.pixelFormat
					textureType:MTLTextureType2D levels:NSMakeRange(0, baseTexture.mipmapLevelCount)
					slices:NSMakeRange((NSUInteger)a.layer, 1)];
				a.view_source = a.view ? baseTexture : nil;
			}
			texture = a.view;
		} else {
			texture = baseTexture;
		}
	}
	if ((texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DMultisample) ||
		(texture.usage & MTLTextureUsageRenderTarget) == 0) return nil;
	if (texture.textureType == MTLTextureType2DMultisample && a.level != 0) return nil;
	return texture;
}

static void traceRenderPassAttachmentSet(const char *phase, const char *callerFile, int callerLine) {
	if (!renderPassTraceEnabled()) return;
	const GLuint passFbo = ctx.pass_draw_fbo;
	GL4METAL_LOG("render-pass %s fbo=%u serial=%llu caller=%s:%d size=%lux%lu samples=%lu colorMask=0x%X depthFormat=%lu stencilFormat=%lu",
		phase ? phase : "?", passFbo,
		(unsigned long long)ctx.render_pass_serial,
		callerFile ? callerFile : "?", callerLine,
		(unsigned long)ctx.render_target_width, (unsigned long)ctx.render_target_height,
		(unsigned long)ctx.pass_sample_count, (unsigned int)ctx.pass_color_attachment_mask,
		(unsigned long)ctx.pass_depth_format, (unsigned long)ctx.pass_stencil_format);

	if (passFbo == UINT32_MAX) return;
	if (passFbo == 0) {
		id<MTLTexture> colorTexture = ctx.currentDrawable.texture;
		GL4METAL_LOG("render-pass %s attachment=DEFAULT_COLOR texture=%p format=%lu size=%lux%lu",
			phase ? phase : "?", (__bridge void *)colorTexture,
			(unsigned long)colorTexture.pixelFormat, (unsigned long)colorTexture.width,
			(unsigned long)colorTexture.height);
		id<MTLTexture> depthTexture = ctx.depthTexture;
		GL4METAL_LOG("render-pass %s attachment=DEFAULT_DEPTH_STENCIL texture=%p format=%lu size=%lux%lu",
			phase ? phase : "?", (__bridge void *)depthTexture,
			(unsigned long)depthTexture.pixelFormat, (unsigned long)depthTexture.width,
			(unsigned long)depthTexture.height);
		return;
	}

	auto fboIt = GL4METAL_FBO_FIND(passFbo);
	if (fboIt == ctx.fbo_objects.end()) {
		GL4METAL_LOG("render-pass %s fbo=%u attachment-state=untracked", phase ? phase : "?", passFbo);
		return;
	}
	const FBO &fbo = fboIt->second;
	for (int logicalSlot = 0; logicalSlot < GL4METAL_MAX_COLOR_ATTACHMENTS; ++logicalSlot) {
		const GLenum drawBuffer = fbo.draw_buffers[logicalSlot];
		if (drawBuffer < GL_COLOR_ATTACHMENT0 || drawBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) continue;
		const FBOAttachment &attachment = fbo.color[drawBuffer - GL_COLOR_ATTACHMENT0];
		id<MTLTexture> texture = attachmentTexture(attachment);
		GL4METAL_LOG("render-pass %s color logical=%d drawBuffer=0x%X texture=%u renderbuffer=%u level=%d metal=%p format=%lu size=%lux%lu",
			phase ? phase : "?", logicalSlot, (unsigned int)drawBuffer,
			attachment.texture, attachment.renderbuffer, attachment.level,
			(__bridge void *)texture, (unsigned long)texture.pixelFormat,
			(unsigned long)texture.width, (unsigned long)texture.height);
	}
	const FBOAttachment &depthAttachment = fbo.depth;
	id<MTLTexture> depthTexture = attachmentTexture(depthAttachment);
	GL4METAL_LOG("render-pass %s depth texture=%u renderbuffer=%u level=%d metal=%p format=%lu size=%lux%lu",
		phase ? phase : "?", depthAttachment.texture, depthAttachment.renderbuffer,
		depthAttachment.level, (__bridge void *)depthTexture, (unsigned long)depthTexture.pixelFormat,
		(unsigned long)depthTexture.width, (unsigned long)depthTexture.height);
	const FBOAttachment &stencilAttachment = fbo.stencil;
	id<MTLTexture> stencilTexture = attachmentTexture(stencilAttachment);
	GL4METAL_LOG("render-pass %s stencil texture=%u renderbuffer=%u level=%d metal=%p format=%lu size=%lux%lu",
		phase ? phase : "?", stencilAttachment.texture, stencilAttachment.renderbuffer,
		stencilAttachment.level, (__bridge void *)stencilTexture, (unsigned long)stencilTexture.pixelFormat,
		(unsigned long)stencilTexture.width, (unsigned long)stencilTexture.height);
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
	std::string selectors = "FBO " + std::to_string(framebufferName) + " buffer selectors: read=0x";
	char readValue[16];
	snprintf(readValue, sizeof(readValue), "%04X", (unsigned int)fbo.read_buffer);
	selectors += readValue;
	selectors += " draw=[";
	for (int drawIndex = 0; drawIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++drawIndex) {
		if (drawIndex) selectors += ",";
		char drawValue[16];
		snprintf(drawValue, sizeof(drawValue), "%04X", (unsigned int)fbo.draw_buffers[drawIndex]);
		selectors += drawValue;
	}
	selectors += "]";
	GL4METAL_LOG("%s", selectors.c_str());
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

	const void *encoder_key = (__bridge const void *)ctx.currentEncoder;
	if (viewportScissorStateCache.encoder_key != encoder_key) {
		viewportScissorStateCache.encoder_key = encoder_key;
		viewportScissorStateCache.viewport_valid = false;
		viewportScissorStateCache.scissor_valid = false;
	}
	if (!viewportScissorStateCache.viewport_valid ||
		!sameViewport(viewportScissorStateCache.viewport, vp)) {
		[ctx.currentEncoder setViewport:vp];
		viewportScissorStateCache.viewport = vp;
		viewportScissorStateCache.viewport_valid = true;
	}

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
	if (!viewportScissorStateCache.scissor_valid ||
		!sameScissorRect(viewportScissorStateCache.scissor, sr)) {
		[ctx.currentEncoder setScissorRect:sr];
		viewportScissorStateCache.scissor = sr;
		viewportScissorStateCache.scissor_valid = true;
	}
}

static bool retargetOpenRenderPassForBoundFramebuffer(GLuint newFramebuffer, bool preservePassSerial);

id<MTLRenderCommandEncoder> beginRenderPass(GLbitfield clearMask, bool preservePassSerial) {
	// A GL framebuffer bind changes state; it does not itself execute rendering.
	// Keep the current Metal encoder alive across transient binds and decide only
	// when a draw/clear actually needs an encoder. This allows a sequence such as
	// A -> temporary B -> A to remain one physical Metal pass when no work targets B.
	// A non-zero clear mask still requires a fresh load-action pass.
	if (ctx.currentEncoder && clearMask == 0) {
		if (ctx.pass_draw_fbo == ctx.draw_fbo) {
			++ctx.frame_render_pass_reused_count;
			applyViewportAndScissor();
			return ctx.currentEncoder;
		}
		if (retargetOpenRenderPassForBoundFramebuffer(ctx.draw_fbo, preservePassSerial)) {
			applyViewportAndScissor();
			return ctx.currentEncoder;
		}
	}
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
	if (!gl4metalEnsureClassicCommandBuffer()) return nil;
	ctx.currentEncoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
	if (!ctx.currentEncoder) {
		GL4METAL_LOG("beginRenderPass: Metal could not create render encoder (FBO=%u, size=%lux%lu)",
			ctx.draw_fbo, (unsigned long)rtWidth, (unsigned long)rtHeight);
		return nil;
	}
	ctx.pass_draw_fbo = ctx.draw_fbo;
	++ctx.frame_render_pass_created_count;
	ctx.render_target_width = rtWidth;
	ctx.render_target_height = rtHeight;
	ctx.pass_uses_default_framebuffer = (ctx.draw_fbo == 0);
	if (renderPassDetailTraceEnabled()) {
		GL4METAL_LOG("render-pass CREATE fbo=%u clearMask=0x%X preserveSerial=%d", ctx.draw_fbo,
			(unsigned int)clearMask, preservePassSerial ? 1 : 0);
		traceRenderPassAttachmentSet("BEGIN", __FILE__, __LINE__);
	}
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
		GL4METAL_LOG("glInvalidateFramebuffer: target=0x%X references untracked framebuffer %u", target, framebufferName);
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
					if (ctx.currentEncoder && ctx.pass_draw_fbo == 0 && (ctx.pass_color_attachment_mask & 1u)) {
						ctx.pass_discard_color_store_mask |= 1u;
						activePassTouched = true;
					}
					break;
				case GL_DEPTH:
					ctx.default_depth_has_contents = false;
					if (ctx.currentEncoder && ctx.pass_draw_fbo == 0 && ctx.pass_depth_format != MTLPixelFormatInvalid) {
						ctx.pass_discard_depth_store = true;
						activePassTouched = true;
					}
					break;
				case GL_STENCIL:
					ctx.default_stencil_has_contents = false;
					if (ctx.currentEncoder && ctx.pass_draw_fbo == 0 && ctx.pass_stencil_format != MTLPixelFormatInvalid) {
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
				if (ctx.currentEncoder && ctx.pass_draw_fbo == framebufferName) {
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
				if (ctx.currentEncoder && ctx.pass_draw_fbo == framebufferName && ctx.pass_depth_format != MTLPixelFormatInvalid) {
					ctx.pass_discard_depth_store = true;
					activePassTouched = true;
				}
			}
			if (attachment == GL_STENCIL_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
				fbo.invalidated_stencil = true;
				if (ctx.currentEncoder && ctx.pass_draw_fbo == framebufferName && ctx.pass_stencil_format != MTLPixelFormatInvalid) {
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
		GL4METAL_LOG("glInvalidateSubFramebuffer: target=0x%X references untracked framebuffer %u", target, framebufferName);
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

// The public framebuffer status helper is defined below. During FBO binding we
// use it without setting GL errors so an invalid new target is still reported by
// the ordinary draw/clear operation, not prematurely by glBindFramebuffer.
static GLenum framebufferStatusForBoundTarget(GLenum target, bool checkColorBufferSelectors);

static bool samePhysicalAttachment(const FBOAttachment &leftAttachment, const FBOAttachment &rightAttachment) {
	const bool leftPresent = hasAttachment(leftAttachment);
	const bool rightPresent = hasAttachment(rightAttachment);
	if (!leftPresent || !rightPresent) return leftPresent == rightPresent;
	if (leftAttachment.level != rightAttachment.level) return false;
	id<MTLTexture> leftTexture = attachmentTexture(leftAttachment);
	id<MTLTexture> rightTexture = attachmentTexture(rightAttachment);
	return leftTexture && rightTexture && leftTexture == rightTexture;
}

static bool renderPassTransitionTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_RENDER_PASS_TRANSITIONS");
		return value && value[0] == '1';
	}();
	return enabled;
}

static void traceRenderPassRetarget(GLuint oldFramebuffer, GLuint newFramebuffer,
	bool reused, const char *reason) {
	if (!renderPassTransitionTraceEnabled()) return;
	static std::mutex *traceMutex = new std::mutex();
	static std::unordered_set<std::string> *reported = new std::unordered_set<std::string>();
	std::string key = std::to_string(oldFramebuffer) + ":" + std::to_string(newFramebuffer) +
		(reused ? ":reuse" : ":split:") + (reason ? reason : "unknown");
	bool firstReport = false;
	{
		std::lock_guard<std::mutex> lock(*traceMutex);
		firstReport = reported->insert(key).second;
	}
	if (firstReport) {
		GL4METAL_LOG("render-pass FBO transition: %u -> %u %s%s%s",
			oldFramebuffer, newFramebuffer, reused ? "reused compatible encoder" : "split encoder; ",
			(!reused && reason) ? "reason=" : "", (!reused && reason) ? reason : "");
	}
}

// Retarget an open Metal encoder when two GL FBO names resolve to exactly the
// same physical attachments in the same Metal color slots. FBO names themselves
// are not part of a Metal render-pass descriptor; texture identity, mip level,
// output-slot mapping, depth/stencil attachments and sample count are.
//
// This is deliberately conservative: it only reuses a complete user-FBO pass,
// never a drawable pass or a pass whose attachment signature cannot be proven
// identical. A later glClear with a non-zero mask still opens a fresh load-action pass.
static bool retargetOpenRenderPassForBoundFramebuffer(GLuint newFramebuffer, bool preservePassSerial) {
	if (!ctx.currentEncoder || ctx.pass_uses_default_framebuffer ||
		ctx.pass_draw_fbo == UINT32_MAX || ctx.pass_draw_fbo == 0 || newFramebuffer == 0 ||
		ctx.pass_draw_fbo == newFramebuffer) {
		return false;
	}

	const GLuint oldFramebuffer = ctx.pass_draw_fbo;
	++ctx.frame_render_pass_alias_candidates;
	const char *failureReason = nullptr;

	if (framebufferStatusForBoundTarget(GL_DRAW_FRAMEBUFFER, true) != GL_FRAMEBUFFER_COMPLETE) {
		failureReason = "new framebuffer is incomplete";
	} else {
		auto oldIt = GL4METAL_FBO_FIND(oldFramebuffer);
		auto newIt = GL4METAL_FBO_FIND(newFramebuffer);
		if (oldIt == ctx.fbo_objects.end() || newIt == ctx.fbo_objects.end()) {
			failureReason = "old or new framebuffer is untracked";
		} else {
			const FBO &oldFbo = oldIt->second;
			FBO &newFbo = newIt->second;
			uint32_t expectedColorMask = 0;
			NSUInteger expectedWidth = 0;
			NSUInteger expectedHeight = 0;
			NSUInteger expectedSamples = 1;
			bool haveExtent = false;

			for (int slot_index = 0; slot_index < GL4METAL_MAX_COLOR_ATTACHMENTS && !failureReason; ++slot_index) {
				const GLenum oldBuffer = oldFbo.draw_buffers[slot_index];
				const GLenum newBuffer = newFbo.draw_buffers[slot_index];
				const FBOAttachment *oldAttachment = nullptr;
				const FBOAttachment *newAttachment = nullptr;
				if (oldBuffer != GL_NONE) {
					if (oldBuffer < GL_COLOR_ATTACHMENT0 || oldBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
						failureReason = "old draw-buffer mapping is invalid";
						break;
					}
					oldAttachment = &oldFbo.color[oldBuffer - GL_COLOR_ATTACHMENT0];
				}
				if (newBuffer != GL_NONE) {
					if (newBuffer < GL_COLOR_ATTACHMENT0 || newBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
						failureReason = "new draw-buffer mapping is invalid";
						break;
					}
					newAttachment = &newFbo.color[newBuffer - GL_COLOR_ATTACHMENT0];
				}

				if (oldAttachment && newAttachment) {
					if (!samePhysicalAttachment(*oldAttachment, *newAttachment)) {
						failureReason = "color attachment image or mip level differs";
						break;
					}
					id<MTLTexture> colorTexture = attachmentTexture(*newAttachment);
					if (!colorTexture || ctx.pass_color_format[slot_index] != colorTexture.pixelFormat) {
						failureReason = "color attachment format differs from active pass";
						break;
					}
					expectedColorMask |= (1u << slot_index);
					if (!haveExtent) {
						expectedWidth = std::max<NSUInteger>(1, colorTexture.width >> (NSUInteger)newAttachment->level);
						expectedHeight = std::max<NSUInteger>(1, colorTexture.height >> (NSUInteger)newAttachment->level);
						expectedSamples = colorTexture.sampleCount;
						haveExtent = true;
					}
				} else if (oldAttachment || newAttachment || ctx.pass_color_format[slot_index] != MTLPixelFormatInvalid) {
					failureReason = "color attachment slot mapping differs";
					break;
				}
			}

			if (!failureReason && !samePhysicalAttachment(oldFbo.depth, newFbo.depth)) {
				failureReason = "depth attachment image or mip level differs";
			}
			id<MTLTexture> depthTexture = !failureReason ? attachmentTexture(newFbo.depth) : nil;
			if (!failureReason) {
				const MTLPixelFormat expectedDepthFormat = depthTexture ? depthTexture.pixelFormat : MTLPixelFormatInvalid;
				if (ctx.pass_depth_format != expectedDepthFormat) {
					failureReason = "depth attachment format differs";
				} else if (depthTexture) {
					if (!haveExtent) {
						expectedWidth = std::max<NSUInteger>(1, depthTexture.width >> (NSUInteger)newFbo.depth.level);
						expectedHeight = std::max<NSUInteger>(1, depthTexture.height >> (NSUInteger)newFbo.depth.level);
						expectedSamples = depthTexture.sampleCount;
						haveExtent = true;
					} else if (expectedSamples != depthTexture.sampleCount) {
						failureReason = "depth sample count differs";
					}
				}
			}

			if (!failureReason && !samePhysicalAttachment(oldFbo.stencil, newFbo.stencil)) {
				failureReason = "stencil attachment image or mip level differs";
			}
			id<MTLTexture> stencilTexture = !failureReason ? attachmentTexture(newFbo.stencil) : nil;
			const bool hasMetalStencil = stencilTexture &&
				(stencilTexture.pixelFormat == MTLPixelFormatStencil8 ||
				 stencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8);
			if (!failureReason) {
				const MTLPixelFormat expectedStencilFormat = hasMetalStencil ? stencilTexture.pixelFormat : MTLPixelFormatInvalid;
				if (ctx.pass_stencil_format != expectedStencilFormat) {
					failureReason = "stencil attachment format differs";
				} else if (hasMetalStencil) {
					if (!haveExtent) {
						expectedWidth = std::max<NSUInteger>(1, stencilTexture.width >> (NSUInteger)newFbo.stencil.level);
						expectedHeight = std::max<NSUInteger>(1, stencilTexture.height >> (NSUInteger)newFbo.stencil.level);
						expectedSamples = stencilTexture.sampleCount;
						haveExtent = true;
					} else if (expectedSamples != stencilTexture.sampleCount) {
						failureReason = "stencil sample count differs";
					}
				}
			}

			if (!failureReason && (!haveExtent || expectedColorMask != ctx.pass_color_attachment_mask ||
				expectedWidth != ctx.render_target_width || expectedHeight != ctx.render_target_height ||
				expectedSamples != ctx.pass_sample_count)) {
				failureReason = "attachment mask, extent or sample count differs";
			}

			if (!failureReason) {
				// An invalidation on the newly bound alias refers to the same image.
				// Consume it as a deferred DontCare store action without reopening the pass.
				for (int slot_index = 0; slot_index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++slot_index) {
					const GLenum drawBuffer = newFbo.draw_buffers[slot_index];
					if (drawBuffer < GL_COLOR_ATTACHMENT0 || drawBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) continue;
					const uint32_t attachmentBit = 1u << (drawBuffer - GL_COLOR_ATTACHMENT0);
					if (newFbo.invalidated_color_mask & attachmentBit) {
						ctx.pass_discard_color_store_mask |= (1u << slot_index);
						newFbo.invalidated_color_mask &= ~attachmentBit;
					}
				}
				if (depthTexture && newFbo.invalidated_depth) {
					ctx.pass_discard_depth_store = true;
					newFbo.invalidated_depth = false;
				}
				if (hasMetalStencil && newFbo.invalidated_stencil) {
					ctx.pass_discard_stencil_store = true;
					newFbo.invalidated_stencil = false;
				}

				ctx.pass_draw_fbo = newFramebuffer;
				if (!preservePassSerial) {
					++ctx.render_pass_serial; // new logical GL pass, same physical Metal attachment pass
					if (ctx.render_pass_serial == 0) ++ctx.render_pass_serial;
				}
				++ctx.frame_render_pass_alias_reuses;
				traceRenderPassRetarget(oldFramebuffer, newFramebuffer, true, nullptr);
				return true;
			}
		}
	}

	++ctx.frame_render_pass_alias_misses;
	traceRenderPassRetarget(oldFramebuffer, newFramebuffer, false, failureReason ? failureReason : "unknown mismatch");
	return false;
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
			// Defer render-encoder retarget/split until a draw or clear. Binding an
			// intermediate FBO that receives no rendering must not flush the active pass.
			ctx.draw_fbo = framebuffer;
			ctx.read_fbo = framebuffer;
			break;
		case GL_DRAW_FRAMEBUFFER:
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

// Check the currently bound framebuffer for the requested operation. The GL 3.3
// public status query validates draw/read color-buffer selectors. A depth/stencil-only
// blit, however, never reads or writes those selectors: on macOS' GL 4.1 path used by
// Minecraft, selector-based completeness checks are relaxed, and Mojang's temporary
// depth-copy FBOs intentionally leave their default read buffer at COLOR_ATTACHMENT0.
// Keep glCheckFramebufferStatus strict, but do not reject a valid depth/stencil copy
// solely because its unused color selector has no color attachment.
static GLenum framebufferStatusForBoundTarget(GLenum target, bool checkColorBufferSelectors) {
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
	if (checkColorBufferSelectors) {
		// Validate only the selector used by this framebuffer target.
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
	}
	return GL_FRAMEBUFFER_COMPLETE;
}

GLenum glCheckFramebufferStatus(GLenum target) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glCheckFramebufferStatus: invalid target 0x%X", target);
		return 0;
	}
	return framebufferStatusForBoundTarget(target, true);
}

// MARK: - Framebuffer attachments

void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
	GLenum imageTarget = GL_TEXTURE_2D;
	if (texture != 0) {
		auto object = ctx.texture_objects.find(texture);
		if (object != ctx.texture_objects.end() && object->second.target == GL_TEXTURE_2D_MULTISAMPLE)
			imageTarget = GL_TEXTURE_2D_MULTISAMPLE;
	}
	glFramebufferTexture2D(target, attachment, imageTarget, texture, level);
}

void glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (textarget != GL_TEXTURE_1D) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (level < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const bool validAttachment =
		(attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) ||
		attachment == GL_DEPTH_ATTACHMENT || attachment == GL_STENCIL_ATTACHMENT ||
		attachment == GL_DEPTH_STENCIL_ATTACHMENT;
	if (!validAttachment) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	// Detaching is supported even though a live 1D texture cannot be used as
	// a Metal render-pass attachment. Keep GL object/FBO state consistent.
	if (texture == 0) {
		glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, 0, level);
		return;
	}
	auto textureIt = ctx.texture_objects.find(texture);
	if (textureIt == ctx.texture_objects.end() || textureIt->second.target != GL_TEXTURE_1D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glFramebufferTexture1D: 1D render-target attachments are not supported by iOS Metal");
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
	if (texture != 0 && textarget != GL_TEXTURE_2D && textarget != GL_TEXTURE_2D_MULTISAMPLE) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glFramebufferTexture2D: unsupported textarget 0x%X", textarget);
		return;
	}
	if (texture != 0) {
		auto object = ctx.texture_objects.find(texture);
		if (object == ctx.texture_objects.end() || object->second.target != textarget) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glFramebufferTexture2D: texture %u target 0x%X does not match attachment target 0x%X",
				texture, object == ctx.texture_objects.end() ? 0 : object->second.target, textarget);
			return;
		}
		if (textarget == GL_TEXTURE_2D_MULTISAMPLE && level != 0) {
			gl4metalSetError(GL_INVALID_VALUE);
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
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (textarget != GL_TEXTURE_3D) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (level < 0 || zoffset < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const bool validAttachment =
		(attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) ||
		attachment == GL_DEPTH_ATTACHMENT || attachment == GL_STENCIL_ATTACHMENT ||
		attachment == GL_DEPTH_STENCIL_ATTACHMENT;
	if (!validAttachment) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (texture == 0) {
		glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, 0, level);
		return;
	}
	auto textureIt = ctx.texture_objects.find(texture);
	if (textureIt == ctx.texture_objects.end() || textureIt->second.target != GL_TEXTURE_3D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	// Metal texture views cannot reinterpret a 3D volume as a 2D render-target
	// slice. glFramebufferTextureLayer on GL_TEXTURE_2D_ARRAY is the supported
	// layered-render-target path on iPhone/iPad.
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glFramebufferTexture3D: a 3D-volume slice cannot be attached as a 2D Metal render target; use GL_TEXTURE_2D_ARRAY");
}

void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
	if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (level < 0 || layer < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const GLuint framebufferName = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (framebufferName == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const bool colorAttachment = attachment >= GL_COLOR_ATTACHMENT0 &&
		attachment < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS;
	const bool depthStencilAttachment = attachment == GL_DEPTH_ATTACHMENT ||
		attachment == GL_STENCIL_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT;
	if (!colorAttachment && !depthStencilAttachment) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	FBOAttachment newAttachment;
	newAttachment.texture = texture;
	newAttachment.level = level;
	newAttachment.layer = layer;
	if (texture != 0) {
		auto textureIt = ctx.texture_objects.find(texture);
		if (textureIt == ctx.texture_objects.end() || !textureIt->second.texture ||
			textureIt->second.target != GL_TEXTURE_2D_ARRAY ||
			textureIt->second.texture.textureType != MTLTextureType2DArray) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glFramebufferTextureLayer: texture %u is not an allocated GL_TEXTURE_2D_ARRAY", texture);
			return;
		}
		id<MTLTexture> baseTexture = textureIt->second.texture;
		if ((NSUInteger)level >= baseTexture.mipmapLevelCount || (NSUInteger)layer >= baseTexture.arrayLength) {
			gl4metalSetError(GL_INVALID_VALUE);
			return;
		}
		if (!attachmentTexture(newAttachment)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glFramebufferTextureLayer: Metal could not create a renderable 2D view for texture %u layer %d", texture, layer);
			return;
		}
	}
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
		case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
			*params = (a && a->texture && a->layer >= 0) ? a->layer : 0;
			break;
		case GL_FRAMEBUFFER_ATTACHMENT_LAYERED:
			// The backend supports explicit single-layer views only; whole-array
			// layered rendering (gl_Layer) is not implemented by the shader path.
			*params = GL_FALSE;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetFramebufferAttachmentParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

// MARK: - Draw / read buffers

// The legacy Metal render-pass API used by this backend fixes its color
// attachment layout when the encoder is created. A glDrawBuffers change can
// therefore keep the current encoder only when every logical output slot still
// resolves to the exact same Metal image and mip level. This commonly catches
// redundant routing changes involving aliased GL attachment points without
// relying on the Metal 4 color-attachment-mapping pipeline state.
static bool drawBufferSelectionMatchesOpenMetalPass(const FBO &fbo,
	GLsizei requestedCount, const GLenum *requestedBuffers) {
	if (!ctx.currentEncoder || ctx.pass_draw_fbo != ctx.draw_fbo || ctx.draw_fbo == 0) {
		return false;
	}

	for (int logicalSlot = 0; logicalSlot < GL4METAL_MAX_COLOR_ATTACHMENTS; ++logicalSlot) {
		const GLenum oldSelector = fbo.draw_buffers[logicalSlot];
		const GLenum newSelector = logicalSlot < requestedCount
			? requestedBuffers[logicalSlot] : GL_NONE;

		id<MTLTexture> oldTexture = nil;
		id<MTLTexture> newTexture = nil;
		GLint oldLevel = 0;
		GLint newLevel = 0;

		if (oldSelector >= GL_COLOR_ATTACHMENT0 &&
			oldSelector < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
			const FBOAttachment &oldAttachment = fbo.color[oldSelector - GL_COLOR_ATTACHMENT0];
			oldTexture = attachmentTexture(oldAttachment);
			oldLevel = oldAttachment.level;
		}
		if (newSelector >= GL_COLOR_ATTACHMENT0 &&
			newSelector < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
			const FBOAttachment &newAttachment = fbo.color[newSelector - GL_COLOR_ATTACHMENT0];
			newTexture = attachmentTexture(newAttachment);
			newLevel = newAttachment.level;
		}

		const uint32_t slotBit = 1u << logicalSlot;
		const bool passHasAttachment = (ctx.pass_color_attachment_mask & slotBit) != 0;
		if (oldTexture != newTexture || (oldTexture && oldLevel != newLevel)) {
			return false;
		}
		if ((oldTexture != nil) != passHasAttachment) {
			return false;
		}
		if (oldTexture && ctx.pass_color_format[logicalSlot] != oldTexture.pixelFormat) {
			return false;
		}
	}
	return true;
}

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
	// Repeated glDrawBuffers calls are common in shader packs. Do not split a live
	// render pass when the complete logical-to-physical routing is unchanged.
	bool drawBufferMappingChanged = false;
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; ++i) {
		const GLenum requested = (i < n) ? bufs[i] : GL_NONE;
		if (it->second.draw_buffers[i] != requested) drawBufferMappingChanged = true;
	}
	// Draw buffer i writes to the colour attachment named by bufs[i] (GL_NONE: not written).
	// Preserve an open legacy Metal encoder only when the new routing resolves to
	// exactly the same attachment objects at the same logical output slots. A
	// different target set/order still requires a new render pass.
	const bool openPassTargetsUnchanged = drawBufferMappingChanged &&
		drawBufferSelectionMatchesOpenMetalPass(it->second, n, bufs);
	if (drawBufferMappingChanged) {
		for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
			it->second.draw_buffers[i] = (i < n) ? bufs[i] : GL_NONE;
		}
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
	// Do not tear down an encoder that belongs to a previously bound FBO: r4
	// deliberately defers transient framebuffer binds until the next draw.
	// Likewise, changing draw-buffer state before a render pass exists must not
	// flush an unrelated compute encoder. If the active pass belongs to this FBO,
	// end it only when the physical attachment layout really changed.
	if (drawBufferMappingChanged && ctx.currentEncoder &&
		ctx.pass_draw_fbo == ctx.draw_fbo && !openPassTargetsUnchanged) {
		endRenderEncoder();
	} else if (drawBufferTraceEnabled() && drawBufferMappingChanged) {
		const char *reason = !ctx.currentEncoder ? "no-active-render-encoder" :
			(ctx.pass_draw_fbo != ctx.draw_fbo ? "encoder-belongs-to-other-fbo" :
			(openPassTargetsUnchanged ? "same-physical-attachment-layout" : "deferred"));
		GL4METAL_LOG("glDrawBuffers: kept encoder state (%s) fbo=%u activeFbo=%u",
			reason, ctx.draw_fbo, ctx.pass_draw_fbo);
	}
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
	if (std::isnan(v)) return 0.0;
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

	// MTLLoadActionClear is fixed when an encoder is created. If a compatible pass
	// is already open, clear inside that pass with the clear triangle instead of
	// ending it solely to request a new load action.
	const GLbitfield fastMask = (allowFastLoad && drawbuffer < 0 && !ctx.currentEncoder)
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
	invalidateViewportAndScissorCache();
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
	if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
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
	if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
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
	if (!value) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (buffer == GL_COLOR) {
		if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
			gl4metalSetError(GL_INVALID_VALUE);
			return;
		}
		ClearColorValue color;
		color.kind = ClearValueKind::Float;
		for (int component = 0; component < 4; ++component)
			color.floatValues[component] = std::isnan(value[component]) ? 0.0f : value[component];
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
	if (!rgba) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (drawbuffer < 0 || drawbuffer >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ClearColorValue color;
	color.kind = ClearValueKind::Float;
	for (int component = 0; component < 4; ++component)
		color.floatValues[component] = std::isnan(rgba[component]) ? 0.0f : rgba[component];
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
	NSUInteger color_formats[GL4METAL_MAX_COLOR_ATTACHMENTS] = {};
	NSUInteger depth_format = 0;
	NSUInteger stencil_format = 0;
	NSUInteger sample_count = 1;
	bool operator==(const BlitPipelineCacheKey &other) const {
		if (device_key != other.device_key || depth_format != other.depth_format ||
			stencil_format != other.stencil_format || sample_count != other.sample_count) return false;
		for (int index = 0; index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++index) {
			if (color_formats[index] != other.color_formats[index]) return false;
		}
		return true;
	}
};

struct BlitPipelineCacheKeyHash {
	size_t operator()(const BlitPipelineCacheKey &key) const {
		size_t hash = std::hash<uintptr_t>{}(key.device_key);
		auto combine = [&hash](size_t value) {
			hash ^= value + (size_t)0x9e3779b9u + (hash << 6) + (hash >> 2);
		};
		for (int index = 0; index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++index)
			combine(std::hash<NSUInteger>{}(key.color_formats[index]));
		combine(std::hash<NSUInteger>{}(key.depth_format));
		combine(std::hash<NSUInteger>{}(key.stencil_format));
		combine(std::hash<NSUInteger>{}(key.sample_count));
		return hash;
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

// Compile against the full attachment signature of the render pass. This permits
// the shader blit to run inside an already-open GL render encoder (notably the
// drawable pass), rather than allocating a separate render pass for every blit.
static id<MTLRenderPipelineState> blitPipeline(MTLPixelFormat colorFormat,
	MTLPixelFormat depthFormat, MTLPixelFormat stencilFormat, NSUInteger sampleCount) {
	if (!ctx.device || colorFormat == MTLPixelFormatInvalid) return nil;
	BlitPipelineCache &cache = blitPipelineCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	const uintptr_t deviceKey = reinterpret_cast<uintptr_t>((__bridge void *)ctx.device);
	BlitPipelineCacheKey key{};
	key.device_key = deviceKey;
	key.color_formats[0] = (NSUInteger)colorFormat;
	// A blit writes only logical color output zero. Remaining pass attachments are
	// still described in the pipeline signature but use no fragment color output.
	for (int index = 1; index < GL4METAL_MAX_COLOR_ATTACHMENTS; ++index)
		key.color_formats[index] = (NSUInteger)MTLPixelFormatInvalid;
	key.depth_format = (NSUInteger)depthFormat;
	key.stencil_format = (NSUInteger)stencilFormat;
	key.sample_count = std::max<NSUInteger>(1, sampleCount);

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
	MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
	descriptor.vertexFunction = [library newFunctionWithName:@"blit_vs"];
	descriptor.fragmentFunction = [library newFunctionWithName:@"blit_fs"];
	if (!descriptor.vertexFunction || !descriptor.fragmentFunction) {
		GL4METAL_LOG("glBlitFramebuffer: failed to resolve blit shader entry points");
		return nil;
	}
	descriptor.colorAttachments[0].pixelFormat = colorFormat;
	descriptor.colorAttachments[0].writeMask = MTLColorWriteMaskAll;	descriptor.colorAttachments[0].blendingEnabled = NO;
	descriptor.depthAttachmentPixelFormat = depthFormat;
	descriptor.stencilAttachmentPixelFormat = stencilFormat;
	descriptor.rasterSampleCount = key.sample_count;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:descriptor error:&error];
	if (!pipeline) {
		GL4METAL_LOG("glBlitFramebuffer: blit pipeline color=%d depth=%d stencil=%d samples=%lu failed: %s",
			(int)colorFormat, (int)depthFormat, (int)stencilFormat, (unsigned long)key.sample_count,
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

		if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(source, 0, sourceLevel,
			MTLOriginMake(sourceX, sourceY, 0), MTLSizeMake(width, height, 1),
			destination, 0, destinationLevel, MTLOriginMake(destinationX, destinationY, 0))) {
			return true;
		}

		if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
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
	const MTLBlitOption aspectOption = depthPlane ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;

	if (ctx.metal4Enabled &&
		gl4metalMetal4CopyTextureToBuffer(source, 0, sourceLevel,
			MTLOriginMake(sourceX, sourceY, 0), MTLSizeMake(width, height, 1),
			scratch, 0, rowBytes, 0, aspectOption) &&
		gl4metalMetal4CopyBufferToTextureOptions(scratch, 0, rowBytes, 0,
			MTLSizeMake(width, height, 1), destination, 0, destinationLevel,
			MTLOriginMake(destinationX, destinationY, 0), aspectOption)) {
		return true;
	}

	if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
	id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
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
	if (ctx.read_fbo != 0) {
		const GLenum readStatus = framebufferStatusForBoundTarget(GL_READ_FRAMEBUFFER,
			(mask & GL_COLOR_BUFFER_BIT) != 0);
		if (readStatus != GL_FRAMEBUFFER_COMPLETE) {
			gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: depth/stencil READ framebuffer %u incomplete (status=0x%X)",
				ctx.read_fbo, readStatus);
			auto readFbo = GL4METAL_FBO_FIND(ctx.read_fbo);
			if (readFbo != ctx.fbo_objects.end()) logFBOAttachmentStates(ctx.read_fbo, readFbo->second);
			return false;
		}
	}
	if (ctx.draw_fbo != 0) {
		const GLenum drawStatus = framebufferStatusForBoundTarget(GL_DRAW_FRAMEBUFFER,
			(mask & GL_COLOR_BUFFER_BIT) != 0);
		if (drawStatus != GL_FRAMEBUFFER_COMPLETE) {
			gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: depth/stencil DRAW framebuffer %u incomplete (status=0x%X)",
				ctx.draw_fbo, drawStatus);
			auto drawFbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
			if (drawFbo != ctx.fbo_objects.end()) logFBOAttachmentStates(ctx.draw_fbo, drawFbo->second);
			return false;
		}
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
		const NSUInteger sourceMetalY = blitMetalY(ctx.read_fbo, sourceDepth, sourceDepthLevel, srcY0, height);
		const NSUInteger destinationMetalY = blitMetalY(ctx.draw_fbo, destinationDepth, destinationDepthLevel, dstY0, height);

		if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(sourceDepth, 0, sourceDepthLevel,
			MTLOriginMake((NSUInteger)srcX0, sourceMetalY, 0), MTLSizeMake(width, height, 1),
			destinationDepth, 0, destinationDepthLevel,
			MTLOriginMake((NSUInteger)dstX0, destinationMetalY, 0))) {
			markDefaultDepthStencilContents(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			return true;
		}

		if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
		id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
		if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
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
	// glBlitFramebuffer is one of the few operations for which both READ and DRAW
	// framebuffer completeness are mandatory. Validate color-only blits too; the
	// old implementation checked completeness only in the depth/stencil helper,
	// allowing a broken temporary FBO to fail later with an opaque GL error.
	const bool blitUsesColorBuffers = (mask & GL_COLOR_BUFFER_BIT) != 0;
	if (ctx.read_fbo != 0) {
		const GLenum readStatus = framebufferStatusForBoundTarget(GL_READ_FRAMEBUFFER, blitUsesColorBuffers);
		if (readStatus != GL_FRAMEBUFFER_COMPLETE) {
			gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: READ framebuffer %u incomplete (status=0x%X)", ctx.read_fbo, readStatus);
			auto readFbo = GL4METAL_FBO_FIND(ctx.read_fbo);
			if (readFbo != ctx.fbo_objects.end()) logFBOAttachmentStates(ctx.read_fbo, readFbo->second);
			return;
		}
	}
	if (ctx.draw_fbo != 0) {
		const GLenum drawStatus = framebufferStatusForBoundTarget(GL_DRAW_FRAMEBUFFER, blitUsesColorBuffers);
		if (drawStatus != GL_FRAMEBUFFER_COMPLETE) {
			gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
			GL4METAL_LOG("glBlitFramebuffer: DRAW framebuffer %u incomplete (status=0x%X)", ctx.draw_fbo, drawStatus);
			auto drawFbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
			if (drawFbo != ctx.fbo_objects.end()) logFBOAttachmentStates(ctx.draw_fbo, drawFbo->second);
			return;
		}
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
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: reading the default framebuffer is not supported (the drawable is framebuffer-only)");
		return;
	}
	auto rit = GL4METAL_FBO_FIND(ctx.read_fbo);
	if (rit == ctx.fbo_objects.end() || rit->second.read_buffer < GL_COLOR_ATTACHMENT0 ||
		rit->second.read_buffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
		return;
	}
	const FBOAttachment &sa = rit->second.color[rit->second.read_buffer - GL_COLOR_ATTACHMENT0];
	const GLuint sourceTextureName = sa.texture;
	const GLuint sourceRenderbufferName = sa.renderbuffer;
	id<MTLTexture> src = attachmentTexture(sa);
	if (!src) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: source color attachment has no usable Metal texture");
		return;
	}
	float srcW = (float)std::max<NSUInteger>(src.width >> sa.level, 1);
	float srcH = (float)std::max<NSUInteger>(src.height >> sa.level, 1);

	// Destination: first draw buffer of the draw framebuffer. Keep a compatible
	// render encoder open; exact Metal blits and non-mergeable shader blits end it
	// only when their own encoder is actually required.
	bool dstDefault = (ctx.draw_fbo == 0);
	id<MTLTexture> dst = nil;
	NSUInteger dstLevel = 0;
	GLenum destinationDrawBuffer = GL_NONE;
	GLuint destinationTextureName = 0;
	GLuint destinationRenderbufferName = 0;
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
		destinationDrawBuffer = buf;
		const FBOAttachment &da = dit->second.color[buf - GL_COLOR_ATTACHMENT0];
		destinationTextureName = da.texture;
		destinationRenderbufferName = da.renderbuffer;
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
	auto markDestinationColorWritten = [&]() {
		if (dstDefault) {
			ctx.default_color_has_contents = true;
			return;
		}
		auto destinationFbo = GL4METAL_FBO_FIND(ctx.draw_fbo);
		if (destinationFbo != ctx.fbo_objects.end() &&
			destinationDrawBuffer >= GL_COLOR_ATTACHMENT0 &&
			destinationDrawBuffer < GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) {
			const uint32_t attachmentBit = 1u << (destinationDrawBuffer - GL_COLOR_ATTACHMENT0);
			destinationFbo->second.invalidated_color_mask &= ~attachmentBit;
		}
	};

	// Minecraft 1.21.11 exposes GlCommandEncoder::copyTextureToTexture and owns
	// two temporary FBOs for copy work. For an exact equal-size color copy, use a
	// Metal blit encoder directly: no sampling pipeline,
	// sampler state, fragment draw, or ShaderRead usage is needed.
	const int64_t sourceWidth = (int64_t)srcX1 - (int64_t)srcX0;
	const int64_t sourceHeight = (int64_t)srcY1 - (int64_t)srcY0;
	const int64_t destinationWidth = (int64_t)dstX1 - (int64_t)dstX0;
	const int64_t destinationHeight = (int64_t)dstY1 - (int64_t)dstY0;
	const NSUInteger sourceLevel = (NSUInteger)sa.level;
	const NSUInteger sourceLevelWidth = std::max<NSUInteger>(1, src.width >> sourceLevel);
	const NSUInteger sourceLevelHeight = std::max<NSUInteger>(1, src.height >> sourceLevel);
	const NSUInteger destinationLevelWidth = std::max<NSUInteger>(1, dst.width >> dstLevel);
	const NSUInteger destinationLevelHeight = std::max<NSUInteger>(1, dst.height >> dstLevel);
	const bool exactCopy = mask == GL_COLOR_BUFFER_BIT && filter == GL_NEAREST &&
		ctx.read_fbo != 0 && !dstDefault && src != dst &&
		src.textureType == MTLTextureType2D && dst.textureType == MTLTextureType2D &&
		src.sampleCount == 1 && dst.sampleCount == 1 && src.pixelFormat == dst.pixelFormat &&
		sourceWidth > 0 && sourceHeight > 0 &&
		sourceWidth == destinationWidth && sourceHeight == destinationHeight &&
		srcX0 >= 0 && srcY0 >= 0 && dstX0 >= 0 && dstY0 >= 0 &&
		(uint64_t)srcX1 <= sourceLevelWidth && (uint64_t)srcY1 <= sourceLevelHeight &&
		(uint64_t)dstX1 <= destinationLevelWidth && (uint64_t)dstY1 <= destinationLevelHeight;
	if (exactCopy) {
		if (textureCopyTraceEnabled()) {
			GL4METAL_LOG("glBlitFramebuffer: native texture-copy fast path readFBO=%u drawFBO=%u srcTexture=%u srcRenderbuffer=%u dstTexture=%u dstRenderbuffer=%u level=%lu->%lu rect=%d,%d %lldx%lld -> %d,%d",
				ctx.read_fbo, ctx.draw_fbo, sourceTextureName, sourceRenderbufferName,
				destinationTextureName, destinationRenderbufferName,
				(unsigned long)sourceLevel, (unsigned long)dstLevel, srcX0, srcY0,
				(long long)sourceWidth, (long long)sourceHeight, dstX0, dstY0);
		}
		endRenderEncoder();

		if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(src, 0, sourceLevel,
			MTLOriginMake((NSUInteger)srcX0, (NSUInteger)srcY0, 0),
			MTLSizeMake((NSUInteger)sourceWidth, (NSUInteger)sourceHeight, 1),
			dst, 0, dstLevel, MTLOriginMake((NSUInteger)dstX0, (NSUInteger)dstY0, 0))) {
			markDestinationColorWritten();
			return;
		}

		if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
		if (!ctx.currentCommandBuffer) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
		id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
		if (!copyEncoder) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
		const NSUInteger sourceMetalY = (NSUInteger)srcY0;
		const NSUInteger destinationMetalY = (NSUInteger)dstY0;
		[copyEncoder copyFromTexture:src sourceSlice:0 sourceLevel:sourceLevel
			sourceOrigin:MTLOriginMake((NSUInteger)srcX0, sourceMetalY, 0)
			sourceSize:MTLSizeMake((NSUInteger)sourceWidth, (NSUInteger)sourceHeight, 1)
			toTexture:dst destinationSlice:0 destinationLevel:dstLevel
			destinationOrigin:MTLOriginMake((NSUInteger)dstX0, destinationMetalY, 0)];
		[copyEncoder endEncoding];
		markDestinationColorWritten();
		return;
	}

	if (textureCopyTraceEnabled()) {
		GL4METAL_LOG("glBlitFramebuffer: shader blit path readFBO=%u drawFBO=%u srcFormat=%d dstFormat=%d filter=0x%X rect=%d,%d-%d,%d -> %d,%d-%d,%d",
			ctx.read_fbo, ctx.draw_fbo, (int)src.pixelFormat, (int)dst.pixelFormat, filter,
			srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);
	}
	if ((src.usage & MTLTextureUsageShaderRead) == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBlitFramebuffer: non-exact color blit requires a ShaderRead source texture");
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

	id<MTLRenderCommandEncoder> enc = nil;
	bool mergedIntoGLPass = false;
	if (dstDefault) {
		// The default drawable is by far the most common repeated blit destination
		// (Minecraft's final post-process/composite chain). Reuse its active render
		// encoder so each shader conversion does not become a separate render pass.
		enc = (ctx.currentEncoder && ctx.pass_draw_fbo == ctx.draw_fbo)
			? ctx.currentEncoder : beginRenderPass(0);
		if (enc && ctx.pass_uses_default_framebuffer &&
			ctx.pass_color_format[0] == dst.pixelFormat && ctx.pass_draw_fbo == 0) {
			mergedIntoGLPass = true;
		} else {
			if (enc) endRenderEncoder();
			enc = nil;
		}
	}

	if (!mergedIntoGLPass) {
		// Different attachment sets still require their own pass. Do not try to
		// merge these into the current GL encoder.
		endRenderEncoder();
		if (!gl4metalEnsureClassicCommandBuffer()) return;
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = dst;
		rp.colorAttachments[0].level = dstLevel;
		// A newly acquired drawable has undefined contents. Loading it before the
		// first write is both unnecessary and potentially undefined on tile GPUs.
		rp.colorAttachments[0].loadAction =
			(dstDefault && !ctx.default_color_has_contents) ? MTLLoadActionDontCare : MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		enc = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
		if (!enc) {
			GL4METAL_LOG("glBlitFramebuffer: failed to create Metal render encoder");
			return;
		}
		++ctx.frame_render_pass_created_count;
		if (renderPassDetailTraceEnabled()) {
			GL4METAL_LOG("render-pass BEGIN source=glBlitFramebuffer standalone readFBO=%u drawFBO=%u srcTexture=%u srcRenderbuffer=%u srcLevel=%d srcMetal=%p srcFormat=%lu dstTexture=%u dstRenderbuffer=%u dstLevel=%lu dstMetal=%p dstFormat=%lu",
				ctx.read_fbo, ctx.draw_fbo, sourceTextureName, sourceRenderbufferName, sa.level,
				(__bridge void *)src, (unsigned long)src.pixelFormat,
				destinationTextureName, destinationRenderbufferName, (unsigned long)dstLevel,
				(__bridge void *)dst, (unsigned long)dst.pixelFormat);
		}
	}

	id<MTLRenderPipelineState> pipeline = mergedIntoGLPass
		? blitPipeline(dst.pixelFormat, ctx.pass_depth_format, ctx.pass_stencil_format, ctx.pass_sample_count)
		: blitPipeline(dst.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid, 1);
	if (!pipeline) {
		if (!mergedIntoGLPass) [enc endEncoding];
		return;
	}
	id<MTLSamplerState> sampler = blitSampler(filter == GL_LINEAR);
	if (!sampler) {
		if (!mergedIntoGLPass) [enc endEncoding];
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
	// glBlitFramebuffer ignores the application's depth/stencil test and raster state.
	// The standalone path already had no depth attachment; explicitly neutralize
	// those states when sharing the default framebuffer's live encoder.
	[enc setDepthStencilState:ctx.no_depth_state];
	[enc setCullMode:MTLCullModeNone];
	[enc setTriangleFillMode:MTLTriangleFillModeFill];
	[enc setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
	invalidateViewportAndScissorCache();
	[enc setViewport:vp];
	[enc setScissorRect:sr];
	[enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
	[enc setFragmentTexture:src atIndex:0];
	[enc setFragmentSamplerState:sampler atIndex:0];
	[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	if (mergedIntoGLPass) {
		++ctx.frame_merged_blit_draw_count;
		// This draw changed Metal state behind draw.mm's shadow cache. Restore the
		// GL viewport/scissor and force the next GL draw to rebind its full state.
		applyViewportAndScissor();
		invalidateDrawEncoderStateCache();
		ctx.fragcoord_params_bound = false;
		ctx.pass_discard_color_store_mask &= ~1u;
	} else {
		if (renderPassDetailTraceEnabled()) {
			GL4METAL_LOG("render-pass END source=glBlitFramebuffer standalone readFBO=%u drawFBO=%u dstMetal=%p dstFormat=%lu",
				ctx.read_fbo, ctx.draw_fbo, (__bridge void *)dst, (unsigned long)dst.pixelFormat);
		}
		[enc endEncoding];
	}
	markDestinationColorWritten();
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
	const GLenum readStatus = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
	if (readStatus != GL_FRAMEBUFFER_COMPLETE) {
		gl4metalSetError(GL_INVALID_FRAMEBUFFER_OPERATION);
		GL4METAL_LOG("glReadPixels: read framebuffer %u is incomplete (status=0x%X)", ctx.read_fbo, readStatus);
		auto readFbo = GL4METAL_FBO_FIND(ctx.read_fbo);
		if (readFbo != ctx.fbo_objects.end()) logFBOAttachmentStates(ctx.read_fbo, readFbo->second);
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
		(NSUInteger)width > levelW - (NSUInteger)x || (NSUInteger)height > levelH - (NSUInteger)y) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glReadPixels: region %d,%d %dx%d is outside the %lux%lu attachment", x, y, width, height, (unsigned long)levelW, (unsigned long)levelH);
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

	std::vector<uint8_t> raw;
	try {
		raw.resize(rawBytes);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glReadPixels: CPU readback staging allocation failed (%lu bytes)", (unsigned long)rawBytes);
		return;
	}
	if (tex.storageMode == MTLStorageModeShared) {
		flushCommandBuffer(true); // finish the rendering before CPU reads shared texture storage
		[tex getBytes:raw.data() bytesPerRow:rawRowBytes
			fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height)
			mipmapLevel:(NSUInteger)a.level];
	} else {
		size_t stagingRowBytes = 0, stagingLength = 0;
		if (!checkedReadAlign(rawRowBytes, 256, stagingRowBytes) ||
			!checkedReadSizeMul(stagingRowBytes, (size_t)height, stagingLength) ||
			stagingLength == 0 || stagingLength > ctx.device.maxBufferLength) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("glReadPixels: Metal readback staging buffer size is invalid");
			return;
		}
		id<MTLBuffer> staging = [ctx.device newBufferWithLength:stagingLength options:MTLResourceStorageModeShared];
		if (!staging) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
		bool readbackSucceeded = false;

		if (ctx.metal4Enabled && gl4metalMetal4CopyTextureToBuffer(tex, 0, (NSUInteger)a.level,
			MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0), MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
			staging, 0, stagingRowBytes, 0, MTLBlitOptionNone)) {
			flushCommandBuffer(true);
			readbackSucceeded = ctx.metal4LastComputeEventValue == 0 ||
				(ctx.metal4InteropEvent && ctx.metal4InteropEvent.signaledValue >= ctx.metal4LastComputeEventValue);
		}

		if (!readbackSucceeded) {
			// Drain any partial Metal 4 attempt before submitting the classic blit.
			flushCommandBuffer(true);
			id<MTLCommandQueue> queue = ctx.commandQueue;
			id<MTLCommandBuffer> command_buffer = queue ? [queue commandBuffer] : nil;
			id<MTLBlitCommandEncoder> blit = command_buffer ? [command_buffer blitCommandEncoder] : nil;
			if (!command_buffer || !blit) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
			[blit copyFromTexture:tex sourceSlice:0 sourceLevel:(NSUInteger)a.level
				sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
				sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
				toBuffer:staging destinationOffset:0 destinationBytesPerRow:stagingRowBytes destinationBytesPerImage:0];
			[blit endEncoding];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
			if (command_buffer.status != MTLCommandBufferStatusCompleted) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("glReadPixels: classic private-texture readback failed: %s",
					command_buffer.error.localizedDescription.UTF8String ?: "unknown Metal error");
				return;
			}
		}
		const uint8_t *stagingBytes = (const uint8_t *)staging.contents;
		for (GLsizei row = 0; row < height; ++row) {
			memcpy(raw.data() + (size_t)row * rawRowBytes,
				stagingBytes + (size_t)row * stagingRowBytes, rawRowBytes);
		}
	}
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
