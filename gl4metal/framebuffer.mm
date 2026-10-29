//
// framebuffer.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//

#import "framebuffer.h"

// MARK: - Render passes

// Metal records work into encoders, GL into global state: a render pass is opened by glClear (or the first draw)
// for the current draw framebuffer and closed when the target changes or the frame is presented.

void endRenderEncoder(void) {
	if (ctx.currentEncoder) {
		[ctx.currentEncoder endEncoding];
		ctx.currentEncoder = nil;
	}
}

static id<MTLTexture> attachmentTexture(const FBOAttachment &a) {
	if (a.texture != 0) {
		auto it = ctx.texture_objects.find(a.texture);
		if (it != ctx.texture_objects.end()) {
			return it->second.texture;
		}
	}
	return nil;
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
	ctx.depthTexture = [ctx.device newTextureWithDescriptor:d];
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
		x0 = std::max<NSInteger>(ctx.scissor.x, 0);
		x1 = std::min<NSInteger>((NSInteger)ctx.scissor.x + ctx.scissor.w, (NSInteger)W);
		y0 = std::max<NSInteger>(mirror ? (NSInteger)H - sy1 : sy0, 0);
		y1 = std::min<NSInteger>(mirror ? (NSInteger)H - sy0 : sy1, (NSInteger)H);
	}
	MTLScissorRect sr;
	sr.x = (NSUInteger)x0;
	sr.y = (NSUInteger)y0;
	sr.width = (NSUInteger)std::max<NSInteger>(x1 - x0, 0);
	sr.height = (NSUInteger)std::max<NSInteger>(y1 - y0, 0);
	[ctx.currentEncoder setScissorRect:sr];
}

id<MTLRenderCommandEncoder> beginRenderPass(GLbitfield clearMask) {
	endRenderEncoder();

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
		rtWidth = color.width;
		rtHeight = color.height;

		MTLRenderPassColorAttachmentDescriptor *ca = rp.colorAttachments[0];
		ca.texture = color;
		ca.loadAction = colorLoad;
		ca.storeAction = MTLStoreActionStore;
		ca.clearColor = ctx.clear_color;
		ctx.pass_color_format[0] = color.pixelFormat;

		id<MTLTexture> depth = defaultDepthTexture(rtWidth, rtHeight);
		rp.depthAttachment.texture = depth;
		rp.depthAttachment.loadAction = depthLoad;
		rp.depthAttachment.storeAction = MTLStoreActionStore;
		rp.depthAttachment.clearDepth = ctx.clear_depth;
		rp.stencilAttachment.texture = depth;
		rp.stencilAttachment.loadAction = stencilLoad;
		rp.stencilAttachment.storeAction = MTLStoreActionStore;
		rp.stencilAttachment.clearStencil = (uint32_t)ctx.clear_stencil;
		ctx.pass_depth_format = depth.pixelFormat;
		ctx.pass_stencil_format = depth.pixelFormat;
	} else {
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

	if (!ctx.currentCommandBuffer) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	ctx.currentEncoder = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
	ctx.render_target_width = rtWidth;
	ctx.render_target_height = rtHeight;
	applyViewportAndScissor();
	return ctx.currentEncoder;
}

// MARK: - Framebuffer objects

void glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = ++ctx.next_fbo_id;
		ctx.fbo_objects[id] = FBO();
		framebuffers[i] = id;
	}
}

void glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = framebuffers[i];
		if (id == 0) {
			continue;
		}
		if (ctx.draw_fbo == id) {
			endRenderEncoder();
			ctx.draw_fbo = 0; // GL falls back to the default framebuffer
		}
		if (ctx.read_fbo == id) {
			ctx.read_fbo = 0;
		}
		ctx.fbo_objects.erase(id);
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
			GL4METAL_LOG("glBindFramebuffer: invalid target 0x%X", target);
			break;
	}
}

GLenum glCheckFramebufferStatus(GLenum target) {
	GLuint id = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (id == 0) {
		return GL_FRAMEBUFFER_COMPLETE; // the layer drawable is always complete
	}
	auto it = ctx.fbo_objects.find(id);
	if (it == ctx.fbo_objects.end()) {
		return GL_FRAMEBUFFER_UNDEFINED;
	}
	const FBO &fbo = it->second;

	bool any = false;
	auto check = [&](const FBOAttachment &a) -> bool { // false = attached but has no storage
		if (a.texture == 0) {
			return true;
		}
		any = true;
		return attachmentTexture(a) != nil;
	};
	bool ok = check(fbo.depth) && check(fbo.stencil);
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		ok = check(fbo.color[i]) && ok;
	}
	if (!ok) {
		return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
	}
	return any ? GL_FRAMEBUFFER_COMPLETE : GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
}

// MARK: - Framebuffer attachments

void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
	GL4METAL_LOG("glFramebufferTexture: Unimplemented");
}

void glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
	GL4METAL_LOG("glFramebufferTexture1D: Unimplemented");
}

void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
	GLuint id = (target == GL_READ_FRAMEBUFFER) ? ctx.read_fbo : ctx.draw_fbo;
	if (id == 0) {
		GL4METAL_LOG("glFramebufferTexture2D: the default framebuffer is bound (GL_INVALID_OPERATION)");
		return;
	}
	if (texture != 0 && textarget != GL_TEXTURE_2D) {
		GL4METAL_LOG("glFramebufferTexture2D: unsupported textarget 0x%X", textarget);
		return;
	}
	FBO &fbo = ctx.fbo_objects[id];
	FBOAttachment a;
	a.texture = texture;
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
		GL4METAL_LOG("glFramebufferTexture2D: invalid attachment 0x%X", attachment);
		return;
	}
	if (id == ctx.draw_fbo) {
		endRenderEncoder(); // the open render pass was built from the old attachments
	}
}

void glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset) {
	GL4METAL_LOG("glFramebufferTexture3D: Unimplemented");
}

void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
	GL4METAL_LOG("glFramebufferTextureLayer: Unimplemented");
}

void glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
	GL4METAL_LOG("glFramebufferRenderbuffer: Unimplemented");
}

void glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetFramebufferAttachmentParameteriv: Unimplemented");
}

// MARK: - Draw / read buffers

void glDrawBuffers(GLsizei n, const GLenum *bufs) {
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
	endRenderEncoder(); // the open pass was built from the old draw buffers
}

void glDrawBuffer(GLenum buf) {
	glDrawBuffers(1, &buf);
}

void glReadBuffer(GLenum src) {
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
	GL4METAL_LOG("glClearBufferiv: Unimplemented");
}

void glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint *value) {
	GL4METAL_LOG("glClearBufferuiv: Unimplemented");
}

void glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *value) {
	GL4METAL_LOG("glClearBufferfv: Unimplemented");
}

void glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
	GL4METAL_LOG("glClearBufferfi: Unimplemented");
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

static id<MTLRenderPipelineState> blitPipeline(MTLPixelFormat format) {
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
			GL4METAL_LOG("glBlitFramebuffer: blit shader failed to compile: %s", error.localizedDescription.UTF8String);
			return nil;
		}
	}
	MTLRenderPipelineDescriptor *d = [[MTLRenderPipelineDescriptor alloc] init];
	d.vertexFunction = [library newFunctionWithName:@"blit_vs"];
	d.fragmentFunction = [library newFunctionWithName:@"blit_fs"];
	d.colorAttachments[0].pixelFormat = format;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:d error:&error];
	if (!pipeline) {
		GL4METAL_LOG("glBlitFramebuffer: blit pipeline for pixel format %d failed: %s", (int)format, error.localizedDescription.UTF8String);
	}
	pipelines[(NSUInteger)format] = pipeline;
	return pipeline;
}

static id<MTLSamplerState> blitSampler(bool linear) {
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
	if (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) {
		GL4METAL_LOG("glBlitFramebuffer: depth/stencil blits are not supported yet (colour only)");
	}
	if (!(mask & GL_COLOR_BUFFER_BIT)) {
		return;
	}

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
	if (!src) {
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
	if (!dst) {
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
	rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;

	if (!ctx.currentCommandBuffer) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	id<MTLRenderCommandEncoder> enc = [ctx.currentCommandBuffer renderCommandEncoderWithDescriptor:rp];

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
	[enc setFragmentSamplerState:blitSampler(filter == GL_LINEAR) atIndex:0];
	[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[enc endEncoding];
}

void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels) {
	GL4METAL_LOG("glReadPixels: Unimplemented");
}
