#import "state.h"

#include <cmath>

// Fixed-function state Metal has no equivalent for (or that is not wired up yet): accepted, reported once, ignored.
static void notSupported(const char *name) {
	gl4metalSetError(GL_INVALID_OPERATION);
	static std::unordered_set<std::string> reported;
	if (reported.insert(name).second) {
		GL4METAL_LOG("%s: not supported by the Metal layer, ignored (reported once)", name);
	}
}

// MARK: - Blending

static bool validBlendFactor(GLenum factor) {
	switch (factor) {
		case GL_ZERO: case GL_ONE:
		case GL_SRC_COLOR: case GL_ONE_MINUS_SRC_COLOR:
		case GL_DST_COLOR: case GL_ONE_MINUS_DST_COLOR:
		case GL_SRC_ALPHA: case GL_ONE_MINUS_SRC_ALPHA:
		case GL_DST_ALPHA: case GL_ONE_MINUS_DST_ALPHA:
		case GL_CONSTANT_COLOR: case GL_ONE_MINUS_CONSTANT_COLOR:
		case GL_CONSTANT_ALPHA: case GL_ONE_MINUS_CONSTANT_ALPHA:
		case GL_SRC_ALPHA_SATURATE:
			return true;
		default:
			return false;
	}
}

static bool validBlendEquation(GLenum equation) {
	switch (equation) {
		case GL_FUNC_ADD: case GL_FUNC_SUBTRACT: case GL_FUNC_REVERSE_SUBTRACT:
		case GL_MIN: case GL_MAX:
			return true;
		default:
			return false;
	}
}

static bool validBlendBufferIndex(GLuint bufferIndex) {
	if (bufferIndex >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	return true;
}

static void mirrorBlendBufferZero() {
	const BlendBufferState &state = ctx.blend_buffers[0];
	ctx.blend_src_rgb = state.src_rgb;
	ctx.blend_dst_rgb = state.dst_rgb;
	ctx.blend_src_alpha = state.src_alpha;
	ctx.blend_dst_alpha = state.dst_alpha;
	ctx.blend_eq_rgb = state.equation_rgb;
	ctx.blend_eq_alpha = state.equation_alpha;
}

static bool validBlendFunction(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
	// GL_SRC_ALPHA_SATURATE is legal for source RGB and source alpha (where its
	// alpha component is defined as one), but never as a destination factor.
	if (validBlendFactor(srcRGB) && validBlendFactor(dstRGB) &&
		validBlendFactor(srcAlpha) && validBlendFactor(dstAlpha) &&
		dstRGB != GL_SRC_ALPHA_SATURATE && dstAlpha != GL_SRC_ALPHA_SATURATE) {
		return true;
	}
	gl4metalSetError(GL_INVALID_ENUM);
	return false;
}

void glBlendFunc(GLenum sfactor, GLenum dfactor) {
	glBlendFuncSeparate(sfactor, dfactor, sfactor, dfactor);
}

void glBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
	if (!validBlendFunction(srcRGB, dstRGB, srcAlpha, dstAlpha)) return;
	for (int bufferIndex = 0; bufferIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++bufferIndex) {
		BlendBufferState &state = ctx.blend_buffers[bufferIndex];
		state.src_rgb = srcRGB;
		state.dst_rgb = dstRGB;
		state.src_alpha = srcAlpha;
		state.dst_alpha = dstAlpha;
	}
	mirrorBlendBufferZero();
}

void glBlendFunci(GLuint bufferIndex, GLenum src, GLenum dst) {
	glBlendFuncSeparatei(bufferIndex, src, dst, src, dst);
}

void glBlendFuncSeparatei(GLuint bufferIndex, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
	if (!validBlendBufferIndex(bufferIndex) || !validBlendFunction(srcRGB, dstRGB, srcAlpha, dstAlpha)) return;
	BlendBufferState &state = ctx.blend_buffers[bufferIndex];
	state.src_rgb = srcRGB;
	state.dst_rgb = dstRGB;
	state.src_alpha = srcAlpha;
	state.dst_alpha = dstAlpha;
	if (bufferIndex == 0) mirrorBlendBufferZero();
}

void glBlendEquation(GLenum mode) {
	glBlendEquationSeparate(mode, mode);
}

void glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
	if (!validBlendEquation(modeRGB) || !validBlendEquation(modeAlpha)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	for (int bufferIndex = 0; bufferIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++bufferIndex) {
		ctx.blend_buffers[bufferIndex].equation_rgb = modeRGB;
		ctx.blend_buffers[bufferIndex].equation_alpha = modeAlpha;
	}
	mirrorBlendBufferZero();
}

void glBlendEquationi(GLuint bufferIndex, GLenum mode) {
	glBlendEquationSeparatei(bufferIndex, mode, mode);
}

void glBlendEquationSeparatei(GLuint bufferIndex, GLenum modeRGB, GLenum modeAlpha) {
	if (!validBlendBufferIndex(bufferIndex)) return;
	if (!validBlendEquation(modeRGB) || !validBlendEquation(modeAlpha)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	BlendBufferState &state = ctx.blend_buffers[bufferIndex];
	state.equation_rgb = modeRGB;
	state.equation_alpha = modeAlpha;
	if (bufferIndex == 0) mirrorBlendBufferZero();
}

void glBlendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
	ctx.blend_color[0] = red;
	ctx.blend_color[1] = green;
	ctx.blend_color[2] = blue;
	ctx.blend_color[3] = alpha;
}

void glLogicOp(GLenum opcode) {
	notSupported("glLogicOp");
}

// MARK: - Depth

void glDepthFunc(GLenum func) {
	switch (func) {
		case GL_NEVER: case GL_LESS: case GL_EQUAL: case GL_LEQUAL:
		case GL_GREATER: case GL_NOTEQUAL: case GL_GEQUAL: case GL_ALWAYS:
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			return;
	}
	ctx.depth_func = func;
	updateDepthStencilState();
}

void glDepthMask(GLboolean flag) {
	ctx.depth_write_enabled = flag;
	updateDepthStencilState();
}

void glDepthRange(GLdouble n, GLdouble f) {
	ctx.depth_range_near = (GLfloat)std::min(std::max(n, 0.0), 1.0);
	ctx.depth_range_far = (GLfloat)std::min(std::max(f, 0.0), 1.0);
	applyViewportAndScissor(); // the depth range lives in the viewport
}

// MARK: - Stencil

static bool validStencilFace(GLenum face) {
	return face == GL_FRONT || face == GL_BACK || face == GL_FRONT_AND_BACK;
}

static bool validStencilFunc(GLenum func) {
	switch (func) {
		case GL_NEVER: case GL_LESS: case GL_EQUAL: case GL_LEQUAL:
		case GL_GREATER: case GL_NOTEQUAL: case GL_GEQUAL: case GL_ALWAYS: return true;
		default: return false;
	}
}

static bool validStencilOp(GLenum op) {
	switch (op) {
		case GL_KEEP: case GL_ZERO: case GL_REPLACE: case GL_INCR: case GL_INCR_WRAP:
		case GL_DECR: case GL_DECR_WRAP: case GL_INVERT: return true;
		default: return false;
	}
}

static void setStencilFuncForFace(MTLContext::StencilFaceState &state, GLenum func, GLint ref, GLuint mask) {
	state.func = func;
	state.ref = ref;
	state.value_mask = mask;
}

void glStencilFunc(GLenum func, GLint ref, GLuint mask) {
	glStencilFuncSeparate(GL_FRONT_AND_BACK, func, ref, mask);
}

void glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
	if (!validStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (!validStencilFunc(func)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (face == GL_FRONT || face == GL_FRONT_AND_BACK) setStencilFuncForFace(ctx.stencil_front, func, ref, mask);
	if (face == GL_BACK || face == GL_FRONT_AND_BACK) setStencilFuncForFace(ctx.stencil_back, func, ref, mask);
	updateDepthStencilState();
}

static void setStencilOpsForFace(MTLContext::StencilFaceState &state, GLenum fail, GLenum depthFail, GLenum depthPass) {
	state.fail = fail; state.depth_fail = depthFail; state.depth_pass = depthPass;
}

void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
	glStencilOpSeparate(GL_FRONT_AND_BACK, fail, zfail, zpass);
}

void glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
	if (!validStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (!validStencilOp(sfail) || !validStencilOp(dpfail) || !validStencilOp(dppass)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (face == GL_FRONT || face == GL_FRONT_AND_BACK) setStencilOpsForFace(ctx.stencil_front, sfail, dpfail, dppass);
	if (face == GL_BACK || face == GL_FRONT_AND_BACK) setStencilOpsForFace(ctx.stencil_back, sfail, dpfail, dppass);
	updateDepthStencilState();
}

void glStencilMask(GLuint mask) {
	glStencilMaskSeparate(GL_FRONT_AND_BACK, mask);
}

void glStencilMaskSeparate(GLenum face, GLuint mask) {
	if (!validStencilFace(face)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (face == GL_FRONT || face == GL_FRONT_AND_BACK) ctx.stencil_front.write_mask = mask;
	if (face == GL_BACK || face == GL_FRONT_AND_BACK) ctx.stencil_back.write_mask = mask;
	updateDepthStencilState();
}

// MARK: - Color write mask

void glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
	const GLboolean values[4] = { red, green, blue, alpha };
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment)
		for (int component = 0; component < 4; ++component)
			ctx.color_masks[attachment][component] = values[component];
	for (int component = 0; component < 4; ++component)
		ctx.color_mask[component] = values[component];
}

void glColorMaski(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
	if (index >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const GLboolean values[4] = { r, g, b, a };
	for (int component = 0; component < 4; ++component)
		ctx.color_masks[index][component] = values[component];
	if (index == 0) {
		for (int component = 0; component < 4; ++component)
			ctx.color_mask[component] = values[component];
	}
}

void glClampColor(GLenum target, GLenum clamp) {
	if (target != GL_CLAMP_READ_COLOR && target != GL_CLAMP_VERTEX_COLOR && target != GL_CLAMP_FRAGMENT_COLOR) {
		gl4metalSetError(GL_INVALID_ENUM); return;
	}
	if (clamp != GL_TRUE && clamp != GL_FALSE && clamp != GL_FIXED_ONLY) {
		gl4metalSetError(GL_INVALID_ENUM); return;
	}
	// Metal's normalized render-target formats clamp on write; floating-point targets preserve range.
	// The legacy per-stage clamp controls have no direct Metal fixed-function equivalent.
}

// MARK: - Rasterization

void glCullFace(GLenum mode) {
	if (mode != GL_FRONT && mode != GL_BACK && mode != GL_FRONT_AND_BACK) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	ctx.cull_face_mode = mode;
}

void glFrontFace(GLenum mode) {
	if (mode != GL_CW && mode != GL_CCW) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	ctx.front_face = mode;
}

void glPolygonMode(GLenum face, GLenum mode) {
	if (face != GL_FRONT_AND_BACK) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glPolygonMode: only GL_FRONT_AND_BACK is supported");
		return;
	}
	if (mode != GL_LINE && mode != GL_FILL && mode != GL_POINT) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (mode == GL_POINT) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glPolygonMode(GL_POINT): point polygon mode is not implemented; state unchanged");
		return;
	}
	ctx.polygon_mode = mode;
}

void glPolygonOffset(GLfloat factor, GLfloat units) {
	ctx.polygon_offset_factor = factor;
	ctx.polygon_offset_units = units;
}

void glLineWidth(GLfloat width) {
	if (!(width > 0.0f)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	// Metal exposes line primitives with a fixed 1-pixel width on iOS. Its GL range is [1, 1].
	if (width != 1.0f) GL4METAL_LOG("glLineWidth: iOS Metal supports width 1 only; requested width is clamped");
}

void glPointSize(GLfloat size) {
	if (!(size > 0.0f)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	// This backend currently maps GL_POINTS to Metal point primitives at fixed size 1.
	if (size != 1.0f) GL4METAL_LOG("glPointSize: point-size output is not available in the current MSL vertex interface");
}

void glPointParameterf(GLenum pname, GLfloat param) {
	notSupported("glPointParameterf");
}

void glPointParameterfv(GLenum pname, const GLfloat *params) {
	notSupported("glPointParameterfv");
}

void glPointParameteri(GLenum pname, GLint param) {
	notSupported("glPointParameteri");
}

void glPointParameteriv(GLenum pname, const GLint *params) {
	notSupported("glPointParameteriv");
}

void glProvokingVertex(GLenum mode) {
	notSupported("glProvokingVertex");
}

void glPrimitiveRestartIndex(GLuint index) {
	ctx.primitive_restart_index = index;
}

void glHint(GLenum target, GLenum mode) {
	const bool validTarget = target == GL_PERSPECTIVE_CORRECTION_HINT || target == GL_POINT_SMOOTH_HINT ||
		target == GL_LINE_SMOOTH_HINT || target == GL_POLYGON_SMOOTH_HINT || target == GL_FOG_HINT ||
		target == GL_TEXTURE_COMPRESSION_HINT || target == GL_GENERATE_MIPMAP_HINT;
	if (!validTarget) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (mode != GL_FASTEST && mode != GL_NICEST && mode != GL_DONT_CARE) { gl4metalSetError(GL_INVALID_ENUM); return; }
	// Performance hints do not change rendered results; Metal owns the corresponding optimization choices.
}

// MARK: - Multisample

void glSampleCoverage(GLfloat value, GLboolean invert) {
	if (std::isnan(value)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (value < 0.0f || value > 1.0f) { gl4metalSetError(GL_INVALID_VALUE); return; }
	// Metal iOS has no encoder-level sample-coverage control. Do not claim to apply it.
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glSampleCoverage: sample coverage is not currently emulated by this Metal backend");
}

void glSampleMaski(GLuint maskNumber, GLbitfield mask) {
	if (maskNumber != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
	// MSL sample-mask output can emulate this, but the translator does not currently inject it.
	gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glSampleMaski: sample-mask emulation is not available in the current shader translator");
}

// MARK: - Viewport / scissor

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
	if (width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ctx.viewport.x = x;
	ctx.viewport.y = y;
	ctx.viewport.w = width;
	ctx.viewport.h = height;
	applyViewportAndScissor();
}

void glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
	if (width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ctx.scissor.x = x;
	ctx.scissor.y = y;
	ctx.scissor.w = width;
	ctx.scissor.h = height;
	applyViewportAndScissor();
}
