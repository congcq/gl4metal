#import "state.h"

// MARK: - Blending

void glBlendFunc(GLenum sfactor, GLenum dfactor) {
	ctx.blend_src_rgb = ctx.blend_src_alpha = sfactor;
	ctx.blend_dst_rgb = ctx.blend_dst_alpha = dfactor;
}

void glBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) {
	ctx.blend_src_rgb = srcRGB;
	ctx.blend_dst_rgb = dstRGB;
	ctx.blend_src_alpha = srcAlpha;
	ctx.blend_dst_alpha = dstAlpha;
}

void glBlendEquation(GLenum mode) {
	ctx.blend_eq_rgb = ctx.blend_eq_alpha = mode;
}

void glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha) {
	ctx.blend_eq_rgb = modeRGB;
	ctx.blend_eq_alpha = modeAlpha;
}

void glBlendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
	ctx.blend_color[0] = red;
	ctx.blend_color[1] = green;
	ctx.blend_color[2] = blue;
	ctx.blend_color[3] = alpha;
}

void glLogicOp(GLenum opcode) {
	GL4METAL_LOG("glLogicOp: Unimplemented");
}

// MARK: - Depth

void glDepthFunc(GLenum func) {
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

void glStencilFunc(GLenum func, GLint ref, GLuint mask) {
	GL4METAL_LOG("glStencilFunc: Unimplemented");
}

void glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
	GL4METAL_LOG("glStencilFuncSeparate: Unimplemented");
}

void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
	GL4METAL_LOG("glStencilOp: Unimplemented");
}

void glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
	GL4METAL_LOG("glStencilOpSeparate: Unimplemented");
}

void glStencilMask(GLuint mask) {
	GL4METAL_LOG("glStencilMask: Unimplemented");
}

void glStencilMaskSeparate(GLenum face, GLuint mask) {
	GL4METAL_LOG("glStencilMaskSeparate: Unimplemented");
}

// MARK: - Color write mask

void glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
	ctx.color_mask[0] = red;
	ctx.color_mask[1] = green;
	ctx.color_mask[2] = blue;
	ctx.color_mask[3] = alpha;
}

void glColorMaski(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
	GL4METAL_LOG("glColorMaski: Unimplemented");
}

void glClampColor(GLenum target, GLenum clamp) {
	GL4METAL_LOG("glClampColor: Unimplemented");
}

// MARK: - Rasterization

void glCullFace(GLenum mode) {
	ctx.cull_face_mode = mode;
}

void glFrontFace(GLenum mode) {
	ctx.front_face = mode;
}

void glPolygonMode(GLenum face, GLenum mode) {
	if (face == GL_FRONT_AND_BACK) { // Metal has a single fill mode for both faces
		ctx.polygon_mode = mode;
	} else {
		GL4METAL_LOG("glPolygonMode: only GL_FRONT_AND_BACK is supported");
	}
}

void glPolygonOffset(GLfloat factor, GLfloat units) {
	ctx.polygon_offset_factor = factor;
	ctx.polygon_offset_units = units;
}

void glLineWidth(GLfloat width) {
	GL4METAL_LOG("glLineWidth: Unimplemented");
}

void glPointSize(GLfloat size) {
	GL4METAL_LOG("glPointSize: Unimplemented");
}

void glPointParameterf(GLenum pname, GLfloat param) {
	GL4METAL_LOG("glPointParameterf: Unimplemented");
}

void glPointParameterfv(GLenum pname, const GLfloat *params) {
	GL4METAL_LOG("glPointParameterfv: Unimplemented");
}

void glPointParameteri(GLenum pname, GLint param) {
	GL4METAL_LOG("glPointParameteri: Unimplemented");
}

void glPointParameteriv(GLenum pname, const GLint *params) {
	GL4METAL_LOG("glPointParameteriv: Unimplemented");
}

void glProvokingVertex(GLenum mode) {
	GL4METAL_LOG("glProvokingVertex: Unimplemented");
}

void glPrimitiveRestartIndex(GLuint index) {
	GL4METAL_LOG("glPrimitiveRestartIndex: Unimplemented");
}

void glHint(GLenum target, GLenum mode) {
	GL4METAL_LOG("glHint: Unimplemented");
}

// MARK: - Multisample

void glSampleCoverage(GLfloat value, GLboolean invert) {
	GL4METAL_LOG("glSampleCoverage: Unimplemented");
}

void glSampleMaski(GLuint maskNumber, GLbitfield mask) {
	GL4METAL_LOG("glSampleMaski: Unimplemented");
}

// MARK: - Viewport / scissor

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
	ctx.viewport.x = x;
	ctx.viewport.y = y;
	ctx.viewport.w = width;
	ctx.viewport.h = height;
	applyViewportAndScissor();
}

void glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
	ctx.scissor.x = x;
	ctx.scissor.y = y;
	ctx.scissor.w = width;
	ctx.scissor.h = height;
	applyViewportAndScissor();
}
