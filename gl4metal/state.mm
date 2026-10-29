#import "state.h"

// Fixed-function state Metal has no equivalent for (or that is not wired up yet): accepted, reported once, ignored.
static void notSupported(const char *name) {
	static std::unordered_set<std::string> reported;
	if (reported.insert(name).second) {
		GL4METAL_LOG("%s: not supported by the Metal layer, ignored (reported once)", name);
	}
}

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
	notSupported("glLogicOp");
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
	notSupported("glStencilFunc");
}

void glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
	notSupported("glStencilFuncSeparate");
}

void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
	notSupported("glStencilOp");
}

void glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
	notSupported("glStencilOpSeparate");
}

void glStencilMask(GLuint mask) {
	notSupported("glStencilMask");
}

void glStencilMaskSeparate(GLenum face, GLuint mask) {
	notSupported("glStencilMaskSeparate");
}

// MARK: - Color write mask

void glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
	ctx.color_mask[0] = red;
	ctx.color_mask[1] = green;
	ctx.color_mask[2] = blue;
	ctx.color_mask[3] = alpha;
}

void glColorMaski(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
	if (index == 0) { // one mask for all draw buffers for now
		glColorMask(r, g, b, a);
	}
}

void glClampColor(GLenum target, GLenum clamp) {
	notSupported("glClampColor");
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
	notSupported("glLineWidth");
}

void glPointSize(GLfloat size) {
	notSupported("glPointSize");
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
	notSupported("glPrimitiveRestartIndex");
}

void glHint(GLenum target, GLenum mode) {
	notSupported("glHint");
}

// MARK: - Multisample

void glSampleCoverage(GLfloat value, GLboolean invert) {
	notSupported("glSampleCoverage");
}

void glSampleMaski(GLuint maskNumber, GLbitfield mask) {
	notSupported("glSampleMaski");
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
