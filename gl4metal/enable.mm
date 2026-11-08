//
// enable.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "enable.h"

static bool isCore33EnableCap(GLenum cap) {
	switch (cap) {
		case GL_BLEND: case GL_CULL_FACE: case GL_DEPTH_TEST: case GL_DITHER:
		case GL_POLYGON_OFFSET_FILL: case GL_SAMPLE_ALPHA_TO_COVERAGE: case GL_SAMPLE_ALPHA_TO_ONE:
		case GL_SAMPLE_COVERAGE: case GL_SCISSOR_TEST: case GL_STENCIL_TEST: case GL_MULTISAMPLE:
		case GL_TEXTURE_CUBE_MAP_SEAMLESS: case GL_PRIMITIVE_RESTART: case GL_RASTERIZER_DISCARD:
		case GL_DEPTH_CLAMP: case GL_FRAMEBUFFER_SRGB: case GL_SAMPLE_MASK:
			return true;
		default:
			return cap >= GL_CLIP_DISTANCE0 && cap <= GL_CLIP_DISTANCE7;
	}
}

static void setCap(GLenum cap, GLboolean value) {
	if (!isCore33EnableCap(cap)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (cap == GL_BLEND) {
		for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment)
			ctx.blend_enabled[attachment] = value;
	}
	switch (cap) {
		case GL_DEPTH_TEST:
			ctx.depth_test_enabled = value;
			updateDepthStencilState();
			break;
		case GL_STENCIL_TEST:
			ctx.caps[cap] = value;
			updateDepthStencilState();
			return;
		case GL_SCISSOR_TEST:
			ctx.scissor_test_enabled = value;
			applyViewportAndScissor();
			break;
		default:
			break; // tracked in ctx.caps only, applied when pipelines/draws are implemented
	}
	ctx.caps[cap] = value;
}

void glEnable(GLenum cap) {
	setCap(cap, GL_TRUE);
}

void glDisable(GLenum cap) {
	setCap(cap, GL_FALSE);
}

GLboolean glIsEnabled(GLenum cap) {
	if (!isCore33EnableCap(cap)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return GL_FALSE;
	}
	auto it = ctx.caps.find(cap);
	return it != ctx.caps.end() && it->second;
}

static bool validDrawBufferIndex(GLuint index) {
	if (index >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	return true;
}

void glEnablei(GLenum target, GLuint index) {
	if (target == GL_BLEND) {
		if (!validDrawBufferIndex(index)) return;
		ctx.blend_enabled[index] = GL_TRUE;
		if (index == 0) ctx.caps[GL_BLEND] = GL_TRUE;
		return;
	}
	if (target == GL_SCISSOR_TEST) {
		if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
		glEnable(target);
		return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
}

void glDisablei(GLenum target, GLuint index) {
	if (target == GL_BLEND) {
		if (!validDrawBufferIndex(index)) return;
		ctx.blend_enabled[index] = GL_FALSE;
		if (index == 0) ctx.caps[GL_BLEND] = GL_FALSE;
		return;
	}
	if (target == GL_SCISSOR_TEST) {
		if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
		glDisable(target);
		return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
}

GLboolean glIsEnabledi(GLenum target, GLuint index) {
	if (target == GL_BLEND) {
		if (!validDrawBufferIndex(index)) return GL_FALSE;
		return ctx.blend_enabled[index];
	}
	if (target == GL_SCISSOR_TEST) {
		if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return GL_FALSE; }
		return ctx.scissor_test_enabled;
	}
	gl4metalSetError(GL_INVALID_ENUM);
	return GL_FALSE;
}
