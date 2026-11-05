//
// enable.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "enable.h"

static void setCap(GLenum cap, GLboolean value) {
	switch (cap) {
		case GL_DEPTH_TEST:
			ctx.depth_test_enabled = value;
			updateDepthStencilState();
			break;
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
	auto it = ctx.caps.find(cap);
	return it != ctx.caps.end() && it->second;
}

void glEnablei(GLenum target, GLuint index) {
	glEnable(target); // one state for all draw buffers (per-buffer blending is not supported yet)
}

void glDisablei(GLenum target, GLuint index) {
	glDisable(target);
}

GLboolean glIsEnabledi(GLenum target, GLuint index) {
	return glIsEnabled(target);
}
