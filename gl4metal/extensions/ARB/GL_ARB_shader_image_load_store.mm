#import "gl4metal.h"

void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer,
	GLenum access, GLenum format) {
	if (unit >= GL4METAL_MAX_IMAGE_UNITS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (access != GL_READ_ONLY && access != GL_WRITE_ONLY && access != GL_READ_WRITE) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (texture == 0) {
		ctx.image_unit_bindings[unit] = ImageUnitBinding();
		return;
	}
	if (level != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindImageTexture: only mip level 0 is currently supported");
		return;
	}
	auto textureFound = ctx.texture_objects.find(texture);
	if (textureFound == ctx.texture_objects.end() || !textureFound->second.texture ||
		textureFound->second.width <= 0 || textureFound->second.height <= 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindImageTexture: texture %u is not allocated", texture);
		return;
	}
	const GLenum textureTarget = textureFound->second.target;
	// Per OpenGL, the layered parameter has no effect for a non-layered
	// GL_TEXTURE_2D object. A caller may therefore pass GL_TRUE here; requiring
	// GL_FALSE incorrectly drops the image binding and makes every later compute
	// dispatch fail even though the bound resource is a valid 2D texture.
	const bool is2D = textureTarget == GL_TEXTURE_2D && layer == 0 &&
		textureFound->second.texture.textureType == MTLTextureType2D;
	const bool is3D = textureTarget == GL_TEXTURE_3D && layered == GL_TRUE &&
		textureFound->second.depth > 0 && textureFound->second.texture.textureType == MTLTextureType3D;
	const bool isArray = textureTarget == GL_TEXTURE_2D_ARRAY && layered == GL_TRUE &&
		textureFound->second.depth > 0 && textureFound->second.texture.textureType == MTLTextureType2DArray;
	if (!is2D && !is3D && !isArray) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindImageTexture: supported bindings are GL_TEXTURE_2D (layered flag ignored, layer 0), layered GL_TEXTURE_3D and layered GL_TEXTURE_2D_ARRAY (texture=%u target=0x%X layered=%u layer=%d)",
			texture, textureTarget, (unsigned)layered, layer);
		return;
	}
	if (format != textureFound->second.internalformat) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindImageTexture: image format 0x%X does not match texture %u internal format 0x%X",
			format, texture, textureFound->second.internalformat);
		return;
	}
	if (access != GL_READ_ONLY &&
		!gl4metalEnsureTextureUsage(textureFound->second, MTLTextureUsageShaderWrite, "glBindImageTexture")) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindImageTexture: texture %u cannot be promoted to ShaderWrite usage", texture);
		return;
	}
	ImageUnitBinding binding;
	binding.texture = texture;
	binding.level = level;
	binding.layered = layered;
	binding.layer = layer;
	binding.access = access;
	binding.format = format;
	ctx.image_unit_bindings[unit] = binding;
}

void glMemoryBarrier(GLbitfield barriers) {
	constexpr GLbitfield validBarrierBits =
		GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT |
		GL_UNIFORM_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
		GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
		GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT |
		GL_BUFFER_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT |
		GL_TRANSFORM_FEEDBACK_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT |
		GL_SHADER_STORAGE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
		GL_QUERY_BUFFER_BARRIER_BIT;
	if (barriers != GL_ALL_BARRIER_BITS && (barriers & ~validBarrierBits) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glMemoryBarrier: invalid barrier mask 0x%08X", barriers);
		return;
	}
	// A zero mask has no visibility requirements and must not split a Metal encoder.
	if (barriers == 0) return;
	// Metal 4 keeps consecutive compute dispatches in one encoder. Encode an
	// intra-pass device-visibility barrier so shader writes from earlier dispatches
	// are visible to later dispatches in this same encoder.
	if (ctx.currentMetal4ComputeEncoder) {
		if (ctx.metal4LastWorkStage == 2) {
			[ctx.currentMetal4ComputeEncoder barrierAfterEncoderStages:MTLStageBlit
				beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
		} else if (ctx.metal4LastWorkStage == 1) {
			[ctx.currentMetal4ComputeEncoder barrierAfterEncoderStages:MTLStageDispatch
				beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
		}
		// The barrier prepares the next GL compute access to resources written by
		// either a prior dispatch or a prior MTL4 blit command. The stage barrier
		// publishes every earlier blit write, so its per-resource hazard set resets.
		ctx.metal4BlitReadResourceKeys.clear();
		ctx.metal4BlitWriteResourceKeys.clear();
		ctx.metal4LastWorkStage = 1;
		return;
	}

	// Classic Metal uses tracked resources and a new encoder in the same command
	// buffer to establish the boundary without stalling the CPU.
	endRenderEncoder();
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
		extern void invalidateComputeEncoderStateCache(void);
		invalidateComputeEncoderStateCache();
	}
}
