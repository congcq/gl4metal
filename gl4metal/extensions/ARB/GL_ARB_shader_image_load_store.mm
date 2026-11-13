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
	if (!is2D && !is3D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindImageTexture: supported bindings are GL_TEXTURE_2D (layered flag ignored, layer 0) and layered GL_TEXTURE_3D (texture=%u target=0x%X layered=%u layer=%d)",
			texture, textureTarget, (unsigned)layered, layer);
		return;
	}
	if (format != textureFound->second.internalformat) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindImageTexture: image format 0x%X does not match texture %u internal format 0x%X",
			format, texture, textureFound->second.internalformat);
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
	(void)barriers; // Metal command encoders provide ordered resource hazards for implemented resources.
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
	}
	endRenderEncoder();
}
