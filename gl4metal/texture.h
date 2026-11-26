#ifndef TEXTURE_H
#define TEXTURE_H

#import <Metal/Metal.h>

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

struct TextureObject {
	GLenum target = 0; // 0 until the name is bound for the first time
	id<MTLTexture> texture = nil;
	GLenum internalformat = 0; // 0 until level 0 or a buffer texture format is defined
	GLsizei width = 0, height = 0, depth = 1; // level 0 size; depth > 1 for GL_TEXTURE_3D
	GLuint buffer_object = 0; // GL_TEXTURE_BUFFER backing store name (0 for ordinary textures)
	GLenum buffer_sampler_type = 0; // GL_SAMPLER_BUFFER / GL_INT_SAMPLER_BUFFER / GL_UNSIGNED_INT_SAMPLER_BUFFER
	bool immutable_storage = false; // glTexStorage* / multisample texture storage cannot be redefined

	// sampler parameters (consumed when the texture is bound to a shader)
	GLenum min_filter = GL_NEAREST_MIPMAP_LINEAR;
	GLenum mag_filter = GL_LINEAR;
	GLenum wrap_s = GL_REPEAT, wrap_t = GL_REPEAT, wrap_r = GL_REPEAT;
	GLfloat min_lod = -1000.0f, max_lod = 1000.0f, lod_bias = 0.0f;
	GLint base_level = 0, max_level = 1000;
	GLfloat max_anisotropy = 1.0f;
	GLenum compare_mode = GL_NONE, compare_func = GL_LEQUAL;
	GLfloat border_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	GLenum swizzle[4] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};

	// Sampling views share the original allocation. Keep separate entries for
	// the regular texture and the feedback snapshot so swizzle never changes
	// framebuffer attachment or image-load/store semantics.
	mutable __strong id<MTLTexture> sampled_view = nil;
	mutable void *sampled_view_source_identity = nullptr;
	mutable __strong id<MTLTexture> feedback_sampled_view = nil;
	mutable void *feedback_sampled_view_source_identity = nullptr;

	// Lazily allocated read-only copy used when a shader samples this same image
	// while it is selected as an active framebuffer attachment. Tied to the GL
	// texture object's lifetime; recreated when the source dimensions/format change.
	id<MTLTexture> feedback_snapshot = nil;
	uint64_t feedback_snapshot_pass_serial = UINT64_MAX;
};

MTLPixelFormat metalPixelFormat(GLenum internalformat);
void resetTextureUploadArena(id<MTLCommandBuffer> commandBuffer);
void refreshTextureBufferViews(GLuint bufferObject);
GLint maxTextureBufferTexels(void);
void gl4metalInvalidateTextureSamplingViews(TextureObject &textureObject);
void gl4metalSetTextureStorage(TextureObject &textureObject, id<MTLTexture> texture);
// Lazily promotes image textures to ShaderWrite only when GL binds them writable.
bool gl4metalEnsureTextureUsage(TextureObject &textureObject, MTLTextureUsage requiredUsage, const char *caller);
// sourceTexture is borrowed from TextureObject storage or the active feedback-snapshot map.
id<MTLTexture> gl4metalTextureForSampling(const TextureObject &textureObject, __unsafe_unretained id<MTLTexture> sourceTexture);

// Internal implementation hooks used by GL_ARB_clear_texture entry points.
void gl4metalClearTexImageImpl(GLuint texture, GLint level, GLenum format, GLenum type, const void *data);
void gl4metalClearTexSubImageImpl(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
	GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data);

#endif /* TEXTURE_H */
