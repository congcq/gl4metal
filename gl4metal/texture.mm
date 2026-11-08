#import "texture.h"

#include <cmath>
#include <limits>
#include <new>

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif

// MARK: - Helpers
// iOS notes: CPU-visible textures use MTLStorageModeShared (Managed does not exist on iOS), depth/stencil textures are
// Private, and there are no 24-bit depth formats nor 24-bit RGB formats, so GL_DEPTH24_* maps to Depth32Float* and
// GL_RGB8 to RGBA8.

MTLPixelFormat metalPixelFormat(GLenum internalformat) {
	switch (internalformat) {
		case GL_R8:
		case GL_RED:
			return MTLPixelFormatR8Unorm;
		case GL_RG8:
		case GL_RG:
			return MTLPixelFormatRG8Unorm;
		case GL_RGB8:
		case GL_RGB:
		case GL_RGBA8:
		case GL_RGBA:
			return MTLPixelFormatRGBA8Unorm;
		case GL_SRGB8:
		case GL_SRGB8_ALPHA8:
			return MTLPixelFormatRGBA8Unorm_sRGB;
		case GL_R16:
			return MTLPixelFormatR16Unorm;
		case GL_RG16:
			return MTLPixelFormatRG16Unorm;
		case GL_RGB16:
		case GL_RGBA16:
			// Metal has no three-channel 16-bit UNorm texture; keep RGB data in RGBA16.
			return MTLPixelFormatRGBA16Unorm;
		case GL_R16F:
			return MTLPixelFormatR16Float;
		case GL_RG16F:
			return MTLPixelFormatRG16Float;
		case GL_RGB16F:
		case GL_RGBA16F:
			// Metal has no three-channel 16-bit float texture; use RGBA16Float.
			return MTLPixelFormatRGBA16Float;
		case GL_R32F:
			return MTLPixelFormatR32Float;
		case GL_RG32F:
			return MTLPixelFormatRG32Float;
		case GL_RGB32F:
		case GL_RGBA32F:
			return MTLPixelFormatRGBA32Float;
		case GL_RGB10_A2:
			return MTLPixelFormatRGB10A2Unorm;
		case GL_R11F_G11F_B10F:
			return MTLPixelFormatRG11B10Float;
		case GL_RGB9_E5:
			return MTLPixelFormatRGB9E5Float;
		case GL_DEPTH_COMPONENT16:
			return MTLPixelFormatDepth16Unorm;
		case GL_DEPTH_COMPONENT:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			return MTLPixelFormatDepth32Float;
		case GL_DEPTH_STENCIL:
		case GL_DEPTH24_STENCIL8:
		case GL_DEPTH32F_STENCIL8:
			return MTLPixelFormatDepth32Float_Stencil8;
		default:
			return MTLPixelFormatInvalid;
	}
}

static bool isDepthStencilFormat(MTLPixelFormat f) {
	return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8;
}


static bool usesMipmaps(GLenum filter) {
	return filter == GL_NEAREST_MIPMAP_NEAREST || filter == GL_LINEAR_MIPMAP_NEAREST ||
		   filter == GL_NEAREST_MIPMAP_LINEAR || filter == GL_LINEAR_MIPMAP_LINEAR;
}

// Metal fixes the mip count at creation. Honour GL_TEXTURE_MAX_LEVEL when the app set it (Minecraft does, before
// specifying the levels), otherwise only mipmapped minification filters get a full chain.
static NSUInteger mipLevelCount(const TextureObject &t, GLsizei w, GLsizei h) {
	NSUInteger full = 1;
	for (GLsizei m = std::max(w, h); m > 1; m >>= 1) {
		full++;
	}
	if (t.max_level < 1000) {
		return std::min<NSUInteger>(full, (NSUInteger)std::max(t.max_level, 0) + 1);
	}
	return usesMipmaps(t.min_filter) ? full : 1;
}

// GL_TEXTURE_CUBE_MAP_POSITIVE_X ... NEGATIVE_Z -> slice 0...5 of the cube texture, -1 for any other target.
static int cubeFace(GLenum target) {
	return (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) ? (int)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : -1;
}

static TextureObject *boundTexture(GLenum target) {
	GLuint id = ctx.bound_textures[ctx.active_texture_unit][target];
	if (id == 0) {
		return nullptr;
	}
	auto it = ctx.texture_objects.find(id);
	return it == ctx.texture_objects.end() ? nullptr : &it->second;
}

// Checked size arithmetic shared by the CPU pixel-transfer paths.
static bool checkedSizeAdd(size_t a, size_t b, size_t &out) {
	if (b > SIZE_MAX - a) return false;
	out = a + b;
	return true;
}

static bool checkedSizeMul(size_t a, size_t b, size_t &out) {
	if (a != 0 && b > SIZE_MAX / a) return false;
	out = a * b;
	return true;
}

static bool checkedAlignSize(size_t value, size_t alignment, size_t &out) {
	if (alignment == 0) return false;
	size_t sum = 0;
	if (!checkedSizeAdd(value, alignment - 1, sum)) return false;
	out = (sum / alignment) * alignment;
	return true;
}

// GL_PIXEL_UNPACK_BUFFER bound: `pixels` is an offset into that buffer. Validate the
// complete accessed span before pointer arithmetic; otherwise a short PBO can become
// an invalid CPU read before Metal is ever involved.
static const uint8_t *resolveUnpackPointer(const void *pixels, size_t byteOffset, size_t accessedBytes, const char *caller) {
	const GLuint pbo = boundBuffer(GL_PIXEL_UNPACK_BUFFER);
	if (pbo == 0) {
		if (!pixels) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("%s: null pixel pointer without GL_PIXEL_UNPACK_BUFFER", caller);
			return nullptr;
		}
		if (byteOffset > SIZE_MAX - (uintptr_t)pixels) {
			gl4metalSetError(GL_INVALID_VALUE);
			return nullptr;
		}
		return (const uint8_t *)pixels + byteOffset;
	}
	auto it = ctx.buffer_objects.find(pbo);
	if (it == ctx.buffer_objects.end() || !it->second || !it->second.contents) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: GL_PIXEL_UNPACK_BUFFER %u has no CPU-visible storage", caller, pbo);
		return nullptr;
	}
	const uintptr_t baseOffset = (uintptr_t)pixels;
	const NSUInteger length = it->second.length;
	if (baseOffset > length || byteOffset > length - baseOffset || accessedBytes > length - baseOffset - byteOffset) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: pixel-unpack buffer range offset=%llu span=%llu exceeds %llu-byte buffer %u",
			caller, (unsigned long long)baseOffset, (unsigned long long)(byteOffset + accessedBytes),
			(unsigned long long)length, pbo);
		return nullptr;
	}
	return (const uint8_t *)it->second.contents + baseOffset + byteOffset;
}

// Position of the R,G,B,A channels inside one source pixel (-1 = channel not present).
static bool sourceLayout(GLenum format, int &comps, int idx[4]) {
	idx[0] = idx[1] = idx[2] = idx[3] = -1;
	switch (format) {
		case GL_RED:
		case GL_DEPTH_COMPONENT: comps = 1; idx[0] = 0; return true;
		case GL_RG:   comps = 2; idx[0] = 0; idx[1] = 1; return true;
		case GL_RGB:  comps = 3; idx[0] = 0; idx[1] = 1; idx[2] = 2; return true;
		case GL_BGR:  comps = 3; idx[2] = 0; idx[1] = 1; idx[0] = 2; return true;
		case GL_RGBA: comps = 4; idx[0] = 0; idx[1] = 1; idx[2] = 2; idx[3] = 3; return true;
		case GL_BGRA: comps = 4; idx[2] = 0; idx[1] = 1; idx[0] = 2; idx[3] = 3; return true;
		default: return false;
	}
}

// Components and bytes per component of the Metal formats we can write from the CPU.
static bool destLayout(MTLPixelFormat f, int &comps, int &compSize) {
	switch (f) {
		case MTLPixelFormatR8Unorm:         comps = 1; compSize = 1; return true;
		case MTLPixelFormatRG8Unorm:        comps = 2; compSize = 1; return true;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB: comps = 4; compSize = 1; return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Float:        comps = 1; compSize = 2; return true;
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Float:       comps = 2; compSize = 2; return true;
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Float:     comps = 4; compSize = 2; return true;
		case MTLPixelFormatR32Float:        comps = 1; compSize = 4; return true;
		case MTLPixelFormatRG32Float:       comps = 2; compSize = 4; return true;
		case MTLPixelFormatRGBA32Float:     comps = 4; compSize = 4; return true;
		default: return false;
	}
}

static void uploadPixels(TextureObject &t, GLint level, GLint x, GLint y, GLsizei w, GLsizei h,
						 GLenum format, GLenum type, const void *pixels, const char *caller, NSUInteger slice = 0) {
	if (!t.texture || level < 0 || (NSUInteger)level >= t.texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: invalid or unallocated mip level %d", caller, level);
		return;
	}
	const GLsizei lw = std::max<GLsizei>(1, t.width >> level);
	const GLsizei lh = std::max<GLsizei>(1, t.height >> level);
	if (x < 0 || y < 0 || w < 0 || h < 0 || x > lw || y > lh || w > lw - x || h > lh - y) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: region %d,%d %dx%d does not fit level %d (%dx%d)", caller, x, y, w, h, level, lw, lh);
		return;
	}
	if (w == 0 || h == 0) return;
	if (t.texture.storageMode == MTLStorageModePrivate) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: CPU pixel upload to a private texture is unsupported; use a staging/blit path", caller);
		return;
	}

	int srcComps = 0, idx[4];
	int dstComps = 0, dstCompSize = 0;
	const int srcCompSize = (type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV) ? 1 :
		(type == GL_UNSIGNED_SHORT || type == GL_HALF_FLOAT) ? 2 : (type == GL_FLOAT ? 4 : 0);
	const MTLPixelFormat dstFormat = t.texture.pixelFormat;
	const bool dstIs16Unorm = dstFormat == MTLPixelFormatR16Unorm || dstFormat == MTLPixelFormatRG16Unorm || dstFormat == MTLPixelFormatRGBA16Unorm;
	const bool dstIs16Float = dstFormat == MTLPixelFormatR16Float || dstFormat == MTLPixelFormatRG16Float || dstFormat == MTLPixelFormatRGBA16Float;
	const bool incompatible16Type = (dstIs16Unorm && type != GL_UNSIGNED_SHORT) || (dstIs16Float && type != GL_HALF_FLOAT);
	if (!sourceLayout(format, srcComps, idx) || srcCompSize == 0 || !destLayout(dstFormat, dstComps, dstCompSize) ||
		srcCompSize != dstCompSize || incompatible16Type || (type == GL_UNSIGNED_INT_8_8_8_8_REV && srcComps != 4)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("%s: unsupported upload, format 0x%X type 0x%X into MTLPixelFormat %d", caller, format, type, (int)t.texture.pixelFormat);
		return;
	}

	// Source row layout, honouring GL_UNPACK_ROW_LENGTH / ALIGNMENT / SKIP_*.
	size_t srcBpp = 0, rowRawBytes = 0, srcRowBytes = 0;
	const size_t rowLength = ctx.unpack_row_length > 0 ? (size_t)ctx.unpack_row_length : (size_t)w;
	const size_t align = ctx.unpack_alignment > 0 ? (size_t)ctx.unpack_alignment : 1;
	if (!checkedSizeMul((size_t)srcComps, (size_t)srcCompSize, srcBpp) ||
		!checkedSizeMul(rowLength, srcBpp, rowRawBytes) || !checkedAlignSize(rowRawBytes, align, srcRowBytes)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: unpack row layout overflows addressable memory", caller);
		return;
	}
	size_t skipRowsBytes = 0, skipPixelsBytes = 0, lastRowBytes = 0, copyRowBytes = 0;
	size_t byteOffset = 0, accessedBytes = 0;
	if (!checkedSizeMul((size_t)ctx.unpack_skip_rows, srcRowBytes, skipRowsBytes) ||
		!checkedSizeMul((size_t)ctx.unpack_skip_pixels, srcBpp, skipPixelsBytes) ||
		!checkedSizeMul((size_t)(h - 1), srcRowBytes, lastRowBytes) ||
		!checkedSizeMul((size_t)w, srcBpp, copyRowBytes) ||
		!checkedSizeAdd(skipRowsBytes, skipPixelsBytes, byteOffset) ||
		!checkedSizeAdd(byteOffset, lastRowBytes, accessedBytes) ||
		!checkedSizeAdd(accessedBytes, copyRowBytes, accessedBytes)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: unpack skip/extent calculation overflows addressable memory", caller);
		return;
	}
	const size_t spanBytes = accessedBytes - byteOffset;
	const uint8_t *src = resolveUnpackPointer(pixels, byteOffset, spanBytes, caller);
	if (!src) return;

	size_t dstBpp = 0, tmpRowBytes = 0, tmpBytes = 0;
	if (!checkedSizeMul((size_t)dstComps, (size_t)dstCompSize, dstBpp) ||
		!checkedSizeMul((size_t)w, dstBpp, tmpRowBytes) ||
		!checkedSizeMul(tmpRowBytes, (size_t)h, tmpBytes) || tmpBytes > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: upload staging size is too large", caller);
		return;
	}
	MTLRegion region = MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h);

	bool identical = (srcComps == dstComps);
	for (int c = 0; c < dstComps && identical; c++) identical = (idx[c] == c);
	if (srcRowBytes < copyRowBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: GL_UNPACK_ROW_LENGTH produces a row stride (%lu) smaller than the copied row (%lu)",
			caller, (unsigned long)srcRowBytes, (unsigned long)copyRowBytes);
		return;
	}
	if (identical && srcRowBytes % dstBpp == 0) {
		flushCommandBuffer(true); // CPU writes to Shared textures must follow earlier GPU reads.
		[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:src bytesPerRow:srcRowBytes bytesPerImage:0];
		return;
	}

	// Channel swizzle / expansion into a tightly packed temporary.
	std::vector<uint8_t> tmp;
	try {
		tmp.resize(tmpBytes);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: failed to allocate %lu-byte CPU upload staging buffer", caller, (unsigned long)tmpBytes);
		return;
	}
	const uint8_t one8 = 255;
	const uint16_t one16 = dstFormat == MTLPixelFormatRGBA16Float ? 0x3c00u : 0xffffu;
	const float one32 = 1.0f;
	for (GLsizei row = 0; row < h; row++) {
		const uint8_t *s = src + (size_t)row * srcRowBytes;
		uint8_t *d = tmp.data() + (size_t)row * tmpRowBytes;
		for (GLsizei col = 0; col < w; col++, s += srcBpp, d += dstBpp) {
			for (int c = 0; c < dstComps; c++) {
				uint8_t *out = d + (size_t)c * dstCompSize;
				if (idx[c] >= 0) {
					memcpy(out, s + (size_t)idx[c] * srcCompSize, (size_t)dstCompSize);
				} else if (c == 3) {
					const void *one = dstCompSize == 1 ? (const void *)&one8 :
						(dstCompSize == 2 ? (const void *)&one16 : (const void *)&one32);
					memcpy(out, one, (size_t)dstCompSize);
				} else {
					memset(out, 0, (size_t)dstCompSize);
				}
			}
		}
	}
	flushCommandBuffer(true); // preserve GL ordering before mutating a Shared texture used by earlier draws.
	[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:tmp.data() bytesPerRow:tmpRowBytes bytesPerImage:0];
}

// glCopyTexSubImage2D reads the attachment matching the destination base format:
// colour textures read GL_READ_BUFFER; depth textures read the read framebuffer's depth attachment.
// The default drawable is framebuffer-only, so it cannot be used as a copy source here.
static id<MTLTexture> readAttachment(GLint &level, bool depth) {
	level = 0;
	if (ctx.read_fbo == 0) return nil;
	auto it = ctx.fbo_objects.find(ctx.read_fbo);
	if (it == ctx.fbo_objects.end()) return nil;
	const FBOAttachment *attachment = nullptr;
	if (depth) {
		attachment = &it->second.depth;
	} else {
		const GLenum readBuffer = it->second.read_buffer;
		if (readBuffer < GL_COLOR_ATTACHMENT0 || readBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) return nil;
		attachment = &it->second.color[readBuffer - GL_COLOR_ATTACHMENT0];
	}
	if (!attachment || attachment->texture == 0) return nil;
	auto texture = ctx.texture_objects.find(attachment->texture);
	if (texture == ctx.texture_objects.end() || !texture->second.texture) return nil;
	id<MTLTexture> mtlTexture = texture->second.texture;
	if (attachment->level < 0 || (NSUInteger)attachment->level >= mtlTexture.mipmapLevelCount ||
		mtlTexture.textureType != MTLTextureType2D || (mtlTexture.usage & MTLTextureUsageRenderTarget) == 0) return nil;
	level = attachment->level;
	return mtlTexture;
}

// MARK: - Texture objects

void glGenTextures(GLsizei n, GLuint *textures) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = ++ctx.next_texture_id;
		ctx.texture_objects[id] = TextureObject();
		textures[i] = id;
	}
}

void glDeleteTextures(GLsizei n, const GLuint *textures) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = textures[i];
		if (id == 0) {
			continue;
		}
		for (auto &unit : ctx.bound_textures) {
			for (auto &binding : unit) {
				if (binding.second == id) {
					binding.second = 0;
				}
			}
		}
		// GL detaches a deleted texture from the framebuffers it is attached to.
		bool detached = false;
		auto detach = [&](FBOAttachment &a) {
			if (a.texture == id) {
				a.texture = 0;
				detached = true;
			}
		};
		for (auto &fbo : ctx.fbo_objects) {
			for (auto &a : fbo.second.color) {
				detach(a);
			}
			detach(fbo.second.depth);
			detach(fbo.second.stencil);
		}
		if (detached) {
			endRenderEncoder();
		}
		ctx.texture_objects.erase(id);
	}
}

GLboolean glIsTexture(GLuint texture) {
	auto it = ctx.texture_objects.find(texture);
	return texture != 0 && it != ctx.texture_objects.end() && it->second.target != 0; // a name only becomes a texture when first bound
}

void glActiveTexture(GLenum texture) {
	if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + GL4METAL_MAX_TEXTURE_UNITS) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glActiveTexture: invalid texture unit 0x%X", texture);
		return;
	}
	ctx.active_texture_unit = texture - GL_TEXTURE0;
}

void glBindTexture(GLenum target, GLuint texture) {
	if (texture != 0) {
		TextureObject &t = ctx.texture_objects[texture]; // creates the object when the name was never generated
		if (t.target == 0) {
			t.target = target;
		} else if (t.target != target) {
			GL4METAL_LOG("glBindTexture: texture %u was created as 0x%X, can't bind it to 0x%X", texture, t.target, target);
			return;
		}
	}
	ctx.bound_textures[ctx.active_texture_unit][target] = texture;
}

void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer) {
	GL4METAL_LOG("glTexBuffer: Unimplemented");
}

void glGenerateMipmap(GLenum target) {
	TextureObject *tex = boundTexture(target);
	if (!tex || !tex->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGenerateMipmap: no allocated texture bound to target 0x%X (GL_INVALID_OPERATION)", target);
		return;
	}
	if (tex->texture.mipmapLevelCount < 2 || isDepthStencilFormat(tex->texture.pixelFormat)) {
		return; // nothing to generate (levels are allocated from GL_TEXTURE_MAX_LEVEL / the min filter)
	}
	endRenderEncoder();
	if (!ctx.commandQueue) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGenerateMipmap: Metal command queue is unavailable");
		return;
	}
	if (!ctx.currentCommandBuffer) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	if (!ctx.currentCommandBuffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGenerateMipmap: failed to create Metal command buffer");
		return;
	}
	id<MTLBlitCommandEncoder> blit = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!blit) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGenerateMipmap: failed to create Metal blit encoder");
		return;
	}
	[blit generateMipmapsForTexture:tex->texture];
	[blit endEncoding];
}

// MARK: - Pixel storage

void glPixelStoref(GLenum pname, GLfloat param) {
	glPixelStorei(pname, (GLint)param);
}

void glPixelStorei(GLenum pname, GLint param) {
	switch (pname) {
		case GL_UNPACK_ALIGNMENT:
		case GL_PACK_ALIGNMENT:
			if (param != 1 && param != 2 && param != 4 && param != 8) {
				gl4metalSetError(GL_INVALID_VALUE);
				GL4METAL_LOG("glPixelStorei: invalid alignment %d", param);
				return;
			}
			(pname == GL_UNPACK_ALIGNMENT ? ctx.unpack_alignment : ctx.pack_alignment) = param;
			break;
		case GL_UNPACK_ROW_LENGTH:
		case GL_UNPACK_SKIP_PIXELS:
		case GL_UNPACK_SKIP_ROWS:
		case GL_PACK_ROW_LENGTH:
		case GL_PACK_SKIP_PIXELS:
		case GL_PACK_SKIP_ROWS:
			if (param < 0) {
				gl4metalSetError(GL_INVALID_VALUE);
				GL4METAL_LOG("glPixelStorei: negative row length/skip value %d for pname 0x%X", param, pname);
				return;
			}
			switch (pname) {
				case GL_UNPACK_ROW_LENGTH:  ctx.unpack_row_length = param; break;
				case GL_UNPACK_SKIP_PIXELS: ctx.unpack_skip_pixels = param; break;
				case GL_UNPACK_SKIP_ROWS:   ctx.unpack_skip_rows = param; break;
				case GL_PACK_ROW_LENGTH:    ctx.pack_row_length = param; break;
				case GL_PACK_SKIP_PIXELS:  ctx.pack_skip_pixels = param; break;
				case GL_PACK_SKIP_ROWS:    ctx.pack_skip_rows = param; break;
			}
			break;
		case GL_PACK_SWAP_BYTES:
			ctx.pack_swap_bytes = param ? GL_TRUE : GL_FALSE;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glPixelStorei: unhandled pname 0x%X", pname);
			break;
	}
}

// MARK: - Texture images

void glTexImage1D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels) {
	GL4METAL_LOG("glTexImage1D: Unimplemented");
}

void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels) {
	MTLPixelFormat mtlFormat = metalPixelFormat((GLenum)internalformat);
	// OpenGL permits zero-sized images (including empty mip levels); only negative dimensions are invalid.
	bool valid = level >= 0 && border == 0 && mtlFormat != MTLPixelFormatInvalid &&
				 width >= 0 && height >= 0 && width <= GL4METAL_MAX_TEXTURE_SIZE && height <= GL4METAL_MAX_TEXTURE_SIZE;

	if (target == GL_PROXY_TEXTURE_2D) {
		// Only records whether a texture of this size/format could be created (Minecraft probes the max size this way).
		ctx.proxy_texture_2d.w = valid ? width : 0;
		ctx.proxy_texture_2d.h = valid ? height : 0;
		ctx.proxy_texture_2d.internalformat = valid ? (GLenum)internalformat : 0;
		return;
	}
	int face = cubeFace(target); // -1: a plain 2D texture
	if (target != GL_TEXTURE_2D && face < 0) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexImage2D: unsupported target 0x%X", target);
		return;
	}
	TextureObject *t = boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D);
	if (!t) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage2D: no texture bound to target 0x%X", face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D);
		return;
	}
	if (face >= 0 && width != height) {
		GL4METAL_LOG("glTexImage2D: cube map faces must be square (%dx%d)", width, height);
		return;
	}
	if (!valid) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage2D: invalid level %d, size %dx%d or internalformat 0x%X", level, width, height, internalformat);
		return;
	}

	if (width == 0 || height == 0) {
		// A zero-sized level has no Metal storage. At level 0, clear any old allocation; for an empty non-base mip,
		// treat the definition as a no-op because Metal's mip chain is fixed when the texture is allocated.
		if (level == 0) {
			endRenderEncoder();
			t->texture = nil;
			t->width = width;
			t->height = height;
			t->internalformat = (GLenum)internalformat;
		}
		return;
	}

	if (level == 0) {
		if (!t->texture || t->width != width || t->height != height || t->texture.pixelFormat != mtlFormat) {
			NSUInteger levels = mipLevelCount(*t, width, height);
			MTLTextureDescriptor *d = face >= 0
				? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:mtlFormat size:(NSUInteger)width mipmapped:levels > 1]
				: [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtlFormat
																	  width:(NSUInteger)width
																	 height:(NSUInteger)height
																  mipmapped:levels > 1];
			d.mipmapLevelCount = levels;
			d.usage = MTLTextureUsageShaderRead;
			if (mtlFormat != MTLPixelFormatRGB9E5Float) {
				d.usage = d.usage | MTLTextureUsageRenderTarget; // RGB9E5 is not renderable
			}
			d.storageMode = isDepthStencilFormat(mtlFormat) ? MTLStorageModePrivate : MTLStorageModeShared;
			id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:d];
			if (!newTexture) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("glTexImage2D: failed to allocate a %dx%d MTLPixelFormat %d texture; previous storage preserved", width, height, (int)mtlFormat);
				return;
			}
			endRenderEncoder(); // a pass that renders to the old texture must not outlive it
			flushCommandBuffer(true); // the old image may still be sampled by committed work before name replacement
			t->texture = newTexture;
			t->width = width;
			t->height = height;
		}
		t->internalformat = (GLenum)internalformat;
	} else if (!t->texture || (NSUInteger)level >= t->texture.mipmapLevelCount ||
			   width != std::max(1, t->width >> level) || height != std::max(1, t->height >> level)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage2D: level %d (%dx%d) does not fit the allocated mip chain", level, width, height);
		return;
	}

	if (pixels || boundBuffer(GL_PIXEL_UNPACK_BUFFER)) {
		uploadPixels(*t, level, 0, 0, width, height, format, type, pixels, "glTexImage2D", face >= 0 ? (NSUInteger)face : 0);
	}
}

void glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels) {
	GL4METAL_LOG("glTexImage3D: Unimplemented");
}

void glTexImage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) {
	GL4METAL_LOG("glTexImage2DMultisample: Unimplemented");
}

void glTexImage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) {
	GL4METAL_LOG("glTexImage3DMultisample: Unimplemented");
}

void glTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels) {
	GL4METAL_LOG("glTexSubImage1D: Unimplemented");
}

void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
	int face = cubeFace(target);
	TextureObject *t = (target == GL_TEXTURE_2D || face >= 0) ? boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D) : nullptr;
	if (!t || !t->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexSubImage2D: no allocated texture bound (target 0x%X)", target);
		return;
	}
	uploadPixels(*t, level, xoffset, yoffset, width, height, format, type, pixels, "glTexSubImage2D", face >= 0 ? (NSUInteger)face : 0);
}

void glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) {
	GL4METAL_LOG("glTexSubImage3D: Unimplemented");
}

void glCopyTexImage1D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border) {
	GL4METAL_LOG("glCopyTexImage1D: Unimplemented");
}

void glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border) {
	glTexImage2D(target, level, (GLint)internalformat, width, height, border, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glCopyTexSubImage2D(target, level, 0, 0, x, y, width, height);
}

void glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width) {
	GL4METAL_LOG("glCopyTexSubImage1D: Unimplemented");
}

void glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height) {
	int face = cubeFace(target);
	TextureObject *dst = (target == GL_TEXTURE_2D || face >= 0) ? boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D) : nullptr;
	if (!dst || !dst->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: destination texture is not allocated");
		return;
	}
	if (level < 0 || (NSUInteger)level >= dst->texture.mipmapLevelCount || width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCopyTexSubImage2D: invalid mip level or copy size");
		return;
	}
	if (width == 0 || height == 0) return;
	const bool depthCopy = isDepthStencilFormat(dst->texture.pixelFormat);
	GLint srcLevel = 0;
	id<MTLTexture> src = readAttachment(srcLevel, depthCopy);
	if (!src) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: read framebuffer has no usable %s attachment (the default framebuffer is not readable)", depthCopy ? "depth" : "color");
		return;
	}
	NSUInteger srcW = std::max<NSUInteger>(src.width >> srcLevel, 1), srcH = std::max<NSUInteger>(src.height >> srcLevel, 1);
	GLsizei dstW = std::max(1, dst->width >> level), dstH = std::max(1, dst->height >> level);
	if (x < 0 || y < 0 || xoffset < 0 || yoffset < 0 ||
		(NSUInteger)x > srcW || (NSUInteger)y > srcH || (NSUInteger)width > srcW - (NSUInteger)x || (NSUInteger)height > srcH - (NSUInteger)y ||
		xoffset > dstW || yoffset > dstH || width > dstW - xoffset || height > dstH - yoffset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCopyTexSubImage2D: the rectangle does not fit the source or destination (GL_INVALID_VALUE)");
		return;
	}
	if (depthCopy != isDepthStencilFormat(src.pixelFormat)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: selected read attachment is not a %s attachment", depthCopy ? "depth" : "color");
		return;
	}
	if (src.pixelFormat != dst->texture.pixelFormat) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: source/destination pixel formats differ (%d -> %d); conversion is not implemented", (int)src.pixelFormat, (int)dst->texture.pixelFormat);
		return;
	}
	if (src == dst->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: source and destination alias the same Metal texture; copy skipped");
		return;
	}
	endRenderEncoder();
	if (!ctx.commandQueue) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: Metal command queue is unavailable");
		return;
	}
	if (!ctx.currentCommandBuffer) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	if (!ctx.currentCommandBuffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	id<MTLBlitCommandEncoder> blit = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!blit) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glCopyTexSubImage2D: failed to create Metal blit encoder");
		return;
	}
	[blit copyFromTexture:src sourceSlice:0 sourceLevel:(NSUInteger)srcLevel sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
			   sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
			    toTexture:dst->texture destinationSlice:(face >= 0 ? (NSUInteger)face : 0) destinationLevel:(NSUInteger)level
		destinationOrigin:MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, 0)];
	[blit endEncoding];
}

void glCopyTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height) {
	GL4METAL_LOG("glCopyTexSubImage3D: Unimplemented");
}

// MARK: - Compressed texture images

void glCompressedTexImage1D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexImage1D: Unimplemented");
}

void glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexImage2D: Unimplemented");
}

void glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexImage3D: Unimplemented");
}

void glCompressedTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexSubImage1D: Unimplemented");
}

void glCompressedTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexSubImage2D: Unimplemented");
}

void glCompressedTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data) {
	GL4METAL_LOG("glCompressedTexSubImage3D: Unimplemented");
}

// MARK: - Texture parameters

void glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
	TextureObject *t = boundTexture(target);
	if (!t) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexParameterf: no texture bound to target 0x%X", target);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:           t->min_lod = param; break;
		case GL_TEXTURE_MAX_LOD:           t->max_lod = param; break;
		case GL_TEXTURE_LOD_BIAS:          t->lod_bias = param; break;
		case GL_TEXTURE_MAX_ANISOTROPY_EXT: t->max_anisotropy = std::max(param, 1.0f); break;
		default:
			glTexParameteri(target, pname, (GLint)param); // enum-valued parameters passed as float
			break;
	}
}

void glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) {
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		return; // clamp-to-border colours are not available, see addressMode in draw.mm
	}
	glTexParameterf(target, pname, params[0]);
}

void glTexParameteri(GLenum target, GLenum pname, GLint param) {
	TextureObject *t = boundTexture(target);
	if (!t) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexParameteri: no texture bound to target 0x%X", target);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:   t->min_filter = (GLenum)param; break;
		case GL_TEXTURE_MAG_FILTER:   t->mag_filter = (GLenum)param; break;
		case GL_TEXTURE_WRAP_S:       t->wrap_s = (GLenum)param; break;
		case GL_TEXTURE_WRAP_T:       t->wrap_t = (GLenum)param; break;
		case GL_TEXTURE_WRAP_R:       t->wrap_r = (GLenum)param; break;
		case GL_TEXTURE_BASE_LEVEL:   t->base_level = param; break;
		case GL_TEXTURE_MAX_LEVEL:    t->max_level = param; break;
		case GL_TEXTURE_COMPARE_MODE: t->compare_mode = (GLenum)param; break;
		case GL_TEXTURE_COMPARE_FUNC: t->compare_func = (GLenum)param; break;
		case GL_TEXTURE_MIN_LOD:
		case GL_TEXTURE_MAX_LOD:
		case GL_TEXTURE_LOD_BIAS:
		case GL_TEXTURE_MAX_ANISOTROPY_EXT:
			glTexParameterf(target, pname, (GLfloat)param);
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glTexParameteri: unhandled pname 0x%X", pname);
			break;
	}
}

void glTexParameteriv(GLenum target, GLenum pname, const GLint *params) {
	if (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA) {
		return; // border colours and swizzles are not supported yet
	}
	glTexParameteri(target, pname, params[0]);
}

void glTexParameterIiv(GLenum target, GLenum pname, const GLint *params) {
	glTexParameteriv(target, pname, params);
}

void glTexParameterIuiv(GLenum target, GLenum pname, const GLuint *params) {
	glTexParameteriv(target, pname, (const GLint *)params);
}

// MARK: - Texture queries

namespace {

static bool getTexImageBytesPerPixel(MTLPixelFormat format, NSUInteger &bytesPerPixel) {
	switch (format) {
		case MTLPixelFormatR8Unorm: bytesPerPixel = 1; return true;
		case MTLPixelFormatRG8Unorm: bytesPerPixel = 2; return true;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatRGB10A2Unorm:
		case MTLPixelFormatRG11B10Float:
		case MTLPixelFormatRGB9E5Float: bytesPerPixel = 4; return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatDepth16Unorm: bytesPerPixel = 2; return true;
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Float: bytesPerPixel = 4; return true;
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Float: bytesPerPixel = 8; return true;
		case MTLPixelFormatR32Float:
		case MTLPixelFormatDepth32Float: bytesPerPixel = 4; return true;
		case MTLPixelFormatRG32Float: bytesPerPixel = 8; return true;
		case MTLPixelFormatRGBA32Float: bytesPerPixel = 16; return true;
		case MTLPixelFormatDepth32Float_Stencil8: bytesPerPixel = 4; return true; // depth aspect only; stencil is a separate blit plane
		default: return false;
	}
}

static bool readTextureLevel(id<MTLTexture> texture, GLint level, NSUInteger slice,
						 NSUInteger width, NSUInteger height, NSUInteger bytesPerPixel,
						 MTLBlitOption blitOptions, std::vector<uint8_t> &out) {
	if (!texture || width == 0 || height == 0 || bytesPerPixel == 0) return false;
	if (width > SIZE_MAX / bytesPerPixel) return false;
	const NSUInteger rowBytes = width * bytesPerPixel;
	if (height > SIZE_MAX / rowBytes) return false;
	const NSUInteger totalBytes = rowBytes * height;
	try {
		out.resize((size_t)totalBytes);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGetTexImage: CPU readback allocation failed (%lu bytes)", (unsigned long)totalBytes);
		return false;
	}

	// A shared texture is CPU-readable after all earlier GPU commands have completed.
	flushCommandBuffer(true);
	// MTLStorageModeManaged does not available on iOS
	if (texture.storageMode == MTLStorageModeShared) {
		[texture getBytes:out.data() bytesPerRow:rowBytes bytesPerImage:0
			fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:(NSUInteger)level slice:slice];
		return true;
	}

	// Private depth/render-target textures must be copied through a shared buffer on Apple GPUs.
	const NSUInteger alignment = 256; // Metal blit texture-to-buffer row alignment
	if (rowBytes > SIZE_MAX - (alignment - 1)) return false;
	const NSUInteger stagingRowBytes = (rowBytes + alignment - 1) & ~(alignment - 1);
	if (height > SIZE_MAX / stagingRowBytes) return false;
	const NSUInteger stagingBytes = stagingRowBytes * height;
	if (stagingBytes == 0 || stagingBytes > texture.device.maxBufferLength) {
		GL4METAL_LOG("glGetTexImage: private texture readback exceeds maxBufferLength (%lu bytes)", (unsigned long)stagingBytes);
		return false;
	}
	id<MTLBuffer> staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
	if (!staging) return false;
	id<MTLCommandQueue> queue = ctx.commandQueue;
	id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
	if (!commandBuffer || !blit) return false;
	[blit copyFromTexture:texture sourceSlice:slice sourceLevel:(NSUInteger)level
		sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
		toBuffer:staging destinationOffset:0 destinationBytesPerRow:stagingRowBytes
		destinationBytesPerImage:0 options:blitOptions];
	[blit endEncoding];
	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];
	if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
		GL4METAL_LOG("glGetTexImage: private texture blit readback failed: %s",
			commandBuffer.error.localizedDescription.UTF8String ?: "unknown Metal error");
		return false;
	}
	const uint8_t *src = (const uint8_t *)staging.contents;
	for (NSUInteger y = 0; y < height; ++y) {
		memcpy(out.data() + y * rowBytes, src + y * stagingRowBytes, rowBytes);
	}
	return true;
}

static float halfToFloat(uint16_t h) {
	const int sign = (h & 0x8000u) ? -1 : 1;
	const unsigned exponent = (h >> 10) & 0x1fu;
	const unsigned mantissa = h & 0x3ffu;
	if (exponent == 0) {
		if (mantissa == 0) return sign < 0 ? -0.0f : 0.0f;
		return (float)(sign * (double)mantissa * std::ldexp(1.0, -24));
	}
	if (exponent == 31) {
		if (mantissa == 0) return sign < 0 ? -INFINITY : INFINITY;
		return NAN;
	}
	return (float)(sign * (1.0 + (double)mantissa / 1024.0) * std::ldexp(1.0, (int)exponent - 15));
}

static uint16_t floatToHalf(float value) {
	uint32_t bits = 0;
	memcpy(&bits, &value, sizeof(bits));
	const uint32_t sign = (bits >> 16) & 0x8000u;
	const uint32_t mantissa = bits & 0x7fffffu;
	int exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
	if (((bits >> 23) & 0xffu) == 0xffu) {
		return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
	}
	if (exponent <= 0) {
		if (exponent < -10) return (uint16_t)sign;
		uint32_t m = mantissa | 0x800000u;
		const unsigned shift = (unsigned)(14 - exponent);
		uint32_t rounded = (m + (1u << (shift - 1))) >> shift;
		return (uint16_t)(sign | rounded);
	}
	if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
	uint32_t roundedMantissa = mantissa + 0x1000u;
	if (roundedMantissa & 0x800000u) {
		roundedMantissa = 0;
		if (++exponent >= 31) return (uint16_t)(sign | 0x7c00u);
	}
	return (uint16_t)(sign | ((uint32_t)exponent << 10) | (roundedMantissa >> 13));
}

static float decodeUnsignedFloat(uint32_t bits, unsigned mantissaBits) {
	const unsigned mantissaMask = (1u << mantissaBits) - 1u;
	const unsigned mantissa = bits & mantissaMask;
	const unsigned exponent = (bits >> mantissaBits) & 0x1fu;
	if (exponent == 0) return std::ldexp((float)mantissa, 1 - 15 - (int)mantissaBits);
	if (exponent == 31) return mantissa == 0 ? INFINITY : NAN;
	return std::ldexp(1.0f + (float)mantissa / (float)(1u << mantissaBits), (int)exponent - 15);
}

static bool decodeTexturePixel(MTLPixelFormat format, const uint8_t *src, float rgba[4]) {
	rgba[0] = rgba[1] = rgba[2] = 0.0f;
	rgba[3] = 1.0f;
	auto unorm8 = [](uint8_t x) { return (float)x / 255.0f; };
	auto unorm16 = [](uint16_t x) { return (float)x / 65535.0f; };
	switch (format) {
		case MTLPixelFormatR8Unorm: rgba[0] = unorm8(src[0]); return true;
		case MTLPixelFormatRG8Unorm: rgba[0] = unorm8(src[0]); rgba[1] = unorm8(src[1]); return true;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
			for (int i = 0; i < 4; ++i) rgba[i] = unorm8(src[i]);
			return true;
		case MTLPixelFormatBGRA8Unorm:
			rgba[0] = unorm8(src[2]); rgba[1] = unorm8(src[1]); rgba[2] = unorm8(src[0]); rgba[3] = unorm8(src[3]);
			return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatDepth16Unorm: {
			uint16_t v; memcpy(&v, src, sizeof(v)); rgba[0] = unorm16(v); return true;
		}
		case MTLPixelFormatRG16Unorm: {
			uint16_t v[2]; memcpy(v, src, sizeof(v)); rgba[0] = unorm16(v[0]); rgba[1] = unorm16(v[1]); return true;
		}
		case MTLPixelFormatRGBA16Unorm: {
			uint16_t v[4]; memcpy(v, src, sizeof(v)); for (int i = 0; i < 4; ++i) rgba[i] = unorm16(v[i]); return true;
		}
		case MTLPixelFormatR16Float: {
			uint16_t v; memcpy(&v, src, sizeof(v)); rgba[0] = halfToFloat(v); return true;
		}
		case MTLPixelFormatRG16Float: {
			uint16_t v[2]; memcpy(v, src, sizeof(v)); rgba[0] = halfToFloat(v[0]); rgba[1] = halfToFloat(v[1]); return true;
		}
		case MTLPixelFormatRGBA16Float: {
			uint16_t v[4]; memcpy(v, src, sizeof(v)); for (int i = 0; i < 4; ++i) rgba[i] = halfToFloat(v[i]); return true;
		}
		case MTLPixelFormatR32Float:
		case MTLPixelFormatDepth32Float:
		case MTLPixelFormatDepth32Float_Stencil8: memcpy(&rgba[0], src, sizeof(float)); return true;
		case MTLPixelFormatRG32Float: memcpy(rgba, src, 2 * sizeof(float)); return true;
		case MTLPixelFormatRGBA32Float: memcpy(rgba, src, 4 * sizeof(float)); return true;
		case MTLPixelFormatRGB10A2Unorm: {
			uint32_t v; memcpy(&v, src, sizeof(v));
			rgba[0] = (float)(v & 0x3ffu) / 1023.0f;
			rgba[1] = (float)((v >> 10) & 0x3ffu) / 1023.0f;
			rgba[2] = (float)((v >> 20) & 0x3ffu) / 1023.0f;
			rgba[3] = (float)((v >> 30) & 0x3u) / 3.0f;
			return true;
		}
		case MTLPixelFormatRG11B10Float: {
			uint32_t v; memcpy(&v, src, sizeof(v));
			rgba[0] = decodeUnsignedFloat(v & 0x7ffu, 6);
			rgba[1] = decodeUnsignedFloat((v >> 11) & 0x7ffu, 6);
			rgba[2] = decodeUnsignedFloat((v >> 22) & 0x3ffu, 5);
			return true;
		}
		case MTLPixelFormatRGB9E5Float: {
			uint32_t v; memcpy(&v, src, sizeof(v));
			const int exponent = (int)((v >> 27) & 0x1fu) - 24;
			const float scale = std::ldexp(1.0f, exponent);
			rgba[0] = (float)(v & 0x1ffu) * scale;
			rgba[1] = (float)((v >> 9) & 0x1ffu) * scale;
			rgba[2] = (float)((v >> 18) & 0x1ffu) * scale;
			return true;
		}
		default: return false;
	}
}

static size_t packedTypeSize(GLenum type, int components) {
	switch (type) {
		case GL_UNSIGNED_INT_8_8_8_8:
		case GL_UNSIGNED_INT_8_8_8_8_REV:
		case GL_UNSIGNED_INT_10_10_10_2:
		case GL_UNSIGNED_INT_2_10_10_10_REV: return components == 4 ? 4 : 0;
		case GL_UNSIGNED_SHORT_5_6_5:
		case GL_UNSIGNED_SHORT_5_6_5_REV: return components == 3 ? 2 : 0;
		case GL_UNSIGNED_SHORT_4_4_4_4:
		case GL_UNSIGNED_SHORT_4_4_4_4_REV:
		case GL_UNSIGNED_SHORT_5_5_5_1:
		case GL_UNSIGNED_SHORT_1_5_5_5_REV: return components == 4 ? 2 : 0;
		case GL_UNSIGNED_INT_10F_11F_11F_REV:
		case GL_UNSIGNED_INT_5_9_9_9_REV: return components == 3 ? 4 : 0;
		default: return 0;
	}
}

static size_t scalarTypeSize(GLenum type) {
	switch (type) {
		case GL_BYTE: case GL_UNSIGNED_BYTE: return 1;
		case GL_SHORT: case GL_UNSIGNED_SHORT: case GL_HALF_FLOAT: return 2;
		case GL_INT: case GL_UNSIGNED_INT: case GL_FLOAT: return 4;
		default: return 0;
	}
}

static float componentForFormat(const float rgba[4], int component) {
	return rgba[component];
}

static void writeScalarComponent(uint8_t *dst, GLenum type, float value) {
	switch (type) {
		case GL_BYTE: {
			int8_t v = (int8_t)std::lrintf(std::max(-1.0f, std::min(1.0f, value)) * 127.0f); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_UNSIGNED_BYTE: {
			uint8_t v = (uint8_t)std::lrintf(std::max(0.0f, std::min(1.0f, value)) * 255.0f); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_SHORT: {
			int16_t v = (int16_t)std::lrintf(std::max(-1.0f, std::min(1.0f, value)) * 32767.0f); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_UNSIGNED_SHORT: {
			uint16_t v = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, value)) * 65535.0f); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_INT: {
			int32_t v = (int32_t)llround((double)std::max(-1.0f, std::min(1.0f, value)) * 2147483647.0); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_UNSIGNED_INT: {
			uint32_t v = (uint32_t)llround((double)std::max(0.0f, std::min(1.0f, value)) * 4294967295.0); memcpy(dst, &v, sizeof(v)); break;
		}
		case GL_HALF_FLOAT: { uint16_t v = floatToHalf(value); memcpy(dst, &v, sizeof(v)); break; }
		case GL_FLOAT: memcpy(dst, &value, sizeof(value)); break;
	}
}

static uint32_t encodeUnsignedFloat(float value, unsigned mantissaBits) {
	const uint32_t mantissaMax = (1u << mantissaBits) - 1u;
	if (std::isnan(value)) return (31u << mantissaBits) | (1u << (mantissaBits - 1));
	if (value <= 0.0f) return 0;
	if (!std::isfinite(value)) return 31u << mantissaBits;
	int exponent = 0;
	const float fraction = std::frexp(value, &exponent); // value = fraction * 2^exponent, fraction in [0.5, 1)
	int biased = exponent - 1 + 15;
	if (biased <= 0) {
		const float scaled = std::ldexp(value, 14 + (int)mantissaBits);
		const uint32_t subnormal = (uint32_t)std::max<double>(0.0, std::nearbyint(scaled));
		return subnormal > mantissaMax ? (1u << mantissaBits) : subnormal;
	}
	if (biased >= 31) return 31u << mantissaBits;
	uint32_t mantissa = (uint32_t)std::nearbyint(((fraction * 2.0f) - 1.0f) * (float)(1u << mantissaBits));
	if (mantissa > mantissaMax) {
		mantissa = 0;
		if (++biased >= 31) return 31u << mantissaBits;
	}
	return ((uint32_t)biased << mantissaBits) | mantissa;
}

static uint32_t encodeRGB9E5(const float values[4]) {
	const float r = std::min(65408.0f, std::max(0.0f, values[0]));
	const float g = std::min(65408.0f, std::max(0.0f, values[1]));
	const float b = std::min(65408.0f, std::max(0.0f, values[2]));
	const float maximum = std::max(r, std::max(g, b));
	int exponent = std::max(0, (int)std::floor(std::log2(maximum)) + 16);
	float scale = std::ldexp(1.0f, 24 - exponent);
	uint32_t mr = (uint32_t)std::nearbyint(r * scale);
	uint32_t mg = (uint32_t)std::nearbyint(g * scale);
	uint32_t mb = (uint32_t)std::nearbyint(b * scale);
	if (std::max(mr, std::max(mg, mb)) > 511u && exponent < 31) {
		++exponent;
		scale *= 0.5f;
		mr = (uint32_t)std::nearbyint(r * scale);
		mg = (uint32_t)std::nearbyint(g * scale);
		mb = (uint32_t)std::nearbyint(b * scale);
	}
	return (mr & 0x1ffu) | ((mg & 0x1ffu) << 9) | ((mb & 0x1ffu) << 18) | ((uint32_t)exponent << 27);
}

static void writePackedPixel(uint8_t *dst, GLenum type, const float values[4]) {
	if (type == GL_UNSIGNED_INT_8_8_8_8_REV || type == GL_UNSIGNED_INT_8_8_8_8) {
		uint32_t b[4];
		for (int i = 0; i < 4; ++i) b[i] = (uint32_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[i])) * 255.0f);
		uint32_t packed = type == GL_UNSIGNED_INT_8_8_8_8_REV
			? (b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24))
			: ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV) {
		uint16_t r = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[0])) * 31.0f);
		uint16_t g = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[1])) * 63.0f);
		uint16_t b = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[2])) * 31.0f);
		uint16_t packed = type == GL_UNSIGNED_SHORT_5_6_5_REV ? (uint16_t)(r | (g << 5) | (b << 11)) : (uint16_t)((r << 11) | (g << 5) | b);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
		uint16_t c[4]; for (int i = 0; i < 4; ++i) c[i] = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[i])) * 15.0f);
		uint16_t packed = type == GL_UNSIGNED_SHORT_4_4_4_4_REV ? (uint16_t)(c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12)) : (uint16_t)((c[0] << 12) | (c[1] << 8) | (c[2] << 4) | c[3]);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
		uint16_t r = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[0])) * 31.0f);
		uint16_t g = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[1])) * 31.0f);
		uint16_t b = (uint16_t)std::lrintf(std::max(0.0f, std::min(1.0f, values[2])) * 31.0f);
		uint16_t a = values[3] >= 0.5f ? 1 : 0;
		uint16_t packed = type == GL_UNSIGNED_SHORT_1_5_5_5_REV
			? (uint16_t)(r | (g << 5) | (b << 10) | (a << 15))
			: (uint16_t)((r << 11) | (g << 6) | (b << 1) | a);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV) {
		uint32_t r = (uint32_t)std::nearbyint(std::max(0.0f, std::min(1.0f, values[0])) * 1023.0f);
		uint32_t g = (uint32_t)std::nearbyint(std::max(0.0f, std::min(1.0f, values[1])) * 1023.0f);
		uint32_t b = (uint32_t)std::nearbyint(std::max(0.0f, std::min(1.0f, values[2])) * 1023.0f);
		uint32_t a = (uint32_t)std::nearbyint(std::max(0.0f, std::min(1.0f, values[3])) * 3.0f);
		uint32_t packed = type == GL_UNSIGNED_INT_2_10_10_10_REV
			? (r | (g << 10) | (b << 20) | (a << 30))
			: ((r << 22) | (g << 12) | (b << 2) | a);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
		uint32_t r = encodeUnsignedFloat(values[0], 6);
		uint32_t g = encodeUnsignedFloat(values[1], 6);
		uint32_t b = encodeUnsignedFloat(values[2], 5);
		uint32_t packed = r | (g << 11) | (b << 22);
		memcpy(dst, &packed, sizeof(packed));
	} else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
		uint32_t packed = encodeRGB9E5(values);
		memcpy(dst, &packed, sizeof(packed));
	}
}

static bool checkedImageSize(size_t width, size_t height, size_t pixelBytes, size_t rowLength,
						 GLint rowAlignment, GLint skipPixels, GLint skipRows,
						 size_t &rowStride, size_t &startOffset, size_t &requiredBytes) {
	if (pixelBytes == 0 || rowAlignment <= 0 || skipPixels < 0 || skipRows < 0) return false;
	if (rowLength == 0) rowLength = width;
	if (rowLength > SIZE_MAX / pixelBytes) return false;
	const size_t rawRow = rowLength * pixelBytes;
	const size_t align = (size_t)rowAlignment;
	if (rawRow > SIZE_MAX - (align - 1)) return false;
	rowStride = (rawRow + align - 1) / align * align;
	if (rowStride == 0) return false;
	if ((size_t)skipRows > SIZE_MAX / rowStride || (size_t)skipPixels > SIZE_MAX / pixelBytes) return false;
	const size_t skipY = (size_t)skipRows * rowStride;
	const size_t skipX = (size_t)skipPixels * pixelBytes;
	if (skipY > SIZE_MAX - skipX) return false;
	startOffset = skipY + skipX;
	if (height == 0 || width == 0) { requiredBytes = startOffset; return true; }
	if (height - 1 > (SIZE_MAX - startOffset) / rowStride) return false;
	const size_t lastRow = startOffset + (height - 1) * rowStride;
	if (width > (SIZE_MAX - lastRow) / pixelBytes) return false;
	requiredBytes = lastRow + width * pixelBytes;
	return true;
}

} // namespace

void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void *pixels) {
	const int face = cubeFace(target);
	const bool is2D = target == GL_TEXTURE_2D;
	if (!is2D && face < 0) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glGetTexImage: unsupported target 0x%X (use GL_TEXTURE_2D or a cube-map face)", target);
		return;
	}
	TextureObject *tex = boundTexture(is2D ? GL_TEXTURE_2D : GL_TEXTURE_CUBE_MAP);
	if (!tex) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: no texture bound for target 0x%X", target);
		return;
	}
	if (level < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!tex->texture || (NSUInteger)level >= tex->texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: texture image at level %d is not allocated", level);
		return;
	}
	const NSUInteger width = std::max<NSUInteger>(1, (NSUInteger)tex->width >> level);
	const NSUInteger height = std::max<NSUInteger>(1, (NSUInteger)tex->height >> level);
	const MTLPixelFormat mtlFormat = tex->texture.pixelFormat;
	const bool depthFormat = isDepthStencilFormat(mtlFormat);

	int components = 0;
	int channels[4] = {0, 1, 2, 3};
	switch (format) {
		case GL_RED: components = 1; channels[0] = 0; break;
		case GL_RG: components = 2; channels[0] = 0; channels[1] = 1; break;
		case GL_RGB: components = 3; channels[0] = 0; channels[1] = 1; channels[2] = 2; break;
		case GL_BGR: components = 3; channels[0] = 2; channels[1] = 1; channels[2] = 0; break;
		case GL_RGBA: components = 4; channels[0] = 0; channels[1] = 1; channels[2] = 2; channels[3] = 3; break;
		case GL_BGRA: components = 4; channels[0] = 2; channels[1] = 1; channels[2] = 0; channels[3] = 3; break;
		case GL_DEPTH_COMPONENT: components = 1; channels[0] = 0; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetTexImage: unsupported external format 0x%X", format);
			return;
	}
	if (depthFormat != (format == GL_DEPTH_COMPONENT)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: external format 0x%X is incompatible with MTLPixelFormat %d", format, (int)mtlFormat);
		return;
	}
	NSUInteger sourceBytesPerPixel = 0;
	if (!getTexImageBytesPerPixel(mtlFormat, sourceBytesPerPixel)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: readback for MTLPixelFormat %d is not implemented", (int)mtlFormat);
		return;
	}
	const size_t packedBytes = packedTypeSize(type, components);
	const size_t scalarBytes = scalarTypeSize(type);
	const bool packed = packedBytes != 0;
	if (!packed && scalarBytes == 0) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glGetTexImage: unsupported pixel type 0x%X", type);
		return;
	}
	const size_t outputPixelBytes = packed ? packedBytes : (size_t)components * scalarBytes;
	if (packed && (format == GL_DEPTH_COMPONENT ||
		((type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV ||
		  type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_5_9_9_9_REV) ? components != 3 : components != 4))) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: packed type 0x%X is incompatible with format 0x%X", type, format);
		return;
	}
	if (format == GL_DEPTH_COMPONENT && type != GL_UNSIGNED_BYTE && type != GL_BYTE && type != GL_UNSIGNED_SHORT &&
		type != GL_SHORT && type != GL_UNSIGNED_INT && type != GL_INT && type != GL_HALF_FLOAT && type != GL_FLOAT) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (format != GL_DEPTH_COMPONENT && packed && components != 3 && components != 4) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}

	const GLuint packBuffer = boundBuffer(GL_PIXEL_PACK_BUFFER);
	uint8_t *destination = (uint8_t *)pixels;
	NSUInteger bufferOffset = 0;
	id<MTLBuffer> packBufferObject = nil;
	if (packBuffer != 0) {
		auto it = ctx.buffer_objects.find(packBuffer);
		if (it == ctx.buffer_objects.end() || !it->second) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glGetTexImage: GL_PIXEL_PACK_BUFFER has no storage");
			return;
		}
		packBufferObject = it->second;
		bufferOffset = (NSUInteger)(uintptr_t)pixels;
		destination = (uint8_t *)packBufferObject.contents;
	} else if (!destination) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: pixels is null and no pixel-pack buffer is bound");
		return;
	}

	size_t rowStride = 0, startOffset = 0, requiredBytes = 0;
	if (!checkedImageSize((size_t)width, (size_t)height, outputPixelBytes,
		(size_t)ctx.pack_row_length, ctx.pack_alignment, ctx.pack_skip_pixels, ctx.pack_skip_rows,
		rowStride, startOffset, requiredBytes) || bufferOffset > SIZE_MAX - requiredBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetTexImage: pixel-pack layout overflows addressable memory");
		return;
	}
	if (packBufferObject) {
		if (bufferOffset > packBufferObject.length || requiredBytes > packBufferObject.length - bufferOffset) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glGetTexImage: pixel-pack buffer range %lu+%lu exceeds %lu bytes",
				(unsigned long)bufferOffset, (unsigned long)requiredBytes, (unsigned long)packBufferObject.length);
			return;
		}
	}
	destination += bufferOffset + startOffset;

	std::vector<uint8_t> source;
	const MTLBlitOption readbackOptions = mtlFormat == MTLPixelFormatDepth32Float_Stencil8
		? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone;
	if (!readTextureLevel(tex->texture, level, face >= 0 ? (NSUInteger)face : 0,
					  width, height, sourceBytesPerPixel, readbackOptions, source)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: failed to read MTLPixelFormat %d at level %d", (int)mtlFormat, level);
		return;
	}
	for (NSUInteger y = 0; y < height; ++y) {
		const uint8_t *srcRow = source.data() + y * width * sourceBytesPerPixel;
		uint8_t *dstRow = destination + y * rowStride;
		for (NSUInteger x = 0; x < width; ++x) {
			const uint8_t *srcPixel = srcRow + x * sourceBytesPerPixel;
			float rgba[4];
			if (!decodeTexturePixel(mtlFormat, srcPixel, rgba)) {
				gl4metalSetError(GL_INVALID_OPERATION);
				return;
			}
			uint8_t *dstPixel = dstRow + x * outputPixelBytes;
			if (packed) {
				float values[4] = {0, 0, 0, 1};
				for (int c = 0; c < components; ++c) values[c] = componentForFormat(rgba, channels[c]);
				writePackedPixel(dstPixel, type, values);
			} else {
				for (int c = 0; c < components; ++c) {
					writeScalarComponent(dstPixel + (size_t)c * scalarBytes, type, componentForFormat(rgba, channels[c]));
				}
				if (ctx.pack_swap_bytes && scalarBytes > 1) {
					for (int c = 0; c < components; ++c) {
						uint8_t *component = dstPixel + (size_t)c * scalarBytes;
						for (size_t b = 0; b < scalarBytes / 2; ++b) std::swap(component[b], component[scalarBytes - 1 - b]);
					}
				}
			}
			if (packed && ctx.pack_swap_bytes && outputPixelBytes > 1) {
				for (size_t b = 0; b < outputPixelBytes / 2; ++b) std::swap(dstPixel[b], dstPixel[outputPixelBytes - 1 - b]);
			}
		}
	}
}

void glGetCompressedTexImage(GLenum target, GLint level, void *img) {
	GL4METAL_LOG("glGetCompressedTexImage: Unimplemented");
}

void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
	TextureObject *tex = boundTexture(target);
	if (tex && pname == GL_TEXTURE_MIN_LOD)  { *params = tex->min_lod; return; }
	if (tex && pname == GL_TEXTURE_MAX_LOD)  { *params = tex->max_lod; return; }
	if (tex && pname == GL_TEXTURE_LOD_BIAS) { *params = tex->lod_bias; return; }
	GLint value = 0;
	glGetTexParameteriv(target, pname, &value);
	*params = (GLfloat)value;
}

void glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
	TextureObject *tex = boundTexture(target);
	if (!tex) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexParameteriv: no texture bound to target 0x%X (GL_INVALID_OPERATION)", target);
		*params = 0;
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:   *params = (GLint)tex->min_filter; break;
		case GL_TEXTURE_MAG_FILTER:   *params = (GLint)tex->mag_filter; break;
		case GL_TEXTURE_WRAP_S:       *params = (GLint)tex->wrap_s; break;
		case GL_TEXTURE_WRAP_T:       *params = (GLint)tex->wrap_t; break;
		case GL_TEXTURE_WRAP_R:       *params = (GLint)tex->wrap_r; break;
		case GL_TEXTURE_BASE_LEVEL:   *params = tex->base_level; break;
		case GL_TEXTURE_MAX_LEVEL:    *params = tex->max_level; break;
		case GL_TEXTURE_COMPARE_MODE: *params = (GLint)tex->compare_mode; break;
		case GL_TEXTURE_COMPARE_FUNC: *params = (GLint)tex->compare_func; break;
		case GL_TEXTURE_MIN_LOD:      *params = (GLint)tex->min_lod; break;
		case GL_TEXTURE_MAX_LOD:      *params = (GLint)tex->max_lod; break;
		case GL_TEXTURE_LOD_BIAS:     *params = (GLint)tex->lod_bias; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetTexParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetTexParameterIiv(GLenum target, GLenum pname, GLint *params) {
	glGetTexParameteriv(target, pname, params);
}

void glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint *params) {
	glGetTexParameteriv(target, pname, (GLint *)params);
}

void glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat *params) {
	GLint value = 0;
	glGetTexLevelParameteriv(target, level, pname, &value);
	*params = (GLfloat)value;
}

void glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint *params) {
	GLsizei w = 0, h = 0;
	GLenum internalformat = 0;

	if (target == GL_PROXY_TEXTURE_2D) {
		w = ctx.proxy_texture_2d.w;
		h = ctx.proxy_texture_2d.h;
		internalformat = ctx.proxy_texture_2d.internalformat;
	} else {
		TextureObject *t = boundTexture(cubeFace(target) >= 0 ? GL_TEXTURE_CUBE_MAP : target);
		if (t && t->texture && level >= 0 && (NSUInteger)level < t->texture.mipmapLevelCount) {
			w = std::max(1, t->width >> level);
			h = std::max(1, t->height >> level);
			internalformat = t->internalformat;
		}
	}

	switch (pname) {
		case GL_TEXTURE_WIDTH:           *params = w; break;
		case GL_TEXTURE_HEIGHT:          *params = h; break;
		case GL_TEXTURE_DEPTH:           *params = (w > 0) ? 1 : 0; break;
		case GL_TEXTURE_INTERNAL_FORMAT: *params = (GLint)internalformat; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetTexLevelParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}
