#import "texture.h"

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
		case GL_R16F:
			return MTLPixelFormatR16Float;
		case GL_RG16F:
			return MTLPixelFormatRG16Float;
		case GL_RGB16F:
		case GL_RGBA16F:
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

// GL_PIXEL_UNPACK_BUFFER bound: `pixels` is a byte offset into that buffer.
static const uint8_t *resolveUnpackPointer(const void *pixels) {
	GLuint pbo = boundBuffer(GL_PIXEL_UNPACK_BUFFER);
	if (pbo == 0) {
		return (const uint8_t *)pixels;
	}
	auto it = ctx.buffer_objects.find(pbo);
	if (it == ctx.buffer_objects.end() || !it->second) {
		return nullptr;
	}
	return (const uint8_t *)it->second.contents + (uintptr_t)pixels;
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
		case MTLPixelFormatR32Float:        comps = 1; compSize = 4; return true;
		case MTLPixelFormatRG32Float:       comps = 2; compSize = 4; return true;
		case MTLPixelFormatRGBA32Float:     comps = 4; compSize = 4; return true;
		default: return false;
	}
}

static void uploadPixels(TextureObject &t, GLint level, GLint x, GLint y, GLsizei w, GLsizei h,
						 GLenum format, GLenum type, const void *pixels, const char *caller, NSUInteger slice = 0) {
	GLsizei lw = std::max(1, t.width >> level), lh = std::max(1, t.height >> level);
	if (level < 0 || (NSUInteger)level >= t.texture.mipmapLevelCount || x < 0 || y < 0 || w < 0 || h < 0 || x + w > lw || y + h > lh) {
		GL4METAL_LOG("%s: region %d,%d %dx%d does not fit level %d (%dx%d)", caller, x, y, w, h, level, lw, lh);
		return;
	}
	if (w == 0 || h == 0) {
		return;
	}
	const uint8_t *src = resolveUnpackPointer(pixels);
	if (!src) {
		return;
	}

	int srcComps = 0, idx[4];
	int dstComps = 0, dstCompSize = 0;
	int srcCompSize = (type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV) ? 1 : (type == GL_FLOAT ? 4 : 0);
	if (!sourceLayout(format, srcComps, idx) || srcCompSize == 0 || !destLayout(t.texture.pixelFormat, dstComps, dstCompSize) ||
		srcCompSize != dstCompSize || (type == GL_UNSIGNED_INT_8_8_8_8_REV && srcComps != 4)) {
		GL4METAL_LOG("%s: unsupported upload, format 0x%X type 0x%X into MTLPixelFormat %d", caller, format, type, (int)t.texture.pixelFormat);
		return;
	}

	// Source row layout, honouring GL_UNPACK_ROW_LENGTH / ALIGNMENT / SKIP_*.
	size_t srcBpp = (size_t)srcComps * srcCompSize;
	size_t rowLength = ctx.unpack_row_length > 0 ? (size_t)ctx.unpack_row_length : (size_t)w;
	size_t align = ctx.unpack_alignment > 0 ? (size_t)ctx.unpack_alignment : 1;
	size_t srcRowBytes = (rowLength * srcBpp + align - 1) / align * align;
	src += (size_t)ctx.unpack_skip_rows * srcRowBytes + (size_t)ctx.unpack_skip_pixels * srcBpp;

	size_t dstBpp = (size_t)dstComps * dstCompSize;
	MTLRegion region = MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h);

	bool identical = (srcComps == dstComps);
	for (int c = 0; c < dstComps && identical; c++) {
		identical = (idx[c] == c);
	}
	if (identical && srcRowBytes % dstBpp == 0) {
		[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:src bytesPerRow:srcRowBytes bytesPerImage:0];
		return;
	}

	// Channel swizzle / expansion (BGRA, RGB -> RGBA, RED -> RGBA, ...) into a tightly packed temporary.
	std::vector<uint8_t> tmp((size_t)w * h * dstBpp);
	const uint8_t one8 = 255;
	const float one32 = 1.0f;
	for (GLsizei row = 0; row < h; row++) {
		const uint8_t *s = src + (size_t)row * srcRowBytes;
		uint8_t *d = tmp.data() + (size_t)row * w * dstBpp;
		for (GLsizei col = 0; col < w; col++, s += srcBpp, d += dstBpp) {
			for (int c = 0; c < dstComps; c++) {
				uint8_t *out = d + (size_t)c * dstCompSize;
				if (idx[c] >= 0) {
					memcpy(out, s + (size_t)idx[c] * srcCompSize, (size_t)dstCompSize);
				} else if (c == 3) {
					memcpy(out, dstCompSize == 4 ? (const void *)&one32 : (const void *)&one8, (size_t)dstCompSize); // missing alpha = 1
				} else {
					memset(out, 0, (size_t)dstCompSize); // missing colour channel = 0
				}
			}
		}
	}
	[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:tmp.data() bytesPerRow:(size_t)w * dstBpp bytesPerImage:0];
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
	GL4METAL_LOG("glGenerateMipmap: Unimplemented");
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
				GL4METAL_LOG("glPixelStorei: invalid alignment %d", param);
				return;
			}
			(pname == GL_UNPACK_ALIGNMENT ? ctx.unpack_alignment : ctx.pack_alignment) = param;
			break;
		case GL_UNPACK_ROW_LENGTH:   ctx.unpack_row_length = param; break;
		case GL_UNPACK_SKIP_PIXELS:  ctx.unpack_skip_pixels = param; break;
		case GL_UNPACK_SKIP_ROWS:    ctx.unpack_skip_rows = param; break;
		case GL_PACK_ROW_LENGTH:     ctx.pack_row_length = param; break;
		case GL_PACK_SKIP_PIXELS:    ctx.pack_skip_pixels = param; break;
		case GL_PACK_SKIP_ROWS:      ctx.pack_skip_rows = param; break;
		default:
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
	bool valid = level >= 0 && border == 0 && mtlFormat != MTLPixelFormatInvalid &&
				 width > 0 && height > 0 && width <= GL4METAL_MAX_TEXTURE_SIZE && height <= GL4METAL_MAX_TEXTURE_SIZE;

	if (target == GL_PROXY_TEXTURE_2D) {
		// Only records whether a texture of this size/format could be created (Minecraft probes the max size this way).
		ctx.proxy_texture_2d.w = valid ? width : 0;
		ctx.proxy_texture_2d.h = valid ? height : 0;
		ctx.proxy_texture_2d.internalformat = valid ? (GLenum)internalformat : 0;
		return;
	}
	int face = cubeFace(target); // -1: a plain 2D texture
	if (target != GL_TEXTURE_2D && face < 0) {
		GL4METAL_LOG("glTexImage2D: unsupported target 0x%X", target);
		return;
	}
	TextureObject *t = boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D);
	if (!t) {
		GL4METAL_LOG("glTexImage2D: no texture bound to target 0x%X", face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D);
		return;
	}
	if (face >= 0 && width != height) {
		GL4METAL_LOG("glTexImage2D: cube map faces must be square (%dx%d)", width, height);
		return;
	}
	if (!valid) {
		GL4METAL_LOG("glTexImage2D: invalid level %d, size %dx%d or internalformat 0x%X", level, width, height, internalformat);
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
			endRenderEncoder(); // a pass that renders to the old texture must not outlive it
			t->texture = [ctx.device newTextureWithDescriptor:d];
			if (!t->texture) {
				GL4METAL_LOG("glTexImage2D: failed to allocate a %dx%d MTLPixelFormat %d texture", width, height, (int)mtlFormat);
				return;
			}
			t->width = width;
			t->height = height;
		}
		t->internalformat = (GLenum)internalformat;
	} else if (!t->texture || (NSUInteger)level >= t->texture.mipmapLevelCount ||
			   width != std::max(1, t->width >> level) || height != std::max(1, t->height >> level)) {
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
	GL4METAL_LOG("glCopyTexImage2D: Unimplemented");
}

void glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width) {
	GL4METAL_LOG("glCopyTexSubImage1D: Unimplemented");
}

void glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height) {
	GL4METAL_LOG("glCopyTexSubImage2D: Unimplemented");
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
	GL4METAL_LOG("glTexParameterfv: Unimplemented");
}

void glTexParameteri(GLenum target, GLenum pname, GLint param) {
	TextureObject *t = boundTexture(target);
	if (!t) {
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
			GL4METAL_LOG("glTexParameteri: unhandled pname 0x%X", pname);
			break;
	}
}

void glTexParameteriv(GLenum target, GLenum pname, const GLint *params) {
	GL4METAL_LOG("glTexParameteriv: Unimplemented");
}

void glTexParameterIiv(GLenum target, GLenum pname, const GLint *params) {
	GL4METAL_LOG("glTexParameterIiv: Unimplemented");
}

void glTexParameterIuiv(GLenum target, GLenum pname, const GLuint *params) {
	GL4METAL_LOG("glTexParameterIuiv: Unimplemented");
}

// MARK: - Texture queries

void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void *pixels) {
	GL4METAL_LOG("glGetTexImage: Unimplemented");
}

void glGetCompressedTexImage(GLenum target, GLint level, void *img) {
	GL4METAL_LOG("glGetCompressedTexImage: Unimplemented");
}

void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
	GL4METAL_LOG("glGetTexParameterfv: Unimplemented");
}

void glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetTexParameteriv: Unimplemented");
}

void glGetTexParameterIiv(GLenum target, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetTexParameterIiv: Unimplemented");
}

void glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint *params) {
	GL4METAL_LOG("glGetTexParameterIuiv: Unimplemented");
}

void glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat *params) {
	GL4METAL_LOG("glGetTexLevelParameterfv: Unimplemented");
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
			GL4METAL_LOG("glGetTexLevelParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}
