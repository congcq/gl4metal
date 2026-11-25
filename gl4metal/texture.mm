#import "texture.h"

#include <cmath>
#include <limits>
#include <new>


// MARK: - Helpers

// CPU writes these staging allocations sequentially and never reads them back.
// WriteCombined can reduce CPU cache pollution for this write-only upload path.
static constexpr MTLResourceOptions kTextureUploadBufferOptions =
	(MTLResourceOptions)(MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined);

// iOS notes: long-lived GPU-facing textures default to MTLStorageModePrivate.
// Textures that GL may attach to an FBO retain RenderTarget usage from allocation;
// ShaderWrite is added only when GL binds an image for writing. CPU uploads use a
// shared, write-combined staging arena and blit into Private textures. MTLStorageModeManaged
// does not exist on iOS. There are no 24-bit depth or RGB formats, so GL_DEPTH24_* maps
// to Depth32Float* and GL_RGB8 to RGBA8. RGB5_A1 uses the native packed format when supported.
static MTLPixelFormat rgb5A1MetalPixelFormat(void) {
	if (!ctx.device) return MTLPixelFormatRGBA8Unorm;
	static const void *cachedDeviceKey = nullptr;
	static MTLPixelFormat cachedFormat = MTLPixelFormatInvalid;
	const void *deviceKey = (__bridge const void *)ctx.device;
	if (cachedDeviceKey != deviceKey || cachedFormat == MTLPixelFormatInvalid) {
		MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatA1BGR5Unorm
			width:1 height:1 mipmapped:NO];
		descriptor.storageMode = MTLStorageModeShared;
		descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		id<MTLTexture> probeTexture = [ctx.device newTextureWithDescriptor:descriptor];
		cachedFormat = probeTexture ? MTLPixelFormatA1BGR5Unorm : MTLPixelFormatRGBA8Unorm;
		cachedDeviceKey = deviceKey;
	}
	return cachedFormat;
}

MTLPixelFormat metalPixelFormat(GLenum internalformat) {
	switch (internalformat) {
		case GL_R8:
		case GL_RED:
			return MTLPixelFormatR8Unorm;
		case GL_R8_SNORM:
			return MTLPixelFormatR8Snorm;
		case GL_RG8:
		case GL_RG:
			return MTLPixelFormatRG8Unorm;
		case GL_RG8_SNORM:
			return MTLPixelFormatRG8Snorm;
		case GL_RGB5_A1:
			return rgb5A1MetalPixelFormat();
		case GL_R3_G3_B2:
		case GL_RGB8:
		case GL_RGB:
		case GL_RGBA8:
		case GL_RGBA:
			return MTLPixelFormatRGBA8Unorm;
		case GL_RGB8_SNORM:
		case GL_RGBA8_SNORM:
			// Metal has no RGB8 texture format; RGB is expanded to RGBA by uploadPixels().
			return MTLPixelFormatRGBA8Snorm;
		case GL_SRGB8:
		case GL_SRGB8_ALPHA8:
			return MTLPixelFormatRGBA8Unorm_sRGB;
		case GL_R16:
			return MTLPixelFormatR16Unorm;
		case GL_R16_SNORM:
			return MTLPixelFormatR16Snorm;
		case GL_RG16:
			return MTLPixelFormatRG16Unorm;
		case GL_RG16_SNORM:
			return MTLPixelFormatRG16Snorm;
		case GL_RGB16:
		case GL_RGBA16:
			// Metal has no three-channel 16-bit UNorm texture; keep RGB data in RGBA16.
			return MTLPixelFormatRGBA16Unorm;
		case GL_RGB16_SNORM:
		case GL_RGBA16_SNORM:
			// Metal has no three-channel 16-bit SNORM texture; keep RGB data in RGBA16Snorm.
			return MTLPixelFormatRGBA16Snorm;
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
		case GL_STENCIL_INDEX8:
			return MTLPixelFormatStencil8;
		default:
			return MTLPixelFormatInvalid;
	}
}

static bool isDepthStencilFormat(MTLPixelFormat f) {
	return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float ||
		f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatStencil8;
}

static bool isCompressedMetalPixelFormat(MTLPixelFormat format) {
	switch (format) {
		case MTLPixelFormatEAC_R11Unorm: case MTLPixelFormatEAC_R11Snorm:
		case MTLPixelFormatEAC_RG11Unorm: case MTLPixelFormatEAC_RG11Snorm:
		case MTLPixelFormatEAC_RGBA8: case MTLPixelFormatEAC_RGBA8_sRGB:
		case MTLPixelFormatETC2_RGB8: case MTLPixelFormatETC2_RGB8_sRGB:
		case MTLPixelFormatETC2_RGB8A1: case MTLPixelFormatETC2_RGB8A1_sRGB:
		case MTLPixelFormatASTC_4x4_LDR: case MTLPixelFormatASTC_5x4_LDR: case MTLPixelFormatASTC_5x5_LDR:
		case MTLPixelFormatASTC_6x5_LDR: case MTLPixelFormatASTC_6x6_LDR:
		case MTLPixelFormatASTC_8x5_LDR: case MTLPixelFormatASTC_8x6_LDR: case MTLPixelFormatASTC_8x8_LDR:
		case MTLPixelFormatASTC_10x5_LDR: case MTLPixelFormatASTC_10x6_LDR: case MTLPixelFormatASTC_10x8_LDR:
		case MTLPixelFormatASTC_10x10_LDR: case MTLPixelFormatASTC_12x10_LDR: case MTLPixelFormatASTC_12x12_LDR:
		case MTLPixelFormatASTC_4x4_sRGB: case MTLPixelFormatASTC_5x4_sRGB: case MTLPixelFormatASTC_5x5_sRGB:
		case MTLPixelFormatASTC_6x5_sRGB: case MTLPixelFormatASTC_6x6_sRGB:
		case MTLPixelFormatASTC_8x5_sRGB: case MTLPixelFormatASTC_8x6_sRGB: case MTLPixelFormatASTC_8x8_sRGB:
		case MTLPixelFormatASTC_10x5_sRGB: case MTLPixelFormatASTC_10x6_sRGB: case MTLPixelFormatASTC_10x8_sRGB:
		case MTLPixelFormatASTC_10x10_sRGB: case MTLPixelFormatASTC_12x10_sRGB: case MTLPixelFormatASTC_12x12_sRGB:
			return true;
		default: return false;
	}
}

static bool isSizedTextureInternalFormat(GLenum format) {
	switch (format) {
		case GL_R8: case GL_R8_SNORM: case GL_RG8: case GL_RG8_SNORM:
		case GL_RGB5_A1: case GL_RGB8: case GL_RGB8_SNORM: case GL_RGBA8: case GL_RGBA8_SNORM:
		case GL_SRGB8: case GL_SRGB8_ALPHA8:
		case GL_R16: case GL_R16_SNORM: case GL_RG16: case GL_RG16_SNORM:
		case GL_RGB16: case GL_RGBA16: case GL_RGB16_SNORM: case GL_RGBA16_SNORM:
		case GL_R16F: case GL_RG16F: case GL_RGB16F: case GL_RGBA16F:
		case GL_R32F: case GL_RG32F: case GL_RGB32F: case GL_RGBA32F:
		case GL_RGB10_A2: case GL_R11F_G11F_B10F: case GL_RGB9_E5:
		case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F: case GL_DEPTH24_STENCIL8: case GL_DEPTH32F_STENCIL8:
		case GL_STENCIL_INDEX8:
			return true;
		default:
			return false;
	}
}

static GLsizei fullMipLevelCount(GLsizei width, GLsizei height = 1, GLsizei depth = 1) {
	GLsizei largest = std::max(width, std::max(height, depth));
	GLsizei levels = 1;
	while (largest > 1) { largest >>= 1; ++levels; }
	return levels;
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

static NSUInteger mipLevelCount3D(const TextureObject &t, GLsizei w, GLsizei h, GLsizei d) {
	NSUInteger full = 1;
	for (GLsizei m = std::max(w, std::max(h, d)); m > 1; m >>= 1) ++full;
	if (t.max_level < 1000)
		return std::min<NSUInteger>(full, (NSUInteger)std::max(t.max_level, 0) + 1);
	return usesMipmaps(t.min_filter) ? full : 1;
}

// GL_TEXTURE_CUBE_MAP_POSITIVE_X ... NEGATIVE_Z -> slice 0...5 of the cube texture, -1 for any other target.
static int cubeFace(GLenum target) {
	return (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) ? (int)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : -1;
}

static TextureObject *boundTexture(GLenum target) {
	if (ctx.active_texture_unit >= GL4METAL_MAX_TEXTURE_UNITS) return nullptr;
	const auto binding = ctx.bound_textures[ctx.active_texture_unit].find(target);
	if (binding == ctx.bound_textures[ctx.active_texture_unit].end() || binding->second == 0) return nullptr;
	auto textureIt = ctx.texture_objects.find(binding->second);
	return textureIt == ctx.texture_objects.end() ? nullptr : &textureIt->second;
}

void gl4metalInvalidateTextureSamplingViews(TextureObject &textureObject) {
	textureObject.sampled_view = nil;
	textureObject.sampled_view_source_identity = nullptr;
	textureObject.feedback_sampled_view = nil;
	textureObject.feedback_sampled_view_source_identity = nullptr;
}

void gl4metalSetTextureStorage(TextureObject &textureObject, id<MTLTexture> texture) {
	gl4metalInvalidateTextureSamplingViews(textureObject);
	textureObject.feedback_snapshot = nil;
	textureObject.feedback_snapshot_pass_serial = UINT64_MAX;
	textureObject.texture = texture;
}

// Promote a GPU texture to an additional Metal usage only when GL actually needs it.
// FBO-capable textures keep RenderTarget from allocation (GL may attach them later);
// omitting blanket ShaderWrite is the important part for Apple GPU lossless compression.
bool gl4metalEnsureTextureUsage(TextureObject &textureObject, MTLTextureUsage requiredUsage, const char *caller) {
	id<MTLTexture> sourceTexture = textureObject.texture;
	if (!sourceTexture) return false;
	if ((sourceTexture.usage & requiredUsage) == requiredUsage) return true;
	if (textureObject.buffer_object != 0) return false;

	const MTLTextureUsage allowedPromotionBits = MTLTextureUsageShaderWrite;
	if (requiredUsage == 0 || (requiredUsage & ~allowedPromotionBits) != 0) return false;

	const MTLTextureType textureType = sourceTexture.textureType;
	if (textureType != MTLTextureType2D && textureType != MTLTextureType2DArray &&
		textureType != MTLTextureType3D && textureType != MTLTextureTypeCube) {
		GL4METAL_LOG("%s: cannot promote unsupported texture type %lu to usage 0x%X",
			caller ? caller : "texture usage promotion", (unsigned long)textureType, (unsigned)requiredUsage);
		return false;
	}

	MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
	descriptor.textureType = textureType;
	descriptor.pixelFormat = sourceTexture.pixelFormat;
	descriptor.width = sourceTexture.width;
	descriptor.height = sourceTexture.height;
	descriptor.depth = sourceTexture.depth;
	descriptor.mipmapLevelCount = sourceTexture.mipmapLevelCount;
	descriptor.sampleCount = sourceTexture.sampleCount;
	descriptor.arrayLength = sourceTexture.arrayLength;
	descriptor.storageMode = sourceTexture.storageMode;
	descriptor.cpuCacheMode = sourceTexture.cpuCacheMode;
	descriptor.usage = (MTLTextureUsage)(sourceTexture.usage | requiredUsage);

	id<MTLTexture> replacementTexture = [ctx.device newTextureWithDescriptor:descriptor];
	if (!replacementTexture) {
		GL4METAL_LOG("%s: Metal could not promote texture format=%d type=%lu storage=%lu usage=0x%X",
			caller ? caller : "texture usage promotion", (int)sourceTexture.pixelFormat,
			(unsigned long)textureType, (unsigned long)sourceTexture.storageMode,
			(unsigned)descriptor.usage);
		return false;
	}

	// A usage change requires a new MTLTexture allocation. Drain both backends once
	// at promotion time, then copy every mip/slice before publishing the replacement.
	// This is intentionally a cold path; normal sampling and draw submission never wait here.
	endRenderEncoder();
	flushCommandBuffer(true);
	if (!gl4metalEnsureClassicCommandBuffer()) return false;
	id<MTLBlitCommandEncoder> copyEncoder = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!copyEncoder) return false;

	const NSUInteger sliceCount = textureType == MTLTextureType2DArray ? sourceTexture.arrayLength :
		(textureType == MTLTextureTypeCube ? 6u : 1u);
	for (NSUInteger mipLevel = 0; mipLevel < sourceTexture.mipmapLevelCount; ++mipLevel) {
		const NSUInteger levelWidth = std::max<NSUInteger>(1, sourceTexture.width >> mipLevel);
		const NSUInteger levelHeight = std::max<NSUInteger>(1, sourceTexture.height >> mipLevel);
		const NSUInteger levelDepth = textureType == MTLTextureType3D
			? std::max<NSUInteger>(1, sourceTexture.depth >> mipLevel) : 1u;
		const MTLSize copySize = MTLSizeMake(levelWidth, levelHeight, levelDepth);
		if (textureType == MTLTextureType3D) {
			[copyEncoder copyFromTexture:sourceTexture sourceSlice:0 sourceLevel:mipLevel
				sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:copySize
				toTexture:replacementTexture destinationSlice:0 destinationLevel:mipLevel
				destinationOrigin:MTLOriginMake(0, 0, 0)];
		} else {
			for (NSUInteger slice = 0; slice < sliceCount; ++slice) {
				[copyEncoder copyFromTexture:sourceTexture sourceSlice:slice sourceLevel:mipLevel
					sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:copySize
					toTexture:replacementTexture destinationSlice:slice destinationLevel:mipLevel
					destinationOrigin:MTLOriginMake(0, 0, 0)];
			}
		}
	}
	[copyEncoder endEncoding];
	if (sourceTexture.label.length > 0) replacementTexture.label = sourceTexture.label;
	gl4metalSetTextureStorage(textureObject, replacementTexture);
	return true;
}

static bool textureSwizzleTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_TEXTURE_SWIZZLE");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool mapGLTextureSwizzle(GLenum value, MTLTextureSwizzle &mapped) {
	switch (value) {
		case GL_RED: mapped = MTLTextureSwizzleRed; return true;
		case GL_GREEN: mapped = MTLTextureSwizzleGreen; return true;
		case GL_BLUE: mapped = MTLTextureSwizzleBlue; return true;
		case GL_ALPHA: mapped = MTLTextureSwizzleAlpha; return true;
		case GL_ZERO: mapped = MTLTextureSwizzleZero; return true;
		case GL_ONE: mapped = MTLTextureSwizzleOne; return true;
		default: return false;
	}
}

// Keep this list aligned with the Metal texture-view type conversion table. In particular,
// do not assume 1DArray or multisample texture types support the swizzle-view overload.
static bool textureViewSliceRange(id<MTLTexture> sourceTexture, NSUInteger &sliceCount) {
	if (!sourceTexture) return false;
	switch (sourceTexture.textureType) {
		case MTLTextureType1D:
		case MTLTextureType2D:
		case MTLTextureType3D:
			sliceCount = 1;
			return true;
		case MTLTextureType2DArray:
			sliceCount = sourceTexture.arrayLength;
			return sliceCount != 0;
		case MTLTextureTypeCube:
			sliceCount = 6;
			return true;
		case MTLTextureTypeCubeArray:
			if (sourceTexture.arrayLength == 0 || sourceTexture.arrayLength > std::numeric_limits<NSUInteger>::max() / 6) return false;
			sliceCount = sourceTexture.arrayLength * 6;
			return true;
		default:
			return false;
	}
}

id<MTLTexture> gl4metalTextureForSampling(const TextureObject &textureObject, __unsafe_unretained id<MTLTexture> sourceTexture) {
	if (!sourceTexture) return nil;
	MTLTextureSwizzle mapped[4];
	bool identitySwizzle = true;
	for (size_t component = 0; component < 4; ++component) {
		if (!mapGLTextureSwizzle(textureObject.swizzle[component], mapped[component])) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("texture sampling: invalid stored swizzle enum 0x%X", textureObject.swizzle[component]);
			return nil;
		}
	}
	identitySwizzle = textureObject.swizzle[0] == GL_RED && textureObject.swizzle[1] == GL_GREEN &&
		textureObject.swizzle[2] == GL_BLUE && textureObject.swizzle[3] == GL_ALPHA;

	const bool isPrimaryTexture = sourceTexture == textureObject.texture;
	const bool isFeedbackSnapshot = sourceTexture == textureObject.feedback_snapshot;
	if (identitySwizzle) {
		if (isPrimaryTexture) {
			textureObject.sampled_view = nil;
			textureObject.sampled_view_source_identity = nullptr;
		} else if (isFeedbackSnapshot) {
			textureObject.feedback_sampled_view = nil;
			textureObject.feedback_sampled_view_source_identity = nullptr;
		}
		return sourceTexture;
	}

	const bool cachePrimaryView = isPrimaryTexture;
	const bool cacheFeedbackView = isFeedbackSnapshot;
	void *sourceIdentity = (__bridge void *)sourceTexture;
	if (cachePrimaryView && textureObject.sampled_view &&
		textureObject.sampled_view_source_identity == sourceIdentity) return textureObject.sampled_view;
	if (cacheFeedbackView && textureObject.feedback_sampled_view &&
		textureObject.feedback_sampled_view_source_identity == sourceIdentity) return textureObject.feedback_sampled_view;
	if (cachePrimaryView) {
		textureObject.sampled_view = nil;
		textureObject.sampled_view_source_identity = nullptr;
	} else if (cacheFeedbackView) {
		textureObject.feedback_sampled_view = nil;
		textureObject.feedback_sampled_view_source_identity = nullptr;
	}

	NSUInteger sliceCount = 0;
	if (!textureViewSliceRange(sourceTexture, sliceCount) || sourceTexture.mipmapLevelCount == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("texture sampling: Metal texture type %lu has no supported native swizzle-view mapping",
			(unsigned long)sourceTexture.textureType);
		return nil;
	}

	MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
		mapped[0], mapped[1], mapped[2], mapped[3]);
	id<MTLTexture> view = [sourceTexture newTextureViewWithPixelFormat:sourceTexture.pixelFormat
		textureType:sourceTexture.textureType
		levels:NSMakeRange(0, sourceTexture.mipmapLevelCount)
		slices:NSMakeRange(0, sliceCount)
		swizzle:channels];
	if (!view) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("texture sampling: failed to create swizzle view (type=%lu format=%lu mips=%lu slices=%lu)",
			(unsigned long)sourceTexture.textureType, (unsigned long)sourceTexture.pixelFormat,
			(unsigned long)sourceTexture.mipmapLevelCount, (unsigned long)sliceCount);
		return nil;
	}
	view.label = [NSString stringWithFormat:@"GL4Metal swizzle view %@", sourceTexture.label ?: @"texture"];
	if (textureSwizzleTraceEnabled()) {
		GL4METAL_LOG("[TextureSwizzle] view created type=%lu format=%lu mips=%lu slices=%lu rgba=(0x%X,0x%X,0x%X,0x%X)",
			(unsigned long)sourceTexture.textureType, (unsigned long)sourceTexture.pixelFormat,
			(unsigned long)sourceTexture.mipmapLevelCount, (unsigned long)sliceCount,
			textureObject.swizzle[0], textureObject.swizzle[1], textureObject.swizzle[2], textureObject.swizzle[3]);
	}
	if (cachePrimaryView) {
		textureObject.sampled_view = view;
		textureObject.sampled_view_source_identity = sourceIdentity;
	} else if (cacheFeedbackView) {
		textureObject.feedback_sampled_view = view;
		textureObject.feedback_sampled_view_source_identity = sourceIdentity;
	}
	return view;
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
	const NSUInteger length = glBufferLogicalSize(pbo);
	if (baseOffset > length || byteOffset > length - baseOffset || accessedBytes > length - baseOffset - byteOffset) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: pixel-unpack buffer range offset=%llu span=%llu exceeds %llu-byte buffer %u",
			caller, (unsigned long long)baseOffset, (unsigned long long)(byteOffset + accessedBytes),
			(unsigned long long)length, pbo);
		return nullptr;
	}
	return (const uint8_t *)it->second.contents + baseOffset + byteOffset;
}

// Small/medium texture updates share one staging arena per command buffer on the
// owning GL thread. Each encoded blit gets a distinct range, so the CPU never overwrites
// bytes that the GPU may still be reading. The command buffer retains the arena.
struct TextureUploadArena {
	id<MTLCommandBuffer> commandBuffer = nil;
	id<MTL4CommandBuffer> metal4CommandBuffer = nil;
	id<MTLBuffer> buffer = nil;
	NSUInteger used = 0;
};

static thread_local TextureUploadArena textureUploadArena;

void resetTextureUploadArena(id<MTLCommandBuffer> commandBuffer) {
	if (!commandBuffer) return;
	if (textureUploadArena.commandBuffer == commandBuffer) {
		textureUploadArena.commandBuffer = nil;
		textureUploadArena.buffer = nil;
		textureUploadArena.used = 0;
	}
}

void resetTextureUploadArenaMetal4(id<MTL4CommandBuffer> command_buffer) {
	if (!command_buffer) return;
	if (textureUploadArena.metal4CommandBuffer == command_buffer) {
		textureUploadArena.metal4CommandBuffer = nil;
		textureUploadArena.buffer = nil;
		textureUploadArena.used = 0;
	}
}

// Queue texture uploads through a shared staging buffer and a blit encoder. This keeps
// GL command ordering (earlier render work -> upload -> later render work) without
// blocking the CPU on waitUntilCompleted for every glTexSubImage2D call.
static bool enqueueTextureUpload(id<MTLTexture> texture, GLint level, NSUInteger slice,
		GLint x, GLint y, GLsizei width, GLsizei height,
		const uint8_t *source, size_t sourceRowBytes, size_t copyRowBytes) {
	if (!texture || !source || !ctx.device || !ctx.commandQueue || width <= 0 || height <= 0) return false;
	size_t stagingRowBytes = 0, stagingBytes = 0;
	if (!checkedAlignSize(copyRowBytes, 256, stagingRowBytes) ||
		!checkedSizeMul(stagingRowBytes, (size_t)height, stagingBytes) ||
		stagingBytes == 0 || stagingBytes > ctx.device.maxBufferLength) {
		return false;
	}

	// The upload must follow any render pass that has already sampled or rendered to
	// this texture. Metal 4 records buffer-to-texture copies in its unified compute/blit
	// encoder; later classic rendering is ordered by the shared-event bridge.
	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalBeginMetal4Compute()) {
		id<MTL4CommandBuffer> targetMetal4CommandBuffer = ctx.currentMetal4CommandBuffer;
		if (textureUploadArena.metal4CommandBuffer != targetMetal4CommandBuffer) {
			textureUploadArena.commandBuffer = nil;
			textureUploadArena.metal4CommandBuffer = targetMetal4CommandBuffer;
			textureUploadArena.buffer = nil;
			textureUploadArena.used = 0;
		}
		if ((NSUInteger)stagingBytes <= std::numeric_limits<NSUInteger>::max() - textureUploadArena.used) {
			NSUInteger requiredEnd = textureUploadArena.used + (NSUInteger)stagingBytes;
			if (!textureUploadArena.buffer || requiredEnd > textureUploadArena.buffer.length) {
				NSUInteger capacity = std::max<NSUInteger>(256 * 1024, requiredEnd);
				if (textureUploadArena.buffer && textureUploadArena.buffer.length <= ctx.device.maxBufferLength / 2)
					capacity = std::max(capacity, textureUploadArena.buffer.length * 2);
				capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
				if (capacity >= requiredEnd) {
					id<MTLBuffer> replacement = [ctx.device newBufferWithLength:capacity options:kTextureUploadBufferOptions];
					if (replacement) {
						textureUploadArena.buffer = replacement;
						textureUploadArena.used = 0;
						requiredEnd = (NSUInteger)stagingBytes;
					}
				}
			}
			if (textureUploadArena.buffer && requiredEnd <= textureUploadArena.buffer.length) {
				const NSUInteger stagingOffset = textureUploadArena.used;
				uint8_t *destination = (uint8_t *)textureUploadArena.buffer.contents + stagingOffset;
				for (GLsizei row = 0; row < height; ++row) {
					memcpy(destination + (size_t)row * stagingRowBytes,
						source + (size_t)row * sourceRowBytes, copyRowBytes);
				}
				if (gl4metalMetal4CopyBufferToTexture(textureUploadArena.buffer, stagingOffset,
						(NSUInteger)stagingRowBytes, 0, MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
						texture, slice, (NSUInteger)level, MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0))) {
					textureUploadArena.used = requiredEnd;
					return true;
				}
			}
		}
	}

	if (!gl4metalEnsureClassicCommandBuffer()) return false;
	id<MTLCommandBuffer> targetCommandBuffer = ctx.currentCommandBuffer;

	id<MTLBuffer> staging = nil;
	NSUInteger stagingOffset = 0;
	{
		if (textureUploadArena.commandBuffer != targetCommandBuffer) {
			textureUploadArena.commandBuffer = targetCommandBuffer;
			textureUploadArena.metal4CommandBuffer = nil;
			textureUploadArena.buffer = nil;
			textureUploadArena.used = 0;
		}
		NSUInteger requiredEnd = 0;
		if ((NSUInteger)stagingBytes <= std::numeric_limits<NSUInteger>::max() - textureUploadArena.used) {
			requiredEnd = textureUploadArena.used + (NSUInteger)stagingBytes;
		} else {
			return false;
		}
		if (!textureUploadArena.buffer || requiredEnd > textureUploadArena.buffer.length) {
			NSUInteger capacity = std::max<NSUInteger>(256 * 1024, (NSUInteger)stagingBytes);
			if (textureUploadArena.buffer && textureUploadArena.buffer.length <= ctx.device.maxBufferLength / 2) {
				capacity = std::max(capacity, textureUploadArena.buffer.length * 2);
			}
			capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
			id<MTLBuffer> newArena = [ctx.device newBufferWithLength:capacity options:kTextureUploadBufferOptions];
			if (!newArena) return false;
			// Previous arena allocations remain valid: the command buffer retains the old
			// MTLBuffer even after this context points at the larger replacement.
			textureUploadArena.buffer = newArena;
			textureUploadArena.used = 0;
			requiredEnd = (NSUInteger)stagingBytes;
		}
		staging = textureUploadArena.buffer;
		stagingOffset = textureUploadArena.used;
		uint8_t *destination = (uint8_t *)staging.contents + stagingOffset;
		for (GLsizei row = 0; row < height; ++row) {
			memcpy(destination + (size_t)row * stagingRowBytes,
				source + (size_t)row * sourceRowBytes, copyRowBytes);
		}
		textureUploadArena.used = requiredEnd;
	}

	id<MTLBlitCommandEncoder> blit = [targetCommandBuffer blitCommandEncoder];
	if (!blit) return false;
	// For 2D and cube-slice copies, sourceBytesPerImage must be zero (there is one image).
	[blit copyFromBuffer:staging sourceOffset:stagingOffset sourceBytesPerRow:(NSUInteger)stagingRowBytes
		sourceBytesPerImage:0 sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
		toTexture:texture destinationSlice:slice destinationLevel:(NSUInteger)level
		destinationOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)];
	[blit endEncoding];
	return true;
}

// 3D variant of the staging upload path. sourceBytesPerImage and destinationOrigin.z
// are essential here: treating a volume as a stack of unrelated 2D slices breaks
// row/image pixel-store semantics and can corrupt volume textures.
static bool enqueueTextureUpload3D(id<MTLTexture> texture, GLint level,
		GLint x, GLint y, GLint z, GLsizei width, GLsizei height, GLsizei depth,
		const uint8_t *source, size_t sourceRowBytes, size_t sourceImageBytes, size_t copyRowBytes) {
	const bool isVolume = texture && texture.textureType == MTLTextureType3D;
	const bool isArray = texture && texture.textureType == MTLTextureType2DArray;
	if (!texture || (!isVolume && !isArray) || !source || !ctx.device ||
		!ctx.commandQueue || width <= 0 || height <= 0 || depth <= 0) return false;
	size_t stagingRowBytes = 0, stagingImageBytes = 0, stagingBytes = 0;
	if (!checkedAlignSize(copyRowBytes, 256, stagingRowBytes) ||
		!checkedSizeMul(stagingRowBytes, (size_t)height, stagingImageBytes) ||
		!checkedSizeMul(stagingImageBytes, (size_t)depth, stagingBytes) ||
		stagingBytes == 0 || stagingBytes > ctx.device.maxBufferLength) return false;

	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalBeginMetal4Compute()) {
		id<MTL4CommandBuffer> targetMetal4CommandBuffer = ctx.currentMetal4CommandBuffer;
		if (textureUploadArena.metal4CommandBuffer != targetMetal4CommandBuffer) {
			textureUploadArena.commandBuffer = nil;
			textureUploadArena.metal4CommandBuffer = targetMetal4CommandBuffer;
			textureUploadArena.buffer = nil;
			textureUploadArena.used = 0;
		}
		if ((NSUInteger)stagingBytes <= std::numeric_limits<NSUInteger>::max() - textureUploadArena.used) {
			NSUInteger requiredEnd = textureUploadArena.used + (NSUInteger)stagingBytes;
			if (!textureUploadArena.buffer || requiredEnd > textureUploadArena.buffer.length) {
				NSUInteger capacity = std::max<NSUInteger>(256 * 1024, requiredEnd);
				if (textureUploadArena.buffer && textureUploadArena.buffer.length <= ctx.device.maxBufferLength / 2)
					capacity = std::max(capacity, textureUploadArena.buffer.length * 2);
				capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
				if (capacity >= requiredEnd) {
					id<MTLBuffer> replacement = [ctx.device newBufferWithLength:capacity options:kTextureUploadBufferOptions];
					if (replacement) {
						textureUploadArena.buffer = replacement;
						textureUploadArena.used = 0;
						requiredEnd = (NSUInteger)stagingBytes;
					}
				}
			}
			if (textureUploadArena.buffer && requiredEnd <= textureUploadArena.buffer.length) {
				const NSUInteger stagingOffset = textureUploadArena.used;
				uint8_t *destination = (uint8_t *)textureUploadArena.buffer.contents + stagingOffset;
				for (GLsizei image = 0; image < depth; ++image) {
					const uint8_t *sourceImage = source + (size_t)image * sourceImageBytes;
					uint8_t *destinationImage = destination + (size_t)image * stagingImageBytes;
					for (GLsizei row = 0; row < height; ++row) {
						memcpy(destinationImage + (size_t)row * stagingRowBytes,
							sourceImage + (size_t)row * sourceRowBytes, copyRowBytes);
					}
				}
				bool copied = true;
				if (isArray) {
					for (GLsizei image = 0; image < depth; ++image) {
						copied = gl4metalMetal4CopyBufferToTexture(textureUploadArena.buffer,
							stagingOffset + (NSUInteger)image * (NSUInteger)stagingImageBytes,
							(NSUInteger)stagingRowBytes, 0, MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
							texture, (NSUInteger)(z + image), (NSUInteger)level, MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)) && copied;
					}
				} else {
					copied = gl4metalMetal4CopyBufferToTexture(textureUploadArena.buffer, stagingOffset,
						(NSUInteger)stagingRowBytes, (NSUInteger)stagingImageBytes,
						MTLSizeMake((NSUInteger)width, (NSUInteger)height, (NSUInteger)depth),
						texture, 0, (NSUInteger)level, MTLOriginMake((NSUInteger)x, (NSUInteger)y, (NSUInteger)z));
				}
				if (copied) {
					textureUploadArena.used = requiredEnd;
					return true;
				}
			}
		}
	}

	if (!gl4metalEnsureClassicCommandBuffer()) return false;
	id<MTLCommandBuffer> targetCommandBuffer = ctx.currentCommandBuffer;

	id<MTLBuffer> staging = nil;
	NSUInteger stagingOffset = 0;
	{
		if (textureUploadArena.commandBuffer != targetCommandBuffer) {
			textureUploadArena.commandBuffer = targetCommandBuffer;
			textureUploadArena.metal4CommandBuffer = nil;
			textureUploadArena.buffer = nil;
			textureUploadArena.used = 0;
		}
		if ((NSUInteger)stagingBytes > std::numeric_limits<NSUInteger>::max() - textureUploadArena.used) return false;
		NSUInteger requiredEnd = textureUploadArena.used + (NSUInteger)stagingBytes;
		if (!textureUploadArena.buffer || requiredEnd > textureUploadArena.buffer.length) {
			NSUInteger capacity = std::max<NSUInteger>(256 * 1024, (NSUInteger)stagingBytes);
			if (textureUploadArena.buffer && textureUploadArena.buffer.length <= ctx.device.maxBufferLength / 2)
				capacity = std::max(capacity, textureUploadArena.buffer.length * 2);
			capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
			id<MTLBuffer> newArena = [ctx.device newBufferWithLength:capacity options:kTextureUploadBufferOptions];
			if (!newArena) return false;
			textureUploadArena.buffer = newArena;
			textureUploadArena.used = 0;
			requiredEnd = (NSUInteger)stagingBytes;
		}
		staging = textureUploadArena.buffer;
		stagingOffset = textureUploadArena.used;
		uint8_t *destination = (uint8_t *)staging.contents + stagingOffset;
		for (GLsizei image = 0; image < depth; ++image) {
			const uint8_t *sourceImage = source + (size_t)image * sourceImageBytes;
			uint8_t *destinationImage = destination + (size_t)image * stagingImageBytes;
			for (GLsizei row = 0; row < height; ++row) {
				memcpy(destinationImage + (size_t)row * stagingRowBytes,
					sourceImage + (size_t)row * sourceRowBytes, copyRowBytes);
			}
		}
		textureUploadArena.used = requiredEnd;
	}

	id<MTLBlitCommandEncoder> blit = [targetCommandBuffer blitCommandEncoder];
	if (!blit) return false;
	if (isArray) {
		// A 2D-array texture is a sequence of independent 2D slices, not a 3D
		// volume. Copy each layer explicitly; Metal's 3D destination-origin path
		// would otherwise address the wrong resource type.
		for (GLsizei image = 0; image < depth; ++image) {
			[blit copyFromBuffer:staging sourceOffset:stagingOffset + (NSUInteger)image * (NSUInteger)stagingImageBytes
				sourceBytesPerRow:(NSUInteger)stagingRowBytes sourceBytesPerImage:0
				sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
				toTexture:texture destinationSlice:(NSUInteger)(z + image) destinationLevel:(NSUInteger)level
				destinationOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)];
		}
	} else {
		[blit copyFromBuffer:staging sourceOffset:stagingOffset
			sourceBytesPerRow:(NSUInteger)stagingRowBytes sourceBytesPerImage:(NSUInteger)stagingImageBytes
			sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, (NSUInteger)depth)
			toTexture:texture destinationSlice:0 destinationLevel:(NSUInteger)level
			destinationOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, (NSUInteger)z)];
	}
	[blit endEncoding];
	return true;
}

// Position of the R,G,B,A channels inside one source pixel (-1 = channel not present).
static bool sourceLayout(GLenum format, int &comps, int idx[4]) {
	idx[0] = idx[1] = idx[2] = idx[3] = -1;
	switch (format) {
		case GL_RED:
		case GL_DEPTH_COMPONENT:
		case GL_STENCIL_INDEX: comps = 1; idx[0] = 0; return true;
		case GL_GREEN: comps = 1; idx[1] = 0; return true;
		case GL_BLUE:  comps = 1; idx[2] = 0; return true;
		case GL_RG:   comps = 2; idx[0] = 0; idx[1] = 1; return true;
		case GL_RGB:  comps = 3; idx[0] = 0; idx[1] = 1; idx[2] = 2; return true;
		case GL_BGR:  comps = 3; idx[2] = 0; idx[1] = 1; idx[0] = 2; return true;
		case GL_RGBA: comps = 4; idx[0] = 0; idx[1] = 1; idx[2] = 2; idx[3] = 3; return true;
		case GL_BGRA: comps = 4; idx[2] = 0; idx[1] = 1; idx[0] = 2; idx[3] = 3; return true;
		default: return false;
	}
}

// GL_R3_G3_B2 has no matching Metal pixel format. RGBA8 provides at least the
// requested channel precision, remains renderable, and supplies the implicit alpha=1.
static bool isRGB332PackedType(GLenum type) {
	return type == GL_UNSIGNED_BYTE_3_3_2 || type == GL_UNSIGNED_BYTE_2_3_3_REV;
}

static void decodeRGB332Pixel(const uint8_t *source, GLenum type, uint8_t rgba[4]) {
	const uint8_t packed = *source;
	unsigned red = 0, green = 0, blue = 0;
	if (type == GL_UNSIGNED_BYTE_3_3_2) {
		red = (packed >> 5) & 0x7u;
		green = (packed >> 2) & 0x7u;
		blue = packed & 0x3u;
	} else {
		red = packed & 0x7u;
		green = (packed >> 3) & 0x7u;
		blue = (packed >> 6) & 0x3u;
	}
	rgba[0] = (uint8_t)((red * 255u + 3u) / 7u);
	rgba[1] = (uint8_t)((green * 255u + 3u) / 7u);
	rgba[2] = (uint8_t)((blue * 255u + 1u) / 3u);
	rgba[3] = 0xffu;
}

// RGB5_A1 uploads target A1BGR5Unorm directly when available. Decode the two GL
// 5:5:5:1 packed layouts into source-channel order before applying GL_RGBA/GL_BGRA swizzles;
// scalar inputs are normalized and quantized to five color bits plus one alpha bit.
static bool isRGBA5551PackedType(GLenum type) {
	return type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV;
}

static void decodeRGBA5551Pixel(const uint8_t *source, GLenum type, bool swapBytes, uint8_t channels[4]) {
	uint16_t packed = 0;
	memcpy(&packed, source, sizeof(packed));
	if (swapBytes) packed = (uint16_t)((packed >> 8) | (packed << 8));
	unsigned first = 0, second = 0, third = 0, alpha = 0;
	if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
		first = (packed >> 11) & 0x1fu;
		second = (packed >> 6) & 0x1fu;
		third = (packed >> 1) & 0x1fu;
		alpha = packed & 0x1u;
	} else {
		first = packed & 0x1fu;
		second = (packed >> 5) & 0x1fu;
		third = (packed >> 10) & 0x1fu;
		alpha = (packed >> 15) & 0x1u;
	}
	channels[0] = (uint8_t)((first * 255u + 15u) / 31u);
	channels[1] = (uint8_t)((second * 255u + 15u) / 31u);
	channels[2] = (uint8_t)((third * 255u + 15u) / 31u);
	channels[3] = alpha ? 0xffu : 0u;
}

static float readA1BGR5SourceComponent(const uint8_t *pixel, GLenum type, int component, bool swapBytes) {
	switch (type) {
		case GL_BYTE: {
			int8_t value = 0;
			memcpy(&value, pixel + (size_t)component, sizeof(value));
			return std::max(0.0f, (float)value / 127.0f);
		}
		case GL_UNSIGNED_BYTE:
			return (float)pixel[component] / 255.0f;
		case GL_UNSIGNED_INT_8_8_8_8_REV: {
			// On iOS (little-endian), REV stores R,G,B,A bytes in memory order.
			// GL_UNPACK_SWAP_BYTES reverses the packed 32-bit word as a whole.
			const int byteIndex = swapBytes ? 3 - component : component;
			return (float)pixel[byteIndex] / 255.0f;
		}
		case GL_SHORT: {
			int16_t value = 0;
			memcpy(&value, pixel + (size_t)component * 2, sizeof(value));
			if (swapBytes) value = (int16_t)(((uint16_t)value >> 8) | ((uint16_t)value << 8));
			return std::max(0.0f, (float)value / 32767.0f);
		}
		case GL_UNSIGNED_SHORT: {
			uint16_t value = 0;
			memcpy(&value, pixel + (size_t)component * 2, sizeof(value));
			if (swapBytes) value = (uint16_t)((value >> 8) | (value << 8));
			return (float)value / 65535.0f;
		}
		case GL_INT: {
			int32_t value = 0;
			const uint8_t *bytes = pixel + (size_t)component * 4;
			if (!swapBytes) memcpy(&value, bytes, sizeof(value));
			else { uint8_t reversed[4] = {bytes[3], bytes[2], bytes[1], bytes[0]}; memcpy(&value, reversed, sizeof(value)); }
			return std::max(0.0f, (float)((double)value / 2147483647.0));
		}
		case GL_UNSIGNED_INT: {
			uint32_t value = 0;
			const uint8_t *bytes = pixel + (size_t)component * 4;
			if (!swapBytes) memcpy(&value, bytes, sizeof(value));
			else { uint8_t reversed[4] = {bytes[3], bytes[2], bytes[1], bytes[0]}; memcpy(&value, reversed, sizeof(value)); }
			return (float)((double)value / 4294967295.0);
		}
		case GL_HALF_FLOAT: {
			uint16_t value = 0;
			memcpy(&value, pixel + (size_t)component * 2, sizeof(value));
			if (swapBytes) value = (uint16_t)((value >> 8) | (value << 8));
			const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
			uint32_t exponent = (value >> 10) & 0x1fu;
			uint32_t mantissa = value & 0x3ffu;
			uint32_t bits = 0;
			if (exponent == 0) {
				if (mantissa == 0) bits = sign;
				else {
					int shift = 0;
					while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
					mantissa &= 0x3ffu;
					bits = sign | (uint32_t)(127 - 14 - shift) << 23 | (mantissa << 13);
				}
			} else if (exponent == 31) {
				bits = sign | 0x7f800000u | (mantissa << 13);
			} else {
				bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
			}
			float result = 0.0f;
			memcpy(&result, &bits, sizeof(result));
			return result;
		}
		case GL_FLOAT: {
			float value = 0.0f;
			const uint8_t *componentBytes = pixel + (size_t)component * 4;
			if (!swapBytes) memcpy(&value, componentBytes, sizeof(value));
			else {
				uint8_t ordered[4] = { componentBytes[3], componentBytes[2], componentBytes[1], componentBytes[0] };
				memcpy(&value, ordered, sizeof(value));
			}
			return value;
		}
		default: return 0.0f;
	}
}

static uint16_t packA1BGR5(const float rgba[4]) {
	auto quantize5 = [](float value) -> uint16_t {
		if (!std::isfinite(value)) value = 0.0f;
		value = std::max(0.0f, std::min(1.0f, value));
		return (uint16_t)(value * 31.0f + 0.5f);
	};
	const uint16_t red = quantize5(rgba[0]);
	const uint16_t green = quantize5(rgba[1]);
	const uint16_t blue = quantize5(rgba[2]);
	const uint16_t alpha = std::isfinite(rgba[3]) && rgba[3] >= 0.5f ? 1u : 0u;
	return (uint16_t)((alpha << 15) | (blue << 10) | (green << 5) | red);
}

static bool packA1BGR5Pixel(const uint8_t *source, GLenum type,
	const int channelMap[4], int sourceComponents, bool swapBytes, uint16_t &packedOut) {
	float rgba[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	if (isRGBA5551PackedType(type)) {
		uint8_t decoded[4];
		decodeRGBA5551Pixel(source, type, swapBytes, decoded);
		for (int channel = 0; channel < 4; ++channel) {
			if (channelMap[channel] >= 0) rgba[channel] = (float)decoded[channelMap[channel]] / 255.0f;
		}
	} else {
		const int scalarBytes = (type == GL_UNSIGNED_BYTE || type == GL_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV) ? 1 :
			(type == GL_UNSIGNED_SHORT || type == GL_SHORT || type == GL_HALF_FLOAT) ? 2 :
			(type == GL_INT || type == GL_UNSIGNED_INT || type == GL_FLOAT) ? 4 : 0;
		if (scalarBytes == 0) return false;
		for (int channel = 0; channel < 4; ++channel) {
			if (channelMap[channel] >= 0 && channelMap[channel] < sourceComponents) {
				rgba[channel] = readA1BGR5SourceComponent(source, type, channelMap[channel], swapBytes);
			}
		}
	}
	packedOut = packA1BGR5(rgba);
	return true;
}

// Components and bytes per component of the Metal formats we can write from the CPU.
static bool destLayout(MTLPixelFormat f, int &comps, int &compSize) {
	switch (f) {
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Snorm:         comps = 1; compSize = 1; return true;
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Snorm:        comps = 2; compSize = 1; return true;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:      comps = 4; compSize = 1; return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatR16Float:        comps = 1; compSize = 2; return true;
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float:       comps = 2; compSize = 2; return true;
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Snorm:
		case MTLPixelFormatRGBA16Float:     comps = 4; compSize = 2; return true;
		case MTLPixelFormatR32Float:        comps = 1; compSize = 4; return true;
		case MTLPixelFormatRG32Float:       comps = 2; compSize = 4; return true;
		case MTLPixelFormatRGBA32Float:     comps = 4; compSize = 4; return true;
		default: return false;
	}
}

static void uploadPixels(TextureObject &t, GLint level, GLint x, GLint y, GLsizei w, GLsizei h,
						 GLenum format, GLenum type, const void *pixels, const char *caller,
						 NSUInteger slice = 0, bool freshTextureStorage = false) {
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

	int srcComps = 0, idx[4];
	int dstComps = 0, dstCompSize = 0;
	const bool sourceFormatValid = sourceLayout(format, srcComps, idx);
	const bool packedRGB332 = sourceFormatValid && format == GL_RGB && srcComps == 3 && isRGB332PackedType(type);
	const bool packedRGBA5551 = sourceFormatValid && srcComps == 4 &&
		(format == GL_RGBA || format == GL_BGRA) && isRGBA5551PackedType(type);
	const MTLPixelFormat dstFormat = t.texture.pixelFormat;
	const bool nativeA1BGR5 = dstFormat == MTLPixelFormatA1BGR5Unorm;
	const bool rgb5A1Fallback = t.internalformat == GL_RGB5_A1 && !nativeA1BGR5 && dstFormat == MTLPixelFormatRGBA8Unorm;
	const bool rgb5A1Path = nativeA1BGR5 || rgb5A1Fallback;
	if (nativeA1BGR5) { dstComps = 1; dstCompSize = 2; }
	else destLayout(dstFormat, dstComps, dstCompSize);
	const int srcCompSize = (type == GL_UNSIGNED_BYTE || type == GL_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV) ? 1 :
		(type == GL_UNSIGNED_SHORT || type == GL_SHORT || type == GL_HALF_FLOAT) ? 2 :
		(type == GL_INT || type == GL_UNSIGNED_INT || type == GL_FLOAT) ? 4 : 0;
	const bool dstIs16Unorm = dstFormat == MTLPixelFormatR16Unorm || dstFormat == MTLPixelFormatRG16Unorm || dstFormat == MTLPixelFormatRGBA16Unorm;
	const bool dstIs16Float = dstFormat == MTLPixelFormatR16Float || dstFormat == MTLPixelFormatRG16Float || dstFormat == MTLPixelFormatRGBA16Float;
	const bool dstIs8Snorm = dstFormat == MTLPixelFormatR8Snorm || dstFormat == MTLPixelFormatRG8Snorm || dstFormat == MTLPixelFormatRGBA8Snorm;
	const bool dstIs16Snorm = dstFormat == MTLPixelFormatR16Snorm || dstFormat == MTLPixelFormatRG16Snorm || dstFormat == MTLPixelFormatRGBA16Snorm;
	const bool incompatible16Type = (dstIs16Unorm && type != GL_UNSIGNED_SHORT) || (dstIs16Float && type != GL_HALF_FLOAT);
	const bool incompatibleSignedType = (dstIs8Snorm && type != GL_BYTE) || (dstIs16Snorm && type != GL_SHORT) ||
		(!dstIs8Snorm && !dstIs16Snorm && (type == GL_BYTE || type == GL_SHORT));
	if (!sourceFormatValid || (!packedRGB332 && !packedRGBA5551 && srcCompSize == 0) ||
		(!rgb5A1Path && !destLayout(dstFormat, dstComps, dstCompSize)) ||
		(!rgb5A1Path && !packedRGB332 && !packedRGBA5551 && srcCompSize != dstCompSize) ||
		(packedRGB332 && (dstFormat != MTLPixelFormatRGBA8Unorm || dstComps != 4 || dstCompSize != 1)) ||
		(packedRGBA5551 && !nativeA1BGR5 && (dstFormat != MTLPixelFormatRGBA8Unorm || dstComps != 4 || dstCompSize != 1)) ||
		(!rgb5A1Path && (incompatible16Type || incompatibleSignedType)) ||
		(isRGBA5551PackedType(type) && !packedRGBA5551) ||
		(type == GL_UNSIGNED_INT_8_8_8_8_REV && srcComps != 4)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("%s: unsupported upload, format 0x%X type 0x%X into MTLPixelFormat %d", caller, format, type, (int)t.texture.pixelFormat);
		return;
	}

	// Source row layout, honouring GL_UNPACK_ROW_LENGTH / ALIGNMENT / SKIP_*.
	size_t srcBpp = 0, rowRawBytes = 0, srcRowBytes = 0;
	const size_t rowLength = ctx.unpack_row_length > 0 ? (size_t)ctx.unpack_row_length : (size_t)w;
	const size_t align = ctx.unpack_alignment > 0 ? (size_t)ctx.unpack_alignment : 1;
	srcBpp = packedRGB332 ? 1u : (packedRGBA5551 ? 2u : (size_t)srcComps * (size_t)srcCompSize);
	if (srcBpp == 0 || !checkedSizeMul(rowLength, srcBpp, rowRawBytes) ||
		!checkedAlignSize(rowRawBytes, align, srcRowBytes)) {
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
	const bool is1D = t.texture.textureType == MTLTextureType1D;
	MTLRegion region = is1D ? MTLRegionMake1D((NSUInteger)x, (NSUInteger)w)
		: MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h);

	// Metal stores GL_RGB formats in a four-component texture. Preserve GL's
	// default alpha of +1 even when the upload source happens to be GL_RGBA.
	const bool directNativeA1BGR5 = nativeA1BGR5 && format == GL_RGBA &&
		type == GL_UNSIGNED_SHORT_1_5_5_5_REV && !ctx.unpack_swap_bytes;
	const bool logicalRgb = t.internalformat == GL_R3_G3_B2 || t.internalformat == GL_RGB || t.internalformat == GL_RGB8 ||
		t.internalformat == GL_RGB8_SNORM || t.internalformat == GL_SRGB8 ||
		t.internalformat == GL_RGB16 || t.internalformat == GL_RGB16_SNORM ||
		t.internalformat == GL_RGB16F || t.internalformat == GL_RGB32F;
	bool identical = (srcComps == dstComps) && !logicalRgb && !packedRGB332 && !packedRGBA5551 && !rgb5A1Path;
	for (int c = 0; c < dstComps && identical; c++) identical = (idx[c] == c);
	if (srcRowBytes < copyRowBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: GL_UNPACK_ROW_LENGTH produces a row stride (%lu) smaller than the copied row (%lu)",
			caller, (unsigned long)srcRowBytes, (unsigned long)copyRowBytes);
		return;
	}
	if ((identical || directNativeA1BGR5) && srcRowBytes % dstBpp == 0 && (!ctx.unpack_swap_bytes || srcCompSize == 1)) {
		const bool canReplaceDirectly = freshTextureStorage && t.texture.storageMode == MTLStorageModeShared;
		if (canReplaceDirectly) {
			if (is1D) {
				[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:src bytesPerRow:0];
			} else {
				[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:src bytesPerRow:srcRowBytes bytesPerImage:0];
			}
		} else if (!enqueueTextureUpload(t.texture, level, slice, x, y, w, h, src, srcRowBytes, copyRowBytes)) {
			if (t.texture.storageMode == MTLStorageModeShared) {
				// A bounded synchronous fallback is valid only for CPU-visible Shared textures.
				flushCommandBuffer(true);
				if (is1D) [t.texture replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:src bytesPerRow:0];
				else [t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:src bytesPerRow:srcRowBytes bytesPerImage:0];
			} else {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: could not allocate staging for private texture upload (%dx%d)", caller, w, h);
			}
		}
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
	const uint8_t one8 = type == GL_BYTE ? 0x7f : 0xff;
	const uint16_t one16 = dstIs16Snorm ? 0x7fffu : (dstFormat == MTLPixelFormatRGBA16Float ? 0x3c00u : 0xffffu);
	const float one32 = 1.0f;
	for (GLsizei row = 0; row < h; row++) {
		const uint8_t *s = src + (size_t)row * srcRowBytes;
		uint8_t *d = tmp.data() + (size_t)row * tmpRowBytes;
		for (GLsizei col = 0; col < w; col++, s += srcBpp, d += dstBpp) {
			if (rgb5A1Path) {
				uint16_t packed = 0;
				if (!packA1BGR5Pixel(s, type, idx, srcComps, ctx.unpack_swap_bytes != GL_FALSE, packed)) {
					gl4metalSetError(GL_INVALID_ENUM);
					GL4METAL_LOG("%s: unsupported source type 0x%X for RGB5_A1 upload", caller, type);
					return;
				}
				if (nativeA1BGR5) {
					memcpy(d, &packed, sizeof(packed));
				} else {
					d[0] = (uint8_t)((((packed & 0x1fu) * 255u) + 15u) / 31u);
					d[1] = (uint8_t)(((((packed >> 5) & 0x1fu) * 255u) + 15u) / 31u);
					d[2] = (uint8_t)(((((packed >> 10) & 0x1fu) * 255u) + 15u) / 31u);
					d[3] = (packed & 0x8000u) ? 255u : 0u;
				}
				continue;
			}
			if (packedRGB332) {
				decodeRGB332Pixel(s, type, d);
				continue;
			}
			if (packedRGBA5551) {
				uint8_t sourceChannels[4];
				decodeRGBA5551Pixel(s, type, ctx.unpack_swap_bytes != GL_FALSE, sourceChannels);
				for (int c = 0; c < 4; ++c) d[c] = sourceChannels[idx[c]];
				continue;
			}
			for (int c = 0; c < dstComps; c++) {
				uint8_t *out = d + (size_t)c * dstCompSize;
				if (logicalRgb && c == 3) {
					const void *one = dstCompSize == 1 ? (const void *)&one8 :
						(dstCompSize == 2 ? (const void *)&one16 : (const void *)&one32);
					memcpy(out, one, (size_t)dstCompSize);
				} else if (idx[c] >= 0) {
					const uint8_t *component = s + (size_t)idx[c] * srcCompSize;
					if (ctx.unpack_swap_bytes && srcCompSize > 1) {
						for (int b = 0; b < srcCompSize; ++b) out[b] = component[srcCompSize - 1 - b];
					} else {
						memcpy(out, component, (size_t)dstCompSize);
					}
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
	const bool canReplaceConvertedDirectly = freshTextureStorage && t.texture.storageMode == MTLStorageModeShared;
	if (canReplaceConvertedDirectly) {
		if (is1D) {
			[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:tmp.data() bytesPerRow:0];
		} else {
			[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:tmp.data() bytesPerRow:tmpRowBytes bytesPerImage:0];
		}
	} else if (!enqueueTextureUpload(t.texture, level, slice, x, y, w, h, tmp.data(), tmpRowBytes, tmpRowBytes)) {
		if (t.texture.storageMode == MTLStorageModeShared) {
			flushCommandBuffer(true);
			if (is1D) [t.texture replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:tmp.data() bytesPerRow:0];
			else [t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:slice withBytes:tmp.data() bytesPerRow:tmpRowBytes bytesPerImage:0];
		} else {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("%s: could not allocate staging for converted private texture upload (%dx%d)", caller, w, h);
		}
	}
}

// Volume counterpart to uploadPixels(). It observes all 3D pixel-unpack state and
// uses the same format conversion rules as 2D uploads.
static void uploadPixels3D(TextureObject &t, GLint level, GLint x, GLint y, GLint z,
		GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const void *pixels,
		const char *caller, bool freshTextureStorage = false) {
	const bool isVolume = t.texture && t.texture.textureType == MTLTextureType3D;
	const bool isArray = t.texture && t.texture.textureType == MTLTextureType2DArray;
	if (!t.texture || (!isVolume && !isArray) || level < 0 ||
		(NSUInteger)level >= t.texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: invalid or unallocated 3D/array mip level %d", caller, level);
		return;
	}
	const GLsizei lw = std::max<GLsizei>(1, t.width >> level);
	const GLsizei lh = std::max<GLsizei>(1, t.height >> level);
	// Array layer count is invariant across mip levels; volume depth shrinks.
	const GLsizei ld = isArray ? t.depth : std::max<GLsizei>(1, t.depth >> level);
	if (x < 0 || y < 0 || z < 0 || w < 0 || h < 0 || d < 0 ||
		x > lw || y > lh || z > ld || w > lw - x || h > lh - y || d > ld - z) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: region %d,%d,%d %dx%dx%d does not fit level %d (%dx%dx%d)",
			caller, x, y, z, w, h, d, level, lw, lh, ld);
		return;
	}
	if (w == 0 || h == 0 || d == 0) return;
	int srcComps = 0, idx[4], dstComps = 0, dstCompSize = 0;
	const bool sourceFormatValid = sourceLayout(format, srcComps, idx);
	const bool packedRGB332 = sourceFormatValid && format == GL_RGB && srcComps == 3 && isRGB332PackedType(type);
	const bool packedRGBA5551 = sourceFormatValid && srcComps == 4 &&
		(format == GL_RGBA || format == GL_BGRA) && isRGBA5551PackedType(type);
	const MTLPixelFormat dstFormat = t.texture.pixelFormat;
	const bool nativeA1BGR5 = dstFormat == MTLPixelFormatA1BGR5Unorm;
	const bool rgb5A1Fallback = t.internalformat == GL_RGB5_A1 && !nativeA1BGR5 && dstFormat == MTLPixelFormatRGBA8Unorm;
	const bool rgb5A1Path = nativeA1BGR5 || rgb5A1Fallback;
	if (nativeA1BGR5) { dstComps = 1; dstCompSize = 2; }
	else destLayout(dstFormat, dstComps, dstCompSize);
	const bool directNativeA1BGR5 = nativeA1BGR5 && format == GL_RGBA &&
		type == GL_UNSIGNED_SHORT_1_5_5_5_REV && !ctx.unpack_swap_bytes;
	const int srcCompSize = (type == GL_UNSIGNED_BYTE || type == GL_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV) ? 1 :
		(type == GL_UNSIGNED_SHORT || type == GL_SHORT || type == GL_HALF_FLOAT) ? 2 :
		(type == GL_INT || type == GL_UNSIGNED_INT || type == GL_FLOAT) ? 4 : 0;
	const bool dstIs16Unorm = dstFormat == MTLPixelFormatR16Unorm || dstFormat == MTLPixelFormatRG16Unorm || dstFormat == MTLPixelFormatRGBA16Unorm;
	const bool dstIs16Float = dstFormat == MTLPixelFormatR16Float || dstFormat == MTLPixelFormatRG16Float || dstFormat == MTLPixelFormatRGBA16Float;
	const bool dstIs8Snorm = dstFormat == MTLPixelFormatR8Snorm || dstFormat == MTLPixelFormatRG8Snorm || dstFormat == MTLPixelFormatRGBA8Snorm;
	const bool dstIs16Snorm = dstFormat == MTLPixelFormatR16Snorm || dstFormat == MTLPixelFormatRG16Snorm || dstFormat == MTLPixelFormatRGBA16Snorm;
	const bool incompatible16Type = (dstIs16Unorm && type != GL_UNSIGNED_SHORT) || (dstIs16Float && type != GL_HALF_FLOAT);
	const bool incompatibleSignedType = (dstIs8Snorm && type != GL_BYTE) || (dstIs16Snorm && type != GL_SHORT) ||
		(!dstIs8Snorm && !dstIs16Snorm && (type == GL_BYTE || type == GL_SHORT));
	if (!sourceFormatValid || (!packedRGB332 && !packedRGBA5551 && srcCompSize == 0) ||
		(!rgb5A1Path && !destLayout(dstFormat, dstComps, dstCompSize)) ||
		(!rgb5A1Path && !packedRGB332 && !packedRGBA5551 && srcCompSize != dstCompSize) ||
		(packedRGB332 && (dstFormat != MTLPixelFormatRGBA8Unorm || dstComps != 4 || dstCompSize != 1)) ||
		(packedRGBA5551 && !nativeA1BGR5 && (dstFormat != MTLPixelFormatRGBA8Unorm || dstComps != 4 || dstCompSize != 1)) ||
		(!rgb5A1Path && (incompatible16Type || incompatibleSignedType)) ||
		(isRGBA5551PackedType(type) && !packedRGBA5551) ||
		(type == GL_UNSIGNED_INT_8_8_8_8_REV && srcComps != 4)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("%s: unsupported 3D/array upload, format 0x%X type 0x%X into MTLPixelFormat %d",
			caller, format, type, (int)dstFormat);
		return;
	}

	size_t srcBpp = 0, rowRawBytes = 0, srcRowBytes = 0, srcImageBytes = 0;
	const size_t rowLength = ctx.unpack_row_length > 0 ? (size_t)ctx.unpack_row_length : (size_t)w;
	const size_t imageHeight = ctx.unpack_image_height > 0 ? (size_t)ctx.unpack_image_height : (size_t)h;
	const size_t align = ctx.unpack_alignment > 0 ? (size_t)ctx.unpack_alignment : 1;
	srcBpp = packedRGB332 ? 1u : (packedRGBA5551 ? 2u : (size_t)srcComps * (size_t)srcCompSize);
	if (srcBpp == 0 || !checkedSizeMul(rowLength, srcBpp, rowRawBytes) ||
		!checkedAlignSize(rowRawBytes, align, srcRowBytes) ||
		!checkedSizeMul(srcRowBytes, imageHeight, srcImageBytes) || srcRowBytes < (size_t)w * srcBpp) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: invalid or overflowing 3D/array unpack row/image layout", caller);
		return;
	}
	size_t skipImagesBytes = 0, skipRowsBytes = 0, skipPixelsBytes = 0;
	size_t lastImageBytes = 0, lastRowBytes = 0, copyRowBytes = 0;
	size_t byteOffset = 0, accessedBytes = 0;
	if (!checkedSizeMul((size_t)ctx.unpack_skip_images, srcImageBytes, skipImagesBytes) ||
		!checkedSizeMul((size_t)ctx.unpack_skip_rows, srcRowBytes, skipRowsBytes) ||
		!checkedSizeMul((size_t)ctx.unpack_skip_pixels, srcBpp, skipPixelsBytes) ||
		!checkedSizeMul((size_t)(d - 1), srcImageBytes, lastImageBytes) ||
		!checkedSizeMul((size_t)(h - 1), srcRowBytes, lastRowBytes) ||
		!checkedSizeMul((size_t)w, srcBpp, copyRowBytes) ||
		!checkedSizeAdd(skipImagesBytes, skipRowsBytes, byteOffset) ||
		!checkedSizeAdd(byteOffset, skipPixelsBytes, byteOffset) ||
		!checkedSizeAdd(byteOffset, lastImageBytes, accessedBytes) ||
		!checkedSizeAdd(accessedBytes, lastRowBytes, accessedBytes) ||
		!checkedSizeAdd(accessedBytes, copyRowBytes, accessedBytes)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: 3D/array unpack skip/extent calculation overflows addressable memory", caller);
		return;
	}
	const uint8_t *src = resolveUnpackPointer(pixels, byteOffset, accessedBytes - byteOffset, caller);
	if (!src) return;

	size_t dstBpp = 0, tmpRowBytes = 0, tmpImageBytes = 0, tmpBytes = 0;
	if (!checkedSizeMul((size_t)dstComps, (size_t)dstCompSize, dstBpp) ||
		!checkedSizeMul((size_t)w, dstBpp, tmpRowBytes) ||
		!checkedSizeMul(tmpRowBytes, (size_t)h, tmpImageBytes) ||
		!checkedSizeMul(tmpImageBytes, (size_t)d, tmpBytes) || tmpBytes > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: 3D/array upload staging size is too large", caller);
		return;
	}
	const MTLRegion region = MTLRegionMake3D((NSUInteger)x, (NSUInteger)y, (NSUInteger)z,
		(NSUInteger)w, (NSUInteger)h, (NSUInteger)d);
	const bool logicalRgb = t.internalformat == GL_R3_G3_B2 || t.internalformat == GL_RGB || t.internalformat == GL_RGB8 ||
		t.internalformat == GL_RGB8_SNORM || t.internalformat == GL_SRGB8 ||
		t.internalformat == GL_RGB16 || t.internalformat == GL_RGB16_SNORM ||
		t.internalformat == GL_RGB16F || t.internalformat == GL_RGB32F;
	bool identical = (srcComps == dstComps) && !logicalRgb && !packedRGB332 && !packedRGBA5551 && !rgb5A1Path;
	for (int c = 0; c < dstComps && identical; ++c) identical = (idx[c] == c);
	if (srcRowBytes < copyRowBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: GL_UNPACK_ROW_LENGTH produces a row stride smaller than the copied row", caller);
		return;
	}

	// An image stride shorter than all copied rows is legal input memory overlap, but
	// not a safe bytesPerImage for Metal's replaceRegion API; route it through packing.
	const bool directRows = (identical || directNativeA1BGR5) && srcRowBytes % dstBpp == 0 && (!ctx.unpack_swap_bytes || srcCompSize == 1);
	const bool directImages = srcImageBytes >= srcRowBytes * (size_t)h && srcImageBytes % dstBpp == 0;
	if (directRows && directImages) {
		if (t.texture.storageMode == MTLStorageModePrivate) {
			// Private GPU-local volumes cannot be written by replaceRegion. Upload through
			// the command-buffer staging arena; this remains asynchronous and preserves
			// ordering with draws already recorded on the same GL context.
			if (!enqueueTextureUpload3D(t.texture, level, x, y, z, w, h, d,
				src, srcRowBytes, srcImageBytes, copyRowBytes)) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: could not stage %dx%dx%d private 3D texture upload", caller, w, h, d);
			}
		} else if (freshTextureStorage) {
			// A new shared texture has no prior GPU readers. 3D storage accepts a
			// single volume replace; array storage needs one replace per layer.
			if (isArray) {
				for (GLsizei image = 0; image < d; ++image) {
					[t.texture replaceRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h)
						mipmapLevel:(NSUInteger)level slice:(NSUInteger)(z + image)
						withBytes:src + (size_t)image * srcImageBytes bytesPerRow:srcRowBytes bytesPerImage:0];
				}
			} else {
				[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:0 withBytes:src
					bytesPerRow:srcRowBytes bytesPerImage:(d > 1 ? srcImageBytes : 0)];
			}
		} else if (!enqueueTextureUpload3D(t.texture, level, x, y, z, w, h, d,
			src, srcRowBytes, srcImageBytes, copyRowBytes)) {
			flushCommandBuffer(true);
			if (isArray) {
				for (GLsizei image = 0; image < d; ++image) {
					[t.texture replaceRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h)
						mipmapLevel:(NSUInteger)level slice:(NSUInteger)(z + image)
						withBytes:src + (size_t)image * srcImageBytes bytesPerRow:srcRowBytes bytesPerImage:0];
				}
			} else {
				[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:0 withBytes:src
					bytesPerRow:srcRowBytes bytesPerImage:(d > 1 ? srcImageBytes : 0)];
			}
		}
		return;
	}

	std::vector<uint8_t> tmp;
	try { tmp.resize(tmpBytes); }
	catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: failed to allocate %lu-byte 3D/array CPU upload conversion buffer", caller, (unsigned long)tmpBytes);
		return;
	}
	const uint8_t one8 = type == GL_BYTE ? 0x7f : 0xff;
	const uint16_t one16 = dstIs16Snorm ? 0x7fffu : (dstFormat == MTLPixelFormatRGBA16Float ? 0x3c00u : 0xffffu);
	const float one32 = 1.0f;
	for (GLsizei image = 0; image < d; ++image) {
		for (GLsizei row = 0; row < h; ++row) {
			const uint8_t *sourceRow = src + (size_t)image * srcImageBytes + (size_t)row * srcRowBytes;
			uint8_t *destRow = tmp.data() + (size_t)image * tmpImageBytes + (size_t)row * tmpRowBytes;
			for (GLsizei col = 0; col < w; ++col) {
				const uint8_t *sp = sourceRow + (size_t)col * srcBpp;
				uint8_t *dp = destRow + (size_t)col * dstBpp;
				if (rgb5A1Path) {
					uint16_t packed = 0;
					if (!packA1BGR5Pixel(sp, type, idx, srcComps, ctx.unpack_swap_bytes != GL_FALSE, packed)) {
						gl4metalSetError(GL_INVALID_ENUM);
						GL4METAL_LOG("%s: unsupported source type 0x%X for 3D/array RGB5_A1 upload", caller, type);
						return;
					}
					if (nativeA1BGR5) {
						memcpy(dp, &packed, sizeof(packed));
					} else {
						dp[0] = (uint8_t)((((packed & 0x1fu) * 255u) + 15u) / 31u);
						dp[1] = (uint8_t)(((((packed >> 5) & 0x1fu) * 255u) + 15u) / 31u);
						dp[2] = (uint8_t)(((((packed >> 10) & 0x1fu) * 255u) + 15u) / 31u);
						dp[3] = (packed & 0x8000u) ? 255u : 0u;
					}
					continue;
				}
				if (packedRGB332) {
					decodeRGB332Pixel(sp, type, dp);
					continue;
				}
				if (packedRGBA5551) {
					uint8_t sourceChannels[4];
					decodeRGBA5551Pixel(sp, type, ctx.unpack_swap_bytes != GL_FALSE, sourceChannels);
					for (int c = 0; c < 4; ++c) dp[c] = sourceChannels[idx[c]];
					continue;
				}
				for (int c = 0; c < dstComps; ++c) {
					uint8_t *out = dp + (size_t)c * dstCompSize;
					if (logicalRgb && c == 3) {
						const void *one = dstCompSize == 1 ? (const void *)&one8 :
							(dstCompSize == 2 ? (const void *)&one16 : (const void *)&one32);
						memcpy(out, one, (size_t)dstCompSize);
					} else if (idx[c] >= 0) {
						const uint8_t *component = sp + (size_t)idx[c] * srcCompSize;
						if (ctx.unpack_swap_bytes && srcCompSize > 1) {
							for (int b = 0; b < srcCompSize; ++b) out[b] = component[srcCompSize - 1 - b];
						} else {
							memcpy(out, component, (size_t)dstCompSize);
						}
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
	}
	if (t.texture.storageMode == MTLStorageModePrivate) {
		if (!enqueueTextureUpload3D(t.texture, level, x, y, z, w, h, d,
			tmp.data(), tmpRowBytes, tmpImageBytes, tmpRowBytes)) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("%s: could not stage converted %dx%dx%d private 3D texture upload", caller, w, h, d);
		}
	} else if (freshTextureStorage) {
		if (isArray) {
			for (GLsizei image = 0; image < d; ++image) {
				[t.texture replaceRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h)
					mipmapLevel:(NSUInteger)level slice:(NSUInteger)(z + image)
					withBytes:tmp.data() + (size_t)image * tmpImageBytes bytesPerRow:tmpRowBytes bytesPerImage:0];
			}
		} else {
			[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:0 withBytes:tmp.data()
				bytesPerRow:tmpRowBytes bytesPerImage:(d > 1 ? tmpImageBytes : 0)];
		}
	} else if (!enqueueTextureUpload3D(t.texture, level, x, y, z, w, h, d,
		tmp.data(), tmpRowBytes, tmpImageBytes, tmpRowBytes)) {
		flushCommandBuffer(true);
		if (isArray) {
			for (GLsizei image = 0; image < d; ++image) {
				[t.texture replaceRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h)
					mipmapLevel:(NSUInteger)level slice:(NSUInteger)(z + image)
					withBytes:tmp.data() + (size_t)image * tmpImageBytes bytesPerRow:tmpRowBytes bytesPerImage:0];
			}
		} else {
			[t.texture replaceRegion:region mipmapLevel:(NSUInteger)level slice:0 withBytes:tmp.data()
				bytesPerRow:tmpRowBytes bytesPerImage:(d > 1 ? tmpImageBytes : 0)];
		}
	}
}

// glCopyTexSubImage2D reads the attachment matching the destination base format:
// colour textures read GL_READ_BUFFER; depth textures read the read framebuffer's depth attachment.
// The default drawable is framebuffer-only, so it cannot be used as a copy source here.
static id<MTLTexture> readAttachment(GLint &level, bool depth) {
	level = 0;
	if (ctx.read_fbo == 0) return nil;
	auto it = GL4METAL_FBO_FIND(ctx.read_fbo);
	if (it == ctx.fbo_objects.end()) return nil;
	const FBOAttachment *attachment = nullptr;
	if (depth) {
		attachment = &it->second.depth;
	} else {
		const GLenum readBuffer = it->second.read_buffer;
		if (readBuffer < GL_COLOR_ATTACHMENT0 || readBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) return nil;
		attachment = &it->second.color[readBuffer - GL_COLOR_ATTACHMENT0];
	}
	if (!attachment || (attachment->texture == 0 && attachment->renderbuffer == 0) ||
		(attachment->texture != 0 && attachment->renderbuffer != 0)) return nil;
	id<MTLTexture> mtlTexture = nil;
	if (attachment->renderbuffer != 0) {
		auto renderbuffer = ctx.renderbuffer_objects.find(attachment->renderbuffer);
		if (renderbuffer == ctx.renderbuffer_objects.end() || !renderbuffer->second.texture) return nil;
		mtlTexture = renderbuffer->second.texture;
		if (attachment->level != 0 || mtlTexture.sampleCount != 1) return nil;
		level = 0;
	} else {
		auto texture = ctx.texture_objects.find(attachment->texture);
		if (texture == ctx.texture_objects.end() || !texture->second.texture) return nil;
		mtlTexture = texture->second.texture;
		if (attachment->level < 0 || (NSUInteger)attachment->level >= mtlTexture.mipmapLevelCount ||
			mtlTexture.sampleCount != 1) return nil;
		level = attachment->level;
	}
	if (mtlTexture.textureType != MTLTextureType2D || (mtlTexture.usage & MTLTextureUsageRenderTarget) == 0) return nil;
	return mtlTexture;
}

// MARK: - Texture objects

void glGenTextures(GLsizei n, GLuint *textures) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint tex_id = ++ctx.next_texture_id;
		ctx.texture_objects[tex_id] = TextureObject();
		textures[i] = tex_id;
	}
}

void glDeleteTextures(GLsizei n, const GLuint *textures) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint tex_id = textures[i];
		if (tex_id == 0) {
			continue;
		}
		for (auto &unit : ctx.bound_textures) {
			for (auto &binding : unit) {
				if (binding.second == tex_id) binding.second = 0;
			}
		}
		for (GLuint image_unit = 0; image_unit < GL4METAL_MAX_IMAGE_UNITS; ++image_unit) {
			ImageUnitBinding &binding = ctx.image_unit_bindings[image_unit];
			if (binding.texture == tex_id) {
				binding = ImageUnitBinding();
			}
		}
		// GL detaches a deleted texture from the framebuffers it is attached to.
		bool detached = false;
		auto detach = [&](FBOAttachment &a) {
			if (a.texture == tex_id) {
				a.texture = 0;
				a.layer = -1;
				a.view = nil;
				a.view_source = nil;
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
		ctx.texture_objects.erase(tex_id);
	}
}

GLboolean glIsTexture(GLuint texture) {
	auto it = ctx.texture_objects.find(texture);
	return texture != 0 && it != ctx.texture_objects.end() && it->second.target != 0; // a name only becomes a texture when first bound
}

void glActiveTexture(GLenum texture) {
	if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + (GLenum)GL4METAL_MAX_TEXTURE_UNITS) {
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

// SPIRV-Cross's MSL backend emulates samplerBuffer with a 2D texture and
// spvTexelBufferCoord(index) = uint2(index % 4096, index / 4096). Keep that
// ABI stable: a linear MTLTexture view over the GL buffer is bound as texture2d<T>.
static const NSUInteger kTexelBufferWidth = 4096;

struct BufferTextureFormat {
	MTLPixelFormat pixelFormat;
	NSUInteger bytesPerTexel;
	GLenum samplerType;
};

static bool bufferTextureFormat(GLenum internalformat, BufferTextureFormat &out) {
	switch (internalformat) {
		case GL_R8:         out = { MTLPixelFormatR8Unorm, 1, GL_SAMPLER_BUFFER }; return true;
		case GL_R8_SNORM:   out = { MTLPixelFormatR8Snorm, 1, GL_SAMPLER_BUFFER }; return true;
		case GL_R8I:        out = { MTLPixelFormatR8Sint, 1, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_R8UI:       out = { MTLPixelFormatR8Uint, 1, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_R16:        out = { MTLPixelFormatR16Unorm, 2, GL_SAMPLER_BUFFER }; return true;
		case GL_R16_SNORM:  out = { MTLPixelFormatR16Snorm, 2, GL_SAMPLER_BUFFER }; return true;
		case GL_R16F:       out = { MTLPixelFormatR16Float, 2, GL_SAMPLER_BUFFER }; return true;
		case GL_R16I:       out = { MTLPixelFormatR16Sint, 2, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_R16UI:      out = { MTLPixelFormatR16Uint, 2, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_R32F:       out = { MTLPixelFormatR32Float, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_R32I:       out = { MTLPixelFormatR32Sint, 4, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_R32UI:      out = { MTLPixelFormatR32Uint, 4, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RG8:        out = { MTLPixelFormatRG8Unorm, 2, GL_SAMPLER_BUFFER }; return true;
		case GL_RG8_SNORM:  out = { MTLPixelFormatRG8Snorm, 2, GL_SAMPLER_BUFFER }; return true;
		case GL_RG8I:       out = { MTLPixelFormatRG8Sint, 2, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RG8UI:      out = { MTLPixelFormatRG8Uint, 2, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RG16:       out = { MTLPixelFormatRG16Unorm, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_RG16_SNORM: out = { MTLPixelFormatRG16Snorm, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_RG16F:      out = { MTLPixelFormatRG16Float, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_RG16I:      out = { MTLPixelFormatRG16Sint, 4, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RG16UI:     out = { MTLPixelFormatRG16Uint, 4, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RG32F:      out = { MTLPixelFormatRG32Float, 8, GL_SAMPLER_BUFFER }; return true;
		case GL_RG32I:      out = { MTLPixelFormatRG32Sint, 8, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RG32UI:     out = { MTLPixelFormatRG32Uint, 8, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA8:      out = { MTLPixelFormatRGBA8Unorm, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA8_SNORM:out = { MTLPixelFormatRGBA8Snorm, 4, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA8I:     out = { MTLPixelFormatRGBA8Sint, 4, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA8UI:    out = { MTLPixelFormatRGBA8Uint, 4, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA16:     out = { MTLPixelFormatRGBA16Unorm, 8, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA16_SNORM:out = { MTLPixelFormatRGBA16Snorm, 8, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA16F:    out = { MTLPixelFormatRGBA16Float, 8, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA16I:    out = { MTLPixelFormatRGBA16Sint, 8, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA16UI:   out = { MTLPixelFormatRGBA16Uint, 8, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA32F:    out = { MTLPixelFormatRGBA32Float, 16, GL_SAMPLER_BUFFER }; return true;
		case GL_RGBA32I:    out = { MTLPixelFormatRGBA32Sint, 16, GL_INT_SAMPLER_BUFFER }; return true;
		case GL_RGBA32UI:   out = { MTLPixelFormatRGBA32Uint, 16, GL_UNSIGNED_INT_SAMPLER_BUFFER }; return true;
		default: return false;
	}
}

static NSUInteger maxTextureBufferHeight(id<MTLDevice> device) {
	if (!device || kTexelBufferWidth == 0) return 0;
	NSUInteger maxHeight = 8192;
	if (@available(iOS 13.0, *)) {
		if ([device supportsFamily:MTLGPUFamilyApple3]) maxHeight = GL4METAL_MAX_TEXTURE_SIZE;
	}
	// GL_MAX_TEXTURE_BUFFER_SIZE is a device-wide texel limit. Advertise a limit
	// representable even by the largest format supported below (RGBA32*, 16 B/texel),
	// rather than exposing a limit that only R8 views can allocate.
	const NSUInteger largestRowBytes = kTexelBufferWidth * 16u;
	maxHeight = std::min(maxHeight, (NSUInteger)GL4METAL_MAX_TEXTURE_SIZE);
	maxHeight = std::min(maxHeight, device.maxBufferLength / largestRowBytes);
	return maxHeight;
}

GLint maxTextureBufferTexels(void) {
	const NSUInteger rows = maxTextureBufferHeight(ctx.device);
	const NSUInteger maxGLint = (NSUInteger)std::numeric_limits<GLint>::max();
	if (rows > maxGLint / kTexelBufferWidth) return (GLint)maxGLint;
	return (GLint)(rows * kTexelBufferWidth);
}

static bool bufferTextureLayout(GLenum internalformat, NSUInteger logicalSize,
	BufferTextureFormat &viewFormat, NSUInteger &texelCount, NSUInteger &height,
	NSUInteger &rowBytes, NSUInteger &requiredStorage) {
	if (!bufferTextureFormat(internalformat, viewFormat)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexBuffer: unsupported texture-buffer internalformat 0x%X", internalformat);
		return false;
	}
	if (!ctx.device || kTexelBufferWidth > GL4METAL_MAX_TEXTURE_SIZE) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	const NSUInteger availableTexels = logicalSize / viewFormat.bytesPerTexel;
	const NSUInteger maximumTexels = (NSUInteger)maxTextureBufferTexels();
	texelCount = std::min(availableTexels, maximumTexels);
	if (maximumTexels == 0 && availableTexels != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: device cannot represent a linear texture-buffer row");
		return false;
	}
	height = std::max<NSUInteger>(1, texelCount / kTexelBufferWidth + (texelCount % kTexelBufferWidth ? 1 : 0));
	if (kTexelBufferWidth > NSUIntegerMax / viewFormat.bytesPerTexel) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return false;
	}
	rowBytes = kTexelBufferWidth * viewFormat.bytesPerTexel;
	const NSUInteger alignment = [ctx.device minimumLinearTextureAlignmentForPixelFormat:viewFormat.pixelFormat];
	if (alignment == 0 || (rowBytes % alignment) != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: row pitch %lu does not meet Metal linear-texture alignment %lu for format %d",
			(unsigned long)rowBytes, (unsigned long)alignment, (int)viewFormat.pixelFormat);
		return false;
	}
	if (height > NSUIntegerMax / rowBytes) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return false;
	}
	requiredStorage = rowBytes * height;
	return true;
}

// Grow the Metal storage once to cover every texture-buffer view attached to this GL buffer.
// Multiple views may have different pixel sizes/row pitches; growing during the per-view loop
// would leave earlier views retaining an obsolete MTLBuffer.
static bool ensureBufferTextureStorage(GLuint bufferObject, NSUInteger logicalSize, NSUInteger requiredStorage) {
	auto bufferIt = ctx.buffer_objects.find(bufferObject);
	if (bufferObject == 0 || bufferIt == ctx.buffer_objects.end() || !bufferIt->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: buffer %u has no allocated storage", bufferObject);
		return false;
	}
	id<MTLBuffer> backing = bufferIt->second;
	if (backing.length >= requiredStorage) return true;
	if (requiredStorage > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glTexBuffer: padded backing store %llu exceeds Metal maxBufferLength",
			(unsigned long long)requiredStorage);
		return false;
	}
	id<MTLBuffer> expanded = [ctx.device newBufferWithLength:requiredStorage options:MTLResourceStorageModeShared];
	if (!expanded) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glTexBuffer: failed to allocate %llu-byte padded Metal buffer",
			(unsigned long long)requiredStorage);
		return false;
	}
	memset(expanded.contents, 0, (size_t)requiredStorage);
	const NSUInteger copyBytes = std::min<NSUInteger>(logicalSize, backing.length);
	if (copyBytes != 0) memcpy(expanded.contents, backing.contents, (size_t)copyBytes);
	ctx.buffer_objects[bufferObject] = expanded;
	return true;
}

static id<MTLTexture> makeBufferTextureView(GLuint bufferObject, GLenum internalformat, NSUInteger logicalSize) {
	BufferTextureFormat viewFormat;
	NSUInteger texelCount = 0, height = 0, rowBytes = 0, requiredStorage = 0;
	if (!bufferTextureLayout(internalformat, logicalSize, viewFormat, texelCount, height, rowBytes, requiredStorage)) return nil;
	if (!ensureBufferTextureStorage(bufferObject, logicalSize, requiredStorage)) return nil;
	auto bufferIt = ctx.buffer_objects.find(bufferObject);
	if (bufferIt == ctx.buffer_objects.end() || !bufferIt->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return nil;
	}

	MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
	descriptor.textureType = MTLTextureType2D;
	descriptor.pixelFormat = viewFormat.pixelFormat;
	descriptor.width = kTexelBufferWidth;
	descriptor.height = height;
	descriptor.depth = 1;
	descriptor.mipmapLevelCount = 1;
	descriptor.sampleCount = 1;
	descriptor.arrayLength = 1;
	descriptor.usage = MTLTextureUsageShaderRead;
	descriptor.storageMode = MTLStorageModeShared;
	id<MTLTexture> view = [bufferIt->second newTextureWithDescriptor:descriptor offset:0 bytesPerRow:rowBytes];
	if (!view) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: Metal rejected a %lux%lu linear texture view (format=%d rowBytes=%lu)",
			(unsigned long)descriptor.width, (unsigned long)descriptor.height, (int)viewFormat.pixelFormat, (unsigned long)rowBytes);
		return nil;
	}
	return view;
}

void refreshTextureBufferViews(GLuint bufferObject) {
	if (bufferObject == 0) return;
	const NSUInteger logicalSize = glBufferLogicalSize(bufferObject);
	NSUInteger maximumStorage = 0;
	// First pass: find the largest physical layout required by any view of this GL buffer.
	for (const auto &entry : ctx.texture_objects) {
		const TextureObject &textureObject = entry.second;
		if (textureObject.target != GL_TEXTURE_BUFFER || textureObject.buffer_object != bufferObject) continue;
		BufferTextureFormat viewFormat;
		NSUInteger texelCount = 0, height = 0, rowBytes = 0, requiredStorage = 0;
		if (bufferTextureLayout(textureObject.internalformat, logicalSize, viewFormat, texelCount, height, rowBytes, requiredStorage)) {
			maximumStorage = std::max(maximumStorage, requiredStorage);
		}
	}
	if (maximumStorage != 0 && !ensureBufferTextureStorage(bufferObject, logicalSize, maximumStorage)) {
		for (auto &entry : ctx.texture_objects) {
			TextureObject &textureObject = entry.second;
			if (textureObject.target == GL_TEXTURE_BUFFER && textureObject.buffer_object == bufferObject) gl4metalSetTextureStorage(textureObject, nil);
		}
		return;
	}
	// Second pass: all views now alias the same current MTLBuffer backing store.
	for (auto &entry : ctx.texture_objects) {
		TextureObject &textureObject = entry.second;
		if (textureObject.target != GL_TEXTURE_BUFFER || textureObject.buffer_object != bufferObject) continue;
		BufferTextureFormat viewFormat;
		if (!bufferTextureFormat(textureObject.internalformat, viewFormat)) {
			gl4metalSetTextureStorage(textureObject, nil);
			textureObject.width = 0;
			textureObject.height = 1;
			textureObject.buffer_sampler_type = 0;
			continue;
		}
		textureObject.buffer_sampler_type = viewFormat.samplerType;
		const NSUInteger visibleTexels = std::min<NSUInteger>(
			logicalSize / viewFormat.bytesPerTexel, (NSUInteger)maxTextureBufferTexels());
		textureObject.width = (GLsizei)std::min<NSUInteger>(visibleTexels, (NSUInteger)std::numeric_limits<GLsizei>::max());
		textureObject.height = 1; // Buffer textures are logically one-dimensional.
		gl4metalSetTextureStorage(textureObject, makeBufferTextureView(bufferObject, textureObject.internalformat, logicalSize));
	}
}

void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer) {
	if (target != GL_TEXTURE_BUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *textureObject = boundTexture(GL_TEXTURE_BUFFER);
	if (!textureObject) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: no GL_TEXTURE_BUFFER texture is bound");
		return;
	}
	BufferTextureFormat viewFormat;
	if (!bufferTextureFormat(internalformat, viewFormat)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexBuffer: unsupported internalformat 0x%X", internalformat);
		return;
	}
	if (buffer == 0) {
		gl4metalSetTextureStorage(*textureObject, nil);
		textureObject->buffer_object = 0;
		textureObject->width = textureObject->height = 0;
		textureObject->buffer_sampler_type = viewFormat.samplerType;
		textureObject->internalformat = internalformat;
		return;
	}
	auto bufferIt = ctx.buffer_objects.find(buffer);
	if (bufferIt == ctx.buffer_objects.end()) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexBuffer: buffer %u is not an existing buffer object", buffer);
		return;
	}
	// An existing buffer with no data store is a legal zero-sized buffer texture.
	if (!bufferIt->second) {
		gl4metalSetTextureStorage(*textureObject, nil);
		textureObject->buffer_object = buffer;
		textureObject->buffer_sampler_type = viewFormat.samplerType;
		textureObject->internalformat = internalformat;
		textureObject->width = 0;
		textureObject->height = 1;
		return;
	}
	// Attach first, then rebuild every view of this buffer together. A different
	// format may require a larger padded backing store; creating only this view
	// could otherwise orphan previously attached views onto the old MTLBuffer.
	const TextureObject previous = *textureObject;
	gl4metalSetTextureStorage(*textureObject, nil);
	textureObject->buffer_object = buffer;
	textureObject->buffer_sampler_type = viewFormat.samplerType;
	textureObject->internalformat = internalformat;
	const NSUInteger visibleTexels = std::min<NSUInteger>(
		glBufferLogicalSize(buffer) / viewFormat.bytesPerTexel, (NSUInteger)maxTextureBufferTexels());
	textureObject->width = (GLsizei)std::min<NSUInteger>(visibleTexels, (NSUInteger)std::numeric_limits<GLsizei>::max());
	textureObject->height = 1; // Buffer textures are logically one-dimensional.
	refreshTextureBufferViews(buffer);
	if (!textureObject->texture) {
		*textureObject = previous;
		// Rebuild both sets: refresh may have invalidated older views while attempting
		// to grow storage for the rejected attachment.
		refreshTextureBufferViews(buffer);
		if (previous.buffer_object != 0 && previous.buffer_object != buffer)
			refreshTextureBufferViews(previous.buffer_object);
		return;
	}
}

void glGenerateMipmap(GLenum target) {
	TextureObject *tex = boundTexture(target);
	if (!tex || !tex->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGenerateMipmap: no allocated texture bound to target 0x%X (GL_INVALID_OPERATION)", target);
		return;
	}
	if (isCompressedMetalPixelFormat(tex->texture.pixelFormat)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGenerateMipmap: Metal cannot generate mipmaps for compressed texture storage; provide compressed mip levels explicitly");
		return;
	}
	if (tex->texture.mipmapLevelCount < 2 || isDepthStencilFormat(tex->texture.pixelFormat)) {
		return; // nothing to generate (levels are allocated from GL_TEXTURE_MAX_LEVEL / the min filter)
	}
	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalMetal4GenerateMipmaps(tex->texture)) return;

	if (!ctx.commandQueue) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGenerateMipmap: Metal command queue is unavailable");
		return;
	}
	if (!gl4metalEnsureClassicCommandBuffer()) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGenerateMipmap: failed to create ordered Metal command buffer");
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
		case GL_UNPACK_IMAGE_HEIGHT:
		case GL_UNPACK_SKIP_IMAGES:
		case GL_PACK_ROW_LENGTH:
		case GL_PACK_SKIP_PIXELS:
		case GL_PACK_SKIP_ROWS:
		case GL_PACK_IMAGE_HEIGHT:
		case GL_PACK_SKIP_IMAGES:
			if (param < 0) {
				gl4metalSetError(GL_INVALID_VALUE);
				GL4METAL_LOG("glPixelStorei: negative row length/skip value %d for pname 0x%X", param, pname);
				return;
			}
			switch (pname) {
				case GL_UNPACK_ROW_LENGTH:  ctx.unpack_row_length = param; break;
				case GL_UNPACK_SKIP_PIXELS: ctx.unpack_skip_pixels = param; break;
				case GL_UNPACK_SKIP_ROWS:   ctx.unpack_skip_rows = param; break;
				case GL_UNPACK_IMAGE_HEIGHT: ctx.unpack_image_height = param; break;
				case GL_UNPACK_SKIP_IMAGES: ctx.unpack_skip_images = param; break;
				case GL_PACK_ROW_LENGTH:    ctx.pack_row_length = param; break;
				case GL_PACK_SKIP_PIXELS:  ctx.pack_skip_pixels = param; break;
				case GL_PACK_SKIP_ROWS:    ctx.pack_skip_rows = param; break;
				case GL_PACK_IMAGE_HEIGHT: ctx.pack_image_height = param; break;
				case GL_PACK_SKIP_IMAGES: ctx.pack_skip_images = param; break;
			}
			break;
		case GL_PACK_SWAP_BYTES:
			ctx.pack_swap_bytes = param ? GL_TRUE : GL_FALSE;
			break;
		case GL_UNPACK_SWAP_BYTES:
			ctx.unpack_swap_bytes = param ? GL_TRUE : GL_FALSE;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glPixelStorei: unhandled pname 0x%X", pname);
			break;
	}
}

// MARK: - Texture images


void glTexImage1D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border,
		GLenum format, GLenum type, const void *pixels) {
	if (target != GL_TEXTURE_1D && target != GL_PROXY_TEXTURE_1D) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	const MTLPixelFormat pixelFormat = metalPixelFormat((GLenum)internalformat);
	const bool valid = level >= 0 && border == 0 && width >= 0 &&
		width <= GL4METAL_MAX_TEXTURE_SIZE && pixelFormat != MTLPixelFormatInvalid;
	if (target == GL_PROXY_TEXTURE_1D) {
		ctx.proxy_texture_1d.w = valid ? width : 0;
		ctx.proxy_texture_1d.internalformat = valid ? (GLenum)internalformat : 0;
		return;
	}
	TextureObject *textureObject = boundTexture(GL_TEXTURE_1D);
	if (!textureObject) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage1D: no GL_TEXTURE_1D object is bound");
		return;
	}
	if (textureObject->immutable_storage) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage1D: immutable texture storage cannot be redefined");
		return;
	}
	if (!valid) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage1D: invalid level=%d width=%d border=%d internalformat=0x%X",
			level, width, border, internalformat);
		return;
	}
	if (width == 0) {
		if (level == 0) {
			endRenderEncoder();
			gl4metalSetTextureStorage(*textureObject, nil);
			textureObject->width = 0;
			textureObject->height = 1;
			textureObject->depth = 1;
			textureObject->internalformat = (GLenum)internalformat;
		}
		return;
	}
	bool freshTextureStorage = false;
	if (level == 0) {
		if (!textureObject->texture || textureObject->texture.textureType != MTLTextureType1D ||
			textureObject->width != width || textureObject->texture.pixelFormat != pixelFormat) {
			const NSUInteger levels = mipLevelCount(*textureObject, width, 1);
			MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
			descriptor.textureType = MTLTextureType1D;
			descriptor.pixelFormat = pixelFormat;
			descriptor.width = (NSUInteger)width;
			descriptor.height = 1;
			descriptor.depth = 1;
			descriptor.arrayLength = 1;
			descriptor.mipmapLevelCount = levels;
			descriptor.sampleCount = 1;
			descriptor.usage = MTLTextureUsageShaderRead;
			descriptor.storageMode = isDepthStencilFormat(pixelFormat) ? MTLStorageModePrivate : MTLStorageModeShared;
			id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
			if (!newTexture) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("glTexImage1D: Metal allocation failed for width=%d format=0x%X", width, internalformat);
				return;
			}
			endRenderEncoder();
			gl4metalSetTextureStorage(*textureObject, newTexture);
			textureObject->width = width;
			textureObject->height = 1;
			textureObject->depth = 1;
			freshTextureStorage = true;
		}
		textureObject->internalformat = (GLenum)internalformat;
	} else if (!textureObject->texture || textureObject->texture.textureType != MTLTextureType1D ||
		(NSUInteger)level >= textureObject->texture.mipmapLevelCount ||
		width != std::max(1, textureObject->width >> level)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage1D: mip level %d width=%d does not fit the allocated 1D mip chain", level, width);
		return;
	}
	if (pixels || boundBuffer(GL_PIXEL_UNPACK_BUFFER)) {
		uploadPixels(*textureObject, level, 0, 0, width, 1, format, type, pixels, "glTexImage1D", 0, freshTextureStorage);
	}
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
	if (t->immutable_storage) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage2D: immutable texture storage cannot be redefined");
		return;
	}
	if (face >= 0 && width != height) {
		GL4METAL_LOG("glTexImage2D: cube map faces must be square (%dx%d)", width, height);
		return;
	}
	if (!valid) {
		gl4metalSetError(GL_INVALID_VALUE);
		const GLuint texture_name = ctx.bound_textures[ctx.active_texture_unit][face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D];
		GL4METAL_LOG("glTexImage2D: rejected level=%d size=%dx%d internalformat=0x%X mappedMetalFormat=%d texture=%u priorFormat=0x%X priorSize=%dx%d hasStorage=%d",
			level, width, height, internalformat, (int)mtlFormat, texture_name, t->internalformat,
			t->width, t->height, t->texture ? 1 : 0);
		return;
	}

	if (width == 0 || height == 0) {
		// A zero-sized level has no Metal storage. At level 0, clear any old allocation; for an empty non-base mip,
		// treat the definition as a no-op because Metal's mip chain is fixed when the texture is allocated.
		if (level == 0) {
			endRenderEncoder();
			gl4metalSetTextureStorage(*t, nil);
			t->width = width;
			t->height = height;
			t->internalformat = (GLenum)internalformat;
		}
		return;
	}

	bool freshTextureStorage = false;
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
			// GL textures can be attached to an FBO after allocation, so retain the
			// RenderTarget capability, but do not blanket-enable ShaderWrite: that flag
			// can disable lossless texture compression on Apple GPUs.
			d.usage = MTLTextureUsageShaderRead;
			if (mtlFormat != MTLPixelFormatRGB9E5Float)
				d.usage = d.usage | MTLTextureUsageRenderTarget;
			d.storageMode = MTLStorageModePrivate;
			id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:d];
			if (!newTexture) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("glTexImage2D: failed to allocate a %dx%d MTLPixelFormat %d texture; previous storage preserved", width, height, (int)mtlFormat);
				return;
			}
			endRenderEncoder(); // close a pass targeting the old texture before replacing its GL storage
			// In-flight command buffers retain the old MTLTexture until their work completes.
			gl4metalSetTextureStorage(*t, newTexture);
			t->width = width;
			t->height = height;
			freshTextureStorage = true;
		}
		t->internalformat = (GLenum)internalformat;
	} else if (!t->texture || (NSUInteger)level >= t->texture.mipmapLevelCount ||
			   width != std::max(1, t->width >> level) || height != std::max(1, t->height >> level)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage2D: level %d (%dx%d) does not fit the allocated mip chain", level, width, height);
		return;
	}

	if (pixels || boundBuffer(GL_PIXEL_UNPACK_BUFFER)) {
		uploadPixels(*t, level, 0, 0, width, height, format, type, pixels, "glTexImage2D",
			face >= 0 ? (NSUInteger)face : 0, freshTextureStorage);
	}
}

void glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height,
		GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels) {
	const bool isArray = target == GL_TEXTURE_2D_ARRAY;
	const bool isVolume = target == GL_TEXTURE_3D;
	const bool isArrayProxy = target == GL_PROXY_TEXTURE_2D_ARRAY;
	const bool isVolumeProxy = target == GL_PROXY_TEXTURE_3D;
	if (!isArray && !isVolume && !isArrayProxy && !isVolumeProxy) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexImage3D: unsupported target 0x%X", target);
		return;
	}

	const MTLPixelFormat mtlFormat = metalPixelFormat((GLenum)internalformat);
	// Metal 3D volume textures cannot be render targets or depth/stencil images.
	// 2D-array textures can be render targets; depth-array allocation is allowed,
	// although the CPU colour-conversion uploader below intentionally rejects a
	// non-null initial pixel payload for depth formats.
	const bool depthStencilFormat = isDepthStencilFormat(mtlFormat);
	const GLsizei maxWidth = isArray ? GL4METAL_MAX_TEXTURE_SIZE : GL4METAL_MAX_3D_TEXTURE_SIZE;
	const GLsizei maxHeight = isArray ? GL4METAL_MAX_TEXTURE_SIZE : GL4METAL_MAX_3D_TEXTURE_SIZE;
	const bool valid = level >= 0 && border == 0 && mtlFormat != MTLPixelFormatInvalid &&
		width >= 0 && height >= 0 && depth >= 0 && width <= maxWidth && height <= maxHeight &&
		depth <= (isArray ? GL4METAL_MAX_ARRAY_TEXTURE_LAYERS : GL4METAL_MAX_3D_TEXTURE_SIZE) &&
		(!depthStencilFormat || isArray);
	if (isArrayProxy) {
		ctx.proxy_texture_2d_array.w = valid ? width : 0;
		ctx.proxy_texture_2d_array.h = valid ? height : 0;
		ctx.proxy_texture_2d_array.layers = valid ? depth : 0;
		ctx.proxy_texture_2d_array.internalformat = valid ? (GLenum)internalformat : 0;
		return;
	}
	if (isVolumeProxy) {
		ctx.proxy_texture_3d.w = valid ? width : 0;
		ctx.proxy_texture_3d.h = valid ? height : 0;
		ctx.proxy_texture_3d.d = valid ? depth : 0;
		ctx.proxy_texture_3d.internalformat = valid ? (GLenum)internalformat : 0;
		return;
	}

	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage3D: no texture bound to target 0x%X", target);
		return;
	}
	if (textureObject->immutable_storage) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage3D: immutable texture storage cannot be redefined");
		return;
	}
	if (!valid) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage3D: invalid level %d, size %dx%dx%d or internalformat 0x%X",
			level, width, height, depth, internalformat);
		return;
	}
	if (width == 0 || height == 0 || depth == 0) {
		if (level == 0) {
			endRenderEncoder();
			gl4metalSetTextureStorage(*textureObject, nil);
			textureObject->width = width;
			textureObject->height = height;
			textureObject->depth = depth;
			textureObject->internalformat = (GLenum)internalformat;
		}
		return;
	}

	bool freshTextureStorage = false;
	if (level == 0) {
		const MTLTextureType expectedType = isArray ? MTLTextureType2DArray : MTLTextureType3D;
		if (!textureObject->texture || textureObject->width != width || textureObject->height != height ||
			textureObject->depth != depth || textureObject->texture.pixelFormat != mtlFormat ||
			textureObject->texture.textureType != expectedType) {
			const NSUInteger levels = isArray
				? mipLevelCount(*textureObject, width, height)
				: mipLevelCount3D(*textureObject, width, height, depth);
			MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
			descriptor.textureType = expectedType;
			descriptor.pixelFormat = mtlFormat;
			descriptor.width = (NSUInteger)width;
			descriptor.height = (NSUInteger)height;
			descriptor.depth = isArray ? 1 : (NSUInteger)depth;
			descriptor.arrayLength = isArray ? (NSUInteger)depth : 1;
			descriptor.mipmapLevelCount = levels;
			descriptor.sampleCount = 1;
			// 2D-array layers may be FBO attachments; 3D volumes are sampling/image
			// resources only. Neither receives ShaderWrite unless GL binds it writable.
			descriptor.usage = MTLTextureUsageShaderRead;
			if (isArray && mtlFormat != MTLPixelFormatRGB9E5Float)
				descriptor.usage = descriptor.usage | MTLTextureUsageRenderTarget;
			descriptor.storageMode = MTLStorageModePrivate;
			id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
			if (!newTexture) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("glTexImage3D: failed to allocate %dx%dx%d texture type=%lu format=%d; previous storage preserved",
					width, height, depth, (unsigned long)expectedType, (int)mtlFormat);
				return;
			}
			endRenderEncoder();
			gl4metalSetTextureStorage(*textureObject, newTexture);
			textureObject->width = width;
			textureObject->height = height;
			textureObject->depth = depth;
			freshTextureStorage = true;
		}
		textureObject->internalformat = (GLenum)internalformat;
	} else {
		const GLsizei expectedDepth = isArray ? textureObject->depth : std::max(1, textureObject->depth >> level);
		if (!textureObject->texture || textureObject->texture.textureType != (isArray ? MTLTextureType2DArray : MTLTextureType3D) ||
			(NSUInteger)level >= textureObject->texture.mipmapLevelCount ||
			width != std::max(1, textureObject->width >> level) ||
			height != std::max(1, textureObject->height >> level) || depth != expectedDepth) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glTexImage3D: level %d (%dx%dx%d) does not fit the allocated %s texture",
				level, width, height, depth, isArray ? "array" : "volume");
			return;
		}
	}

	const bool hasPixelInput = pixels != nullptr || boundBuffer(GL_PIXEL_UNPACK_BUFFER) != 0;
	if (hasPixelInput && !depthStencilFormat) {
		uploadPixels3D(*textureObject, level, 0, 0, 0, width, height, depth, format, type, pixels,
			isArray ? "glTexImage3D(GL_TEXTURE_2D_ARRAY)" : "glTexImage3D", freshTextureStorage);
	} else if (hasPixelInput && depthStencilFormat) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage3D: initial depth-array pixel upload is not supported; allocate the image without pixels and clear/render it afterward");
	}
}

namespace {

static bool validateImmutableStorageArguments(GLsizei levels, GLenum internalformat,
		GLsizei width, GLsizei height, GLsizei depth, GLsizei maxWidth,
		GLsizei maxHeight, GLsizei maxDepth, bool allowDepthStencil, bool arrayTexture = false) {
	const MTLPixelFormat pixelFormat = metalPixelFormat(internalformat);
	if (!isSizedTextureInternalFormat(internalformat) || pixelFormat == MTLPixelFormatInvalid) {
		gl4metalSetError(GL_INVALID_ENUM);
		return false;
	}
	if (levels < 1 || width <= 0 || height <= 0 || depth <= 0 ||
		width > maxWidth || height > maxHeight || depth > maxDepth) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	if (!allowDepthStencil && isDepthStencilFormat(pixelFormat)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	if (levels > fullMipLevelCount(width, height, arrayTexture ? 1 : depth)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	return true;
}

static bool allocateImmutableTexture(TextureObject &textureObject, MTLTextureType textureType,
		GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth,
		GLsizei levels, bool cube, bool arrayTexture) {
	const MTLPixelFormat pixelFormat = metalPixelFormat(internalformat);
	MTLTextureDescriptor *descriptor = nil;
	if (cube) {
		descriptor = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:pixelFormat
			size:(NSUInteger)width mipmapped:levels > 1];
	} else {
		descriptor = [[MTLTextureDescriptor alloc] init];
	}
	descriptor.textureType = textureType;
	descriptor.pixelFormat = pixelFormat;
	descriptor.width = (NSUInteger)width;
	descriptor.height = (NSUInteger)height;
	descriptor.depth = textureType == MTLTextureType3D ? (NSUInteger)depth : 1;
	descriptor.arrayLength = arrayTexture ? (NSUInteger)depth : 1;
	descriptor.mipmapLevelCount = (NSUInteger)levels;
	descriptor.sampleCount = 1;
	// GL 2D/cube/array textures may later become framebuffer attachments. Keep
	// RenderTarget capability but omit ShaderWrite unless GL explicitly binds it writable.
	descriptor.usage = MTLTextureUsageShaderRead;
	if ((textureType == MTLTextureType2D || textureType == MTLTextureTypeCube || arrayTexture) &&
		pixelFormat != MTLPixelFormatRGB9E5Float)
		descriptor.usage = descriptor.usage | MTLTextureUsageRenderTarget;
	descriptor.storageMode = textureType == MTLTextureType1D ? MTLStorageModeShared : MTLStorageModePrivate;

	id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
	if (!newTexture) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glTexStorage: Metal allocation failed type=%lu size=%dx%dx%d levels=%d format=0x%X",
			(unsigned long)textureType, width, height, depth, levels, internalformat);
		return false;
	}
	endRenderEncoder();
	gl4metalSetTextureStorage(textureObject, newTexture);
	textureObject.width = width;
	textureObject.height = height;
	textureObject.depth = depth;
	textureObject.internalformat = internalformat;
	textureObject.buffer_object = 0;
	textureObject.immutable_storage = true;
	return true;
}

} // namespace

void glTexStorage1D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width) {
	if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (textureObject->immutable_storage || textureObject->texture || textureObject->internalformat != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (!validateImmutableStorageArguments(levels, internalformat, width, 1, 1,
		GL4METAL_MAX_TEXTURE_SIZE, 1, 1, false)) return;
	if (!ctx.device) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	allocateImmutableTexture(*textureObject, MTLTextureType1D, internalformat,
		width, 1, 1, levels, false, false);
}

void glTexStorage2D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
	const bool cube = target == GL_TEXTURE_CUBE_MAP;
	if (target != GL_TEXTURE_2D && !cube) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (textureObject->immutable_storage || textureObject->texture || textureObject->internalformat != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if ((cube && width != height) || !validateImmutableStorageArguments(levels, internalformat,
		width, height, 1, GL4METAL_MAX_TEXTURE_SIZE, GL4METAL_MAX_TEXTURE_SIZE, 1, true)) {
		if (cube && width != height) gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!ctx.device) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	allocateImmutableTexture(*textureObject, cube ? MTLTextureTypeCube : MTLTextureType2D,
		internalformat, width, height, 1, levels, cube, false);
}

void glTexStorage3D(GLenum target, GLsizei levels, GLenum internalformat,
		GLsizei width, GLsizei height, GLsizei depth) {
	const bool arrayTexture = target == GL_TEXTURE_2D_ARRAY;
	if (target != GL_TEXTURE_3D && !arrayTexture) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (textureObject->immutable_storage || textureObject->texture || textureObject->internalformat != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (!validateImmutableStorageArguments(levels, internalformat, width, height, depth,
		arrayTexture ? GL4METAL_MAX_TEXTURE_SIZE : GL4METAL_MAX_3D_TEXTURE_SIZE,
		arrayTexture ? GL4METAL_MAX_TEXTURE_SIZE : GL4METAL_MAX_3D_TEXTURE_SIZE,
		arrayTexture ? GL4METAL_MAX_ARRAY_TEXTURE_LAYERS : GL4METAL_MAX_3D_TEXTURE_SIZE,
		arrayTexture, arrayTexture)) return;
	if (!ctx.device) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	allocateImmutableTexture(*textureObject, arrayTexture ? MTLTextureType2DArray : MTLTextureType3D,
		internalformat, width, height, depth, levels, false, arrayTexture);
}

void glTexStorage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat,
		GLsizei width, GLsizei height, GLboolean fixedsamplelocations) {
	glTexImage2DMultisample(target, samples, internalformat, width, height, fixedsamplelocations);
}

void glTexStorage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat,
		GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) {
	glTexImage3DMultisample(target, samples, internalformat, width, height, depth, fixedsamplelocations);
}

void glTexImage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) {
	if (target != GL_TEXTURE_2D_MULTISAMPLE) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexImage2DMultisample: target must be GL_TEXTURE_2D_MULTISAMPLE (0x%X)", target);
		return;
	}
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage2DMultisample: no multisample texture is bound");
		return;
	}
	if (textureObject->immutable_storage) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexImage2DMultisample: immutable texture storage cannot be redefined");
		return;
	}
	const MTLPixelFormat pixelFormat = metalPixelFormat(internalformat);
	if (samples < 1 || width <= 0 || height <= 0 ||
		width > GL4METAL_MAX_TEXTURE_SIZE || height > GL4METAL_MAX_TEXTURE_SIZE) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage2DMultisample: invalid samples=%d size=%dx%d", samples, width, height);
		return;
	}
	if (pixelFormat == MTLPixelFormatInvalid) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexImage2DMultisample: unsupported internal format 0x%X", internalformat);
		return;
	}
	if (![ctx.device supportsTextureSampleCount:(NSUInteger)samples]) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glTexImage2DMultisample: sample count %d is not supported by this iOS Metal device", samples);
		return;
	}
	MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
	descriptor.textureType = MTLTextureType2DMultisample;
	descriptor.pixelFormat = pixelFormat;
	descriptor.width = (NSUInteger)width;
	descriptor.height = (NSUInteger)height;
	descriptor.depth = 1;
	descriptor.arrayLength = 1;
	descriptor.mipmapLevelCount = 1;
	descriptor.sampleCount = (NSUInteger)samples;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	if (pixelFormat != MTLPixelFormatStencil8)
		descriptor.usage = descriptor.usage | MTLTextureUsageShaderRead;
	id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
	if (!newTexture) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glTexImage2DMultisample: Metal allocation failed for %dx%d samples=%d format=0x%X",
			width, height, samples, internalformat);
		return;
	}
	// Metal uses fixed sample locations. OpenGL permits the implementation to use
	// fixed locations even when the caller passes GL_FALSE, so both values map safely.
	(void)fixedsamplelocations;
	endRenderEncoder();
	gl4metalSetTextureStorage(*textureObject, newTexture);
	textureObject->width = width;
	textureObject->height = height;
	textureObject->depth = 1;
	textureObject->internalformat = internalformat;
	textureObject->immutable_storage = true;
}

void glTexImage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) {
	(void)samples; (void)internalformat; (void)width; (void)height; (void)depth; (void)fixedsamplelocations;
	if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) gl4metalSetError(GL_INVALID_ENUM);
	else gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glTexImage3DMultisample: 2D multisample-array textures are not exposed by the current iOS shader/FBO backend");
}

void glTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels) {
	if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *textureObject = boundTexture(GL_TEXTURE_1D);
	if (!textureObject || !textureObject->texture || textureObject->texture.textureType != MTLTextureType1D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexSubImage1D: no allocated 1D texture is bound");
		return;
	}
	uploadPixels(*textureObject, level, xoffset, 0, width, 1, format, type, pixels, "glTexSubImage1D");
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

void glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
		GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) {
	const bool isArray = target == GL_TEXTURE_2D_ARRAY;
	const bool isVolume = target == GL_TEXTURE_3D;
	if (!isVolume && !isArray) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glTexSubImage3D: unsupported target 0x%X", target);
		return;
	}
	TextureObject *t = boundTexture(target);
	const MTLTextureType expectedType = isArray ? MTLTextureType2DArray : MTLTextureType3D;
	if (!t || !t->texture || t->texture.textureType != expectedType) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexSubImage3D: no allocated %s texture bound", isArray ? "2D-array" : "3D");
		return;
	}
	uploadPixels3D(*t, level, xoffset, yoffset, zoffset, width, height, depth,
		format, type, pixels, isArray ? "glTexSubImage3D(GL_TEXTURE_2D_ARRAY)" : "glTexSubImage3D");
}

void glCopyTexImage1D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border) {
	if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (width < 0 || border != 0 || level < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
	TextureObject *textureObject = boundTexture(GL_TEXTURE_1D);
	if (textureObject && textureObject->immutable_storage) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	glTexImage1D(target, level, (GLint)internalformat, width, border, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	if (width > 0) glCopyTexSubImage1D(target, level, 0, x, y, width);
}

void glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border) {
	const int face = cubeFace(target);
	TextureObject *textureObject = (target == GL_TEXTURE_2D || face >= 0)
		? boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D) : nullptr;
	if (textureObject && textureObject->immutable_storage) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	glTexImage2D(target, level, (GLint)internalformat, width, height, border, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glCopyTexSubImage2D(target, level, 0, 0, x, y, width, height);
}

void glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width) {
	if (target != GL_TEXTURE_1D) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *destination = boundTexture(GL_TEXTURE_1D);
	if (!destination || !destination->texture || destination->texture.textureType != MTLTextureType1D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage1D: no allocated 1D destination texture is bound");
		return;
	}
	if (level < 0 || (NSUInteger)level >= destination->texture.mipmapLevelCount || width < 0) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	if (width == 0) return;
	GLint sourceLevel = 0;
	const bool copyDepth = isDepthStencilFormat(destination->texture.pixelFormat);
	id<MTLTexture> source = readAttachment(sourceLevel, copyDepth);
	if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage1D: read framebuffer has no compatible single-sample 2D source");
		return;
	}
	const NSUInteger sourceWidth = std::max<NSUInteger>(source.width >> sourceLevel, 1);
	const NSUInteger sourceHeight = std::max<NSUInteger>(source.height >> sourceLevel, 1);
	const GLsizei destinationWidth = std::max(1, destination->width >> level);
	if (x < 0 || y < 0 || xoffset < 0 || x > (GLint)sourceWidth || y >= (GLint)sourceHeight ||
		width > (GLint)sourceWidth - x || xoffset > destinationWidth || width > destinationWidth - xoffset) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (source.pixelFormat != destination->texture.pixelFormat || copyDepth != isDepthStencilFormat(source.pixelFormat)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage1D: source/destination pixel formats differ (%lu -> %lu)",
			(unsigned long)source.pixelFormat, (unsigned long)destination->texture.pixelFormat);
		return;
	}
	if (source == destination->texture) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(source, 0, (NSUInteger)sourceLevel,
		MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0), MTLSizeMake((NSUInteger)width, 1, 1),
		destination->texture, 0, (NSUInteger)level, MTLOriginMake((NSUInteger)xoffset, 0, 0))) return;

	if (!ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
	id<MTLBlitCommandEncoder> blit = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!blit) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
	[blit copyFromTexture:source sourceSlice:0 sourceLevel:(NSUInteger)sourceLevel
		sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
		sourceSize:MTLSizeMake((NSUInteger)width, 1, 1)
		toTexture:destination->texture destinationSlice:0 destinationLevel:(NSUInteger)level
		destinationOrigin:MTLOriginMake((NSUInteger)xoffset, 0, 0)];
	[blit endEncoding];
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

	if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(src, 0, (NSUInteger)srcLevel,
		MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0), MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
		dst->texture, (face >= 0 ? (NSUInteger)face : 0), (NSUInteger)level,
		MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, 0))) {
		return;
	}

	if (!ctx.commandQueue) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage2D: Metal command queue is unavailable");
		return;
	}
	if (!gl4metalEnsureClassicCommandBuffer()) {
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

void glCopyTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
		GLint x, GLint y, GLsizei width, GLsizei height) {
	const bool isArray = target == GL_TEXTURE_2D_ARRAY;
	const bool isVolume = target == GL_TEXTURE_3D;
	if (!isVolume && !isArray) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glCopyTexSubImage3D: unsupported target 0x%X", target);
		return;
	}
	TextureObject *dst = boundTexture(target);
	const MTLTextureType expectedType = isArray ? MTLTextureType2DArray : MTLTextureType3D;
	if (!dst || !dst->texture || dst->texture.textureType != expectedType) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage3D: destination %s texture is not allocated", isArray ? "2D-array" : "3D");
		return;
	}
	if (level < 0 || (NSUInteger)level >= dst->texture.mipmapLevelCount || width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCopyTexSubImage3D: invalid mip level or copy size");
		return;
	}
	if (width == 0 || height == 0) return;
	const bool depthCopy = isDepthStencilFormat(dst->texture.pixelFormat);
	GLint srcLevel = 0;
	id<MTLTexture> src = readAttachment(srcLevel, depthCopy);
	if (!src || src.textureType != MTLTextureType2D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage3D: read framebuffer has no usable 2D %s attachment", depthCopy ? "depth" : "color");
		return;
	}
	const NSUInteger srcW = std::max<NSUInteger>(src.width >> srcLevel, 1);
	const NSUInteger srcH = std::max<NSUInteger>(src.height >> srcLevel, 1);
	const GLsizei dstW = std::max(1, dst->width >> level);
	const GLsizei dstH = std::max(1, dst->height >> level);
	const GLsizei dstD = isArray ? dst->depth : std::max(1, dst->depth >> level);
	if (x < 0 || y < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
		(NSUInteger)x > srcW || (NSUInteger)y > srcH || (NSUInteger)width > srcW - (NSUInteger)x ||
		(NSUInteger)height > srcH - (NSUInteger)y || xoffset > dstW || yoffset > dstH || zoffset >= dstD ||
		width > dstW - xoffset || height > dstH - yoffset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCopyTexSubImage3D: source rectangle or destination slice is out of range");
		return;
	}
	if (depthCopy != isDepthStencilFormat(src.pixelFormat) || src.pixelFormat != dst->texture.pixelFormat) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage3D: source/destination attachment formats are incompatible");
		return;
	}
	if (src == dst->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glCopyTexSubImage3D: source and destination alias the same Metal texture");
		return;
	}
	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalMetal4CopyTextureRegion(src, 0, (NSUInteger)srcLevel,
		MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0), MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
		dst->texture, isArray ? (NSUInteger)zoffset : 0, (NSUInteger)level,
		MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, isArray ? 0 : (NSUInteger)zoffset))) {
		return;
	}

	if (!ctx.commandQueue) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (!gl4metalEnsureClassicCommandBuffer()) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
	id<MTLBlitCommandEncoder> blit = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!blit) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
	if (isArray) {
		[blit copyFromTexture:src sourceSlice:0 sourceLevel:(NSUInteger)srcLevel
			sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
			sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
			toTexture:dst->texture destinationSlice:(NSUInteger)zoffset destinationLevel:(NSUInteger)level
			destinationOrigin:MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, 0)];
	} else {
		[blit copyFromTexture:src sourceSlice:0 sourceLevel:(NSUInteger)srcLevel
			sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
			sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
			toTexture:dst->texture destinationSlice:0 destinationLevel:(NSUInteger)level
			destinationOrigin:MTLOriginMake((NSUInteger)xoffset, (NSUInteger)yoffset, (NSUInteger)zoffset)];
	}
	[blit endEncoding];
}

// MARK: - Compressed texture images

namespace {
struct CompressedFormatInfo {
	MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
	NSUInteger blockWidth = 4;
	NSUInteger blockHeight = 4;
	NSUInteger bytesPerBlock = 0;
};

static bool compressedFormatInfo(GLenum format, CompressedFormatInfo &info) {
	info = {};
	switch (format) {
		case GL_COMPRESSED_R11_EAC: info.pixelFormat = MTLPixelFormatEAC_R11Unorm; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_SIGNED_R11_EAC: info.pixelFormat = MTLPixelFormatEAC_R11Snorm; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_RG11_EAC: info.pixelFormat = MTLPixelFormatEAC_RG11Unorm; info.bytesPerBlock = 16; return true;
		case GL_COMPRESSED_SIGNED_RG11_EAC: info.pixelFormat = MTLPixelFormatEAC_RG11Snorm; info.bytesPerBlock = 16; return true;
		case GL_COMPRESSED_RGB8_ETC2: info.pixelFormat = MTLPixelFormatETC2_RGB8; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_SRGB8_ETC2: info.pixelFormat = MTLPixelFormatETC2_RGB8_sRGB; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: info.pixelFormat = MTLPixelFormatETC2_RGB8A1; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: info.pixelFormat = MTLPixelFormatETC2_RGB8A1_sRGB; info.bytesPerBlock = 8; return true;
		case GL_COMPRESSED_RGBA8_ETC2_EAC: info.pixelFormat = MTLPixelFormatEAC_RGBA8; info.bytesPerBlock = 16; return true;
		case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: info.pixelFormat = MTLPixelFormatEAC_RGBA8_sRGB; info.bytesPerBlock = 16; return true;
#define GL4METAL_ASTC_FORMAT(glFormat, metalFormat, bw, bh) \
		case glFormat: info.pixelFormat = metalFormat; info.blockWidth = bw; info.blockHeight = bh; info.bytesPerBlock = 16; return true
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_4x4_KHR, MTLPixelFormatASTC_4x4_LDR, 4, 4);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_5x4_KHR, MTLPixelFormatASTC_5x4_LDR, 5, 4);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_5x5_KHR, MTLPixelFormatASTC_5x5_LDR, 5, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_6x5_KHR, MTLPixelFormatASTC_6x5_LDR, 6, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_6x6_KHR, MTLPixelFormatASTC_6x6_LDR, 6, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_8x5_KHR, MTLPixelFormatASTC_8x5_LDR, 8, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_8x6_KHR, MTLPixelFormatASTC_8x6_LDR, 8, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_8x8_KHR, MTLPixelFormatASTC_8x8_LDR, 8, 8);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_10x5_KHR, MTLPixelFormatASTC_10x5_LDR, 10, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_10x6_KHR, MTLPixelFormatASTC_10x6_LDR, 10, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_10x8_KHR, MTLPixelFormatASTC_10x8_LDR, 10, 8);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_10x10_KHR, MTLPixelFormatASTC_10x10_LDR, 10, 10);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_12x10_KHR, MTLPixelFormatASTC_12x10_LDR, 12, 10);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_RGBA_ASTC_12x12_KHR, MTLPixelFormatASTC_12x12_LDR, 12, 12);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR, MTLPixelFormatASTC_4x4_sRGB, 4, 4);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR, MTLPixelFormatASTC_5x4_sRGB, 5, 4);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR, MTLPixelFormatASTC_5x5_sRGB, 5, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR, MTLPixelFormatASTC_6x5_sRGB, 6, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR, MTLPixelFormatASTC_6x6_sRGB, 6, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR, MTLPixelFormatASTC_8x5_sRGB, 8, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR, MTLPixelFormatASTC_8x6_sRGB, 8, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR, MTLPixelFormatASTC_8x8_sRGB, 8, 8);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR, MTLPixelFormatASTC_10x5_sRGB, 10, 5);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR, MTLPixelFormatASTC_10x6_sRGB, 10, 6);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR, MTLPixelFormatASTC_10x8_sRGB, 10, 8);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR, MTLPixelFormatASTC_10x10_sRGB, 10, 10);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR, MTLPixelFormatASTC_12x10_sRGB, 12, 10);
		GL4METAL_ASTC_FORMAT(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR, MTLPixelFormatASTC_12x12_sRGB, 12, 12);
#undef GL4METAL_ASTC_FORMAT
		default: return false;
	}
}

static bool compressedImageLayout(GLsizei width, GLsizei height, const CompressedFormatInfo &info,
		NSUInteger &rowBytes, NSUInteger &rows, size_t &totalBytes) {
	if (width <= 0 || height <= 0 || info.bytesPerBlock == 0 || info.blockWidth == 0 || info.blockHeight == 0)
		return false;
	const NSUInteger blocksWide = ((NSUInteger)width + info.blockWidth - 1) / info.blockWidth;
	rows = ((NSUInteger)height + info.blockHeight - 1) / info.blockHeight;
	if (blocksWide > SIZE_MAX / info.bytesPerBlock) return false;
	rowBytes = blocksWide * info.bytesPerBlock;
	if (rows > SIZE_MAX / rowBytes) return false;
	totalBytes = rowBytes * rows;
	return totalBytes != 0;
}

static bool compressedSubRegionValid(GLint x, GLint y, GLsizei width, GLsizei height,
		GLsizei textureWidth, GLsizei textureHeight, const CompressedFormatInfo &info) {
	if (x < 0 || y < 0 || width < 0 || height < 0 ||
		x > textureWidth || y > textureHeight || width > textureWidth - x || height > textureHeight - y)
		return false;
	if (width == 0 || height == 0) return true;
	const bool rightEdge = x + width == textureWidth;
	const bool bottomEdge = y + height == textureHeight;
	return ((NSUInteger)x % info.blockWidth) == 0 && ((NSUInteger)y % info.blockHeight) == 0 &&
		(((NSUInteger)width % info.blockWidth) == 0 || rightEdge) &&
		(((NSUInteger)height % info.blockHeight) == 0 || bottomEdge);
}

static bool uploadCompressedSlices(TextureObject &textureObject, GLint level, NSUInteger firstSlice,
		NSUInteger sliceCount, GLint x, GLint y, GLsizei width, GLsizei height,
		const CompressedFormatInfo &info, const uint8_t *bytes, NSUInteger rowBytes) {
	id<MTLTexture> destinationTexture = textureObject.texture;
	const NSUInteger availableSlices = destinationTexture && destinationTexture.textureType == MTLTextureTypeCube
		? 6 : (destinationTexture ? destinationTexture.arrayLength : 0);
	if (!destinationTexture || !bytes || !ctx.device || !ctx.commandQueue || sliceCount == 0 ||
		width <= 0 || height <= 0 || rowBytes == 0 || info.blockHeight == 0 ||
		level < 0 || (NSUInteger)level >= destinationTexture.mipmapLevelCount ||
		firstSlice >= availableSlices || sliceCount > availableSlices - firstSlice) return false;

	const NSUInteger blockRows = ((NSUInteger)height + info.blockHeight - 1) / info.blockHeight;
	if (blockRows > SIZE_MAX / rowBytes) return false;
	const size_t bytesPerImage = (size_t)rowBytes * (size_t)blockRows;
	if (bytesPerImage == 0 || bytesPerImage > ctx.device.maxBufferLength ||
		sliceCount > SIZE_MAX / bytesPerImage) return false;

	// Keep compressed uploads asynchronous and ordered with surrounding GL work.
	// Metal 4 routes block-linear copies through the unified compute/blit encoder;
	// sourceBytesPerRow is the byte distance between compressed block rows.
	endRenderEncoder();

	if (ctx.metal4Enabled && gl4metalBeginMetal4Compute()) {
		// Metal buffer/texture copies require an aligned row pitch. Compressed
		// payloads arrive tightly packed by GL, so repack block rows into a
		// 256-byte-aligned staging image before encoding the MTL4 copy.
		if (rowBytes <= std::numeric_limits<NSUInteger>::max() - 255u) {
			const NSUInteger stagingRowBytes = (rowBytes + 255u) & ~(NSUInteger)255u;
			if (blockRows <= SIZE_MAX / stagingRowBytes) {
				const NSUInteger stagingImageBytes = stagingRowBytes * blockRows;
				if (stagingImageBytes > 0 && stagingImageBytes <= ctx.device.maxBufferLength) {
					const NSUInteger maxMetal4Slices = std::max<NSUInteger>(1, ctx.device.maxBufferLength / stagingImageBytes);
					NSUInteger uploadedSlices = 0;
					bool allCopiesEncoded = true;
					while (uploadedSlices < sliceCount && allCopiesEncoded) {
						const NSUInteger batchSlices = std::min<NSUInteger>(sliceCount - uploadedSlices, maxMetal4Slices);
						if (batchSlices > SIZE_MAX / stagingImageBytes) { allCopiesEncoded = false; break; }
						const NSUInteger batchBytes = batchSlices * stagingImageBytes;
						std::vector<uint8_t> paddedBytes;
						try { paddedBytes.resize((size_t)batchBytes); }
						catch (const std::bad_alloc &) { allCopiesEncoded = false; break; }
						for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
							const uint8_t *sourceImage = bytes + (uploadedSlices + sliceIndex) * bytesPerImage;
							uint8_t *destinationImage = paddedBytes.data() + sliceIndex * stagingImageBytes;
							for (NSUInteger blockRow = 0; blockRow < blockRows; ++blockRow) {
								memcpy(destinationImage + blockRow * stagingRowBytes,
									sourceImage + blockRow * rowBytes, rowBytes);
							}
						}
						id<MTLBuffer> stagingBuffer = nil;
						NSUInteger stagingOffset = 0;
						if (!gl4metalMetal4StageBytes(paddedBytes.data(), batchBytes, &stagingBuffer, &stagingOffset)) {
							allCopiesEncoded = false;
							break;
						}
						for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
							if (!gl4metalMetal4CopyBufferToTexture(stagingBuffer,
									stagingOffset + sliceIndex * stagingImageBytes, stagingRowBytes, 0,
									MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1), destinationTexture,
									firstSlice + uploadedSlices + sliceIndex, (NSUInteger)level,
									MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0))) {
								allCopiesEncoded = false;
								break;
							}
						}
						if (allCopiesEncoded) uploadedSlices += batchSlices;
					}
					if (allCopiesEncoded && uploadedSlices == sliceCount) return true;
				}
			}
		}
		// If setup failed after a partial MTL4 batch, the classic helper below
		// commits and waits on the shared-event bridge before replaying all slices.
	}

	if (!gl4metalEnsureClassicCommandBuffer()) return false;
	id<MTLCommandBuffer> targetCommandBuffer = ctx.currentCommandBuffer;

	NSUInteger uploadedSlices = 0;
	const NSUInteger maxSlicesPerBatch = std::max<NSUInteger>(1, ctx.device.maxBufferLength / bytesPerImage);
	while (uploadedSlices < sliceCount) {
		const NSUInteger batchSlices = std::min<NSUInteger>(sliceCount - uploadedSlices, maxSlicesPerBatch);
		if (batchSlices > SIZE_MAX / bytesPerImage) return false;
		const size_t batchBytes = bytesPerImage * batchSlices;

		if (textureUploadArena.commandBuffer != targetCommandBuffer) {
			textureUploadArena.commandBuffer = targetCommandBuffer;
			textureUploadArena.metal4CommandBuffer = nil;
			textureUploadArena.buffer = nil;
			textureUploadArena.used = 0;
		}

		size_t alignedOffset = 0;
		if (!checkedAlignSize((size_t)textureUploadArena.used, (size_t)info.bytesPerBlock, alignedOffset) ||
			alignedOffset > std::numeric_limits<NSUInteger>::max() ||
			batchBytes > std::numeric_limits<NSUInteger>::max() - alignedOffset) return false;
		NSUInteger stagingOffset = (NSUInteger)alignedOffset;
		NSUInteger requiredEnd = stagingOffset + (NSUInteger)batchBytes;

		if (!textureUploadArena.buffer || requiredEnd > textureUploadArena.buffer.length) {
			NSUInteger capacity = std::max<NSUInteger>(256 * 1024, requiredEnd);
			if (textureUploadArena.buffer && textureUploadArena.buffer.length <= ctx.device.maxBufferLength / 2) {
				capacity = std::max(capacity, textureUploadArena.buffer.length * 2);
			}
			capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
			if (capacity < requiredEnd) return false;
			id<MTLBuffer> replacement = [ctx.device newBufferWithLength:capacity options:kTextureUploadBufferOptions];
			if (!replacement) return false;
			// The command buffer retains previously encoded staging buffers. Reset the
			// offset only when switching to a newly allocated arena buffer.
			textureUploadArena.buffer = replacement;
			textureUploadArena.used = 0;
			stagingOffset = 0;
			requiredEnd = (NSUInteger)batchBytes;
		}

		id<MTLBuffer> stagingBuffer = textureUploadArena.buffer;
		uint8_t *stagingBytes = (uint8_t *)stagingBuffer.contents + stagingOffset;
		memcpy(stagingBytes, bytes + uploadedSlices * bytesPerImage, batchBytes);
		textureUploadArena.used = requiredEnd;

		id<MTLBlitCommandEncoder> blitEncoder = [targetCommandBuffer blitCommandEncoder];
		if (!blitEncoder) return false;
		for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
			[blitEncoder copyFromBuffer:stagingBuffer
				sourceOffset:stagingOffset + sliceIndex * bytesPerImage
				sourceBytesPerRow:rowBytes
				sourceBytesPerImage:0
				sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
				toTexture:destinationTexture
				destinationSlice:firstSlice + uploadedSlices + sliceIndex
				destinationLevel:(NSUInteger)level
				destinationOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)];
		}
		[blitEncoder endEncoding];
		uploadedSlices += batchSlices;
	}
	return true;
}

static bool readbackCompressedSlices(id<MTLTexture> sourceTexture, GLint level, NSUInteger firstSlice,
		NSUInteger sliceCount, GLsizei width, GLsizei height, const CompressedFormatInfo &info,
		uint8_t *destination, NSUInteger rowBytes) {
	const NSUInteger availableSlices = sourceTexture && sourceTexture.textureType == MTLTextureTypeCube
		? 6 : (sourceTexture ? sourceTexture.arrayLength : 0);
	if (!sourceTexture || !destination || !ctx.device || !ctx.commandQueue || sliceCount == 0 ||
		width <= 0 || height <= 0 || rowBytes == 0 || level < 0 ||
		(NSUInteger)level >= sourceTexture.mipmapLevelCount || firstSlice >= availableSlices ||
		sliceCount > availableSlices - firstSlice || info.blockHeight == 0) return false;

	const NSUInteger blockRows = ((NSUInteger)height + info.blockHeight - 1) / info.blockHeight;
	if (blockRows > SIZE_MAX / rowBytes) return false;
	const size_t bytesPerImage = (size_t)rowBytes * (size_t)blockRows;
	if (bytesPerImage == 0 || bytesPerImage > ctx.device.maxBufferLength ||
		sliceCount > SIZE_MAX / bytesPerImage) return false;
	const NSUInteger maxSlicesPerBatch = std::max<NSUInteger>(1, ctx.device.maxBufferLength / bytesPerImage);

	NSUInteger copiedSlices = 0;
	while (copiedSlices < sliceCount) {
		const NSUInteger batchSlices = std::min<NSUInteger>(sliceCount - copiedSlices, maxSlicesPerBatch);
		if (batchSlices > SIZE_MAX / bytesPerImage) return false;
		const NSUInteger stagingLength = (NSUInteger)(bytesPerImage * batchSlices);
		if (stagingLength == 0 || stagingLength > ctx.device.maxBufferLength) return false;
		endRenderEncoder();
		if (rowBytes > std::numeric_limits<NSUInteger>::max() - 255u) return false;
		const NSUInteger stagingRowBytes = (rowBytes + 255u) & ~(NSUInteger)255u;
		if (blockRows > SIZE_MAX / stagingRowBytes) return false;
		const NSUInteger stagingImageBytes = stagingRowBytes * blockRows;
		if (stagingImageBytes == 0 || batchSlices > SIZE_MAX / stagingImageBytes) return false;
		const NSUInteger paddedStagingLength = stagingImageBytes * batchSlices;
		bool metal4ReadbackSucceeded = false;

		id<MTLBuffer> metal4StagingBuffer = nil;
		if (ctx.metal4Enabled && paddedStagingLength <= ctx.device.maxBufferLength) {
			metal4StagingBuffer = [ctx.device newBufferWithLength:paddedStagingLength options:MTLResourceStorageModeShared];
			if (metal4StagingBuffer) {
				metal4ReadbackSucceeded = true;
				for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
					if (!gl4metalMetal4CopyTextureToBuffer(sourceTexture, firstSlice + copiedSlices + sliceIndex,
							(NSUInteger)level, MTLOriginMake(0, 0, 0), MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1),
							metal4StagingBuffer, sliceIndex * stagingImageBytes, stagingRowBytes, 0, MTLBlitOptionNone)) {
						metal4ReadbackSucceeded = false;
						break;
					}
				}
				if (metal4ReadbackSucceeded) {
					flushCommandBuffer(true);
					metal4ReadbackSucceeded = ctx.metal4InteropEvent &&
						ctx.metal4InteropEvent.signaledValue >= ctx.metal4LastComputeEventValue &&
						ctx.metal4LastComputeEventValue != 0;
				}
			}
		}

		if (metal4ReadbackSucceeded) {
			const uint8_t *stagingBytes = (const uint8_t *)metal4StagingBuffer.contents;
			for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
				for (NSUInteger blockRow = 0; blockRow < blockRows; ++blockRow) {
					memcpy(destination + (copiedSlices + sliceIndex) * bytesPerImage + blockRow * rowBytes,
						stagingBytes + sliceIndex * stagingImageBytes + blockRow * stagingRowBytes, rowBytes);
				}
			}
		} else {
			// A partial Metal 4 attempt must finish before the classic queue reads
			// the same texture. Reuse its staging allocation when possible and keep
			// the Metal-required padded row pitch for compressed blocks.
			flushCommandBuffer(true);

			if (!metal4StagingBuffer) metal4StagingBuffer = [ctx.device newBufferWithLength:paddedStagingLength options:MTLResourceStorageModeShared];
			id<MTLBuffer> stagingBuffer = metal4StagingBuffer;
			if (!stagingBuffer) {
				stagingBuffer = [ctx.device newBufferWithLength:paddedStagingLength options:MTLResourceStorageModeShared];
			}
			// id<MTLBuffer> stagingBuffer = [ctx.device newBufferWithLength:paddedStagingLength options:MTLResourceStorageModeShared];
			if (!stagingBuffer || !gl4metalEnsureClassicCommandBuffer()) return false;
			id<MTLCommandBuffer> targetCommandBuffer = ctx.currentCommandBuffer;
			id<MTLBlitCommandEncoder> blitEncoder = [targetCommandBuffer blitCommandEncoder];
			if (!blitEncoder) return false;
			for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
				[blitEncoder copyFromTexture:sourceTexture
					sourceSlice:firstSlice + copiedSlices + sliceIndex
					sourceLevel:(NSUInteger)level
					sourceOrigin:MTLOriginMake(0, 0, 0)
					sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
					toBuffer:stagingBuffer
					destinationOffset:sliceIndex * stagingImageBytes
					destinationBytesPerRow:stagingRowBytes
					destinationBytesPerImage:0];
			}
			[blitEncoder endEncoding];
			flushCommandBuffer(true);
			if (!ctx.last_committed_buffer || ctx.last_committed_buffer.status != MTLCommandBufferStatusCompleted) return false;
			const uint8_t *stagingBytes = (const uint8_t *)stagingBuffer.contents;
			for (NSUInteger sliceIndex = 0; sliceIndex < batchSlices; ++sliceIndex) {
				for (NSUInteger blockRow = 0; blockRow < blockRows; ++blockRow) {
					memcpy(destination + (copiedSlices + sliceIndex) * bytesPerImage + blockRow * rowBytes,
						stagingBytes + sliceIndex * stagingImageBytes + blockRow * stagingRowBytes, rowBytes);
				}
			}
		}
		copiedSlices += batchSlices;
	}
	return true;
}

static bool defineCompressed2D(TextureObject &textureObject, GLenum textureTarget, GLint level,
		GLenum internalformat, GLsizei width, GLsizei height, GLint border,
		GLsizei imageSize, const void *data) {
	CompressedFormatInfo info;
	if (!compressedFormatInfo(internalformat, info)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
	if (level < 0 || width <= 0 || height <= 0 || border != 0 ||
		width > GL4METAL_MAX_TEXTURE_SIZE || height > GL4METAL_MAX_TEXTURE_SIZE) {
		gl4metalSetError(GL_INVALID_VALUE); return false;
	}
	NSUInteger rowBytes = 0, blockRows = 0;
	size_t expectedBytes = 0;
	if (!compressedImageLayout(width, height, info, rowBytes, blockRows, expectedBytes) ||
		expectedBytes > (size_t)INT_MAX || imageSize < 0 || (size_t)imageSize != expectedBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCompressedTexImage2D: imageSize=%d does not match required compressed payload %lu", imageSize, (unsigned long)expectedBytes);
		return false;
	}
	const bool cube = cubeFace(textureTarget) >= 0;
	if (textureTarget != GL_TEXTURE_2D && !cube) { gl4metalSetError(GL_INVALID_ENUM); return false; }
	if (cube && width != height) { gl4metalSetError(GL_INVALID_VALUE); return false; }
	if (textureObject.immutable_storage) { gl4metalSetError(GL_INVALID_OPERATION); return false; }
	if (textureObject.internalformat != 0 && textureObject.internalformat != internalformat) {
		gl4metalSetError(GL_INVALID_OPERATION); return false;
	}
	if (level == 0) {
		const MTLTextureType expectedType = cube ? MTLTextureTypeCube : MTLTextureType2D;
		if (!textureObject.texture || textureObject.texture.textureType != expectedType ||
			textureObject.width != width || textureObject.height != height ||
			textureObject.texture.pixelFormat != info.pixelFormat) {
			MTLTextureDescriptor *descriptor = cube
				? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:info.pixelFormat size:(NSUInteger)width mipmapped:mipLevelCount(textureObject, width, height) > 1]
				: [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.pixelFormat width:(NSUInteger)width height:(NSUInteger)height mipmapped:mipLevelCount(textureObject, width, height) > 1];
			descriptor.textureType = expectedType;
			descriptor.mipmapLevelCount = mipLevelCount(textureObject, width, height);
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead;
			id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
			if (!newTexture) { gl4metalSetError(GL_OUT_OF_MEMORY); return false; }
			endRenderEncoder();
			gl4metalSetTextureStorage(textureObject, newTexture);
			textureObject.width = width;
			textureObject.height = height;
			textureObject.depth = 1;
			textureObject.internalformat = internalformat;
		} else {
			textureObject.internalformat = internalformat;
		}
	} else {
		const MTLTextureType expectedType = cube ? MTLTextureTypeCube : MTLTextureType2D;
		if (!textureObject.texture || textureObject.texture.textureType != expectedType || textureObject.texture.pixelFormat != info.pixelFormat ||
			(NSUInteger)level >= textureObject.texture.mipmapLevelCount ||
			width != std::max(1, textureObject.width >> level) ||
			height != std::max(1, textureObject.height >> level)) {
			gl4metalSetError(GL_INVALID_VALUE); return false;
		}
	}
	if (data) {
		const uint8_t *source = resolveUnpackPointer(data, 0, expectedBytes, "glCompressedTexImage2D");
		if (!source) return false;
		const NSUInteger slice = cube ? (NSUInteger)cubeFace(textureTarget) : 0;
		if (!uploadCompressedSlices(textureObject, level, slice, 1, 0, 0, width, height, info, source, rowBytes)) {
			gl4metalSetError(GL_INVALID_OPERATION); return false;
		}
	}
	return true;
}
} // namespace

void glCompressedTexImage1D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data) {
	(void)level; (void)internalformat; (void)width; (void)border; (void)imageSize; (void)data;
	if (target != GL_TEXTURE_1D) gl4metalSetError(GL_INVALID_ENUM);
	else gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glCompressedTexImage1D: compressed 1D textures are not supported by Metal on iOS");
}

void glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data) {
	if (target != GL_TEXTURE_2D && cubeFace(target) < 0) { gl4metalSetError(GL_INVALID_ENUM); return; }
	const GLenum bindingTarget = cubeFace(target) >= 0 ? GL_TEXTURE_CUBE_MAP : target;
	TextureObject *textureObject = boundTexture(bindingTarget);
	if (!textureObject) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	defineCompressed2D(*textureObject, target, level, internalformat, width, height, border, imageSize, data);
}

void glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data) {
	if (target != GL_TEXTURE_2D_ARRAY) {
		gl4metalSetError(target == GL_TEXTURE_3D ? GL_INVALID_OPERATION : GL_INVALID_ENUM);
		GL4METAL_LOG("glCompressedTexImage3D: compressed Metal textures support 2D arrays, not compressed 3D volumes");
		return;
	}
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	CompressedFormatInfo info;
	if (!compressedFormatInfo(internalformat, info)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (level < 0 || width <= 0 || height <= 0 || depth <= 0 || border != 0 ||
		width > GL4METAL_MAX_TEXTURE_SIZE || height > GL4METAL_MAX_TEXTURE_SIZE || depth > GL4METAL_MAX_ARRAY_TEXTURE_LAYERS) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	NSUInteger rowBytes = 0, blockRows = 0;
	size_t imageBytes = 0;
	if (!compressedImageLayout(width, height, info, rowBytes, blockRows, imageBytes) ||
		imageBytes > SIZE_MAX / (size_t)depth || (size_t)imageSize != imageBytes * (size_t)depth) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	if (textureObject->immutable_storage || (textureObject->internalformat && textureObject->internalformat != internalformat)) {
		gl4metalSetError(GL_INVALID_OPERATION); return;
	}
	if (level != 0) {
		if (!textureObject->texture || textureObject->texture.textureType != MTLTextureType2DArray ||
			textureObject->texture.pixelFormat != info.pixelFormat || (NSUInteger)level >= textureObject->texture.mipmapLevelCount ||
			width != std::max(1, textureObject->width >> level) || height != std::max(1, textureObject->height >> level) || depth != textureObject->depth) {
			gl4metalSetError(GL_INVALID_VALUE); return;
		}
	} else if (!textureObject->texture || textureObject->texture.textureType != MTLTextureType2DArray ||
		textureObject->width != width || textureObject->height != height || textureObject->depth != depth ||
		textureObject->texture.pixelFormat != info.pixelFormat) {
		MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
		descriptor.textureType = MTLTextureType2DArray;
		descriptor.pixelFormat = info.pixelFormat;
		descriptor.width = (NSUInteger)width;
		descriptor.height = (NSUInteger)height;
		descriptor.depth = 1;
		descriptor.arrayLength = (NSUInteger)depth;
		descriptor.mipmapLevelCount = mipLevelCount(*textureObject, width, height);
		descriptor.sampleCount = 1;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> newTexture = [ctx.device newTextureWithDescriptor:descriptor];
		if (!newTexture) { gl4metalSetError(GL_OUT_OF_MEMORY); return; }
		endRenderEncoder();
		gl4metalSetTextureStorage(*textureObject, newTexture);
		textureObject->width = width; textureObject->height = height; textureObject->depth = depth;
		textureObject->internalformat = internalformat;
	}
	if (data) {
		const uint8_t *source = resolveUnpackPointer(data, 0, (size_t)imageSize, "glCompressedTexImage3D");
		if (!source) return;
		if (!uploadCompressedSlices(*textureObject, level, 0, (NSUInteger)depth,
			0, 0, width, height, info, source, rowBytes)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
	}
}

void glCompressedTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data) {
	(void)level; (void)xoffset; (void)width; (void)format; (void)imageSize; (void)data;
	if (target != GL_TEXTURE_1D) gl4metalSetError(GL_INVALID_ENUM); else gl4metalSetError(GL_INVALID_OPERATION);
	GL4METAL_LOG("glCompressedTexSubImage1D: compressed 1D textures are not supported by Metal on iOS");
}

void glCompressedTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data) {
	if (target != GL_TEXTURE_2D && cubeFace(target) < 0) { gl4metalSetError(GL_INVALID_ENUM); return; }
	const bool cube = cubeFace(target) >= 0;
	TextureObject *textureObject = boundTexture(cube ? GL_TEXTURE_CUBE_MAP : target);
	if (!textureObject || !textureObject->texture) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	CompressedFormatInfo info;
	if (!compressedFormatInfo(format, info)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (format != textureObject->internalformat || textureObject->texture.pixelFormat != info.pixelFormat) {
		gl4metalSetError(GL_INVALID_OPERATION); return;
	}
	if (level < 0 || (NSUInteger)level >= textureObject->texture.mipmapLevelCount || width < 0 || height < 0) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	const GLsizei levelWidth = std::max(1, textureObject->width >> level);
	const GLsizei levelHeight = std::max(1, textureObject->height >> level);
	if (!compressedSubRegionValid(xoffset, yoffset, width, height, levelWidth, levelHeight, info)) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	if (width == 0 || height == 0) return;
	NSUInteger rowBytes = 0, blockRows = 0;
	size_t expectedBytes = 0;
	if (!compressedImageLayout(width, height, info, rowBytes, blockRows, expectedBytes) || imageSize < 0 || (size_t)imageSize != expectedBytes) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	const uint8_t *source = resolveUnpackPointer(data, 0, expectedBytes, "glCompressedTexSubImage2D");
	if (!source) return;
	const NSUInteger slice = cube ? (NSUInteger)cubeFace(target) : 0;
	if (!uploadCompressedSlices(*textureObject, level, slice, 1, xoffset, yoffset,
		width, height, info, source, rowBytes))
		gl4metalSetError(GL_INVALID_OPERATION);
}

void glCompressedTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data) {
	if (target != GL_TEXTURE_2D_ARRAY) { gl4metalSetError(target == GL_TEXTURE_3D ? GL_INVALID_OPERATION : GL_INVALID_ENUM); return; }
	TextureObject *textureObject = boundTexture(target);
	if (!textureObject || !textureObject->texture || textureObject->texture.textureType != MTLTextureType2DArray) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	CompressedFormatInfo info;
	if (!compressedFormatInfo(format, info)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (format != textureObject->internalformat || textureObject->texture.pixelFormat != info.pixelFormat) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (level < 0 || (NSUInteger)level >= textureObject->texture.mipmapLevelCount || depth < 0 ||
		zoffset < 0 || zoffset > textureObject->depth || depth > textureObject->depth - zoffset) { gl4metalSetError(GL_INVALID_VALUE); return; }
	const GLsizei levelWidth = std::max(1, textureObject->width >> level);
	const GLsizei levelHeight = std::max(1, textureObject->height >> level);
	if (!compressedSubRegionValid(xoffset, yoffset, width, height, levelWidth, levelHeight, info)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (width == 0 || height == 0 || depth == 0) return;
	NSUInteger rowBytes = 0, blockRows = 0;
	size_t imageBytes = 0;
	if (!compressedImageLayout(width, height, info, rowBytes, blockRows, imageBytes) || imageBytes > SIZE_MAX / (size_t)depth || (size_t)imageSize != imageBytes * (size_t)depth) { gl4metalSetError(GL_INVALID_VALUE); return; }
	const uint8_t *source = resolveUnpackPointer(data, 0, (size_t)imageSize, "glCompressedTexSubImage3D");
	if (!source) return;
	if (!uploadCompressedSlices(*textureObject, level, (NSUInteger)zoffset, (NSUInteger)depth,
		xoffset, yoffset, width, height, info, source, rowBytes)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
}

// MARK: - Texture parameters

static bool validTextureMinFilter(GLint value) {
	return value == GL_NEAREST || value == GL_LINEAR || value == GL_NEAREST_MIPMAP_NEAREST ||
		value == GL_LINEAR_MIPMAP_NEAREST || value == GL_NEAREST_MIPMAP_LINEAR || value == GL_LINEAR_MIPMAP_LINEAR;
}

static bool validTextureMagFilter(GLint value) {
	return value == GL_NEAREST || value == GL_LINEAR;
}

static bool validTextureWrap(GLint value) {
	return value == GL_REPEAT || value == GL_CLAMP_TO_EDGE || value == GL_MIRRORED_REPEAT || value == GL_CLAMP_TO_BORDER;
}

static bool validTextureCompareFunc(GLint value) {
	switch (value) {
		case GL_NEVER: case GL_LESS: case GL_EQUAL: case GL_LEQUAL:
		case GL_GREATER: case GL_NOTEQUAL: case GL_GEQUAL: case GL_ALWAYS: return true;
		default: return false;
	}
}

static bool validTextureSwizzle(GLint value) {
	switch (value) {
		case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_ZERO: case GL_ONE:
			return true;
		default:
			return false;
	}
}

static GLint textureSwizzleComponentIndex(GLenum pname) {
	switch (pname) {
		case GL_TEXTURE_SWIZZLE_R: return 0;
		case GL_TEXTURE_SWIZZLE_G: return 1;
		case GL_TEXTURE_SWIZZLE_B: return 2;
		case GL_TEXTURE_SWIZZLE_A: return 3;
		default: return -1;
	}
}

static bool setTextureSwizzle(TextureObject &texture, const GLint values[4]) {
	for (int component = 0; component < 4; ++component) {
		if (!validTextureSwizzle(values[component])) {
			gl4metalSetError(GL_INVALID_ENUM);
			return false;
		}
	}
	bool changed = false;
	for (int component = 0; component < 4; ++component) {
		const GLenum value = (GLenum)values[component];
		if (texture.swizzle[component] != value) changed = true;
	}
	if (!changed) return true;
	for (int component = 0; component < 4; ++component) texture.swizzle[component] = (GLenum)values[component];
	gl4metalInvalidateTextureSamplingViews(texture);
	return true;
}

static bool setTextureSwizzleComponent(TextureObject &texture, GLenum pname, GLint value) {
	const GLint component = textureSwizzleComponentIndex(pname);
	if (component < 0) return false;
	if (!validTextureSwizzle(value)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return true;
	}
	if (texture.swizzle[component] != (GLenum)value) {
		texture.swizzle[component] = (GLenum)value;
		gl4metalInvalidateTextureSamplingViews(texture);
	}
	return true;
}

static bool textureSwizzleFloatToInt(GLfloat value, GLint &converted) {
	const double wideValue = (double)value;
	if (!std::isfinite(value) || wideValue < (double)std::numeric_limits<GLint>::min() ||
		wideValue > (double)std::numeric_limits<GLint>::max()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	converted = (GLint)value;
	return true;
}

static bool validTextureParameterTarget(GLenum target) {
	switch (target) {
		case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D:
		case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP:
		case GL_TEXTURE_BUFFER: case GL_TEXTURE_CUBE_MAP_ARRAY:
		case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
			return true;
		default: return false;
	}
}

static GLint textureParameterFloatToInt(GLfloat value) {
	if (std::isnan(value)) return 0;
	if (value >= (GLfloat)std::numeric_limits<GLint>::max()) return std::numeric_limits<GLint>::max();
	if (value <= (GLfloat)std::numeric_limits<GLint>::min()) return std::numeric_limits<GLint>::min();
	return (GLint)value;
}

static GLint textureParameterQueryInt(GLfloat value) {
	return textureParameterFloatToInt(value);
}

void glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *texture = boundTexture(target);
	if (!texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexParameterf: no texture bound to target 0x%X", target);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:
		case GL_TEXTURE_MAX_LOD:
		case GL_TEXTURE_LOD_BIAS:
			if (!std::isfinite(param)) { gl4metalSetError(GL_INVALID_VALUE); return; }
			if (pname == GL_TEXTURE_MIN_LOD) texture->min_lod = param;
			else if (pname == GL_TEXTURE_MAX_LOD) texture->max_lod = param;
			else texture->lod_bias = param;
			return;
		case GL_TEXTURE_MAX_ANISOTROPY_EXT:
			if (!std::isfinite(param)) { gl4metalSetError(GL_INVALID_VALUE); return; }
			texture->max_anisotropy = std::min(std::max(param, 1.0f), 16.0f);
			return;
		default:
			if (!std::isfinite(param) || (double)param > (double)std::numeric_limits<GLint>::max() ||
				(double)param < (double)std::numeric_limits<GLint>::min()) {
				gl4metalSetError(GL_INVALID_VALUE);
				return;
			}
			glTexParameteri(target, pname, (GLint)param);
			return;
	}
}

void glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *texture = boundTexture(target);
	if (!texture) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (pname == GL_TEXTURE_SWIZZLE_RGBA) {
		GLint converted[4];
		for (int component = 0; component < 4; ++component) {
			if (!textureSwizzleFloatToInt(params[component], converted[component])) return;
		}
		setTextureSwizzle(*texture, converted);
		return;
	}
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = params[0] == 0.0f && params[1] == 0.0f && params[2] == 0.0f && params[3] == 0.0f;
		const bool opaqueBlack = params[0] == 0.0f && params[1] == 0.0f && params[2] == 0.0f && params[3] == 1.0f;
		const bool opaqueWhite = params[0] == 1.0f && params[1] == 1.0f && params[2] == 1.0f && params[3] == 1.0f;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glTexParameterfv: Metal supports transparent-black, opaque-black and opaque-white border colours only");
			return;
		}
		for (int component = 0; component < 4; ++component) texture->border_color[component] = params[component];
		return;
	}
	glTexParameterf(target, pname, params[0]);
}

void glTexParameteri(GLenum target, GLenum pname, GLint param) {
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *texture = boundTexture(target);
	if (!texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glTexParameteri: no texture bound to target 0x%X", target);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_SWIZZLE_R:
		case GL_TEXTURE_SWIZZLE_G:
		case GL_TEXTURE_SWIZZLE_B:
		case GL_TEXTURE_SWIZZLE_A:
			setTextureSwizzleComponent(*texture, pname, param);
			return;
		case GL_TEXTURE_MIN_FILTER:
			if (!validTextureMinFilter(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->min_filter = (GLenum)param; break;
		case GL_TEXTURE_MAG_FILTER:
			if (!validTextureMagFilter(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->mag_filter = (GLenum)param; break;
		case GL_TEXTURE_WRAP_S:
			if (!validTextureWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->wrap_s = (GLenum)param; break;
		case GL_TEXTURE_WRAP_T:
			if (!validTextureWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->wrap_t = (GLenum)param; break;
		case GL_TEXTURE_WRAP_R:
			if (!validTextureWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->wrap_r = (GLenum)param; break;
		case GL_TEXTURE_BASE_LEVEL:
			if (param < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
			texture->base_level = param; break;
		case GL_TEXTURE_MAX_LEVEL:
			if (param < 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
			texture->max_level = param; break;
		case GL_TEXTURE_COMPARE_MODE:
			if (param != GL_NONE && param != GL_COMPARE_REF_TO_TEXTURE) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->compare_mode = (GLenum)param; break;
		case GL_TEXTURE_COMPARE_FUNC:
			if (!validTextureCompareFunc(param)) { gl4metalSetError(GL_INVALID_ENUM); return; }
			texture->compare_func = (GLenum)param; break;
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
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *texture = boundTexture(target);
	if (!texture) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = params[0] == 0 && params[1] == 0 && params[2] == 0 && params[3] == 0;
		const bool opaqueBlack = params[0] == 0 && params[1] == 0 && params[2] == 0 && params[3] == 1;
		const bool opaqueWhite = params[0] == 1 && params[1] == 1 && params[2] == 1 && params[3] == 1;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		for (int component = 0; component < 4; ++component) texture->border_color[component] = (GLfloat)params[component];
		return;
	}
	if (pname == GL_TEXTURE_SWIZZLE_RGBA) {
		setTextureSwizzle(*texture, params);
		return;
	}
	if (pname == GL_TEXTURE_SWIZZLE_R || pname == GL_TEXTURE_SWIZZLE_G ||
		pname == GL_TEXTURE_SWIZZLE_B || pname == GL_TEXTURE_SWIZZLE_A) {
		setTextureSwizzleComponent(*texture, pname, params[0]);
		return;
	}
	glTexParameteri(target, pname, params[0]);
}

void glTexParameterIiv(GLenum target, GLenum pname, const GLint *params) {
	glTexParameteriv(target, pname, params);
}

void glTexParameterIuiv(GLenum target, GLenum pname, const GLuint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (!boundTexture(target)) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (pname == GL_TEXTURE_SWIZZLE_RGBA) {
		GLint converted[4];
		for (int component = 0; component < 4; ++component) {
			if (params[component] > (GLuint)std::numeric_limits<GLint>::max()) { gl4metalSetError(GL_INVALID_ENUM); return; }
			converted[component] = (GLint)params[component];
		}
		glTexParameteriv(target, pname, converted);
		return;
	}
	if (pname == GL_TEXTURE_SWIZZLE_R || pname == GL_TEXTURE_SWIZZLE_G ||
		pname == GL_TEXTURE_SWIZZLE_B || pname == GL_TEXTURE_SWIZZLE_A) {
		if (params[0] > (GLuint)std::numeric_limits<GLint>::max()) { gl4metalSetError(GL_INVALID_ENUM); return; }
		glTexParameteri(target, pname, (GLint)params[0]);
		return;
	}
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = params[0] == 0u && params[1] == 0u && params[2] == 0u && params[3] == 0u;
		const bool opaqueBlack = params[0] == 0u && params[1] == 0u && params[2] == 0u && params[3] == 1u;
		const bool opaqueWhite = params[0] == 1u && params[1] == 1u && params[2] == 1u && params[3] == 1u;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		GLint converted[4] = {(GLint)params[0], (GLint)params[1], (GLint)params[2], (GLint)params[3]};
		glTexParameteriv(target, pname, converted);
		return;
	}
	if (params[0] > (GLuint)std::numeric_limits<GLint>::max()) { gl4metalSetError(GL_INVALID_VALUE); return; }
	glTexParameteri(target, pname, (GLint)params[0]);
}

// MARK: - Texture queries

namespace {

static bool getTexImageBytesPerPixel(MTLPixelFormat format, NSUInteger &bytesPerPixel) {
	switch (format) {
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Snorm: bytesPerPixel = 1; return true;
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Snorm: bytesPerPixel = 2; return true;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatRGB10A2Unorm:
		case MTLPixelFormatRG11B10Float:
		case MTLPixelFormatRGB9E5Float: bytesPerPixel = 4; return true;
		case MTLPixelFormatA1BGR5Unorm:
			bytesPerPixel = 2; return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatDepth16Unorm: bytesPerPixel = 2; return true;
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float: bytesPerPixel = 4; return true;
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Snorm:
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
	const bool is1D = texture.textureType == MTLTextureType1D;
	if (is1D && height != 1) return false;
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
		if (is1D) {
			[texture getBytes:out.data() bytesPerRow:0 fromRegion:MTLRegionMake1D(0, width) mipmapLevel:(NSUInteger)level];
		} else {
			[texture getBytes:out.data() bytesPerRow:rowBytes bytesPerImage:0
				fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:(NSUInteger)level slice:slice];
		}
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
	id<MTLBuffer> staging = nil;
	bool readbackCompleted = false;

	if (ctx.metal4Enabled) {
		staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		if (staging && gl4metalMetal4CopyTextureToBuffer(texture, slice, (NSUInteger)level,
				MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, 1), staging, 0, stagingRowBytes, 0, blitOptions)) {
			flushCommandBuffer(true);
			readbackCompleted = ctx.metal4LastComputeEventValue == 0 ||
				(ctx.metal4InteropEvent && ctx.metal4InteropEvent.signaledValue >= ctx.metal4LastComputeEventValue);
		}
	}

	if (!readbackCompleted) {
		// Reuse an MTL4 staging allocation if its command was only partially encoded.
		flushCommandBuffer(true);
		if (!staging) staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		if (!staging) return false;
		id<MTLCommandQueue> queue = ctx.commandQueue;
		id<MTLCommandBuffer> commandBuffer = queue ? [queue commandBuffer] : nil;
		id<MTLBlitCommandEncoder> blit = commandBuffer ? [commandBuffer blitCommandEncoder] : nil;
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
	}
	const uint8_t *src = (const uint8_t *)staging.contents;
	for (NSUInteger y = 0; y < height; ++y) {
		memcpy(out.data() + y * rowBytes, src + y * stagingRowBytes, rowBytes);
	}
	return true;
}

// Read an entire 3D mip level. Shared textures use the direct CPU-visible path;
// private textures are copied through one aligned staging buffer with all images
// represented by source/destinationBytesPerImage.
static bool readTextureLevel3D(id<MTLTexture> texture, GLint level,
						   NSUInteger width, NSUInteger height, NSUInteger depth,
						   NSUInteger bytesPerPixel, MTLBlitOption blitOptions,
						   std::vector<uint8_t> &out) {
	if (!texture || texture.textureType != MTLTextureType3D || width == 0 || height == 0 ||
		depth == 0 || bytesPerPixel == 0 || width > SIZE_MAX / bytesPerPixel) return false;
	const NSUInteger rowBytes = width * bytesPerPixel;
	if (height > SIZE_MAX / rowBytes) return false;
	const NSUInteger imageBytes = rowBytes * height;
	if (depth > SIZE_MAX / imageBytes) return false;
	const NSUInteger totalBytes = imageBytes * depth;
	try { out.resize((size_t)totalBytes); }
	catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGetTexImage: 3D CPU readback allocation failed (%lu bytes)", (unsigned long)totalBytes);
		return false;
	}

	flushCommandBuffer(true);
	const MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, depth);
	if (texture.storageMode == MTLStorageModeShared) {
		[texture getBytes:out.data() bytesPerRow:rowBytes bytesPerImage:imageBytes
			fromRegion:region mipmapLevel:(NSUInteger)level slice:0];
		return true;
	}

	const NSUInteger alignment = 256;
	if (rowBytes > SIZE_MAX - (alignment - 1)) return false;
	const NSUInteger stagingRowBytes = (rowBytes + alignment - 1) & ~(alignment - 1);
	if (height > SIZE_MAX / stagingRowBytes) return false;
	const NSUInteger stagingImageBytes = stagingRowBytes * height;
	if (depth > SIZE_MAX / stagingImageBytes) return false;
	const NSUInteger stagingBytes = stagingImageBytes * depth;
	if (stagingBytes == 0 || stagingBytes > texture.device.maxBufferLength) {
		GL4METAL_LOG("glGetTexImage: private 3D readback exceeds maxBufferLength (%lu bytes)", (unsigned long)stagingBytes);
		return false;
	}
	id<MTLBuffer> staging = nil;
	bool readbackCompleted = false;

	if (ctx.metal4Enabled) {
		staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		if (staging && gl4metalMetal4CopyTextureToBuffer(texture, 0, (NSUInteger)level,
				MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, depth), staging, 0,
				stagingRowBytes, stagingImageBytes, blitOptions)) {
			flushCommandBuffer(true);
			readbackCompleted = ctx.metal4LastComputeEventValue == 0 ||
				(ctx.metal4InteropEvent && ctx.metal4InteropEvent.signaledValue >= ctx.metal4LastComputeEventValue);
		}
	}

	if (!readbackCompleted) {
		flushCommandBuffer(true);
		if (!staging) staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		if (!staging) return false;
		id<MTLCommandQueue> queue = ctx.commandQueue;
		id<MTLCommandBuffer> commandBuffer = queue ? [queue commandBuffer] : nil;
		id<MTLBlitCommandEncoder> blit = commandBuffer ? [commandBuffer blitCommandEncoder] : nil;
		if (!commandBuffer || !blit) return false;
		[blit copyFromTexture:texture sourceSlice:0 sourceLevel:(NSUInteger)level
			sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, depth)
			toBuffer:staging destinationOffset:0 destinationBytesPerRow:stagingRowBytes
			destinationBytesPerImage:stagingImageBytes options:blitOptions];
		[blit endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
			GL4METAL_LOG("glGetTexImage: private 3D blit readback failed: %s",
				commandBuffer.error.localizedDescription.UTF8String ?: "unknown Metal error");
			return false;
		}
	}
	const uint8_t *source = (const uint8_t *)staging.contents;
	for (NSUInteger image = 0; image < depth; ++image) {
		for (NSUInteger row = 0; row < height; ++row) {
			memcpy(out.data() + image * imageBytes + row * rowBytes,
				source + image * stagingImageBytes + row * stagingRowBytes, rowBytes);
		}
	}
	return true;
}

// Read a 2D-array mip as a tightly packed sequence of images.  Unlike a 3D
// texture, each array layer is an independent 2D slice, so the Metal blit API
// must be issued once per slice.  The shared-buffer path still uses one
// allocation and one command buffer for the entire array.
static bool readTextureLevelArray(id<MTLTexture> texture, GLint level,
							  NSUInteger width, NSUInteger height, NSUInteger layers,
							  NSUInteger bytesPerPixel, MTLBlitOption blitOptions,
							  std::vector<uint8_t> &out) {
	if (!texture || texture.textureType != MTLTextureType2DArray || width == 0 || height == 0 ||
		layers == 0 || bytesPerPixel == 0 || width > SIZE_MAX / bytesPerPixel) return false;
	const NSUInteger rowBytes = width * bytesPerPixel;
	if (height > SIZE_MAX / rowBytes) return false;
	const NSUInteger imageBytes = rowBytes * height;
	if (layers > SIZE_MAX / imageBytes) return false;
	const NSUInteger totalBytes = imageBytes * layers;
	try { out.resize((size_t)totalBytes); }
	catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glGetTexImage: array CPU readback allocation failed (%lu bytes)", (unsigned long)totalBytes);
		return false;
	}

	flushCommandBuffer(true);
	if (texture.storageMode == MTLStorageModeShared) {
		for (NSUInteger layer = 0; layer < layers; ++layer) {
			[texture getBytes:out.data() + layer * imageBytes bytesPerRow:rowBytes bytesPerImage:0
				fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:(NSUInteger)level slice:layer];
		}
		return true;
	}

	const NSUInteger alignment = 256;
	if (rowBytes > SIZE_MAX - (alignment - 1)) return false;
	const NSUInteger stagingRowBytes = (rowBytes + alignment - 1) & ~(alignment - 1);
	if (height > SIZE_MAX / stagingRowBytes) return false;
	const NSUInteger stagingImageBytes = stagingRowBytes * height;
	if (layers > SIZE_MAX / stagingImageBytes) return false;
	const NSUInteger stagingBytes = stagingImageBytes * layers;
	if (stagingBytes == 0 || stagingBytes > texture.device.maxBufferLength) return false;
	id<MTLBuffer> staging = nil;
	bool metal4ReadbackSucceeded = false;

	if (ctx.metal4Enabled) {
		staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		metal4ReadbackSucceeded = staging != nil;
		for (NSUInteger layer = 0; metal4ReadbackSucceeded && layer < layers; ++layer) {
			if (!gl4metalMetal4CopyTextureToBuffer(texture, layer, (NSUInteger)level,
					MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, 1), staging,
					layer * stagingImageBytes, stagingRowBytes, 0, blitOptions)) {
				metal4ReadbackSucceeded = false;
			}
		}
		if (metal4ReadbackSucceeded) {
			flushCommandBuffer(true);
			metal4ReadbackSucceeded = ctx.metal4LastComputeEventValue == 0 ||
				(ctx.metal4InteropEvent && ctx.metal4InteropEvent.signaledValue >= ctx.metal4LastComputeEventValue);
		}
	}

	if (!metal4ReadbackSucceeded) {
		// Complete any partially encoded MTL4 copies before submitting classic work.
		flushCommandBuffer(true);
		if (!staging) staging = [texture.device newBufferWithLength:stagingBytes options:MTLResourceStorageModeShared];
		if (!staging) return false;
		id<MTLCommandQueue> queue = ctx.commandQueue;
		id<MTLCommandBuffer> commandBuffer = queue ? [queue commandBuffer] : nil;
		id<MTLBlitCommandEncoder> blit = commandBuffer ? [commandBuffer blitCommandEncoder] : nil;
		if (!commandBuffer || !blit) return false;
		for (NSUInteger layer = 0; layer < layers; ++layer) {
			[blit copyFromTexture:texture sourceSlice:layer sourceLevel:(NSUInteger)level
				sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
				toBuffer:staging destinationOffset:layer * stagingImageBytes
				destinationBytesPerRow:stagingRowBytes destinationBytesPerImage:0 options:blitOptions];
		}
		[blit endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
			GL4METAL_LOG("glGetTexImage: private array blit readback failed: %s",
				commandBuffer.error.localizedDescription.UTF8String ?: "unknown Metal error");
			return false;
		}
	}
	const uint8_t *source = (const uint8_t *)staging.contents;
	for (NSUInteger layer = 0; layer < layers; ++layer) {
		for (NSUInteger row = 0; row < height; ++row) {
			memcpy(out.data() + layer * imageBytes + row * rowBytes,
				source + layer * stagingImageBytes + row * stagingRowBytes, rowBytes);
		}
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
		case MTLPixelFormatA1BGR5Unorm: {
			uint16_t packed = 0; memcpy(&packed, src, sizeof(packed));
			rgba[0] = (float)(packed & 0x1fu) / 31.0f;
			rgba[1] = (float)((packed >> 5) & 0x1fu) / 31.0f;
			rgba[2] = (float)((packed >> 10) & 0x1fu) / 31.0f;
			rgba[3] = (packed & 0x8000u) ? 1.0f : 0.0f;
			return true;
		}
		case MTLPixelFormatRG8Unorm: rgba[0] = unorm8(src[0]); rgba[1] = unorm8(src[1]); return true;
		case MTLPixelFormatR8Snorm: rgba[0] = std::max(-1.0f, (float)*(const int8_t *)src / 127.0f); return true;
		case MTLPixelFormatRG8Snorm: {
			const int8_t *values = (const int8_t *)src;
			rgba[0] = std::max(-1.0f, (float)values[0] / 127.0f);
			rgba[1] = std::max(-1.0f, (float)values[1] / 127.0f);
			return true;
		}
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
			for (int i = 0; i < 4; ++i) rgba[i] = unorm8(src[i]);
			return true;
		case MTLPixelFormatRGBA8Snorm: {
			const int8_t *values = (const int8_t *)src;
			for (int i = 0; i < 4; ++i) rgba[i] = std::max(-1.0f, (float)values[i] / 127.0f);
			return true;
		}
		case MTLPixelFormatBGRA8Unorm:
			rgba[0] = unorm8(src[2]); rgba[1] = unorm8(src[1]); rgba[2] = unorm8(src[0]); rgba[3] = unorm8(src[3]);
			return true;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatDepth16Unorm: {
			uint16_t v; memcpy(&v, src, sizeof(v)); rgba[0] = unorm16(v); return true;
		}
		case MTLPixelFormatR16Snorm: {
			int16_t v; memcpy(&v, src, sizeof(v)); rgba[0] = std::max(-1.0f, (float)v / 32767.0f); return true;
		}
		case MTLPixelFormatRG16Unorm: {
			uint16_t v[2]; memcpy(v, src, sizeof(v)); rgba[0] = unorm16(v[0]); rgba[1] = unorm16(v[1]); return true;
		}
		case MTLPixelFormatRG16Snorm: {
			int16_t v[2]; memcpy(v, src, sizeof(v));
			rgba[0] = std::max(-1.0f, (float)v[0] / 32767.0f);
			rgba[1] = std::max(-1.0f, (float)v[1] / 32767.0f);
			return true;
		}
		case MTLPixelFormatRGBA16Unorm: {
			uint16_t v[4]; memcpy(v, src, sizeof(v)); for (int i = 0; i < 4; ++i) rgba[i] = unorm16(v[i]); return true;
		}
		case MTLPixelFormatRGBA16Snorm: {
			int16_t v[4]; memcpy(v, src, sizeof(v));
			for (int i = 0; i < 4; ++i) rgba[i] = std::max(-1.0f, (float)v[i] / 32767.0f);
			return true;
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
		case GL_UNSIGNED_BYTE_3_3_2:
		case GL_UNSIGNED_BYTE_2_3_3_REV: return components == 3 ? 1 : 0;
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
		// Packed pixel types must never fall back to scalar-component parsing when
		// their external format has the wrong component count.
		case GL_UNSIGNED_BYTE_3_3_2:
		case GL_UNSIGNED_BYTE_2_3_3_REV:
		case GL_UNSIGNED_SHORT_5_6_5:
		case GL_UNSIGNED_SHORT_5_6_5_REV:
		case GL_UNSIGNED_SHORT_4_4_4_4:
		case GL_UNSIGNED_SHORT_4_4_4_4_REV:
		case GL_UNSIGNED_SHORT_5_5_5_1:
		case GL_UNSIGNED_SHORT_1_5_5_5_REV:
		case GL_UNSIGNED_INT_8_8_8_8:
		case GL_UNSIGNED_INT_8_8_8_8_REV:
		case GL_UNSIGNED_INT_10_10_10_2:
		case GL_UNSIGNED_INT_2_10_10_10_REV:
		case GL_UNSIGNED_INT_10F_11F_11F_REV:
		case GL_UNSIGNED_INT_5_9_9_9_REV:
			return 0;
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

static bool checkedImageSize3D(size_t width, size_t height, size_t depth, size_t pixelBytes,
							 size_t rowLength, size_t imageHeight, GLint rowAlignment,
							 GLint skipPixels, GLint skipRows, GLint skipImages,
							 size_t &rowStride, size_t &imageStride, size_t &startOffset,
							 size_t &requiredBytes) {
	if (pixelBytes == 0 || rowAlignment <= 0 || skipPixels < 0 || skipRows < 0 || skipImages < 0) return false;
	if (rowLength == 0) rowLength = width;
	if (imageHeight == 0) imageHeight = height;
	if (rowLength > SIZE_MAX / pixelBytes) return false;
	const size_t rawRowBytes = rowLength * pixelBytes;
	const size_t alignment = (size_t)rowAlignment;
	if (rawRowBytes > SIZE_MAX - (alignment - 1)) return false;
	rowStride = ((rawRowBytes + alignment - 1) / alignment) * alignment;
	if (rowStride == 0 || imageHeight > SIZE_MAX / rowStride) return false;
	imageStride = rowStride * imageHeight;
	if (imageStride == 0) return false;
	if ((size_t)skipImages > SIZE_MAX / imageStride ||
		(size_t)skipRows > SIZE_MAX / rowStride || (size_t)skipPixels > SIZE_MAX / pixelBytes) return false;
	const size_t skipImageBytes = (size_t)skipImages * imageStride;
	const size_t skipRowBytes = (size_t)skipRows * rowStride;
	const size_t skipPixelBytes = (size_t)skipPixels * pixelBytes;
	if (skipImageBytes > SIZE_MAX - skipRowBytes ||
		skipImageBytes + skipRowBytes > SIZE_MAX - skipPixelBytes) return false;
	startOffset = skipImageBytes + skipRowBytes + skipPixelBytes;
	if (width == 0 || height == 0 || depth == 0) { requiredBytes = startOffset; return true; }
	if (depth - 1 > (SIZE_MAX - startOffset) / imageStride) return false;
	const size_t lastImage = startOffset + (depth - 1) * imageStride;
	if (height - 1 > (SIZE_MAX - lastImage) / rowStride) return false;
	const size_t lastRow = lastImage + (height - 1) * rowStride;
	if (width > (SIZE_MAX - lastRow) / pixelBytes) return false;
	requiredBytes = lastRow + width * pixelBytes;
	return true;
}

// Decode one external GL pixel used by glClearTexImage/glClearTexSubImage. These
// entry points do not use GL_UNPACK_ROW_LENGTH/SKIP state: `data` addresses one
// clear value, or a byte offset in GL_PIXEL_UNPACK_BUFFER when one is bound.
static bool decodeClearScalar(const uint8_t *src, GLenum type, float &out) {
	switch (type) {
		case GL_BYTE: { int8_t v; memcpy(&v, src, sizeof(v)); out = std::max(-1.0f, (float)v / 127.0f); return true; }
		case GL_UNSIGNED_BYTE: { uint8_t v; memcpy(&v, src, sizeof(v)); out = (float)v / 255.0f; return true; }
		case GL_SHORT: { int16_t v; memcpy(&v, src, sizeof(v)); out = std::max(-1.0f, (float)v / 32767.0f); return true; }
		case GL_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, src, sizeof(v)); out = (float)v / 65535.0f; return true; }
		case GL_INT: { int32_t v; memcpy(&v, src, sizeof(v)); out = (float)std::max(-1.0, (double)v / 2147483647.0); return true; }
		case GL_UNSIGNED_INT: { uint32_t v; memcpy(&v, src, sizeof(v)); out = (float)((double)v / 4294967295.0); return true; }
		case GL_HALF_FLOAT: { uint16_t v; memcpy(&v, src, sizeof(v)); out = halfToFloat(v); return true; }
		case GL_FLOAT: memcpy(&out, src, sizeof(out)); return true;
		default: return false;
	}
}

static bool decodeClearPacked(const uint8_t *src, GLenum type, float values[4]) {
	values[0] = values[1] = values[2] = 0.0f;
	values[3] = 1.0f;
	if (type == GL_UNSIGNED_BYTE_3_3_2 || type == GL_UNSIGNED_BYTE_2_3_3_REV) {
		const uint8_t v = *src;
		if (type == GL_UNSIGNED_BYTE_3_3_2) {
			values[0] = (float)((v >> 5) & 7u) / 7.0f;
			values[1] = (float)((v >> 2) & 7u) / 7.0f;
			values[2] = (float)(v & 3u) / 3.0f;
		} else {
			values[0] = (float)(v & 7u) / 7.0f;
			values[1] = (float)((v >> 3) & 7u) / 7.0f;
			values[2] = (float)((v >> 6) & 3u) / 3.0f;
		}
		return true;
	}
	if (type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV) {
		uint32_t v; memcpy(&v, src, sizeof(v));
		for (unsigned c = 0; c < 4; ++c) {
			const unsigned shift = type == GL_UNSIGNED_INT_8_8_8_8_REV ? c * 8 : (3 - c) * 8;
			values[c] = (float)((v >> shift) & 0xffu) / 255.0f;
		}
		return true;
	}
	if (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV) {
		uint16_t v; memcpy(&v, src, sizeof(v));
		const bool rev = type == GL_UNSIGNED_SHORT_5_6_5_REV;
		const unsigned r = rev ? (v & 31u) : ((v >> 11) & 31u);
		const unsigned g = (v >> 5) & 63u;
		const unsigned b = rev ? ((v >> 11) & 31u) : (v & 31u);
		values[0] = (float)r / 31.0f; values[1] = (float)g / 63.0f; values[2] = (float)b / 31.0f;
		return true;
	}
	if (type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
		uint16_t v; memcpy(&v, src, sizeof(v));
		for (unsigned c = 0; c < 4; ++c) {
			const unsigned shift = type == GL_UNSIGNED_SHORT_4_4_4_4_REV ? c * 4 : (3 - c) * 4;
			values[c] = (float)((v >> shift) & 15u) / 15.0f;
		}
		return true;
	}
	if (type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
		uint16_t v; memcpy(&v, src, sizeof(v));
		if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
			values[0] = (float)(v & 31u) / 31.0f;
			values[1] = (float)((v >> 5) & 31u) / 31.0f;
			values[2] = (float)((v >> 10) & 31u) / 31.0f;
			values[3] = (float)((v >> 15) & 1u);
		} else {
			values[0] = (float)((v >> 11) & 31u) / 31.0f;
			values[1] = (float)((v >> 6) & 31u) / 31.0f;
			values[2] = (float)((v >> 1) & 31u) / 31.0f;
			values[3] = (float)(v & 1u);
		}
		return true;
	}
	if (type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV) {
		uint32_t v; memcpy(&v, src, sizeof(v));
		if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
			values[0] = (float)(v & 1023u) / 1023.0f;
			values[1] = (float)((v >> 10) & 1023u) / 1023.0f;
			values[2] = (float)((v >> 20) & 1023u) / 1023.0f;
			values[3] = (float)(v >> 30) / 3.0f;
		} else {
			values[0] = (float)((v >> 22) & 1023u) / 1023.0f;
			values[1] = (float)((v >> 12) & 1023u) / 1023.0f;
			values[2] = (float)((v >> 2) & 1023u) / 1023.0f;
			values[3] = (float)(v & 3u) / 3.0f;
		}
		return true;
	}
	if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
		uint32_t v; memcpy(&v, src, sizeof(v));
		values[0] = decodeUnsignedFloat(v & 0x7ffu, 6);
		values[1] = decodeUnsignedFloat((v >> 11) & 0x7ffu, 6);
		values[2] = decodeUnsignedFloat((v >> 22) & 0x3ffu, 5);
		return true;
	}
	if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
		uint32_t v; memcpy(&v, src, sizeof(v));
		const int exponent = (int)((v >> 27) & 31u) - 24;
		const float scale = std::ldexp(1.0f, exponent);
		values[0] = (float)(v & 0x1ffu) * scale;
		values[1] = (float)((v >> 9) & 0x1ffu) * scale;
		values[2] = (float)((v >> 18) & 0x1ffu) * scale;
		return true;
	}
	return false;
}

static bool encodeClearTexel(MTLPixelFormat format, GLenum internalformat, GLenum externalFormat,
		GLenum type, const void *data, uint8_t out[16], size_t &bytesPerPixel, size_t &sourceBytes) {
	int sourceComponents = 0, channelMap[4];
	if (!sourceLayout(externalFormat, sourceComponents, channelMap)) return false;
	const size_t packedBytes = packedTypeSize(type, sourceComponents);
	const size_t scalarBytes = scalarTypeSize(type);
	const bool packed = packedBytes != 0;
	if ((!packed && scalarBytes == 0) || (packed && packedBytes == 0)) return false;
	sourceBytes = packed ? packedBytes : (size_t)sourceComponents * scalarBytes;

	const bool depthFormat = format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float;
	const bool stencilFormat = format == MTLPixelFormatStencil8;
	if ((depthFormat && externalFormat != GL_DEPTH_COMPONENT) ||
		(stencilFormat && externalFormat != GL_STENCIL_INDEX) ||
		(!depthFormat && !stencilFormat && (externalFormat == GL_DEPTH_COMPONENT || externalFormat == GL_STENCIL_INDEX)))
		return false;

	float rgba[4] = {0.0f, 0.0f, 0.0f, data ? 1.0f : 0.0f};
	float source[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	if (data) {
		const uint8_t *bytes = (const uint8_t *)data;
		if (packed) {
			if (!decodeClearPacked(bytes, type, source)) return false;
		} else {
			for (int c = 0; c < sourceComponents; ++c) {
				if (!decodeClearScalar(bytes + (size_t)c * scalarBytes, type, source[c])) return false;
			}
		}
		for (int c = 0; c < 4; ++c) {
			if (channelMap[c] >= 0) rgba[c] = source[channelMap[c]];
		}
	}

	const bool logicalRgb = internalformat == GL_R3_G3_B2 || internalformat == GL_RGB || internalformat == GL_RGB8 ||
		internalformat == GL_RGB8_SNORM || internalformat == GL_SRGB8 ||
		internalformat == GL_RGB16 || internalformat == GL_RGB16_SNORM ||
		internalformat == GL_RGB16F || internalformat == GL_RGB32F;
	if (logicalRgb) rgba[3] = 1.0f;
	// The fallback backing texture is RGBA8, but GL_RGB5_A1 semantics still quantize
	// each clear to 5:5:5:1 before expanding it into the 8-bit Metal storage.
	if (internalformat == GL_RGB5_A1 && format != MTLPixelFormatA1BGR5Unorm) {
		const uint16_t packed = packA1BGR5(rgba);
		rgba[0] = (float)(packed & 0x1fu) / 31.0f;
		rgba[1] = (float)((packed >> 5) & 0x1fu) / 31.0f;
		rgba[2] = (float)((packed >> 10) & 0x1fu) / 31.0f;
		rgba[3] = (packed & 0x8000u) ? 1.0f : 0.0f;
	}
	memset(out, 0, 16);
	if (format == MTLPixelFormatDepth16Unorm) {
		bytesPerPixel = 2;
		writeScalarComponent(out, GL_UNSIGNED_SHORT, rgba[0]);
		return true;
	}
	if (format == MTLPixelFormatDepth32Float) {
		bytesPerPixel = 4;
		writeScalarComponent(out, GL_FLOAT, rgba[0]);
		return true;
	}
	if (format == MTLPixelFormatStencil8) {
		bytesPerPixel = 1;
		if (data) {
			// Stencil is an integer index, not a normalized color component.
			const uint8_t *bytes = (const uint8_t *)data;
			switch (type) {
				case GL_UNSIGNED_BYTE: out[0] = *bytes; break;
				case GL_BYTE: out[0] = (uint8_t)*bytes; break;
				case GL_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, bytes, 2); out[0] = (uint8_t)v; break; }
				case GL_UNSIGNED_INT: { uint32_t v; memcpy(&v, bytes, 4); out[0] = (uint8_t)v; break; }
				default: return false;
			}
		}
		return true;
	}
	if (format == MTLPixelFormatDepth32Float_Stencil8) return false; // combined aspect clear requires two aspect-specific blits.

	if (format == MTLPixelFormatA1BGR5Unorm) {
		bytesPerPixel = 2;
		const uint16_t packed = packA1BGR5(rgba);
		memcpy(out, &packed, sizeof(packed));
		return true;
	}
	int destinationComponents = 0, componentBytes = 0;
	if (destLayout(format, destinationComponents, componentBytes)) {
		bytesPerPixel = (size_t)destinationComponents * (size_t)componentBytes;
		GLenum destinationType = GL_UNSIGNED_BYTE;
		switch (format) {
			case MTLPixelFormatR8Snorm: case MTLPixelFormatRG8Snorm: case MTLPixelFormatRGBA8Snorm: destinationType = GL_BYTE; break;
			case MTLPixelFormatR16Unorm: case MTLPixelFormatRG16Unorm: case MTLPixelFormatRGBA16Unorm: destinationType = GL_UNSIGNED_SHORT; break;
			case MTLPixelFormatR16Snorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRGBA16Snorm: destinationType = GL_SHORT; break;
			case MTLPixelFormatR16Float: case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA16Float: destinationType = GL_HALF_FLOAT; break;
			case MTLPixelFormatR32Float: case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float: destinationType = GL_FLOAT; break;
			default: destinationType = GL_UNSIGNED_BYTE; break;
		}
		for (int c = 0; c < destinationComponents; ++c)
			writeScalarComponent(out + (size_t)c * (size_t)componentBytes, destinationType, rgba[c]);
		return true;
	}
	if (format == MTLPixelFormatRGB10A2Unorm) {
		bytesPerPixel = 4; writePackedPixel(out, GL_UNSIGNED_INT_2_10_10_10_REV, rgba); return true;
	}
	if (format == MTLPixelFormatRG11B10Float) {
		bytesPerPixel = 4; writePackedPixel(out, GL_UNSIGNED_INT_10F_11F_11F_REV, rgba); return true;
	}
	if (format == MTLPixelFormatRGB9E5Float) {
		bytesPerPixel = 4; writePackedPixel(out, GL_UNSIGNED_INT_5_9_9_9_REV, rgba); return true;
	}
	return false;
}

static bool clearTextureRegion(TextureObject &textureObject, GLint level,
		GLint x, GLint y, GLint z, GLsizei width, GLsizei height, GLsizei depth,
		GLenum format, GLenum type, const void *data, const char *caller) {
	id<MTLTexture> texture = textureObject.texture;
	if (textureObject.buffer_object != 0 || textureObject.internalformat == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: buffer textures or undefined internal formats cannot be cleared", caller);
		return false;
	}
	if (!texture || level < 0 || (NSUInteger)level >= texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: texture has no allocated mip level %d", caller, level);
		return false;
	}
	if (width < 0 || height < 0 || depth < 0 || x < 0 || y < 0 || z < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	const GLint levelWidth = (GLint)std::max<NSUInteger>(1, texture.width >> level);
	GLint levelHeight = (GLint)std::max<NSUInteger>(1, texture.height >> level);
	GLint levelDepth = 1;
	const bool is1D = texture.textureType == MTLTextureType1D;
	const bool is3D = texture.textureType == MTLTextureType3D;
	const bool isArray = texture.textureType == MTLTextureType2DArray;
	const bool isCube = texture.textureType == MTLTextureTypeCube;
	if (is1D) levelHeight = 1;
	if (is3D) levelDepth = (GLint)std::max<NSUInteger>(1, texture.depth >> level);
	else if (isArray) levelDepth = (GLint)texture.arrayLength;
	else if (isCube) levelDepth = 6;
	else if (!is1D && texture.textureType != MTLTextureType2D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: texture type %lu is not supported by the clear path", caller, (unsigned long)texture.textureType);
		return false;
	}
	if (x > levelWidth || y > levelHeight || z > levelDepth ||
		width > levelWidth - x || height > levelHeight - y || depth > levelDepth - z) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: region (%d,%d,%d %dx%dx%d) exceeds mip level (%dx%dx%d)", caller,
			x, y, z, width, height, depth, levelWidth, levelHeight, levelDepth);
		return false;
	}
	if (width == 0 || height == 0 || depth == 0) return true;
	if (!ctx.device || !ctx.commandQueue) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: Metal device/queue is unavailable", caller);
		return false;
	}

	uint8_t clearPixel[16] = {};
	size_t bytesPerPixel = 0, sourceBytes = 0;
	const GLuint unpackBuffer = boundBuffer(GL_PIXEL_UNPACK_BUFFER);
	// A PBO pointer is an integer byte offset, never a CPU address. Validate the
	// format first with a zero texel, then decode the PBO/client value below.
	const void *initialValue = (unpackBuffer == 0) ? data : nullptr;
	if (!encodeClearTexel(texture.pixelFormat, textureObject.internalformat, format, type,
		initialValue, clearPixel, bytesPerPixel, sourceBytes)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("%s: unsupported clear value format=0x%X type=0x%X for MTLPixelFormat %d",
			caller, format, type, (int)texture.pixelFormat);
		return false;
	}

	const bool hasSourceValue = data != nullptr || unpackBuffer != 0;
	const uint8_t *sourceValue = nullptr;
	uint8_t convertedPixel[16];
	if (hasSourceValue) {
		sourceValue = resolveUnpackPointer(data, 0, sourceBytes, caller);
		if (!sourceValue) return false;
		if (!encodeClearTexel(texture.pixelFormat, textureObject.internalformat, format, type,
			sourceValue, convertedPixel, bytesPerPixel, sourceBytes)) {
			gl4metalSetError(GL_INVALID_ENUM);
			return false;
		}
		memcpy(clearPixel, convertedPixel, bytesPerPixel);
	}

	size_t rowBytes = 0;
	if (!checkedSizeMul((size_t)width, bytesPerPixel, rowBytes) || rowBytes == 0 ||
		rowBytes > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return false;
	}
	std::vector<uint8_t> row;
	try {
		row.resize(rowBytes);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return false;
	}
	for (GLsizei column = 0; column < width; ++column)
		memcpy(row.data() + (size_t)column * bytesPerPixel, clearPixel, bytesPerPixel);

	if (is3D || isArray) {
		// Chunk by z so even a maximum-size volume doesn't require one staging buffer
		// larger than Metal's per-buffer limit. Each chunk remains ordered on the GL queue.
		size_t alignedRow = 0, imageBytes = 0;
		if (!checkedAlignSize(rowBytes, 256, alignedRow) ||
			!checkedSizeMul(alignedRow, (size_t)height, imageBytes) || imageBytes == 0) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			return false;
		}
		const size_t maxDepthPerUpload = std::max<size_t>(1, (size_t)ctx.device.maxBufferLength / imageBytes);
		for (GLsizei done = 0; done < depth;) {
			const GLsizei chunkDepth = (GLsizei)std::min<size_t>((size_t)(depth - done), maxDepthPerUpload);
			if (!enqueueTextureUpload3D(texture, level, x, y, z + done, width, height, chunkDepth,
				row.data(), 0, 0, rowBytes)) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: failed to stage %s clear at slice/depth=%d count=%d",
					caller, isArray ? "array" : "3D", z + done, chunkDepth);
				return false;
			}
			done += chunkDepth;
		}
		return true;
	}

	for (GLsizei slice = 0; slice < depth; ++slice) {
		const NSUInteger destinationSlice = isCube ? (NSUInteger)(z + slice) : 0;
		if (!enqueueTextureUpload(texture, level, destinationSlice, x, y, width, height,
			row.data(), 0, rowBytes)) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("%s: failed to stage texture clear for slice %lu", caller, (unsigned long)destinationSlice);
			return false;
		}
	}
	return true;
}

} // namespace



void gl4metalClearTexImageImpl(GLuint texture, GLint level, GLenum format, GLenum type, const void *data) {
	auto it = ctx.texture_objects.find(texture);
	if (texture == 0 || it == ctx.texture_objects.end() || it->second.target == 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glClearTexImage: invalid texture name %u", texture);
		return;
	}
	TextureObject &object = it->second;
	if (!object.texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glClearTexImage: texture %u has no allocated image", texture);
		return;
	}
	if (level < 0 || (NSUInteger)level >= object.texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const bool is3D = object.texture.textureType == MTLTextureType3D;
	const bool isArray = object.texture.textureType == MTLTextureType2DArray;
	const bool isCube = object.texture.textureType == MTLTextureTypeCube;
	const GLsizei width = (GLsizei)std::max<NSUInteger>(1, object.texture.width >> level);
	const GLsizei height = (GLsizei)std::max<NSUInteger>(1, object.texture.height >> level);
	const GLsizei depth = is3D ? (GLsizei)std::max<NSUInteger>(1, object.texture.depth >> level)
		: (isArray ? (GLsizei)object.texture.arrayLength : (isCube ? 6 : 1));
	clearTextureRegion(object, level, 0, 0, 0, width, height, depth, format, type, data, "glClearTexImage");
}

void gl4metalClearTexSubImageImpl(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
		GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data) {
	if (level < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.texture_objects.find(texture);
	if (texture == 0 || it == ctx.texture_objects.end() || it->second.target == 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glClearTexSubImage: invalid texture name %u", texture);
		return;
	}
	clearTextureRegion(it->second, level, xoffset, yoffset, zoffset,
		width, height, depth, format, type, data, "glClearTexSubImage");
}


void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void *pixels) {
	const int face = cubeFace(target);
	const bool is1D = target == GL_TEXTURE_1D;
	const bool is2D = target == GL_TEXTURE_2D;
	const bool is3D = target == GL_TEXTURE_3D;
	const bool isArray = target == GL_TEXTURE_2D_ARRAY;
	if (!is1D && !is2D && !is3D && !isArray && face < 0) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glGetTexImage: unsupported target 0x%X", target);
		return;
	}
	TextureObject *tex = boundTexture(is1D || is2D || is3D || isArray ? target : GL_TEXTURE_CUBE_MAP);
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
	if (is1D && tex->texture.textureType != MTLTextureType1D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: GL_TEXTURE_1D is not backed by a 1D Metal texture");
		return;
	}
	if (is3D && tex->texture.textureType != MTLTextureType3D) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: GL_TEXTURE_3D is not backed by a 3D Metal texture");
		return;
	}
	if (isArray && tex->texture.textureType != MTLTextureType2DArray) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: GL_TEXTURE_2D_ARRAY is not backed by a 2D-array Metal texture");
		return;
	}
	const NSUInteger width = std::max<NSUInteger>(1, (NSUInteger)tex->width >> level);
	const NSUInteger height = is1D ? 1 : std::max<NSUInteger>(1, (NSUInteger)tex->height >> level);
	const NSUInteger depth = is3D ? std::max<NSUInteger>(1, (NSUInteger)tex->depth >> level)
		: (isArray ? tex->texture.arrayLength : 1);
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

	size_t rowStride = 0, imageStride = 0, startOffset = 0, requiredBytes = 0;
	if (is3D || isArray) {
		if (!checkedImageSize3D((size_t)width, (size_t)height, (size_t)depth, outputPixelBytes,
			(size_t)ctx.pack_row_length, (size_t)ctx.pack_image_height, ctx.pack_alignment,
			ctx.pack_skip_pixels, ctx.pack_skip_rows, ctx.pack_skip_images,
			rowStride, imageStride, startOffset, requiredBytes)) {
			gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetTexImage: 3D/array pixel-pack layout overflows addressable memory");
			return;
		}
	} else {
		if (!checkedImageSize((size_t)width, (size_t)height, outputPixelBytes,
			(size_t)ctx.pack_row_length, ctx.pack_alignment, ctx.pack_skip_pixels, ctx.pack_skip_rows,
			rowStride, startOffset, requiredBytes)) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glGetTexImage: pixel-pack layout overflows addressable memory");
			return;
		}
		imageStride = rowStride * (size_t)height;
	}
	if (bufferOffset > SIZE_MAX - requiredBytes) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (packBufferObject) {
		const NSUInteger logicalLength = glBufferLogicalSize(packBuffer);
		if (bufferOffset > logicalLength || requiredBytes > logicalLength - bufferOffset) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glGetTexImage: pixel-pack buffer range %lu+%lu exceeds %lu bytes",
				(unsigned long)bufferOffset, (unsigned long)requiredBytes, (unsigned long)logicalLength);
			return;
		}
	}
	destination += bufferOffset + startOffset;

	std::vector<uint8_t> source;
	const MTLBlitOption readbackOptions = mtlFormat == MTLPixelFormatDepth32Float_Stencil8
		? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone;
	const bool readSucceeded = is3D
		? readTextureLevel3D(tex->texture, level, width, height, depth, sourceBytesPerPixel, readbackOptions, source)
		: (isArray
			? readTextureLevelArray(tex->texture, level, width, height, depth, sourceBytesPerPixel, readbackOptions, source)
			: readTextureLevel(tex->texture, level, face >= 0 ? (NSUInteger)face : 0,
				width, height, sourceBytesPerPixel, readbackOptions, source));
	if (!readSucceeded) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetTexImage: failed to read MTLPixelFormat %d at level %d", (int)mtlFormat, level);
		return;
	}
	for (NSUInteger image = 0; image < depth; ++image) {
		for (NSUInteger y = 0; y < height; ++y) {
			const uint8_t *srcRow = source.data() + image * width * height * sourceBytesPerPixel + y * width * sourceBytesPerPixel;
			uint8_t *dstRow = destination + image * imageStride + y * rowStride;
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
}

void glGetCompressedTexImage(GLenum target, GLint level, void *img) {
	const int face = cubeFace(target);
	const bool isArray = target == GL_TEXTURE_2D_ARRAY;
	if (target != GL_TEXTURE_2D && !isArray && face < 0) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	TextureObject *textureObject = boundTexture(face >= 0 ? GL_TEXTURE_CUBE_MAP : target);
	if (!textureObject || !textureObject->texture) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	CompressedFormatInfo info;
	if (!compressedFormatInfo(textureObject->internalformat, info) || textureObject->texture.pixelFormat != info.pixelFormat) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetCompressedTexImage: bound texture does not contain a supported compressed image");
		return;
	}
	if (level < 0 || (NSUInteger)level >= textureObject->texture.mipmapLevelCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	const NSUInteger width = std::max<NSUInteger>(1, (NSUInteger)textureObject->width >> level);
	const NSUInteger height = std::max<NSUInteger>(1, (NSUInteger)textureObject->height >> level);
	NSUInteger rowBytes = 0, blockRows = 0;
	size_t imageBytes = 0;
	if (!compressedImageLayout((GLsizei)width, (GLsizei)height, info, rowBytes, blockRows, imageBytes)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const NSUInteger layerCount = isArray ? textureObject->texture.arrayLength : 1;
	if (imageBytes > SIZE_MAX / layerCount) { gl4metalSetError(GL_INVALID_VALUE); return; }
	const size_t totalBytes = imageBytes * layerCount;
	const GLuint packBuffer = boundBuffer(GL_PIXEL_PACK_BUFFER);
	uint8_t *destination = (uint8_t *)img;
	NSUInteger bufferOffset = 0;
	id<MTLBuffer> packBufferObject = nil;
	if (packBuffer != 0) {
		auto bufferIt = ctx.buffer_objects.find(packBuffer);
		if (bufferIt == ctx.buffer_objects.end() || !bufferIt->second || !bufferIt->second.contents) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		packBufferObject = bufferIt->second;
		bufferOffset = (NSUInteger)(uintptr_t)img;
		const NSUInteger logicalLength = glBufferLogicalSize(packBuffer);
		if (bufferOffset > logicalLength || totalBytes > logicalLength - bufferOffset) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glGetCompressedTexImage: pixel-pack buffer range exceeds buffer storage");
			return;
		}
		destination = (uint8_t *)packBufferObject.contents + bufferOffset;
	} else if (!destination) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const NSUInteger firstSlice = face >= 0 ? (NSUInteger)face : 0;
	if (!readbackCompressedSlices(textureObject->texture, level, firstSlice, layerCount,
		(GLsizei)width, (GLsizei)height, info, destination, rowBytes)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetCompressedTexImage: Metal blit readback failed");
		return;
	}
}

void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) { gl4metalSetError(GL_INVALID_ENUM); return; }
	TextureObject *tex = boundTexture(target);
	if (!tex) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (pname == GL_TEXTURE_MIN_LOD)  { *params = tex->min_lod; return; }
	if (pname == GL_TEXTURE_MAX_LOD)  { *params = tex->max_lod; return; }
	if (pname == GL_TEXTURE_LOD_BIAS) { *params = tex->lod_bias; return; }
	if (tex && pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) { *params = tex->max_anisotropy; return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) { for (int component = 0; component < 4; ++component) params[component] = tex->border_color[component]; return; }
	GLint swizzleValues[4] = {0, 0, 0, 0};
	if (pname == GL_TEXTURE_SWIZZLE_RGBA || textureSwizzleComponentIndex(pname) >= 0) {
		glGetTexParameteriv(target, pname, swizzleValues);
		if (pname == GL_TEXTURE_SWIZZLE_RGBA) { for (int component = 0; component < 4; ++component) params[component] = (GLfloat)swizzleValues[component]; }
		else params[0] = (GLfloat)swizzleValues[0];
		return;
	}
	GLint value = 0;
	glGetTexParameteriv(target, pname, &value);
	*params = (GLfloat)value;
}

void glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); *params = 0; return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
		TextureObject *multisampleTexture = boundTexture(target);
		if (!multisampleTexture) {
			gl4metalSetError(GL_INVALID_OPERATION);
			*params = 0;
			return;
		}
		if (pname == GL_TEXTURE_IMMUTABLE_FORMAT) {
			*params = multisampleTexture && multisampleTexture->immutable_storage ? GL_TRUE : GL_FALSE;
			return;
		}
		if (pname == GL_TEXTURE_IMMUTABLE_LEVELS) {
			*params = multisampleTexture && multisampleTexture->immutable_storage ? 1 : 0;
			return;
		}
		gl4metalSetError(GL_INVALID_ENUM);
		*params = 0;
		return;
	}
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
		case GL_TEXTURE_IMMUTABLE_FORMAT: *params = tex->immutable_storage ? GL_TRUE : GL_FALSE; break;
		case GL_TEXTURE_IMMUTABLE_LEVELS: *params = tex->immutable_storage && tex->texture ? (GLint)tex->texture.mipmapLevelCount : 0; break;
		case GL_TEXTURE_MIN_LOD:      *params = textureParameterQueryInt(tex->min_lod); break;
		case GL_TEXTURE_MAX_LOD:      *params = textureParameterQueryInt(tex->max_lod); break;
		case GL_TEXTURE_LOD_BIAS:     *params = textureParameterQueryInt(tex->lod_bias); break;
		case GL_TEXTURE_MAX_ANISOTROPY_EXT: *params = textureParameterQueryInt(tex->max_anisotropy); break;
		case GL_TEXTURE_BORDER_COLOR:
			for (int component = 0; component < 4; ++component) params[component] = (GLint)tex->border_color[component];
			break;
		case GL_TEXTURE_SWIZZLE_RGBA:
			for (int component = 0; component < 4; ++component) params[component] = (GLint)tex->swizzle[component];
			break;
		case GL_TEXTURE_SWIZZLE_R: *params = (GLint)tex->swizzle[0]; break;
		case GL_TEXTURE_SWIZZLE_G: *params = (GLint)tex->swizzle[1]; break;
		case GL_TEXTURE_SWIZZLE_B: *params = (GLint)tex->swizzle[2]; break;
		case GL_TEXTURE_SWIZZLE_A: *params = (GLint)tex->swizzle[3]; break;
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
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!validTextureParameterTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) { gl4metalSetError(GL_INVALID_ENUM); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		TextureObject *texture = boundTexture(target);
		if (!texture) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		for (int component = 0; component < 4; ++component) params[component] = (GLuint)std::max(texture->border_color[component], 0.0f);
		return;
	}
	if (pname == GL_TEXTURE_SWIZZLE_RGBA) {
		GLint values[4] = {0, 0, 0, 0};
		glGetTexParameteriv(target, pname, values);
		for (int component = 0; component < 4; ++component) params[component] = (GLuint)values[component];
		return;
	}
	GLint value = 0;
	glGetTexParameteriv(target, pname, &value);
	params[0] = (GLuint)value;
}

void glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	GLint value = 0;
	glGetTexLevelParameteriv(target, level, pname, &value);
	*params = (GLfloat)value;
}

void glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (target == GL_TEXTURE_BUFFER && level != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	GLsizei w = 0, h = 0;
	GLenum internalformat = 0;
	GLuint bufferObject = 0;

	GLsizei d = 0;
	GLint samples = 0;
	GLboolean fixedSampleLocations = GL_FALSE;
	if (target == GL_PROXY_TEXTURE_1D) {
		w = ctx.proxy_texture_1d.w;
		h = w > 0 ? 1 : 0;
		d = w > 0 ? 1 : 0;
		internalformat = ctx.proxy_texture_1d.internalformat;
	} else if (target == GL_PROXY_TEXTURE_2D) {
		w = ctx.proxy_texture_2d.w;
		h = ctx.proxy_texture_2d.h;
		d = 1;
		internalformat = ctx.proxy_texture_2d.internalformat;
	} else if (target == GL_PROXY_TEXTURE_2D_ARRAY) {
		w = ctx.proxy_texture_2d_array.w;
		h = ctx.proxy_texture_2d_array.h;
		d = ctx.proxy_texture_2d_array.layers;
		internalformat = ctx.proxy_texture_2d_array.internalformat;
	} else if (target == GL_PROXY_TEXTURE_3D) {
		w = ctx.proxy_texture_3d.w;
		h = ctx.proxy_texture_3d.h;
		d = ctx.proxy_texture_3d.d;
		internalformat = ctx.proxy_texture_3d.internalformat;
	} else {
		TextureObject *t = boundTexture(cubeFace(target) >= 0 ? GL_TEXTURE_CUBE_MAP : target);
		if (t && level >= 0 && t->target == GL_TEXTURE_BUFFER && target == GL_TEXTURE_BUFFER) {
			w = t->width;
			h = t->height;
			d = 1;
			internalformat = t->internalformat;
			bufferObject = t->buffer_object;
		} else if (t && t->texture && level >= 0 && (NSUInteger)level < t->texture.mipmapLevelCount) {
			w = std::max(1, t->width >> level);
			h = std::max(1, t->height >> level);
			if (t->texture.textureType == MTLTextureType3D)
				d = std::max(1, t->depth >> level);
			else if (t->texture.textureType == MTLTextureType2DArray)
				d = t->texture.arrayLength;
			else
				d = 1;
			if (t->texture.textureType == MTLTextureType2DMultisample) {
				samples = (GLint)t->texture.sampleCount;
				fixedSampleLocations = GL_TRUE;
			}
			internalformat = t->internalformat;
		}
	}

	switch (pname) {
		case GL_TEXTURE_WIDTH:           *params = w; break;
		case GL_TEXTURE_HEIGHT:          *params = h; break;
		case GL_TEXTURE_DEPTH:           *params = d; break;
		case GL_TEXTURE_INTERNAL_FORMAT: *params = (GLint)internalformat; break;
		case GL_TEXTURE_SAMPLES: *params = samples; break;
		case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS: *params = (GLint)fixedSampleLocations; break;
		case GL_TEXTURE_COMPRESSED: {
			CompressedFormatInfo info;
			*params = compressedFormatInfo(internalformat, info) ? GL_TRUE : GL_FALSE;
			break;
		}
		case GL_TEXTURE_COMPRESSED_IMAGE_SIZE: {
			CompressedFormatInfo info;
			if (!compressedFormatInfo(internalformat, info) || w <= 0 || h <= 0 || d <= 0) { *params = 0; break; }
			NSUInteger compressedRowBytes = 0, compressedRows = 0;
			size_t compressedImageBytes = 0;
			if (!compressedImageLayout(w, h, info, compressedRowBytes, compressedRows, compressedImageBytes) ||
				compressedImageBytes > SIZE_MAX / (size_t)d || compressedImageBytes * (size_t)d > (size_t)INT_MAX) {
				gl4metalSetError(GL_INVALID_OPERATION); *params = 0; break;
			}
			*params = (GLint)(compressedImageBytes * (size_t)d);
			break;
		}
		case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH: {
			CompressedFormatInfo info;
			if (!compressedFormatInfo(internalformat, info)) { gl4metalSetError(GL_INVALID_OPERATION); *params = 0; }
			else *params = (GLint)info.blockWidth;
			break;
		}
		case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT: {
			CompressedFormatInfo info;
			if (!compressedFormatInfo(internalformat, info)) { gl4metalSetError(GL_INVALID_OPERATION); *params = 0; }
			else *params = (GLint)info.blockHeight;
			break;
		}
		case GL_TEXTURE_COMPRESSED_BLOCK_SIZE: {
			CompressedFormatInfo info;
			if (!compressedFormatInfo(internalformat, info)) { gl4metalSetError(GL_INVALID_OPERATION); *params = 0; }
			else *params = (GLint)info.bytesPerBlock;
			break;
		}
		case GL_TEXTURE_BUFFER_DATA_STORE_BINDING:
			if (target != GL_TEXTURE_BUFFER) { gl4metalSetError(GL_INVALID_ENUM); *params = 0; break; }
			*params = (GLint)bufferObject;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetTexLevelParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}
