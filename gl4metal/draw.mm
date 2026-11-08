#import "draw.h"

#include <mutex>
#include <atomic>
#include <limits>
#include <new>

// These caches are process-wide / context-wide and may be reached by more than one
// JVM thread. std::unordered_map and std::unordered_set do not support concurrent
// mutation, even when the keys differ.
static std::mutex convertedVertexBufferCacheMutex;
static std::mutex samplerCacheMutex;
static std::mutex pipelineCacheMutex;
static std::mutex feedbackLoopSetMutex;
static std::mutex fallbackTextureMutex;
static std::mutex fallbackSamplerMutex;

static bool vertexFetchValidationEnabled() {
	// Always protect GPU vertex fetches by default. Set this to 0 only for a controlled
	// performance comparison after the driver has been validated on the target device.
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_VALIDATE_VERTEX_FETCH");
		return !(value && value[0] == '0');
	}();
	return enabled;
}

static bool vertexSamplerTraceEnabled() {
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_TRACE_VERTEX_SAMPLERS");
		return value && value[0] && value[0] != '0';
	}();
	return enabled;
}

static bool vertexFetchTraceEnabled() {
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_TRACE_VERTEX_FETCH");
		return value && value[0] && value[0] != '0';
	}();
	return enabled;
}

static bool fragmentSamplerTraceEnabled() {
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_TRACE_FRAGMENT_SAMPLERS");
		return value && value[0] && value[0] != '0';
	}();
	return enabled;
}

// MARK: - Helpers

static bool capOn(GLenum cap) {
	auto it = ctx.caps.find(cap);
	return it != ctx.caps.end() && it->second;
}

static MTLBlendFactor blendFactor(GLenum f) {
	switch (f) {
		case GL_ZERO:                     return MTLBlendFactorZero;
		case GL_ONE:                      return MTLBlendFactorOne;
		case GL_SRC_COLOR:                return MTLBlendFactorSourceColor;
		case GL_ONE_MINUS_SRC_COLOR:      return MTLBlendFactorOneMinusSourceColor;
		case GL_DST_COLOR:                return MTLBlendFactorDestinationColor;
		case GL_ONE_MINUS_DST_COLOR:      return MTLBlendFactorOneMinusDestinationColor;
		case GL_SRC_ALPHA:                return MTLBlendFactorSourceAlpha;
		case GL_ONE_MINUS_SRC_ALPHA:      return MTLBlendFactorOneMinusSourceAlpha;
		case GL_DST_ALPHA:                return MTLBlendFactorDestinationAlpha;
		case GL_ONE_MINUS_DST_ALPHA:      return MTLBlendFactorOneMinusDestinationAlpha;
		case GL_CONSTANT_COLOR:           return MTLBlendFactorBlendColor;
		case GL_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
		case GL_CONSTANT_ALPHA:           return MTLBlendFactorBlendAlpha;
		case GL_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
		case GL_SRC_ALPHA_SATURATE:       return MTLBlendFactorSourceAlphaSaturated;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("unsupported blend factor 0x%X, using GL_ONE", f);
			return MTLBlendFactorOne;
	}
}

static MTLBlendOperation blendOperation(GLenum e) {
	switch (e) {
		case GL_FUNC_ADD:              return MTLBlendOperationAdd;
		case GL_FUNC_SUBTRACT:         return MTLBlendOperationSubtract;
		case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
		case GL_MIN:                   return MTLBlendOperationMin;
		case GL_MAX:                   return MTLBlendOperationMax;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("unsupported blend equation 0x%X, using GL_FUNC_ADD", e);
			return MTLBlendOperationAdd;
	}
}

// Vertex formats by component count (index 1...4).
static const MTLVertexFormat kFloat[5]  = { MTLVertexFormatInvalid, MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4 };
static const MTLVertexFormat kHalf[5]   = { MTLVertexFormatInvalid, MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4 };
static const MTLVertexFormat kChar[5]   = { MTLVertexFormatInvalid, MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4 };
static const MTLVertexFormat kCharN[5]  = { MTLVertexFormatInvalid, MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized };
static const MTLVertexFormat kUChar[5]  = { MTLVertexFormatInvalid, MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4 };
static const MTLVertexFormat kUCharN[5] = { MTLVertexFormatInvalid, MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized };
static const MTLVertexFormat kShort[5]  = { MTLVertexFormatInvalid, MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4 };
static const MTLVertexFormat kShortN[5] = { MTLVertexFormatInvalid, MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized };
static const MTLVertexFormat kUShort[5] = { MTLVertexFormatInvalid, MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4 };
static const MTLVertexFormat kUShortN[5]= { MTLVertexFormatInvalid, MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized };
static const MTLVertexFormat kInt[5]    = { MTLVertexFormatInvalid, MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4 };
static const MTLVertexFormat kUInt[5]   = { MTLVertexFormatInvalid, MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4 };

static NSUInteger componentBytes(GLenum type) {
	switch (type) {
		case GL_BYTE:
		case GL_UNSIGNED_BYTE:  return 1;
		case GL_SHORT:
		case GL_UNSIGNED_SHORT:
		case GL_HALF_FLOAT:     return 2;
		default:                return 4;
	}
}

// `tight` receives the size of one element in bytes (the stride when GL's stride is 0).
static MTLVertexFormat vertexFormatFor(const VertexAttrib &a, NSUInteger &tight) {
	int n = (a.size >= 1 && a.size <= 4) ? a.size : 4; // GL_BGRA (0x80E1) is treated as 4 components
	bool norm = a.normalized;
	tight = (NSUInteger)n * componentBytes(a.type);
	switch (a.type) {
		case GL_FLOAT:          return kFloat[n];
		case GL_HALF_FLOAT:     return kHalf[n];
		case GL_BYTE:           return (norm ? kCharN : kChar)[n];
		case GL_UNSIGNED_BYTE:  return (norm ? kUCharN : kUChar)[n];
		case GL_SHORT:          return (norm ? kShortN : kShort)[n];
		case GL_UNSIGNED_SHORT: return (norm ? kUShortN : kUShort)[n];
		case GL_INT:            return kInt[n];
		case GL_UNSIGNED_INT:   return kUInt[n];
		case GL_INT_2_10_10_10_REV:
			tight = 4;
			return MTLVertexFormatInt1010102Normalized;
		case GL_UNSIGNED_INT_2_10_10_10_REV:
			tight = 4;
			return MTLVertexFormatUInt1010102Normalized;
		default:
			return MTLVertexFormatInvalid;
	}
}


// Metal can widen a short/ushort vertex format to a shader integer of the SAME signedness,
// but it rejects a signedness change (for example UShort3 -> int3). Preserve the GL source
// value by extending each component to 32 bits only for that mismatched case.
static bool integerVertexTypeInfo(GLenum type, bool &isUnsigned, NSUInteger &componentSize) {
	switch (type) {
		case GL_BYTE:           isUnsigned = false; componentSize = 1; return true;
		case GL_UNSIGNED_BYTE:  isUnsigned = true;  componentSize = 1; return true;
		case GL_SHORT:          isUnsigned = false; componentSize = 2; return true;
		case GL_UNSIGNED_SHORT: isUnsigned = true;  componentSize = 2; return true;
		case GL_INT:            isUnsigned = false; componentSize = 4; return true;
		case GL_UNSIGNED_INT:   isUnsigned = true;  componentSize = 4; return true;
		default: return false;
	}
}

static uint32_t loadExtendedIntegerComponent(const uint8_t *src, GLenum type) {
	switch (type) {
		case GL_BYTE: {
			int8_t value; memcpy(&value, src, sizeof(value));
			return (uint32_t)(int32_t)value;
		}
		case GL_UNSIGNED_BYTE: {
			uint8_t value; memcpy(&value, src, sizeof(value));
			return (uint32_t)value;
		}
		case GL_SHORT: {
			int16_t value; memcpy(&value, src, sizeof(value));
			return (uint32_t)(int32_t)value;
		}
		case GL_UNSIGNED_SHORT: {
			uint16_t value; memcpy(&value, src, sizeof(value));
			return (uint32_t)value;
		}
		case GL_INT: {
			int32_t value; memcpy(&value, src, sizeof(value));
			return (uint32_t)value;
		}
		case GL_UNSIGNED_INT: {
			uint32_t value; memcpy(&value, src, sizeof(value));
			return value;
		}
		default: return 0;
	}
}

static std::unordered_map<GLuint, std::unordered_map<std::string, id<MTLBuffer>>> convertedVertexBufferCache;

void invalidateConvertedVertexBuffers(GLuint buffer) {
	std::lock_guard<std::mutex> lock(convertedVertexBufferCacheMutex);
	convertedVertexBufferCache.erase(buffer);
}

static bool makeSignednessConvertedVertexBuffer(GLuint bufferId, const VertexAttrib &va, id<MTLBuffer> source, NSUInteger &convertedStride, __strong id<MTLBuffer> &converted) {
	if (!source || !source.contents || va.size < 1 || va.size > 4) return false;
	const NSUInteger sourceLength = source.length;
	bool sourceUnsigned = false;
	NSUInteger componentSize = 0;
	if (!integerVertexTypeInfo(va.type, sourceUnsigned, componentSize)) return false;
	(void)sourceUnsigned;
	const NSUInteger components = (NSUInteger)va.size;
	const NSUInteger sourceElementSize = components * componentSize;
	const NSUInteger sourceStride = va.stride > 0 ? (NSUInteger)va.stride : sourceElementSize;
	const NSUInteger sourceOffset = (NSUInteger)(uintptr_t)va.pointer;
	if (sourceStride < sourceElementSize || sourceOffset > source.length || source.length - sourceOffset < sourceElementSize) return false;
	convertedStride = components * sizeof(uint32_t);
	std::string cacheKey;
	cacheKey.append((const char *)&va.type, sizeof(va.type));
	cacheKey.append((const char *)&va.size, sizeof(va.size));
	cacheKey.append((const char *)&va.stride, sizeof(va.stride));
	cacheKey.append((const char *)&sourceOffset, sizeof(sourceOffset));
	cacheKey.append((const char *)&sourceLength, sizeof(sourceLength));
	std::lock_guard<std::mutex> lock(convertedVertexBufferCacheMutex);
	auto &perBuffer = convertedVertexBufferCache[bufferId];
	auto cached = perBuffer.find(cacheKey);
	if (cached != perBuffer.end() && cached->second) {
		converted = cached->second;
		return true;
	}
	const NSUInteger records = (source.length - sourceOffset - sourceElementSize) / sourceStride + 1;
	if (records > SIZE_MAX / convertedStride) return false;
	const NSUInteger convertedSize = records * convertedStride;
	if (convertedSize == 0 || convertedSize > ctx.device.maxBufferLength) return false;

	std::vector<uint8_t> bytes;
	try {
		bytes.resize((size_t)convertedSize);
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("vertex conversion: CPU staging allocation failed (%llu bytes)", (unsigned long long)convertedSize);
		return false;
	}
	const uint8_t *sourceBytes = (const uint8_t *)source.contents + sourceOffset;
	for (NSUInteger record = 0; record < records; ++record) {
		const uint8_t *srcRecord = sourceBytes + record * sourceStride;
		uint8_t *dstRecord = bytes.data() + record * convertedStride;
		for (NSUInteger component = 0; component < components; ++component) {
			const uint32_t value = loadExtendedIntegerComponent(srcRecord + component * componentSize, va.type);
			memcpy(dstRecord + component * sizeof(value), &value, sizeof(value));
		}
	}
	converted = [ctx.device newBufferWithBytes:bytes.data() length:convertedSize options:MTLResourceStorageModeShared];
	if (converted) perBuffer[cacheKey] = converted;
	return converted != nil;
}

static MTLSamplerAddressMode addressMode(GLenum wrap) {
	switch (wrap) {
		case GL_CLAMP_TO_EDGE:
		case GL_CLAMP_TO_BORDER: // clamp-to-border colours are not available on every iOS GPU, edge clamp is the closest
			return MTLSamplerAddressModeClampToEdge;
		case GL_MIRRORED_REPEAT:
			return MTLSamplerAddressModeMirrorRepeat;
		default:
			return MTLSamplerAddressModeRepeat;
	}
}

static void putBytes(std::string &key, const void *data, size_t size) {
	key.append((const char *)data, size);
}

static bool textureIsAttachedToDrawFramebuffer(GLuint textureName) {
	if (textureName == 0 || ctx.draw_fbo == 0) return false;
	auto fbo = ctx.fbo_objects.find(ctx.draw_fbo);
	if (fbo == ctx.fbo_objects.end()) return false;
	const FBO &attachments = fbo->second;
	for (const FBOAttachment &a : attachments.color) if (a.texture == textureName) return true;
	return attachments.depth.texture == textureName || attachments.stencil.texture == textureName;
}

static std::unordered_set<uint64_t> reportedFeedbackLoops;
static std::mutex vertexSamplerTraceMutex;
static std::unordered_set<std::string> reportedVertexSamplerBindings;
static std::mutex fragmentSamplerTraceMutex;
static std::unordered_set<std::string> reportedFragmentSamplerBindings;
static std::mutex vertexFetchTraceMutex;
static std::unordered_set<std::string> reportedVertexFetchBindings;

static bool firstVertexSamplerTrace(const std::string &key) {
	std::lock_guard<std::mutex> lock(vertexSamplerTraceMutex);
	return reportedVertexSamplerBindings.insert(key).second;
}

static bool firstFragmentSamplerTrace(const std::string &key) {
	std::lock_guard<std::mutex> lock(fragmentSamplerTraceMutex);
	return reportedFragmentSamplerBindings.insert(key).second;
}

static bool firstVertexFetchTrace(const std::string &key) {
	std::lock_guard<std::mutex> lock(vertexFetchTraceMutex);
	return reportedVertexFetchBindings.insert(key).second;
}

static id<MTLSamplerState> samplerFor(const TextureObject &t, GLuint textureUnit) {
	// Snapshot the bound sampler and its parameters while sampler-object state is locked.
	GLenum minFilter = t.min_filter, magFilter = t.mag_filter;
	GLenum wrapS = t.wrap_s, wrapT = t.wrap_t, wrapR = t.wrap_r;
	GLfloat minLod = t.min_lod, maxLod = t.max_lod, anisotropy = t.max_anisotropy;
	GLenum compareMode = t.compare_mode, compareFunc = t.compare_func;
	{
		std::lock_guard<std::recursive_mutex> objectLock(gl4metalSamplerObjectMutex);
		const GLuint samplerName = ctx.bound_samplers[textureUnit];
		if (samplerName != 0) {
			auto found = ctx.sampler_objects.find(samplerName);
			if (found != ctx.sampler_objects.end()) {
				const SamplerObject &so = found->second;
				minFilter = so.min_filter; magFilter = so.mag_filter;
				wrapS = so.wrap_s; wrapT = so.wrap_t; wrapR = so.wrap_r;
				minLod = so.min_lod; maxLod = so.max_lod; anisotropy = so.max_anisotropy;
				compareMode = so.compare_mode; compareFunc = so.compare_func;
			}
		}
	}
	std::lock_guard<std::mutex> cacheLock(samplerCacheMutex);
	// Parameters of the texture, overridden by the sampler object bound to the unit (glBindSampler) when there is one.
	std::string key;
	putBytes(key, &minFilter, sizeof(minFilter));
	putBytes(key, &magFilter, sizeof(magFilter));
	putBytes(key, &wrapS, sizeof(wrapS));
	putBytes(key, &wrapT, sizeof(wrapT));
	putBytes(key, &wrapR, sizeof(wrapR));
	putBytes(key, &minLod, sizeof(minLod));
	putBytes(key, &maxLod, sizeof(maxLod));
	putBytes(key, &t.base_level, sizeof(t.base_level));
	putBytes(key, &t.max_level, sizeof(t.max_level));
	putBytes(key, &anisotropy, sizeof(anisotropy));
	putBytes(key, &compareMode, sizeof(compareMode));
	putBytes(key, &compareFunc, sizeof(compareFunc));

	auto it = ctx.sampler_cache.find(key);
	if (it != ctx.sampler_cache.end()) {
		return it->second;
	}

	bool mipmapped = minFilter == GL_NEAREST_MIPMAP_NEAREST || minFilter == GL_LINEAR_MIPMAP_NEAREST ||
					 minFilter == GL_NEAREST_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_LINEAR;
	bool minLinear = minFilter == GL_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST || minFilter == GL_LINEAR_MIPMAP_LINEAR;
	bool mipLinear = minFilter == GL_NEAREST_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_LINEAR;

	MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
	d.minFilter = minLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	d.magFilter = magFilter == GL_NEAREST ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
	d.mipFilter = !mipmapped ? MTLSamplerMipFilterNotMipmapped : (mipLinear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest);
	d.sAddressMode = addressMode(wrapS);
	d.tAddressMode = addressMode(wrapT);
	d.rAddressMode = addressMode(wrapR);
	float lodMin = std::max(std::max(minLod, (float)t.base_level), 0.0f);
	float lodMax = std::max(std::min(maxLod, (float)t.max_level), lodMin);
	d.lodMinClamp = lodMin;
	d.lodMaxClamp = lodMax;
	d.maxAnisotropy = (NSUInteger)std::min(std::max(anisotropy, 1.0f), 16.0f);
	if (compareMode == GL_COMPARE_REF_TO_TEXTURE) {
		d.compareFunction = getMetalCompareFunc(compareFunc);
	}

	id<MTLSamplerState> sampler = [ctx.device newSamplerStateWithDescriptor:d];
	if (!sampler) {
		GL4METAL_LOG("samplerFor: Metal sampler allocation failed (min=0x%X mag=0x%X wrap=%X/%X/%X)",
			minFilter, magFilter, wrapS, wrapT, wrapR);
		return nil;
	}
	ctx.sampler_cache.emplace(key, sampler);
	return sampler;
}

// What GL samples from a texture unit with nothing (usable) bound: (0, 0, 0, 1).
static GLenum targetForDim(TranslatedUniform::Dim dim) {
	switch (dim) {
		case TranslatedUniform::DimCube:    return GL_TEXTURE_CUBE_MAP;
		case TranslatedUniform::Dim3D:      return GL_TEXTURE_3D;
		case TranslatedUniform::Dim2DArray: return GL_TEXTURE_2D_ARRAY;
		default:                            return GL_TEXTURE_2D;
	}
}

static MTLTextureType typeForDim(TranslatedUniform::Dim dim) {
	switch (dim) {
		case TranslatedUniform::DimCube:    return MTLTextureTypeCube;
		case TranslatedUniform::Dim3D:      return MTLTextureType3D;
		case TranslatedUniform::Dim2DArray: return MTLTextureType2DArray;
		default:                            return MTLTextureType2D;
	}
}

// What GL samples from a texture unit with nothing (usable) bound: (0, 0, 0, 1). The shader decides the texture type,
// and Metal wants exactly that type bound, so there is one fallback per sampler dimensionality.
static id<MTLTexture> fallbackTexture(TranslatedUniform::Dim dim) {
	const int dimensionIndex = (int)dim;
	if (dimensionIndex < 0 || dimensionIndex >= 4 || !ctx.device) return nil;
	std::lock_guard<std::mutex> lock(fallbackTextureMutex);
	static id<MTLTexture> textures[4] = { nil, nil, nil, nil };
	__strong id<MTLTexture> &slot = textures[dimensionIndex];
	if (!slot) {
		MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
		d.textureType = typeForDim(dim);
		d.pixelFormat = MTLPixelFormatRGBA8Unorm;
		d.width = d.height = 1;
		d.depth = 1;
		d.arrayLength = 1;
		d.usage = MTLTextureUsageShaderRead;
		d.storageMode = MTLStorageModeShared;
		slot = [ctx.device newTextureWithDescriptor:d];
		if (!slot) return nil;
		const uint8_t black[4] = { 0, 0, 0, 255 };
		NSUInteger slices = dim == TranslatedUniform::DimCube ? 6 : 1;
		for (NSUInteger slice = 0; slice < slices; slice++) {
			[slot replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:slice withBytes:black bytesPerRow:4 bytesPerImage:4];
		}
	}
	return slot;
}

static id<MTLSamplerState> fallbackSampler() {
	if (!ctx.device) return nil;
	std::lock_guard<std::mutex> lock(fallbackSamplerMutex);
	static id<MTLSamplerState> sampler = nil;
	if (!sampler) {
		MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
		d.minFilter = d.magFilter = MTLSamplerMinMagFilterNearest;
		sampler = [ctx.device newSamplerStateWithDescriptor:d];
	}
	return sampler;
}

// One entry per vertex attribute the program uses. Every attribute gets its own Metal buffer slot (0...15), so the
// byte offset of the attribute inside its GL buffer is simply the binding offset and Metal's "attribute must fit in the
// stride" rule can't be violated by separate streams that live in the same GL buffer.
struct ActiveAttrib {
	const ProgramAttrib *pa = nullptr;
	bool constant = true;  // no usable array: GL's default (0, 0, 0, 1)
	MTLVertexFormat format = MTLVertexFormatInvalid;
	id<MTLBuffer> buffer = nil;
	NSUInteger offset = 0;
	NSUInteger stride = 0;
	NSUInteger elementBytes = 0; // bytes consumed by one vertex/instance attribute element
	GLuint divisor = 0;
	NSUInteger slot = 0;
};

static id<MTLRenderPipelineState> pipelineFor(GLuint programId, ProgramObject &p, std::vector<ActiveAttrib> &attribs, const char *caller) {
	std::lock_guard<std::mutex> cacheLock(pipelineCacheMutex);
	bool flip = ctx.pass_flipped;
	bool blend = capOn(GL_BLEND);

	NSUInteger slot = 0;
	for (ActiveAttrib &aa : attribs) {
		if (!aa.pa || aa.pa->msl_location >= 31 || slot >= 16 || aa.format == MTLVertexFormatInvalid) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: invalid MSL vertex attribute location/slot/format (location=%u slot=%lu format=%lu)",
				caller, aa.pa ? aa.pa->msl_location : UINT32_MAX, (unsigned long)slot, (unsigned long)aa.format);
			return nil;
		}
		aa.slot = slot++;
	}

	std::string key;
	auto put = [&key](uint64_t v) { putBytes(key, &v, sizeof(v)); };
	put(programId);
	put(flip ? 1 : 0);
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		put((uint64_t)ctx.pass_color_format[i]);
	}
	put((uint64_t)ctx.pass_depth_format);
	put((uint64_t)ctx.pass_stencil_format);
	put(blend ? 1 : 0);
	if (blend) {
		put(ctx.blend_src_rgb); put(ctx.blend_dst_rgb); put(ctx.blend_src_alpha); put(ctx.blend_dst_alpha);
		put(ctx.blend_eq_rgb); put(ctx.blend_eq_alpha);
	}
	put((uint64_t)ctx.color_mask[0] | ((uint64_t)ctx.color_mask[1] << 1) | ((uint64_t)ctx.color_mask[2] << 2) | ((uint64_t)ctx.color_mask[3] << 3));
	for (const ActiveAttrib &aa : attribs) {
		put(aa.pa->msl_location);
		put((uint64_t)aa.format);
		put(aa.slot);
		put(aa.stride);
		put(aa.divisor);
		put(aa.constant ? 1 : 0);
	}

	auto cached = ctx.pipeline_cache.find(key);
	if (cached != ctx.pipeline_cache.end()) {
		return cached->second; // may be nil: a combination that already failed was reported once
	}

	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	for (const ActiveAttrib &aa : attribs) {
		MTLVertexAttributeDescriptor *ad = vd.attributes[aa.pa->msl_location];
		ad.format = aa.format;
		ad.offset = 0;
		ad.bufferIndex = aa.slot;

		MTLVertexBufferLayoutDescriptor *ld = vd.layouts[aa.slot];
		if (aa.constant) {
			ld.stepFunction = MTLVertexStepFunctionConstant;
			ld.stepRate = 0;
			// Metal serializes the layout even for constant attributes; zero stride is rejected.
			ld.stride = 16;
		} else if (aa.divisor == 0) {
			ld.stepFunction = MTLVertexStepFunctionPerVertex;
			ld.stepRate = 1;
			ld.stride = aa.stride;
		} else {
			ld.stepFunction = MTLVertexStepFunctionPerInstance;
			ld.stepRate = aa.divisor;
			ld.stride = aa.stride;
		}
	}

	MTLRenderPipelineDescriptor *d = [[MTLRenderPipelineDescriptor alloc] init];
	d.vertexFunction = flip ? p.vertex_function_flipped : p.vertex_function;
	d.fragmentFunction = p.fragment_function;
	d.vertexDescriptor = vd;

	MTLColorWriteMask mask = 0;
	if (ctx.color_mask[0]) mask |= MTLColorWriteMaskRed;
	if (ctx.color_mask[1]) mask |= MTLColorWriteMaskGreen;
	if (ctx.color_mask[2]) mask |= MTLColorWriteMaskBlue;
	if (ctx.color_mask[3]) mask |= MTLColorWriteMaskAlpha;

	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		if (ctx.pass_color_format[i] == MTLPixelFormatInvalid) {
			continue;
		}
		MTLRenderPipelineColorAttachmentDescriptor *ca = d.colorAttachments[i];
		ca.pixelFormat = ctx.pass_color_format[i];
		ca.writeMask = mask;
		if (blend) {
			ca.blendingEnabled = YES;
			ca.sourceRGBBlendFactor = blendFactor(ctx.blend_src_rgb);
			ca.destinationRGBBlendFactor = blendFactor(ctx.blend_dst_rgb);
			ca.sourceAlphaBlendFactor = blendFactor(ctx.blend_src_alpha);
			ca.destinationAlphaBlendFactor = blendFactor(ctx.blend_dst_alpha);
			ca.rgbBlendOperation = blendOperation(ctx.blend_eq_rgb);
			ca.alphaBlendOperation = blendOperation(ctx.blend_eq_alpha);
		}
	}
	d.depthAttachmentPixelFormat = ctx.pass_depth_format;
	d.stencilAttachmentPixelFormat = ctx.pass_stencil_format;

	NSError *error = nil;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:d error:&error];
	if (!pipeline) {
		GL4METAL_LOG("%s: pipeline creation failed for program %u: %s", caller, programId,
			error ? error.localizedDescription.UTF8String : "unknown Metal pipeline error");
	}
	ctx.pipeline_cache[key] = pipeline;
	return pipeline;
}

struct DrawCall {
	GLenum mode;
	GLint first;          // glDrawArrays
	GLsizei count;
	bool indexed;
	GLenum indexType;
	GLintptr indexOffset; // glDrawElements: byte offset into the element array buffer
	GLsizei instances;
	GLint baseVertex;
};

struct VertexIndexRange {
	uint64_t minIndex = 0;
	uint64_t maxIndex = 0;
};

static id<MTLBuffer> createTemporaryIndexBuffer(const std::vector<uint32_t> &indices, const char *caller) {
	if (indices.empty() || !ctx.device || indices.size() > SIZE_MAX / sizeof(uint32_t)) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: temporary index-buffer size overflow or empty input", caller);
		return nil;
	}
	const NSUInteger byteLength = (NSUInteger)(indices.size() * sizeof(uint32_t));
	if (byteLength > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: temporary index buffer (%llu bytes) exceeds Metal maxBufferLength", caller, (unsigned long long)byteLength);
		return nil;
	}
	id<MTLBuffer> buffer = [ctx.device newBufferWithBytes:indices.data() length:byteLength options:MTLResourceStorageModeShared];
	if (!buffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: Metal failed to allocate %llu-byte temporary index buffer", caller, (unsigned long long)byteLength);
	}
	return buffer;
}

static bool validateVertexFetchRanges(const DrawCall &dc, const char *caller,
		const std::vector<ActiveAttrib> &attribs, id<MTLBuffer> elementBuffer,
		NSUInteger indexSize) {
	if (!vertexFetchValidationEnabled()) return true;

	VertexIndexRange range;
	if (!dc.indexed) {
		if (dc.first < 0) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("%s: negative first vertex %d; draw skipped", caller, dc.first);
			return false;
		}
		const uint64_t first = (uint64_t)dc.first;
		const uint64_t last = first + (uint64_t)dc.count - 1;
		if (last < first) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("%s: vertex range overflows; draw skipped", caller);
			return false;
		}
		range.minIndex = first;
		range.maxIndex = last;
	} else {
		if (!elementBuffer || !elementBuffer.contents || indexSize == 0) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: cannot validate vertex range because the element buffer is not CPU-readable; draw skipped", caller);
			return false;
		}
		const uint8_t *raw = (const uint8_t *)elementBuffer.contents + (NSUInteger)dc.indexOffset;
		uint64_t minIndex = UINT64_MAX;
		uint64_t maxIndex = 0;
		for (GLsizei i = 0; i < dc.count; ++i) {
			uint32_t index = 0;
			const uint8_t *entry = raw + (NSUInteger)i * indexSize;
			if (dc.indexType == GL_UNSIGNED_INT) {
				memcpy(&index, entry, sizeof(uint32_t));
			} else if (dc.indexType == GL_UNSIGNED_SHORT) {
				uint16_t value = 0;
				memcpy(&value, entry, sizeof(value));
				index = value;
			} else {
				index = *entry;
			}
			const int64_t effective = (int64_t)index + (int64_t)dc.baseVertex;
			if (effective < 0) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: index %u plus baseVertex %d is negative; draw skipped", caller, index, dc.baseVertex);
				return false;
			}
			const uint64_t adjusted = (uint64_t)effective;
			minIndex = std::min(minIndex, adjusted);
			maxIndex = std::max(maxIndex, adjusted);
		}
		if (minIndex == UINT64_MAX) return true;
		range.minIndex = minIndex;
		range.maxIndex = maxIndex;
	}

	for (const ActiveAttrib &aa : attribs) {
		if (aa.constant || !aa.buffer) continue;
		const uint64_t elementIndex = aa.divisor == 0
			? range.maxIndex
			: ((uint64_t)dc.instances - 1) / (uint64_t)aa.divisor;
		if (aa.elementBytes == 0 || aa.stride == 0 || aa.offset > aa.buffer.length) {
			GL4METAL_LOG("%s: invalid vertex binding for '%s' (offset=%lu stride=%lu element=%lu length=%lu); draw skipped",
				caller, aa.pa ? aa.pa->name.c_str() : "?", (unsigned long)aa.offset,
				(unsigned long)aa.stride, (unsigned long)aa.elementBytes, (unsigned long)aa.buffer.length);
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
		const uint64_t length = (uint64_t)aa.buffer.length;
		const uint64_t offset = (uint64_t)aa.offset;
		const uint64_t stride = (uint64_t)aa.stride;
		const uint64_t bytes = (uint64_t)aa.elementBytes;
		if (bytes > UINT64_MAX - offset ||
			elementIndex > (UINT64_MAX - offset - bytes) / stride ||
			offset + elementIndex * stride + bytes > length) {
			GL4METAL_LOG("%s: vertex fetch out of range: program=%u attribute='%s' index=%llu offset=%llu stride=%llu element=%llu buffer=%llu; draw skipped",
				caller, ctx.current_program, aa.pa ? aa.pa->name.c_str() : "?",
				(unsigned long long)elementIndex, (unsigned long long)offset,
				(unsigned long long)stride, (unsigned long long)bytes, (unsigned long long)length);
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
	}
	return true;
}

static void executeDraw(const DrawCall &dc, const char *caller) {
	if (dc.count < 0 || dc.instances < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: negative count (%d) or instance count (%d)", caller, dc.count, dc.instances);
		return;
	}
	if (dc.count == 0 || dc.instances == 0) return;
	if (!dc.indexed && dc.first < 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: negative first vertex %d (GL_INVALID_VALUE)", caller, dc.first);
		return;
	}

	auto pit = ctx.program_objects.find(ctx.current_program);
	if (ctx.current_program == 0 || pit == ctx.program_objects.end() || !pit->second.linked) {
		static bool warned = false;
		if (!warned) {
			warned = true;
			GL4METAL_LOG("%s: no linked program in use, draw skipped (reported once)", caller);
		}
		return;
	}
	ProgramObject &p = pit->second;

	MTLPrimitiveType prim = MTLPrimitiveTypeTriangle;
	bool fan = false;
	bool triangles = true;
	switch (dc.mode) {
		case GL_POINTS:         prim = MTLPrimitiveTypePoint; triangles = false; break;
		case GL_LINES:          prim = MTLPrimitiveTypeLine; triangles = false; break;
		case GL_LINE_STRIP:     prim = MTLPrimitiveTypeLineStrip; triangles = false; break;
		case GL_TRIANGLES:      prim = MTLPrimitiveTypeTriangle; break;
		case GL_TRIANGLE_STRIP: prim = MTLPrimitiveTypeTriangleStrip; break;
		case GL_TRIANGLE_FAN:   prim = MTLPrimitiveTypeTriangle; fan = true; break; // Metal has no fans: re-indexed below
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("%s: unsupported primitive mode 0x%X", caller, dc.mode);
			return;
	}
	// A triangle fan with fewer than three vertices produces no primitives. Return before
	// expanding (count - 2), which is unsigned and would otherwise underflow.
	if (fan && dc.count < 3) return;

	id<MTLRenderCommandEncoder> enc = ctx.currentEncoder ? ctx.currentEncoder : beginRenderPass(0);
	if (!enc) {
		return;
	}

	// Vertex attributes of the current VAO, for the attributes this program reads.
	VertexArray &vao = ctx.vao_objects[ctx.current_vao];
	std::vector<ActiveAttrib> attribs;
	for (const ProgramAttrib &pa : p.attribs) {
		if (pa.gl_index < 0 || pa.gl_index >= 16) {
			continue;
		}
		const VertexAttrib &va = vao.attribs[pa.gl_index];
		if (va.enabled && va.bound_vbo != 0) {
			auto storage = ctx.buffer_objects.find(va.bound_vbo);
			if (storage == ctx.buffer_objects.end() || !storage->second) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: enabled attribute '%s' references VBO %u without valid Metal storage; draw skipped",
					caller, pa.name.c_str(), va.bound_vbo);
				return;
			}
		}
		ActiveAttrib aa;
		aa.pa = &pa;
		auto bit = ctx.buffer_objects.find(va.bound_vbo);
		if (va.enabled && va.bound_vbo != 0 && bit != ctx.buffer_objects.end() && bit->second != nil) {
			NSUInteger tight = 0;
			MTLVertexFormat format = vertexFormatFor(va, tight);
			if (format == MTLVertexFormatInvalid) {
				GL4METAL_LOG("%s: attribute '%s' has an unsupported type 0x%X/size %d", caller, pa.name.c_str(), va.type, va.size);
				return;
			}
			aa.constant = false;
			aa.format = format;
			aa.buffer = bit->second;
			aa.offset = (NSUInteger)(uintptr_t)va.pointer;
			aa.stride = va.stride > 0 ? (NSUInteger)va.stride : tight;
			aa.elementBytes = tight; // native vertex fetch width; converted integer formats override this below
			aa.divisor = va.divisor;

			if (pa.is_integer && va.integer) {
				bool sourceUnsigned = false;
				NSUInteger sourceComponentSize = 0;
				const bool hasIntegerStorageType = integerVertexTypeInfo(va.type, sourceUnsigned, sourceComponentSize);
				(void)sourceComponentSize;
				if (hasIntegerStorageType && sourceUnsigned != pa.is_unsigned) {
					NSUInteger convertedStride = 0;
					id<MTLBuffer> converted = nil;
					if (!makeSignednessConvertedVertexBuffer(va.bound_vbo, va, bit->second, convertedStride, converted)) {
						GL4METAL_LOG("%s: cannot convert integer vertex attribute '%s' from GL type 0x%X for the shader's %s input", caller, pa.name.c_str(), va.type, pa.is_unsigned ? "unsigned" : "signed");
						return;
					}
					static const MTLVertexFormat signedFormats[5] = { MTLVertexFormatInvalid, MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4 };
					static const MTLVertexFormat unsignedFormats[5] = { MTLVertexFormatInvalid, MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4 };
					aa.format = pa.is_unsigned ? unsignedFormats[va.size] : signedFormats[va.size];
					aa.buffer = converted;
					aa.offset = 0;
					aa.stride = convertedStride;
					aa.elementBytes = (NSUInteger)va.size * sizeof(uint32_t);
				}
			}
		} else {
			aa.constant = true;
			aa.format = pa.is_integer ? (pa.is_unsigned ? MTLVertexFormatUInt4 : MTLVertexFormatInt4) : MTLVertexFormatFloat4;
		}
		if (vertexFetchTraceEnabled()) {
			const std::string traceKey = std::to_string(ctx.current_program) + ":" + pa.name + ":" +
				std::to_string(va.bound_vbo) + ":" + std::to_string((uintptr_t)(__bridge void *)aa.buffer) +
				":" + std::to_string((unsigned long)aa.format) + ":" + std::to_string(aa.stride);
			if (firstVertexFetchTrace(traceKey)) {
				GL4METAL_LOG("vertex attribute binding: program=%u name='%s' GLindex=%d VBO=%u integer=%d unsigned=%d normalized=%d sourceType=0x%X size=%d stride=%lu offset=%lu divisor=%u MetalFormat=%lu MetalBuffer=%p bytes=%lu elementBytes=%lu",
					ctx.current_program, pa.name.c_str(), pa.gl_index, va.bound_vbo, va.integer ? 1 : 0,
					pa.is_unsigned ? 1 : 0, va.normalized ? 1 : 0, va.type, va.size,
					(unsigned long)va.stride, (unsigned long)(uintptr_t)va.pointer, va.divisor,
					(unsigned long)aa.format, (__bridge void *)aa.buffer,
					(unsigned long)(aa.buffer ? aa.buffer.length : 0), (unsigned long)aa.elementBytes);
			}
		}
		attribs.push_back(aa);
	}

	id<MTLRenderPipelineState> pipeline = pipelineFor(ctx.current_program, p, attribs, caller);
	if (!pipeline) {
		return;
	}

	// Fixed-function state.
	MTLCullMode cull = MTLCullModeNone;
	if (capOn(GL_CULL_FACE) && triangles) {
		if (ctx.cull_face_mode == GL_FRONT_AND_BACK) {
			return; // every triangle is culled
		}
		cull = ctx.cull_face_mode == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
	}
	// Texture targets are drawn Y-flipped, which reverses the winding as seen by Metal.
	bool ccw = (ctx.front_face == GL_CCW) != ctx.pass_flipped;

	// The command-buffer label is only a hint: Metal reports the last encoded draw in the buffer,
	// not necessarily the exact draw that first caused an asynchronous GPU page fault.
	static std::atomic<uint64_t> drawSequence{0};
	const uint64_t sequence = drawSequence.fetch_add(1, std::memory_order_relaxed) + 1;
	NSString *drawLabel = [NSString stringWithFormat:
		@"GL4Metal last-draw#%llu %s program=%u FBO=%u mode=0x%X indexed=%d count=%d instances=%d first=%d baseVertex=%d indexType=0x%X indexOffset=%lld",
		(unsigned long long)sequence, caller, ctx.current_program, ctx.draw_fbo, dc.mode,
		dc.indexed ? 1 : 0, dc.count, dc.instances, dc.first, dc.baseVertex, dc.indexType, (long long)dc.indexOffset];
	if (ctx.currentCommandBuffer) ctx.currentCommandBuffer.label = drawLabel;
	enc.label = drawLabel;
	id<MTLDepthStencilState> activeDepthState = (ctx.pass_depth_format != MTLPixelFormatInvalid ? ctx.depthStencilState : ctx.no_depth_state);
	if (!activeDepthState) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: no valid Metal depth-stencil state; draw skipped", caller);
		return;
	}
	[enc setRenderPipelineState:pipeline];
	[enc setDepthStencilState:activeDepthState];
	[enc setFrontFacingWinding:ccw ? MTLWindingCounterClockwise : MTLWindingClockwise];
	[enc setCullMode:cull];
	[enc setTriangleFillMode:ctx.polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
	if (capOn(GL_POLYGON_OFFSET_FILL)) {
		[enc setDepthBias:ctx.polygon_offset_units slopeScale:ctx.polygon_offset_factor clamp:0.0f];
	} else {
		[enc setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
	}
	[enc setBlendColorRed:ctx.blend_color[0] green:ctx.blend_color[1] blue:ctx.blend_color[2] alpha:ctx.blend_color[3]];

	// Vertex buffers.
	for (const ActiveAttrib &aa : attribs) {
		if (aa.constant) {
			uint8_t value[16]; // glVertexAttrib*: the attribute's current value
			genericAttribValue((GLuint)aa.pa->gl_index, aa.pa->is_integer, value);
			[enc setVertexBytes:value length:sizeof(value) atIndex:aa.slot];
		} else {
			[enc setVertexBuffer:aa.buffer offset:aa.offset atIndex:aa.slot];
		}
	}

	// Default uniform block, one CPU copy per stage (see glUniform*).
	for (int s = 0; s < 2; s++) {
		const std::vector<uint8_t> &data = p.uniform_data[s];
		if (data.empty()) {
			continue;
		}
		if (data.size() <= 4096) { // setBytes limit
			if (s == 0) {
				[enc setVertexBytes:data.data() length:data.size() atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			} else {
				[enc setFragmentBytes:data.data() length:data.size() atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			}
		} else {
			if (data.size() > ctx.device.maxBufferLength) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: default uniform block (%lu bytes) exceeds Metal maxBufferLength; draw skipped", caller, (unsigned long)data.size());
				return;
			}
			id<MTLBuffer> buffer = [ctx.device newBufferWithBytes:data.data() length:data.size() options:MTLResourceStorageModeShared];
			if (!buffer) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: failed to allocate default uniform block (%lu bytes); draw skipped", caller, (unsigned long)data.size());
				return;
			}
			if (s == 0) {
				[enc setVertexBuffer:buffer offset:0 atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			} else {
				[enc setFragmentBuffer:buffer offset:0 atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			}
		}
	}

	// Uniform blocks: never leave an active MSL buffer slot unset or reuse a stale binding.
	// The translator reserves 16...29 for real UBOs and 30 for GL's default uniform block.
	std::unordered_set<uint32_t> usedVertexBlockSlots;
	std::unordered_set<uint32_t> usedFragmentBlockSlots;
	for (const ProgramBlock &block : p.blocks) {
		if (!block.info.stage[0].present && !block.info.stage[1].present) continue;
		if (block.binding_point >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' uses invalid GL binding point %u; draw skipped",
				caller, block.info.name.c_str(), block.binding_point);
			return;
		}
		if ((block.info.stage[0].present && (block.info.stage[0].slot < 16 || block.info.stage[0].slot >= GL4METAL_UNIFORM_BUFFER_INDEX)) ||
			(block.info.stage[1].present && (block.info.stage[1].slot < 16 || block.info.stage[1].slot >= GL4METAL_UNIFORM_BUFFER_INDEX))) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' maps to a reserved/out-of-range Metal buffer slot (VS=%u FS=%u); draw skipped",
				caller, block.info.name.c_str(), block.info.stage[0].slot, block.info.stage[1].slot);
			return;
		}
		if ((block.info.stage[0].present && !usedVertexBlockSlots.insert(block.info.stage[0].slot).second) ||
			(block.info.stage[1].present && !usedFragmentBlockSlots.insert(block.info.stage[1].slot).second)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: multiple uniform blocks collide on a Metal buffer slot; block '%s' draw skipped",
				caller, block.info.name.c_str());
			return;
		}
		const UniformBufferBinding &binding = ctx.uniform_buffer_bindings[block.binding_point];
		auto bound = ctx.buffer_objects.find(binding.buffer);
		if (binding.buffer == 0 || bound == ctx.buffer_objects.end() || !bound->second) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: active uniform block '%s' has no buffer bound at GL binding point %u; draw skipped",
				caller, block.info.name.c_str(), block.binding_point);
			return;
		}
		if (binding.offset < 0 || (binding.offset % 256) != 0) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' has invalid/alignment-mismatched offset %lld; draw skipped",
				caller, block.info.name.c_str(), (long long)binding.offset);
			return;
		}
		const NSUInteger length = bound->second.length;
		const NSUInteger offset = (NSUInteger)binding.offset;
		if (offset > length) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' offset %llu exceeds buffer length %llu; draw skipped",
				caller, block.info.name.c_str(), (unsigned long long)offset, (unsigned long long)length);
			return;
		}
		const NSUInteger available = length - offset;
		const NSUInteger boundSize = binding.size == 0 ? available : (binding.size < 0 ? 0 : (NSUInteger)binding.size);
		if (binding.size < 0 || boundSize > available || boundSize < block.info.size) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' needs %u bytes but binding %u provides %llu bytes (buffer=%u offset=%llu length=%llu); draw skipped",
				caller, block.info.name.c_str(), block.info.size, block.binding_point,
				(unsigned long long)boundSize, binding.buffer, (unsigned long long)offset, (unsigned long long)length);
			return;
		}
		if (block.info.stage[0].present) {
			[enc setVertexBuffer:bound->second offset:offset atIndex:block.info.stage[0].slot];
		}
		if (block.info.stage[1].present) {
			[enc setFragmentBuffer:bound->second offset:offset atIndex:block.info.stage[1].slot];
		}
	}

	// Textures: the sampler uniform holds a texture unit, the unit holds a texture.
	// A feedback-loop texture cannot safely be replaced with the generic 1x1 fallback: a
	// vertex-stage texelFetch/read (Iris uses this for iris_overlay) may address beyond it.
	// Since OpenGL defines sampling from the active render attachment as undefined here,
	// skip this draw rather than issuing a Metal draw with an unsafe substitute.
	bool hasTextureFeedbackLoop = false;
	for (const ProgramUniform &u : p.uniforms) {
		if (u.info.kind != TranslatedUniform::Sampler) {
			continue;
		}
		if (u.info.array_size <= 0 ||
			(u.info.stage[0].present && (u.info.stage[0].slot >= 16 || (uint64_t)u.info.stage[0].slot + (uint64_t)u.info.array_size > 16)) ||
			(u.info.stage[1].present && (u.info.stage[1].slot >= 16 || (uint64_t)u.info.stage[1].slot + (uint64_t)u.info.array_size > 16))) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: sampler '%s' has invalid MSL texture/sampler slot range (VS=%u FS=%u count=%d); draw skipped",
				caller, u.info.name.c_str(), u.info.stage[0].slot, u.info.stage[1].slot, u.info.array_size);
			return;
		}
		for (int e = 0; e < u.info.array_size; e++) {
			GLint unit = u.units[(size_t)e];
			// Sampler uniforms index the combined texture-unit namespace. The per-stage
			// limits constrain the number of active samplers, not the numeric unit ID.
			if ((u.info.stage[0].present || u.info.stage[1].present) &&
				(unit < 0 || unit >= GL4METAL_MAX_TEXTURE_UNITS)) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: sampler '%s' references combined texture unit %d outside 0...%d; draw skipped",
					caller, u.info.name.c_str(), unit, GL4METAL_MAX_TEXTURE_UNITS - 1);
				return;
			}
			id<MTLTexture> texture = nil;
			id<MTLSamplerState> sampler = nil;
			GLuint textureNameForTrace = 0;
			bool usedFallbackTexture = false;
			if (unit >= 0 && unit < GL4METAL_MAX_TEXTURE_UNITS) {
				auto bound = ctx.bound_textures[unit].find(targetForDim(u.info.dim));
				if (bound != ctx.bound_textures[unit].end() && bound->second != 0) {
					const GLuint textureName = bound->second;
					textureNameForTrace = textureName;
					if (textureIsAttachedToDrawFramebuffer(textureName)) {
						const uint64_t key = ((uint64_t)ctx.draw_fbo << 32) | textureName;
						bool firstReport = false;
						{
							std::lock_guard<std::mutex> lock(feedbackLoopSetMutex);
							firstReport = reportedFeedbackLoops.insert(key).second;
						}
						if (firstReport) {
							GL4METAL_LOG("draw: skipping texture feedback loop (program=%u, framebuffer=%u, texture=%u); sampling the active render attachment is undefined",
								ctx.current_program, ctx.draw_fbo, textureName);
						}
						hasTextureFeedbackLoop = true;
					} else {
						auto to = ctx.texture_objects.find(textureName);
						if (to != ctx.texture_objects.end() && to->second.texture &&
							to->second.texture.textureType == typeForDim(u.info.dim) &&
							(to->second.texture.usage & MTLTextureUsageShaderRead) != 0) {
							texture = to->second.texture;
							sampler = samplerFor(to->second, (GLuint)unit);
						}
					}
				}
			}
			if (!texture) {
				usedFallbackTexture = true;
				texture = fallbackTexture(u.info.dim);
				sampler = fallbackSampler();
			}
			// Never encode a shader read against a nil or non-readable Metal resource. A nil
			// texture can surface later as an asynchronous GPU page fault at address zero.
			if (!texture || (texture.usage & MTLTextureUsageShaderRead) == 0) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: sampler '%s' (unit=%d, program=%u) has no valid ShaderRead texture; fallback allocation failed; draw skipped",
					caller, u.info.name.c_str(), unit, ctx.current_program);
				return;
			}
			if (!sampler) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: sampler state creation failed for '%s' (unit=%d, program=%u); draw skipped",
					caller, u.info.name.c_str(), unit, ctx.current_program);
				return;
			}
			if (vertexSamplerTraceEnabled() && u.info.stage[0].present && texture) {
				std::string traceKey = std::to_string(ctx.current_program) + ":" + u.info.name + ":" +
					std::to_string(unit) + ":" + std::to_string(textureNameForTrace) + ":" +
					std::to_string((uintptr_t)(__bridge void *)texture);
				if (firstVertexSamplerTrace(traceKey)) {
					GL4METAL_LOG("VS sampler binding: program=%u uniform='%s' unit=%d GLtexture=%u fallback=%d MetalTexture=%p type=%lu size=%lux%lux%lu mips=%lu format=%lu usage=0x%lx storage=%lu slot=%lu",
						ctx.current_program, u.info.name.c_str(), unit, textureNameForTrace, usedFallbackTexture ? 1 : 0,
						(__bridge void *)texture, (unsigned long)texture.textureType,
						(unsigned long)texture.width, (unsigned long)texture.height, (unsigned long)texture.depth,
						(unsigned long)texture.mipmapLevelCount, (unsigned long)texture.pixelFormat,
						(unsigned long)texture.usage, (unsigned long)texture.storageMode,
						(unsigned long)u.info.stage[0].slot);
				}
			}
			if (fragmentSamplerTraceEnabled() && u.info.stage[1].present && texture) {
				std::string traceKey = std::to_string(ctx.current_program) + ":" + u.info.name + ":" +
					std::to_string(unit) + ":" + std::to_string(textureNameForTrace) + ":" +
					std::to_string((uintptr_t)(__bridge void *)texture);
				if (firstFragmentSamplerTrace(traceKey)) {
					GL4METAL_LOG("FS sampler binding: program=%u uniform='%s' unit=%d GLtexture=%u fallback=%d MetalTexture=%p type=%lu size=%lux%lux%lu mips=%lu format=%lu usage=0x%lx storage=%lu slot=%lu",
						ctx.current_program, u.info.name.c_str(), unit, textureNameForTrace, usedFallbackTexture ? 1 : 0,
						(__bridge void *)texture, (unsigned long)texture.textureType,
						(unsigned long)texture.width, (unsigned long)texture.height, (unsigned long)texture.depth,
						(unsigned long)texture.mipmapLevelCount, (unsigned long)texture.pixelFormat,
						(unsigned long)texture.usage, (unsigned long)texture.storageMode,
						(unsigned long)u.info.stage[1].slot);
				}
			}
			if (u.info.stage[0].present) {
				NSUInteger index = u.info.stage[0].slot + (NSUInteger)e;
				[enc setVertexTexture:texture atIndex:index];
				[enc setVertexSamplerState:sampler atIndex:index];
			}
			if (u.info.stage[1].present) {
				NSUInteger index = u.info.stage[1].slot + (NSUInteger)e;
				[enc setFragmentTexture:texture atIndex:index];
				[enc setFragmentSamplerState:sampler atIndex:index];
			}
		}
	}
	if (hasTextureFeedbackLoop) {
		return; // Do not sample a render attachment, including through a too-small fallback texture.
	}

	// The draw itself.
	if (!dc.indexed) {
		if (!validateVertexFetchRanges(dc, caller, attribs, nil, 0)) return;
		if (!fan) {
			[enc drawPrimitives:prim vertexStart:(NSUInteger)dc.first vertexCount:(NSUInteger)dc.count
				  instanceCount:(NSUInteger)dc.instances baseInstance:0];
			return;
		}
		const uint64_t triangleCount = (uint64_t)dc.count - 2;
		if (triangleCount > (uint64_t)ctx.device.maxBufferLength / (3u * sizeof(uint32_t))) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("%s: triangle-fan expansion exceeds Metal maxBufferLength", caller);
			return;
		}
		std::vector<uint32_t> indices;
		try {
			indices.reserve((size_t)(triangleCount * 3));
			for (uint64_t i = 1; i + 1 < (uint64_t)dc.count; i++) {
				indices.push_back((uint32_t)((uint64_t)dc.first));
				indices.push_back((uint32_t)((uint64_t)dc.first + i));
				indices.push_back((uint32_t)((uint64_t)dc.first + i + 1));
			}
		} catch (const std::bad_alloc &) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			return;
		}
		id<MTLBuffer> tmp = createTemporaryIndexBuffer(indices, caller);
		if (!tmp) return;
		[enc drawIndexedPrimitives:prim indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:tmp indexBufferOffset:0
					 instanceCount:(NSUInteger)dc.instances baseVertex:0 baseInstance:0];
		return;
	}

	auto eit = ctx.buffer_objects.find(vao.element_array_buffer_id);
	if (vao.element_array_buffer_id == 0 || eit == ctx.buffer_objects.end() || !eit->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: no element array buffer bound to the current VAO", caller);
		return;
	}
	id<MTLBuffer> ebo = eit->second;
	NSUInteger indexSize = dc.indexType == GL_UNSIGNED_INT ? 4 : (dc.indexType == GL_UNSIGNED_SHORT ? 2 : 1);
	if (dc.indexType != GL_UNSIGNED_INT && dc.indexType != GL_UNSIGNED_SHORT && dc.indexType != GL_UNSIGNED_BYTE) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("%s: invalid index type 0x%X", caller, dc.indexType);
		return;
	}
	if (dc.indexOffset < 0 || (NSUInteger)dc.indexOffset > ebo.length ||
		(NSUInteger)dc.count > (ebo.length - (NSUInteger)dc.indexOffset) / indexSize) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: %d indices at offset %ld do not fit the element buffer (%lu bytes)", caller, dc.count, (long)dc.indexOffset, (unsigned long)ebo.length);
		return;
	}
	if (((NSUInteger)dc.indexOffset % indexSize) != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: index offset %ld is not aligned to %lu bytes; draw skipped", caller, (long)dc.indexOffset, (unsigned long)indexSize);
		return;
	}
	if (!validateVertexFetchRanges(dc, caller, attribs, ebo, indexSize)) return;

	if (dc.indexType != GL_UNSIGNED_BYTE && !fan) {
		[enc drawIndexedPrimitives:prim indexCount:(NSUInteger)dc.count
						 indexType:(dc.indexType == GL_UNSIGNED_INT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16)
					   indexBuffer:ebo indexBufferOffset:(NSUInteger)dc.indexOffset
					 instanceCount:(NSUInteger)dc.instances baseVertex:dc.baseVertex baseInstance:0];
		return;
	}

	// 8-bit indices (Metal has none) and triangle fans: rebuild as a 32-bit index list.
	const uint8_t *raw = (const uint8_t *)ebo.contents + dc.indexOffset;
	auto indexAt = [&](GLsizei i) -> uint32_t {
		switch (dc.indexType) {
			case GL_UNSIGNED_INT:   return ((const uint32_t *)raw)[i];
			case GL_UNSIGNED_SHORT: return ((const uint16_t *)raw)[i];
			default:                return raw[i];
		}
	};
	const uint64_t outputIndexCount = fan ? ((uint64_t)dc.count - 2) * 3 : (uint64_t)dc.count;
	if (outputIndexCount > (uint64_t)ctx.device.maxBufferLength / sizeof(uint32_t) || outputIndexCount > SIZE_MAX / sizeof(uint32_t)) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: converted index list exceeds Metal maxBufferLength", caller);
		return;
	}
	std::vector<uint32_t> indices;
	try {
		indices.reserve((size_t)outputIndexCount);
		if (fan) {
			for (GLsizei i = 1; i + 1 < dc.count; i++) {
				indices.push_back(indexAt(0));
				indices.push_back(indexAt(i));
				indices.push_back(indexAt(i + 1));
			}
		} else {
			for (GLsizei i = 0; i < dc.count; i++) indices.push_back(indexAt(i));
		}
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	if (indices.empty()) return;
	id<MTLBuffer> tmp = createTemporaryIndexBuffer(indices, caller);
	if (!tmp) return;
	[enc drawIndexedPrimitives:prim indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:tmp indexBufferOffset:0
				 instanceCount:(NSUInteger)dc.instances baseVertex:dc.baseVertex baseInstance:0];
}


void glDrawArrays(GLenum mode, GLint first, GLsizei count) {
	DrawCall dc = { mode, first, count, false, 0, 0, 1, 0 };
	executeDraw(dc, "glDrawArrays");
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
	DrawCall dc = { mode, 0, count, true, type, (GLintptr)(uintptr_t)indices, 1, 0 };
	executeDraw(dc, "glDrawElements");
}

void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {
	DrawCall dc = { mode, 0, count, true, type, (GLintptr)(uintptr_t)indices, 1, 0 }; // start/end are only a hint
	executeDraw(dc, "glDrawRangeElements");
}

void glMultiDrawArrays(GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount) {
	for (GLsizei i = 0; i < drawcount; i++) {
		glDrawArrays(mode, first[i], count[i]);
	}
}

void glMultiDrawElements(GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount) {
	for (GLsizei i = 0; i < drawcount; i++) {
		glDrawElements(mode, count[i], type, indices[i]);
	}
}

// MARK: - Instanced

void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
	DrawCall dc = { mode, first, count, false, 0, 0, instancecount, 0 };
	executeDraw(dc, "glDrawArraysInstanced");
}

void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
	DrawCall dc = { mode, 0, count, true, type, (GLintptr)(uintptr_t)indices, instancecount, 0 };
	executeDraw(dc, "glDrawElementsInstanced");
}

// MARK: - Base vertex

void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
	DrawCall dc = { mode, 0, count, true, type, (GLintptr)(uintptr_t)indices, 1, basevertex };
	executeDraw(dc, "glDrawElementsBaseVertex");
}

void glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
	glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
}

void glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex) {
	DrawCall dc = { mode, 0, count, true, type, (GLintptr)(uintptr_t)indices, instancecount, basevertex };
	executeDraw(dc, "glDrawElementsInstancedBaseVertex");
}

void glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex) {
	for (GLsizei i = 0; i < drawcount; i++) {
		glDrawElementsBaseVertex(mode, count[i], type, indices[i], basevertex[i]);
	}
}
