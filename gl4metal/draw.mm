#import "draw.h"

#include <mutex>
#include <atomic>
#include <limits>
#include <new>
#include <array>

// These caches are process-wide / context-wide and may be reached by more than one
// JVM thread. std::unordered_map and std::unordered_set do not support concurrent
// mutation, even when the keys differ.
static std::mutex convertedVertexBufferCacheMutex;
static std::mutex samplerCacheMutex;
static std::mutex pipelineCacheMutex;
static std::mutex feedbackLoopSetMutex;
static std::mutex fallbackTextureMutex;
static std::mutex fallbackSamplerMutex;

// Large default uniform blocks use one shared upload arena per in-flight command
// buffer instead of allocating a new MTLBuffer for every draw. Each slice is aligned
// to 256 bytes, matching the constant-buffer offset alignment used by GL4Metal UBOs.
struct DrawUniformUploadArena {
	id<MTLCommandBuffer> commandBuffer = nil;
	id<MTLBuffer> buffer = nil;
	NSUInteger used = 0;
};

static DrawUniformUploadArena drawUniformUploadArena;
static std::mutex drawUniformUploadArenaMutex;

void resetDrawUniformUploadArena(id<MTLCommandBuffer> commandBuffer) {
	std::lock_guard<std::mutex> lock(drawUniformUploadArenaMutex);
	if (!commandBuffer || drawUniformUploadArena.commandBuffer == commandBuffer) {
		drawUniformUploadArena.commandBuffer = nil;
		drawUniformUploadArena.buffer = nil;
		drawUniformUploadArena.used = 0;
	}
}

bool uploadLargeDefaultUniformBlock(const uint8_t *source, NSUInteger length,
	__strong id<MTLBuffer> &destinationBuffer, NSUInteger &destinationOffset) {
	if (!source || length == 0 || !ctx.device || !ctx.currentCommandBuffer ||
		length > ctx.device.maxBufferLength) return false;

	constexpr NSUInteger alignment = 256;
	std::lock_guard<std::mutex> lock(drawUniformUploadArenaMutex);
	id<MTLCommandBuffer> commandBuffer = ctx.currentCommandBuffer;
	if (drawUniformUploadArena.commandBuffer != commandBuffer) {
		drawUniformUploadArena.commandBuffer = commandBuffer;
		drawUniformUploadArena.buffer = nil;
		drawUniformUploadArena.used = 0;
	}

	if (drawUniformUploadArena.used > std::numeric_limits<NSUInteger>::max() - (alignment - 1)) return false;
	NSUInteger offset = (drawUniformUploadArena.used + (alignment - 1)) & ~(alignment - 1);
	if (offset > ctx.device.maxBufferLength || length > ctx.device.maxBufferLength - offset) return false;
	NSUInteger requiredEnd = offset + length;
	if (!drawUniformUploadArena.buffer || requiredEnd > drawUniformUploadArena.buffer.length) {
		NSUInteger capacity = std::max<NSUInteger>(256 * 1024, requiredEnd);
		if (drawUniformUploadArena.buffer && drawUniformUploadArena.buffer.length <= ctx.device.maxBufferLength / 2) {
			capacity = std::max(capacity, drawUniformUploadArena.buffer.length * 2);
		}
		capacity = std::min<NSUInteger>(capacity, ctx.device.maxBufferLength);
		if (capacity < requiredEnd) return false;
		id<MTLBuffer> expanded = [ctx.device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if (!expanded) return false;
		// Earlier encoder bindings retain the previous arena buffer until their command
		// buffer completes; new allocations can safely continue in the replacement.
		drawUniformUploadArena.buffer = expanded;
		drawUniformUploadArena.used = 0;
		offset = 0;
		requiredEnd = length;
	}

	destinationBuffer = drawUniformUploadArena.buffer;
	destinationOffset = offset;
	memcpy((uint8_t *)destinationBuffer.contents + destinationOffset, source, (size_t)length);
	drawUniformUploadArena.used = requiredEnd;
	return true;
}

static bool vertexFetchValidationEnabled() {
	// Full validation scans all indices on the CPU for every indexed draw. Keep it opt-in
	// for diagnostics; EBO byte-range validation remains enabled unconditionally.
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_VALIDATE_VERTEX_FETCH");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool drawLabelingEnabled() {
	// Formatting an NSString for every draw adds CPU and autorelease-pool pressure.
	// Enable only while diagnosing a specific GPU command-buffer fault.
	static const bool enabled = [] {
		const char *value = getenv("GL4METAL_TRACE_DRAW_LABELS");
		return value && value[0] == '1';
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

// Metal's render-pass descriptor contains only the color attachments selected by
// glDrawBuffers; other images retained on the GL FBO are not bound as Metal render
// targets. Iris/OptiFine shaderpacks commonly keep additional colortex images
// attached while sampling them in another pass. Checking every GL attachment here
// reports a false Metal read/write hazard and drops otherwise renderable passes.
// Match the resources that beginRenderPass() actually installs on the encoder.
static bool textureIsAttachedToDrawFramebuffer(GLuint textureName,
	GLenum *attachmentOut = nullptr, GLint *levelOut = nullptr, GLint *drawBufferOut = nullptr) {
	if (textureName == 0 || ctx.draw_fbo == 0) return false;
	auto fbo = ctx.fbo_objects.find(ctx.draw_fbo);
	if (fbo == ctx.fbo_objects.end()) return false;
	const FBO &attachments = fbo->second;

	// The i-th fragment output writes to the attachment named by draw_buffers[i].
	// Ignore GL_NONE and attachments that are present but not selected for drawing.
	for (GLint outputIndex = 0; outputIndex < GL4METAL_MAX_COLOR_ATTACHMENTS; ++outputIndex) {
		const GLenum drawBuffer = attachments.draw_buffers[outputIndex];
		if (drawBuffer < GL_COLOR_ATTACHMENT0 ||
			drawBuffer >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS) continue;
		const GLenum attachment = drawBuffer;
		const FBOAttachment &image = attachments.color[attachment - GL_COLOR_ATTACHMENT0];
		if (image.texture == textureName) {
			if (attachmentOut) *attachmentOut = attachment;
			if (levelOut) *levelOut = image.level;
			if (drawBufferOut) *drawBufferOut = outputIndex;
			return true;
		}
	}

	// Unlike color attachments, depth/stencil are installed independently of
	// glDrawBuffers. Only classify them as write targets when their tests and
	// write masks can actually modify the attached image.
	if (ctx.depth_test_enabled && ctx.depth_write_enabled &&
		attachments.depth.texture == textureName) {
		if (attachmentOut) *attachmentOut = GL_DEPTH_ATTACHMENT;
		if (levelOut) *levelOut = attachments.depth.level;
		if (drawBufferOut) *drawBufferOut = -1;
		return true;
	}
	const bool stencilWrites = capOn(GL_STENCIL_TEST) && (
		(ctx.stencil_front.write_mask != 0 &&
		 (ctx.stencil_front.fail != GL_KEEP || ctx.stencil_front.depth_fail != GL_KEEP || ctx.stencil_front.depth_pass != GL_KEEP)) ||
		(ctx.stencil_back.write_mask != 0 &&
		 (ctx.stencil_back.fail != GL_KEEP || ctx.stencil_back.depth_fail != GL_KEEP || ctx.stencil_back.depth_pass != GL_KEEP)));
	if (stencilWrites && attachments.stencil.texture == textureName) {
		if (attachmentOut) *attachmentOut = GL_STENCIL_ATTACHMENT;
		if (levelOut) *levelOut = attachments.stencil.level;
		if (drawBufferOut) *drawBufferOut = -1;
		return true;
	}
	return false;
}

static std::unordered_set<uint64_t> reportedFeedbackLoops;

struct FeedbackSnapshotRequest {
	GLuint textureName = 0;
	GLenum attachment = GL_NONE;
	GLint level = 0;
	GLint drawBufferIndex = -1;
	GLint unit = -1;
	std::string samplerName;
	id<MTLTexture> source = nil;
	id<MTLTexture> snapshot = nil;
};

// Resolve feedback before any per-draw state/resources are installed. A blit encoder
// cannot be interleaved into an active render encoder, so we first collect every read
// alias, end the pass once, copy all aliases in one blit encoder, then resume the same
// framebuffer with LOAD. The caller will bind the returned snapshots on the new encoder.
static bool prepareFeedbackSnapshots(const ProgramObject &program, const char *caller,
	__strong id<MTLRenderCommandEncoder> &encoder,
	std::unordered_map<GLuint, id<MTLTexture>> &snapshotsByTexture) {
	std::vector<FeedbackSnapshotRequest> requests;
	std::unordered_set<GLuint> seenTextures;

	for (const ProgramUniform &uniform : program.uniforms) {
		if (uniform.info.kind != TranslatedUniform::Sampler ||
			(!uniform.info.stage[0].present && !uniform.info.stage[1].present)) continue;
		if (uniform.info.array_size <= 0 || (size_t)uniform.info.array_size > uniform.units.size()) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: sampler '%s' has an invalid unit array; draw skipped", caller, uniform.info.name.c_str());
			return false;
		}
		if (uniform.info.dim == TranslatedUniform::DimBuffer) continue;

		for (int element = 0; element < uniform.info.array_size; ++element) {
			const GLint unit = uniform.units[(size_t)element];
			if (unit < 0 || unit >= GL4METAL_MAX_TEXTURE_UNITS) continue; // validated in the normal binding path
			auto bound = ctx.bound_textures[unit].find(targetForDim(uniform.info.dim));
			if (bound == ctx.bound_textures[unit].end() || bound->second == 0) continue;

			const GLuint textureName = bound->second;
			GLenum attachment = GL_NONE;
			GLint level = -1;
			GLint drawBufferIndex = -1;
			if (!textureIsAttachedToDrawFramebuffer(textureName, &attachment, &level, &drawBufferIndex)) continue;
			if (!seenTextures.insert(textureName).second) continue; // one snapshot per GL texture, even if bound on several units

			// Snapshot ordinary single-sample 2D color images. Copying the complete mip
			// chain below preserves explicit-LOD reads even if the attached level is not 0.
			// Depth/stencil and multisample aliases need a different copy interpretation;
			// never silently substitute a wrong image for those cases.
			if (attachment < GL_COLOR_ATTACHMENT0 ||
				attachment >= GL_COLOR_ATTACHMENT0 + GL4METAL_MAX_COLOR_ATTACHMENTS || level < 0) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: unsupported feedback image (texture=%u attachment=0x%X level=%d); draw skipped",
					caller, textureName, attachment, level);
				return false;
			}
			auto textureObject = ctx.texture_objects.find(textureName);
			if (textureObject == ctx.texture_objects.end() || !textureObject->second.texture) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: feedback texture %u has no Metal storage; draw skipped", caller, textureName);
				return false;
			}
			id<MTLTexture> source = textureObject->second.texture;
			if (source.textureType != MTLTextureType2D || source.sampleCount != 1 ||
				source.storageMode == MTLStorageModeMemoryless ||
				(source.usage & MTLTextureUsageShaderRead) == 0 ||
				source.pixelFormat == MTLPixelFormatInvalid || source.width == 0 || source.height == 0) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("%s: feedback texture %u cannot be snapshotted (type=%lu samples=%lu usage=0x%lx storage=%lu); draw skipped",
					caller, textureName, (unsigned long)source.textureType, (unsigned long)source.sampleCount,
					(unsigned long)source.usage, (unsigned long)source.storageMode);
				return false;
			}

			TextureObject &object = textureObject->second;
			id<MTLTexture> snapshot = object.feedback_snapshot;
			if (!snapshot || snapshot.textureType != MTLTextureType2D ||
				snapshot.width != source.width || snapshot.height != source.height ||
				snapshot.mipmapLevelCount != source.mipmapLevelCount ||
				snapshot.pixelFormat != source.pixelFormat || snapshot.sampleCount != 1 ||
				(snapshot.usage & MTLTextureUsageShaderRead) == 0) {
				MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
					width:source.width height:source.height mipmapped:(source.mipmapLevelCount > 1)];
				descriptor.mipmapLevelCount = source.mipmapLevelCount;
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageShaderRead;
				snapshot = [ctx.device newTextureWithDescriptor:descriptor];
				if (!snapshot) {
					gl4metalSetError(GL_OUT_OF_MEMORY);
					GL4METAL_LOG("%s: failed to allocate %lux%lu feedback snapshot for texture %u (format=%lu); draw skipped",
						caller, (unsigned long)source.width, (unsigned long)source.height, textureName, (unsigned long)source.pixelFormat);
					return false;
				}
				snapshot.label = [NSString stringWithFormat:@"GL4Metal feedback snapshot texture %u", textureName];
				object.feedback_snapshot = snapshot;
			}

			snapshotsByTexture.emplace(textureName, snapshot);
			if (object.feedback_snapshot_pass_serial != ctx.render_pass_serial) {
				FeedbackSnapshotRequest request;
				request.textureName = textureName;
				request.attachment = attachment;
				request.level = level;
				request.drawBufferIndex = drawBufferIndex;
				request.unit = unit;
				request.samplerName = uniform.info.name;
				request.source = source;
				request.snapshot = snapshot;
				requests.push_back(std::move(request));
			}
		}
	}

	if (requests.empty()) return true;
	if (!ctx.currentCommandBuffer) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: feedback snapshot requested without an active Metal command buffer; draw skipped", caller);
		return false;
	}

	// Finish earlier writes to the active render targets before reading them in a blit.
	endRenderEncoder();
	id<MTLBlitCommandEncoder> blit = [ctx.currentCommandBuffer blitCommandEncoder];
	if (!blit) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: failed to create Metal blit encoder for %zu feedback snapshot(s)", caller, requests.size());
		encoder = beginRenderPass(0, true); // restore the same logical pass for subsequent GL commands
		return false;
	}
	for (const FeedbackSnapshotRequest &request : requests) {
		// Copy every mip level, not just the attached level: the shader may issue
		// explicit-LOD samples from other levels of the same GL texture object.
		for (NSUInteger level = 0; level < request.source.mipmapLevelCount; ++level) {
			const NSUInteger width = std::max<NSUInteger>(1, request.source.width >> level);
			const NSUInteger height = std::max<NSUInteger>(1, request.source.height >> level);
			const MTLSize size = MTLSizeMake(width, height, 1);
			[blit copyFromTexture:request.source sourceSlice:0 sourceLevel:level
				sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:size
				toTexture:request.snapshot destinationSlice:0 destinationLevel:level
				destinationOrigin:MTLOriginMake(0, 0, 0)];
		}
	}
	[blit endEncoding];
	for (const FeedbackSnapshotRequest &request : requests) {
		auto textureObject = ctx.texture_objects.find(request.textureName);
		if (textureObject != ctx.texture_objects.end()) {
			textureObject->second.feedback_snapshot_pass_serial = ctx.render_pass_serial;
		}
	}

	// Re-open every selected color/depth attachment with LOAD so the snapshot operation
	// is transparent to subsequent draws in the same GL render pass. No pipeline or
	// per-draw resource bindings have been issued yet at this point.
	encoder = beginRenderPass(0, true);
	if (!encoder) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: failed to resume framebuffer %u after feedback snapshot", caller, ctx.draw_fbo);
		return false;
	}

	for (const FeedbackSnapshotRequest &request : requests) {
		const uint64_t key = ((uint64_t)ctx.draw_fbo << 32) | request.textureName;
		bool firstReport = false;
		{
			std::lock_guard<std::mutex> lock(feedbackLoopSetMutex);
			firstReport = reportedFeedbackLoops.insert(key).second;
		}
		if (firstReport) {
			GL4METAL_LOG("draw: snapshotting feedback image (program=%u framebuffer=%u texture=%u sampler='%s' unit=%d attachment=0x%X level=%d drawBufferIndex=%d snapshot=%p size=%lux%lu format=%lu)",
				ctx.current_program, ctx.draw_fbo, request.textureName, request.samplerName.c_str(), request.unit,
				request.attachment, request.level, request.drawBufferIndex, (__bridge void *)request.snapshot,
				(unsigned long)request.source.width, (unsigned long)request.source.height, (unsigned long)request.source.pixelFormat);
		}
	}
	return true;
}

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
	// Reuse thread-local key storage; samplerFor is called for every active sampler on every draw.
	static thread_local std::string samplerKeyScratch;
	std::string &key = samplerKeyScratch;
	key.clear();
	if (key.capacity() < 128) key.reserve(128);
	// Parameters of the texture, overridden by the sampler object bound to the unit (glBindSampler) when there is one.
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
		case TranslatedUniform::DimBuffer:   return GL_TEXTURE_BUFFER;
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
static id<MTLTexture> fallbackTexture(const TranslatedUniform &uniform) {
	const TranslatedUniform::Dim dim = uniform.dim;
	if (!ctx.device) return nil;
	std::lock_guard<std::mutex> lock(fallbackTextureMutex);
	if (dim == TranslatedUniform::DimBuffer) {
		static id<MTLTexture> bufferFallbacks[3] = { nil, nil, nil }; // float, signed int, unsigned int
		const NSUInteger formatIndex = uniform.sampler_gl_type == 0x8DD0u ? 1 :
			(uniform.sampler_gl_type == 0x8DD8u ? 2 : 0);
		__strong id<MTLTexture> &slot = bufferFallbacks[formatIndex];
		if (!slot) {
			MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
			descriptor.textureType = MTLTextureType2D;
			descriptor.pixelFormat = formatIndex == 1 ? MTLPixelFormatR32Sint :
				(formatIndex == 2 ? MTLPixelFormatR32Uint : MTLPixelFormatR32Float);
			descriptor.width = 4096;
			descriptor.height = 1;
			descriptor.depth = 1;
			descriptor.arrayLength = 1;
			descriptor.usage = MTLTextureUsageShaderRead;
			descriptor.storageMode = MTLStorageModeShared;
			slot = [ctx.device newTextureWithDescriptor:descriptor];
			if (!slot) return nil;
			std::vector<uint32_t> zeroes(4096, 0);
			[slot replaceRegion:MTLRegionMake2D(0, 0, 4096, 1) mipmapLevel:0
				withBytes:zeroes.data() bytesPerRow:4096 * sizeof(uint32_t)];
		}
		return slot;
	}
	const int dimensionIndex = (int)dim;
	if (dimensionIndex < 0 || dimensionIndex >= 4) return nil;
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

// Shared with the compute dispatch path so graphics and compute use the same
// sampler-state cache and the same dimension-correct zero texture fallback.
id<MTLTexture> gl4metalFallbackTextureForUniform(const TranslatedUniform &uniform) {
	return fallbackTexture(uniform);
}

id<MTLSamplerState> gl4metalFallbackSampler(void) {
	return fallbackSampler();
}

id<MTLSamplerState> gl4metalSamplerForTexture(const TextureObject &texture, GLuint textureUnit) {
	return samplerFor(texture, textureUnit);
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
	GLuint source_buffer = 0; // zero when buffer points at an internally converted stream
	NSUInteger slot = 0;
};

static id<MTLRenderPipelineState> pipelineFor(GLuint programId, ProgramObject &p, std::vector<ActiveAttrib> &attribs, const char *caller) {
	bool flip = ctx.pass_flipped;
	bool blend = false;
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment)
		blend = blend || ctx.blend_enabled[attachment];

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

	// Reuse key storage across draws. The unordered_map copies keys on insertion.
	static thread_local std::string pipelineKeyScratch;
	std::string &key = pipelineKeyScratch;
	key.clear();
	if (key.capacity() < 1024) key.reserve(1024);
	auto put = [&key](uint64_t v) { putBytes(key, &v, sizeof(v)); };
	put(programId);
	put(flip ? 1 : 0);
	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		put((uint64_t)ctx.pass_color_format[i]);
	}
	put((uint64_t)ctx.pass_depth_format);
	put((uint64_t)ctx.pass_stencil_format);
	put((uint64_t)ctx.pass_sample_count);
	put(blend ? 1 : 0);
	put(capOn(GL_SAMPLE_ALPHA_TO_COVERAGE) ? 1 : 0);
	put(capOn(GL_SAMPLE_ALPHA_TO_ONE) ? 1 : 0);
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		const BlendBufferState &blendState = ctx.blend_buffers[attachment];
		put(ctx.blend_enabled[attachment] ? 1 : 0);
		put((uint64_t)ctx.color_masks[attachment][0] | ((uint64_t)ctx.color_masks[attachment][1] << 1) |
			((uint64_t)ctx.color_masks[attachment][2] << 2) | ((uint64_t)ctx.color_masks[attachment][3] << 3));
		if (ctx.blend_enabled[attachment]) {
			put(blendState.src_rgb); put(blendState.dst_rgb);
			put(blendState.src_alpha); put(blendState.dst_alpha);
			put(blendState.equation_rgb); put(blendState.equation_alpha);
		}
	}
	for (const ActiveAttrib &aa : attribs) {
		put(aa.pa->msl_location);
		put((uint64_t)aa.format);
		put(aa.slot);
		put(aa.stride);
		put(aa.divisor);
		put(aa.constant ? 1 : 0);
	}

	{
		std::lock_guard<std::mutex> cacheLock(pipelineCacheMutex);
		auto cached = ctx.pipeline_cache.find(key);
		if (cached != ctx.pipeline_cache.end()) {
			return cached->second; // may be nil: a combination that already failed was reported once
		}
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
	d.alphaToCoverageEnabled = capOn(GL_SAMPLE_ALPHA_TO_COVERAGE);
	d.alphaToOneEnabled = capOn(GL_SAMPLE_ALPHA_TO_ONE);

	for (int i = 0; i < GL4METAL_MAX_COLOR_ATTACHMENTS; i++) {
		if (ctx.pass_color_format[i] == MTLPixelFormatInvalid) {
			continue;
		}
		MTLColorWriteMask mask = 0;
		if (ctx.color_masks[i][0]) mask |= MTLColorWriteMaskRed;
		if (ctx.color_masks[i][1]) mask |= MTLColorWriteMaskGreen;
		if (ctx.color_masks[i][2]) mask |= MTLColorWriteMaskBlue;
		if (ctx.color_masks[i][3]) mask |= MTLColorWriteMaskAlpha;
		MTLRenderPipelineColorAttachmentDescriptor *ca = d.colorAttachments[i];
		ca.pixelFormat = ctx.pass_color_format[i];
		ca.writeMask = mask;
		if (ctx.blend_enabled[i]) {
			const BlendBufferState &blendState = ctx.blend_buffers[i];
			ca.blendingEnabled = YES;
			ca.sourceRGBBlendFactor = blendFactor(blendState.src_rgb);
			ca.destinationRGBBlendFactor = blendFactor(blendState.dst_rgb);
			ca.sourceAlphaBlendFactor = blendFactor(blendState.src_alpha);
			ca.destinationAlphaBlendFactor = blendFactor(blendState.dst_alpha);
			ca.rgbBlendOperation = blendOperation(blendState.equation_rgb);
			ca.alphaBlendOperation = blendOperation(blendState.equation_alpha);
		}
	}
	d.depthAttachmentPixelFormat = ctx.pass_depth_format;
	d.stencilAttachmentPixelFormat = ctx.pass_stencil_format;
	d.rasterSampleCount = std::max<NSUInteger>(1, ctx.pass_sample_count);

	NSError *error = nil;
	id<MTLRenderPipelineState> pipeline = [ctx.device newRenderPipelineStateWithDescriptor:d error:&error];
	if (!pipeline) {
		GL4METAL_LOG("%s: pipeline creation failed for program %u: %s", caller, programId,
			error ? error.localizedDescription.UTF8String : "unknown Metal pipeline error");
	}
	// Pipeline compilation can take milliseconds on iOS. Do it outside the shared
	// cache mutex so unrelated variants can compile concurrently on worker threads.
	// A racing thread may have populated the same key; use that canonical entry.
	{
		std::lock_guard<std::mutex> cacheLock(pipelineCacheMutex);
		auto inserted = ctx.pipeline_cache.emplace(key, pipeline);
		if (!inserted.second) return inserted.first->second;
	}
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
			if (capOn(GL_PRIMITIVE_RESTART) && index == (dc.indexType == GL_UNSIGNED_BYTE ? (ctx.primitive_restart_index & 0xFFu) :
				(dc.indexType == GL_UNSIGNED_SHORT ? (ctx.primitive_restart_index & 0xFFFFu) : ctx.primitive_restart_index))) {
				continue;
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
		const NSUInteger visibleLength = aa.source_buffer != 0 ? glBufferLogicalSize(aa.source_buffer) : aa.buffer.length;
		if (aa.elementBytes == 0 || aa.stride == 0 || aa.offset > visibleLength) {
			GL4METAL_LOG("%s: invalid vertex binding for '%s' (offset=%lu stride=%lu element=%lu length=%lu); draw skipped",
				caller, aa.pa ? aa.pa->name.c_str() : "?", (unsigned long)aa.offset,
				(unsigned long)aa.stride, (unsigned long)aa.elementBytes, (unsigned long)visibleLength);
			gl4metalSetError(GL_INVALID_OPERATION);
			return false;
		}
		const uint64_t length = (uint64_t)visibleLength;
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
	if (capOn(GL_RASTERIZER_DISCARD)) return;
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
	if (p.is_compute) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: compute program cannot be used for draw calls", caller);
		return;
	}

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
	// Reuse the active-attribute vector on the render thread; its capacity is stable for
	// a linked program and avoids one heap allocation per GL draw call.
	static thread_local std::vector<ActiveAttrib> attribScratch;
	std::vector<ActiveAttrib> &attribs = attribScratch;
	attribs.clear();
	if (attribs.capacity() < p.attribs.size()) attribs.reserve(p.attribs.size());
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
			aa.source_buffer = va.bound_vbo;
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
					aa.source_buffer = 0;
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

	// Resolve active sampler/attachment aliases before setting any render state. The
	// snapshot helper may end and recreate the encoder, so it must run before pipeline,
	// vertex, uniform, or sampler bindings are recorded on the current encoder.
	std::unordered_map<GLuint, id<MTLTexture>> feedbackSnapshots;
	if (!prepareFeedbackSnapshots(p, caller, enc, feedbackSnapshots)) {
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

	// Per-draw NSString formatting and autorelease traffic are disabled in normal builds.
	// The label remains available as an opt-in diagnostic when investigating GPU faults.
	if (drawLabelingEnabled()) {
		static std::atomic<uint64_t> drawSequence{0};
		const uint64_t sequence = drawSequence.fetch_add(1, std::memory_order_relaxed) + 1;
		NSString *drawLabel = [NSString stringWithFormat:
			@"GL4Metal last-draw#%llu %s program=%u FBO=%u mode=0x%X indexed=%d count=%d instances=%d first=%d baseVertex=%d indexType=0x%X indexOffset=%lld",
			(unsigned long long)sequence, caller, ctx.current_program, ctx.draw_fbo, dc.mode,
			dc.indexed ? 1 : 0, dc.count, dc.instances, dc.first, dc.baseVertex, dc.indexType, (long long)dc.indexOffset];
		if (ctx.currentCommandBuffer) ctx.currentCommandBuffer.label = drawLabel;
		enc.label = drawLabel;
	}
	id<MTLDepthStencilState> activeDepthState =
		((ctx.pass_depth_format != MTLPixelFormatInvalid || ctx.pass_stencil_format != MTLPixelFormatInvalid)
			? ctx.depthStencilState : ctx.no_depth_state);
	if (!activeDepthState) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("%s: no valid Metal depth-stencil state; draw skipped", caller);
		return;
	}
	[enc setRenderPipelineState:pipeline];
	if (p.uses_gl_frag_coord && !ctx.fragcoord_params_bound) {
		struct alignas(8) FragmentCoordFixupParams {
			uint32_t target_height;
			uint32_t flip_y;
		};
		static_assert(sizeof(FragmentCoordFixupParams) == 8, "MSL fragment-coordinate parameter layout changed");
		if (ctx.render_target_height == 0 || ctx.render_target_height > (NSUInteger)UINT32_MAX) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: invalid render-target height for gl_FragCoord conversion (%llu); draw skipped",
				caller, (unsigned long long)ctx.render_target_height);
			return;
		}
		FragmentCoordFixupParams fragCoordParams = {
			(uint32_t)ctx.render_target_height,
			ctx.pass_flipped ? 0u : 1u
		};
		[enc setFragmentBytes:&fragCoordParams length:sizeof(fragCoordParams) atIndex:GL4METAL_FRAGCOORD_BUFFER_INDEX];
		ctx.fragcoord_params_bound = true;
	}
	[enc setDepthStencilState:activeDepthState];
	[enc setStencilFrontReferenceValue:(uint32_t)ctx.stencil_front.ref backReferenceValue:(uint32_t)ctx.stencil_back.ref];
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

	// Record all GL buffers consumed by this draw in one locked batch. CPU updates can
	// rename only in-flight storage without taking a mutex once per vertex attribute/UBO.
	auto markActiveDrawBuffers = [&](GLuint elementBuffer) {
		std::array<GLuint, 48> names{}; // 16 vertex streams + 28 combined-stage UBOs + one EBO
		size_t count = 0;
		auto appendUnique = [&](GLuint draw_id) {
			if (draw_id == 0) return;
			for (size_t index = 0; index < count; ++index) {
				if (names[index] == draw_id) return;
			}
			if (count < names.size()) names[count++] = draw_id;
		};
		for (const ActiveAttrib &aa : attribs) {
			if (aa.constant || !aa.pa || aa.pa->gl_index < 0 || aa.pa->gl_index >= 16) continue;
			appendUnique(vao.attribs[aa.pa->gl_index].bound_vbo);
		}
		for (const ProgramBlock &block : p.blocks) {
			if (block.info.stage[0].present || block.info.stage[1].present) {
				appendUnique(ctx.uniform_buffer_bindings[block.binding_point].buffer);
			}
		}
		appendUnique(elementBuffer);
		markBuffersUsed(names.data(), count);
	};

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
			id<MTLBuffer> buffer = nil;
			NSUInteger offset = 0;
			if (!uploadLargeDefaultUniformBlock(data.data(), (NSUInteger)data.size(), buffer, offset)) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: failed to reserve %lu bytes in the default-uniform upload arena; draw skipped", caller, (unsigned long)data.size());
				return;
			}
			if (s == 0) {
				[enc setVertexBuffer:buffer offset:offset atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			} else {
				[enc setFragmentBuffer:buffer offset:offset atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			}
		}
	}

	// Uniform blocks: never leave an active MSL buffer slot unset or reuse a stale binding.
	// The translator allocates MSL slots 16...29 per stage independently of GL binding points; slot 30 is the default block.
	// MSL UBO slots are a small fixed range (16..29); stack bitsets avoid two
	// unordered_set allocations and hash lookups on every draw.
	std::array<bool, GL4METAL_UNIFORM_BUFFER_INDEX - 16> usedVertexBlockSlots{};
	std::array<bool, GL4METAL_UNIFORM_BUFFER_INDEX - 16> usedFragmentBlockSlots{};
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
		if ((block.info.stage[0].present && usedVertexBlockSlots[block.info.stage[0].slot - 16]) ||
			(block.info.stage[1].present && usedFragmentBlockSlots[block.info.stage[1].slot - 16])) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: multiple uniform blocks collide on a Metal buffer slot; block '%s' draw skipped",
				caller, block.info.name.c_str());
			return;
		}
		if (block.info.stage[0].present) usedVertexBlockSlots[block.info.stage[0].slot - 16] = true;
		if (block.info.stage[1].present) usedFragmentBlockSlots[block.info.stage[1].slot - 16] = true;
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
		const NSUInteger length = glBufferLogicalSize(binding.buffer);
		const NSUInteger offset = (NSUInteger)binding.offset;
		if (offset > length) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' offset %llu exceeds buffer length %llu; draw skipped",
				caller, block.info.name.c_str(), (unsigned long long)offset, (unsigned long long)length);
			return;
		}
		const NSUInteger available = length - offset;
		const NSUInteger boundSize = binding.size == 0 ? available : (binding.size < 0 ? 0 : (NSUInteger)binding.size);
		if (binding.size < 0 || boundSize > available || boundSize < block.info.required_size) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: uniform block '%s' needs %u bytes of member data (std140 block size %u) but binding %u provides %llu bytes (buffer=%u offset=%llu length=%llu); draw skipped",
				caller, block.info.name.c_str(), block.info.required_size, block.info.size, block.binding_point,
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

	// Textures: ordinary samplers bind their GL image; active attachment aliases bind
	// the just-copied snapshot prepared above, never a too-small generic fallback.
	for (const ProgramUniform &u : p.uniforms) {
		if (u.info.kind != TranslatedUniform::Sampler) {
			continue;
		}
		if (u.info.array_size <= 0 || (size_t)u.info.array_size > u.units.size() ||
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
			const bool isBufferSampler = u.info.dim == TranslatedUniform::DimBuffer;
			GLuint textureNameForTrace = 0;
			bool usedFallbackTexture = false;
			if (unit >= 0 && unit < GL4METAL_MAX_TEXTURE_UNITS) {
				auto bound = ctx.bound_textures[unit].find(targetForDim(u.info.dim));
				if (bound != ctx.bound_textures[unit].end() && bound->second != 0) {
					const GLuint textureName = bound->second;
					textureNameForTrace = textureName;
					auto feedbackSnapshot = feedbackSnapshots.find(textureName);
					auto to = ctx.texture_objects.find(textureName);
					if (feedbackSnapshot != feedbackSnapshots.end()) {
						texture = feedbackSnapshot->second;
						if (to != ctx.texture_objects.end() && !isBufferSampler) {
							sampler = samplerFor(to->second, (GLuint)unit);
						}
					} else {
						if (to != ctx.texture_objects.end() && isBufferSampler &&
							to->second.buffer_sampler_type != 0 &&
							to->second.buffer_sampler_type != u.info.sampler_gl_type) {
							gl4metalSetError(GL_INVALID_OPERATION);
							GL4METAL_LOG("%s: buffer texture '%s' format class 0x%X is incompatible with sampler type 0x%X; draw skipped",
								caller, u.info.name.c_str(), to->second.buffer_sampler_type, u.info.sampler_gl_type);
							return;
						}
						if (to != ctx.texture_objects.end() && to->second.texture &&
							to->second.texture.textureType == typeForDim(u.info.dim) &&
							(to->second.texture.usage & MTLTextureUsageShaderRead) != 0) {
							texture = to->second.texture;
							if (isBufferSampler) {
								markBufferUsed(to->second.buffer_object);
							} else {
								sampler = samplerFor(to->second, (GLuint)unit);
							}
						}
					}
				}
			}
			if (!texture) {
				usedFallbackTexture = true;
				texture = fallbackTexture(u.info);
				if (!isBufferSampler) sampler = fallbackSampler();
			}
			// Never encode a shader read against a nil or non-readable Metal resource. A nil
			// texture can surface later as an asynchronous GPU page fault at address zero.
			if (!texture || (texture.usage & MTLTextureUsageShaderRead) == 0) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("%s: sampler '%s' (unit=%d, program=%u) has no valid ShaderRead texture; fallback allocation failed; draw skipped",
					caller, u.info.name.c_str(), unit, ctx.current_program);
				return;
			}
			if (!isBufferSampler && !sampler) {
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
				if (!isBufferSampler) [enc setVertexSamplerState:sampler atIndex:index];
			}
			if (u.info.stage[1].present) {
				NSUInteger index = u.info.stage[1].slot + (NSUInteger)e;
				[enc setFragmentTexture:texture atIndex:index];
				if (!isBufferSampler) [enc setFragmentSamplerState:sampler atIndex:index];
			}
		}
	}

	auto markDefaultFramebufferAttachmentsWritten = [&]() {
		if (!ctx.pass_uses_default_framebuffer || capOn(GL_RASTERIZER_DISCARD)) return;
		if (ctx.pass_color_format[0] != MTLPixelFormatInvalid &&
			(ctx.color_masks[0][0] || ctx.color_masks[0][1] || ctx.color_masks[0][2] || ctx.color_masks[0][3])) {
			ctx.default_color_has_contents = true;
		}
		if (ctx.depth_test_enabled && ctx.depth_write_enabled) ctx.default_depth_has_contents = true;
		auto stencilTest = ctx.caps.find(GL_STENCIL_TEST);
		if (stencilTest != ctx.caps.end() && stencilTest->second &&
			(ctx.stencil_front.write_mask != 0 || ctx.stencil_back.write_mask != 0)) {
			ctx.default_stencil_has_contents = true;
		}
	};

	// The draw itself.
	if (!dc.indexed) {
		if (!validateVertexFetchRanges(dc, caller, attribs, nil, 0)) return;
		if (!fan) {
			markActiveDrawBuffers(0);
			[enc drawPrimitives:prim vertexStart:(NSUInteger)dc.first vertexCount:(NSUInteger)dc.count
				  instanceCount:(NSUInteger)dc.instances baseInstance:0];
			markDefaultFramebufferAttachmentsWritten();
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
		markActiveDrawBuffers(0);
		[enc drawIndexedPrimitives:prim indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:tmp indexBufferOffset:0
					 instanceCount:(NSUInteger)dc.instances baseVertex:0 baseInstance:0];
		markDefaultFramebufferAttachmentsWritten();
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
	const NSUInteger eboLogicalLength = glBufferLogicalSize(vao.element_array_buffer_id);
	if (dc.indexOffset < 0 || (NSUInteger)dc.indexOffset > eboLogicalLength ||
		(NSUInteger)dc.count > (eboLogicalLength - (NSUInteger)dc.indexOffset) / indexSize) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: %d indices at offset %ld do not fit the element buffer (%lu bytes)", caller, dc.count, (long)dc.indexOffset, (unsigned long)eboLogicalLength);
		return;
	}
	if (((NSUInteger)dc.indexOffset % indexSize) != 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: index offset %ld is not aligned to %lu bytes; draw skipped", caller, (long)dc.indexOffset, (unsigned long)indexSize);
		return;
	}
	if (!validateVertexFetchRanges(dc, caller, attribs, ebo, indexSize)) return;

	const bool primitiveRestart = capOn(GL_PRIMITIVE_RESTART);
	if (dc.indexType != GL_UNSIGNED_BYTE && !fan && !primitiveRestart) {
		markActiveDrawBuffers(vao.element_array_buffer_id);
		[enc drawIndexedPrimitives:prim indexCount:(NSUInteger)dc.count
						 indexType:(dc.indexType == GL_UNSIGNED_INT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16)
					   indexBuffer:ebo indexBufferOffset:(NSUInteger)dc.indexOffset
					 instanceCount:(NSUInteger)dc.instances baseVertex:dc.baseVertex baseInstance:0];
		markDefaultFramebufferAttachmentsWritten();
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
	std::vector<std::pair<size_t, size_t>> segments;
	try {
		indices.reserve((size_t)outputIndexCount);
		if (fan && primitiveRestart) {
			const uint32_t restartIndex = dc.indexType == GL_UNSIGNED_BYTE ? (ctx.primitive_restart_index & 0xFFu) :
				(dc.indexType == GL_UNSIGNED_SHORT ? (ctx.primitive_restart_index & 0xFFFFu) : ctx.primitive_restart_index);
			std::vector<uint32_t> fanSegment;
			fanSegment.reserve((size_t)dc.count);
			for (GLsizei i = 0; i <= dc.count; ++i) {
				const bool atEnd = i == dc.count;
				const uint32_t currentIndex = atEnd ? restartIndex : indexAt(i);
				if (atEnd || currentIndex == restartIndex) {
					for (size_t fanIndex = 1; fanIndex + 1 < fanSegment.size(); ++fanIndex) {
						indices.push_back(fanSegment[0]);
						indices.push_back(fanSegment[fanIndex]);
						indices.push_back(fanSegment[fanIndex + 1]);
					}
					fanSegment.clear();
				} else {
					fanSegment.push_back(currentIndex);
				}
			}
		} else if (fan) {
			for (GLsizei i = 1; i + 1 < dc.count; i++) {
				indices.push_back(indexAt(0));
				indices.push_back(indexAt(i));
				indices.push_back(indexAt(i + 1));
			}
		} else if (primitiveRestart) {
			const uint32_t restartIndex = dc.indexType == GL_UNSIGNED_BYTE ? (ctx.primitive_restart_index & 0xFFu) :
				(dc.indexType == GL_UNSIGNED_SHORT ? (ctx.primitive_restart_index & 0xFFFFu) : ctx.primitive_restart_index);
			size_t segmentStart = 0;
			for (GLsizei i = 0; i < dc.count; ++i) {
				const uint32_t currentIndex = indexAt(i);
				if (currentIndex == restartIndex) {
					if (indices.size() > segmentStart) segments.emplace_back(segmentStart, indices.size() - segmentStart);
					segmentStart = indices.size();
				} else {
					indices.push_back(currentIndex);
				}
			}
			if (indices.size() > segmentStart) segments.emplace_back(segmentStart, indices.size() - segmentStart);
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
	markActiveDrawBuffers(vao.element_array_buffer_id);
	if (primitiveRestart && !fan) {
		for (const auto &segment : segments) {
			if (segment.second == 0) continue;
			[enc drawIndexedPrimitives:prim indexCount:segment.second indexType:MTLIndexTypeUInt32 indexBuffer:tmp
				indexBufferOffset:segment.first * sizeof(uint32_t) instanceCount:(NSUInteger)dc.instances
				baseVertex:dc.baseVertex baseInstance:0];
		}
	} else {
		[enc drawIndexedPrimitives:prim indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:tmp indexBufferOffset:0
				 instanceCount:(NSUInteger)dc.instances baseVertex:dc.baseVertex baseInstance:0];
	}
	markDefaultFramebufferAttachmentsWritten();
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
