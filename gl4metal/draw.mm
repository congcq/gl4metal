#import "draw.h"

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

static id<MTLSamplerState> samplerFor(const TextureObject &t, GLuint samplerName) {
	// Parameters of the texture, overridden by the sampler object bound to the unit (glBindSampler) when there is one.
	GLenum minFilter = t.min_filter, magFilter = t.mag_filter;
	GLenum wrapS = t.wrap_s, wrapT = t.wrap_t, wrapR = t.wrap_r;
	GLfloat minLod = t.min_lod, maxLod = t.max_lod, anisotropy = t.max_anisotropy;
	GLenum compareMode = t.compare_mode, compareFunc = t.compare_func;
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
	ctx.sampler_cache[key] = sampler;
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
	static id<MTLTexture> textures[4] = { nil, nil, nil, nil };
	__strong id<MTLTexture> &slot = textures[(int)dim];
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
		const uint8_t black[4] = { 0, 0, 0, 255 };
		NSUInteger slices = dim == TranslatedUniform::DimCube ? 6 : 1;
		for (NSUInteger s = 0; s < slices; s++) {
			[slot replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:s withBytes:black bytesPerRow:4 bytesPerImage:4];
		}
	}
	return slot;
}

static id<MTLSamplerState> fallbackSampler() {
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
	GLuint divisor = 0;
	NSUInteger slot = 0;
};

static id<MTLRenderPipelineState> pipelineFor(GLuint programId, ProgramObject &p, std::vector<ActiveAttrib> &attribs, const char *caller) {
	bool flip = ctx.pass_flipped;
	bool blend = capOn(GL_BLEND);

	NSUInteger slot = 0;
	for (ActiveAttrib &aa : attribs) {
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
			ld.stride = 0;
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
		GL4METAL_LOG("%s: pipeline creation failed for program %u: %s", caller, programId, error.localizedDescription.UTF8String);
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

static void executeDraw(const DrawCall &dc, const char *caller) {
	if (dc.count <= 0 || dc.instances <= 0) {
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
			aa.divisor = va.divisor;
		} else {
			aa.constant = true;
			aa.format = pa.is_integer ? MTLVertexFormatInt4 : MTLVertexFormatFloat4;
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

	[enc setRenderPipelineState:pipeline];
	[enc setDepthStencilState:(ctx.pass_depth_format != MTLPixelFormatInvalid ? ctx.depthStencilState : ctx.no_depth_state)];
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
			id<MTLBuffer> buffer = [ctx.device newBufferWithBytes:data.data() length:data.size() options:MTLResourceStorageModeShared];
			if (s == 0) {
				[enc setVertexBuffer:buffer offset:0 atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			} else {
				[enc setFragmentBuffer:buffer offset:0 atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
			}
		}
	}

	// Uniform blocks: GL binding point -> buffer range, at the Metal slot the translator gave the block.
	for (const ProgramBlock &block : p.blocks) {
		if (block.binding_point >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
			continue;
		}
		const UniformBufferBinding &binding = ctx.uniform_buffer_bindings[block.binding_point];
		auto bound = ctx.buffer_objects.find(binding.buffer);
		if (binding.buffer == 0 || bound == ctx.buffer_objects.end() || !bound->second) {
			continue;
		}
		if (block.info.stage[0].present) {
			[enc setVertexBuffer:bound->second offset:(NSUInteger)binding.offset atIndex:block.info.stage[0].slot];
		}
		if (block.info.stage[1].present) {
			[enc setFragmentBuffer:bound->second offset:(NSUInteger)binding.offset atIndex:block.info.stage[1].slot];
		}
	}

	// Textures: the sampler uniform holds a texture unit, the unit holds a texture.
	for (const ProgramUniform &u : p.uniforms) {
		if (u.info.kind != TranslatedUniform::Sampler) {
			continue;
		}
		for (int e = 0; e < u.info.array_size; e++) {
			GLint unit = u.units[(size_t)e];
			id<MTLTexture> texture = nil;
			id<MTLSamplerState> sampler = nil;
			if (unit >= 0 && unit < GL4METAL_MAX_TEXTURE_UNITS) {
				auto bound = ctx.bound_textures[unit].find(targetForDim(u.info.dim));
				if (bound != ctx.bound_textures[unit].end() && bound->second != 0) {
					auto to = ctx.texture_objects.find(bound->second);
					if (to != ctx.texture_objects.end() && to->second.texture && to->second.texture.textureType == typeForDim(u.info.dim)) {
						texture = to->second.texture;
						sampler = samplerFor(to->second, ctx.bound_samplers[unit]);
					}
				}
			}
			if (!texture) {
				texture = fallbackTexture(u.info.dim);
				sampler = fallbackSampler();
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

	// The draw itself.
	if (!dc.indexed) {
		if (!fan) {
			[enc drawPrimitives:prim vertexStart:(NSUInteger)dc.first vertexCount:(NSUInteger)dc.count
				  instanceCount:(NSUInteger)dc.instances baseInstance:0];
			return;
		}
		std::vector<uint32_t> indices;
		for (GLsizei i = 1; i + 1 < dc.count; i++) {
			indices.push_back((uint32_t)dc.first);
			indices.push_back((uint32_t)(dc.first + i));
			indices.push_back((uint32_t)(dc.first + i + 1));
		}
		if (indices.empty()) {
			return;
		}
		id<MTLBuffer> tmp = [ctx.device newBufferWithBytes:indices.data() length:indices.size() * 4 options:MTLResourceStorageModeShared];
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
	if (dc.indexOffset < 0 || (NSUInteger)dc.indexOffset + (NSUInteger)dc.count * indexSize > ebo.length) {
		GL4METAL_LOG("%s: %d indices at offset %ld do not fit the element buffer (%lu bytes)", caller, dc.count, (long)dc.indexOffset, (unsigned long)ebo.length);
		return;
	}

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
	std::vector<uint32_t> indices;
	if (fan) {
		for (GLsizei i = 1; i + 1 < dc.count; i++) {
			indices.push_back(indexAt(0));
			indices.push_back(indexAt(i));
			indices.push_back(indexAt(i + 1));
		}
	} else {
		for (GLsizei i = 0; i < dc.count; i++) {
			indices.push_back(indexAt(i));
		}
	}
	if (indices.empty()) {
		return;
	}
	id<MTLBuffer> tmp = [ctx.device newBufferWithBytes:indices.data() length:indices.size() * 4 options:MTLResourceStorageModeShared];
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
