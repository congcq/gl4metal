#import "program.h"
#import "shaders.h"
#import "draw.h"
#import "sync.h"

#include <mutex>

// MARK: - Helpers

static ProgramObject *findProgram(GLuint program) {
	auto it = ctx.program_objects.find(program);
	return it == ctx.program_objects.end() ? nullptr : &it->second;
}

static void destroyProgram(GLuint program) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end()) {
		return;
	}
	std::vector<GLuint> shaders = it->second.shaders;
	ctx.program_objects.erase(it);
	for (GLuint shader : shaders) {
		releaseShaderIfUnused(shader); // shaders deleted while attached go away with their last program
	}
}

static bool dumpEnabled() {
	const char *dump = getenv("GL4METAL_DUMP_SHADERS");
	return dump && dump[0] == '1';
}

// Compute pipelines are expensive to compile and independent of GL object names.
// Cache by the complete generated MSL source so shader-pack reloads can reuse them.
static id<MTLComputePipelineState> computePipelineForMSL(const std::string &mslSource, std::string &errorText) {
	if (!ctx.device) {
		errorText = "Metal device is not initialized";
		return nil;
	}
	static std::mutex cacheMutex;
	static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
	{
		std::lock_guard<std::mutex> lock(cacheMutex);
		auto cached = cache.find(mslSource);
		if (cached != cache.end()) return cached->second;
	}

	MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
	options.languageVersion = MTLLanguageVersion2_2;
	NSError *libraryError = nil;
	NSString *source = [NSString stringWithUTF8String:mslSource.c_str()];
	id<MTLLibrary> library = source ? [ctx.device newLibraryWithSource:source options:options error:&libraryError] : nil;
	id<MTLFunction> computeFunction = [library newFunctionWithName:@"main0"];
	if (!computeFunction) {
		errorText = libraryError ? (libraryError.localizedDescription.UTF8String ?: "Metal library compilation failed") : "no main0 compute entry point";
		return nil;
	}
	NSError *pipelineError = nil;
	id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:computeFunction error:&pipelineError];
	if (!pipeline) {
		errorText = pipelineError ? (pipelineError.localizedDescription.UTF8String ?: "Metal compute pipeline creation failed") : "unknown Metal compute pipeline error";
		return nil;
	}
	{
		std::lock_guard<std::mutex> lock(cacheMutex);
		auto inserted = cache.emplace(mslSource, pipeline);
		if (!inserted.second) return inserted.first->second;
		if (cache.size() > 64) {
			auto victim = cache.begin();
			if (victim == inserted.first) ++victim;
			if (victim != cache.end()) cache.erase(victim);
		}
	}
	return pipeline;
}

// Active-uniform type mapping for GL reflection. The runtime uniform-location table
// remains limited to default-block uniforms; named-block members are introspection-only.
static GLenum activeUniformGLType(const TranslatedUniform &u) {
	if (u.kind == TranslatedUniform::Sampler) {
		if (u.sampler_gl_type != 0) return (GLenum)u.sampler_gl_type;
		switch (u.dim) {
			case TranslatedUniform::DimCube: return GL_SAMPLER_CUBE;
			case TranslatedUniform::Dim3D: return GL_SAMPLER_3D;
			case TranslatedUniform::Dim2DArray: return GL_SAMPLER_2D_ARRAY;
			default: return GL_SAMPLER_2D;
		}
	}
	const int rows = std::max(1, std::min(u.vecsize, 4));
	if (u.columns > 1) {
		if (u.columns == rows) {
			static const GLenum squareTypes[] = { GL_FLOAT_MAT2, GL_FLOAT_MAT3, GL_FLOAT_MAT4 };
			return (u.columns >= 2 && u.columns <= 4) ? squareTypes[u.columns - 2] : GL_FLOAT_MAT4;
		}
		static const GLenum matrixTypes[4][4] = {
			{ 0, 0, 0, 0 },
			{ 0, 0, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4 },
			{ 0, GL_FLOAT_MAT3x2, 0, GL_FLOAT_MAT3x4 },
			{ 0, GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, 0 }
		};
		if (u.columns >= 2 && u.columns <= 4 && rows >= 2 && rows <= 4)
			return matrixTypes[u.columns][rows];
	}
	static const GLenum floatTypes[] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum intTypes[] = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	static const GLenum uintTypes[] = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	static const GLenum boolTypes[] = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
	const size_t componentIndex = (size_t)rows - 1;
	switch (u.kind) {
		case TranslatedUniform::Int: return intTypes[componentIndex];
		case TranslatedUniform::UInt: return uintTypes[componentIndex];
		case TranslatedUniform::Bool: return boolTypes[componentIndex];
		default: return floatTypes[componentIndex];
	}
}

static std::string activeUniformName(const TranslatedUniform &u) {
	return u.array_size > 1 ? u.name + "[0]" : u.name;
}

static bool linkComputeProgram(GLuint programName, ProgramObject &program, const ShaderObject &computeShader) {
	TranslatedProgram translated;
	if (!translateComputeProgram(computeShader.source, translated)) {
		program.info_log = translated.log;
		GL4METAL_LOG("glLinkProgram(%u): compute translation failed:\n%s", programName, translated.log.c_str());
		return false;
	}
	if (dumpEnabled()) {
		if (!translated.log.empty()) GL4METAL_LOG("compute program %u translation notes:\n%s", programName, translated.log.c_str());
		GL4METAL_LOG("---- MSL compute, program %u ----\n%s\n---- end MSL ----", programName, translated.msl[2].c_str());
	}

	std::string pipelineError;
	id<MTLComputePipelineState> computePipeline = computePipelineForMSL(translated.msl[2], pipelineError);
	if (!computePipeline) {
		program.info_log = "compute Metal pipeline compilation failed:\n" + pipelineError + "\n";
		GL4METAL_LOG("glLinkProgram(%u): %s", programName, program.info_log.c_str());
		return false;
	}

	program.compute_pipeline = computePipeline;
	program.compute_workgroup_size[0] = translated.workgroup_size[0];
	program.compute_workgroup_size[1] = translated.workgroup_size[1];
	program.compute_workgroup_size[2] = translated.workgroup_size[2];
	program.storage_buffers = translated.storage_buffers;
	program.storage_images = translated.storage_images;
	program.uniforms.reserve(translated.uniforms.size());
	GLint nextLocation = 0;
	for (const TranslatedUniform &uniformInfo : translated.uniforms) {
		ProgramUniform uniform;
		uniform.info = uniformInfo;
		uniform.base_location = nextLocation;
		uniform.units.assign((size_t)uniformInfo.array_size, 0);
		for (int element = 0; element < uniformInfo.array_size; ++element) {
			program.locations.emplace_back((GLint)program.uniforms.size(), element);
		}
		nextLocation += uniformInfo.array_size;
		program.uniforms.push_back(std::move(uniform));
	}
	program.uniform_data[2].assign(translated.block_size[2], 0);
	for (const TranslatedBlock &blockInfo : translated.blocks) {
		ProgramBlock block;
		block.info = blockInfo;
		block.binding_point = blockInfo.binding;
		program.blocks.push_back(std::move(block));
	}
	for (const ProgramUniform &uniform : program.uniforms) {
		ActiveUniformInfo active;
		active.name = activeUniformName(uniform.info);
		active.type = activeUniformGLType(uniform.info);
		active.size = uniform.info.array_size;
		active.block_index = -1;
		const TranslatedUniform::Stage &stage = uniform.info.stage[2];
		active.offset = (GLint)stage.offset;
		active.array_stride = (GLint)stage.array_stride;
		active.matrix_stride = (GLint)stage.matrix_stride;
		active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
		program.active_uniforms.push_back(std::move(active));
	}
	for (size_t blockIndex = 0; blockIndex < program.blocks.size(); ++blockIndex) {
		ProgramBlock &block = program.blocks[blockIndex];
		for (const TranslatedUniform &member : block.info.members) {
			ActiveUniformInfo active;
			active.name = activeUniformName(member);
			active.type = activeUniformGLType(member);
			active.size = member.array_size;
			active.block_index = (GLint)blockIndex;
			const TranslatedUniform::Stage &stage = member.stage[2];
			active.offset = (GLint)stage.offset;
			active.array_stride = (GLint)stage.array_stride;
			active.matrix_stride = (GLint)stage.matrix_stride;
			active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
			block.active_uniform_indices.push_back((GLuint)program.active_uniforms.size());
			program.active_uniforms.push_back(std::move(active));
		}
	}
	program.is_compute = true;
	program.linked = GL_TRUE;
	return true;
}

// MARK: - Compute resource binding

static GLenum computeTargetForDim(TranslatedUniform::Dim dimension) {
	switch (dimension) {
		case TranslatedUniform::DimCube: return GL_TEXTURE_CUBE_MAP;
		case TranslatedUniform::Dim3D: return GL_TEXTURE_3D;
		case TranslatedUniform::Dim2DArray: return GL_TEXTURE_2D_ARRAY;
		case TranslatedUniform::DimBuffer: return GL_TEXTURE_BUFFER;
		default: return GL_TEXTURE_2D;
	}
}

static MTLTextureType computeTextureTypeForDim(TranslatedUniform::Dim dimension) {
	switch (dimension) {
		case TranslatedUniform::DimCube: return MTLTextureTypeCube;
		case TranslatedUniform::Dim3D: return MTLTextureType3D;
		case TranslatedUniform::Dim2DArray: return MTLTextureType2DArray;
		default: return MTLTextureType2D;
	}
}

static bool resolveComputeBuffer(GLuint buffer_id, GLintptr offset, GLsizeiptr boundSize,
	uint32_t requiredSize, bool runtimeSized, __strong id<MTLBuffer> &metalBuffer,
	NSUInteger &metalOffset, NSUInteger &availableSize) {
	auto bufferFound = ctx.buffer_objects.find(buffer_id);
	if (buffer_id == 0 || bufferFound == ctx.buffer_objects.end() || !bufferFound->second || offset < 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	const NSUInteger logicalSize = glBufferLogicalSize(buffer_id);
	const NSUInteger start = (NSUInteger)offset;
	if (start > logicalSize) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	const NSUInteger remaining = logicalSize - start;
	const NSUInteger requested = boundSize > 0 ? (NSUInteger)boundSize : remaining;
	if (requested > remaining || requested < (NSUInteger)requiredSize ||
		(runtimeSized && requested != remaining) || start > bufferFound->second.length ||
		requested > bufferFound->second.length - start) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	metalBuffer = bufferFound->second;
	metalOffset = start;
	availableSize = requested;
	return true;
}

struct ComputeBufferBinding {
	id<MTLBuffer> buffer = nil;
	NSUInteger offset = 0;
	NSUInteger size = 0;
	NSUInteger slot = 0;
};

struct ComputeTextureBinding {
	id<MTLTexture> texture = nil;
	id<MTLSamplerState> sampler = nil;
	NSUInteger slot = 0;
	bool usesSampler = false;
};

static void addUniqueBufferName(std::vector<GLuint> &bufferNames, GLuint buffer_id) {
	if (buffer_id == 0) return;
	if (std::find(bufferNames.begin(), bufferNames.end(), buffer_id) == bufferNames.end()) {
		bufferNames.push_back(buffer_id);
	}
}

// MARK: - Compute entry points

void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer,
	GLenum access, GLenum format) {
	if (unit >= GL4METAL_MAX_TEXTURE_UNITS) {
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
	if (level != 0 || layered != GL_FALSE || layer != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindImageTexture: only non-layered 2D level 0 images are supported on iOS");
		return;
	}
	auto textureFound = ctx.texture_objects.find(texture);
	if (textureFound == ctx.texture_objects.end() || textureFound->second.target != GL_TEXTURE_2D ||
		!textureFound->second.texture || textureFound->second.width <= 0 || textureFound->second.height <= 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindImageTexture: texture %u is not an allocated GL_TEXTURE_2D", texture);
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

void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
	ProgramObject *program = findProgram(ctx.current_program);
	if (!program || !program->linked || !program->is_compute || !program->compute_pipeline) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchCompute: current program is not a linked compute program");
		return;
	}
	if (num_groups_x > 65535 || num_groups_y > 65535 || num_groups_z > 65535) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDispatchCompute: workgroup count exceeds the iOS limit of 65535 per dimension");
		return;
	}
	if (num_groups_x == 0 || num_groups_y == 0 || num_groups_z == 0) return;

	const uint64_t localThreadCount = (uint64_t)program->compute_workgroup_size[0] *
		(uint64_t)program->compute_workgroup_size[1] * (uint64_t)program->compute_workgroup_size[2];
	if (localThreadCount == 0 || localThreadCount > GL4METAL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS ||
		localThreadCount > program->compute_pipeline.maxTotalThreadsPerThreadgroup) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchCompute: local workgroup size (%u,%u,%u) exceeds GL4Metal/iOS limit (%lu)",
			program->compute_workgroup_size[0], program->compute_workgroup_size[1],
			program->compute_workgroup_size[2], (unsigned long)std::min<NSUInteger>(
				program->compute_pipeline.maxTotalThreadsPerThreadgroup, GL4METAL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS));
		return;
	}

	// Keep the per-dispatch scratch capacity on the GL submission thread. Compute-heavy
	// frames often dispatch the same small set of kernels repeatedly; reusing these
	// vectors avoids allocator churn without retaining resources past the next dispatch.
	struct DispatchScratch {
		std::vector<ComputeBufferBinding> storageBuffers;
		std::vector<ComputeBufferBinding> uniformBuffers;
		std::vector<ComputeTextureBinding> sampledTextures;
		std::vector<ComputeTextureBinding> imageTextures;
		std::vector<GLuint> usedBuffers;
	};
	static thread_local DispatchScratch scratch;
	scratch.storageBuffers.clear();
	scratch.uniformBuffers.clear();
	scratch.sampledTextures.clear();
	scratch.imageTextures.clear();
	scratch.usedBuffers.clear();
	auto &storageBufferBindings = scratch.storageBuffers;
	auto &uniformBufferBindings = scratch.uniformBuffers;
	auto &sampledTextureBindings = scratch.sampledTextures;
	auto &imageTextureBindings = scratch.imageTextures;
	auto &usedBufferNames = scratch.usedBuffers;
	storageBufferBindings.reserve(program->storage_buffers.size());
	uniformBufferBindings.reserve(program->blocks.size());
	sampledTextureBindings.reserve(program->uniforms.size());
	imageTextureBindings.reserve(program->storage_images.size());
	usedBufferNames.reserve(program->storage_buffers.size() + program->blocks.size() + program->uniforms.size());

	// Resolve every buffer and check its logical GL range before creating an encoder.
	for (const TranslatedStorageBuffer &storageInfo : program->storage_buffers) {
		if (storageInfo.binding >= GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS || storageInfo.slot >= 16) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: invalid SSBO binding/Metal slot for '%s'", storageInfo.name.c_str());
			return;
		}
		const ShaderStorageBufferBinding &binding = ctx.shader_storage_buffer_bindings[storageInfo.binding];
		ComputeBufferBinding resolved;
		if (!resolveComputeBuffer(binding.buffer, binding.offset, binding.size, storageInfo.required_size,
			storageInfo.runtime_sized, resolved.buffer, resolved.offset, resolved.size)) {
			GL4METAL_LOG("glDispatchCompute: SSBO '%s' has no compatible buffer range (binding=%u)",
				storageInfo.name.c_str(), storageInfo.binding);
			return;
		}
		resolved.slot = storageInfo.slot;
		storageBufferBindings.push_back(resolved);
		addUniqueBufferName(usedBufferNames, binding.buffer);
	}

	// UBO indices map into Metal buffer slots 16...29; loose uniforms use slot 30.
	for (const ProgramBlock &block : program->blocks) {
		const TranslatedBlock::Stage &stage = block.info.stage[2];
		if (!stage.present) continue;
		if (block.binding_point >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS || stage.slot < 16 ||
			stage.slot >= GL4METAL_UNIFORM_BUFFER_INDEX) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: invalid UBO binding/Metal slot for '%s'", block.info.name.c_str());
			return;
		}
		const UniformBufferBinding &binding = ctx.uniform_buffer_bindings[block.binding_point];
		ComputeBufferBinding resolved;
		if (!resolveComputeBuffer(binding.buffer, binding.offset, binding.size, block.info.required_size,
			false, resolved.buffer, resolved.offset, resolved.size)) {
			GL4METAL_LOG("glDispatchCompute: UBO '%s' is missing or smaller than its reflected layout (binding=%u size=%u)",
				block.info.name.c_str(), block.binding_point, block.info.required_size);
			return;
		}
		resolved.slot = stage.slot;
		uniformBufferBindings.push_back(resolved);
		addUniqueBufferName(usedBufferNames, binding.buffer);
	}

	// Resolve regular samplers, including texture-buffer views, using the same fallback
	// and sampler-object caches as the graphics path.
	for (const ProgramUniform &uniform : program->uniforms) {
		const TranslatedUniform &uniformInfo = uniform.info;
		const TranslatedUniform::Stage &stage = uniformInfo.stage[2];
		if (uniformInfo.kind != TranslatedUniform::Sampler || !stage.present) continue;
		const bool isBufferSampler = uniformInfo.dim == TranslatedUniform::DimBuffer;
		const GLenum target = computeTargetForDim(uniformInfo.dim);
		const MTLTextureType expectedType = computeTextureTypeForDim(uniformInfo.dim);
		for (int element = 0; element < uniformInfo.array_size; ++element) {
			const NSUInteger metalSlot = (NSUInteger)stage.slot + (NSUInteger)element;
			if (metalSlot >= 32) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("glDispatchCompute: sampler '%s' exceeds the 32 Metal texture slots", uniformInfo.name.c_str());
				return;
			}
			const GLint textureUnit = uniform.units[(size_t)element];
			if (textureUnit < 0 || textureUnit >= GL4METAL_MAX_TEXTURE_UNITS) {
				gl4metalSetError(GL_INVALID_VALUE);
				GL4METAL_LOG("glDispatchCompute: sampler '%s' references texture unit %d out of range", uniformInfo.name.c_str(), textureUnit);
				return;
			}
			id<MTLTexture> texture = nil;
			id<MTLSamplerState> sampler = nil;
			auto boundTexture = ctx.bound_textures[(GLuint)textureUnit].find(target);
			if (boundTexture != ctx.bound_textures[(GLuint)textureUnit].end() && boundTexture->second != 0) {
				auto textureObject = ctx.texture_objects.find(boundTexture->second);
				if (textureObject != ctx.texture_objects.end()) {
					const TextureObject &object = textureObject->second;
					if (isBufferSampler && object.buffer_sampler_type != 0 &&
						object.buffer_sampler_type != uniformInfo.sampler_gl_type) {
						gl4metalSetError(GL_INVALID_OPERATION);
						GL4METAL_LOG("glDispatchCompute: buffer texture '%s' has an incompatible sampler type", uniformInfo.name.c_str());
						return;
					}
					if (object.texture && object.texture.textureType == expectedType &&
						(object.texture.usage & MTLTextureUsageShaderRead) != 0) {
						texture = object.texture;
						if (isBufferSampler) addUniqueBufferName(usedBufferNames, object.buffer_object);
						else sampler = gl4metalSamplerForTexture(object, (GLuint)textureUnit);
					}
				}
			}
			if (!texture) {
				texture = gl4metalFallbackTextureForUniform(uniformInfo);
				if (!isBufferSampler) sampler = gl4metalFallbackSampler();
			}
			if (!texture || (texture.usage & MTLTextureUsageShaderRead) == 0 || (!isBufferSampler && !sampler)) {
				gl4metalSetError(GL_OUT_OF_MEMORY);
				GL4METAL_LOG("glDispatchCompute: no usable texture/sampler for '%s' (unit=%d)", uniformInfo.name.c_str(), textureUnit);
				return;
			}
			ComputeTextureBinding resolved;
			resolved.texture = texture;
			resolved.sampler = sampler;
			resolved.slot = metalSlot;
			resolved.usesSampler = !isBufferSampler;
			sampledTextureBindings.push_back(resolved);
		}
	}

	// Image binding points are GL image units; Metal texture slots are remapped
	// independently by the translator after all sampled textures.
	for (const TranslatedImage &imageInfo : program->storage_images) {
		if (imageInfo.binding >= GL4METAL_MAX_TEXTURE_UNITS || imageInfo.slot >= 32) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		const ImageUnitBinding &binding = ctx.image_unit_bindings[imageInfo.binding];
		auto textureObject = ctx.texture_objects.find(binding.texture);
		if (binding.texture == 0 || textureObject == ctx.texture_objects.end() ||
			textureObject->second.target != GL_TEXTURE_2D || !textureObject->second.texture ||
			textureObject->second.texture.textureType != MTLTextureType2D || binding.level != 0 ||
			binding.layered != GL_FALSE || binding.layer != 0 || binding.format != textureObject->second.internalformat ||
			(imageInfo.format != 0 && binding.format != imageInfo.format)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: storage image '%s' has no compatible image2D binding on unit %u",
				imageInfo.name.c_str(), imageInfo.binding);
			return;
		}
		if ((imageInfo.read_only && binding.access == GL_WRITE_ONLY) ||
			(imageInfo.write_only && binding.access == GL_READ_ONLY)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: image access for '%s' is incompatible with glBindImageTexture", imageInfo.name.c_str());
			return;
		}
		const MTLTextureUsage usage = textureObject->second.texture.usage;
		if ((!imageInfo.read_only && (usage & MTLTextureUsageShaderWrite) == 0) ||
			(!imageInfo.write_only && (usage & MTLTextureUsageShaderRead) == 0)) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: image '%s' does not have the required Metal ShaderRead/ShaderWrite usage", imageInfo.name.c_str());
			return;
		}
		ComputeTextureBinding resolved;
		resolved.texture = textureObject->second.texture;
		resolved.slot = imageInfo.slot;
		imageTextureBindings.push_back(resolved);
	}

	// Close render work before compute work. Encoders remain asynchronous in the same
	// command buffer, allowing Metal's hazard tracking to order dependent resources.
	endRenderEncoder();
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
	}
	if (!ctx.currentCommandBuffer && ctx.commandQueue) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
	}
	if (!ctx.currentCommandBuffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	id<MTLComputeCommandEncoder> computeEncoder = [ctx.currentCommandBuffer computeCommandEncoder];
	if (!computeEncoder) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glDispatchCompute: Metal could not create a compute command encoder");
		return;
	}
	ctx.currentComputeEncoder = computeEncoder;
	[computeEncoder setComputePipelineState:program->compute_pipeline];
	for (const ComputeBufferBinding &binding : storageBufferBindings) {
		[computeEncoder setBuffer:binding.buffer offset:binding.offset atIndex:binding.slot];
	}
	for (const ComputeBufferBinding &binding : uniformBufferBindings) {
		[computeEncoder setBuffer:binding.buffer offset:binding.offset atIndex:binding.slot];
	}
	if (!program->uniform_data[2].empty()) {
		const std::vector<uint8_t> &uniformBytes = program->uniform_data[2];
		if (uniformBytes.size() <= 4096) {
			[computeEncoder setBytes:uniformBytes.data() length:uniformBytes.size() atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
		} else {
			__strong id<MTLBuffer> uniformBuffer = nil;
			NSUInteger uniformOffset = 0;
			if (!uploadLargeDefaultUniformBlock(uniformBytes.data(), uniformBytes.size(), uniformBuffer, uniformOffset)) {
				[computeEncoder endEncoding];
				ctx.currentComputeEncoder = nil;
				gl4metalSetError(GL_OUT_OF_MEMORY);
				return;
			}
			[computeEncoder setBuffer:uniformBuffer offset:uniformOffset atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
		}
	}
	for (const ComputeTextureBinding &binding : sampledTextureBindings) {
		[computeEncoder setTexture:binding.texture atIndex:binding.slot];
		if (binding.usesSampler) [computeEncoder setSamplerState:binding.sampler atIndex:binding.slot];
	}
	for (const ComputeTextureBinding &binding : imageTextureBindings) {
		[computeEncoder setTexture:binding.texture atIndex:binding.slot];
	}
	if (!usedBufferNames.empty()) markBuffersUsed(usedBufferNames.data(), usedBufferNames.size());
	[computeEncoder dispatchThreadgroups:MTLSizeMake(num_groups_x, num_groups_y, num_groups_z)
		threadsPerThreadgroup:MTLSizeMake(program->compute_workgroup_size[0],
			program->compute_workgroup_size[1], program->compute_workgroup_size[2])];
	[computeEncoder endEncoding];
	ctx.currentComputeEncoder = nil;
}

void glMemoryBarrier(GLbitfield barriers) {
	(void)barriers; // Metal's tracked-resource hazards cover the implemented image/SSBO/texture accesses.
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
	}
	endRenderEncoder();
	// Deliberately do not commit or wait: command-buffer ordering provides the
	// dependency while keeping the frame pipeline asynchronous.
}

// MARK: - Program objects

GLuint glCreateProgram(void) {
	GLuint program = ++ctx.next_shader_id; // GL uses one name space for shaders and programs
	ctx.program_objects[program] = ProgramObject();
	return program;
}

void glDeleteProgram(GLuint program) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		return;
	}
	if (ctx.current_program == program) {
		p->deleted = GL_TRUE; // destroyed when another program is made current
		return;
	}
	destroyProgram(program);
}

GLboolean glIsProgram(GLuint program) {
	return findProgram(program) != nullptr;
}

void glAttachShader(GLuint program, GLuint shader) {
	ProgramObject *p = findProgram(program);
	if (!p || ctx.shader_objects.find(shader) == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glAttachShader: invalid program %u or shader %u", program, shader);
		return;
	}
	if (std::find(p->shaders.begin(), p->shaders.end(), shader) != p->shaders.end()) {
		GL4METAL_LOG("glAttachShader: shader %u is already attached to program %u", shader, program);
		return;
	}
	p->shaders.push_back(shader);
}

void glDetachShader(GLuint program, GLuint shader) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDetachShader: invalid program %u", program);
		return;
	}
	auto it = std::find(p->shaders.begin(), p->shaders.end(), shader);
	if (it == p->shaders.end()) {
		GL4METAL_LOG("glDetachShader: shader %u is not attached to program %u", shader, program);
		return;
	}
	p->shaders.erase(it);
	releaseShaderIfUnused(shader);
}

void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders) {
	ProgramObject *p = findProgram(program);
	GLsizei n = 0;
	if (p) {
		for (GLuint shader : p->shaders) {
			if (n >= maxCount) {
				break;
			}
			shaders[n++] = shader;
		}
	}
	if (count) {
		*count = n;
	}
}

void glLinkProgram(GLuint program) {
	ProgramObject *pp = findProgram(program);
	if (!pp) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glLinkProgram: invalid program %u", program);
		return;
	}
	ProgramObject &p = *pp;

	// A failed link must not keep anything from an earlier successful one.
	p.linked = GL_FALSE;
	p.info_log.clear();
	p.vertex_function = nil;
	p.fragment_function = nil;
	p.vertex_function_flipped = nil;
	p.compute_pipeline = nil;
	p.compute_workgroup_size[0] = p.compute_workgroup_size[1] = p.compute_workgroup_size[2] = 1;
	p.is_compute = false;
	p.storage_buffers.clear();
	p.storage_images.clear();
	p.attribs.clear();
	p.uniforms.clear();
	p.active_uniforms.clear();
	p.locations.clear();
	p.uniform_data[0].clear();
	p.uniform_data[1].clear();
	p.uniform_data[2].clear();
	p.frag_outputs.clear();
	p.blocks.clear();
	p.uses_gl_frag_coord = false;

	const ShaderObject *vs = nullptr;
	const ShaderObject *fs = nullptr;
	const ShaderObject *cs = nullptr;
	unsigned computeShaderCount = 0;
	for (GLuint shader : p.shaders) {
		auto it = ctx.shader_objects.find(shader);
		if (it == ctx.shader_objects.end()) {
			continue;
		}
		if (it->second.type == GL_VERTEX_SHADER) {
			vs = &it->second;
		} else if (it->second.type == GL_FRAGMENT_SHADER) {
			fs = &it->second;
		} else if (it->second.type == GL_COMPUTE_SHADER) {
			cs = &it->second;
			++computeShaderCount;
		}
	}
	if (cs) {
		if (vs || fs || computeShaderCount != 1) {
			p.info_log = "a compute program must contain exactly one compute shader and no graphics stages\n";
			return;
		}
		if (!cs->compiled) {
			p.info_log = "the attached compute shader is not compiled\n";
			return;
		}
		linkComputeProgram(program, p, *cs);
		return;
	}
	if (!vs || !fs) {
		p.info_log = "a program needs one vertex and one fragment shader attached\n";
		return;
	}
	if (!vs->compiled || !fs->compiled) {
		p.info_log = "an attached shader is not compiled\n";
		return;
	}

	// GLSL -> SPIR-V -> MSL for both stages at once (see shadertranslator.cpp).
	TranslatedProgram tp;
	if (!translateProgram(vs->source, fs->source, tp)) {
		p.info_log = tp.log;
		GL4METAL_LOG("glLinkProgram(%u): translation failed:\n%s", program, tp.log.c_str());
		return;
	}
	if (dumpEnabled()) {
		if (!tp.log.empty()) {
			GL4METAL_LOG("program %u translation notes:\n%s", program, tp.log.c_str());
		}
		GL4METAL_LOG("---- MSL vertex, program %u ----\n%s\n---- MSL fragment, program %u ----\n%s\n---- end MSL ----",
					 program, tp.msl[0].c_str(), program, tp.msl[1].c_str());
	}

	// MSL -> MTLFunction. Compile one vertex library and specialize its static Y-flip
	// function constant into two functions. The old path compiled the same vertex
	// shader twice as separate Metal libraries for default-FB and texture-FB draws.
	auto makeCompileOptions = []() -> MTLCompileOptions * {
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		options.languageVersion = MTLLanguageVersion2_2; // matches set_msl_version(2, 2) in the translator
		// Leave Metal's default fast-math mode enabled, as in the previous compilation path.
		return options;
	};

	id<MTLLibrary> vertexLibrary = nil;
	id<MTLFunction> vertexFunction = nil;
	id<MTLFunction> flippedFunction = nil;
	{
		MTLCompileOptions *options = makeCompileOptions();
		NSError *libraryError = nil;
		NSString *source = [NSString stringWithUTF8String:tp.msl[0].c_str()];
		vertexLibrary = [ctx.device newLibraryWithSource:source options:options error:&libraryError];
		if (vertexLibrary) {
			MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
			bool flipY = false;
			[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
			NSError *normalError = nil;
			vertexFunction = [vertexLibrary newFunctionWithName:@"main0" constantValues:constants error:&normalError];
			flipY = true;
			[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
			NSError *flippedError = nil;
			flippedFunction = [vertexLibrary newFunctionWithName:@"main0" constantValues:constants error:&flippedError];
			if (!vertexFunction || !flippedFunction) {
				NSString *details = normalError ? normalError.localizedDescription :
					(flippedError ? flippedError.localizedDescription : @"Metal did not create both specialized vertex functions");
				p.info_log += std::string("vertex stage function specialization failed:\n") + details.UTF8String + "\n";
			}
		} else {
			p.info_log += std::string("vertex stage failed to compile as Metal:\n") +
				(libraryError ? libraryError.localizedDescription.UTF8String : "no Metal library") + "\n";
		}
	}

	id<MTLFunction> fragmentFunction = nil;
	{
		MTLCompileOptions *options = makeCompileOptions();
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:tp.msl[1].c_str()];
		id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:options error:&error];
		fragmentFunction = [library newFunctionWithName:@"main0"]; // SPIRV-Cross renames main() to main0()
		if (!fragmentFunction) {
			p.info_log += std::string("fragment stage failed to compile as Metal:\n") +
				(error ? error.localizedDescription.UTF8String : "no main0 entry point") + "\n";
		}
	}
	if (!vertexFunction || !fragmentFunction || !flippedFunction) {
		GL4METAL_LOG("glLinkProgram(%u): %s", program, p.info_log.c_str());
		return;
	}

	// Vertex attributes: honour glBindAttribLocation first, then give the rest the lowest free index.
	bool used[16] = {};
	for (const TranslatedAttrib &a : tp.attribs) {
		auto bound = p.attrib_bindings.find(a.name);
		if (bound != p.attrib_bindings.end() && bound->second < 16) {
			used[bound->second] = true;
		}
	}
	for (const TranslatedAttrib &a : tp.attribs) {
		ProgramAttrib pa;
		pa.name = a.name;
		pa.msl_location = a.location;
		pa.vecsize = a.vecsize;
		pa.is_integer = a.is_integer;
		pa.is_unsigned = a.is_unsigned;
		auto bound = p.attrib_bindings.find(a.name);
		if (bound != p.attrib_bindings.end() && bound->second < 16) {
			pa.gl_index = (GLint)bound->second;
		} else {
			int slot = 0;
			while (slot < 16 && used[slot]) {
				slot++;
			}
			if (slot == 16) {
				p.info_log = "too many active vertex attributes\n";
				return;
			}
			used[slot] = true;
			pa.gl_index = slot;
		}
		p.attribs.push_back(pa);
	}

	// Uniform locations: an array uniform of N elements takes N consecutive locations.
	GLint nextLocation = 0;
	for (const TranslatedUniform &u : tp.uniforms) {
		ProgramUniform pu;
		pu.info = u;
		pu.base_location = nextLocation;
		pu.units.assign((size_t)u.array_size, 0);
		for (int element = 0; element < u.array_size; element++) {
			p.locations.emplace_back((GLint)p.uniforms.size(), element);
		}
		nextLocation += u.array_size;
		p.uniforms.push_back(pu);
	}
	p.uniform_data[0].assign(tp.block_size[0], 0); // GL uniforms start at 0
	p.uniform_data[1].assign(tp.block_size[1], 0);
	for (const auto &output : tp.frag_outputs) {
		p.frag_outputs[output.first] = (GLint)output.second;
	}

	for (const TranslatedBlock &tb : tp.blocks) {
		ProgramBlock pb;
		pb.info = tb;
		pb.binding_point = 0; // graphics UBO binding defaults to 0 until glUniformBlockBinding
		p.blocks.push_back(pb);
	}

	// GL_ACTIVE_UNIFORMS includes both default-block uniforms and named-block members.
	// Only the former have locations and can be modified through glUniform*.
	for (const ProgramUniform &uniform : p.uniforms) {
		ActiveUniformInfo active;
		active.name = activeUniformName(uniform.info);
		active.type = activeUniformGLType(uniform.info);
		active.size = uniform.info.array_size;
		active.block_index = -1;
		const TranslatedUniform::Stage &stage = uniform.info.stage[uniform.info.stage[0].present ? 0 : 1];
		active.offset = (GLint)stage.offset;
		active.array_stride = (GLint)stage.array_stride;
		active.matrix_stride = (GLint)stage.matrix_stride;
		active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
		p.active_uniforms.push_back(std::move(active));
	}
	for (size_t blockIndex = 0; blockIndex < p.blocks.size(); ++blockIndex) {
		ProgramBlock &block = p.blocks[blockIndex];
		for (const TranslatedUniform &member : block.info.members) {
			ActiveUniformInfo active;
			active.name = activeUniformName(member);
			active.type = activeUniformGLType(member);
			active.size = member.array_size;
			active.block_index = (GLint)blockIndex;
			const TranslatedUniform::Stage &stage = member.stage[member.stage[0].present ? 0 : 1];
			active.offset = (GLint)stage.offset;
			active.array_stride = (GLint)stage.array_stride;
			active.matrix_stride = (GLint)stage.matrix_stride;
			active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
			block.active_uniform_indices.push_back((GLuint)p.active_uniforms.size());
			p.active_uniforms.push_back(std::move(active));
		}
	}

	p.vertex_function = vertexFunction;
	p.fragment_function = fragmentFunction;
	p.vertex_function_flipped = flippedFunction;
	p.uses_gl_frag_coord = tp.uses_gl_frag_coord;
	p.linked = GL_TRUE;

	if (dumpEnabled()) {
		for (const ProgramAttrib &a : p.attribs) {
			GL4METAL_LOG("program %u attrib '%s': GL index %d, [[attribute(%u)]]", program, a.name.c_str(), a.gl_index, a.msl_location);
		}
		for (const ProgramUniform &u : p.uniforms) {
			GL4METAL_LOG("program %u uniform '%s': location %d x%d, vs offset %d, fs offset %d", program, u.info.name.c_str(),
						 u.base_location, u.info.array_size, u.info.stage[0].present ? (int)u.info.stage[0].offset : -1,
						 u.info.stage[1].present ? (int)u.info.stage[1].offset : -1);
		}
	}
}

void glUseProgram(GLuint program) {
	if (program != 0) {
		ProgramObject *p = findProgram(program);
		if (!p || !p->linked) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glUseProgram: program %u does not exist or is not linked (GL_INVALID_OPERATION)", program);
			return;
		}
	}
	GLuint previous = ctx.current_program;
	ctx.current_program = program;
	if (previous != program) {
		ProgramObject *old = findProgram(previous);
		if (old && old->deleted) {
			destroyProgram(previous); // glDeleteProgram was waiting for it to stop being current
		}
	}
}

void glValidateProgram(GLuint program) {
	ProgramObject *p = findProgram(program);
	if (p) {
		p->validated = p->linked;
	}
}

void glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetProgramiv: invalid program %u", program);
		*params = 0;
		return;
	}
	switch (pname) {
		case GL_DELETE_STATUS:
			*params = p->deleted;
			break;
		case GL_LINK_STATUS:
			*params = p->linked;
			break;
		case GL_VALIDATE_STATUS:
			*params = p->validated;
			break;
		case GL_INFO_LOG_LENGTH:
			*params = p->info_log.empty() ? 0 : (GLint)p->info_log.size() + 1; // includes the NUL terminator
			break;
		case GL_ATTACHED_SHADERS:
			*params = (GLint)p->shaders.size();
			break;
		case GL_ACTIVE_ATTRIBUTES:
			*params = (GLint)p->attribs.size();
			break;
		case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: {
			size_t longest = 0;
			for (const ProgramAttrib &a : p->attribs) {
				longest = std::max(longest, a.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		case GL_ACTIVE_UNIFORMS:
			*params = (GLint)p->active_uniforms.size();
			break;
		case GL_ACTIVE_UNIFORM_MAX_LENGTH: {
			size_t longest = 0;
			for (const ActiveUniformInfo &uniform : p->active_uniforms) {
				longest = std::max(longest, uniform.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		case GL_ACTIVE_UNIFORM_BLOCKS:
			*params = (GLint)p->blocks.size();
			break;
		case GL_COMPUTE_WORK_GROUP_SIZE:
			if (p->is_compute) {
				params[0] = (GLint)p->compute_workgroup_size[0];
				params[1] = (GLint)p->compute_workgroup_size[1];
				params[2] = (GLint)p->compute_workgroup_size[2];
			} else {
				params[0] = params[1] = params[2] = 0;
			}
			break;
		case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: {
			size_t longest = 0;
			for (const ProgramBlock &b : p->blocks) {
				longest = std::max(longest, b.info.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetProgramiv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
	ProgramObject *p = findProgram(program);
	copyGLString(p ? p->info_log : std::string(), bufSize, length, infoLog);
}

// MARK: - Vertex attribute locations

void glBindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || index >= 16 || !name || strncmp(name, "gl_", 3) == 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindAttribLocation: invalid program %u, index %u or name", program, index);
		return;
	}
	p->attrib_bindings[name] = index; // takes effect at the next glLinkProgram
}

GLint glGetAttribLocation(GLuint program, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || !p->linked || !name) {
		return -1;
	}
	for (const ProgramAttrib &a : p->attribs) {
		if (a.name == name) {
			return a.gl_index;
		}
	}
	return -1;
}

void glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || index >= it->second.attribs.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveAttrib: invalid program %u or attribute index %u (GL_INVALID_VALUE)", program, index);
		if (length) {
			*length = 0;
		}
		return;
	}
	const ProgramAttrib &a = it->second.attribs[index];
	static const GLenum floats[4] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum ints[4]   = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	static const GLenum uints[4]  = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	int c = std::min(std::max((int)a.vecsize, 1), 4) - 1;
	if (size) {
		*size = 1;
	}
	if (type) {
		*type = a.is_integer ? (a.is_unsigned ? uints[c] : ints[c]) : floats[c];
	}
	copyGLString(a.name, bufSize, length, name);
}

// MARK: - Fragment output locations

void glBindFragDataLocation(GLuint program, GLuint color, const GLchar *name) {
	// one colour output: the translator numbers it 0
}

void glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber, GLuint index, const GLchar *name) {
	// dual-source blending is not supported
}

GLint glGetFragDataLocation(GLuint program, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || !p->linked || !name) {
		return -1;
	}
	auto it = p->frag_outputs.find(name);
	return it == p->frag_outputs.end() ? -1 : it->second;
}

GLint glGetFragDataIndex(GLuint program, const GLchar *name) {
	return glGetFragDataLocation(program, name) >= 0 ? 0 : -1;
}
