#import "program.h"
#import "shaders.h"
#import "draw.h"
#import "sync.h"

#include <mutex>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <exception>

// MARK: - Helpers

static uint64_t nextProgramCacheUse(uint64_t &clock) {
	++clock;
	if (clock == 0) ++clock;
	return clock;
}

static std::string deviceScopedMSLCacheKey(id<MTLDevice> device, const std::string &mslSource) {
	const uintptr_t deviceIdentity = (uintptr_t)(__bridge void *)device;
	std::string key;
	key.reserve(sizeof(deviceIdentity) + mslSource.size());
	key.append((const char *)&deviceIdentity, sizeof(deviceIdentity));
	key.append(mslSource);
	return key;
}

static ProgramObject *findProgram(GLuint program) {
	auto it = ctx.program_objects.find(program);
	return it == ctx.program_objects.end() ? nullptr : &it->second;
}

static void clearProgramExecutable(ProgramObject &program) {
	program.has_executable = false;
	program.vertex_function = nil;
	program.fragment_function = nil;
	program.vertex_function_flipped = nil;
	program.vertex_stage_cache_id = 0;
	program.fragment_stage_cache_id = 0;
	program.compute_pipeline = nil;
	program.compute_workgroup_size[0] = program.compute_workgroup_size[1] = program.compute_workgroup_size[2] = 1;
	program.is_compute = false;
	for (int stage = 0; stage < 3; ++stage) program.needs_buffer_size_buffer[stage] = false;
	program.storage_buffers.clear();
	program.storage_images.clear();
	program.image_units.clear();
	program.attribs.clear();
	program.uniforms.clear();
	program.sampler_uniform_indices.clear();
	program.active_uniforms.clear();
	program.locations.clear();
	for (int stage = 0; stage < 3; ++stage) {
		program.uniform_data[stage].clear();
		program.uniform_gpu_buffer[stage] = nil;
		program.uniform_gpu_revision[stage] = 0;
		program.uniform_gpu_device_identity[stage] = 0;
		program.uniform_gpu_arena_generation[stage] = 0;
		program.uniform_gpu_offset[stage] = 0;
		program.uniform_gpu_is_transient[stage] = false;
	}
	program.frag_outputs.clear();
	program.blocks.clear();
	program.uses_gl_frag_coord = false;
}

// If a currently-used program is relinked unsuccessfully, OpenGL keeps its old
// executable active until a later glUseProgram removes it. Keep a snapshot only
// for that case; non-current programs can discard their previous executable.
struct ProgramRelinkGuard {
	ProgramObject &target;
	std::unique_ptr<ProgramObject> previous_executable;
	bool committed = false;

	ProgramRelinkGuard(ProgramObject &program, bool preserveCurrentExecutable) : target(program) {
		if (preserveCurrentExecutable && program.has_executable) {
			previous_executable = std::make_unique<ProgramObject>(program);
		}
	}

	void commit() { committed = true; }

	~ProgramRelinkGuard() {
		if (!previous_executable || committed) return;
		std::string failedLinkLog = std::move(target.info_log);
		target = std::move(*previous_executable);
		target.linked = GL_FALSE;
		target.validated = GL_FALSE;
		target.has_executable = true;
		target.info_log = std::move(failedLinkLog);
	}
};

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

// Metal reports static threadgroup-memory overflow only when it creates the
// compute pipeline. Include the generated declarations on this failure path so
// the exact GLSL shared variables can be identified without dumping every shader.
static std::string threadgroupDeclarationsForLog(const std::string &mslSource) {
	std::istringstream input(mslSource);
	std::string line;
	std::string declarations;
	while (std::getline(input, line)) {
		if (line.find("threadgroup ") != std::string::npos) {
			declarations += "    ";
			declarations += line;
			declarations += '\n';
		}
	}
	return declarations;
}

// Compute pipelines are expensive to compile and independent of GL object names.
// Keep both successes and failures: Iris may link the same failing compute program
// more than once during one shader-pack reload. The cache is intentionally process-
// lifetime so late JVM/Metal teardown cannot lock a destroyed static mutex.
struct ComputePipelineCacheEntry {
	__strong id<MTLDevice> device = nil;
	__strong id<MTLLibrary> library = nil;
	__strong id<MTLComputePipelineState> pipeline = nil;
	std::string error;
	uint64_t last_used = 0;
};

struct ComputePipelineCache {
	std::mutex mutex;
	std::unordered_map<std::string, ComputePipelineCacheEntry> entries;
	uint64_t use_clock = 0;
};

static ComputePipelineCache &computePipelineCache() {
	static ComputePipelineCache *cache = [] {
		auto *created = new ComputePipelineCache();
		created->entries.max_load_factor(0.75f);
		created->entries.reserve(64);
		return created;
	}();
	return *cache;
}

static id<MTLComputePipelineState> computePipelineForMSL(const std::string &mslSource, std::string &errorText) {
	id<MTLDevice> compileDevice = ctx.device;
	if (!compileDevice) {
		errorText = "Metal device is not initialized";
		return nil;
	}
	const std::string cacheKey = deviceScopedMSLCacheKey(compileDevice, mslSource);
	ComputePipelineCache &cache = computePipelineCache();
	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto cached = cache.entries.find(cacheKey);
		if (cached != cache.entries.end()) {
			cached->second.last_used = nextProgramCacheUse(cache.use_clock);
			errorText = cached->second.error;
			return cached->second.pipeline;
		}
	}

	ComputePipelineCacheEntry compiled;
	compiled.device = compileDevice;
	MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
	options.languageVersion = MTLLanguageVersion2_2;
	NSError *libraryError = nil;
	NSError *metal4LibraryError = nil;
	NSString *source = [NSString stringWithUTF8String:mslSource.c_str()];
	id<MTLLibrary> library = nil;

	if (ctx.metal4Enabled && ctx.metal4Compiler && source) {
		MTL4LibraryDescriptor *libraryDescriptor = [[MTL4LibraryDescriptor alloc] init];
		libraryDescriptor.source = source;
		libraryDescriptor.options = options;
		library = [ctx.metal4Compiler newLibraryWithDescriptor:libraryDescriptor error:&metal4LibraryError];
		if (!library) {
			GL4METAL_LOG("Metal 4 compute library compile failed; retrying classic compiler: %s",
				metal4LibraryError.localizedDescription.UTF8String ?: "unknown Metal 4 library error");
		}
	}

	if (!library && source) library = [compileDevice newLibraryWithSource:source options:options error:&libraryError];
	compiled.library = library;
	id<MTLFunction> computeFunction = [library newFunctionWithName:@"main0"];
	if (!computeFunction) {
		const char *classicErrorText = libraryError ? libraryError.localizedDescription.UTF8String : nullptr;
		const char *metal4ErrorText = metal4LibraryError ? metal4LibraryError.localizedDescription.UTF8String : nullptr;
		compiled.error = std::string("Metal compute library compilation failed: classic=") +
			(classicErrorText ? classicErrorText : "not attempted/unknown") + "; Metal4=" +
			(metal4ErrorText ? metal4ErrorText : "not attempted/unknown");
	} else {
		NSError *pipelineError = nil;
		NSError *metal4PipelineError = nil;

		if (ctx.metal4Enabled && ctx.metal4Compiler && library) {
			MTL4LibraryFunctionDescriptor *functionDescriptor = [[MTL4LibraryFunctionDescriptor alloc] init];
			functionDescriptor.library = library;
			functionDescriptor.name = @"main0";
			MTL4ComputePipelineDescriptor *pipelineDescriptor = [[MTL4ComputePipelineDescriptor alloc] init];
			pipelineDescriptor.computeFunctionDescriptor = functionDescriptor;
			MTL4CompilerTaskOptions *taskOptions = [[MTL4CompilerTaskOptions alloc] init];
			compiled.pipeline = [ctx.metal4Compiler newComputePipelineStateWithDescriptor:pipelineDescriptor
				compilerTaskOptions:taskOptions error:&metal4PipelineError];
			if (!compiled.pipeline) {
				GL4METAL_LOG("Metal 4 compute PSO failed; retrying classic compiler: %s",
					metal4PipelineError.localizedDescription.UTF8String ?: "unknown Metal 4 PSO error");
			}
		}

		if (!compiled.pipeline) {
			compiled.pipeline = [compileDevice newComputePipelineStateWithFunction:computeFunction error:&pipelineError];
		}
		if (!compiled.pipeline) {
			const char *classicErrorText = pipelineError ? pipelineError.localizedDescription.UTF8String : nullptr;
			const char *metal4ErrorText = metal4PipelineError ? metal4PipelineError.localizedDescription.UTF8String : nullptr;
			compiled.error = std::string("Metal compute pipeline creation failed: classic=") +
				(classicErrorText ? classicErrorText : "unknown error") + "; Metal4=" +
				(metal4ErrorText ? metal4ErrorText : "not attempted/unknown");
		}
	}

	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		compiled.last_used = nextProgramCacheUse(cache.use_clock);
		auto inserted = cache.entries.emplace(cacheKey, compiled);
		if (!inserted.second) {
			inserted.first->second.last_used = compiled.last_used;
			compiled = inserted.first->second;
		}
		if (cache.entries.size() > 64) {
			auto victim = cache.entries.begin();
			for (auto candidate = cache.entries.begin(); candidate != cache.entries.end(); ++candidate) {
				if (candidate->second.last_used < victim->second.last_used) victim = candidate;
			}
			if (victim != cache.entries.end()) cache.entries.erase(victim);
		}
	}
	errorText = compiled.error;
	return compiled.pipeline;
}

struct VertexStageCacheEntry {
	__strong id<MTLDevice> device = nil;
	__strong id<MTLFunction> normalFunction = nil;
	__strong id<MTLFunction> flippedFunction = nil;
	uint64_t cacheId = 0;
	uint64_t last_used = 0;
	std::string error;
};

struct VertexStageCache {
	std::mutex mutex;
	std::unordered_map<std::string, VertexStageCacheEntry> entries;
	uint64_t nextCacheId = 1;
	uint64_t use_clock = 0;
};

static VertexStageCache &vertexStageCache() {
	static VertexStageCache *cache = [] {
		auto *created = new VertexStageCache();
		created->entries.max_load_factor(0.75f);
		created->entries.reserve(96);
		return created;
	}();
	return *cache;
}

static bool vertexStageFunctionsForMSL(const std::string &mslSource, VertexStageCacheEntry &outEntry) {
	id<MTLDevice> compileDevice = ctx.device;
	if (!compileDevice) { outEntry.error = "Metal device is not initialized"; return false; }
	const std::string cacheKey = deviceScopedMSLCacheKey(compileDevice, mslSource);
	VertexStageCache &cache = vertexStageCache();
	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto cached = cache.entries.find(cacheKey);
		if (cached != cache.entries.end()) {
			cached->second.last_used = nextProgramCacheUse(cache.use_clock);
			outEntry = cached->second;
			return outEntry.normalFunction && outEntry.flippedFunction;
		}
	}

	VertexStageCacheEntry compiled;
	compiled.device = compileDevice;
	NSString *source = [NSString stringWithUTF8String:mslSource.c_str()];
	MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
	options.languageVersion = MTLLanguageVersion2_2;
	NSError *libraryError = nil;
	id<MTLLibrary> library = source ? [compileDevice newLibraryWithSource:source options:options error:&libraryError] : nil;
	if (!library) {
		compiled.error = libraryError ? (libraryError.localizedDescription.UTF8String ?: "Metal vertex library compilation failed") :
			(source ? "no Metal vertex library" : "vertex MSL is not valid UTF-8");
	} else {
		MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
		bool flipY = false;
		[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
		NSError *normalError = nil;
		compiled.normalFunction = [library newFunctionWithName:@"main0" constantValues:constants error:&normalError];
		flipY = true;
		[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
		NSError *flippedError = nil;
		compiled.flippedFunction = [library newFunctionWithName:@"main0" constantValues:constants error:&flippedError];
		if (!compiled.normalFunction || !compiled.flippedFunction) {
			NSString *details = normalError ? normalError.localizedDescription :
				(flippedError ? flippedError.localizedDescription : @"Metal did not create both specialized vertex functions");
			compiled.error = details.UTF8String ?: "vertex function specialization failed";
		}
	}

	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto existing = cache.entries.find(cacheKey);
		if (existing != cache.entries.end()) {
			existing->second.last_used = nextProgramCacheUse(cache.use_clock);
			compiled = existing->second;
		} else {
			compiled.cacheId = cache.nextCacheId++;
			if (compiled.cacheId == 0) compiled.cacheId = cache.nextCacheId++;
			compiled.last_used = nextProgramCacheUse(cache.use_clock);
			cache.entries.emplace(cacheKey, compiled);
			if (cache.entries.size() > 96) {
				auto victim = cache.entries.begin();
				for (auto candidate = cache.entries.begin(); candidate != cache.entries.end(); ++candidate) {
					if (candidate->second.last_used < victim->second.last_used) victim = candidate;
				}
				if (victim != cache.entries.end()) cache.entries.erase(victim);
			}
		}
	}
	outEntry = compiled;
	return outEntry.normalFunction && outEntry.flippedFunction;
}

struct FragmentStageCacheEntry {
	__strong id<MTLDevice> device = nil;
	__strong id<MTLFunction> function = nil;
	uint64_t cacheId = 0;
	uint64_t last_used = 0;
	std::string error;
};

struct FragmentStageCache {
	std::mutex mutex;
	std::unordered_map<std::string, FragmentStageCacheEntry> entries;
	uint64_t nextCacheId = 1;
	uint64_t use_clock = 0;
};

static FragmentStageCache &fragmentStageCache() {
	static FragmentStageCache *cache = [] {
		auto *created = new FragmentStageCache();
		created->entries.max_load_factor(0.75f);
		created->entries.reserve(96);
		return created;
	}();
	return *cache;
}

static bool fragmentStageFunctionForMSL(const std::string &mslSource, FragmentStageCacheEntry &outEntry) {
	id<MTLDevice> compileDevice = ctx.device;
	if (!compileDevice) { outEntry.error = "Metal device is not initialized"; return false; }
	const std::string cacheKey = deviceScopedMSLCacheKey(compileDevice, mslSource);
	FragmentStageCache &cache = fragmentStageCache();
	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto cached = cache.entries.find(cacheKey);
		if (cached != cache.entries.end()) {
			cached->second.last_used = nextProgramCacheUse(cache.use_clock);
			outEntry = cached->second;
			return outEntry.function != nil;
		}
	}

	FragmentStageCacheEntry compiled;
	compiled.device = compileDevice;
	NSString *source = [NSString stringWithUTF8String:mslSource.c_str()];
	MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
	options.languageVersion = MTLLanguageVersion2_2;
	NSError *libraryError = nil;
	id<MTLLibrary> library = source ? [compileDevice newLibraryWithSource:source options:options error:&libraryError] : nil;
	compiled.function = [library newFunctionWithName:@"main0"];
	if (!compiled.function) {
		compiled.error = libraryError ? (libraryError.localizedDescription.UTF8String ?: "Metal fragment library compilation failed") :
			(library ? "no main0 fragment entry point" : (source ? "no Metal fragment library" : "fragment MSL is not valid UTF-8"));
	}

	{
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto existing = cache.entries.find(cacheKey);
		if (existing != cache.entries.end()) {
			existing->second.last_used = nextProgramCacheUse(cache.use_clock);
			compiled = existing->second;
		} else {
			compiled.cacheId = cache.nextCacheId++;
			if (compiled.cacheId == 0) compiled.cacheId = cache.nextCacheId++;
			compiled.last_used = nextProgramCacheUse(cache.use_clock);
			cache.entries.emplace(cacheKey, compiled);
			if (cache.entries.size() > 96) {
				auto victim = cache.entries.begin();
				for (auto candidate = cache.entries.begin(); candidate != cache.entries.end(); ++candidate) {
					if (candidate->second.last_used < victim->second.last_used) victim = candidate;
				}
				if (victim != cache.entries.end()) cache.entries.erase(victim);
			}
		}
	}
	outEntry = compiled;
	return outEntry.function != nil;
}

// Compute encoders remain open across consecutive dispatches. Avoid repeating
// Metal state calls when a kernel reuses the same pipeline and resource bindings.
static constexpr NSUInteger kComputeBufferBindingSlots = GL4METAL_UNIFORM_BUFFER_INDEX + 1;
static constexpr NSUInteger kComputeTextureBindingSlots = 32;

struct ComputeEncoderBufferSlot {
	bool valid = false;
	__strong id<MTLBuffer> buffer = nil;
	NSUInteger offset = 0;
};

struct ComputeEncoderTextureSlot {
	bool valid = false;
	__strong id<MTLTexture> texture = nil;
};

struct ComputeEncoderSamplerSlot {
	bool valid = false;
	__strong id<MTLSamplerState> sampler = nil;
};

struct ComputeEncoderUniformSlot {
	bool valid = false;
	GLuint program = 0;
	uint64_t revision = 0;
	NSUInteger length = 0;
};

struct ComputeEncoderStateCache {
	__weak id<MTLComputeCommandEncoder> encoder = nil;
	bool pipelineValid = false;
	__strong id<MTLComputePipelineState> pipeline = nil;
	std::array<ComputeEncoderBufferSlot, kComputeBufferBindingSlots> buffers{};
	std::array<ComputeEncoderTextureSlot, kComputeTextureBindingSlots> textures{};
	std::array<ComputeEncoderSamplerSlot, kComputeTextureBindingSlots> samplers{};
	bool bufferSizesValid = false;
	std::array<uint32_t, GL4METAL_BUFFER_SIZE_BUFFER_INDEX + 1> bufferSizes{};
	ComputeEncoderUniformSlot defaultUniform{};

	void resetForEncoder(id<MTLComputeCommandEncoder> newEncoder) {
		pipelineValid = false;
		pipeline = nil;
		bufferSizesValid = false;
		defaultUniform.valid = false;
		for (auto &binding : buffers) {
			binding.valid = false;
			binding.buffer = nil;
			binding.offset = 0;
		}
		for (auto &binding : textures) {
			binding.valid = false;
			binding.texture = nil;
		}
		for (auto &binding : samplers) {
			binding.valid = false;
			binding.sampler = nil;
		}
		encoder = newEncoder;
	}
};

static ComputeEncoderStateCache &threadComputeEncoderStateCache(void) {
	static thread_local ComputeEncoderStateCache cache;
	return cache;
}

static ComputeEncoderStateCache &computeEncoderStateFor(id<MTLComputeCommandEncoder> encoder) {
	ComputeEncoderStateCache &cache = threadComputeEncoderStateCache();
	if (cache.encoder != encoder) cache.resetForEncoder(encoder);
	return cache;
}

void invalidateComputeEncoderStateCache(void) {
	threadComputeEncoderStateCache().resetForEncoder(nil);
}

static inline void bindComputePipelineCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
	if (cache.pipelineValid && cache.pipeline == pipeline) return;
	[encoder setComputePipelineState:pipeline];
	cache.pipeline = pipeline;
	cache.pipelineValid = true;
}

static inline void bindComputeBufferCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger slot) {
	if (slot >= cache.buffers.size()) {
		[encoder setBuffer:buffer offset:offset atIndex:slot];
		return;
	}
	ComputeEncoderBufferSlot &binding = cache.buffers[slot];
	if (slot == GL4METAL_UNIFORM_BUFFER_INDEX) cache.defaultUniform.valid = false;
	if (slot == GL4METAL_BUFFER_SIZE_BUFFER_INDEX) cache.bufferSizesValid = false;
	if (binding.valid && binding.buffer == buffer && binding.offset == offset) return;
	[encoder setBuffer:buffer offset:offset atIndex:slot];
	binding.buffer = buffer;
	binding.offset = offset;
	binding.valid = true;
}

static inline void bindComputeBufferSizesCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, const std::array<uint32_t, GL4METAL_BUFFER_SIZE_BUFFER_INDEX + 1> &sizes) {
	if (cache.bufferSizesValid && cache.bufferSizes == sizes) return;
	[encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:GL4METAL_BUFFER_SIZE_BUFFER_INDEX];
	cache.bufferSizes = sizes;
	cache.bufferSizesValid = true;
	cache.buffers[GL4METAL_BUFFER_SIZE_BUFFER_INDEX].valid = false;
}

static inline void bindComputeDefaultUniformCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, GLuint programId, uint64_t revision,
	const std::vector<uint8_t> &bytes) {
	ComputeEncoderUniformSlot &binding = cache.defaultUniform;
	if (binding.valid && binding.program == programId && binding.revision == revision &&
		binding.length == bytes.size()) return;
	[encoder setBytes:bytes.data() length:bytes.size() atIndex:GL4METAL_UNIFORM_BUFFER_INDEX];
	binding.program = programId;
	binding.revision = revision;
	binding.length = bytes.size();
	binding.valid = true;
	cache.buffers[GL4METAL_UNIFORM_BUFFER_INDEX].valid = false;
}

static inline void bindComputeTextureCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, id<MTLTexture> texture, NSUInteger slot) {
	if (slot >= cache.textures.size()) {
		[encoder setTexture:texture atIndex:slot];
		return;
	}
	ComputeEncoderTextureSlot &binding = cache.textures[slot];
	if (binding.valid && binding.texture == texture) return;
	[encoder setTexture:texture atIndex:slot];
	binding.texture = texture;
	binding.valid = true;
}

static inline void bindComputeSamplerCached(ComputeEncoderStateCache &cache,
	id<MTLComputeCommandEncoder> encoder, id<MTLSamplerState> sampler, NSUInteger slot) {
	if (slot >= cache.samplers.size()) {
		[encoder setSamplerState:sampler atIndex:slot];
		return;
	}
	ComputeEncoderSamplerSlot &binding = cache.samplers[slot];
	if (binding.valid && binding.sampler == sampler) return;
	[encoder setSamplerState:sampler atIndex:slot];
	binding.sampler = sampler;
	binding.valid = true;
}

// Active-uniform type mapping for ordinary uniforms and samplers. Opaque image
// uniforms use the exact GL_IMAGE_* type carried by ProgramUniform::opaque_gl_type.
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
	try {
		if (!translateComputeProgram(computeShader.compiled_stage, translated)) {
			program.info_log = translated.log;
			GL4METAL_LOG("glLinkProgram(%u): compute translation failed:\n%s", programName, translated.log.c_str());
			return false;
		}
	} catch (const std::exception &exception) {
		program.info_log = std::string("compute shader translation raised an exception: ") + exception.what() + "\n";
		GL4METAL_LOG("glLinkProgram(%u): %s", programName, program.info_log.c_str());
		return false;
	} catch (...) {
		program.info_log = "compute shader translation raised an unknown exception\n";
		return false;
	}
	std::string pipelineError;
	const NSUInteger deviceLimit = ctx.device ? ctx.device.maxThreadgroupMemoryLength : 0;
	std::string compactionNotes;
	std::string compactedMSL = translated.msl[2];
	const bool precompacted = compactMetalThreadgroupFloat3Arrays(
		compactedMSL, (std::size_t)deviceLimit, compactionNotes, false);
	if (precompacted) {
		translated.msl[2].swap(compactedMSL);
		translated.log += "threadgroup storage optimization: " + compactionNotes + "\n";
	}
	id<MTLComputePipelineState> computePipeline = computePipelineForMSL(translated.msl[2], pipelineError);
	if (!computePipeline && !precompacted &&
		(pipelineError.find("Threadgroup memory size") != std::string::npos ||
		 pipelineError.find("threadgroup memory") != std::string::npos)) {
		// Metal's error is authoritative for aggregate storage: retry once with all
		// eligible nested float3 arrays packed, even if each array alone is below the
		// device limit but other threadgroup declarations contribute to the total.
		compactedMSL = translated.msl[2];
		compactionNotes.clear();
		if (compactMetalThreadgroupFloat3Arrays(compactedMSL, (std::size_t)deviceLimit, compactionNotes, true)) {
			translated.msl[2].swap(compactedMSL);
			translated.log += "threadgroup storage fallback optimization: " + compactionNotes + "\n";
			std::string compactPipelineError;
			computePipeline = computePipelineForMSL(translated.msl[2], compactPipelineError);
			if (!computePipeline) pipelineError += "\npacked threadgroup representation also failed:\n" + compactPipelineError;
			else pipelineError.clear();
		}
	}
	if (dumpEnabled()) {
		if (!translated.log.empty()) GL4METAL_LOG("compute program %u translation notes:\n%s", programName, translated.log.c_str());
		GL4METAL_LOG("---- MSL compute, program %u ----\n%s\n---- end MSL ----", programName, translated.msl[2].c_str());
	}

	if (!computePipeline) {
		program.info_log = "compute Metal pipeline compilation failed:\n" + pipelineError + "\n";
		if (pipelineError.find("Threadgroup memory size") != std::string::npos ||
			pipelineError.find("threadgroup memory") != std::string::npos) {
			const NSUInteger deviceLimit = ctx.device ? ctx.device.maxThreadgroupMemoryLength : 0;
			program.info_log += "device static/dynamic threadgroup-memory limit: " +
				std::to_string((unsigned long long)deviceLimit) + " bytes\n";
			program.info_log += "GL4Metal attempted a semantics-preserving packed_float3 representation for eligible threadgroup arrays; workgroup size and barrier positions were not changed. Other shared-memory layouts are not rewritten automatically.\n";
			const std::string declarations = threadgroupDeclarationsForLog(translated.msl[2]);
			if (!declarations.empty()) {
				program.info_log += "generated MSL threadgroup declarations:\n" + declarations;
			}
		}
		if (dumpEnabled()) {
			program.info_log += "generated MSL compute source:\n" + translated.msl[2] + "\n";
		}
		GL4METAL_LOG("glLinkProgram(%u): %s", programName, program.info_log.c_str());
		return false;
	}

	program.compute_pipeline = computePipeline;
	program.compute_workgroup_size[0] = translated.workgroup_size[0];
	program.compute_workgroup_size[1] = translated.workgroup_size[1];
	program.compute_workgroup_size[2] = translated.workgroup_size[2];
	program.storage_buffers = translated.storage_buffers;
	program.storage_images = translated.storage_images;
	for (int stage = 0; stage < 3; ++stage)
		program.needs_buffer_size_buffer[stage] = translated.needs_buffer_size_buffer[stage];
	program.image_units.clear();
	program.image_units.reserve(program.storage_images.size());
	for (const TranslatedImage &imageInfo : program.storage_images) {
		program.image_units.push_back((GLint)imageInfo.binding);
	}
	program.uniforms.reserve(translated.uniforms.size() + translated.storage_images.size());
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

	// Storage images are opaque OpenGL uniforms too. They were previously kept only in
	// storage_images for MSL binding, so Iris could not discover them through
	// GL_ACTIVE_UNIFORMS / glGetUniformLocation and consequently never called
	// glBindImageTexture. Build one location-bearing ProgramUniform per image resource;
	// flattened array elements share one uniform and consecutive locations.
	for (size_t imageIndex = 0; imageIndex < program.storage_images.size();) {
		const TranslatedImage &firstImage = program.storage_images[imageIndex];
		std::string imageBaseName = firstImage.name;
		uint32_t arrayCount = 1;
		const size_t bracket = imageBaseName.rfind('[');
		if (bracket != std::string::npos && !imageBaseName.empty() && imageBaseName.back() == ']') {
			const std::string indexText = imageBaseName.substr(bracket + 1, imageBaseName.size() - bracket - 2);
			char *endPtr = nullptr;
			const unsigned long parsedIndex = std::strtoul(indexText.c_str(), &endPtr, 10);
			if (endPtr && *endPtr == '\0' && parsedIndex == 0) {
				const std::string candidateBase = imageBaseName.substr(0, bracket);
				uint32_t candidateCount = 0;
				while (imageIndex + candidateCount < program.storage_images.size()) {
					const std::string expectedName = candidateBase + "[" + std::to_string(candidateCount) + "]";
					if (program.storage_images[imageIndex + candidateCount].name != expectedName) break;
					++candidateCount;
				}
				if (candidateCount != 0) {
					imageBaseName = candidateBase;
					arrayCount = candidateCount;
				}
			}
		}

		ProgramUniform imageUniform;
		imageUniform.info.name = imageBaseName;
		imageUniform.info.kind = TranslatedUniform::Int; // GL image-unit values are set/read as GLint
		imageUniform.info.vecsize = 1;
		imageUniform.info.columns = 1;
		imageUniform.info.array_size = (int)arrayCount;
		imageUniform.base_location = nextLocation;
		imageUniform.is_image = true;
		imageUniform.opaque_gl_type = firstImage.is_3d ? GL_IMAGE_3D :
			(firstImage.is_array ? GL_IMAGE_2D_ARRAY : GL_IMAGE_2D);
		imageUniform.image_index = (uint32_t)imageIndex;
		for (uint32_t element = 0; element < arrayCount; ++element) {
			program.locations.emplace_back((GLint)program.uniforms.size(), (GLint)element);
		}
		nextLocation += (GLint)arrayCount;
		program.uniforms.push_back(std::move(imageUniform));
		imageIndex += arrayCount;
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
		active.type = uniform.is_image ? uniform.opaque_gl_type : activeUniformGLType(uniform.info);
		active.size = uniform.info.array_size;
		active.block_index = -1;
		if (!uniform.is_image) {
			const TranslatedUniform::Stage &stage = uniform.info.stage[2];
			active.offset = (GLint)stage.offset;
			active.array_stride = (GLint)stage.array_stride;
			active.matrix_stride = (GLint)stage.matrix_stride;
			active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
		}
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
	program.has_executable = true;
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
	// A GL range binding may intentionally expose only a subrange of the backing
	// store. Runtime-sized GLSL arrays are bounded by that range; requiring the
	// range to equal the entire allocation incorrectly rejects valid bindBufferRange.
	(void)runtimeSized;
	if (requested > remaining || requested < (NSUInteger)requiredSize ||
		start > bufferFound->second.length ||
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


void dispatchComputeInternal(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z,
	id<MTLBuffer> indirectBuffer, NSUInteger indirectOffset, GLuint indirectBufferName) {
	const bool indirect = indirectBuffer != nil;
	const NSUInteger maxWorkGroupInvocations = gl4metalMaxComputeWorkGroupInvocations();
	if (maxWorkGroupInvocations == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchCompute: compute shaders are unavailable on the current Metal device");
		return;
	}
	ProgramObject *program = findProgram(ctx.current_program);
	if (!program || !program->has_executable || !program->is_compute || !program->compute_pipeline) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchCompute: current program is not a linked compute program");
		return;
	}
	if (!indirect) {
		if (num_groups_x > 65535 || num_groups_y > 65535 || num_groups_z > 65535) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glDispatchCompute: workgroup count exceeds the iOS limit of 65535 per dimension");
			return;
		}
		if (num_groups_x == 0 || num_groups_y == 0 || num_groups_z == 0) return;
	}
	if (indirect && ((indirectOffset & 3u) != 0 || indirectOffset > indirectBuffer.length ||
		indirectBuffer.length - indirectOffset < sizeof(uint32_t) * 3u || indirectBuffer.gpuAddress == 0 ||
		UINT64_MAX - indirectBuffer.gpuAddress < (MTLGPUAddress)indirectOffset)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDispatchComputeIndirect: invalid indirect argument range, GPU address, or alignment");
		return;
	}

	const uint64_t localThreadCount = (uint64_t)program->compute_workgroup_size[0] *
		(uint64_t)program->compute_workgroup_size[1] * (uint64_t)program->compute_workgroup_size[2];
	if (localThreadCount == 0 || localThreadCount > maxWorkGroupInvocations ||
		localThreadCount > program->compute_pipeline.maxTotalThreadsPerThreadgroup) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchCompute: local workgroup size (%u,%u,%u) exceeds GL4Metal/iOS limit (%lu)",
			program->compute_workgroup_size[0], program->compute_workgroup_size[1],
			program->compute_workgroup_size[2], (unsigned long)std::min<NSUInteger>(
				program->compute_pipeline.maxTotalThreadsPerThreadgroup, maxWorkGroupInvocations));
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
	usedBufferNames.reserve(program->storage_buffers.size() + program->blocks.size() + program->uniforms.size() + (indirect ? 1u : 0u));
	if (indirect) addUniqueBufferName(usedBufferNames, indirectBufferName);

	// Resolve every buffer and check its logical GL range before creating an encoder.
	std::array<uint32_t, GL4METAL_BUFFER_SIZE_BUFFER_INDEX + 1> computeBufferSizes{};
	for (const TranslatedStorageBuffer &storageInfo : program->storage_buffers) {
		if (storageInfo.binding >= GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS || storageInfo.slot >= 16) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: invalid SSBO binding/Metal slot for '%s'", storageInfo.name.c_str());
			return;
		}
		const ShaderStorageBufferBinding &binding = ctx.shader_storage_buffer_bindings[storageInfo.binding];
		if (binding.offset < 0 || (binding.offset & 15) != 0) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: SSBO '%s' has invalid 16-byte-aligned offset %lld",
				storageInfo.name.c_str(), (long long)binding.offset);
			return;
		}
		ComputeBufferBinding resolved;
		if (!resolveComputeBuffer(binding.buffer, binding.offset, binding.size, storageInfo.required_size,
			storageInfo.runtime_sized, resolved.buffer, resolved.offset, resolved.size)) {
			GL4METAL_LOG("glDispatchCompute: SSBO '%s' has no compatible buffer range (binding=%u)",
				storageInfo.name.c_str(), storageInfo.binding);
			return;
		}
		resolved.slot = storageInfo.slot;
		if (program->needs_buffer_size_buffer[2]) {
			if (resolved.size > (NSUInteger)UINT32_MAX) {
				gl4metalSetError(GL_INVALID_OPERATION);
				GL4METAL_LOG("glDispatchCompute: SSBO '%s' range exceeds SPIRV-Cross 32-bit OpArrayLength size limits", storageInfo.name.c_str());
				return;
			}
			computeBufferSizes[storageInfo.slot] = (uint32_t)resolved.size;
		}
		storageBufferBindings.push_back(resolved);
		addUniqueBufferName(usedBufferNames, binding.buffer);
	}

	// UBO indices map into Metal buffer slots 16...28; slot 29 carries SSBO lengths and loose uniforms use slot 30.
	for (const ProgramBlock &block : program->blocks) {
		const TranslatedBlock::Stage &stage = block.info.stage[2];
		if (!stage.present) continue;
		if (block.binding_point >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS || stage.slot < 16 ||
			stage.slot >= GL4METAL_BUFFER_SIZE_BUFFER_INDEX) {
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
						texture = gl4metalTextureForSampling(object, object.texture);
						if (!texture) return;
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
	// independently by the translator after all sampled textures. image3D maps to
	// a layered MTLTextureType3D; image2D remains a non-layered MTLTextureType2D.
	for (size_t imageIndex = 0; imageIndex < program->storage_images.size(); ++imageIndex) {
		const TranslatedImage &imageInfo = program->storage_images[imageIndex];
		const GLint imageUnit = imageIndex < program->image_units.size()
			? program->image_units[imageIndex] : -1;
		if (imageUnit < 0 || imageUnit >= GL4METAL_MAX_IMAGE_UNITS || imageInfo.slot >= 32) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glDispatchCompute: image '%s' has invalid image unit %d or Metal slot %u",
				imageInfo.name.c_str(), imageUnit, imageInfo.slot);
			return;
		}
		const ImageUnitBinding &binding = ctx.image_unit_bindings[(GLuint)imageUnit];
		auto textureObject = ctx.texture_objects.find(binding.texture);
		const GLenum expectedTarget = imageInfo.is_3d ? GL_TEXTURE_3D :
			(imageInfo.is_array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D);
		const MTLTextureType expectedType = imageInfo.is_3d ? MTLTextureType3D :
			(imageInfo.is_array ? MTLTextureType2DArray : MTLTextureType2D);
		const bool invalidLayerMode = (imageInfo.is_3d || imageInfo.is_array)
			? binding.layered != GL_TRUE
			: binding.layer != 0;
		// GL ignores the layered flag for a plain GL_TEXTURE_2D image binding;
		// its only valid layer index is zero. image3D requires a layered binding.
		const bool invalidTextureBinding = binding.texture == 0 || textureObject == ctx.texture_objects.end() ||
			textureObject->second.target != expectedTarget || !textureObject->second.texture ||
			textureObject->second.texture.textureType != expectedType || binding.level != 0 ||
			invalidLayerMode || binding.format != (textureObject == ctx.texture_objects.end() ? 0 : textureObject->second.internalformat) ||
			(imageInfo.format != 0 && binding.format != imageInfo.format);
		if (invalidTextureBinding) {
			gl4metalSetError(GL_INVALID_OPERATION);
			static std::atomic<uint64_t> imageBindingFailureCount{0};
			const uint64_t failureNumber = imageBindingFailureCount.fetch_add(1, std::memory_order_relaxed);
			if (failureNumber < 8 || (failureNumber & (failureNumber - 1)) == 0) {
				const GLenum actualTarget = textureObject == ctx.texture_objects.end() ? 0 : textureObject->second.target;
				const MTLTextureType actualType = (textureObject == ctx.texture_objects.end() || !textureObject->second.texture)
					? MTLTextureType2D : textureObject->second.texture.textureType;
				const GLenum actualFormat = textureObject == ctx.texture_objects.end() ? 0 : textureObject->second.internalformat;
				GL4METAL_LOG("glDispatchCompute: ctx=%p image '%s' binding mismatch unit=%u texture=%u target=0x%X type=%u level=%d layered=%u layer=%d access=0x%X format=0x%X storage=0x%X expected=%s; occurrence=%llu",
					(void *)&ctx, imageInfo.name.c_str(), (GLuint)imageUnit, binding.texture, actualTarget,
					(unsigned)actualType, binding.level, (unsigned)binding.layered,
					binding.layer, binding.access, binding.format, actualFormat,
					imageInfo.is_3d ? "layered GL_TEXTURE_3D" :
					(imageInfo.is_array ? "layered GL_TEXTURE_2D_ARRAY" : "GL_TEXTURE_2D"),
					(unsigned long long)(failureNumber + 1));
			}
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

	// Prefer the Metal 4 compute path on MTLGPUFamilyMetal4 devices. Graphics rendering
	// remains on the classic queue for this migration phase; queue transitions are
	// ordered through MTLSharedEvent by gl4metalBeginMetal4Compute/gl4metalEnsureClassicCommandBuffer.
	if (ctx.metal4Enabled && gl4metalBeginMetal4Compute()) {
		id<MTL4ComputeCommandEncoder> metal4Encoder = ctx.currentMetal4ComputeEncoder;
		bool bindingsValid = metal4Encoder != nil;
		if (bindingsValid) {
			gl4metalMetal4UseComputePipeline(program->compute_pipeline);
			[metal4Encoder setComputePipelineState:program->compute_pipeline];
		}
		for (const ComputeBufferBinding &binding : storageBufferBindings) {
			bindingsValid = gl4metalMetal4BindBuffer(binding.buffer, binding.offset, binding.slot) && bindingsValid;
		}
		for (const ComputeBufferBinding &binding : uniformBufferBindings) {
			bindingsValid = gl4metalMetal4BindBuffer(binding.buffer, binding.offset, binding.slot) && bindingsValid;
		}
		if (program->needs_buffer_size_buffer[2]) {
			bindingsValid = gl4metalMetal4BindBytes(computeBufferSizes.data(), sizeof(computeBufferSizes),
				GL4METAL_BUFFER_SIZE_BUFFER_INDEX) && bindingsValid;
		}
		if (!program->uniform_data[2].empty()) {
			const std::vector<uint8_t> &uniformBytes = program->uniform_data[2];
			if (uniformBytes.size() <= 4096) {
				bindingsValid = gl4metalMetal4BindBytes(uniformBytes.data(), uniformBytes.size(),
					GL4METAL_UNIFORM_BUFFER_INDEX) && bindingsValid;
			} else {
				__strong id<MTLBuffer> uniformBuffer = nil;
				NSUInteger uniformOffset = 0;
				if (!programUniformBufferForStage(*program, 2, uniformBuffer, uniformOffset)) {
					bindingsValid = false;
				} else {
					bindingsValid = gl4metalMetal4BindBuffer(uniformBuffer, uniformOffset,
						GL4METAL_UNIFORM_BUFFER_INDEX) && bindingsValid;
				}
			}
		}
		for (const ComputeTextureBinding &binding : sampledTextureBindings) {
			bindingsValid = gl4metalMetal4BindTexture(binding.texture, binding.slot) && bindingsValid;
			if (binding.usesSampler) {
				bindingsValid = gl4metalMetal4BindSampler(binding.sampler, binding.slot) && bindingsValid;
			}
		}
		for (const ComputeTextureBinding &binding : imageTextureBindings) {
			bindingsValid = gl4metalMetal4BindTexture(binding.texture, binding.slot) && bindingsValid;
		}
		if (indirect) {
			bindingsValid = gl4metalMetal4UseBuffer(indirectBuffer) && bindingsValid;
		}
		if (bindingsValid && gl4metalMetal4PrepareDispatch()) {
			[metal4Encoder setArgumentTable:ctx.metal4ComputeArgumentTable];
			for (GLuint buffer_id : usedBufferNames) gl4metalMarkMetal4BufferUsed(buffer_id);
			const MTLSize threadsPerGroup = MTLSizeMake(program->compute_workgroup_size[0],
				program->compute_workgroup_size[1], program->compute_workgroup_size[2]);
			if (indirect) {
				[metal4Encoder dispatchThreadgroupsWithIndirectBuffer:
					indirectBuffer.gpuAddress + (MTLGPUAddress)indirectOffset threadsPerThreadgroup:threadsPerGroup];
			} else {
				[metal4Encoder dispatchThreadgroups:MTLSizeMake(num_groups_x, num_groups_y, num_groups_z)
					threadsPerThreadgroup:threadsPerGroup];
			}
			return;
		}
		// No dispatch has been encoded yet. Commit any earlier MTL4 work in this batch,
		// then execute this dispatch through the classic path rather than dropping it.
		GL4METAL_LOG("glDispatchCompute: Metal 4 binding/encoder setup failed; falling back to classic compute");
		gl4metalCommitMetal4Compute();
	}

	// Classic fallback: close a render pass before compute work. If the previous GL
	// command was another compute dispatch, keep its encoder open and reuse its state cache.
	if (ctx.currentEncoder) endRenderEncoder();
	if (!gl4metalEnsureClassicCommandBuffer()) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	if (!ctx.currentCommandBuffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	id<MTLComputeCommandEncoder> computeEncoder = ctx.currentComputeEncoder;
	if (!computeEncoder) {
		computeEncoder = [ctx.currentCommandBuffer computeCommandEncoder];
		if (!computeEncoder) {
			gl4metalSetError(GL_OUT_OF_MEMORY);
			GL4METAL_LOG("glDispatchCompute: Metal could not create a compute command encoder");
			return;
		}
		ctx.currentComputeEncoder = computeEncoder;
	}
	ComputeEncoderStateCache &encoderCache = computeEncoderStateFor(computeEncoder);
	bindComputePipelineCached(encoderCache, computeEncoder, program->compute_pipeline);
	for (const ComputeBufferBinding &binding : storageBufferBindings) {
		bindComputeBufferCached(encoderCache, computeEncoder, binding.buffer, binding.offset, binding.slot);
	}
	for (const ComputeBufferBinding &binding : uniformBufferBindings) {
		bindComputeBufferCached(encoderCache, computeEncoder, binding.buffer, binding.offset, binding.slot);
	}
	if (program->needs_buffer_size_buffer[2]) {
		bindComputeBufferSizesCached(encoderCache, computeEncoder, computeBufferSizes);
	}
	if (!program->uniform_data[2].empty()) {
		const std::vector<uint8_t> &uniformBytes = program->uniform_data[2];
		if (uniformBytes.size() <= 4096) {
			bindComputeDefaultUniformCached(encoderCache, computeEncoder, ctx.current_program,
				program->uniform_data_revision[2], uniformBytes);
		} else {
			__strong id<MTLBuffer> uniformBuffer = nil;
			NSUInteger uniformOffset = 0;
			if (!programUniformBufferForStage(*program, 2, uniformBuffer, uniformOffset)) {
				[computeEncoder endEncoding];
				ctx.currentComputeEncoder = nil;
				invalidateComputeEncoderStateCache();
				gl4metalSetError(GL_OUT_OF_MEMORY);
				return;
			}
			bindComputeBufferCached(encoderCache, computeEncoder, uniformBuffer, uniformOffset, GL4METAL_UNIFORM_BUFFER_INDEX);
		}
	}
	for (const ComputeTextureBinding &binding : sampledTextureBindings) {
		bindComputeTextureCached(encoderCache, computeEncoder, binding.texture, binding.slot);
		if (binding.usesSampler) bindComputeSamplerCached(encoderCache, computeEncoder, binding.sampler, binding.slot);
	}
	for (const ComputeTextureBinding &binding : imageTextureBindings) {
		bindComputeTextureCached(encoderCache, computeEncoder, binding.texture, binding.slot);
	}
	if (!usedBufferNames.empty()) markBuffersUsed(usedBufferNames.data(), usedBufferNames.size());
	const MTLSize threadsPerGroup = MTLSizeMake(program->compute_workgroup_size[0],
		program->compute_workgroup_size[1], program->compute_workgroup_size[2]);
	if (indirect) {
		[computeEncoder dispatchThreadgroupsWithIndirectBuffer:indirectBuffer
			indirectBufferOffset:indirectOffset threadsPerThreadgroup:threadsPerGroup];
	} else {
		[computeEncoder dispatchThreadgroups:MTLSizeMake(num_groups_x, num_groups_y, num_groups_z)
			threadsPerThreadgroup:threadsPerGroup];
	}
	// Leave the encoder active for the next dispatch. It is ended by a memory
	// barrier, a graphics/transfer transition, or command-buffer submission.
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
		gl4metalSetError(GL_INVALID_OPERATION);
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
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDetachShader: shader %u is not attached to program %u", shader, program);
		return;
	}
	p->shaders.erase(it);
	releaseShaderIfUnused(shader);
}

void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders) {
	if (!count || maxCount < 0 || (maxCount > 0 && !shaders)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ProgramObject *programObject = findProgram(program);
	if (!programObject) {
		gl4metalSetError(GL_INVALID_VALUE);
		*count = 0;
		return;
	}
	GLsizei copiedCount = 0;
	for (GLuint shaderName : programObject->shaders) {
		if (copiedCount >= maxCount) break;
		shaders[copiedCount++] = shaderName;
	}
	*count = copiedCount;
}

void glLinkProgram(GLuint program) {
	ProgramObject *pp = findProgram(program);
	if (!pp) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glLinkProgram: invalid program %u", program);
		return;
	}
	ProgramObject &p = *pp;
	ProgramRelinkGuard relinkGuard(p, ctx.current_program == program);

	// Link status is reset immediately. If this is the current program and the
	// relink fails, relinkGuard preserves its old executable until glUseProgram
	// removes it from current state, matching GL's active-program rule.
	p.linked = GL_FALSE;
	p.has_executable = false;
	p.info_log.clear();
	p.vertex_function = nil;
	p.fragment_function = nil;
	p.vertex_function_flipped = nil;
	p.vertex_stage_cache_id = 0;
	p.fragment_stage_cache_id = 0;
	p.compute_pipeline = nil;
	p.compute_workgroup_size[0] = p.compute_workgroup_size[1] = p.compute_workgroup_size[2] = 1;
	p.is_compute = false;
	for (int stage = 0; stage < 3; ++stage) p.needs_buffer_size_buffer[stage] = false;
	p.validated = GL_FALSE;
	p.storage_buffers.clear();
	p.storage_images.clear();
	p.image_units.clear();
	p.attribs.clear();
	p.uniforms.clear();
	p.sampler_uniform_indices.clear();
	p.active_uniforms.clear();
	p.locations.clear();
	p.uniform_data[0].clear();
	p.uniform_data[1].clear();
	p.uniform_data[2].clear();
	for (NSUInteger stage = 0; stage < 3; ++stage) {
		p.uniform_gpu_buffer[stage] = nil;
		p.uniform_gpu_revision[stage] = 0;
		p.uniform_gpu_device_identity[stage] = 0;
		p.uniform_gpu_arena_generation[stage] = 0;
		p.uniform_gpu_offset[stage] = 0;
		p.uniform_gpu_is_transient[stage] = false;
		++p.uniform_data_revision[stage];
		if (p.uniform_data_revision[stage] == 0) ++p.uniform_data_revision[stage];
	}
	p.frag_outputs.clear();
	p.blocks.clear();
	p.uses_gl_frag_coord = false;

	const ShaderObject *vs = nullptr;
	const ShaderObject *fs = nullptr;
	const ShaderObject *cs = nullptr;
	unsigned vertexShaderCount = 0;
	unsigned fragmentShaderCount = 0;
	unsigned computeShaderCount = 0;
	for (GLuint shaderName : p.shaders) {
		auto shaderIt = ctx.shader_objects.find(shaderName);
		if (shaderIt == ctx.shader_objects.end()) {
			p.info_log = "an attached shader object no longer exists\n";
			return;
		}
		const ShaderObject &attachedShader = shaderIt->second;
		if (!attachedShader.compiled || !attachedShader.compiled_stage) {
			p.info_log = "an attached shader is not compiled\n";
			return;
		}
		switch (attachedShader.type) {
			case GL_VERTEX_SHADER:
				vs = &attachedShader;
				++vertexShaderCount;
				break;
			case GL_FRAGMENT_SHADER:
				fs = &attachedShader;
				++fragmentShaderCount;
				break;
			case GL_COMPUTE_SHADER:
				cs = &attachedShader;
				++computeShaderCount;
				break;
			default:
				p.info_log = "an attached shader has an unsupported stage type\n";
				return;
		}
	}
	if (computeShaderCount != 0) {
		if (computeShaderCount != 1 || vertexShaderCount != 0 || fragmentShaderCount != 0) {
			p.info_log = "a compute program must contain exactly one compute shader and no graphics stages\n";
			return;
		}
		if (linkComputeProgram(program, p, *cs)) relinkGuard.commit();
		return;
	}
	if (vertexShaderCount != 1 || fragmentShaderCount != 1 || !vs || !fs) {
		p.info_log = "a graphics program must contain exactly one vertex shader and one fragment shader\n";
		return;
	}

	// GLSL -> SPIR-V -> MSL for both stages at once (see shadertranslator.cpp).
	TranslatedProgram tp;
	try {
		if (!translateProgram(vs->compiled_stage, fs->compiled_stage, tp, p.frag_data_bindings)) {
			p.info_log = tp.log;
			GL4METAL_LOG("glLinkProgram(%u): translation failed:\n%s", program, tp.log.c_str());
			return;
		}
	} catch (const std::exception &exception) {
		p.info_log = std::string("shader translation raised an exception: ") + exception.what() + "\n";
		GL4METAL_LOG("glLinkProgram(%u): %s", program, p.info_log.c_str());
		return;
	} catch (...) {
		p.info_log = "shader translation raised an unknown exception\n";
		return;
	}
	if (dumpEnabled()) {
		if (!tp.log.empty()) {
			GL4METAL_LOG("program %u translation notes:\n%s", program, tp.log.c_str());
		}
		GL4METAL_LOG("---- MSL vertex, program %u ----\n%s\n---- MSL fragment, program %u ----\n%s\n---- end MSL ----",
					 program, tp.msl[0].c_str(), program, tp.msl[1].c_str());
	}

	// MSL -> MTLFunction. Cache stage functions by exact generated source so
	// identical Minecraft/Iris stages shared by multiple GL program objects are
	// compiled only once. Function-constant Y-flip variants share one Metal library.
	VertexStageCacheEntry vertexEntry;
	FragmentStageCacheEntry fragmentEntry;
	const bool vertexReady = vertexStageFunctionsForMSL(tp.msl[0], vertexEntry);
	const bool fragmentReady = fragmentStageFunctionForMSL(tp.msl[1], fragmentEntry);
	id<MTLFunction> vertexFunction = vertexEntry.normalFunction;
	id<MTLFunction> flippedFunction = vertexEntry.flippedFunction;
	id<MTLFunction> fragmentFunction = fragmentEntry.function;
	if (!vertexReady) {
		p.info_log += "vertex stage failed to compile/specialize as Metal:\n" + vertexEntry.error + "\n";
	}
	if (!fragmentReady) {
		p.info_log += "fragment stage failed to compile as Metal:\n" + fragmentEntry.error + "\n";
	}
	if (!vertexFunction || !fragmentFunction || !flippedFunction) {
		GL4METAL_LOG("glLinkProgram(%u): %s", program, p.info_log.c_str());
		return;
	}

	// The translator reflects storage blocks independently of UBOs and samplers.
	// Preserve this GL-facing reflection for indexed binding and program-interface queries.
	p.storage_buffers = tp.storage_buffers;
	for (int stage = 0; stage < 3; ++stage)
		p.needs_buffer_size_buffer[stage] = tp.needs_buffer_size_buffer[stage];

	// GLSL layout(location=...) has priority over glBindAttribLocation. The latter
	// only supplies a GL VAO index for attributes without an explicit source location.
	// Reserve every explicit location and API binding before assigning automatic slots,
	// otherwise an unbound attribute can accidentally alias a shader-declared location.
	bool used[16] = {};
	std::vector<bool> hasExplicitLocation;
	collectCompiledStageExplicitVertexAttributeLocations(vs->compiled_stage, tp.attribs, hasExplicitLocation);
	for (size_t attribIndex = 0; attribIndex < tp.attribs.size(); ++attribIndex) {
		const TranslatedAttrib &attribute = tp.attribs[attribIndex];
		if (hasExplicitLocation[attribIndex]) {
			if (attribute.location >= 16) {
				p.info_log = "explicit vertex attribute location exceeds GL_MAX_VERTEX_ATTRIBS for '" + attribute.name + "'\n";
				return;
			}
			used[attribute.location] = true;
			continue;
		}
		auto bound = p.attrib_bindings.find(attribute.name);
		if (bound != p.attrib_bindings.end()) {
			if (bound->second >= 16) {
				p.info_log = "glBindAttribLocation index exceeds GL_MAX_VERTEX_ATTRIBS for '" + attribute.name + "'\n";
				return;
			}
			used[bound->second] = true;
		}
	}
	for (size_t attribIndex = 0; attribIndex < tp.attribs.size(); ++attribIndex) {
		const TranslatedAttrib &attribute = tp.attribs[attribIndex];
		ProgramAttrib pa;
		pa.name = attribute.name;
		pa.msl_location = attribute.location;
		pa.vecsize = attribute.vecsize;
		pa.is_integer = attribute.is_integer;
		pa.is_unsigned = attribute.is_unsigned;
		if (hasExplicitLocation[attribIndex]) {
			pa.gl_index = (GLint)attribute.location;
		} else {
			auto bound = p.attrib_bindings.find(attribute.name);
			if (bound != p.attrib_bindings.end()) {
				pa.gl_index = (GLint)bound->second;
			} else {
				int slot = 0;
				while (slot < 16 && used[slot]) ++slot;
				if (slot == 16) {
					p.info_log = "too many active vertex attributes\n";
					return;
				}
				used[slot] = true;
				pa.gl_index = slot;
			}
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
		const uint32_t uniformIndex = (uint32_t)p.uniforms.size();
		p.uniforms.push_back(pu);
		if (u.kind == TranslatedUniform::Sampler &&
			(u.stage[0].present || u.stage[1].present)) {
			p.sampler_uniform_indices.push_back(uniformIndex);
		}
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
	p.vertex_stage_cache_id = vertexEntry.cacheId;
	p.fragment_stage_cache_id = fragmentEntry.cacheId;
	p.uses_gl_frag_coord = tp.uses_gl_frag_coord;
	p.has_executable = true;
	p.linked = GL_TRUE;
	relinkGuard.commit();

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
		} else if (old && !old->linked && old->has_executable) {
			clearProgramExecutable(*old);
		}
	}
}

void glValidateProgram(GLuint program) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	p->validated = p->linked;
}

void glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
	if (!params) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
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
		case GL_TRANSFORM_FEEDBACK_VARYINGS:
			*params = 0; // capture layout is rejected until vertex-output instrumentation exists
			break;
		case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:
			*params = GL_INTERLEAVED_ATTRIBS;
			break;
		case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
			*params = 0;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetProgramiv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
	if (bufSize < 0 || (bufSize > 0 && !infoLog)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		return;
	}
	copyGLString(p->info_log, bufSize, length, infoLog);
}

// MARK: - Vertex attribute locations

void glBindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || index >= 16 || !name) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindAttribLocation: invalid program %u, index %u or null name", program, index);
		return;
	}
	if (strncmp(name, "gl_", 3) == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindAttribLocation: attribute name '%s' uses the reserved gl_ prefix", name);
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
	if (bufSize < 0 || (bufSize > 0 && !name)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
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
	ProgramObject *p = findProgram(program);
	if (!p || !name || !name[0] || color >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindFragDataLocation: invalid program %u, color %u or output name", program, color);
		return;
	}
	if (strncmp(name, "gl_", 3) == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindFragDataLocation: output name '%s' uses the reserved gl_ prefix", name);
		return;
	}
	p->frag_data_bindings[name] = color; // takes effect at the next glLinkProgram
}

void glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber, GLuint index, const GLchar *name) {
	if (!findProgram(program) || !name || !name[0] || colorNumber >= GL4METAL_MAX_COLOR_ATTACHMENTS) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (index != 0) {
		gl4metalSetError(GL_INVALID_OPERATION); // dual-source fragment outputs are not supported
		return;
	}
	glBindFragDataLocation(program, colorNumber, name);
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
