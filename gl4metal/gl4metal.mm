#import "gl4metal.h"

#include <mutex>
#include <unistd.h>
#include <chrono>
#include <limits>

// CPU-write/GPU-read staging allocations are never read back by the CPU.
static constexpr MTLResourceOptions kMetal4TransientUploadOptions =
	(MTLResourceOptions)(MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined);

thread_local MTLContext ctx; // independent GL/Metal state for the calling thread

static std::mutex *gl4metalInitMutex = new std::mutex();
// MTLContext is thread_local, so its initialization guard must be thread_local too.
// A process-wide flag made later GL threads return success while their own ctx still
// had a null device/queue and otherwise-zero GL state.
static thread_local bool gl4metalInitialized = false;

bool gl4metalWaitForMetal4Event(uint64_t targetValue, uint64_t timeoutMilliseconds) {
	if (!ctx.metal4InteropEvent || targetValue == 0) return true;
	const auto startTime = std::chrono::steady_clock::now();
	while (ctx.metal4InteropEvent.signaledValue < targetValue) {
		if (timeoutMilliseconds != UINT64_MAX) {
			const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - startTime).count();
			if ((uint64_t)elapsed >= timeoutMilliseconds) return false;
		}
		usleep(200);
	}
	return true;
}

static bool initializeMetal4Backend(void) {
	if (!ctx.device) return false;
	const char *disableValue = std::getenv("GL4METAL_DISABLE_METAL4");
	if (disableValue && disableValue[0] == '1') {
		GL4METAL_LOG("Metal 4 disabled by GL4METAL_DISABLE_METAL4=1");
		return false;
	}
	// Metal 4 command APIs are not available on Apple6 (A13). Do not infer
	// support from the OS version alone: require both the Apple7 GPU feature set
	// and the Metal4 feature set before creating any MTL4 object.
	if (@available(iOS 26.0, *)) {
		const bool apple7 = [ctx.device supportsFamily:MTLGPUFamilyApple7];
		const bool metal4 = [ctx.device supportsFamily:MTLGPUFamilyMetal4];
		if (!apple7 || !metal4) {
			if (!apple7 && !metal4) {
				GL4METAL_LOG("Metal 4 unavailable on this device, using complete classic Metal path");
			}
			
			GL4METAL_LOG("Metal 4 command path unavailable: Apple7=%d Metal4=%d, using classic Metal", apple7 ? 1 : 0, metal4 ? 1 : 0);
			return false;
		}

		NSError *error = nil;
		MTL4CompilerDescriptor *compilerDescriptor = [[MTL4CompilerDescriptor alloc] init];
		compilerDescriptor.label = @"GL4Metal Metal 4 compiler";
		ctx.metal4Compiler = [ctx.device newCompilerWithDescriptor:compilerDescriptor error:&error];
		ctx.metal4CommandQueue = [ctx.device newMTL4CommandQueue];
		ctx.metal4InteropEvent = [ctx.device newSharedEvent];
		MTLResidencySetDescriptor *residencyDescriptor = [[MTLResidencySetDescriptor alloc] init];
		residencyDescriptor.initialCapacity = 64;
		for (NSUInteger slot = 0; slot < 3; ++slot) {
			ctx.metal4CommandAllocators[slot] = [ctx.device newCommandAllocator];
			ctx.metal4CommandBuffers[slot] = [ctx.device newCommandBuffer];
			residencyDescriptor.label = [NSString stringWithFormat:@"GL4Metal compute residency slot %lu", (unsigned long)slot];
			ctx.metal4ResidencySets[slot] = [ctx.device newResidencySetWithDescriptor:residencyDescriptor error:&error];
		}
		MTL4ArgumentTableDescriptor *tableDescriptor = [[MTL4ArgumentTableDescriptor alloc] init];
		tableDescriptor.maxBufferBindCount = GL4METAL_UNIFORM_BUFFER_INDEX + 1;
		tableDescriptor.maxTextureBindCount = 32;
		tableDescriptor.maxSamplerStateBindCount = 32;
		ctx.metal4ComputeArgumentTable = [ctx.device newArgumentTableWithDescriptor:tableDescriptor error:&error];
		// Every ring slot needs a residency set as well as an allocator and command
		// buffer. Enabling Metal 4 with a nil residency set makes BeginMetal4Compute()
		// fail on every submission and silently routes work back to the classic queue.
		if (!ctx.metal4CommandQueue || !ctx.metal4InteropEvent || !ctx.metal4Compiler ||
			!ctx.metal4CommandAllocators[0] || !ctx.metal4CommandAllocators[1] ||
			!ctx.metal4CommandAllocators[2] || !ctx.metal4CommandBuffers[0] ||
			!ctx.metal4CommandBuffers[1] || !ctx.metal4CommandBuffers[2] ||
			!ctx.metal4ResidencySets[0] || !ctx.metal4ResidencySets[1] ||
			!ctx.metal4ResidencySets[2] || !ctx.metal4ComputeArgumentTable) {
			GL4METAL_LOG("Metal 4 initialization incomplete; using classic path: %s",
				error.localizedDescription.UTF8String ?: "queue/event/allocator/residency-set/argument-table creation failed");
			ctx.metal4Compiler = nil;
			ctx.metal4CommandQueue = nil;
			ctx.metal4InteropEvent = nil;
			ctx.metal4ComputeArgumentTable = nil;
			for (NSUInteger slot = 0; slot < 3; ++slot) {
				ctx.metal4CommandAllocators[slot] = nil;
				ctx.metal4CommandBuffers[slot] = nil;
				ctx.metal4ResidencySets[slot] = nil;
				ctx.metal4TransientBuffers[slot] = nil;
				ctx.metal4TransientBufferOffsets[slot] = 0;
			}
			return false;
		}
		ctx.metal4Enabled = true;
		GL4METAL_LOG("Metal 4 compute command path enabled on %s (Apple7=1 Metal4=1)",
			ctx.device.name.UTF8String ?: "(unnamed)");
		return true;
	}
	return false;
}

void gl4metalReapMetal4Submissions(void) {
	if (!ctx.metal4Enabled || !ctx.metal4InteropEvent) return;
	const uint64_t completedValue = ctx.metal4InteropEvent.signaledValue;
	while (!ctx.metal4InFlightSubmissions.empty() &&
		ctx.metal4InFlightSubmissions.front().event_value <= completedValue) {
		Metal4InFlightSubmission &completed = ctx.metal4InFlightSubmissions.front();
		// Reusable residency sets hold allocation references after command completion.
		// Remove them now (not only when the ring slot is reused) to bound idle memory.
		if (completed.residency_set) {
			[completed.residency_set removeAllAllocations];
			[completed.residency_set commit];
		}
		ctx.metal4InFlightSubmissions.pop_front();
	}
}

static void registerMetal4Resource(id<MTLResource> resource) {
	if (!resource || !ctx.currentMetal4ResidencySet) return;
	const void *resourceKey = (__bridge const void *)resource;
	if (!ctx.metal4CurrentResourceKeys.insert(resourceKey).second) return;
	[ctx.currentMetal4ResidencySet addAllocation:resource];
	ctx.metal4CurrentResources.push_back(resource);
}

static void registerMetal4Sampler(id<MTLSamplerState> sampler) {
	if (!sampler) return;
	for (const auto &existing : ctx.metal4CurrentSamplers) {
		if (existing == sampler) return;
	}
	ctx.metal4CurrentSamplers.push_back(sampler);
}

bool gl4metalBeginMetal4Compute(void) {
	if (!ctx.metal4Enabled || !ctx.metal4CommandQueue || !ctx.metal4InteropEvent ||
		!ctx.metal4ComputeArgumentTable) return false;
	if (ctx.currentMetal4CommandBuffer && ctx.currentMetal4ComputeEncoder) return true;

	// Metal 4 and classic work are on distinct queues. Finish any classic work and
	// make the Metal 4 queue wait on its event before submitting dependent compute.
	if (ctx.currentEncoder) endRenderEncoder();
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
		extern void invalidateComputeEncoderStateCache(void);
		invalidateComputeEncoderStateCache();
	}
	if (ctx.currentCommandBuffer) gl4metalCommitClassicCommandBuffer();

	const uint32_t allocatorIndex = ctx.metal4AllocatorCursor++ % 3u;
	const uint64_t allocatorCompletionValue = ctx.metal4AllocatorEventValues[allocatorIndex];
	if (allocatorCompletionValue && !gl4metalWaitForMetal4Event(allocatorCompletionValue, 5000)) {
		GL4METAL_LOG("Metal 4 allocator %u is still in flight at event value %llu; falling back to classic compute",
			allocatorIndex, (unsigned long long)allocatorCompletionValue);
		return false;
	}
	gl4metalReapMetal4Submissions();
	id<MTL4CommandAllocator> allocator = ctx.metal4CommandAllocators[allocatorIndex];
	if (!allocator || !ctx.metal4CommandBuffers[allocatorIndex] ||
		!ctx.metal4ResidencySets[allocatorIndex]) return false;
	[allocator reset];
	ctx.metal4CurrentAllocatorIndex = allocatorIndex;
	ctx.metal4TransientBufferOffsets[allocatorIndex] = 0;

	if (ctx.metal4LastClassicEventValue > 0) {
		[ctx.metal4CommandQueue waitForEvent:ctx.metal4InteropEvent value:ctx.metal4LastClassicEventValue];
	}

	id<MTL4CommandBuffer> commandBuffer = ctx.metal4CommandBuffers[allocatorIndex];
	if (!commandBuffer) return false;
	[commandBuffer beginCommandBufferWithAllocator:allocator];
	commandBuffer.label = @"GL4Metal compute workload";

	id<MTLResidencySet> residencySet = ctx.metal4ResidencySets[allocatorIndex];
	// gl4metalReapMetal4Submissions() clears this slot's previous completed set.
	// Keep the set attached while new allocations are staged and committed below.
	[commandBuffer useResidencySet:residencySet];
	id<MTL4ComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
	if (!computeEncoder) {
		[commandBuffer endCommandBuffer];
		return false;
	}

	// Metal 4 doesn't automatically track resource hazards. A residency set only
	// makes allocations resident; it does not make writes visible to later stages.
	// Carry producer stages across command-buffer boundaries and place the consumer
	// barrier in the new encoder, before any command can access those resources.
	const MTLStages queue_producer_stages = (MTLStages)ctx.metal4PendingQueueBarrierStages;
	if (queue_producer_stages != 0) {
		const MTLStages queue_consumer_stages = (MTLStages)(MTLStageDispatch | MTLStageBlit);
		[computeEncoder barrierAfterQueueStages:queue_producer_stages
			beforeStages:queue_consumer_stages visibilityOptions:MTL4VisibilityOptionDevice];
		ctx.metal4PendingQueueBarrierStages = 0;
	}

	ctx.currentMetal4CommandBuffer = commandBuffer;
	ctx.currentMetal4ComputeEncoder = computeEncoder;
	ctx.currentMetal4ResidencySet = residencySet;
	ctx.metal4CurrentResources.clear();
	ctx.metal4CurrentSamplers.clear();
	ctx.metal4CurrentComputePipelines.clear();
	ctx.metal4CurrentResourceKeys.clear();
	ctx.metal4BlitReadResourceKeys.clear();
	ctx.metal4BlitWriteResourceKeys.clear();
	ctx.metal4CurrentEventValue = ++ctx.metal4NextEventValue;
	ctx.metal4LastWorkStage = 0;
	ctx.metal4CurrentEncoderStages = 0;
	return true;
}

bool gl4metalMetal4PrepareDispatch(void) {
	if (!gl4metalBeginMetal4Compute() || !ctx.currentMetal4ComputeEncoder) return false;
	if (ctx.metal4LastWorkStage == 2) {
		[ctx.currentMetal4ComputeEncoder barrierAfterEncoderStages:MTLStageBlit
			beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
		// The stage barrier makes all earlier blit writes visible to dispatch.
		ctx.metal4BlitReadResourceKeys.clear();
		ctx.metal4BlitWriteResourceKeys.clear();
	}
	ctx.metal4LastWorkStage = 1;
	ctx.metal4CurrentEncoderStages |= (uint32_t)MTLStageDispatch;
	return true;
}

static const void *metal4BlitResourceKey(id<MTLResource> resource) {
	if (!resource) return nullptr;

	// Texture views alias their parent allocation, and buffer-backed textures alias
	// the source MTLBuffer. Key the owning allocation rather than the view object so
	// a copy through one view cannot miss a dependency recorded through another.
	if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
		id<MTLTexture> texture = (id<MTLTexture>)resource;
		id<MTLTexture> parentTexture = texture.parentTexture;
		while (parentTexture) {
			texture = parentTexture;
			parentTexture = texture.parentTexture;
		}
		if (texture.buffer) return (__bridge const void *)texture.buffer;
		return (__bridge const void *)texture;
	}
	return (__bridge const void *)resource;
}

static bool metal4BlitContainsResourceKey(const std::vector<const void *> &keys, const void *key) {
	return key && std::find(keys.begin(), keys.end(), key) != keys.end();
}

static void metal4BlitRememberResourceKey(std::vector<const void *> &keys, const void *key) {
	if (key && !metal4BlitContainsResourceKey(keys, key)) keys.push_back(key);
}

static bool prepareMetal4BlitStage(id<MTLResource> readResource, id<MTLResource> writeResource) {
	if (!gl4metalBeginMetal4Compute() || !ctx.currentMetal4ComputeEncoder) return false;

	const void *readKey = metal4BlitResourceKey(readResource);
	const void *writeKey = metal4BlitResourceKey(writeResource);
	if (ctx.metal4LastWorkStage == 1) {
		[ctx.currentMetal4ComputeEncoder barrierAfterEncoderStages:MTLStageDispatch
			beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
		// Dispatch->blit orders all shader writes. Blit hazard tracking starts fresh.
		ctx.metal4BlitReadResourceKeys.clear();
		ctx.metal4BlitWriteResourceKeys.clear();
	} else if (ctx.metal4LastWorkStage == 2) {
		// Do not serialize unrelated texture uploads/copies. A blit->blit barrier is
		// needed only for RAW, WAR, or WAW access to the same Metal resource.
		const bool readAfterWrite = metal4BlitContainsResourceKey(ctx.metal4BlitWriteResourceKeys, readKey);
		const bool writeAfterRead = metal4BlitContainsResourceKey(ctx.metal4BlitReadResourceKeys, writeKey);
		const bool writeAfterWrite = metal4BlitContainsResourceKey(ctx.metal4BlitWriteResourceKeys, writeKey);
		if (readAfterWrite || writeAfterRead || writeAfterWrite) {
			[ctx.currentMetal4ComputeEncoder barrierAfterEncoderStages:MTLStageBlit
				beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
			// This barrier publishes every prior blit write, not just one resource.
			ctx.metal4BlitReadResourceKeys.clear();
			ctx.metal4BlitWriteResourceKeys.clear();
		}
	}

	metal4BlitRememberResourceKey(ctx.metal4BlitReadResourceKeys, readKey);
	metal4BlitRememberResourceKey(ctx.metal4BlitWriteResourceKeys, writeKey);
	ctx.metal4LastWorkStage = 2;
	ctx.metal4CurrentEncoderStages |= (uint32_t)MTLStageBlit;
	return true;
}

bool gl4metalMetal4StageBytes(const void *bytes, NSUInteger length, id<MTLBuffer> __strong *out_buffer, NSUInteger *out_offset) {
	if (!bytes || length == 0 || !out_buffer || !out_offset || !ctx.device ||
		length > ctx.device.maxBufferLength || !gl4metalBeginMetal4Compute()) return false;
	const uint32_t allocatorIndex = ctx.metal4CurrentAllocatorIndex;
	const NSUInteger currentOffset = ctx.metal4TransientBufferOffsets[allocatorIndex];
	if (currentOffset > std::numeric_limits<NSUInteger>::max() - 255u) return false;
	const NSUInteger alignedOffset = (currentOffset + 255u) & ~(NSUInteger)255u;
	if (alignedOffset > std::numeric_limits<NSUInteger>::max() - length) return false;
	const NSUInteger requiredEnd = alignedOffset + length;
	id<MTLBuffer> transientBuffer = ctx.metal4TransientBuffers[allocatorIndex];
	if (!transientBuffer || requiredEnd > transientBuffer.length) {
		NSUInteger capacity = transientBuffer ? transientBuffer.length : (NSUInteger)(256u * 1024u);
		capacity = std::max<NSUInteger>(capacity, 256u * 1024u);
		while (capacity < requiredEnd) {
			if (capacity > ctx.device.maxBufferLength / 2u) {
				capacity = ctx.device.maxBufferLength;
				break;
			}
			capacity *= 2u;
		}
		if (capacity < requiredEnd || capacity > ctx.device.maxBufferLength) return false;
		id<MTLBuffer> replacement = [ctx.device newBufferWithLength:capacity options:kMetal4TransientUploadOptions];
		if (!replacement) return false;
		ctx.metal4TransientBuffers[allocatorIndex] = replacement;
		transientBuffer = replacement;
	}
	memcpy((uint8_t *)transientBuffer.contents + alignedOffset, bytes, (size_t)length);
	registerMetal4Resource(transientBuffer);
	ctx.metal4TransientBufferOffsets[allocatorIndex] = requiredEnd;
	*out_buffer = transientBuffer;
	*out_offset = alignedOffset;
	return true;
}

bool gl4metalMetal4CopyBuffer(id<MTLBuffer> source_buffer, NSUInteger source_offset,
	id<MTLBuffer> destination_buffer, NSUInteger destination_offset, NSUInteger size) {
	if (!source_buffer || !destination_buffer || size == 0 ||
		source_offset > source_buffer.length || size > source_buffer.length - source_offset ||
		destination_offset > destination_buffer.length || size > destination_buffer.length - destination_offset) return false;
	if (source_buffer == destination_buffer) {
		const NSUInteger source_end = source_offset + size;
		const NSUInteger destination_end = destination_offset + size;
		if (source_offset < destination_end && destination_offset < source_end) return false;
	}
	if (!prepareMetal4BlitStage(source_buffer, destination_buffer)) return false;
	registerMetal4Resource(source_buffer);
	registerMetal4Resource(destination_buffer);
	[ctx.currentMetal4ComputeEncoder copyFromBuffer:source_buffer sourceOffset:source_offset
		toBuffer:destination_buffer destinationOffset:destination_offset size:size];
	return true;
}

bool gl4metalMetal4CopyBufferToTexture(id<MTLBuffer> source_buffer, NSUInteger source_offset,
	NSUInteger source_bytes_per_row, NSUInteger source_bytes_per_image, MTLSize source_size,
	id<MTLTexture> destination_texture, NSUInteger destination_slice, NSUInteger destination_level, MTLOrigin destination_origin) {
	if (!source_buffer || !destination_texture || source_size.width == 0 || source_size.height == 0 || source_size.depth == 0 ||
		source_offset > source_buffer.length || !prepareMetal4BlitStage(source_buffer, destination_texture)) return false;
	registerMetal4Resource(source_buffer);
	registerMetal4Resource(destination_texture);
	[ctx.currentMetal4ComputeEncoder copyFromBuffer:source_buffer sourceOffset:source_offset
		sourceBytesPerRow:source_bytes_per_row sourceBytesPerImage:source_bytes_per_image sourceSize:source_size
		toTexture:destination_texture destinationSlice:destination_slice destinationLevel:destination_level
		destinationOrigin:destination_origin];
	return true;
}

bool gl4metalMetal4CopyBufferToTextureOptions(id<MTLBuffer> source_buffer, NSUInteger source_offset,
	NSUInteger source_bytes_per_row, NSUInteger source_bytes_per_image, MTLSize source_size,
	id<MTLTexture> destination_texture, NSUInteger destination_slice, NSUInteger destination_level,
	MTLOrigin destination_origin, MTLBlitOption options) {
	if (!source_buffer || !destination_texture || source_size.width == 0 || source_size.height == 0 || source_size.depth == 0 ||
		source_offset > source_buffer.length || !prepareMetal4BlitStage(source_buffer, destination_texture)) return false;
	registerMetal4Resource(source_buffer);
	registerMetal4Resource(destination_texture);
	[ctx.currentMetal4ComputeEncoder copyFromBuffer:source_buffer sourceOffset:source_offset
		sourceBytesPerRow:source_bytes_per_row sourceBytesPerImage:source_bytes_per_image sourceSize:source_size
		toTexture:destination_texture destinationSlice:destination_slice destinationLevel:destination_level
		destinationOrigin:destination_origin options:options];
	return true;
}

bool gl4metalMetal4CopyTextureRegion(id<MTLTexture> source_texture, NSUInteger source_slice, NSUInteger source_level,
	MTLOrigin source_origin, MTLSize source_size, id<MTLTexture> destination_texture,
	NSUInteger destination_slice, NSUInteger destination_level, MTLOrigin destination_origin) {
	if (!source_texture || !destination_texture || source_size.width == 0 || source_size.height == 0 || source_size.depth == 0 ||
		source_texture == destination_texture || source_texture.pixelFormat != destination_texture.pixelFormat ||
		source_texture.sampleCount != destination_texture.sampleCount ||
		!prepareMetal4BlitStage(source_texture, destination_texture)) return false;
	registerMetal4Resource(source_texture);
	registerMetal4Resource(destination_texture);
	[ctx.currentMetal4ComputeEncoder copyFromTexture:source_texture sourceSlice:source_slice sourceLevel:source_level
		sourceOrigin:source_origin sourceSize:source_size toTexture:destination_texture
		destinationSlice:destination_slice destinationLevel:destination_level destinationOrigin:destination_origin];
	return true;
}

bool gl4metalMetal4GenerateMipmaps(id<MTLTexture> texture) {
	if (!texture || texture.mipmapLevelCount < 2 || !prepareMetal4BlitStage(texture, texture)) return false;
	registerMetal4Resource(texture);
	[ctx.currentMetal4ComputeEncoder generateMipmapsForTexture:texture];
	return true;
}

bool gl4metalMetal4CopyTextureToBuffer(id<MTLTexture> source_texture, NSUInteger source_slice, NSUInteger source_level,
	MTLOrigin source_origin, MTLSize source_size, id<MTLBuffer> destination_buffer,
	NSUInteger destination_offset, NSUInteger destination_bytes_per_row, NSUInteger destination_bytes_per_image, MTLBlitOption options) {
	if (!source_texture || !destination_buffer || source_size.width == 0 || source_size.height == 0 || source_size.depth == 0 ||
		destination_offset > destination_buffer.length ||
		!prepareMetal4BlitStage(source_texture, destination_buffer)) return false;
	registerMetal4Resource(source_texture);
	registerMetal4Resource(destination_buffer);
	[ctx.currentMetal4ComputeEncoder copyFromTexture:source_texture sourceSlice:source_slice sourceLevel:source_level
		sourceOrigin:source_origin sourceSize:source_size toBuffer:destination_buffer destinationOffset:destination_offset
		destinationBytesPerRow:destination_bytes_per_row destinationBytesPerImage:destination_bytes_per_image options:options];
	return true;
}

bool gl4metalMetal4BindBuffer(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger slot) {
	if (!ctx.metal4Enabled || !ctx.currentMetal4ComputeEncoder || !ctx.metal4ComputeArgumentTable ||
		!buffer || slot > GL4METAL_UNIFORM_BUFFER_INDEX || offset > buffer.length || buffer.gpuAddress == 0) return false;
	const MTLGPUAddress baseAddress = buffer.gpuAddress;
	if (UINT64_MAX - baseAddress < (MTLGPUAddress)offset) return false;
	registerMetal4Resource(buffer);
	[ctx.metal4ComputeArgumentTable setAddress:baseAddress + (MTLGPUAddress)offset atIndex:slot];
	return true;
}

bool gl4metalMetal4UseBuffer(id<MTLBuffer> buffer) {
	if (!ctx.metal4Enabled || !ctx.currentMetal4ComputeEncoder || !buffer || buffer.gpuAddress == 0) return false;
	registerMetal4Resource(buffer);
	return true;
}

bool gl4metalMetal4BindTexture(id<MTLTexture> texture, NSUInteger slot) {
	if (!ctx.metal4Enabled || !ctx.currentMetal4ComputeEncoder || !ctx.metal4ComputeArgumentTable ||
		!texture || slot >= 32) return false;
	registerMetal4Resource(texture);
	[ctx.metal4ComputeArgumentTable setTexture:texture.gpuResourceID atIndex:slot];
	return true;
}

bool gl4metalMetal4BindSampler(id<MTLSamplerState> sampler, NSUInteger slot) {
	if (!ctx.metal4Enabled || !ctx.currentMetal4ComputeEncoder || !ctx.metal4ComputeArgumentTable ||
		!sampler || slot >= 32) return false;
	registerMetal4Sampler(sampler);
	[ctx.metal4ComputeArgumentTable setSamplerState:sampler.gpuResourceID atIndex:slot];
	return true;
}

bool gl4metalMetal4BindBytes(const void *bytes, NSUInteger length, NSUInteger slot) {
	if (slot > GL4METAL_UNIFORM_BUFFER_INDEX) return false;
	id<MTLBuffer> transientBuffer = nil;
	NSUInteger alignedOffset = 0;
	if (!gl4metalMetal4StageBytes(bytes, length, &transientBuffer, &alignedOffset)) return false;
	return gl4metalMetal4BindBuffer(transientBuffer, alignedOffset, slot);
}

void gl4metalMetal4UseComputePipeline(id<MTLComputePipelineState> pipeline) {
	if (!pipeline || !ctx.currentMetal4CommandBuffer) return;
	for (const auto &existing : ctx.metal4CurrentComputePipelines) {
		if (existing == pipeline) return;
	}
	ctx.metal4CurrentComputePipelines.push_back(pipeline);
}

void gl4metalMarkMetal4BufferUsed(GLuint buffer_id) {
	if (!ctx.currentMetal4CommandBuffer || buffer_id == 0) return;
	ctx.buffer_last_use_event_value[buffer_id] = ctx.metal4CurrentEventValue;
}

void gl4metalCommitMetal4Compute(void) {
	if (!ctx.metal4Enabled || !ctx.currentMetal4CommandBuffer) return;
	if (ctx.currentMetal4ComputeEncoder) {
		[ctx.currentMetal4ComputeEncoder endEncoding];
		ctx.currentMetal4ComputeEncoder = nil;
	}
	// Preserve all producer stages so the next MTL4 encoder can insert a
	// consumer queue barrier. The next command buffer may access resources
	// written by any stage in this submission.
	ctx.metal4PendingQueueBarrierStages |= ctx.metal4CurrentEncoderStages;
	ctx.metal4CurrentEncoderStages = 0;
	[ctx.currentMetal4ResidencySet commit];
	[ctx.currentMetal4CommandBuffer endCommandBuffer];
	id<MTL4CommandBuffer> commandBuffer = ctx.currentMetal4CommandBuffer;
	const uint64_t completionValue = ctx.metal4CurrentEventValue;
	const uint32_t allocatorIndex = (ctx.metal4AllocatorCursor - 1u) % 3u;
	ctx.metal4AllocatorEventValues[allocatorIndex] = completionValue;
	[ctx.metal4CommandQueue commit:&commandBuffer count:1];
	[ctx.metal4CommandQueue signalEvent:ctx.metal4InteropEvent value:completionValue];

	Metal4InFlightSubmission submission;
	submission.event_value = completionValue;
	submission.command_buffer = commandBuffer;
	submission.residency_set = ctx.currentMetal4ResidencySet;
	submission.resources.swap(ctx.metal4CurrentResources);
	submission.samplers.swap(ctx.metal4CurrentSamplers);
	submission.compute_pipelines.swap(ctx.metal4CurrentComputePipelines);
	ctx.metal4InFlightSubmissions.push_back(std::move(submission));
	ctx.metal4LastComputeEventValue = completionValue;
	ctx.currentMetal4CommandBuffer = nil;
	ctx.currentMetal4ResidencySet = nil;
	ctx.metal4CurrentEventValue = 0;
	ctx.metal4LastWorkStage = 0;
	ctx.metal4CurrentResourceKeys.clear();
	gl4metalReapMetal4Submissions();
	extern void resetTextureUploadArenaMetal4(id<MTL4CommandBuffer> command_buffer);
	resetTextureUploadArenaMetal4(commandBuffer);
}

bool gl4metalEnsureClassicCommandBuffer(void) {
	if (ctx.metal4Enabled && ctx.currentMetal4CommandBuffer) gl4metalCommitMetal4Compute();

	if (!ctx.currentCommandBuffer) {
		if (!ctx.commandQueue) return false;
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer];
		if (!ctx.currentCommandBuffer) return false;

		if (ctx.metal4Enabled && ctx.metal4LastComputeEventValue > ctx.classicWaitEventValue) {
			[ctx.currentCommandBuffer encodeWaitForEvent:ctx.metal4InteropEvent value:ctx.metal4LastComputeEventValue];
			ctx.classicWaitEventValue = ctx.metal4LastComputeEventValue;
		}
	}
	return ctx.currentCommandBuffer != nil;
}

void gl4metalCommitClassicCommandBuffer(void) {
	if (!ctx.currentCommandBuffer) return;

	if (ctx.metal4Enabled && ctx.metal4InteropEvent) {
		const uint64_t signalValue = ++ctx.metal4NextEventValue;
		[ctx.currentCommandBuffer encodeSignalEvent:ctx.metal4InteropEvent value:signalValue];
		ctx.metal4LastClassicEventValue = signalValue;
	}
	[ctx.currentCommandBuffer commit];
	resetDrawTransientUploadArena(ctx.currentCommandBuffer);
	resetTextureUploadArena(ctx.currentCommandBuffer);
	ctx.last_committed_buffer = ctx.currentCommandBuffer;
	ctx.currentCommandBuffer = nil;
}

// Compute-shader compatibility is exposed only when the active Metal device
// meets the OpenGL compute baseline used by Iris. Device identity is stable for
// the lifetime of MTLContext, so cache this capability instead of repeating
// Objective-C feature queries from every GL limit/string query.
bool gl4metalSupportsComputeShaders(void) {
	if (!ctx.device) return false;
	// MTLContext is thread-local, so cache this query per context/device. Querying a
	// separate system-default device behind ctx.device can advertise compute support
	// for a different adapter and hide a capability mismatch from the active device.
	struct ComputeCapabilityCache {
		__strong id<MTLDevice> device = nil;
		bool initialized = false;
		bool supported = false;
	};
	static thread_local ComputeCapabilityCache cache;
	if (cache.initialized && cache.device == ctx.device) return cache.supported;

	cache.device = ctx.device;
	cache.initialized = true;
	NSUInteger sharedMemoryLimit = 0;
	MTLSize threadgroupLimits = MTLSizeMake(0, 0, 0);
	bool apple6 = false;
	bool supported = false;
	if (@available(iOS 14.0, *)) {
		apple6 = [ctx.device supportsFamily:MTLGPUFamilyApple6];
		sharedMemoryLimit = ctx.device.maxThreadgroupMemoryLength;
		threadgroupLimits = ctx.device.maxThreadsPerThreadgroup;
		supported = apple6 && sharedMemoryLimit >= 32768 &&
			threadgroupLimits.width >= 1024 && threadgroupLimits.height >= 1024 && threadgroupLimits.depth >= 64;
	}
	cache.supported = supported;
	GL4METAL_LOG("Compute capability: Apple6=%d, threads/group=%lu x %lu x %lu, threadgroup memory=%lu, enabled=%d",
		apple6 ? 1 : 0, (unsigned long)threadgroupLimits.width,
		(unsigned long)threadgroupLimits.height, (unsigned long)threadgroupLimits.depth,
		(unsigned long)sharedMemoryLimit, supported ? 1 : 0);
	return cache.supported;
}

NSUInteger gl4metalMaxComputeWorkGroupInvocations(void) {
	return gl4metalSupportsComputeShaders() ? GL4METAL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS : 0u;
}

void initDevice() {
	// Keep the device identity stable. Replacing it after resources have been created
	// would leave textures, buffers and pipeline states belonging to the old device.
	if (!ctx.device) {
		ctx.device = MTLCreateSystemDefaultDevice();
		if (ctx.device) {
			GL4METAL_LOG("Initialized context on device: %s", ctx.device.name.UTF8String ?: "(unnamed)");
		}
	}
	if (ctx.device && !ctx.commandQueue) {
		ctx.commandQueue = [ctx.device newCommandQueue];
	}
	if (ctx.device && !ctx.metal4Enabled) initializeMetal4Backend();
}

GLboolean gl4metalInit() {
	std::lock_guard<std::mutex> initLock(*gl4metalInitMutex);
	if (gl4metalInitialized) return GL_TRUE;
	initDevice();
	if (!ctx.device || !ctx.commandQueue) {
		GL4METAL_LOG("gl4metalInit: Metal device or classic command queue creation failed");
		return GL_FALSE;
	}
	const char *check = getenv("GL4METAL_CHECK_EXPORTS");
	if (check && check[0] == '1') {
		gl4metalCheckExports();
	}

	// Shader packs create many transient FBOs while building/reloading pipelines.
	// Reserve the common working set once so frame-time FBO insertions do not
	// repeatedly rehash the context table.
	ctx.fbo_objects.max_load_factor(0.75f);
	ctx.fbo_objects.reserve(128);
	ctx.pipeline_cache.max_load_factor(0.75f);
	ctx.pipeline_cache.reserve(512);
	ctx.sampler_cache.max_load_factor(0.75f);
	ctx.sampler_cache.reserve(64);

	// OpenGL initial state
	ctx.depth_write_enabled = GL_TRUE;
	ctx.depth_func = GL_LESS;
	ctx.clear_color = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
	ctx.clear_depth = 1.0;
	ctx.clear_stencil = 0;
	ctx.unpack_alignment = 4;
	ctx.pack_alignment = 4;
	ctx.caps[GL_DITHER] = GL_TRUE;
	ctx.caps[GL_MULTISAMPLE] = GL_TRUE;

	ctx.blend_src_rgb = ctx.blend_src_alpha = GL_ONE;
	ctx.blend_dst_rgb = ctx.blend_dst_alpha = GL_ZERO;
	ctx.blend_eq_rgb = ctx.blend_eq_alpha = GL_FUNC_ADD;
	for (int i = 0; i < 4; i++) {
		ctx.blend_color[i] = 0.0f;
		ctx.color_mask[i] = GL_TRUE;
	}
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		ctx.blend_enabled[attachment] = GL_FALSE;
		ctx.blend_buffers[attachment] = BlendBufferState();
		for (int component = 0; component < 4; ++component)
			ctx.color_masks[attachment][component] = GL_TRUE;
	}
	ctx.cull_face_mode = GL_BACK;
	ctx.front_face = GL_CCW;
	ctx.polygon_mode = GL_FILL;
	ctx.polygon_offset_factor = ctx.polygon_offset_units = 0.0f;
	ctx.depth_range_near = 0.0f;
	ctx.depth_range_far = 1.0f;

	// Depth-stencil state for passes without a depth attachment: always pass, never write.
	MTLDepthStencilDescriptor *noDepthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
	noDepthDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
	noDepthDescriptor.depthWriteEnabled = NO;
	ctx.no_depth_state = [ctx.device newDepthStencilStateWithDescriptor:noDepthDescriptor];
	if (!ctx.no_depth_state) {
		GL4METAL_LOG("gl4metalInit: failed to create no-depth-stencil state");
		return GL_FALSE;
	}

	// GL's default value of a generic vertex attribute that has no array enabled is (0, 0, 0, 1).
	const float f[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const int32_t n[4] = { 0, 0, 0, 1 };
	uint8_t bytes[32];
	memcpy(bytes, f, 16);
	memcpy(bytes + 16, n, 16);
	ctx.constant_attrib_buffer = [ctx.device newBufferWithBytes:bytes length:sizeof(bytes) options:MTLResourceStorageModeShared];
	if (!ctx.constant_attrib_buffer) {
		GL4METAL_LOG("gl4metalInit: failed to create constant vertex-attribute buffer");
		return GL_FALSE;
	}

	updateDepthStencilState();
	if (!ctx.depthStencilState) {
		GL4METAL_LOG("gl4metalInit: failed to create initial depth-stencil state");
		return GL_FALSE;
	}
	gl4metalInitialized = true;
	return GL_TRUE;
}

void* GL4MetalCreateContext(GLenum format, MTLContext sharelist) {
	return NULL; // don't need yet
}

MTLCompareFunction getMetalCompareFunc(GLenum func) {
	switch(func) {
		case GL_NEVER:
			return MTLCompareFunctionNever;
		case GL_LESS:
			return MTLCompareFunctionLess;
		case GL_EQUAL:
			return MTLCompareFunctionEqual;
		case GL_LEQUAL:
			return MTLCompareFunctionLessEqual;
		case GL_GREATER:
			return MTLCompareFunctionGreater;
		case GL_NOTEQUAL:
			return MTLCompareFunctionNotEqual;
		case GL_GEQUAL:
			return MTLCompareFunctionGreaterEqual;
		case GL_ALWAYS:
			return MTLCompareFunctionAlways;
		default:
			GL4METAL_LOG("Missing compare function: 0x%X, defaulting to GL_ALWAYS", func);
			return MTLCompareFunctionAlways;
	}
}

static MTLStencilOperation getMetalStencilOperation(GLenum operation) {
	switch (operation) {
		case GL_KEEP: return MTLStencilOperationKeep;
		case GL_ZERO: return MTLStencilOperationZero;
		case GL_REPLACE: return MTLStencilOperationReplace;
		case GL_INCR: return MTLStencilOperationIncrementClamp;
		case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
		case GL_DECR: return MTLStencilOperationDecrementClamp;
		case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
		case GL_INVERT: return MTLStencilOperationInvert;
		default: return MTLStencilOperationKeep;
	}
}

static MTLStencilDescriptor *makeStencilDescriptor(const MTLContext::StencilFaceState &state) {
	MTLStencilDescriptor *descriptor = [[MTLStencilDescriptor alloc] init];
	descriptor.stencilCompareFunction = getMetalCompareFunc(state.func);
	descriptor.stencilFailureOperation = getMetalStencilOperation(state.fail);
	descriptor.depthFailureOperation = getMetalStencilOperation(state.depth_fail);
	descriptor.depthStencilPassOperation = getMetalStencilOperation(state.depth_pass);
	descriptor.readMask = state.value_mask;
	descriptor.writeMask = state.write_mask;
	return descriptor;
}

void updateDepthStencilState() {
	// Context/Metal teardown can still trigger GL state callbacks while the JVM is
	// exiting. Keep both the cache and its mutex alive until process termination;
	// destroying a static unordered_map of strong Objective-C objects can race a
	// late callback and leave a subsequent find() operating on a destroyed table.
	using DepthStencilCache = std::unordered_map<std::string, id<MTLDepthStencilState>>;
	static DepthStencilCache *cache = new DepthStencilCache();
	static std::mutex *cacheMutex = new std::mutex();
	if (!ctx.device) {
		ctx.depthStencilState = nil;
		return;
	}
	std::lock_guard<std::mutex> lock(*cacheMutex);

	const bool depthEnabled = ctx.depth_test_enabled != GL_FALSE;
	const bool stencilEnabled = ctx.caps[GL_STENCIL_TEST] != GL_FALSE;
	const auto &front = ctx.stencil_front;
	const auto &back = ctx.stencil_back;
	std::string key;
	auto append = [&key](uint64_t value) {
		key.append(reinterpret_cast<const char *>(&value), sizeof(value));
	};
	append(depthEnabled); append(ctx.depth_write_enabled); append(ctx.depth_func);
	append(stencilEnabled);
	append(front.func); append((uint32_t)front.ref); append(front.value_mask); append(front.write_mask);
	append(front.fail); append(front.depth_fail); append(front.depth_pass);
	append(back.func); append((uint32_t)back.ref); append(back.value_mask); append(back.write_mask);
	append(back.fail); append(back.depth_fail); append(back.depth_pass);

	auto it = cache->find(key);
	if (it == cache->end() || !it->second) {
		MTLDepthStencilDescriptor *descriptor = [[MTLDepthStencilDescriptor alloc] init];
		descriptor.depthCompareFunction = depthEnabled ? getMetalCompareFunc(ctx.depth_func) : MTLCompareFunctionAlways;
		descriptor.depthWriteEnabled = depthEnabled && ctx.depth_write_enabled;
		if (stencilEnabled) {
			descriptor.frontFaceStencil = makeStencilDescriptor(front);
			descriptor.backFaceStencil = makeStencilDescriptor(back);
		} else {
			// A disabled stencil test must not mutate the stencil attachment.
			MTLStencilDescriptor *disabled = [[MTLStencilDescriptor alloc] init];
			disabled.stencilCompareFunction = MTLCompareFunctionAlways;
			disabled.stencilFailureOperation = MTLStencilOperationKeep;
			disabled.depthFailureOperation = MTLStencilOperationKeep;
			disabled.depthStencilPassOperation = MTLStencilOperationKeep;
			disabled.readMask = 0xFFFFFFFFu;
			disabled.writeMask = 0;
			descriptor.frontFaceStencil = disabled;
			descriptor.backFaceStencil = disabled;
		}
		id<MTLDepthStencilState> created = [ctx.device newDepthStencilStateWithDescriptor:descriptor];
		if (it == cache->end()) it = cache->emplace(key, created).first;
		else it->second = created;
	}
	ctx.depthStencilState = it->second;
}
