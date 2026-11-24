//
// context.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//

#import "context.h"
#import "draw.h"
#import "buffers.h"

#import <mach/mach.h>
#import <mach/task_info.h>

extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *pool);

// the render thread is a plain JVM thread: nothing drains an autorelease pool on it, so every autoreleased Metal object
// command buffers, encoders, descriptors, drawables would live forever together with the buffers and textures those
// command buffers retain. One pool per frame, drained when glSwapBuffers returns, fixes that.
static void *&framePoolForCurrentThread() {
	static thread_local void *pool = nullptr;
	return pool;
}

static void ensureFramePool() {
	void *&pool = framePoolForCurrentThread();
	if (!pool) pool = objc_autoreleasePoolPush();
}

static void cycleFramePool() {
	void *&pool = framePoolForCurrentThread();
	if (pool) objc_autoreleasePoolPop(pool);
	pool = objc_autoreleasePoolPush();
}

static bool renderPassTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_RENDER_PASSES");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool memoryTraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_MEMORY");
		return value && value[0] == '1';
	}();
	return enabled;
}

static bool metal4TraceEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("GL4METAL_TRACE_METAL4");
		return value && value[0] == '1';
	}();
	return enabled;
}

static void logMemorySnapshot() {
	if (!memoryTraceEnabled()) return;
	static thread_local uint64_t swapCount = 0;
	if ((++swapCount % 120) != 0) return;

	static thread_local std::vector<const void *> seenResources;
	seenResources.clear();
	uint64_t trackedResourceBytes = 0;
	size_t trackedTextureCount = 0;
	size_t trackedBufferCount = 0;
	size_t trackedRenderbufferCount = 0;
	size_t feedbackSnapshotCount = 0;
	size_t swizzleViewCount = 0;
	size_t feedbackSwizzleViewCount = 0;
	auto addResource = [&](id<MTLResource> resource) {
		if (!resource) return;
		const void *resourceKey = (__bridge const void *)resource;
		if (std::find(seenResources.begin(), seenResources.end(), resourceKey) != seenResources.end()) return;
		seenResources.push_back(resourceKey);
		const uint64_t resourceBytes = (uint64_t)resource.allocatedSize;
		if (UINT64_MAX - trackedResourceBytes < resourceBytes) trackedResourceBytes = UINT64_MAX;
		else trackedResourceBytes += resourceBytes;
	};

	for (const auto &entry : ctx.texture_objects) {
		const TextureObject &textureObject = entry.second;
		// Buffer textures alias MTLBuffer storage and are accounted with the buffer,
		// not again as a texture view.
		if (textureObject.texture && textureObject.buffer_object == 0) {
			addResource(textureObject.texture);
			++trackedTextureCount;
		}
		if (textureObject.feedback_snapshot) {
			addResource(textureObject.feedback_snapshot);
			++feedbackSnapshotCount;
		}
		if (textureObject.sampled_view) ++swizzleViewCount;
		if (textureObject.feedback_sampled_view) ++feedbackSwizzleViewCount;
	}
	for (const auto &entry : ctx.renderbuffer_objects) {
		if (entry.second.texture) {
			addResource(entry.second.texture);
			++trackedRenderbufferCount;
		}
	}
	for (const auto &entry : ctx.buffer_objects) {
		if (entry.second) {
			addResource(entry.second);
			++trackedBufferCount;
		}
	}

	task_vm_info_data_t vmInfo = {};
	mach_msg_type_number_t vmInfoCount = TASK_VM_INFO_COUNT;
	const kern_return_t vmResult = task_info(mach_task_self(), TASK_VM_INFO,
		(task_info_t)&vmInfo, &vmInfoCount);
	const uint64_t physicalFootprint = vmResult == KERN_SUCCESS ? (uint64_t)vmInfo.phys_footprint : 0;
	const uint64_t metalAllocated = ctx.device ? (uint64_t)ctx.device.currentAllocatedSize : 0;
	const uint64_t recommendedMetalWorkingSet = ctx.device ? (uint64_t)ctx.device.recommendedMaxWorkingSetSize : 0;
	GL4METAL_LOG("[Memory] swap=%llu MetalDevice=%lluMiB recommendedMetal=%lluMiB GLTracked=%lluMiB physFootprint=%s%lluMiB textures=%zu buffers=%zu renderbuffers=%zu feedbackSnapshots=%zu swizzleViews=%zu feedbackSwizzleViews=%zu shaders=%zu programs=%zu pipelines=%zu samplers=%zu",
		(unsigned long long)swapCount,
		(unsigned long long)(metalAllocated / (1024u * 1024u)),
		(unsigned long long)(recommendedMetalWorkingSet / (1024u * 1024u)),
		(unsigned long long)(trackedResourceBytes / (1024u * 1024u)),
		vmResult == KERN_SUCCESS ? "" : "unavailable/",
		(unsigned long long)(physicalFootprint / (1024u * 1024u)),
		trackedTextureCount, trackedBufferCount, trackedRenderbufferCount, feedbackSnapshotCount,
		swizzleViewCount, feedbackSwizzleViewCount,
		ctx.shader_objects.size(), ctx.program_objects.size(), ctx.pipeline_cache.size(), ctx.sampler_cache.size());

	if (metal4TraceEnabled()) {
		uint64_t transientBufferBytes = 0;
		for (NSUInteger slot = 0; slot < 3; ++slot) {
			if (ctx.metal4TransientBuffers[slot]) transientBufferBytes += ctx.metal4TransientBuffers[slot].length;
		}
		GL4METAL_LOG("[Metal4] enabled=%d inFlight=%zu eventSignaled=%llu nextEvent=%llu lastClassic=%llu lastCompute=%llu transientBuffers=%lluKiB",
			ctx.metal4Enabled ? 1 : 0, ctx.metal4InFlightSubmissions.size(),
			(unsigned long long)(ctx.metal4InteropEvent ? ctx.metal4InteropEvent.signaledValue : 0),
			(unsigned long long)ctx.metal4NextEventValue,
			(unsigned long long)ctx.metal4LastClassicEventValue,
			(unsigned long long)ctx.metal4LastComputeEventValue,
			(unsigned long long)(transientBufferBytes / 1024u));
	}
}

struct FramePoolCycle {
	~FramePoolCycle() { cycleFramePool(); }
};

void glMakeCurrent(void *metalLayerPtr) {
	ensureFramePool();
	if (!ctx.device || !ctx.commandQueue || !metalLayerPtr) {
		GL4METAL_LOG("glMakeCurrent: no device or no layer (%p), ignored", metalLayerPtr);
		return;
	}
	CAMetalLayer *layer = (__bridge CAMetalLayer *)metalLayerPtr;
	if (ctx.metalLayer != layer) endRenderEncoder();
	layer.device = ctx.device;
	layer.framebufferOnly = YES;
	ctx.metalLayer = layer;
	GL4METAL_LOG("glMakeCurrent: layer %p", metalLayerPtr);
}

void glSwapBuffers(void) {
	FramePoolCycle cycle; // drains the frame's autorelease pool on every return path
	reportCommandBufferErrorIfNeeded(ctx.last_committed_buffer);
	pruneCompletedBufferUses();

	// Submissions may complete after their commit call. Reap on every frame even
	// when no further compute dispatch occurs, otherwise retained residency sets,
	// resources and PSOs can stay alive until the next Metal 4 submission.
	gl4metalReapMetal4Submissions();

	logMemorySnapshot();
	// The drawable's depth/stencil is not consumed after present. endRenderEncoder()
	// finalizes deferred Metal store actions as DontCare for this final default-FB pass.
	ctx.discard_default_depth_stencil_on_end = true;
	endRenderEncoder(); // a render pass must be closed before the command buffer is committed
	if (renderPassTraceEnabled()) {
		GL4METAL_LOG("frame render passes: created=%llu reused=%llu fboAliasCandidates=%llu fboAliasReuses=%llu fboAliasMisses=%llu mergedBlitDraws=%llu",
			(unsigned long long)ctx.frame_render_pass_created_count,
			(unsigned long long)ctx.frame_render_pass_reused_count,
			(unsigned long long)ctx.frame_render_pass_alias_candidates,
			(unsigned long long)ctx.frame_render_pass_alias_reuses,
			(unsigned long long)ctx.frame_render_pass_alias_misses,
			(unsigned long long)ctx.frame_merged_blit_draw_count);
	}
	ctx.frame_render_pass_created_count = 0;
	ctx.frame_render_pass_reused_count = 0;
	ctx.frame_merged_blit_draw_count = 0;
	ctx.frame_render_pass_alias_candidates = 0;
	ctx.frame_render_pass_alias_reuses = 0;
	ctx.frame_render_pass_alias_misses = 0;
	ctx.default_color_has_contents = false;
	ctx.default_depth_has_contents = false;
	ctx.default_stencil_has_contents = false;

	// The default framebuffer is presented by the classic CAMetalLayer path. Commit
	// compute first; the next classic command buffer waits on its shared-event value.
	gl4metalCommitMetal4Compute();

	if (ctx.currentDrawable && !gl4metalEnsureClassicCommandBuffer()) {
		GL4METAL_LOG("glSwapBuffers: failed to create ordered classic command buffer for present");
		ctx.currentDrawable = nil;
		return;
	}
	if (ctx.currentCommandBuffer) {
		if (ctx.currentDrawable) [ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
		gl4metalCommitClassicCommandBuffer();
	}
	ctx.currentDrawable = nil; // next frame acquires a new drawable
}

void glSwapInterval(GLint interval) {
	if (!ctx.metalLayer) return;
	// nothing to be implement
}

void glFlush(void) {
	flushCommandBuffer(false);
}

void glFinish(void) {
	flushCommandBuffer(true);
}
