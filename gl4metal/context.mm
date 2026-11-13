//
// context.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//

#import "context.h"
#import "draw.h"
#import "buffers.h"

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
	layer.device = ctx.device;
	layer.framebufferOnly = YES;
	ctx.metalLayer = layer;
	GL4METAL_LOG("glMakeCurrent: layer %p", metalLayerPtr);
}

void glSwapBuffers(void) {
	FramePoolCycle cycle; // drains the frame's autorelease pool on every return path
	reportCommandBufferErrorIfNeeded(ctx.last_committed_buffer);
	pruneCompletedBufferUses();
	// The drawable's depth/stencil is not consumed after present. endRenderEncoder()
	// finalizes deferred Metal store actions as DontCare for this final default-FB pass.
	ctx.discard_default_depth_stencil_on_end = true;
	endRenderEncoder(); // a render pass must be closed before the command buffer is committed
	ctx.default_color_has_contents = false;
	ctx.default_depth_has_contents = false;
	ctx.default_stencil_has_contents = false;
	if (!ctx.currentCommandBuffer && ctx.currentDrawable && ctx.commandQueue) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer]; // glFlush/fences may have committed this frame's work already
	}
	if (!ctx.currentCommandBuffer) {
		return;
	}
	if (ctx.currentDrawable) {
		[ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
	}
	[ctx.currentCommandBuffer commit];
	resetDrawUniformUploadArena(ctx.currentCommandBuffer);
	resetTextureUploadArena(ctx.currentCommandBuffer);
	ctx.last_committed_buffer = ctx.currentCommandBuffer;
	ctx.currentCommandBuffer = nil;
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
