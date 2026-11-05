//
// context.mm
// gl4metal
//
// Created by congcq on 16.10.26.
//

#import "context.h"

extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *pool);

// the render thread is a plain JVM thread: nothing drains an autorelease pool on it, so every autoreleased Metal object
// command buffers, encoders, descriptors, drawables would live forever together with the buffers and textures those
// command buffers retain. One pool per frame, drained when glSwapBuffers returns, fixes that.
static void cycleFramePool() {
	static void *pool = nullptr;
	static pthread_t poolThread;
	pthread_t self = pthread_self();
	if (pool && pthread_equal(poolThread, self)) {
		objc_autoreleasePoolPop(pool);
	}
	pool = objc_autoreleasePoolPush();
	poolThread = self;
}

struct FramePoolCycle {
	~FramePoolCycle() { cycleFramePool(); }
};

void glMakeCurrent(void *metalLayerPtr) {
	if (!ctx.device || !metalLayerPtr) {
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
	endRenderEncoder(); // a render pass must be closed before the command buffer is committed
	if (!ctx.currentCommandBuffer && ctx.currentDrawable) {
		ctx.currentCommandBuffer = [ctx.commandQueue commandBuffer]; // glFlush/fences may have committed this frame's work already
	}
	if (!ctx.currentCommandBuffer) {
		return;
	}
	if (ctx.currentDrawable) {
		[ctx.currentCommandBuffer presentDrawable:ctx.currentDrawable];
	}
	[ctx.currentCommandBuffer commit];
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
