#import "sync.h"
#import "draw.h"

#include <unistd.h>

// A fence completes when the command buffer that holds the commands recorded before it has completed.
struct SyncObject {
	id<MTLCommandBuffer> buffer = nil; // nil: nothing was pending, already signaled
};

static bool isSignaled(const SyncObject *s) {
	if (!s->buffer) {
		return true;
	}
	MTLCommandBufferStatus status = s->buffer.status;
	if (status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError) {
		return true;
	}
	// A buffer that was dropped without being committed (app in the background) can never complete.
	return status < MTLCommandBufferStatusCommitted && s->buffer != ctx.currentCommandBuffer;
}

// Metal completion handlers can run on arbitrary driver queues, including while the
// app is tearing down its GL context. Poll the last submitted command buffer from the
// GL thread instead; this keeps all GL4Metal state and diagnostics single-thread-affine.
static __strong id<MTLCommandBuffer> lastReportedCommandBufferError = nil;

void reportCommandBufferErrorIfNeeded(id<MTLCommandBuffer> commandBuffer) {
	if (!commandBuffer || commandBuffer.status != MTLCommandBufferStatusError ||
		lastReportedCommandBufferError == commandBuffer) return;
	// Keep only the most recently reported command buffer alive. A raw pointer set
	// could grow forever and become incorrect if Objective-C reuses an address.
	lastReportedCommandBufferError = commandBuffer;
	NSError *error = commandBuffer.error;
	GL4METAL_LOG("Metal command buffer failed (status=%ld, error=%ld): %s; label is only the last encoded draw hint, not a guaranteed fault location: '%s'",
		(long)commandBuffer.status, (long)error.code,
		error.localizedDescription.UTF8String ?: "unknown Metal GPU error",
		commandBuffer.label.UTF8String ?: "(unlabelled)");
	if (error.userInfo.count > 0) {
		GL4METAL_LOG("Metal command buffer error details: %s", error.userInfo.description.UTF8String ?: "(none)");
	}
}

// Commits the open command buffer so that fences waiting for it can complete.
void flushCommandBuffer(bool waitUntilCompleted) {
	reportCommandBufferErrorIfNeeded(ctx.last_committed_buffer);
	endRenderEncoder();
	if (ctx.currentComputeEncoder) {
		[ctx.currentComputeEncoder endEncoding];
		ctx.currentComputeEncoder = nil;
	}
	if (ctx.currentCommandBuffer) {
		[ctx.currentCommandBuffer commit];
		resetDrawUniformUploadArena(ctx.currentCommandBuffer);
		resetTextureUploadArena(ctx.currentCommandBuffer);
		ctx.last_committed_buffer = ctx.currentCommandBuffer;
		ctx.currentCommandBuffer = nil;
	}
	if (waitUntilCompleted && ctx.last_committed_buffer) {
		[ctx.last_committed_buffer waitUntilCompleted];
		reportCommandBufferErrorIfNeeded(ctx.last_committed_buffer);
	}
}

static SyncObject *findSync(GLsync sync) {
	return ctx.sync_objects.count(sync) ? (SyncObject *)sync : nullptr;
}

GLsync glFenceSync(GLenum condition, GLbitfield flags) {
	if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE || flags != 0) {
		GL4METAL_LOG("glFenceSync: invalid condition 0x%X or flags 0x%X (GL_INVALID_ENUM)", condition, flags);
		return NULL;
	}
	SyncObject *s = new SyncObject();
	s->buffer = ctx.currentCommandBuffer ? ctx.currentCommandBuffer : ctx.last_committed_buffer;
	ctx.sync_objects.insert(s);
	return (GLsync)s;
}

GLboolean glIsSync(GLsync sync) {
	return findSync(sync) != nullptr;
}

void glDeleteSync(GLsync sync) {
	SyncObject *s = findSync(sync);
	if (!s) {
		return;
	}
	ctx.sync_objects.erase(sync);
	delete s;
}

GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
	SyncObject *s = findSync(sync);
	if (!s) {
		GL4METAL_LOG("glClientWaitSync: invalid sync object (GL_INVALID_VALUE)");
		return GL_WAIT_FAILED;
	}
	if (isSignaled(s)) {
		return GL_ALREADY_SIGNALED;
	}
	if (flags & GL_SYNC_FLUSH_COMMANDS_BIT) {
		flushCommandBuffer(false); // otherwise the fence's buffer might never be submitted
	}
	if (timeout == 0) {
		return GL_TIMEOUT_EXPIRED;
	}
	flushCommandBuffer(false);
	if (timeout == GL_TIMEOUT_IGNORED) {
		[s->buffer waitUntilCompleted];
		return GL_CONDITION_SATISFIED;
	}
	const uint64_t pollNs = 200000; // 0.2 ms
	for (uint64_t waited = 0; waited < timeout; waited += pollNs) {
		if (isSignaled(s)) {
			return GL_CONDITION_SATISFIED;
		}
		usleep((useconds_t)(pollNs / 1000));
	}
	return isSignaled(s) ? GL_CONDITION_SATISFIED : GL_TIMEOUT_EXPIRED;
}

void glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
	// Server-side wait: commands of one queue already execute in order, nothing to do.
}

void glGetSynciv(GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values) {
	SyncObject *s = findSync(sync);
	GLint value = 0;
	if (!s) {
		GL4METAL_LOG("glGetSynciv: invalid sync object");
	} else {
		switch (pname) {
			case GL_OBJECT_TYPE:    value = GL_SYNC_FENCE; break;
			case GL_SYNC_CONDITION: value = GL_SYNC_GPU_COMMANDS_COMPLETE; break;
			case GL_SYNC_STATUS:    value = isSignaled(s) ? GL_SIGNALED : GL_UNSIGNALED; break;
			case GL_SYNC_FLAGS:     value = 0; break;
			default:
				GL4METAL_LOG("glGetSynciv: unhandled pname 0x%X", pname);
				break;
		}
	}
	if (count > 0) {
		values[0] = value;
	}
	if (length) {
		*length = count > 0 ? 1 : 0;
	}
}
