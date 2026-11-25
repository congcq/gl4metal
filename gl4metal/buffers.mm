//
// buffers.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "buffers.h"

#import "sync.h"
#import "texture.h"


// MARK: - Buffer objects

static bool isBufferTarget(GLenum target) {
	return target == GL_ARRAY_BUFFER || target == GL_ELEMENT_ARRAY_BUFFER ||
		target == GL_UNIFORM_BUFFER || target == GL_SHADER_STORAGE_BUFFER || target == GL_COPY_READ_BUFFER ||
		target == GL_COPY_WRITE_BUFFER || target == GL_PIXEL_PACK_BUFFER ||
		target == GL_PIXEL_UNPACK_BUFFER || target == GL_TEXTURE_BUFFER ||
		target == GL_DISPATCH_INDIRECT_BUFFER;
}

// GL4Metal's MTLContext is single-thread-affine just like the current OpenGL context.
// Keep last-use tracking on the GL submission thread; never mutate MTLContext from a
// Metal completion callback. This avoids callback/context teardown races and removes a
// mutex from the per-draw hot path.
static uint32_t bufferUsePruneCounter = 0;

void markBufferMayBeGPUWritten(GLuint buf_id) {
	if (buf_id != 0) ctx.buffer_may_be_gpu_written.insert(buf_id);
}

static bool commandBufferPending(id<MTLCommandBuffer> commandBuffer) {
	if (!commandBuffer) return false;
	const MTLCommandBufferStatus status = commandBuffer.status;
	return status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError;
}

// Metal completion handlers are intentionally not used for GL state. Metal may invoke
// them on an arbitrary queue, including during GL context teardown. Prune completed
// command buffers lazily on the GL submission thread instead.
void pruneCompletedBufferUses(void) {
	for (auto it = ctx.buffer_last_use.begin(); it != ctx.buffer_last_use.end();) {
		if (!commandBufferPending(it->second)) it = ctx.buffer_last_use.erase(it);
		else ++it;
	}

	if (ctx.metal4InteropEvent) {
		const uint64_t completedValue = ctx.metal4InteropEvent.signaledValue;
		for (auto it = ctx.buffer_last_use_event_value.begin(); it != ctx.buffer_last_use_event_value.end();) {
			if (it->second <= completedValue) it = ctx.buffer_last_use_event_value.erase(it);
			else ++it;
		}
	}
}

void markBuffersUsed(const GLuint *buf_ids, size_t objectCount) {
	if (!buf_ids || objectCount == 0) return;

	if (ctx.metal4Enabled && ctx.currentMetal4CommandBuffer) {
		for (size_t index = 0; index < objectCount; ++index) {
			if (buf_ids[index] != 0) gl4metalMarkMetal4BufferUsed(buf_ids[index]);
		}
		if (++bufferUsePruneCounter >= 128 || ctx.buffer_last_use_event_value.size() > 2048) {
			bufferUsePruneCounter = 0;
			pruneCompletedBufferUses();
		}
		return;
	}

	id<MTLCommandBuffer> activeBuffer = ctx.currentCommandBuffer;
	if (!activeBuffer) return;
	for (size_t index = 0; index < objectCount; ++index) {
		const GLuint buf_id = buf_ids[index];
		if (buf_id == 0) continue;
		auto useIt = ctx.buffer_last_use.find(buf_id);
		if (useIt == ctx.buffer_last_use.end()) ctx.buffer_last_use.emplace(buf_id, activeBuffer);
		else if (useIt->second != activeBuffer) useIt->second = activeBuffer;
	}
	// Prune infrequently instead of scanning the whole map on every draw. When the
	// table becomes large, prune immediately to keep memory bounded.
	if (++bufferUsePruneCounter >= 128 || ctx.buffer_last_use.size() > 2048) {
		bufferUsePruneCounter = 0;
		pruneCompletedBufferUses();
	}
}

void markBufferUsed(GLuint buf_id) {
	markBuffersUsed(&buf_id, 1);
}

static bool bufferHasPendingGPUUse(GLuint buf_id) {
	auto eventIt = ctx.buffer_last_use_event_value.find(buf_id);
	if (eventIt != ctx.buffer_last_use_event_value.end()) {
		if (ctx.metal4InteropEvent && eventIt->second > ctx.metal4InteropEvent.signaledValue) return true;
		ctx.buffer_last_use_event_value.erase(eventIt);
	}

	auto it = ctx.buffer_last_use.find(buf_id);
	if (it == ctx.buffer_last_use.end()) return false;
	if (commandBufferPending(it->second)) return true;
	ctx.buffer_last_use.erase(it);
	return false;
}

static void waitForBufferGPUUse(GLuint buf_id);

static bool makeBufferWritableWithoutStall(GLuint buf_id, __strong id<MTLBuffer> &buffer, bool preserveContents = true) {
	if (!bufferHasPendingGPUUse(buf_id)) return true;
	// On Apple GPUs the CPU and GPU share this MTLBuffer's system-memory allocation.
	// If the in-flight workload only reads it (vertex/index/UBO input), clone the
	// current CPU-visible bytes into a new shared allocation and redirect future GL
	// commands to that allocation. Earlier command buffers keep the old MTLBuffer, so
	// the CPU can update the new store without waiting for those GPU reads to finish.
	// Shader-storage buffers and GPU blit destinations are conservatively classified
	// as potentially GPU-written and still take the synchronization path.
	const bool mayBeGPUWritten = ctx.buffer_may_be_gpu_written.find(buf_id) !=
		ctx.buffer_may_be_gpu_written.end();
	if (preserveContents && mayBeGPUWritten) {
		waitForBufferGPUUse(buf_id);
		if (!bufferHasPendingGPUUse(buf_id)) return true;
		// Never clone a store while an untracked GPU write could still be in flight.
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("buffer update: refusing to rename potentially GPU-written buffer %u while its last use is pending", buf_id);
		return false;
	}
	if (!ctx.device || !buffer || buffer.length > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return false;
	}
	id<MTLBuffer> replacement = [ctx.device newBufferWithLength:std::max<NSUInteger>(buffer.length, 1)
		options:MTLResourceStorageModeShared];
	if (!replacement) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("buffer update: failed to rename in-flight buffer %u (%llu bytes)",
			buf_id, (unsigned long long)buffer.length);
		return false;
	}
	if (preserveContents && buffer.length != 0) {
		memcpy(replacement.contents, buffer.contents, (size_t)buffer.length);
	}
	ctx.buffer_objects[buf_id] = replacement;
	ctx.buffer_last_use.erase(buf_id);
	ctx.buffer_last_use_event_value.erase(buf_id);
	buffer = replacement;
	refreshTextureBufferViews(buf_id);
	// Rebuilding linear texture-buffer views may grow the replacement again to add
	// row padding. Keep the caller's reference on the final shared backing store;
	// otherwise the following CPU write would update an obsolete intermediate buffer.
	auto refreshed = ctx.buffer_objects.find(buf_id);
	if (refreshed == ctx.buffer_objects.end() || !refreshed->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return false;
	}
	buffer = refreshed->second;
	return true;
}

static void waitForBufferGPUUse(GLuint buf_id) {
	auto eventIt = ctx.buffer_last_use_event_value.find(buf_id);
	if (eventIt != ctx.buffer_last_use_event_value.end()) {
		const uint64_t targetValue = eventIt->second;
		if (targetValue > (ctx.metal4InteropEvent ? ctx.metal4InteropEvent.signaledValue : 0)) {
			if (ctx.currentMetal4CommandBuffer && ctx.metal4CurrentEventValue == targetValue) {
				gl4metalCommitMetal4Compute();
			}
			gl4metalWaitForMetal4Event(targetValue, UINT64_MAX);
		}
		auto completedIt = ctx.buffer_last_use_event_value.find(buf_id);
		if (completedIt != ctx.buffer_last_use_event_value.end() && completedIt->second == targetValue)
			ctx.buffer_last_use_event_value.erase(completedIt);
	}

	id<MTLCommandBuffer> lastUse = nil;
	auto useIt = ctx.buffer_last_use.find(buf_id);
	if (useIt == ctx.buffer_last_use.end()) return;
	lastUse = useIt->second;
	if (!commandBufferPending(lastUse)) {
		ctx.buffer_last_use.erase(useIt);
		return;
	}
	// If this is the still-open command buffer, submit it before waiting. Do not flush
	// unrelated work when the buffer was last used by an already committed command buffer.
	if (lastUse == ctx.currentCommandBuffer) {
		flushCommandBuffer(false);
	}
	[lastUse waitUntilCompleted];
	auto it = ctx.buffer_last_use.find(buf_id);
	if (it != ctx.buffer_last_use.end() && it->second == lastUse) ctx.buffer_last_use.erase(it);
}


void glGenBuffers(GLsizei n, GLuint *buffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint buf_id = ++ctx.next_buffer_id; // names start at 1, 0 is "no buffer"
		ctx.buffer_objects[buf_id] = nil;     // name exists, storage comes with glBufferData
		gl4metalSetBufferLogicalSize(buf_id, 0);
		buffers[i] = buf_id;
	}
}

void glDeleteBuffers(GLsizei n, const GLuint *buffers) {
	if (n < 0 || (n > 0 && !buffers)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei i = 0; i < n; i++) {
		const GLuint buf_id = buffers[i];
		if (buf_id == 0) continue;

		auto storageIt = ctx.buffer_objects.find(buf_id);
		__strong id<MTLBuffer> deletedStorage =
			(storageIt != ctx.buffer_objects.end()) ? storageIt->second : nil;
		const NSUInteger deletedLogicalSize = glBufferLogicalSize(buf_id);
		const GLuint currentVaoId = ctx.current_vao;
		bool detachedCurrentElement = false;

		// GL_ARRAY_BUFFER and the other generic targets are context bindings. The
		// element-array binding is VAO state, so update it from the current VAO below.
		for (auto &binding : ctx.bound_buffers) {
			if (binding.first != GL_ELEMENT_ARRAY_BUFFER && binding.second == buf_id) {
				binding.second = 0;
			}
		}

		for (auto &vaoEntry : ctx.vao_objects) {
			const bool isCurrentVao = vaoEntry.first == currentVaoId;
			VertexArray &vao = vaoEntry.second;
			if (vao.element_array_buffer_id == buf_id && !vao.retains_deleted_element_buffer) {
				if (isCurrentVao) {
					// Deletion detaches the buffer from container objects bound to this
					// context. The current VAO loses the EBO association; other VAOs keep
					// their reference and therefore must retain the old Metal storage.
					vao.element_array_buffer_id = 0;
					vao.retains_deleted_element_buffer = false;
					vao.retained_deleted_element_buffer = nil;
					vao.retained_deleted_element_buffer_size = 0;
					detachedCurrentElement = true;
				} else if (!vao.retains_deleted_element_buffer) {
					vao.retains_deleted_element_buffer = true;
					vao.retained_deleted_element_buffer = deletedStorage;
					vao.retained_deleted_element_buffer_size = deletedLogicalSize;
				}
			}
			for (VertexAttrib &attrib : vao.attribs) {
				if (attrib.bound_vbo != buf_id || attrib.retains_deleted_buffer) continue;
				if (isCurrentVao) {
					// Keep the enable bit and byte offset: enable state is independent
					// of the buffer association. An enabled array with no buffer will be
					// rejected by executeDraw until the client supplies a new pointer.
					attrib.bound_vbo = 0;
					attrib.retains_deleted_buffer = false;
					attrib.retained_deleted_buffer = nil;
					attrib.retained_deleted_buffer_size = 0;
				} else if (!attrib.retains_deleted_buffer) {
					attrib.retains_deleted_buffer = true;
					attrib.retained_deleted_buffer = deletedStorage;
					attrib.retained_deleted_buffer_size = deletedLogicalSize;
				}
			}
		}
		if (detachedCurrentElement) {
			ctx.bound_buffers[GL_ELEMENT_ARRAY_BUFFER] = 0;
		} else {
			auto currentVaoIt = ctx.vao_objects.find(currentVaoId);
			ctx.bound_buffers[GL_ELEMENT_ARRAY_BUFFER] =
				(currentVaoIt != ctx.vao_objects.end()) ? currentVaoIt->second.element_array_buffer_id : 0;
		}

		for (UniformBufferBinding &binding : ctx.uniform_buffer_bindings) {
			if (binding.buffer == buf_id) binding = UniformBufferBinding();
		}
		for (ShaderStorageBufferBinding &binding : ctx.shader_storage_buffer_bindings) {
			if (binding.buffer == buf_id) binding = ShaderStorageBufferBinding();
		}
		// Detach texture-buffer views as in the existing backend implementation.
		for (auto &textureEntry : ctx.texture_objects) {
			TextureObject &textureObject = textureEntry.second;
			if (textureObject.buffer_object == buf_id) {
				textureObject.buffer_object = 0;
				textureObject.buffer_sampler_type = 0;
				gl4metalSetTextureStorage(textureObject, nil);
				textureObject.width = 0;
				textureObject.height = 0;
			}
		}

		// Drop the GL name immediately. Non-current VAOs hold strong references to
		// the old MTLBuffer in their own attachment records, so reusing this numeric
		// GL name creates a new buffer without changing those VAOs' backing storage.
		invalidateConvertedVertexBuffers(buf_id);
		ctx.buffer_last_use.erase(buf_id);
		ctx.buffer_last_use_event_value.erase(buf_id);
		ctx.buffer_may_be_gpu_written.erase(buf_id);
		ctx.buffer_objects.erase(buf_id);
		gl4metalEraseBufferLogicalSize(buf_id);

	}
}

GLboolean glIsBuffer(GLuint buffer) {
	return buffer != 0 && ctx.buffer_objects.find(buffer) != ctx.buffer_objects.end();
}

void glBindBuffer(GLenum target, GLuint buffer) {
	if (!isBufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBindBuffer: invalid target 0x%X", target);
		return;
	}
	if (buffer != 0 && ctx.buffer_objects.find(buffer) == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil; // GL allows binding a name that was never generated
		gl4metalSetBufferLogicalSize(buffer, 0);
	}
	ctx.bound_buffers[target] = buffer;
	if (target == GL_ELEMENT_ARRAY_BUFFER) {
		VertexArray &vao = ctx.vao_objects[ctx.current_vao];
		vao.element_array_buffer_id = buffer; // element buffer binding is VAO state
		vao.retains_deleted_element_buffer = false;
		vao.retained_deleted_element_buffer = nil;
		vao.retained_deleted_element_buffer_size = 0;
	}
}

void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
	const bool uniformTarget = target == GL_UNIFORM_BUFFER;
	const bool storageTarget = target == GL_SHADER_STORAGE_BUFFER;
	if (!uniformTarget && !storageTarget) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBindBufferBase: unsupported target 0x%X", target);
		return;
	}
	const GLuint bindingCount = uniformTarget ? GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS : GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS;
	if (index >= bindingCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferBase: binding point %u is out of range", index);
		return;
	}
	if (buffer != 0 && ctx.buffer_objects.find(buffer) == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil; // names are materialized on first bind, storage is allocated separately
		gl4metalSetBufferLogicalSize(buffer, 0);
	}
	if (uniformTarget) {
		UniformBufferBinding newBinding;
		newBinding.buffer = buffer;
		newBinding.offset = 0;
		newBinding.size = 0; // size 0 means the whole current store
		ctx.uniform_buffer_bindings[index] = newBinding;
	} else {
		ShaderStorageBufferBinding newBinding;
		newBinding.buffer = buffer;
		newBinding.offset = 0;
		newBinding.size = 0;
		ctx.shader_storage_buffer_bindings[index] = newBinding;
	}
	ctx.bound_buffers[target] = buffer;
}

void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
	const bool uniformTarget = target == GL_UNIFORM_BUFFER;
	const bool storageTarget = target == GL_SHADER_STORAGE_BUFFER;
	if (!uniformTarget && !storageTarget) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBindBufferRange: unsupported target 0x%X", target);
		return;
	}
	const GLuint bindingCount = uniformTarget ? GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS : GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS;
	if (index >= bindingCount) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: binding point %u is out of range", index);
		return;
	}
	if (buffer == 0) {
		// Match the indexed-binding unbind semantics without leaving a stale range behind.
		if (uniformTarget) ctx.uniform_buffer_bindings[index] = UniformBufferBinding();
		else ctx.shader_storage_buffer_bindings[index] = ShaderStorageBufferBinding();
		ctx.bound_buffers[target] = 0;
		return;
	}
	if (offset < 0 || size <= 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: offset must be non-negative and size must be positive (offset=%lld size=%lld)", (long long)offset, (long long)size);
		return;
	}
	// UBO offsets follow the reported 256-byte GL alignment. Metal's setBuffer
	// SSBO range starts must preserve std430 base alignment (up to 16 bytes).
	// Metal accepts 4-byte buffer offsets, but 16-byte alignment keeps vec4/matrix
	// block members naturally aligned after the GL range offset is applied.
	const GLintptr alignment = uniformTarget ? 256 : 16;
	if ((offset % alignment) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: offset %lld is not aligned to %lld bytes", (long long)offset, (long long)alignment);
		return;
	}
	auto object = ctx.buffer_objects.find(buffer);
	if (object == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil;
		gl4metalSetBufferLogicalSize(buffer, 0);
		object = ctx.buffer_objects.find(buffer);
	}
	if (object == ctx.buffer_objects.end() || !object->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindBufferRange: buffer %u has no data store", buffer);
		return;
	}
	const NSUInteger length = glBufferLogicalSize(buffer);
	const NSUInteger off = (NSUInteger)offset;
	const NSUInteger bytes = (NSUInteger)size;
	if (off > length || bytes > length - off) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: range %llu+%llu exceeds buffer %u length %llu",
			(unsigned long long)off, (unsigned long long)bytes, buffer, (unsigned long long)length);
		return;
	}
	if (uniformTarget) {
		UniformBufferBinding newBinding;
		newBinding.buffer = buffer;
		newBinding.offset = offset;
		newBinding.size = size;
		ctx.uniform_buffer_bindings[index] = newBinding;
	} else {
		ShaderStorageBufferBinding newBinding;
		newBinding.buffer = buffer;
		newBinding.offset = offset;
		newBinding.size = size;
		ctx.shader_storage_buffer_bindings[index] = newBinding;
	}
	ctx.bound_buffers[target] = buffer; // also the generic binding point
}

// MARK: - Buffer data

void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
	if (!isBufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (usage != GL_STREAM_DRAW && usage != GL_STREAM_READ && usage != GL_STREAM_COPY &&
		usage != GL_STATIC_DRAW && usage != GL_STATIC_READ && usage != GL_STATIC_COPY &&
		usage != GL_DYNAMIC_DRAW && usage != GL_DYNAMIC_READ && usage != GL_DYNAMIC_COPY) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	GLuint buf_id = boundBuffer(target);
	if (buf_id == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBufferData: no buffer bound to target 0x%X", target);
		return;
	}
	if (size < 0 || (NSUInteger)size > ctx.device.maxBufferLength) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBufferData: invalid size %ld", (long)size);
		return;
	}

	// Reuse an existing shared buffer when it has enough capacity and the GPU has
	// finished using it. This avoids allocating a fresh MTLBuffer for every dynamic
	// Minecraft vertex/index upload. If an earlier draw still references the store,
	// keep the no-stall rename path below and let the command buffer retain the old one.
	NSUInteger length = std::max<NSUInteger>((NSUInteger)size, 1); // Metal rejects zero-length buffers
	auto existingIt = ctx.buffer_objects.find(buf_id);
	if (existingIt != ctx.buffer_objects.end() && existingIt->second &&
		existingIt->second.length >= length && !bufferHasPendingGPUUse(buf_id)) {
		id<MTLBuffer> reusableBuffer = existingIt->second;
		if (data && size > 0) {
			memcpy(reusableBuffer.contents, data, (size_t)size);
		}
		invalidateConvertedVertexBuffers(buf_id);
		ctx.buffer_last_use.erase(buf_id);
		ctx.buffer_last_use_event_value.erase(buf_id);
		ctx.buffer_may_be_gpu_written.erase(buf_id);
		gl4metalSetBufferLogicalSize(buf_id, (NSUInteger)size);
		refreshTextureBufferViews(buf_id);
		return;
	}

	// iOS uses shared storage, so CPU writes are directly visible to Metal. When the
	// old allocation is still in flight or too small, allocate a replacement rather
	// than stalling the render thread. Existing command encoders retain the old store.
	id<MTLBuffer> buffer = nil;
	if (data && size > 0) {
		buffer = [ctx.device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
	} else {
		buffer = [ctx.device newBufferWithLength:length options:MTLResourceStorageModeShared];
	}
	if (!buffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glBufferData: Metal allocation failed for %llu bytes; previous buffer storage preserved", (unsigned long long)length);
		return;
	}
	invalidateConvertedVertexBuffers(buf_id);
	ctx.buffer_last_use.erase(buf_id);
	ctx.buffer_last_use_event_value.erase(buf_id);
	ctx.buffer_may_be_gpu_written.erase(buf_id);
	ctx.buffer_objects[buf_id] = buffer;
	gl4metalSetBufferLogicalSize(buf_id, (NSUInteger)size);
	refreshTextureBufferViews(buf_id);
}

void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
	if (!isBufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBufferSubData: no buffer storage bound to target 0x%X", target);
		return;
	}
	id<MTLBuffer> buffer = it->second;
	const NSUInteger logicalLength = glBufferLogicalSize(buf_id);
	if (offset < 0 || size < 0 || (NSUInteger)offset > logicalLength || (NSUInteger)size > logicalLength - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBufferSubData: range %ld+%ld is outside the buffer", (long)offset, (long)size);
		return;
	}
	if (size > 0) {
		if (!data) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glBufferSubData: data is null for a non-empty update");
			return;
		}
		const bool replacesWholeStore = offset == 0 && (NSUInteger)size == logicalLength;
		if (!makeBufferWritableWithoutStall(buf_id, buffer, !replacesWholeStore)) return;
		invalidateConvertedVertexBuffers(buf_id);
		memcpy((uint8_t *)buffer.contents + (NSUInteger)offset, data, (size_t)size);
	}
}

void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void *data) {
	if (!isBufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetBufferSubData: no buffer storage bound to target 0x%X", target);
		return;
	}
	id<MTLBuffer> buffer = it->second;
	const NSUInteger logicalLength = glBufferLogicalSize(buf_id);
	if (offset < 0 || size < 0 || (NSUInteger)offset > logicalLength || (NSUInteger)size > logicalLength - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetBufferSubData: range %ld+%ld is outside the buffer", (long)offset, (long)size);
		return;
	}
	if (size > 0 && !data) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetBufferSubData: data is null for a non-empty read");
		return;
	}
	if (size > 0) {
		waitForBufferGPUUse(buf_id);
		buffer = ctx.buffer_objects[buf_id];
		memcpy(data, (const uint8_t *)buffer.contents + (NSUInteger)offset, (size_t)size);
	}
}

void glCopyBufferSubData(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) {
	if (!isBufferTarget(readTarget) || !isBufferTarget(writeTarget)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	auto src = ctx.buffer_objects.find(boundBuffer(readTarget));
	auto dst = ctx.buffer_objects.find(boundBuffer(writeTarget));
	if (src == ctx.buffer_objects.end() || dst == ctx.buffer_objects.end() || !src->second || !dst->second) {
		GL4METAL_LOG("glCopyBufferSubData: both targets need a buffer with storage");
		return;
	}
	id<MTLBuffer> from = src->second;
	id<MTLBuffer> to = dst->second;
	const NSUInteger sourceLogicalLength = glBufferLogicalSize(boundBuffer(readTarget));
	const NSUInteger destinationLogicalLength = glBufferLogicalSize(boundBuffer(writeTarget));
	if (readOffset < 0 || writeOffset < 0 || size < 0 || (NSUInteger)readOffset > sourceLogicalLength || (NSUInteger)size > sourceLogicalLength - (NSUInteger)readOffset || (NSUInteger)writeOffset > destinationLogicalLength || (NSUInteger)size > destinationLogicalLength - (NSUInteger)writeOffset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCopyBufferSubData: range does not fit the buffers");
		return;
	}
	// GL rejects overlapping copies within the same buffer; memmove would silently change that API behaviour.
	if (from == to && size > 0) {
		const NSUInteger r0 = (NSUInteger)readOffset, w0 = (NSUInteger)writeOffset, n = (NSUInteger)size;
		if (r0 < w0 + n && w0 < r0 + n) {
			gl4metalSetError(GL_INVALID_VALUE);
			GL4METAL_LOG("glCopyBufferSubData: overlapping ranges in the same buffer");
			return;
		}
	}
	// Prefer an asynchronous MTL4 blit-stage copy on Apple7+/Metal4. The command
	// encoder records resource dependencies and the per-buffer event tracker prevents
	// later CPU mappings/updates from racing this copy. Keep the shared-storage CPU
	// copy as the compatibility fallback.
	if (size > 0) {
		const GLuint sourceName = boundBuffer(readTarget);
		const GLuint destinationName = boundBuffer(writeTarget);
		if (ctx.metal4Enabled && gl4metalMetal4CopyBuffer(from, (NSUInteger)readOffset,
			to, (NSUInteger)writeOffset, (NSUInteger)size)) {
			markBufferMayBeGPUWritten(destinationName);
			invalidateConvertedVertexBuffers(destinationName);
			gl4metalMarkMetal4BufferUsed(sourceName);
			gl4metalMarkMetal4BufferUsed(destinationName);
			return;
		}
		waitForBufferGPUUse(sourceName);
		auto sourceIt = ctx.buffer_objects.find(sourceName);
		if (sourceIt == ctx.buffer_objects.end() || !sourceIt->second) {
			gl4metalSetError(GL_INVALID_OPERATION);
			return;
		}
		from = sourceIt->second;
		if (!makeBufferWritableWithoutStall(destinationName, to)) return;
		if (sourceName == destinationName) from = to;
		invalidateConvertedVertexBuffers(destinationName);
		memmove((uint8_t *)to.contents + (NSUInteger)writeOffset, (const uint8_t *)from.contents + (NSUInteger)readOffset, (size_t)size);
	}
}

// MARK: - Mapping

void *glMapBuffer(GLenum target, GLenum access) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return NULL; }
	if (access != GL_READ_ONLY && access != GL_WRITE_ONLY && access != GL_READ_WRITE) { gl4metalSetError(GL_INVALID_ENUM); return NULL; }
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glMapBuffer: no buffer storage bound to target 0x%X", target);
		return NULL;
	}
	id<MTLBuffer> buffer = it->second;
	if (access == GL_READ_ONLY || access == GL_READ_WRITE) {
		waitForBufferGPUUse(buf_id);
	} else if (!makeBufferWritableWithoutStall(buf_id, buffer)) {
		return NULL;
	}
	invalidateConvertedVertexBuffers(buf_id);
	return buffer.contents; // shared storage: the buffer is the CPU pointer itself
}

void *glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
	if (!isBufferTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return NULL;
	}
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glMapBufferRange: no buffer bound to target 0x%X", target);
		return NULL;
	}
	id<MTLBuffer> buffer = it->second;
	const NSUInteger logicalLength = glBufferLogicalSize(buf_id);
	if (offset < 0 || length < 0 || (NSUInteger)offset > logicalLength || (NSUInteger)length > logicalLength - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glMapBufferRange: range %ld+%ld outside %lu-byte buffer", (long)offset, (long)length, (unsigned long)buffer.length);
		return NULL;
	}
	if (access & GL_MAP_READ_BIT) {
		waitForBufferGPUUse(buf_id);
		buffer = ctx.buffer_objects[buf_id];
	} else if (!makeBufferWritableWithoutStall(buf_id, buffer,
		(access & GL_MAP_INVALIDATE_BUFFER_BIT) == 0)) {
		return NULL;
	}
	invalidateConvertedVertexBuffers(buf_id);
	return (uint8_t *)buffer.contents + offset;
}

void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	const NSUInteger logicalLength = glBufferLogicalSize(buf_id);
	if (offset < 0 || length < 0 || (NSUInteger)offset > logicalLength || (NSUInteger)length > logicalLength - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	// Shared storage is coherent on Apple GPUs; invalidate cached repacks after CPU writes.
	invalidateConvertedVertexBuffers(buf_id);
}

GLboolean glUnmapBuffer(GLenum target) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return GL_FALSE; }
	GLuint buf_id = boundBuffer(target);
	if (buf_id == 0 || ctx.buffer_objects.find(buf_id) == ctx.buffer_objects.end() || !ctx.buffer_objects[buf_id]) { gl4metalSetError(GL_INVALID_OPERATION); return GL_FALSE; }
	invalidateConvertedVertexBuffers(buf_id);
	return GL_TRUE;
}

// MARK: - Buffer queries

void glGetBufferParameteriv(GLenum target, GLenum pname, GLint *params) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); if (params) *params = 0; return; }
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		*params = 0;
		return;
	}
	id<MTLBuffer> buffer = it->second;
	switch (pname) {
		case GL_BUFFER_SIZE:       *params = (GLint)glBufferLogicalSize(buf_id); break;
		case GL_BUFFER_USAGE:      *params = GL_STATIC_DRAW;       break;
		case GL_BUFFER_MAPPED:     *params = GL_FALSE;             break;
		case GL_BUFFER_ACCESS:     *params = GL_READ_WRITE;        break;
		case GL_BUFFER_MAP_OFFSET: *params = 0;                    break;
		case GL_BUFFER_MAP_LENGTH: *params = (GLint)glBufferLogicalSize(buf_id); break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetBufferParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64 *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	GLint v = 0;
	glGetBufferParameteriv(target, pname, &v);
	*params = v;
}

void glGetBufferPointerv(GLenum target, GLenum pname, void **params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!isBufferTarget(target) || pname != GL_BUFFER_MAP_POINTER) { gl4metalSetError(GL_INVALID_ENUM); *params = NULL; return; }
	*params = NULL; // mappings are not tracked: glMapBuffer hands out the CPU pointer of the shared buffer
}
