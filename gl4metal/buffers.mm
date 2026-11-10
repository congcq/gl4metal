//
// buffers.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "buffers.h"

#import "sync.h"
#import "texture.h"

#include <mutex>
#include <unordered_set>

// MARK: - Buffer objects

static bool isBufferTarget(GLenum target) {
	return target == GL_ARRAY_BUFFER || target == GL_ELEMENT_ARRAY_BUFFER ||
		target == GL_UNIFORM_BUFFER || target == GL_SHADER_STORAGE_BUFFER || target == GL_COPY_READ_BUFFER ||
		target == GL_COPY_WRITE_BUFFER || target == GL_PIXEL_PACK_BUFFER ||
		target == GL_PIXEL_UNPACK_BUFFER || target == GL_TEXTURE_BUFFER;
}

// A buffer used by an encoded draw cannot be mutated in place while its command buffer
// is pending. Track the last command buffer per GL object and rename only that object's
// Metal storage when a CPU write overlaps in-flight GPU work.
static std::mutex bufferUseMutex;
static std::unordered_set<const void *> bufferCleanupRegistered;

static bool commandBufferPending(id<MTLCommandBuffer> commandBuffer) {
	if (!commandBuffer) return false;
	const MTLCommandBufferStatus status = commandBuffer.status;
	return status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError;
}

void markBuffersUsed(const GLuint *buf_ids, size_t objectCount) {
	id<MTLCommandBuffer> activeBuffer = ctx.currentCommandBuffer;
	if (!activeBuffer || !buf_ids || objectCount == 0) return;
	const void *bufferToken = (__bridge const void *)activeBuffer;
	bool installCleanup = false;
	{
		std::lock_guard<std::mutex> lock(bufferUseMutex);
		for (size_t index = 0; index < objectCount; ++index) {
			const GLuint buf_id = buf_ids[index];
			if (buf_id != 0) ctx.buffer_last_use[buf_id] = activeBuffer;
		}
		installCleanup = bufferCleanupRegistered.insert(bufferToken).second;
	}
	if (installCleanup) {
		[activeBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
			const void *completedToken = (__bridge const void *)completed;
			std::lock_guard<std::mutex> lock(bufferUseMutex);
			for (auto it = ctx.buffer_last_use.begin(); it != ctx.buffer_last_use.end();) {
				if (it->second == completed) it = ctx.buffer_last_use.erase(it);
				else ++it;
			}
			bufferCleanupRegistered.erase(completedToken);
		}];
	}
}

void markBufferUsed(GLuint buf_id) {
	markBuffersUsed(&buf_id, 1);
}

static bool bufferHasPendingGPUUse(GLuint buf_id) {
	std::lock_guard<std::mutex> lock(bufferUseMutex);
	auto it = ctx.buffer_last_use.find(buf_id);
	if (it == ctx.buffer_last_use.end()) return false;
	if (commandBufferPending(it->second)) return true;
	ctx.buffer_last_use.erase(it);
	return false;
}

static bool makeBufferWritableWithoutStall(GLuint buf_id, __strong id<MTLBuffer> &buffer, bool preserveContents = true) {
	if (!bufferHasPendingGPUUse(buf_id)) return true;
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
	{
		std::lock_guard<std::mutex> lock(bufferUseMutex);
		ctx.buffer_last_use.erase(buf_id);
	}
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
	id<MTLCommandBuffer> lastUse = nil;
	{
		std::lock_guard<std::mutex> lock(bufferUseMutex);
		auto it = ctx.buffer_last_use.find(buf_id);
		if (it == ctx.buffer_last_use.end()) return;
		lastUse = it->second;
		if (!commandBufferPending(lastUse)) {
			ctx.buffer_last_use.erase(it);
			return;
		}
	}
	// If this is the still-open command buffer, submit it before waiting. Do not flush
	// unrelated work when the buffer was last used by an already committed command buffer.
	if (lastUse == ctx.currentCommandBuffer) {
		flushCommandBuffer(false);
	}
	[lastUse waitUntilCompleted];
	{
		std::lock_guard<std::mutex> lock(bufferUseMutex);
		auto it = ctx.buffer_last_use.find(buf_id);
		if (it != ctx.buffer_last_use.end() && it->second == lastUse) ctx.buffer_last_use.erase(it);
	}
}

void glGenBuffers(GLsizei n, GLuint *buffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint buf_id = ++ctx.next_buffer_id; // names start at 1, 0 is "no buffer"
		ctx.buffer_objects[buf_id] = nil;     // name exists, storage comes with glBufferData
		ctx.buffer_sizes[buf_id] = 0;
		buffers[i] = buf_id;
	}
}

void glDeleteBuffers(GLsizei n, const GLuint *buffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint buf_id = buffers[i];
		if (buf_id == 0) {
			continue;
		}
		for (auto &binding : ctx.bound_buffers) {
			if (binding.second == buf_id) {
				binding.second = 0;
			}
		}
		for (auto &vao : ctx.vao_objects) {
			if (vao.second.element_array_buffer_id == buf_id) {
				vao.second.element_array_buffer_id = 0;
			}
			for (VertexAttrib &attrib : vao.second.attribs) {
				if (attrib.bound_vbo == buf_id) {
					attrib.bound_vbo = 0;
					attrib.pointer = nullptr;
					attrib.enabled = false;
				}
			}
		}
		for (UniformBufferBinding &binding : ctx.uniform_buffer_bindings) {
			if (binding.buffer == buf_id) binding = UniformBufferBinding();
		}
		for (ShaderStorageBufferBinding &binding : ctx.shader_storage_buffer_bindings) {
			if (binding.buffer == buf_id) binding = ShaderStorageBufferBinding();
		}
		// Deleting a GL buffer detaches it from every texture-buffer object. The
		// command buffer retains any old Metal resources already referenced by draws.
		for (auto &textureEntry : ctx.texture_objects) {
			TextureObject &textureObject = textureEntry.second;
			if (textureObject.buffer_object == buf_id) {
				textureObject.buffer_object = 0;
				textureObject.buffer_sampler_type = 0;
				textureObject.texture = nil;
				textureObject.width = 0;
				textureObject.height = 0;
			}
		}
		invalidateConvertedVertexBuffers(buf_id);
		{
			std::lock_guard<std::mutex> lock(bufferUseMutex);
			ctx.buffer_last_use.erase(buf_id);
		}
		ctx.buffer_objects.erase(buf_id); // deleted buffer references were detached from texture-buffer objects above
		ctx.buffer_sizes.erase(buf_id);
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
		ctx.buffer_sizes[buffer] = 0;
	}
	ctx.bound_buffers[target] = buffer;
	if (target == GL_ELEMENT_ARRAY_BUFFER) {
		ctx.vao_objects[ctx.current_vao].element_array_buffer_id = buffer; // element buffer binding is VAO state
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
		ctx.buffer_sizes[buffer] = 0;
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
	// offset for SSBOs only needs 4-byte alignment on iOS.
	const GLintptr alignment = uniformTarget ? 256 : 4;
	if ((offset % alignment) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: offset %lld is not aligned to %lld bytes", (long long)offset, (long long)alignment);
		return;
	}
	auto object = ctx.buffer_objects.find(buffer);
	if (object == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil;
		ctx.buffer_sizes[buffer] = 0;
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

	// iOS only has shared (unified memory) buffers: no Managed mode, so no didModifyRange: is needed after CPU writes.
	NSUInteger length = std::max<NSUInteger>((NSUInteger)size, 1); // Metal rejects zero-length buffers
	id<MTLBuffer> buffer = nil;
	if (data && size > 0) {
		buffer = [ctx.device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
	} else {
		buffer = [ctx.device newBufferWithLength:length options:MTLResourceStorageModeShared]; // zero filled
	}
	if (!buffer) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		GL4METAL_LOG("glBufferData: Metal allocation failed for %llu bytes; previous buffer storage preserved", (unsigned long long)length);
		return;
	}
	invalidateConvertedVertexBuffers(buf_id);
	{
		std::lock_guard<std::mutex> lock(bufferUseMutex);
		ctx.buffer_last_use.erase(buf_id);
	}
	ctx.buffer_objects[buf_id] = buffer;
	ctx.buffer_sizes[buf_id] = (NSUInteger)size;
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
	// Shared storage: a CPU copy. If the destination is still referenced by GPU work,
	// rename it instead of serializing the full command queue. `from` remains the prior
	// version when readTarget and writeTarget name the same object.
	if (size > 0) {
		GLuint destinationName = boundBuffer(writeTarget);
		if (!makeBufferWritableWithoutStall(destinationName, to)) return;
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
