//
// buffers.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "buffers.h"

#import "sync.h"

// MARK: - Buffer objects

static bool isBufferTarget(GLenum target) {
	return target == GL_ARRAY_BUFFER || target == GL_ELEMENT_ARRAY_BUFFER ||
		target == GL_UNIFORM_BUFFER || target == GL_COPY_READ_BUFFER ||
		target == GL_COPY_WRITE_BUFFER || target == GL_PIXEL_PACK_BUFFER ||
		target == GL_PIXEL_UNPACK_BUFFER;
}

// Shared MTLBuffer contents are CPU-visible, but the CPU must not mutate bytes while a
// previously encoded GPU command may still read them. Conservative barrier for map/subdata
// APIs: only waits when work is pending; glBufferData uses storage renaming and needs no wait.
static void waitForPriorGPUBufferUse(void) {
	if (ctx.currentEncoder || ctx.currentCommandBuffer ||
		(ctx.last_committed_buffer && ctx.last_committed_buffer.status != MTLCommandBufferStatusCompleted &&
		 ctx.last_committed_buffer.status != MTLCommandBufferStatusError)) {
		flushCommandBuffer(true);
	}
}

void glGenBuffers(GLsizei n, GLuint *buffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = ++ctx.next_buffer_id; // names start at 1, 0 is "no buffer"
		ctx.buffer_objects[id] = nil;     // name exists, storage comes with glBufferData
		buffers[i] = id;
	}
}

void glDeleteBuffers(GLsizei n, const GLuint *buffers) {
	for (GLsizei i = 0; i < n; i++) {
		GLuint id = buffers[i];
		if (id == 0) {
			continue;
		}
		for (auto &binding : ctx.bound_buffers) {
			if (binding.second == id) {
				binding.second = 0;
			}
		}
		for (auto &vao : ctx.vao_objects) {
			if (vao.second.element_array_buffer_id == id) {
				vao.second.element_array_buffer_id = 0;
			}
			for (VertexAttrib &attrib : vao.second.attribs) {
				if (attrib.bound_vbo == id) {
					attrib.bound_vbo = 0;
					attrib.pointer = nullptr;
					attrib.enabled = false;
				}
			}
		}
		for (UniformBufferBinding &binding : ctx.uniform_buffer_bindings) {
			if (binding.buffer == id) {
				binding = UniformBufferBinding();
			}
		}
		invalidateConvertedVertexBuffers(id);
		ctx.buffer_objects.erase(id); // ARC releases the MTLBuffer once in-flight command buffers are done with it
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
	}
	ctx.bound_buffers[target] = buffer;
	if (target == GL_ELEMENT_ARRAY_BUFFER) {
		ctx.vao_objects[ctx.current_vao].element_array_buffer_id = buffer; // element buffer binding is VAO state
	}
}

void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
	if (target != GL_UNIFORM_BUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBindBufferBase: unsupported target 0x%X", target);
		return;
	}
	if (index >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferBase: binding point %u is out of range", index);
		return;
	}
	if (buffer != 0 && ctx.buffer_objects.find(buffer) == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil; // names are materialized on first bind, storage is allocated separately
	}
	UniformBufferBinding newBinding;
	newBinding.buffer = buffer;
	newBinding.offset = 0;
	newBinding.size = 0; // size 0 means the whole current store
	ctx.uniform_buffer_bindings[index] = newBinding;
	ctx.bound_buffers[target] = buffer;
}

void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
	if (target != GL_UNIFORM_BUFFER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glBindBufferRange: unsupported target 0x%X", target);
		return;
	}
	if (index >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: binding point %u is out of range", index);
		return;
	}
	if (buffer == 0) {
		// Match the indexed-binding unbind semantics without leaving a stale range behind.
		ctx.uniform_buffer_bindings[index] = UniformBufferBinding();
		ctx.bound_buffers[target] = 0;
		return;
	}
	if (offset < 0 || size <= 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: offset must be non-negative and size must be positive (offset=%lld size=%lld)", (long long)offset, (long long)size);
		return;
	}
	// The value returned by GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT is 256 in getter.mm.
	if ((offset % 256) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: offset %lld is not aligned to GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT (256)", (long long)offset);
		return;
	}
	auto object = ctx.buffer_objects.find(buffer);
	if (object == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil;
		object = ctx.buffer_objects.find(buffer);
	}
	if (object == ctx.buffer_objects.end() || !object->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glBindBufferRange: buffer %u has no data store", buffer);
		return;
	}
	const NSUInteger length = object->second.length;
	const NSUInteger off = (NSUInteger)offset;
	const NSUInteger bytes = (NSUInteger)size;
	if (off > length || bytes > length - off) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindBufferRange: range %llu+%llu exceeds buffer %u length %llu",
			(unsigned long long)off, (unsigned long long)bytes, buffer, (unsigned long long)length);
		return;
	}
	UniformBufferBinding newBinding;
	newBinding.buffer = buffer;
	newBinding.offset = offset;
	newBinding.size = size;
	ctx.uniform_buffer_bindings[index] = newBinding;
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
	ctx.buffer_objects[buf_id] = buffer;
}

void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
	if (target != GL_ARRAY_BUFFER && target != GL_ELEMENT_ARRAY_BUFFER && target != GL_UNIFORM_BUFFER &&
		target != GL_COPY_READ_BUFFER && target != GL_COPY_WRITE_BUFFER && target != GL_PIXEL_PACK_BUFFER && target != GL_PIXEL_UNPACK_BUFFER) {
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
	if (offset < 0 || size < 0 || (NSUInteger)offset > buffer.length || (NSUInteger)size > buffer.length - (NSUInteger)offset) {
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
		waitForPriorGPUBufferUse();
		invalidateConvertedVertexBuffers(buf_id);
		memcpy((uint8_t *)buffer.contents + (NSUInteger)offset, data, (size_t)size);
	}
}

void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void *data) {
	if (target != GL_ARRAY_BUFFER && target != GL_ELEMENT_ARRAY_BUFFER && target != GL_UNIFORM_BUFFER &&
		target != GL_COPY_READ_BUFFER && target != GL_COPY_WRITE_BUFFER && target != GL_PIXEL_PACK_BUFFER && target != GL_PIXEL_UNPACK_BUFFER) {
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
	if (offset < 0 || size < 0 || (NSUInteger)offset > buffer.length || (NSUInteger)size > buffer.length - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetBufferSubData: range %ld+%ld is outside the buffer", (long)offset, (long)size);
		return;
	}
	if (size > 0 && !data) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetBufferSubData: data is null for a non-empty read");
		return;
	}
	if (size > 0) memcpy(data, (const uint8_t *)buffer.contents + (NSUInteger)offset, (size_t)size);
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
	if (readOffset < 0 || writeOffset < 0 || size < 0 || (NSUInteger)readOffset > from.length || (NSUInteger)size > from.length - (NSUInteger)readOffset || (NSUInteger)writeOffset > to.length || (NSUInteger)size > to.length - (NSUInteger)writeOffset) {
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
	// Shared storage: a CPU copy. memmove is safe for disjoint ranges in the same object.
	if (size > 0) {
		waitForPriorGPUBufferUse();
		invalidateConvertedVertexBuffers(boundBuffer(writeTarget));
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
	waitForPriorGPUBufferUse();
	invalidateConvertedVertexBuffers(buf_id);
	return it->second.contents; // shared storage: the buffer is the CPU pointer itself
}

void *glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
	if (target != GL_ARRAY_BUFFER && target != GL_ELEMENT_ARRAY_BUFFER && target != GL_UNIFORM_BUFFER &&
		target != GL_COPY_READ_BUFFER && target != GL_COPY_WRITE_BUFFER && target != GL_PIXEL_PACK_BUFFER && target != GL_PIXEL_UNPACK_BUFFER) {
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
	if (offset < 0 || length < 0 || (NSUInteger)offset > buffer.length || (NSUInteger)length > buffer.length - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glMapBufferRange: range %ld+%ld outside %lu-byte buffer", (long)offset, (long)length, (unsigned long)buffer.length);
		return NULL;
	}
	waitForPriorGPUBufferUse();
	invalidateConvertedVertexBuffers(buf_id);
	return (uint8_t *)buffer.contents + offset;
}

void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return; }
	GLuint id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(id);
	if (id == 0 || it == ctx.buffer_objects.end() || !it->second) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	if (offset < 0 || length < 0 || (NSUInteger)offset > it->second.length || (NSUInteger)length > it->second.length - (NSUInteger)offset) {
		gl4metalSetError(GL_INVALID_VALUE); return;
	}
	// Shared storage is coherent on Apple GPUs; invalidate cached repacks after CPU writes.
	invalidateConvertedVertexBuffers(id);
}

GLboolean glUnmapBuffer(GLenum target) {
	if (!isBufferTarget(target)) { gl4metalSetError(GL_INVALID_ENUM); return GL_FALSE; }
	GLuint id = boundBuffer(target);
	if (id == 0 || ctx.buffer_objects.find(id) == ctx.buffer_objects.end() || !ctx.buffer_objects[id]) { gl4metalSetError(GL_INVALID_OPERATION); return GL_FALSE; }
	invalidateConvertedVertexBuffers(id);
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
		case GL_BUFFER_SIZE:       *params = (GLint)buffer.length; break;
		case GL_BUFFER_USAGE:      *params = GL_STATIC_DRAW;       break;
		case GL_BUFFER_MAPPED:     *params = GL_FALSE;             break;
		case GL_BUFFER_ACCESS:     *params = GL_READ_WRITE;        break;
		case GL_BUFFER_MAP_OFFSET: *params = 0;                    break;
		case GL_BUFFER_MAP_LENGTH: *params = (GLint)buffer.length; break;
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
