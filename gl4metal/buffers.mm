//
// buffers.mm
// gl4metal
//
// Created by congcq on 17.10.26.
//

#import "buffers.h"

// MARK: - Buffer objects

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
		}
		ctx.buffer_objects.erase(id); // ARC releases the MTLBuffer once in-flight command buffers are done with it
	}
}

GLboolean glIsBuffer(GLuint buffer) {
	return buffer != 0 && ctx.buffer_objects.find(buffer) != ctx.buffer_objects.end();
}

void glBindBuffer(GLenum target, GLuint buffer) {
	if (buffer != 0 && ctx.buffer_objects.find(buffer) == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil; // GL allows binding a name that was never generated
	}
	ctx.bound_buffers[target] = buffer;
	if (target == GL_ELEMENT_ARRAY_BUFFER) {
		ctx.vao_objects[ctx.current_vao].element_array_buffer_id = buffer; // element buffer binding is VAO state
	}
}

void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
	glBindBufferRange(target, index, buffer, 0, 0); // size 0 = the whole buffer
}

void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
	if (target != GL_UNIFORM_BUFFER) {
		GL4METAL_LOG("glBindBufferRange: unsupported target 0x%X", target);
		return;
	}
	if (index >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
		GL4METAL_LOG("glBindBufferRange: binding point %u is out of range", index);
		return;
	}
	if (buffer != 0 && ctx.buffer_objects.find(buffer) == ctx.buffer_objects.end()) {
		ctx.buffer_objects[buffer] = nil; // GL allows binding a name that was never generated
	}
	UniformBufferBinding &binding = ctx.uniform_buffer_bindings[index];
	binding.buffer = buffer;
	binding.offset = offset;
	binding.size = size;
	ctx.bound_buffers[target] = buffer; // also the generic binding point
}

// MARK: - Buffer data

void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
	GLuint buf_id = boundBuffer(target);
	if (buf_id == 0) {
		GL4METAL_LOG("glBufferData: no buffer bound to target 0x%X", target);
		return;
	}
	if (size < 0 || (NSUInteger)size > ctx.device.maxBufferLength) {
		GL4METAL_LOG("glBufferData: invalid size %ld", (long)size);
		return;
	}

	// iOS only has shared (unified memory) buffers: no Managed mode, so no didModifyRange: is needed after CPU writes.
	NSUInteger length = std::max<NSUInteger>((NSUInteger)size, 1); // Metal rejects zero-length buffers
	id<MTLBuffer> buffer;
	if (data && size > 0) {
		buffer = [ctx.device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
	} else {
		buffer = [ctx.device newBufferWithLength:length options:MTLResourceStorageModeShared]; // zero filled
	}
	ctx.buffer_objects[buf_id] = buffer;
}

void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		GL4METAL_LOG("glBufferSubData: no buffer storage bound to target 0x%X", target);
		return;
	}
	id<MTLBuffer> buffer = it->second;
	if (offset < 0 || size < 0 || (NSUInteger)(offset + size) > buffer.length) {
		GL4METAL_LOG("glBufferSubData: range %ld+%ld is outside the buffer", (long)offset, (long)size);
		return;
	}
	if (size > 0) {
		memcpy((uint8_t *)buffer.contents + offset, data, (size_t)size);
	}
}

void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void *data) {
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		GL4METAL_LOG("glGetBufferSubData: no buffer storage bound to target 0x%X", target);
		return;
	}
	id<MTLBuffer> buffer = it->second;
	if (offset < 0 || size < 0 || (NSUInteger)(offset + size) > buffer.length) {
		GL4METAL_LOG("glGetBufferSubData: range %ld+%ld is outside the buffer", (long)offset, (long)size);
		return;
	}
	memcpy(data, (const uint8_t *)buffer.contents + offset, (size_t)size);
}

void glCopyBufferSubData(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) {
	auto src = ctx.buffer_objects.find(boundBuffer(readTarget));
	auto dst = ctx.buffer_objects.find(boundBuffer(writeTarget));
	if (src == ctx.buffer_objects.end() || dst == ctx.buffer_objects.end() || !src->second || !dst->second) {
		GL4METAL_LOG("glCopyBufferSubData: both targets need a buffer with storage");
		return;
	}
	id<MTLBuffer> from = src->second;
	id<MTLBuffer> to = dst->second;
	if (readOffset < 0 || writeOffset < 0 || size < 0 || (NSUInteger)(readOffset + size) > from.length || (NSUInteger)(writeOffset + size) > to.length) {
		GL4METAL_LOG("glCopyBufferSubData: range does not fit the buffers");
		return;
	}
	// Shared storage: a CPU copy. memmove because source and destination can be the same buffer.
	memmove((uint8_t *)to.contents + writeOffset, (const uint8_t *)from.contents + readOffset, (size_t)size);
}

// MARK: - Mapping

void *glMapBuffer(GLenum target, GLenum access) {
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		GL4METAL_LOG("glMapBuffer: no buffer storage bound to target 0x%X", target);
		return NULL;
	}
	return it->second.contents; // shared storage: the CPU pointer is the buffer itself
}

void *glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
	GLuint buf_id = boundBuffer(target);
	auto it = ctx.buffer_objects.find(buf_id);
	if (buf_id == 0 || it == ctx.buffer_objects.end() || !it->second) {
		GL4METAL_LOG("glMapBufferRange: no buffer bound to target 0x%X", target);
		return NULL;
	}
	id<MTLBuffer> buffer = it->second;
	if (offset < 0 || length < 0 || (NSUInteger)(offset + length) > buffer.length) {
		GL4METAL_LOG("glMapBufferRange: range %ld+%ld outside %lu-byte buffer", (long)offset, (long)length, (unsigned long)buffer.length);
		return NULL;
	}
	return (uint8_t *)buffer.contents + offset;
}

void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
	// Shared storage is coherent on Apple GPUs: nothing to flush.
}

GLboolean glUnmapBuffer(GLenum target) {
	return GL_TRUE;
}

// MARK: - Buffer queries

void glGetBufferParameteriv(GLenum target, GLenum pname, GLint *params) {
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
			GL4METAL_LOG("glGetBufferParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64 *params) {
	GLint v = 0;
	glGetBufferParameteriv(target, pname, &v);
	*params = v;
}

void glGetBufferPointerv(GLenum target, GLenum pname, void **params) {
	GL4METAL_LOG("glGetBufferPointerv: Unimplemented");
}
