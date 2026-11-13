#import "program.h"

// GL_ARB_compute_shader public entry points. The backend implementation and
// all resource resolution stay in program.mm.
void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
	dispatchComputeInternal(num_groups_x, num_groups_y, num_groups_z);
}

void glDispatchComputeIndirect(GLintptr indirect) {
	if (indirect < 0 || (indirect & 3) != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDispatchComputeIndirect: offset must be non-negative and 4-byte aligned");
		return;
	}
	const GLuint indirectBufferName = boundBuffer(GL_DISPATCH_INDIRECT_BUFFER);
	if (indirectBufferName == 0) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchComputeIndirect: no GL_DISPATCH_INDIRECT_BUFFER is bound");
		return;
	}
	auto bufferIt = ctx.buffer_objects.find(indirectBufferName);
	if (bufferIt == ctx.buffer_objects.end() || !bufferIt->second) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const NSUInteger offset = (NSUInteger)indirect;
	const NSUInteger logicalSize = glBufferLogicalSize(indirectBufferName);
	if (offset > logicalSize || logicalSize - offset < 3 * sizeof(GLuint) ||
		offset > bufferIt->second.length || bufferIt->second.length - offset < 3 * sizeof(GLuint)) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glDispatchComputeIndirect: argument range exceeds buffer %u", indirectBufferName);
		return;
	}
	dispatchComputeInternal(0, 0, 0, bufferIt->second, offset, indirectBufferName);
}
