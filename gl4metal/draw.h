#ifndef DRAW_H
#define DRAW_H

#import "gl4metal.h"
#import "framebuffer.h"

// Release the CPU-side reference to the per-command-buffer large-uniform upload arena after submission.
void resetDrawUniformUploadArena(id<MTLCommandBuffer> commandBuffer);

static bool makeSignednessConvertedVertexBuffer(GLuint bufferId, const VertexAttrib &va, id<MTLBuffer> source, NSUInteger &convertedStride, __strong id<MTLBuffer> &converted);

#endif // DRAW_H
