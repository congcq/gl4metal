#ifndef DRAW_H
#define DRAW_H

#import "gl4metal.h"
#import "framebuffer.h"

static bool makeSignednessConvertedVertexBuffer(GLuint bufferId, const VertexAttrib &va, id<MTLBuffer> source, NSUInteger &convertedStride, __strong id<MTLBuffer> &converted);

#endif // DRAW_H
