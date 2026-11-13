#ifndef DRAW_H
#define DRAW_H

#import "gl4metal.h"
#import "framebuffer.h"

// Release the CPU-side reference to the per-command-buffer large-uniform upload arena after submission.
void resetDrawUniformUploadArena(id<MTLCommandBuffer> commandBuffer);
id<MTLTexture> gl4metalFallbackTextureForUniform(const TranslatedUniform &uniform);
id<MTLSamplerState> gl4metalFallbackSampler(void);
id<MTLSamplerState> gl4metalSamplerForTexture(const TextureObject &texture, GLuint textureUnit);
bool uploadLargeDefaultUniformBlock(const uint8_t *source, NSUInteger length, __strong id<MTLBuffer> &destinationBuffer, NSUInteger &destinationOffset);

static bool makeSignednessConvertedVertexBuffer(GLuint bufferId, const VertexAttrib &va, id<MTLBuffer> source, NSUInteger &convertedStride, __strong id<MTLBuffer> &converted);

static GLenum targetForDim(TranslatedUniform::Dim dim);

#endif // DRAW_H
