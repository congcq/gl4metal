#ifndef DRAW_H
#define DRAW_H

#import "gl4metal.h"
#import "framebuffer.h"

// Mark the command-buffer upload arena submitted; recycle it only after Metal completes that buffer.
void resetDrawTransientUploadArena(id<MTLCommandBuffer> commandBuffer);
id<MTLTexture> gl4metalFallbackTextureForUniform(const TranslatedUniform &uniform);
id<MTLSamplerState> gl4metalFallbackSampler(void);
id<MTLSamplerState> gl4metalSamplerForTexture(const TextureObject &texture, GLuint textureUnit);
// Returns a safe GPU binding for the current large default-uniform revision: a command-buffer arena slice while changing, or an immutable snapshot when stable.
bool programUniformBufferForStage(ProgramObject &program, NSUInteger stage,
	__strong id<MTLBuffer> &destinationBuffer, NSUInteger &destinationOffset);

static bool makeSignednessConvertedVertexBuffer(GLuint bufferId, const VertexAttrib &va, id<MTLBuffer> source, NSUInteger &convertedStride, __strong id<MTLBuffer> &converted);

static GLenum targetForDim(TranslatedUniform::Dim dim);

#endif // DRAW_H
