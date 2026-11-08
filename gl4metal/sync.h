#ifndef SYNC_H
#define SYNC_H

#import "gl4metal.h"
#import "framebuffer.h"

void flushCommandBuffer(bool waitUntilCompleted); // commits the open command buffer
void installCommandBufferErrorHandler(id<MTLCommandBuffer> commandBuffer);

#endif // SYNC_H
