#ifndef PROGRAM_H
#define PROGRAM_H

#import "gl4metal.h"
#import "shaders.h"

void dispatchComputeInternal(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z, id<MTLBuffer> indirectBuffer = nil, NSUInteger indirectOffset = 0, GLuint indirectBufferName = 0);

#endif // PROGRAM_H
