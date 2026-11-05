//
// framebuffer.h
// gl4metal
//
// Created by congcq on 16.10.26.
//

#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#import "gl4metal.h"

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

id<MTLRenderCommandEncoder> beginRenderPass(GLbitfield clearMask);  // ends the open pass, starts one on glBindFramebuffer's target
void endRenderEncoder(void);
void applyViewportAndScissor(void);
void clearDepthStencil(GLbitfield mask, GLfloat depth, GLint stencil);
void clearOneColorBuffer(GLint drawbuffer, const GLfloat *rgba);

#endif /* FRAMEBUFFER_H */
