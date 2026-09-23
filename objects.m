#import "gl4metal.h"
#import "gl4metal_runtime.h"

void APIENTRY glGenTextures(GLsizei n, GLuint *textures) {
    if (!textures || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        textures[i] = nextTextureId++;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteTextures(GLsizei n, const GLuint *textures) {
    if (!textures || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        [textureObjects removeObjectForKey:@(textures[i])];
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsTexture(GLuint texture) {
    return textureObjects[@(texture)] != nil ? GL_TRUE : GL_FALSE;
}

void APIENTRY glBindTexture(GLenum target, GLuint texture) {
    if (target != GL_TEXTURE_2D && target != GL_TEXTURE_CUBE_MAP) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    boundTextureUnits[@(activeTextureUnit - GL_TEXTURE0)] = @(texture);
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glActiveTexture(GLenum texture) {
    if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + 32) {
        gl4metalSetError(GL_INVALID_ENUM);
        return;
    }
    activeTextureUnit = texture;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexParameteri(GLenum target, GLenum pname, GLint param) {
    (void)target;
    (void)pname;
    (void)param;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexParameteriv(GLenum target, GLenum pname, const GLint *params) {
    (void)target;
    (void)pname;
    (void)params;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
    (void)target;
    (void)pname;
    (void)param;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) {
    (void)target;
    (void)pname;
    (void)params;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
    (void)target;
    (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
    (void)target;
    (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0.0f;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexImage1D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLint border, GLenum format,
                           GLenum type, const void *pixels) {
    (void)target; (void)level; (void)internalformat; (void)width;
    (void)border; (void)format; (void)type; (void)pixels;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexSubImage1D(GLenum target, GLint level, GLint xoffset,
                              GLsizei width, GLenum format, GLenum type,
                              const void *pixels) {
    (void)target; (void)level; (void)xoffset; (void)width;
    (void)format; (void)type; (void)pixels;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint *params) {
    (void)target; (void)level; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat *params) {
    (void)target; (void)level; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0.0f;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenerateMipmap(GLenum target) {
    (void)target;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexParameterIiv(GLenum target, GLenum pname, const GLint *params) {
    glTexParameteriv(target, pname, params);
}

void APIENTRY glTexParameterIuiv(GLenum target, GLenum pname, const GLuint *params) {
    (void)target; (void)pname; (void)params;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetTexParameterIiv(GLenum target, GLenum pname, GLint *params) {
    glGetTexParameteriv(target, pname, params);
}

void APIENTRY glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint *params) {
    (void)target; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLsizei height, GLint border,
                           GLenum format, GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_2D || level < 0 || width <= 0 || height <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    (void)level;
    (void)internalformat;
    (void)width;
    (void)height;
    (void)border;
    (void)format;
    (void)type;
    GLuint texture = [boundTextureUnits[@(activeTextureUnit - GL_TEXTURE0)] unsignedIntValue];
    if (texture == 0 || !ctx.device) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }

    MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (format == GL_BGRA) pixelFormat = MTLPixelFormatBGRA8Unorm;
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                                             width:(NSUInteger)width
                                                                                            height:(NSUInteger)height
                                                                                         mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> textureObject = [ctx.device newTextureWithDescriptor:descriptor];
    if (!textureObject) {
        gl4metalSetError(GL_OUT_OF_MEMORY);
        return;
    }
    if (pixels && type == GL_UNSIGNED_BYTE) {
        NSUInteger bytesPerRow = (NSUInteger)width * 4;
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
        [textureObject replaceRegion:region mipmapLevel:0 withBytes:pixels bytesPerRow:bytesPerRow];
    }
    textureObjects[@(texture)] = textureObject;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                              GLint yoffset, GLsizei width, GLsizei height,
                              GLenum format, GLenum type, const void *pixels) {
    (void)target;
    (void)level;
    (void)xoffset;
    (void)yoffset;
    (void)width;
    (void)height;
    (void)format;
    (void)type;
    GLuint texture = [boundTextureUnits[@(activeTextureUnit - GL_TEXTURE0)] unsignedIntValue];
    id<MTLTexture> textureObject = textureObjects[@(texture)];
    if (!textureObject || !pixels || type != GL_UNSIGNED_BYTE) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return;
    }
    NSUInteger bytesPerRow = (NSUInteger)width * 4;
    MTLRegion region = MTLRegionMake2D((NSUInteger)xoffset, (NSUInteger)yoffset, (NSUInteger)width, (NSUInteger)height);
    [textureObject replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:pixels bytesPerRow:bytesPerRow];
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    if (!framebuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++) {
        GLuint framebuffer = ++nextBufferId;
        framebuffers[i] = framebuffer;
        framebufferObjects[@(framebuffer)] = @(framebuffer);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer) {
    (void)target;
    if (framebuffer == 0) {
        ctx.currentFramebuffer = 0;
        ctx.currentReadFramebuffer = 0;
    } else {
        ctx.currentFramebuffer = framebuffer;
        ctx.currentReadFramebuffer = framebuffer;
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLenum APIENTRY glCheckFramebufferStatus(GLenum target) {
    (void)target;
    if (!ctx) {
        gl4metalSetError(GL_INVALID_OPERATION);
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }

    gl4metalSetError(GL_NO_ERROR);
    return GL_FRAMEBUFFER_COMPLETE;
}

void APIENTRY glFramebufferTexture2D(GLenum target, GLenum attachment,
                                     GLenum textarget, GLuint texture, GLint level) {
    (void)target;
    (void)attachment;
    (void)textarget;
    (void)texture;
    (void)level;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    glFramebufferTexture2D(target, attachment, textarget, texture, level);
}

void APIENTRY glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset) {
    (void)zoffset;
    glFramebufferTexture2D(target, attachment, textarget, texture, level);
}

void APIENTRY glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    (void)layer;
    glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
}

void APIENTRY glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
    glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
}

void APIENTRY glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    (void)target; (void)attachment; (void)renderbuffertarget; (void)renderbuffer;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint *params) {
    (void)target; (void)attachment; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) {
    (void)samples;
    glRenderbufferStorage(target, internalformat, width, height);
}

void APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    if (!framebuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        [framebufferObjects removeObjectForKey:@(framebuffers[i])];
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsFramebuffer(GLuint framebuffer) {
    return framebufferObjects[@(framebuffer)] != nil ? GL_TRUE : GL_FALSE;
}

void APIENTRY glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) {
    if (!renderbuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++) {
        GLuint renderbuffer = ++nextBufferId;
        renderbuffers[i] = renderbuffer;
        renderbufferObjects[@(renderbuffer)] = @(renderbuffer);
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    (void)target;
    (void)renderbuffer;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glRenderbufferStorage(GLenum target, GLenum internalformat,
                                    GLsizei width, GLsizei height) {
    (void)target;
    (void)internalformat;
    (void)width;
    (void)height;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    if (!renderbuffers || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        [renderbufferObjects removeObjectForKey:@(renderbuffers[i])];
    }
    gl4metalSetError(GL_NO_ERROR);
}

GLboolean APIENTRY glIsRenderbuffer(GLuint renderbuffer) {
    return renderbufferObjects[@(renderbuffer)] != nil ? GL_TRUE : GL_FALSE;
}

void APIENTRY glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint *params) {
    (void)target; (void)pname;
    if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
    params[0] = 0;
    gl4metalSetError(GL_NO_ERROR);
}
