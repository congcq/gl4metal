#import "gl4metal.h"
#import "gl4metal_runtime.h"

void APIENTRY glGenTextures(GLsizei n, GLuint *textures) {
    if (!textures || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    for (GLsizei i = 0; i < n; i++) {
        textures[i] = nextBufferId++;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glDeleteTextures(GLsizei n, const GLuint *textures) {
    if (!textures || n <= 0) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glBindTexture(GLenum target, GLuint texture) {
    (void)target;
    (void)texture;
    gl4metalSetError(GL_NO_ERROR);
}

void APIENTRY glActiveTexture(GLenum texture) {
    (void)texture;
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

void APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat,
                           GLsizei width, GLsizei height, GLint border,
                           GLenum format, GLenum type, const void *pixels) {
    (void)target;
    (void)level;
    (void)internalformat;
    (void)width;
    (void)height;
    (void)border;
    (void)format;
    (void)type;
    (void)pixels;
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
    (void)pixels;
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
