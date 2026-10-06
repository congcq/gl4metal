#import "gl4metal_runtime.h"
#include "glcorearb.h"

GLenum APIENTRY glGetError() {
    if (!gl4metalPendingErrors || gl4metalPendingErrors.count == 0) {
        lastGL4MetalError = GL_NO_ERROR;
        return GL_NO_ERROR;
    }
    GLenum error = gl4metalPendingErrors.firstObject.unsignedIntValue;
    [gl4metalPendingErrors removeObjectAtIndex:0];
    lastGL4MetalError = gl4metalPendingErrors.count
        ? gl4metalPendingErrors.firstObject.unsignedIntValue : GL_NO_ERROR;
    return error;
}

void APIENTRY glGetPointerv(GLenum pname, void**params) {
    if (!params) {
        gl4metalSetError(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_BUFFER_MAP_POINTER:
            params[0] = NULL;
            gl4metalSetError(GL_NO_ERROR);
            return;
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return;
    }
}

static NSArray<NSString *> *gl4metalGetExtensionList(void) {
    static NSArray<NSString *> *cachedExtensions = nil;
    if (cachedExtensions == nil) {
        // No optional GL extensions are exposed. The previous list claimed
        // multisample, ASTC, framebuffer-fetch, and VAO aliases that this
        // backend does not implement as those extensions specify.
        cachedExtensions = @[];
    }
    return cachedExtensions;
}

NSString* getExtensions() {
    NSArray<NSString *> *extensions = gl4metalGetExtensionList();
    return [extensions componentsJoinedByString:@" "];
}

GLint gl4metalGetExtensionCount(void) {
    return (GLint)gl4metalGetExtensionList().count;
}

const GLubyte* APIENTRY glGetString(GLenum name) {
    static const char *vendor = "congcq";
    static const char *version = "3.3.0 GL4Metal";
    static const char *slVersion = "3.30";

    switch(name) {
        case GL_VENDOR:
            return (const GLubyte *)vendor;
        case GL_RENDERER:
            return (const GLubyte *)(gl4metalGetDeviceName() ?: "gl4metal");
        case GL_VERSION:
            return (const GLubyte *)version;
        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)slVersion;
        case GL_EXTENSIONS: {
            gl4metalSetError(GL_INVALID_ENUM);
            return NULL;
        }
        default:
            gl4metalSetError(GL_INVALID_ENUM);
            return NULL;
    }
}

const GLubyte *APIENTRY glGetStringi(GLenum name, GLuint index) {
    if (name != GL_EXTENSIONS) {
        gl4metalSetError(GL_INVALID_ENUM);
        return NULL;
    }
    NSArray<NSString *> *extensions = gl4metalGetExtensionList();
    if (index >= (GLuint)extensions.count) {
        gl4metalSetError(GL_INVALID_VALUE);
        return NULL;
    }
    NSString *extension = extensions[index];
    return (const GLubyte *)extension.UTF8String;
}
