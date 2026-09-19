#import "gl4metal.h"
#include "glcorearb.h"

extern GLenum lastGL4MetalError;

GLenum APIENTRY glGetError() {
    GLenum error = lastGL4MetalError;
    lastGL4MetalError = GL_NO_ERROR;
    return error;
}

void APIENTRY glGetPointerv(GLenum pname, void**params) {
    if (!params) {
        lastGL4MetalError = GL_INVALID_VALUE;
        return;
    }

    switch (pname) {
        case GL_BUFFER_MAP_POINTER:
            params[0] = NULL;
            lastGL4MetalError = GL_NO_ERROR;
            return;
        default:
            lastGL4MetalError = GL_INVALID_ENUM;
            return;
    }
}

NSString* getExtensions() {
    return @"GL_ARB_buffer_object GL_ARB_vertex_array_object GL_ARB_vertex_buffer_object GL_ARB_framebuffer_object GL_EXT_framebuffer_object GL_OES_vertex_array_object GL_EXT_texture_storage";
}

const GLubyte* APIENTRY glGetString(GLenum name) {
    switch(name) {
        case GL_VENDOR:
            return (const GLubyte *)"congcq";
        case GL_RENDERER:
            return (const GLubyte *)"gl4metal 0.1-beta";
        case GL_VERSION:
            return (const GLubyte *)"3.3.0 Core Profile";
        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)"3.3";
        case GL_EXTENSIONS: {
            NSString* extensions = getExtensions();
            return (const GLubyte *)extensions.UTF8String;
        }
        default:
            return NULL;
    }
}

const GLubyte *APIENTRY glGetStringi(GLenum name, GLuint index) {
    if (index == 0) {
        switch (name) {
            case GL_VENDOR:
                return (const GLubyte *)"congcq";
            case GL_RENDERER:
                return (const GLubyte *)"gl4metal 0.1-beta";
            case GL_VERSION:
                return (const GLubyte *)"3.3.0 Core Profile";
            case GL_SHADING_LANGUAGE_VERSION:
                return (const GLubyte *)"3.3.0";
            case GL_EXTENSIONS: {
                NSString *extensions = getExtensions();
                return (const GLubyte *)extensions.UTF8String;
            }
            default:
                return NULL;
        }
    }
    return NULL;
}