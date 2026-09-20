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

static NSArray<NSString *> *gl4metalGetExtensionList(void) {
    static NSArray<NSString *> *cachedExtensions = nil;
    if (cachedExtensions == nil) {
        NSMutableArray<NSString *> *items = [NSMutableArray arrayWithArray:@[
            @"GL_ARB_buffer_object",
            @"GL_ARB_vertex_array_object",
            @"GL_ARB_vertex_buffer_object",
            @"GL_ARB_framebuffer_object",
            @"GL_EXT_framebuffer_object",
            @"GL_OES_vertex_array_object",
            @"GL_EXT_texture_storage",
            @"GL_EXT_texture_filter_anisotropic"
        ]];

        NSUInteger family = gl4metalGetGPUFamily();
        if (family > 0) {
            [items addObject:[NSString stringWithFormat:@"GL_APPLE_gpu_family_%lu", (unsigned long)family]];
            [items addObjectsFromArray:@[
                @"GL_APPLE_framebuffer_multisample",
                @"GL_APPLE_rgb_422"
            ]];
            if (family >= 3) {
                [items addObjectsFromArray:@[
                    @"GL_EXT_color_buffer_float",
                    @"GL_EXT_shader_framebuffer_fetch"
                ]];
            }
            if (family >= 4) {
                [items addObjectsFromArray:@[
                    @"GL_EXT_texture_compression_astc_decode_mode",
                    @"GL_KHR_texture_compression_astc_ldr"
                ]];
            }
        }

        cachedExtensions = [items copy];
    }
    return cachedExtensions;
}

NSString* getExtensions() {
    NSArray<NSString *> *extensions = gl4metalGetExtensionList();
    return [extensions componentsJoinedByString:@" "];
}

const GLubyte* APIENTRY glGetString(GLenum name) {
    static const char *vendor = "congcq";
    static const char *version = "3.3.0 Core Profile";
    static const char *slVersion = "3.3";

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
            NSString *extensions = getExtensions();
            return (const GLubyte *)extensions.UTF8String;
        }
        default:
            return NULL;
    }
}

const GLubyte *APIENTRY glGetStringi(GLenum name, GLuint index) {
    if (name == GL_VENDOR) {
        return index == 0 ? (const GLubyte *)"congcq" : NULL;
    }
    if (name == GL_RENDERER) {
        return index == 0 ? (const GLubyte *)(gl4metalGetDeviceName() ?: "gl4metal") : NULL;
    }
    if (name == GL_VERSION) {
        return index == 0 ? (const GLubyte *)"3.3.0 Core Profile" : NULL;
    }
    if (name == GL_SHADING_LANGUAGE_VERSION) {
        return index == 0 ? (const GLubyte *)"3.3.0" : NULL;
    }
    if (name == GL_EXTENSIONS) {
        NSArray<NSString *> *extensions = gl4metalGetExtensionList();
        if (index >= (GLuint)extensions.count) {
            return NULL;
        }
        NSString *extension = extensions[index];
        return (const GLubyte *)extension.UTF8String;
    }
    return NULL;
}