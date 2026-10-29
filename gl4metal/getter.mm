#import "gl4metal.h"
#import <UIKit/UIKit.h>

#include <sys/utsname.h>

// MARK: - State queries

GLenum glGetError(void) {
	GL4METAL_LOG("glGetError: Unimplemented, defaulting return GL_NO_ERROR");
	return GL_NO_ERROR;
}

void glGetBooleanv(GLenum pname, GLboolean *data) {
	GL4METAL_LOG("glGetBooleanv: Unimplemented");
}

void glGetDoublev(GLenum pname, GLdouble *data) {
	GL4METAL_LOG("glGetDoublev: Unimplemented");
}

void glGetFloatv(GLenum pname, GLfloat *data) {
	GL4METAL_LOG("glGetFloatv: Unimplemented");
}

void glGetIntegerv(GLenum pname, GLint *data) {
	switch(pname) {
		case GL_MAJOR_VERSION:
			*data = 3;
			break;
		case GL_MINOR_VERSION:
			*data = 3;
			break;
		case GL_NUM_EXTENSIONS:
			*data = 0; // does not implement extensions yet
			break;
		case GL_CONTEXT_FLAGS:
			*data = GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT;
			break;
		case GL_CONTEXT_PROFILE_MASK:
			*data = GL_CONTEXT_CORE_PROFILE_BIT;
			break;
		case GL_MAX_DRAW_BUFFERS:
		case GL_MAX_COLOR_ATTACHMENTS:
			*data = 8;
			break;
		case GL_MAX_TEXTURE_IMAGE_UNITS:
		case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS:
		case GL_MAX_VERTEX_ATTRIBS:
			*data = 16;
			break;
		case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS:
			*data = 32;
			break;
		case GL_MAX_TEXTURE_SIZE:
		case GL_MAX_CUBE_MAP_TEXTURE_SIZE:
		case GL_MAX_RENDERBUFFER_SIZE:
			*data = 16384;
			break;
		case GL_MAX_3D_TEXTURE_SIZE:
		case GL_MAX_ARRAY_TEXTURE_LAYERS:
			*data = 2048;
			break;
		case GL_MAX_TEXTURE_BUFFER_SIZE:
			*data = 65536;
			break;
		case GL_MAX_SAMPLES:
			*data = 1; // no multisampling yet
			break;
		case GL_MAX_ELEMENTS_VERTICES:
		case GL_MAX_ELEMENTS_INDICES:
			*data = 1048576;
			break;
		case GL_MIN_PROGRAM_TEXEL_OFFSET:
			*data = -8;
			break;
		case GL_MAX_PROGRAM_TEXEL_OFFSET:
			*data = 7;
			break;
		case GL_MAX_VERTEX_UNIFORM_COMPONENTS:
		case GL_MAX_FRAGMENT_UNIFORM_COMPONENTS:
			*data = 4096;
			break;
		case GL_MAX_VARYING_COMPONENTS:
			*data = 60;
			break;

		// Uniform buffers: Metal accepts any offset that is a multiple of 4 on iOS, 256 is what GL drivers report.
		case GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT:
			*data = 256;
			break;
		case GL_MAX_UNIFORM_BLOCK_SIZE:
			*data = 65536;
			break;
		case GL_MAX_UNIFORM_BUFFER_BINDINGS:
			*data = GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS;
			break;
		case GL_MAX_VERTEX_UNIFORM_BLOCKS:
		case GL_MAX_FRAGMENT_UNIFORM_BLOCKS:
			*data = 14; // Metal slots 16...29 are used for blocks, see shadertranslator.cpp
			break;
		case GL_MAX_COMBINED_UNIFORM_BLOCKS:
			*data = 28;
			break;
		case GL_MAX_GEOMETRY_UNIFORM_BLOCKS:
			*data = 0;
			break;

		// Current state.
		case GL_VIEWPORT:
			data[0] = ctx.viewport.x; data[1] = ctx.viewport.y; data[2] = ctx.viewport.w; data[3] = ctx.viewport.h;
			break;
		case GL_SCISSOR_BOX:
			data[0] = ctx.scissor.x; data[1] = ctx.scissor.y; data[2] = ctx.scissor.w; data[3] = ctx.scissor.h;
			break;
		case GL_MAX_VIEWPORT_DIMS:
			data[0] = 16384; data[1] = 16384;
			break;
		case GL_DRAW_FRAMEBUFFER_BINDING: // = GL_FRAMEBUFFER_BINDING
			*data = (GLint)ctx.draw_fbo;
			break;
		case GL_READ_FRAMEBUFFER_BINDING:
			*data = (GLint)ctx.read_fbo;
			break;
		case GL_ARRAY_BUFFER_BINDING:
			*data = (GLint)boundBuffer(GL_ARRAY_BUFFER);
			break;
		case GL_ELEMENT_ARRAY_BUFFER_BINDING:
			*data = (GLint)ctx.vao_objects[ctx.current_vao].element_array_buffer_id;
			break;
		case GL_UNIFORM_BUFFER_BINDING:
			*data = (GLint)boundBuffer(GL_UNIFORM_BUFFER);
			break;
		case GL_PIXEL_UNPACK_BUFFER_BINDING:
			*data = (GLint)boundBuffer(GL_PIXEL_UNPACK_BUFFER);
			break;
		case GL_PIXEL_PACK_BUFFER_BINDING:
			*data = (GLint)boundBuffer(GL_PIXEL_PACK_BUFFER);
			break;
		case GL_VERTEX_ARRAY_BINDING:
			*data = (GLint)ctx.current_vao;
			break;
		case GL_CURRENT_PROGRAM:
			*data = (GLint)ctx.current_program;
			break;
		case GL_ACTIVE_TEXTURE:
			*data = (GLint)(GL_TEXTURE0 + ctx.active_texture_unit);
			break;
		case GL_TEXTURE_BINDING_2D: {
			auto bound = ctx.bound_textures[ctx.active_texture_unit].find(GL_TEXTURE_2D);
			*data = bound == ctx.bound_textures[ctx.active_texture_unit].end() ? 0 : (GLint)bound->second;
			break;
		}
		case GL_SAMPLER_BINDING:
			*data = (GLint)ctx.bound_samplers[ctx.active_texture_unit];
			break;
		case GL_UNPACK_ALIGNMENT:   *data = ctx.unpack_alignment; break;
		case GL_UNPACK_ROW_LENGTH:  *data = ctx.unpack_row_length; break;
		case GL_UNPACK_SKIP_PIXELS: *data = ctx.unpack_skip_pixels; break;
		case GL_UNPACK_SKIP_ROWS:   *data = ctx.unpack_skip_rows; break;
		default:
			GL4METAL_LOG("glGetIntegerv: missing 0x%X", pname);
			break;
	}
}

void glGetInteger64v(GLenum pname, GLint64 *data) {
	GL4METAL_LOG("glGetInteger64v: Unimplemented");
}

void glGetBooleani_v(GLenum target, GLuint index, GLboolean *data) {
	GL4METAL_LOG("glGetBooleani_v: Unimplemented");
}

void glGetIntegeri_v(GLenum target, GLuint index, GLint *data) {
	GL4METAL_LOG("glGetIntegeri_v: Unimplemented");
}

void glGetInteger64i_v(GLenum target, GLuint index, GLint64 *data) {
	GL4METAL_LOG("glGetInteger64i_v: Unimplemented");
}

void glGetMultisamplefv(GLenum pname, GLuint index, GLfloat *val) {
	GL4METAL_LOG("glGetMultisamplefv: Unimplemented");
}

void glGetPointerv(GLenum pname, void **params) {
    GL4METAL_LOG("glGetPointerv: Unimplemented");
}

NSString *detectDeviceName(utsname si) {
	NSString *id = [NSString stringWithUTF8String:si.machine];
	static NSDictionary *name;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		name = @{
			// iPhone
			@"iPhone8,1"  : @"iPhone 6s",
            @"iPhone8,2"  : @"iPhone 6s Plus",
            @"iPhone8,4"  : @"iPhone SE",
            @"iPhone9,1"  : @"iPhone 7",
            @"iPhone9,3"  : @"iPhone 7",
            @"iPhone9,2"  : @"iPhone 7 Plus",
            @"iPhone9,4"  : @"iPhone 7 Plus",
            @"iPhone10,1" : @"iPhone 8",
            @"iPhone10,4" : @"iPhone 8",
            @"iPhone10,2" : @"iPhone 8 Plus",
            @"iPhone10,5" : @"iPhone 8 Plus",
            @"iPhone10,3" : @"iPhone X",
            @"iPhone10,6" : @"iPhone X",
            @"iPhone11,2" : @"iPhone XS",
            @"iPhone11,4" : @"iPhone XS Max",
            @"iPhone11,6" : @"iPhone XS Max",
            @"iPhone11,8" : @"iPhone XR",
            @"iPhone12,1" : @"iPhone 11",
            @"iPhone12,3" : @"iPhone 11 Pro",
            @"iPhone12,5" : @"iPhone 11 Pro Max",
            @"iPhone12,8" : @"iPhone SE2",
            @"iPhone13,1" : @"iPhone 12 mini",
            @"iPhone13,2" : @"iPhone 12",
            @"iPhone13,3" : @"iPhone 12 Pro",
            @"iPhone13,4" : @"iPhone 12 Pro Max",
            @"iPhone14,4" : @"iPhone 13 mini",
            @"iPhone14,5" : @"iPhone 13",
            @"iPhone14,2" : @"iPhone 13 Pro",
            @"iPhone14,3" : @"iPhone 13 Pro Max",
            @"iPhone14,6" : @"iPhone SE3",
            @"iPhone14,7" : @"iPhone 14",
            @"iPhone14,8" : @"iPhone 14 Plus",
            @"iPhone15,2" : @"iPhone 14 Pro",
            @"iPhone15,3" : @"iPhone 14 Pro Max",
            @"iPhone15,4" : @"iPhone 15",
            @"iPhone15,5" : @"iPhone 15 Plus",
            @"iPhone16,1" : @"iPhone 15 Pro",
            @"iPhone16,2" : @"iPhone 15 Pro Max",
            @"iPhone17,3" : @"iPhone 16",
            @"iPhone17,4" : @"iPhone 16 Plus",
            @"iPhone17,1" : @"iPhone 16 Pro",
            @"iPhone17,2" : @"iPhone 16 Pro Max",
            @"iPhone17,5" : @"iPhone 16e",
            @"iPhone18,3" : @"iPhone 17",
            @"iPhone18,1" : @"iPhone 17 Pro",
            @"iPhone18,2" : @"iPhone 17 Pro Max",
            @"iPhone18,4" : @"iPhone Air",
			@"iPhone18,5" : @"iPhone 17e",
			@"iPhone19,2" : @"iPhone 18 Pro",
			@"iPhone19,3" : @"iPhone 18 Pro Max",
			@"iPhone19,7" : @"iPhone 18 Pro Max",

            // iPad
            @"iPad6,11"  : @"iPad 5th gen",
            @"iPad6,12"  : @"iPad 5th gen",
            @"iPad7,5"   : @"iPad 6th gen",
            @"iPad7,6"   : @"iPad 6th gen",
            @"iPad7,11"  : @"iPad 7th gen",
            @"iPad7,12"  : @"iPad 7th gen",
            @"iPad11,6"  : @"iPad (8th gen, Wi-Fi)",
            @"iPad11,7"  : @"iPad (8th gen, Cellular)",
            @"iPad12,1"  : @"iPad (9th gen, Wi-Fi)",
            @"iPad12,2"  : @"iPad (9th gen, Cellular)",
            @"iPad13,18" : @"iPad (10th gen, Wi-Fi)",
            @"iPad13,19" : @"iPad (10th gen, Cellular)",
            @"iPad15,7"  : @"iPad (A16, Wi-Fi)",
            @"iPad15,8"  : @"iPad (A16, Cellular)",

            // iPad mini
            @"iPad5,1"   : @"iPad mini 4 (Wi-Fi)",
            @"iPad5,2"   : @"iPad mini 4 (Cellular)",
            @"iPad11,1"  : @"iPad mini (5th gen, Wi-Fi)",
            @"iPad11,2"  : @"iPad mini (5th gen, Cellular)",
            @"iPad14,1"  : @"iPad mini (6th gen, Wi-Fi)",
            @"iPad14,2"  : @"iPad mini (6th gen, Cellular)",
            @"iPad16,1"  : @"iPad mini (A17 Pro, Wi-Fi)",
            @"iPad16,2"  : @"iPad mini (A17 Pro, Cellular)",

            // iPad Air
            @"iPad5,3"   : @"iPad Air 2 (Wi-Fi)",
            @"iPad5,4"   : @"iPad Air 2 (Cellular)",
            @"iPad11,3"  : @"iPad Air (3rd gen, Wi-Fi)",
            @"iPad11,4"  : @"iPad Air (3rd gen, Cellular)",
            @"iPad13,1"  : @"iPad Air (4th gen, Wi-Fi)",
            @"iPad13,2"  : @"iPad Air (4th gen, Cellular)",
            @"iPad13,16" : @"iPad Air (5th gen, Wi-Fi)",
            @"iPad13,17" : @"iPad Air (5th gen, Cellular)",
            @"iPad14,8"  : @"iPad Air 11-inch (M2, Wi-Fi)",
            @"iPad14,9"  : @"iPad Air 11-inch (M2, Cellular)",
            @"iPad14,10" : @"iPad Air 13-inch (M2, Wi-Fi)",
            @"iPad14,11" : @"iPad Air 13-inch (M2, Cellular)",

            // iPad Pro
            @"iPad6,3"   : @"iPad Pro 9.7-inch (Wi-Fi)",
            @"iPad6,4"   : @"iPad Pro 9.7-inch (Cellular)",
            @"iPad7,3"   : @"iPad Pro 10.5-inch (Wi-Fi)",
            @"iPad7,4"   : @"iPad Pro 10.5-inch (Cellular)",
            @"iPad8,1"   : @"iPad Pro 11-inch (1st gen, Wi-Fi)",
            @"iPad8,2"   : @"iPad Pro 11-inch (1st gen, Wi-Fi)",
            @"iPad8,3"   : @"iPad Pro 11-inch (1st gen, Cellular)",
            @"iPad8,4"   : @"iPad Pro 11-inch (1st gen, Cellular)",
            @"iPad8,9"   : @"iPad Pro 11-inch (2nd gen, Wi-Fi)",
            @"iPad8,10"  : @"iPad Pro 11-inch (2nd gen, Cellular)",
            @"iPad13,4"  : @"iPad Pro 11-inch (3rd gen, Wi-Fi)",
            @"iPad13,5"  : @"iPad Pro 11-inch (3rd gen, Wi-Fi)",
            @"iPad13,6"  : @"iPad Pro 11-inch (3rd gen, Cellular)",
            @"iPad13,7"  : @"iPad Pro 11-inch (3rd gen, Cellular)",
            @"iPad14,3"  : @"iPad Pro 11-inch (4th gen, Wi-Fi)",
            @"iPad14,4"  : @"iPad Pro 11-inch (4th gen, Cellular)",
            @"iPad16,3"  : @"iPad Pro 11-inch (M4, Wi-Fi)",
            @"iPad16,4"  : @"iPad Pro 11-inch (M4, Cellular)",
            @"iPad6,7"   : @"iPad Pro 12.9-inch (1st gen, Wi-Fi)",
            @"iPad6,8"   : @"iPad Pro 12.9-inch (1st gen, Cellular)",
            @"iPad7,1"   : @"iPad Pro 12.9-inch (2nd gen, Wi-Fi)",
            @"iPad7,2"   : @"iPad Pro 12.9-inch (2nd gen, Cellular)",
            @"iPad8,5"   : @"iPad Pro 12.9-inch (3rd gen, Wi-Fi)",
            @"iPad8,6"   : @"iPad Pro 12.9-inch (3rd gen, Wi-Fi)",
            @"iPad8,7"   : @"iPad Pro 12.9-inch (3rd gen, Cellular)",
            @"iPad8,8"   : @"iPad Pro 12.9-inch (3rd gen, Cellular)",
            @"iPad8,11"  : @"iPad Pro 12.9-inch (4th gen, Wi-Fi)",
            @"iPad8,12"  : @"iPad Pro 12.9-inch (4th gen, Cellular)",
            @"iPad13,8"  : @"iPad Pro 12.9-inch (5th gen, Wi-Fi)",
            @"iPad13,9"  : @"iPad Pro 12.9-inch (5th gen, Wi-Fi)",
            @"iPad13,10" : @"iPad Pro 12.9-inch (5th gen, Cellular)",
            @"iPad13,11" : @"iPad Pro 12.9-inch (5th gen, Cellular)",
            @"iPad14,5"  : @"iPad Pro 12.9-inch (6th gen, Wi-Fi)",
            @"iPad14,6"  : @"iPad Pro 12.9-inch (6th gen, Cellular)",
            @"iPad16,5"  : @"iPad Pro 13-inch (M4, Wi-Fi)",
            @"iPad16,6"  : @"iPad Pro 13-inch (M4, Cellular)"
		};
	});
	return name[id] ?: id;
}

const GLubyte *glGetString(GLenum name) {
	switch(name) {
		case GL_VENDOR:
			return (const GLubyte *)"congcq";
		case GL_RENDERER: {
			struct utsname si; uname(&si);
			UIDevice *d = UIDevice.currentDevice;
			NSString *s = [NSString stringWithFormat:@"GL4Metal beta (%@ | %@ | iOS %@)", detectDeviceName(si), ctx.device.name, d.systemVersion];
			return (const GLubyte *)s.UTF8String;
		}
		case GL_VERSION:
			return (const GLubyte *)"3.3 Core Profile (Metal backend)";
		case GL_SHADING_LANGUAGE_VERSION:
			return (const GLubyte *)"3.3.0";
		case GL_EXTENSIONS:
			return (const GLubyte *)""; // does not implement extensions yet
		default:
			GL4METAL_LOG("glGetString: missing 0x%X", name);
			return NULL;
	}
}

const GLubyte *glGetStringi(GLenum name, GLuint index) {
	switch(name) {
		case GL_EXTENSIONS:
			return (const GLubyte *)"";
		default:
			GL4METAL_LOG("glGetStringi: unhandled 0x%X", name);
			return NULL;
	}
}
