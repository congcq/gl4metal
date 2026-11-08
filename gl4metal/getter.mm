#import "gl4metal.h"
#import <UIKit/UIKit.h>

#include <sys/utsname.h>

// Apple2 devices support 8192; Apple3 and
// later Apple GPU families support at least 16384. GL4Metal intentionally
// caps its advertised 2D limit at GL4METAL_MAX_TEXTURE_SIZE.
static GLint maxMetalTextureDimension2D(id<MTLDevice> metalDevice) {
	if (!metalDevice) {
		return GL4METAL_MAX_TEXTURE_SIZE;
	}

	NSUInteger deviceLimit = 8192;
	if (@available(iOS 13.0, *)) {
		if ([metalDevice supportsFamily:MTLGPUFamilyApple3]) {
			deviceLimit = 16384;
		}
	}

	return (GLint)std::min<NSUInteger>(
		(NSUInteger)GL4METAL_MAX_TEXTURE_SIZE, deviceLimit);
}

// MARK: - State queries

// GL keeps the first error until glGetError reads (and clears) it.
static GLenum sError = GL_NO_ERROR;

void gl4metalSetError(GLenum error) {
	if (sError == GL_NO_ERROR) {
		sError = error;
	}
}

GLenum glGetError(void) {
	GLenum error = sError;
	sError = GL_NO_ERROR;
	return error;
}

void glGetBooleanv(GLenum pname, GLboolean *data) {
	switch (pname) {
		case GL_BLEND:
		case GL_CULL_FACE:
		case GL_DEPTH_TEST:
		case GL_SCISSOR_TEST:
		case GL_POLYGON_OFFSET_FILL:
		case GL_STENCIL_TEST:
		case GL_DITHER:
			*data = glIsEnabled(pname);
			break;
		case GL_DEPTH_WRITEMASK:
			*data = ctx.depth_write_enabled;
			break;
		case GL_COLOR_WRITEMASK:
			for (int i = 0; i < 4; i++) {
				data[i] = ctx.color_mask[i];
			}
			break;
		default: {
			GLint value = 0;
			glGetIntegerv(pname, &value);
			*data = value != 0;
			break;
		}
	}
}

void glGetDoublev(GLenum pname, GLdouble *data) {
	GLfloat values[4] = { 0, 0, 0, 0 };
	glGetFloatv(pname, values);
	int count = (pname == GL_COLOR_CLEAR_VALUE || pname == GL_BLEND_COLOR || pname == GL_VIEWPORT || pname == GL_SCISSOR_BOX) ? 4 :
				((pname == GL_DEPTH_RANGE || pname == GL_MAX_VIEWPORT_DIMS || pname == GL_ALIASED_LINE_WIDTH_RANGE || pname == GL_POINT_SIZE_RANGE) ? 2 : 1);
	for (int i = 0; i < count; i++) {
		data[i] = values[i];
	}
}

void glGetFloatv(GLenum pname, GLfloat *data) {
	switch (pname) {
		case GL_COLOR_CLEAR_VALUE:
			data[0] = (GLfloat)ctx.clear_color.red; data[1] = (GLfloat)ctx.clear_color.green;
			data[2] = (GLfloat)ctx.clear_color.blue; data[3] = (GLfloat)ctx.clear_color.alpha;
			break;
		case GL_DEPTH_CLEAR_VALUE:
			*data = (GLfloat)ctx.clear_depth;
			break;
		case GL_DEPTH_RANGE:
			data[0] = ctx.depth_range_near; data[1] = ctx.depth_range_far;
			break;
		case GL_BLEND_COLOR:
			for (int i = 0; i < 4; i++) {
				data[i] = ctx.blend_color[i];
			}
			break;
		case GL_POLYGON_OFFSET_FACTOR:
			*data = ctx.polygon_offset_factor;
			break;
		case GL_POLYGON_OFFSET_UNITS:
			*data = ctx.polygon_offset_units;
			break;
		case GL_LINE_WIDTH:
			*data = 1.0f; // Metal has no line width
			break;
		case GL_ALIASED_LINE_WIDTH_RANGE:
		case GL_SMOOTH_LINE_WIDTH_RANGE:
			data[0] = 1.0f; data[1] = 1.0f;
			break;
		case GL_POINT_SIZE_RANGE:
			data[0] = 1.0f; data[1] = 1.0f;
			break;
		case GL_MAX_TEXTURE_LOD_BIAS:
			*data = 15.0f;
			break;
		case 0x84FF: // GL_MAX_TEXTURE_MAX_ANISOTROPY
			*data = 16.0f;
			break;
		default: {
			GLint values[4] = { 0, 0, 0, 0 };
			glGetIntegerv(pname, values);
			int count = (pname == GL_VIEWPORT || pname == GL_SCISSOR_BOX) ? 4 : (pname == GL_MAX_VIEWPORT_DIMS ? 2 : 1);
			for (int i = 0; i < count; i++) {
				data[i] = (GLfloat)values[i];
			}
			break;
		}
	}
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
			*data = 1; // GL_ARB_draw_buffers_blend
			break;
		case GL_CONTEXT_FLAGS:
			*data = GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT;
			break;
		case GL_CONTEXT_PROFILE_MASK:
			*data = GL_CONTEXT_CORE_PROFILE_BIT;
			break;
		case GL_MAX_DRAW_BUFFERS:
		case GL_MAX_COLOR_ATTACHMENTS:
			*data = GL4METAL_MAX_COLOR_ATTACHMENTS;
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
			*data = maxMetalTextureDimension2D(ctx.device);
			break;
		case GL_MAX_3D_TEXTURE_SIZE:
			*data = 0; // GL_TEXTURE_3D storage/upload is not implemented yet.
			break;
		case GL_MAX_ARRAY_TEXTURE_LAYERS:
			*data = 0; // GL_TEXTURE_2D_ARRAY storage/upload is not implemented yet.
			break;
		case GL_MAX_TEXTURE_BUFFER_SIZE:
			*data = 0; // glTexBuffer is not implemented yet.
			break;
		case GL_MAX_SAMPLES: {
			GLint maxSamples = 1;
			for (GLint candidate = 2; candidate <= 16; candidate *= 2) {
				if ([ctx.device supportsTextureSampleCount:(NSUInteger)candidate]) maxSamples = candidate;
			}
			*data = maxSamples;
			break;
		}
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
		case GL_MAX_VIEWPORT_DIMS: {
			GLint maxDim = maxMetalTextureDimension2D(ctx.device);
			data[0] = maxDim; data[1] = maxDim;
			break;
		}
		case GL_DRAW_FRAMEBUFFER_BINDING: // = GL_FRAMEBUFFER_BINDING
			*data = (GLint)ctx.draw_fbo;
			break;
		case GL_READ_FRAMEBUFFER_BINDING:
			*data = (GLint)ctx.read_fbo;
			break;
		case GL_RENDERBUFFER_BINDING:
			*data = (GLint)ctx.bound_renderbuffer;
			break;
		case GL_STENCIL_FUNC: *data = (GLint)ctx.stencil_front.func; break;
		case GL_STENCIL_REF: *data = ctx.stencil_front.ref; break;
		case GL_STENCIL_VALUE_MASK: *data = (GLint)ctx.stencil_front.value_mask; break;
		case GL_STENCIL_WRITEMASK: *data = (GLint)ctx.stencil_front.write_mask; break;
		case GL_STENCIL_FAIL: *data = (GLint)ctx.stencil_front.fail; break;
		case GL_STENCIL_PASS_DEPTH_FAIL: *data = (GLint)ctx.stencil_front.depth_fail; break;
		case GL_STENCIL_PASS_DEPTH_PASS: *data = (GLint)ctx.stencil_front.depth_pass; break;
		case GL_STENCIL_BACK_FUNC: *data = (GLint)ctx.stencil_back.func; break;
		case GL_STENCIL_BACK_REF: *data = ctx.stencil_back.ref; break;
		case GL_STENCIL_BACK_VALUE_MASK: *data = (GLint)ctx.stencil_back.value_mask; break;
		case GL_STENCIL_BACK_WRITEMASK: *data = (GLint)ctx.stencil_back.write_mask; break;
		case GL_STENCIL_BACK_FAIL: *data = (GLint)ctx.stencil_back.fail; break;
		case GL_STENCIL_BACK_PASS_DEPTH_FAIL: *data = (GLint)ctx.stencil_back.depth_fail; break;
		case GL_STENCIL_BACK_PASS_DEPTH_PASS: *data = (GLint)ctx.stencil_back.depth_pass; break;
		case GL_PRIMITIVE_RESTART_INDEX: *data = (GLint)ctx.primitive_restart_index; break;
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
		case GL_BLEND_SRC_RGB:      *data = (GLint)ctx.blend_buffers[0].src_rgb; break;
		case GL_BLEND_DST_RGB:      *data = (GLint)ctx.blend_buffers[0].dst_rgb; break;
		case GL_BLEND_SRC_ALPHA:    *data = (GLint)ctx.blend_buffers[0].src_alpha; break;
		case GL_BLEND_DST_ALPHA:    *data = (GLint)ctx.blend_buffers[0].dst_alpha; break;
		case GL_BLEND_EQUATION_RGB: *data = (GLint)ctx.blend_buffers[0].equation_rgb; break;
		case GL_BLEND_EQUATION_ALPHA: *data = (GLint)ctx.blend_buffers[0].equation_alpha; break;
		case GL_UNPACK_ALIGNMENT:   *data = ctx.unpack_alignment; break;
		case GL_UNPACK_ROW_LENGTH:  *data = ctx.unpack_row_length; break;
		case GL_UNPACK_SKIP_PIXELS: *data = ctx.unpack_skip_pixels; break;
		case GL_UNPACK_SKIP_ROWS:   *data = ctx.unpack_skip_rows; break;
		case GL_PACK_ALIGNMENT:    *data = ctx.pack_alignment; break;
		case GL_PACK_ROW_LENGTH:   *data = ctx.pack_row_length; break;
		case GL_PACK_SKIP_PIXELS:  *data = ctx.pack_skip_pixels; break;
		case GL_PACK_SKIP_ROWS:    *data = ctx.pack_skip_rows; break;
		case GL_PACK_SWAP_BYTES:   *data = ctx.pack_swap_bytes; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetIntegerv: missing 0x%X", pname);
			break;
	}
}

void glGetInteger64v(GLenum pname, GLint64 *data) {
	GLint values[4] = { 0, 0, 0, 0 };
	glGetIntegerv(pname, values);
	int count = (pname == GL_VIEWPORT || pname == GL_SCISSOR_BOX) ? 4 : (pname == GL_MAX_VIEWPORT_DIMS ? 2 : 1);
	for (int i = 0; i < count; i++) {
		data[i] = values[i];
	}
}

void glGetBooleani_v(GLenum target, GLuint index, GLboolean *data) {
	if (!data) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (target == GL_BLEND) {
		if (index >= GL4METAL_MAX_COLOR_ATTACHMENTS) { gl4metalSetError(GL_INVALID_VALUE); return; }
		*data = ctx.blend_enabled[index];
		return;
	}
	if (target == GL_SCISSOR_TEST) {
		if (index != 0) { gl4metalSetError(GL_INVALID_VALUE); return; }
		*data = ctx.scissor_test_enabled;
		return;
	}
	if (target == GL_COLOR_WRITEMASK) {
		if (index >= GL4METAL_MAX_COLOR_ATTACHMENTS) { gl4metalSetError(GL_INVALID_VALUE); return; }
		for (int component = 0; component < 4; ++component)
			data[component] = ctx.color_masks[index][component];
		return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
	GL4METAL_LOG("glGetBooleani_v: unhandled target 0x%X", target);
}

void glGetIntegeri_v(GLenum target, GLuint index, GLint *data) {
	if (!data) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (target == GL_BLEND_SRC_RGB || target == GL_BLEND_DST_RGB || target == GL_BLEND_SRC_ALPHA ||
		target == GL_BLEND_DST_ALPHA || target == GL_BLEND_EQUATION_RGB || target == GL_BLEND_EQUATION_ALPHA) {
		if (index >= GL4METAL_MAX_COLOR_ATTACHMENTS) { gl4metalSetError(GL_INVALID_VALUE); return; }
		const BlendBufferState &blendState = ctx.blend_buffers[index];
		switch (target) {
			case GL_BLEND_SRC_RGB: *data = (GLint)blendState.src_rgb; break;
			case GL_BLEND_DST_RGB: *data = (GLint)blendState.dst_rgb; break;
			case GL_BLEND_SRC_ALPHA: *data = (GLint)blendState.src_alpha; break;
			case GL_BLEND_DST_ALPHA: *data = (GLint)blendState.dst_alpha; break;
			case GL_BLEND_EQUATION_RGB: *data = (GLint)blendState.equation_rgb; break;
			case GL_BLEND_EQUATION_ALPHA: *data = (GLint)blendState.equation_alpha; break;
		}
		return;
	}
	if (index >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS && (target == GL_UNIFORM_BUFFER_BINDING || target == GL_UNIFORM_BUFFER_START || target == GL_UNIFORM_BUFFER_SIZE)) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetIntegeri_v: binding point %u is out of range (GL_INVALID_VALUE)", index);
		return;
	}
	switch (target) {
		case GL_UNIFORM_BUFFER_BINDING: *data = (GLint)ctx.uniform_buffer_bindings[index].buffer; break;
		case GL_UNIFORM_BUFFER_START:   *data = (GLint)ctx.uniform_buffer_bindings[index].offset; break;
		case GL_UNIFORM_BUFFER_SIZE: {
			const UniformBufferBinding &binding = ctx.uniform_buffer_bindings[index];
			if (binding.size != 0) {
				*data = (GLint)binding.size;
			} else {
				auto buffer = ctx.buffer_objects.find(binding.buffer);
				*data = (binding.buffer != 0 && buffer != ctx.buffer_objects.end() && buffer->second)
					? (GLint)buffer->second.length : 0;
			}
			break;
		}
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetIntegeri_v: unhandled target 0x%X", target);
			break;
	}
}

void glGetInteger64i_v(GLenum target, GLuint index, GLint64 *data) {
	GLint value = 0;
	glGetIntegeri_v(target, index, &value);
	*data = value;
}

static NSUInteger currentDrawFramebufferSampleCount() {
	if (ctx.draw_fbo == 0) return 1;
	auto fbo = ctx.fbo_objects.find(ctx.draw_fbo);
	if (fbo == ctx.fbo_objects.end()) return 1;
	auto sampleCountForAttachment = [](const FBOAttachment &attachment) -> NSUInteger {
		if (attachment.renderbuffer != 0) {
			auto renderbuffer = ctx.renderbuffer_objects.find(attachment.renderbuffer);
			return renderbuffer != ctx.renderbuffer_objects.end() && renderbuffer->second.texture
				? renderbuffer->second.texture.sampleCount : 1;
		}
		if (attachment.texture != 0) {
			auto texture = ctx.texture_objects.find(attachment.texture);
			return texture != ctx.texture_objects.end() && texture->second.texture ? texture->second.texture.sampleCount : 1;
		}
		return 1;
	};
	for (const FBOAttachment &attachment : fbo->second.color) {
		if (attachment.texture != 0 || attachment.renderbuffer != 0) return sampleCountForAttachment(attachment);
	}
	if (fbo->second.depth.texture || fbo->second.depth.renderbuffer) return sampleCountForAttachment(fbo->second.depth);
	if (fbo->second.stencil.texture || fbo->second.stencil.renderbuffer) return sampleCountForAttachment(fbo->second.stencil);
	return 1;
}

void glGetMultisamplefv(GLenum pname, GLuint index, GLfloat *val) {
	if (!val) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname != GL_SAMPLE_POSITION) { gl4metalSetError(GL_INVALID_ENUM); return; }
	const NSUInteger sampleCount = currentDrawFramebufferSampleCount();
	if ((NSUInteger)index >= sampleCount) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (sampleCount == 1) { val[0] = 0.5f; val[1] = 0.5f; return; }
	// Metal only exposes the default sample pattern on devices with programmable sample positions.
	// On fixed-pattern GPUs the exact locations are implementation-defined and cannot be queried.
	if (!ctx.device || !ctx.device.programmableSamplePositionsSupported) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetMultisamplefv: this iOS GPU has fixed sample locations that Metal does not expose");
		return;
	}
	MTLSamplePosition positions[16] = {};
	if (sampleCount > 16) { gl4metalSetError(GL_INVALID_OPERATION); return; }
	[ctx.device getDefaultSamplePositions:positions count:sampleCount];
	val[0] = positions[index].x;
	val[1] = 1.0f - positions[index].y; // Metal's origin is top-left; GL's window origin is bottom-left.
}

void glGetPointerv(GLenum pname, void **params) {
	*params = NULL; // no debug callback, no client-side arrays
}

NSString *detectDeviceName(utsname si) {
	NSString *device_id = [NSString stringWithUTF8String:si.machine];
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
            @"iPad11,6"  : @"iPad 8th gen",
            @"iPad11,7"  : @"iPad 8th gen",
            @"iPad12,1"  : @"iPad 9th gen",
            @"iPad12,2"  : @"iPad 9th gen",
            @"iPad13,18" : @"iPad 10th gen",
            @"iPad13,19" : @"iPad 10th gen",
            @"iPad15,7"  : @"iPad A16",
            @"iPad15,8"  : @"iPad A16",

            // iPad mini
            @"iPad5,1"   : @"iPad mini 4",
            @"iPad5,2"   : @"iPad mini 4",
            @"iPad11,1"  : @"iPad mini 5th gen",
            @"iPad11,2"  : @"iPad mini 5th gen",
            @"iPad14,1"  : @"iPad mini 6th gen",
            @"iPad14,2"  : @"iPad mini 6th gen",
            @"iPad16,1"  : @"iPad mini A17 Pro",
            @"iPad16,2"  : @"iPad mini A17 Pro",

            // iPad Air
            @"iPad5,3"   : @"iPad Air 2",
            @"iPad5,4"   : @"iPad Air 2",
            @"iPad11,3"  : @"iPad Air 3rd gen",
            @"iPad11,4"  : @"iPad Air 3rd gen",
            @"iPad13,1"  : @"iPad Air 4th gen",
            @"iPad13,2"  : @"iPad Air 4th gen",
            @"iPad13,16" : @"iPad Air 5th gen",
            @"iPad13,17" : @"iPad Air 5th gen",
            @"iPad14,8"  : @"iPad Air M2",
            @"iPad14,9"  : @"iPad Air M2",
            @"iPad14,10" : @"iPad Air M2",
            @"iPad14,11" : @"iPad Air M2",

            // iPad Pro
            @"iPad6,3"   : @"iPad Pro",
            @"iPad6,4"   : @"iPad Pro",
            @"iPad7,3"   : @"iPad Pro",
            @"iPad7,4"   : @"iPad Pro",
            @"iPad8,1"   : @"iPad Pro 1st gen",
            @"iPad8,2"   : @"iPad Pro 1st gen",
            @"iPad8,3"   : @"iPad Pro 1st gen",
            @"iPad8,4"   : @"iPad Pro 1st gen",
            @"iPad8,9"   : @"iPad Pro 2nd gen",
            @"iPad8,10"  : @"iPad Pro 2nd gen",
            @"iPad13,4"  : @"iPad Pro 3rd gen",
            @"iPad13,5"  : @"iPad Pro 3rd gen",
            @"iPad13,6"  : @"iPad Pro 3rd gen",
            @"iPad13,7"  : @"iPad Pro 3rd gen",
            @"iPad14,3"  : @"iPad Pro 4th gen",
            @"iPad14,4"  : @"iPad Pro 4th gen",
            @"iPad16,3"  : @"iPad Pro M4",
            @"iPad16,4"  : @"iPad Pro M4",
            @"iPad6,7"   : @"iPad Pro 1st gen",
            @"iPad6,8"   : @"iPad Pro 1st gen",
            @"iPad7,1"   : @"iPad Pro 2nd gen",
            @"iPad7,2"   : @"iPad Pro 2nd gen",
            @"iPad8,5"   : @"iPad Pro 3rd gen",
            @"iPad8,6"   : @"iPad Pro 3rd gen",
            @"iPad8,7"   : @"iPad Pro 3rd gen",
            @"iPad8,8"   : @"iPad Pro 3rd gen",
            @"iPad8,11"  : @"iPad Pro 4th gen",
            @"iPad8,12"  : @"iPad Pro 4th gen",
            @"iPad13,8"  : @"iPad Pro 5th gen",
            @"iPad13,9"  : @"iPad Pro 5th gen",
            @"iPad13,10" : @"iPad Pro 5th gen",
            @"iPad13,11" : @"iPad Pro 5th gen",
            @"iPad14,5"  : @"iPad Pro 6th gen",
            @"iPad14,6"  : @"iPad Pro 6th gen",
            @"iPad16,5"  : @"iPad Pro M4",
            @"iPad16,6"  : @"iPad Pro M4"
		};
	});
	return name[device_id] ?: device_id;
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
			return (const GLubyte *)"3.3 MetalGL";
		case GL_SHADING_LANGUAGE_VERSION:
			return (const GLubyte *)"3.3.0";
		case GL_EXTENSIONS:
			return (const GLubyte *)"GL_ARB_draw_buffers_blend";
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetString: missing 0x%X", name);
			return NULL;
	}
}

const GLubyte *glGetStringi(GLenum name, GLuint index) {
	if (name != GL_EXTENSIONS) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glGetStringi: unhandled 0x%X", name);
		return NULL;
	}
	if (index != 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		return NULL;
	}
	return (const GLubyte *)"GL_ARB_draw_buffers_blend";
}
