#pragma once

#import <Metal/Metal.h>
// #import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#import "framebuffer.h"
#import "texture.h"
#import "shaders.h"
#import "sync.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

// Some older platform GL headers expose the ARB-suffixed token only, even
// though GL_TEXTURE_BUFFER_FORMAT is part of the OpenGL 3.1 texture-buffer API.
// Keep the unsuffixed core spelling available to the backend on every build.

#define GL_TEXTURE_BUFFER_FORMAT GL_TEXTURE_BUFFER_FORMAT_ARB

#include "shadertranslator.h"

#define GL4METAL_LOG(format, ...) printf("[GL4Metal] " format "\n", ##__VA_ARGS__)

// iOS limits (Apple GPU family 3+ / A9 and newer)
#define GL4METAL_MAX_TEXTURE_SIZE 16384
#define GL4METAL_MAX_COLOR_ATTACHMENTS 8
#define GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS 36
// Metal reserves buffer slots 16...30 for UBOs/default uniforms; compute SSBOs use 0...15.
#define GL4METAL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 512

// MARK: - Textures

struct TextureObject {
	GLenum target = 0; // 0 until the name is bound for the first time
	id<MTLTexture> texture = nil;
	GLenum internalformat = GL_RGBA8;
	GLsizei width = 0, height = 0; // level 0 size
	GLuint buffer_object = 0; // GL_TEXTURE_BUFFER backing store name (0 for ordinary textures)
	GLenum buffer_sampler_type = 0; // GL_SAMPLER_BUFFER / GL_INT_SAMPLER_BUFFER / GL_UNSIGNED_INT_SAMPLER_BUFFER

	// sampler parameters (consumed when the texture is bound to a shader)
	GLenum min_filter = GL_NEAREST_MIPMAP_LINEAR;
	GLenum mag_filter = GL_LINEAR;
	GLenum wrap_s = GL_REPEAT, wrap_t = GL_REPEAT, wrap_r = GL_REPEAT;
	GLfloat min_lod = -1000.0f, max_lod = 1000.0f, lod_bias = 0.0f;
	GLint base_level = 0, max_level = 1000;
	GLfloat max_anisotropy = 1.0f;
	GLenum compare_mode = GL_NONE, compare_func = GL_LEQUAL;

	// Lazily allocated read-only copy used when a shader samples this same image
	// while it is selected as an active framebuffer attachment. Tied to the GL
	// texture object's lifetime; recreated when the source dimensions/format change.
	id<MTLTexture> feedback_snapshot = nil;
	uint64_t feedback_snapshot_pass_serial = UINT64_MAX;
};

// MARK: - Framebuffers

struct FBOAttachment {
	GLuint texture = 0;       // GL_TEXTURE object, 0 when not a texture attachment
	GLuint renderbuffer = 0;  // GL_RENDERBUFFER object, 0 when not a renderbuffer attachment
	GLint level = 0;
};

struct RenderbufferObject {
	id<MTLTexture> texture = nil;
	GLenum internalformat = 0;
	GLsizei width = 0;
	GLsizei height = 0;
	GLsizei samples = 0;
	bool initialized = false;
};

struct FBO {
	FBOAttachment color[GL4METAL_MAX_COLOR_ATTACHMENTS];
	FBOAttachment depth, stencil;
	GLenum draw_buffers[GL4METAL_MAX_COLOR_ATTACHMENTS] = { GL_COLOR_ATTACHMENT0 };
	GLenum read_buffer = GL_COLOR_ATTACHMENT0;
};

struct VAO {
	
};

// MARK: - Vertex arrays

struct VertexAttrib {
	bool enabled;
	GLuint bound_vbo;
	GLint size;
	GLenum type;
	GLboolean normalized;
	GLsizei stride;
	const void *pointer;
	bool integer; // glVertexAttribIPointer
	GLuint divisor;
};

struct VertexArray {
	VertexAttrib attribs[16];
	GLuint element_array_buffer_id;
};

// MARK: - Shaders

struct ShaderObject {
	GLenum type = 0;
	std::string source;
	std::string info_log;
	GLboolean compiled = GL_FALSE;
	GLboolean deleted = GL_FALSE; // glDeleteShader on a shader that is still attached to a program
};

// glGenSamplers objects: they override the sampling parameters of the texture bound to the same unit.
struct SamplerObject {
	GLenum min_filter = GL_NEAREST_MIPMAP_LINEAR;
	GLenum mag_filter = GL_LINEAR;
	GLenum wrap_s = GL_REPEAT, wrap_t = GL_REPEAT, wrap_r = GL_REPEAT;
	GLfloat min_lod = -1000.0f, max_lod = 1000.0f, lod_bias = 0.0f;
	GLfloat max_anisotropy = 1.0f;
	GLenum compare_mode = GL_NONE, compare_func = GL_LEQUAL;
};

// glBindBufferBase / glBindBufferRange on GL_UNIFORM_BUFFER
struct UniformBufferBinding {
	GLuint buffer = 0;
	GLintptr offset = 0;
	GLsizeiptr size = 0; // 0 = the whole buffer
};

// GL_ARB_draw_buffers_blend / GL 4.0: parameters are independent per draw buffer.
struct BlendBufferState {
	GLenum src_rgb = GL_ONE;
	GLenum dst_rgb = GL_ZERO;
	GLenum src_alpha = GL_ONE;
	GLenum dst_alpha = GL_ZERO;
	GLenum equation_rgb = GL_FUNC_ADD;
	GLenum equation_alpha = GL_FUNC_ADD;
};

// MARK: - Programs

struct ProgramAttrib {
	std::string name;
	GLint gl_index;         // what glGetAttribLocation returns / glVertexAttribPointer indexes
	uint32_t msl_location;  // [[attribute(n)]] of the MSL vertex function
	GLint vecsize;
	bool is_integer;        // integer vertex format needed
	bool is_unsigned;       // uint/uvec vertex input
};

struct ProgramUniform {
	TranslatedUniform info; // name, type, per-stage layout in the default uniform block
	GLint base_location;    // an array of N elements owns N consecutive locations
	std::vector<GLint> units; // samplers: texture unit set with glUniform1i
};

struct ProgramBlock {
	TranslatedBlock info;
	GLuint binding_point = 0; // glUniformBlockBinding: GL_UNIFORM_BUFFER binding point the block reads from
	std::vector<GLuint> active_uniform_indices;
};

struct ActiveUniformInfo {
	std::string name;
	GLenum type = GL_FLOAT;
	GLint size = 1;
	GLint block_index = -1;
	GLint offset = -1;
	GLint array_stride = 0;
	GLint matrix_stride = 0;
	GLboolean row_major = GL_FALSE;
};

struct ProgramObject {
	std::vector<GLuint> shaders; // attached
	GLboolean linked = GL_FALSE, validated = GL_FALSE, deleted = GL_FALSE;
	std::string info_log;
	std::unordered_map<std::string, GLuint> attrib_bindings; // glBindAttribLocation, applied at link time

	// results of the last successful link
	id<MTLFunction> vertex_function = nil;
	id<MTLFunction> fragment_function = nil;
	id<MTLFunction> vertex_function_flipped = nil; // vertex_function with gl_Position.y negated (texture targets)
	std::vector<ProgramAttrib> attribs;
	std::vector<ProgramUniform> uniforms; // default-block uniforms with glUniform locations
	std::vector<ActiveUniformInfo> active_uniforms; // GL_ACTIVE_UNIFORMS includes named-block members too
	std::vector<std::pair<GLint, GLint>> locations; // GL uniform location -> (index in uniforms, array element)
	std::vector<uint8_t> uniform_data[3];           // CPU copy of default uniforms: [0] vertex, [1] fragment, [2] compute
	std::unordered_map<std::string, GLint> frag_outputs;
	std::vector<ProgramBlock> blocks;
	std::vector<TranslatedStorageBuffer> storage_buffers;
	std::vector<TranslatedImage> storage_images;
	id<MTLComputePipelineState> compute_pipeline = nil;
	uint32_t compute_workgroup_size[3] = {1, 1, 1};
	bool is_compute = false;
	bool uses_gl_frag_coord = false; // fragment MSL needs the per-render-pass coordinate parameters
};

struct ShaderStorageBufferBinding {
	GLuint buffer = 0;
	GLintptr offset = 0;
	GLsizeiptr size = 0; // 0 = whole store
};

struct ImageUnitBinding {
	GLuint texture = 0;
	GLint level = 0;
	GLboolean layered = GL_FALSE;
	GLint layer = 0;
	GLenum access = GL_READ_ONLY;
	GLenum format = GL_RGBA8;
};

struct MTLContext {
	id<MTLDevice> device;
	id<MTLCommandQueue> commandQueue;
	CAMetalLayer *metalLayer;
	
	id<CAMetalDrawable> currentDrawable;
	id<MTLCommandBuffer> currentCommandBuffer;
	id<MTLRenderCommandEncoder> currentEncoder; // open render pass of the current draw framebuffer, nil if none
	id<MTLRenderPipelineState> pipelineState;
	id<MTLDepthStencilState> depthStencilState;
	id<MTLTexture> depthTexture; // depth/stencil of the default framebuffer
	NSUInteger render_target_width, render_target_height; // size of the open render pass
	
	GLuint current_vao;
	GLuint next_buffer_id;
	GLuint next_vao_id;
	GLuint next_texture_id;
	GLuint next_fbo_id;
	GLuint next_shader_id; // shared with programs later (GL uses one name space for both)
	
	std::unordered_map<GLenum, GLuint> bound_buffers;
	std::unordered_map<GLuint, id<MTLBuffer>> buffer_objects;
	std::unordered_map<GLuint, VertexArray> vao_objects;
	std::unordered_map<GLuint, TextureObject> texture_objects;
	std::unordered_map<GLuint, FBO> fbo_objects;
	std::unordered_map<GLuint, ShaderObject> shader_objects;
	
	// texture bindings: [texture unit][target] -> texture name
	GLuint active_texture_unit;
	std::unordered_map<GLenum, GLuint> bound_textures[GL4METAL_MAX_TEXTURE_UNITS];
	struct {
		GLsizei w, h;
		GLenum internalformat;
	} proxy_texture_2d; // result of the last glTexImage2D(GL_PROXY_TEXTURE_2D)
	
	// framebuffer bindings (0 = the CAMetalLayer drawable)
	GLuint draw_fbo, read_fbo;
	
	// pixel storage
	GLint unpack_alignment, unpack_row_length, unpack_skip_pixels, unpack_skip_rows;
	GLint pack_alignment, pack_row_length, pack_skip_pixels, pack_skip_rows;
	
	// enable/disable caps that have no dedicated field below
	std::unordered_map<GLenum, GLboolean> caps;
	
	GLboolean depth_test_enabled, depth_write_enabled;
	GLenum depth_func;
	
	MTLClearColor clear_color;
	double clear_depth;
	GLint clear_stencil;
	GLbitfield clear_flags;
	
	struct {
		GLint x, y;
		GLsizei w, h;
	} viewport, scissor;
	
	GLboolean scissor_test_enabled;

	// Programs. Kept at the very end on purpose: members added in the middle of this class shift every member after
	// them, and any .mm that is not recompiled against the new header would silently read/write the wrong fields.
	std::unordered_map<GLuint, ProgramObject> program_objects;
	GLuint current_program; // glUseProgram, 0 = none

	// Draw state, also appended at the end (see the note above).
	GLenum blend_src_rgb, blend_dst_rgb, blend_src_alpha, blend_dst_alpha;
	GLenum blend_eq_rgb, blend_eq_alpha;
	GLfloat blend_color[4];
	GLboolean color_mask[4];
	GLenum cull_face_mode, front_face, polygon_mode;
	GLfloat polygon_offset_factor, polygon_offset_units;
	GLfloat depth_range_near, depth_range_far;

	// Attachments of the open render pass (pipelines have to be created for exactly these formats).
	MTLPixelFormat pass_color_format[GL4METAL_MAX_COLOR_ATTACHMENTS];
	MTLPixelFormat pass_depth_format, pass_stencil_format;
	bool pass_flipped; // the pass renders to a texture: Y is flipped so rows are bottom-first like GL

	id<MTLDepthStencilState> no_depth_state;  // used when the pass has no depth attachment
	id<MTLBuffer> constant_attrib_buffer;     // 32 bytes: float (0,0,0,1) then int (0,0,0,1), GL's default generic attribute
	id<MTLTexture> fallback_texture;          // what an unbound/incomplete texture samples: (0,0,0,1)
	std::unordered_map<std::string, id<MTLRenderPipelineState>> pipeline_cache;
	std::unordered_map<std::string, id<MTLSamplerState>> sampler_cache;

	// Sampler objects, uniform buffer bindings, fences (appended at the end, see the note above).
	GLuint next_sampler_id;
	std::unordered_map<GLuint, SamplerObject> sampler_objects;
	GLuint bound_samplers[GL4METAL_MAX_TEXTURE_UNITS]; // glBindSampler, 0 = none
	UniformBufferBinding uniform_buffer_bindings[GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS];
	ShaderStorageBufferBinding shader_storage_buffer_bindings[GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS];
	ImageUnitBinding image_unit_bindings[GL4METAL_MAX_TEXTURE_UNITS];
	std::unordered_set<const void *> sync_objects;
	id<MTLCommandBuffer> last_committed_buffer;
	GLboolean pack_swap_bytes = GL_FALSE; // GL_PACK_SWAP_BYTES; appended to preserve existing member offsets

	// Renderbuffer state is appended to keep offsets of older members stable across the ObjC++ translation units.
	GLuint next_renderbuffer_id = 0;
	GLuint bound_renderbuffer = 0;
	std::unordered_map<GLuint, RenderbufferObject> renderbuffer_objects;
	NSUInteger pass_sample_count = 1;

	// GL stencil state is stored per face; appended to preserve all older member offsets.
	struct StencilFaceState {
		GLenum func = GL_ALWAYS;
		GLint ref = 0;
		GLuint value_mask = 0xFFFFFFFFu;
		GLuint write_mask = 0xFFFFFFFFu;
		GLenum fail = GL_KEEP;
		GLenum depth_fail = GL_KEEP;
		GLenum depth_pass = GL_KEEP;
	};
	StencilFaceState stencil_front;
	StencilFaceState stencil_back;
	GLuint primitive_restart_index = 0xFFFFFFFFu;

	// Per-draw-buffer state from GL 3.0+. Metal stores write masks and blend enable
	// in each render-pipeline color attachment descriptor. Appended to preserve layout.
	GLboolean color_masks[GL4METAL_MAX_COLOR_ATTACHMENTS][4] = {};
	GLboolean blend_enabled[GL4METAL_MAX_COLOR_ATTACHMENTS] = {};

	// Last command buffer that encoded a draw using each GL buffer object. This lets
	// CPU updates rename storage instead of stalling the entire render thread.
	std::unordered_map<GLuint, id<MTLCommandBuffer>> buffer_last_use;

	// Independent blend factors/equations for GL draw-buffer indices 0...7.
	// Appended to keep the offsets of older MTLContext members stable.
	BlendBufferState blend_buffers[GL4METAL_MAX_COLOR_ATTACHMENTS];

	// Exact GL buffer-store sizes. Metal storage may be larger when padded for linear texture-buffer views.
	std::unordered_map<GLuint, NSUInteger> buffer_sizes;

	// Default drawable depth/stencil contents are transient between presents. Appended to preserve layout.
	bool default_color_has_contents = false;
	bool default_depth_has_contents = false;
	bool default_stencil_has_contents = false;
	bool pass_uses_default_framebuffer = false;
	bool discard_default_depth_stencil_on_end = false;
	// Fragment gl_FragCoord parameters are constant for the lifetime of one render encoder.
	bool fragcoord_params_bound = false;

	// Compute encoder is appended to preserve all existing MTLContext member offsets.
	id<MTLComputeCommandEncoder> currentComputeEncoder = nil;
	uint64_t render_pass_serial = 0; // logical pass identity; feedback blits may split Metal encoders
};

extern MTLContext ctx;

inline NSUInteger glBufferLogicalSize(GLuint bufferName) {
	auto sizeIt = ctx.buffer_sizes.find(bufferName);
	if (sizeIt != ctx.buffer_sizes.end()) return sizeIt->second;
	auto bufferIt = ctx.buffer_objects.find(bufferName);
	return bufferIt != ctx.buffer_objects.end() && bufferIt->second ? bufferIt->second.length : 0;
}
extern std::recursive_mutex gl4metalSamplerObjectMutex; // sampler object state shared by GL entry points and draw.mm

static inline GLuint boundBuffer(GLenum target) {
	auto it = ctx.bound_buffers.find(target);
	return it == ctx.bound_buffers.end() ? 0 : it->second;
}

#ifdef __cplusplus
extern "C" {
#endif

GLboolean gl4metalInit();
void updateDepthStencilState();

#ifdef __cplusplus
}
#endif

// MARK: - Internal helpers (C++ linkage, not part of the GL API)

MTLCompareFunction getMetalCompareFunc(GLenum func);                // gl4metal.mm
void gl4metalSetError(GLenum error);                                // getter.mm: what the next glGetError returns
void genericAttribValue(GLuint index, bool integer, void *out16);   // vertexarray.mm: glVertexAttrib* current value
void invalidateConvertedVertexBuffers(GLuint buffer);               // draw.mm: invalidate cached integer attribute repacks
void markBufferUsed(GLuint buffer);                                 // buffers.mm: record one GPU buffer use
void markBuffersUsed(const GLuint *buffers, size_t count);          // buffers.mm: batch GPU buffer-use tracking per draw

#include "exports.h"
