#pragma once

// GLSL (OpenGL 3.3 core) -> SPIR-V (glslang) -> MSL (SPIRV-Cross), done for a whole program at link time so that
// TProgram::mapIO() can give matching locations to vertex outputs and fragment inputs.
// Plain C++ on purpose: no Metal/ObjC types, only shadertranslator.cpp includes glslang and SPIRV-Cross.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Metal buffer argument index of the "default uniform block" (GL's loose uniforms) in both stages.
// Vertex buffers occupy the low indices (0...15), so the uniform data lives at the top of the argument table.
#define GL4METAL_UNIFORM_BUFFER_INDEX 30
// Shared GL-to-Metal binding budgets used by both the plain-C++ translator and ObjC++ backend.
#ifndef GL4METAL_MAX_TEXTURE_UNITS
#  define GL4METAL_MAX_TEXTURE_UNITS 32
#endif
#ifndef GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
#  define GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 16
#endif
// Reserved fragment-stage buffer slot for the dynamic GL gl_FragCoord coordinate fixup.
#define GL4METAL_FRAGCOORD_BUFFER_INDEX 15

struct TranslatedUniform {
	enum Kind { Float, Int, UInt, Bool, Sampler };
	enum Dim { Dim2D, DimCube, Dim3D, Dim2DArray, DimBuffer }; // samplers: what the shader samples

	std::string name;
	Kind kind = Float;
	int vecsize = 1;     // components of a vector / rows of a matrix
	int columns = 1;     // > 1 for matrices
	int array_size = 1;
	Dim dim = Dim2D;     // samplers only
	uint32_t sampler_gl_type = 0; // exact GL sampler enum when source reflection provides it

	struct Stage {
		bool present = false;
		uint32_t offset = 0;        // byte offset inside the default uniform block of that stage
		uint32_t array_stride = 0;
		uint32_t matrix_stride = 0;
		bool row_major = false;
		uint32_t slot = 0;          // samplers: [[texture(slot)]] and [[sampler(slot)]]
	} stage[3];                     // 0 = vertex, 1 = fragment, 2 = compute
};

struct TranslatedBlock { // a real (named) uniform block, e.g. Sodium's ubo_CameraMatrices
	std::string name;
	uint32_t binding = 0; // GL_UNIFORM_BUFFER binding point from layout(binding=...), default 0
	uint32_t size = 0;      // GL_UNIFORM_BLOCK_DATA_SIZE, including std140 tail padding
	struct Stage {
		bool present = false;
		uint32_t slot = 0;  // Metal buffer index in that stage
	} stage[3];             // 0 = vertex, 1 = fragment, 2 = compute
	std::vector<TranslatedUniform> members; // active members of this named uniform block
	uint32_t required_size = 0; // end of the last declared member; Metal reads no std140 tail padding
};

struct TranslatedAttrib {
	std::string name;
	int vecsize;
	uint32_t location; // [[attribute(location)]] in the MSL vertex function
	bool is_integer = false; // ivec/uvec input: needs an integer vertex format
	bool is_unsigned = false; // uvec input: Metal vertex format must also be unsigned
};

struct TranslatedStorageBuffer {
	std::string name;
	uint32_t binding = 0; // GL_SHADER_STORAGE_BUFFER binding point
	uint32_t slot = 0;    // MSL [[buffer(slot)]]
	uint32_t required_size = 0;
	bool runtime_sized = false; // unsized trailing array: Metal must see the full remaining GL store
};

struct TranslatedImage {
	std::string name;
	uint32_t binding = 0; // GL image unit from layout(binding=...)
	uint32_t slot = 0;    // MSL [[texture(slot)]]
	uint32_t format = 0;  // GL sized image format, 0 when the GLSL format is unqualified
	bool read_only = false;
	bool write_only = false;
};

struct TranslatedProgram {
	std::string msl[3];                        // vertex, fragment, compute (entry point "main0")
	uint32_t block_size[3] = {0, 0, 0};           // bytes of the default uniform block, 16 byte aligned, 0 = none
	std::vector<TranslatedUniform> uniforms;
	std::vector<TranslatedAttrib> attribs;     // vertex inputs
	std::vector<TranslatedBlock> blocks;       // uniform blocks other than the default one
	std::vector<TranslatedStorageBuffer> storage_buffers;
	std::vector<TranslatedImage> storage_images;
	uint32_t workgroup_size[3] = {1, 1, 1};
	std::vector<std::pair<std::string, uint32_t>> frag_outputs;
	bool uses_gl_frag_coord = false; // fragment MSL entry point includes the dynamic render-target coordinate parameters
	std::string log;
};

bool translateProgram(const std::string &vertexGLSL, const std::string &fragmentGLSL, TranslatedProgram &out);
bool translateComputeProgram(const std::string &computeGLSL, TranslatedProgram &out);
