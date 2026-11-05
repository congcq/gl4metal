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

struct TranslatedUniform {
	enum Kind { Float, Int, UInt, Bool, Sampler };
	enum Dim { Dim2D, DimCube, Dim3D, Dim2DArray }; // samplers: what the shader samples

	std::string name;
	Kind kind = Float;
	int vecsize = 1;     // components of a vector / rows of a matrix
	int columns = 1;     // > 1 for matrices
	int array_size = 1;
	Dim dim = Dim2D;     // samplers only

	struct Stage {
		bool present = false;
		uint32_t offset = 0;        // byte offset inside the default uniform block of that stage
		uint32_t array_stride = 0;
		uint32_t matrix_stride = 0;
		uint32_t slot = 0;          // samplers: [[texture(slot)]] and [[sampler(slot)]]
	} stage[2];                     // 0 = vertex, 1 = fragment
};

struct TranslatedBlock { // a real (named) uniform block, e.g. Sodium's ubo_CameraMatrices
	std::string name;
	uint32_t size = 0;      // bytes, 16 byte aligned
	struct Stage {
		bool present = false;
		uint32_t slot = 0;  // Metal buffer index in that stage
	} stage[2];             // 0 = vertex, 1 = fragment
};

struct TranslatedAttrib {
	std::string name;
	int vecsize;
	uint32_t location; // [[attribute(location)]] in the MSL vertex function
	bool is_integer = false; // ivec/uvec input: needs an integer vertex format
};

struct TranslatedProgram {
	std::string msl[2];                        // vertex, fragment (entry point "main0")
	std::string msl_vs_flipped;                // vertex stage again with gl_Position.y negated, for render-to-texture draws
	uint32_t block_size[2] = {0, 0};           // bytes of the default uniform block, 16 byte aligned, 0 = none
	std::vector<TranslatedUniform> uniforms;
	std::vector<TranslatedAttrib> attribs;     // vertex inputs
	std::vector<TranslatedBlock> blocks;       // uniform blocks other than the default one
	std::vector<std::pair<std::string, uint32_t>> frag_outputs;
	std::string log;
};

bool translateProgram(const std::string &vertexGLSL, const std::string &fragmentGLSL, TranslatedProgram &out);
