#include "shadertranslator.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h> // GetDefaultResources() (library: glslang-default-resource-limits)
#include <SPIRV/GlslangToSpv.h>
#include <spirv_msl.hpp>

#include <algorithm>
#include <exception>

namespace {

enum Stage { VS = 0, FS = 1 };

void initGlslang() {
	static bool initialized = false;
	if (!initialized) {
		glslang::InitializeProcess();
		initialized = true;
	}
}

// glslang's Vulkan dialect reserves `sampler` (separate samplers), GLSL 330 doesn't: Minecraft's shaders use it as the
// name of a parameter (sampleNearest(sampler2D sampler, ...)). Whole-word occurrences only, sampler2D & co. stay.
std::string renameSamplerIdentifier(const std::string &source) {
	auto isIdentifier = [](char ch) { return ch == '_' || (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'); };
	const std::string word = "sampler";
	std::string out;
	out.reserve(source.size() + 16);
	size_t i = 0;
	while (i < source.size()) {
		bool match = source.compare(i, word.size(), word) == 0 &&
					 (i == 0 || !isIdentifier(source[i - 1])) &&
					 (i + word.size() >= source.size() || !isIdentifier(source[i + word.size()]));
		if (match) {
			out += "sampler_";
			i += word.size();
		} else {
			out += source[i++];
		}
	}
	return out;
}

// Rewrites done on the GLSL: Minecraft ships #version 150, glslang's SPIR-V rules want 330, plus the identifier above.
std::string patchVersion(const std::string &source) {
	std::string out = source;
	const std::string from = "#version 150";
	size_t pos = out.find(from);
	if (pos != std::string::npos) {
		out.replace(pos, from.size(), "#version 330");
	}
	return renameSamplerIdentifier(out);
}

void setupShader(glslang::TShader &shader, EShLanguage stage, const char *const *source) {
	shader.setStrings(source, 1);
	// Vulkan semantics + "relaxed" rules: loose GL uniforms are collected into gl_DefaultUniformBlock (a real buffer,
	// which MSL needs), and in/out without layout(location) are accepted and numbered by mapIO().
	shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
	shader.setEnvInputVulkanRulesRelaxed();
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
	shader.setAutoMapLocations(true);
	shader.setAutoMapBindings(true);
}

TranslatedUniform &findOrAdd(TranslatedProgram &out, const std::string &name) {
	for (TranslatedUniform &u : out.uniforms) {
		if (u.name == name) {
			return u;
		}
	}
	out.uniforms.emplace_back();
	out.uniforms.back().name = name;
	return out.uniforms.back();
}

TranslatedUniform::Kind kindOf(spirv_cross::SPIRType::BaseType base) {
	switch (base) {
		case spirv_cross::SPIRType::Int:     return TranslatedUniform::Int;
		case spirv_cross::SPIRType::UInt:    return TranslatedUniform::UInt;
		case spirv_cross::SPIRType::Boolean: return TranslatedUniform::Bool;
		default:                             return TranslatedUniform::Float;
	}
}

int arraySizeOf(const spirv_cross::SPIRType &type) {
	if (!type.array.empty() && type.array_size_literal[0]) {
		return (int)type.array[0];
	}
	return 1;
}

// flipY: second pass over the vertex stage that only produces msl_vs_flipped (reflection was done by the first pass).
bool compileStage(std::vector<unsigned int> spirv, spv::ExecutionModel model, int stage, bool flipY, TranslatedProgram &out) {
	try {
		spirv_cross::CompilerMSL msl(std::move(spirv));

		spirv_cross::CompilerMSL::Options options;
		options.platform = spirv_cross::CompilerMSL::Options::iOS; // iOS GPUs, not macOS
		options.set_msl_version(2, 2);
		msl.set_msl_options(options);

		spirv_cross::CompilerGLSL::Options common = msl.get_common_options();
		common.vertex.fixup_clipspace = true; // GL clip space z is [-w, w], Metal's is [0, w]
		common.vertex.flip_vert_y = flipY;
		msl.set_common_options(common);

		spirv_cross::ShaderResources resources = msl.get_shader_resources();

		// Uniform blocks: the default block goes to a fixed high buffer index (see GL4METAL_UNIFORM_BUFFER_INDEX),
		// real uniform blocks (not supported yet) are parked at 16 + binding so they never collide with vertex buffers.
		for (const spirv_cross::Resource &r : resources.uniform_buffers) {
			const spirv_cross::SPIRType &type = msl.get_type(r.base_type_id);
			bool isDefault = msl.get_name(r.base_type_id) == "gl_DefaultUniformBlock" || r.name == "gl_DefaultUniformBlock";
			uint32_t binding = msl.get_decoration(r.id, spv::DecorationBinding);

			spirv_cross::MSLResourceBinding b;
			b.stage = model;
			b.desc_set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
			b.binding = binding;
			b.msl_buffer = isDefault ? GL4METAL_UNIFORM_BUFFER_INDEX : 16 + binding;
			msl.add_msl_resource_binding(b);

			if (flipY) {
				continue;
			}
			if (!isDefault) {
				std::string blockName = msl.get_name(r.base_type_id);
				if (blockName.empty()) {
					blockName = r.name;
				}
				TranslatedBlock *block = nullptr;
				for (TranslatedBlock &candidate : out.blocks) {
					if (candidate.name == blockName) {
						block = &candidate;
					}
				}
				if (!block) {
					out.blocks.emplace_back();
					block = &out.blocks.back();
					block->name = blockName;
				}
				size_t blockSize = msl.get_declared_struct_size(type);
				block->size = std::max(block->size, (uint32_t)((blockSize + 15) & ~(size_t)15));
				block->stage[stage].present = true;
				block->stage[stage].slot = b.msl_buffer;
				continue;
			}

			size_t size = msl.get_declared_struct_size(type);
			out.block_size[stage] = (uint32_t)((size + 15) & ~(size_t)15);

			// Reflection has to happen before compile(): compile() may rename members that clash with MSL keywords.
			for (uint32_t i = 0; i < type.member_types.size(); i++) {
				const spirv_cross::SPIRType &member = msl.get_type(type.member_types[i]);
				std::string name = msl.get_member_name(r.base_type_id, i);
				if (member.basetype == spirv_cross::SPIRType::Struct) {
					out.log += "uniform '" + name + "' is a struct, not supported\n";
					continue;
				}
				TranslatedUniform &u = findOrAdd(out, name);
				u.kind = kindOf(member.basetype);
				u.vecsize = (int)member.vecsize;
				u.columns = (int)member.columns;
				u.array_size = arraySizeOf(member);

				TranslatedUniform::Stage &st = u.stage[stage];
				st.present = true;
				st.offset = msl.type_struct_member_offset(type, i);
				st.array_stride = member.array.empty() ? 0 : msl.type_struct_member_array_stride(type, i);
				st.matrix_stride = member.columns > 1 ? msl.type_struct_member_matrix_stride(type, i) : 0;
			}
		}

		// Combined image samplers become a texture + sampler pair sharing one dense slot per stage.
		std::vector<std::pair<uint32_t, const spirv_cross::Resource *>> samplers;
		for (const spirv_cross::Resource &r : resources.sampled_images) {
			samplers.emplace_back(msl.get_decoration(r.id, spv::DecorationBinding), &r);
		}
		std::sort(samplers.begin(), samplers.end(),
				  [](const std::pair<uint32_t, const spirv_cross::Resource *> &a,
					 const std::pair<uint32_t, const spirv_cross::Resource *> &b) { return a.first < b.first; });

		uint32_t slot = 0;
		for (const auto &entry : samplers) {
			const spirv_cross::Resource &r = *entry.second;
			int count = arraySizeOf(msl.get_type(r.type_id));

			spirv_cross::MSLResourceBinding b;
			b.stage = model;
			b.desc_set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
			b.binding = entry.first;
			b.msl_texture = slot;
			b.msl_sampler = slot;
			msl.add_msl_resource_binding(b);

			if (flipY) {
				slot += (uint32_t)count;
				continue;
			}
			TranslatedUniform &u = findOrAdd(out, r.name);
			u.kind = TranslatedUniform::Sampler;
			const spirv_cross::SPIRType &samplerType = msl.get_type(r.type_id);
			if (samplerType.image.dim == spv::DimCube) {
				u.dim = TranslatedUniform::DimCube;
			} else if (samplerType.image.dim == spv::Dim3D) {
				u.dim = TranslatedUniform::Dim3D;
			} else if (samplerType.image.arrayed) {
				u.dim = TranslatedUniform::Dim2DArray;
			} else {
				u.dim = TranslatedUniform::Dim2D;
			}
			u.vecsize = 1;
			u.columns = 1;
			u.array_size = count;
			u.stage[stage].present = true;
			u.stage[stage].slot = slot;
			slot += (uint32_t)count;
		}

		if (flipY) {
			// interface already reflected by the regular vertex pass
		} else if (stage == VS) {
			for (const spirv_cross::Resource &r : resources.stage_inputs) {
				if (r.name.compare(0, 3, "gl_") == 0) {
					continue;
				}
				TranslatedAttrib a;
				a.name = r.name;
				a.vecsize = (int)msl.get_type(r.type_id).vecsize;
				const spirv_cross::SPIRType::BaseType base = msl.get_type(r.type_id).basetype;
				a.is_integer = base == spirv_cross::SPIRType::Int || base == spirv_cross::SPIRType::UInt;
				a.location = msl.get_decoration(r.id, spv::DecorationLocation);
				out.attribs.push_back(a);
			}
		} else {
			for (const spirv_cross::Resource &r : resources.stage_outputs) {
				out.frag_outputs.emplace_back(r.name, msl.get_decoration(r.id, spv::DecorationLocation));
			}
		}

		(flipY ? out.msl_vs_flipped : out.msl[stage]) = msl.compile();
		return true;
	} catch (const std::exception &e) {
		out.log += std::string(stage == VS ? "vertex" : "fragment") + " stage, SPIRV-Cross: " + e.what() + "\n";
		return false;
	}
}

} // namespace

bool translateProgram(const std::string &vertexGLSL, const std::string &fragmentGLSL, TranslatedProgram &out) {
	initGlslang();

	std::string vsText = patchVersion(vertexGLSL);
	std::string fsText = patchVersion(fragmentGLSL);
	const char *vsSource = vsText.c_str();
	const char *fsSource = fsText.c_str();

	glslang::TShader vs(EShLangVertex);
	glslang::TShader fs(EShLangFragment);
	setupShader(vs, EShLangVertex, &vsSource);
	setupShader(fs, EShLangFragment, &fsSource);

	const EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	const TBuiltInResource *resources = GetDefaultResources();

	if (!vs.parse(resources, 330, false, messages)) {
		out.log += std::string("vertex shader:\n") + vs.getInfoLog() + vs.getInfoDebugLog();
		return false;
	}
	if (!fs.parse(resources, 330, false, messages)) {
		out.log += std::string("fragment shader:\n") + fs.getInfoLog() + fs.getInfoDebugLog();
		return false;
	}

	// Both stages in one TProgram: mapIO() numbers varyings by name across the stages, which is what makes
	// vertex outputs and fragment inputs line up (translating each shader on its own can't guarantee that).
	glslang::TProgram program;
	program.addShader(&vs);
	program.addShader(&fs);
	if (!program.link(messages)) {
		out.log += std::string("link:\n") + program.getInfoLog() + program.getInfoDebugLog();
		return false;
	}
	if (!program.mapIO()) {
		out.log += std::string("mapIO:\n") + program.getInfoLog() + program.getInfoDebugLog();
		return false;
	}

	const EShLanguage languages[2] = { EShLangVertex, EShLangFragment };
	std::vector<unsigned int> spirv[2];
	for (int i = 0; i < 2; i++) {
		glslang::TIntermediate *intermediate = program.getIntermediate(languages[i]);
		if (!intermediate) {
			out.log += "no intermediate representation for a stage\n";
			return false;
		}
		spv::SpvBuildLogger logger;
		glslang::GlslangToSpv(*intermediate, spirv[i], &logger);
		if (spirv[i].empty()) {
			out.log += "SPIR-V generation failed: " + logger.getAllMessages() + "\n";
			return false;
		}
	}

	// Texture targets are rendered with Y flipped (so texture rows follow GL's bottom-first order and sampling code
	// behaves exactly like on GL), the default framebuffer is not: hence two vertex variants.
	std::vector<unsigned int> vertexSpirv = spirv[VS];
	return compileStage(std::move(spirv[VS]), spv::ExecutionModelVertex, VS, false, out) &&
		   compileStage(std::move(spirv[FS]), spv::ExecutionModelFragment, FS, false, out) &&
		   compileStage(std::move(vertexSpirv), spv::ExecutionModelVertex, VS, true, out);
}
