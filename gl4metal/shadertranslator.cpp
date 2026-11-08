#include "shadertranslator.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h> // GetDefaultResources() (library: glslang-default-resource-limits)
#include <SPIRV/GlslangToSpv.h>
#include <spirv_msl.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace {

enum Stage { VS = 0, FS = 1 };

void initGlslang() {
	// C++11 function-local static initialization is synchronized across linker threads.
	static const bool initialized = [] { return glslang::InitializeProcess(); }();
	(void)initialized;
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

// Convert pre-4.10 desktop GLSL to desktop GLSL 4.10 before glslang parses it.
// This includes Iris-generated #version 400 core shaders: the varying-alignment pass below inserts
// layout(location) on stage inputs/outputs, which is core syntax starting with GLSL 4.10.
// Keep the original profile suffix (for example, "core") and do not rewrite GLSL ES versions.
// No GL_ARB_separate_shader_objects directive is injected; GL4Metal does not claim to implement it.
std::string patchVersion(const std::string &source) {
	std::string out = source;
	size_t lineStart = 0;
	while (lineStart < out.size()) {
		size_t lineEnd = out.find('\n', lineStart);
		if (lineEnd == std::string::npos) lineEnd = out.size();

		size_t directive = lineStart;
		while (directive < lineEnd && (out[directive] == ' ' || out[directive] == '\t' || out[directive] == '\r')) {
			++directive;
		}
		if (out.compare(directive, 8, "#version") == 0 &&
			(directive + 8 == lineEnd || out[directive + 8] == ' ' || out[directive + 8] == '\t')) {
			size_t numberStart = directive + 8;
			while (numberStart < lineEnd && (out[numberStart] == ' ' || out[numberStart] == '\t')) ++numberStart;
			size_t numberEnd = numberStart;
			while (numberEnd < lineEnd && out[numberEnd] >= '0' && out[numberEnd] <= '9') ++numberEnd;
			if (numberEnd > numberStart) {
				const int version = std::stoi(out.substr(numberStart, numberEnd - numberStart));
				size_t profile = numberEnd;
				while (profile < lineEnd && (out[profile] == ' ' || out[profile] == '\t')) ++profile;
				const bool isGlslEs = out.compare(profile, 2, "es") == 0 &&
					(profile + 2 == lineEnd || out[profile + 2] == ' ' || out[profile + 2] == '\t' || out[profile + 2] == '\r');
				if (!isGlslEs && version < 410) {
					out.replace(numberStart, numberEnd - numberStart, "410");
				}
			}
			break; // only the real leading #version directive controls the shader language version
		}
		lineStart = lineEnd + 1;
	}
	return renameSamplerIdentifier(out);
}


// Preserve source offsets while masking comments, so declaration matching never sees fake `in`/`out` tokens in comments.
std::string maskShaderComments(const std::string &source) {
	std::string masked = source;
	bool lineComment = false, blockComment = false;
	for (size_t i = 0; i < masked.size(); i++) {
		if (lineComment) {
			if (masked[i] == '\n') lineComment = false;
			else masked[i] = ' ';
			continue;
		}
		if (blockComment) {
			if (masked[i] == '*' && i + 1 < masked.size() && masked[i + 1] == '/') {
				masked[i] = masked[i + 1] = ' ';
				i++;
				blockComment = false;
			} else if (masked[i] != '\n') {
				masked[i] = ' ';
			}
			continue;
		}
		if (masked[i] == '/' && i + 1 < masked.size() && masked[i + 1] == '/') {
			masked[i] = masked[i + 1] = ' ';
			i++;
			lineComment = true;
		} else if (masked[i] == '/' && i + 1 < masked.size() && masked[i + 1] == '*') {
			masked[i] = masked[i + 1] = ' ';
			i++;
			blockComment = true;
		}
	}
	return masked;
}

struct SimpleInterfaceVariable {
	std::string name;
	std::string type;
	size_t insertionOffset = 0;
	uint32_t locationCount = 1;
};

static uint32_t interfaceLocationCount(const std::string &type, const std::string &arrayLength) {
	uint32_t count = 1;
	if (type.size() >= 4 && type.compare(0, 3, "mat") == 0 && type[3] >= '2' && type[3] <= '4') {
		count = (uint32_t)(type[3] - '0'); // matrix columns each consume one location
	}
	if (!arrayLength.empty()) {
		try { count *= (uint32_t)std::stoul(arrayLength); } catch (...) { return 1; }
	}
	return std::max<uint32_t>(count, 1);
}

static std::vector<SimpleInterfaceVariable> simpleInterfaceVariables(const std::string &masked, const char *storage) {
	const std::string qualifiers = R"((?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*)";
	const std::string pattern = R"((^|\n)([ \t]*))" + qualifiers + R"(\b)" + storage +
		R"([ \t]+(float|int|uint|bool|[biu]?vec[234]|mat[234](?:x[234])?)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:\[[ \t]*([0-9]+)[ \t]*\])?[ \t]*;)";
	const std::regex re(pattern);
	std::vector<SimpleInterfaceVariable> vars;
	for (std::sregex_iterator it(masked.begin(), masked.end(), re), end; it != end; ++it) {
		const std::smatch &m = *it;
		SimpleInterfaceVariable v;
		v.type = m[3].str();
		v.name = m[4].str();
		v.locationCount = interfaceLocationCount(v.type, m[5].str());
		v.insertionOffset = (size_t)m.position() + m.length(1) + m.length(2);
		vars.push_back(std::move(v));
	}
	return vars;
}

static size_t countSimpleStorageDeclarations(const std::string &masked, const char *storage) {
	const std::string qualifiers = R"((?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*)";
	const std::string pattern = R"((^|[;\n])[ \t]*)" + qualifiers + R"(\b)" + storage + R"(\b[^;\n]*;)";
	const std::regex re(pattern);
	return (size_t)std::distance(std::sregex_iterator(masked.begin(), masked.end(), re), std::sregex_iterator());
}

static bool hasLayoutQualifiedStorage(const std::string &masked, const char *storage) {
	const std::string pattern = std::string(R"(layout[ \t]*\([^)]*\)[ \t]*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\b)") + storage + R"(\b)";
	return std::regex_search(masked, std::regex(pattern));
}

// SPIR-V locations must match across the stages. Metal matches [[user(locnN)]] by location, so explicitly number simple
// unqualified interfaces by vertex-output order and assign that same location to the matching fragment input.
// patchVersion() upgrades desktop GLSL to 410 so these explicit varying locations are core syntax.
void alignUnqualifiedVaryingLocations(std::string &vertexSource, std::string &fragmentSource) {
	const std::string vsMasked = maskShaderComments(vertexSource);
	const std::string fsMasked = maskShaderComments(fragmentSource);
	if (hasLayoutQualifiedStorage(vsMasked, "out") || hasLayoutQualifiedStorage(fsMasked, "in")) return;

	auto outputs = simpleInterfaceVariables(vsMasked, "out");
	auto inputs = simpleInterfaceVariables(fsMasked, "in");
	if (outputs.empty() || inputs.empty() ||
		outputs.size() != countSimpleStorageDeclarations(vsMasked, "out") ||
		inputs.size() != countSimpleStorageDeclarations(fsMasked, "in")) {
		return; // complex blocks, comma declarations, or unsupported types: don't guess
	}

	std::unordered_map<std::string, uint32_t> locationByName;
	std::unordered_map<std::string, std::pair<std::string, uint32_t>> interfaceByName;
	std::vector<uint32_t> outputLocations, inputLocations;
	uint32_t nextLocation = 0;
	for (const SimpleInterfaceVariable &v : outputs) {
		auto existing = locationByName.find(v.name);
		uint32_t location;
		if (existing != locationByName.end()) {
			const auto &known = interfaceByName.at(v.name);
			if (known.first != v.type || known.second != v.locationCount) return;
			location = existing->second;
		} else {
			location = nextLocation;
			locationByName.emplace(v.name, location);
			interfaceByName.emplace(v.name, std::make_pair(v.type, v.locationCount));
			nextLocation += v.locationCount;
		}
		outputLocations.push_back(location);
	}
	for (const SimpleInterfaceVariable &v : inputs) {
		auto found = locationByName.find(v.name);
		auto interface = interfaceByName.find(v.name);
		if (found == locationByName.end() || interface == interfaceByName.end() ||
			interface->second.first != v.type || interface->second.second != v.locationCount) {
			return; // don't partially rewrite an interface that doesn't match by name and type
		}
		inputLocations.push_back(found->second);
	}

	std::vector<std::pair<size_t, std::string>> vsInsertions, fsInsertions;
	for (size_t i = 0; i < outputs.size(); i++) {
		vsInsertions.emplace_back(outputs[i].insertionOffset, "layout(location = " + std::to_string(outputLocations[i]) + ") ");
	}
	for (size_t i = 0; i < inputs.size(); i++) {
		fsInsertions.emplace_back(inputs[i].insertionOffset, "layout(location = " + std::to_string(inputLocations[i]) + ") ");
	}
	auto apply = [](std::string &source, std::vector<std::pair<size_t, std::string>> &insertions) {
		std::sort(insertions.begin(), insertions.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
		for (const auto &entry : insertions) source.insert(entry.first, entry.second);
	};
	apply(vertexSource, vsInsertions);
	apply(fragmentSource, fsInsertions);
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

static bool isMslIdentifierChar(char c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		(c >= '0' && c <= '9') || c == '_';
}

static size_t matchingMslParen(const std::string &source, size_t open) {
	if (open >= source.size() || source[open] != '(') return std::string::npos;
	int depth = 0;
	for (size_t i = open; i < source.size(); ++i) {
		if (source[i] == '(') ++depth;
		else if (source[i] == ')' && --depth == 0) return i;
	}
	return std::string::npos;
}

// MSL texture2d.read() uses integer texel coordinates. GL texelFetch() outside the texture's
// extent is undefined; clamping those coordinates preserves every in-range fetch and prevents
// an invalid integer coordinate (including a fetch through the 1x1 incomplete-texture fallback)
// from turning into a GPU address fault on Apple GPUs.
static size_t clampMslTexture2DReadCoordinates(std::string &source) {
	std::unordered_set<std::string> texture2DNames;
	const std::string declaration = "texture2d<";
	size_t scan = 0;
	while ((scan = source.find(declaration, scan)) != std::string::npos) {
		size_t typeEnd = source.find('>', scan + declaration.size());
		if (typeEnd == std::string::npos) break;
		size_t nameStart = typeEnd + 1;
		while (nameStart < source.size() && (source[nameStart] == ' ' || source[nameStart] == '\t' || source[nameStart] == '\n')) ++nameStart;
		size_t nameEnd = nameStart;
		while (nameEnd < source.size() && isMslIdentifierChar(source[nameEnd])) ++nameEnd;
		if (nameEnd > nameStart) texture2DNames.insert(source.substr(nameStart, nameEnd - nameStart));
		scan = typeEnd + 1;
	}

	size_t rewritten = 0;
	size_t pos = 0;
	const std::string call = ".read(uint2(";
	while ((pos = source.find(call, pos)) != std::string::npos) {
		// Resolve the receiver identifier immediately before `.read`.
		size_t nameEnd = pos;
		size_t nameStart = nameEnd;
		while (nameStart > 0 && isMslIdentifierChar(source[nameStart - 1])) --nameStart;
		if (nameStart == nameEnd) { pos += call.size(); continue; }
		const std::string textureName = source.substr(nameStart, nameEnd - nameStart);
		if (texture2DNames.find(textureName) == texture2DNames.end()) { pos += call.size(); continue; }

		const size_t readOpen = pos + 5; // opening '(' in `.read(`
		const size_t coordOpen = source.find('(', readOpen + 1); // opening '(' in `uint2(`
		if (coordOpen == std::string::npos) break;
		const size_t coordClose = matchingMslParen(source, coordOpen);
		const size_t readClose = matchingMslParen(source, readOpen);
		if (coordClose == std::string::npos || readClose == std::string::npos || coordClose >= readClose) {
			pos += call.size();
			continue;
		}
		size_t comma = coordClose + 1;
		while (comma < readClose && (source[comma] == ' ' || source[comma] == '\t' || source[comma] == '\n')) ++comma;
		if (comma >= readClose || source[comma] != ',') { pos += call.size(); continue; }
		size_t levelStart = comma + 1;
		while (levelStart < readClose && (source[levelStart] == ' ' || source[levelStart] == '\t' || source[levelStart] == '\n')) ++levelStart;
		const std::string level = source.substr(levelStart, readClose - levelStart);
		if (level.empty()) { pos += call.size(); continue; }

		const std::string coordinates = source.substr(coordOpen + 1, coordClose - coordOpen - 1);
		const std::string clamped = "uint2(clamp(int2(" + coordinates + "), int2(0), int2(int(" + textureName +
			".get_width(" + level + ")) - 1, int(" + textureName + ".get_height(" + level + ")) - 1)))";
		source.replace(pos + std::string(".read(").size(), coordClose - (pos + std::string(".read(").size()) + 1, clamped);
		++rewritten;
		pos += std::string(".read(").size() + clamped.size();
	}
	return rewritten;
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

		// Reflect only resources statically accessed by this entry point. Iris injects
		// shared sampler declarations into both stages, even when a stage does not use
		// them. Reflecting every declaration marks those inactive samplers as present,
		// which causes needless Metal bindings and can trigger false feedback-loop skips.
		auto activeVariables = msl.get_active_interface_variables();
		spirv_cross::ShaderResources resources = msl.get_shader_resources(activeVariables);
		msl.set_enabled_interface_variables(std::move(activeVariables));

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
				a.is_unsigned = base == spirv_cross::SPIRType::UInt;
				a.location = msl.get_decoration(r.id, spv::DecorationLocation);
				out.attribs.push_back(a);
			}
		} else {
			for (const spirv_cross::Resource &r : resources.stage_outputs) {
				out.frag_outputs.emplace_back(r.name, msl.get_decoration(r.id, spv::DecorationLocation));
			}
		}

		std::string compiledMsl = msl.compile();
		const size_t clampedReads = clampMslTexture2DReadCoordinates(compiledMsl);
		if (clampedReads != 0) {
			out.log += std::to_string(clampedReads) + " out-of-range-safe texture2d.read coordinate guard(s) added to " +
				(stage == VS ? "vertex" : "fragment") + " MSL stage\n";
		}
		(flipY ? out.msl_vs_flipped : out.msl[stage]) = std::move(compiledMsl);
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
	alignUnqualifiedVaryingLocations(vsText, fsText);
	const char *vsSource = vsText.c_str();
	const char *fsSource = fsText.c_str();

	glslang::TShader vs(EShLangVertex);
	glslang::TShader fs(EShLangFragment);
	setupShader(vs, EShLangVertex, &vsSource);
	setupShader(fs, EShLangFragment, &fsSource);

	const EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	const TBuiltInResource *resources = GetDefaultResources();

	if (!vs.parse(resources, 410, false, messages)) {
		out.log += std::string("vertex shader:\n") + vs.getInfoLog() + vs.getInfoDebugLog();
		return false;
	}
	if (!fs.parse(resources, 410, false, messages)) {
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
