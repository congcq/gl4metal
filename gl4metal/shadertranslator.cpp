#include "shadertranslator.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h> // GetDefaultResources() (library: glslang-default-resource-limits)
#include <glslang/Include/intermediate.h>
#include <glslang/MachineIndependent/localintermediate.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv_msl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>
#include <vector>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glext.h>

namespace {

enum Stage { VS = 0, FS = 1 };

void initGlslang() {
	glslang::InitializeProcess();
}

void finalizeGlslang() {
	glslang::FinalizeProcess();
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
std::string patchVersion(const std::string &source, int minimumVersion = 410) {
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
				if (!isGlslEs && version < minimumVersion) {
					out.replace(numberStart, numberEnd - numberStart, std::to_string(minimumVersion));
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

struct SourceUniformType {
	bool is_bool = false;
	int bool_vecsize = 1;
	uint32_t sampler_gl_type = 0;
};

static void collectSourceUniformTypes(const std::string &source, std::unordered_map<std::string, SourceUniformType> &types) {
	const std::string masked = maskShaderComments(source);
	static const std::regex declaration(
		R"(\buniform[ \t]+(?:(?:lowp|mediump|highp)[ \t]+)?(bool|bvec2|bvec3|bvec4|samplerBuffer|isamplerBuffer|usamplerBuffer)[ \t]+([A-Za-z_][A-Za-z0-9_]*))");
	for (std::sregex_iterator it(masked.begin(), masked.end(), declaration), end; it != end; ++it) {
		const std::string typeName = (*it)[1].str();
		const std::string uniformName = (*it)[2].str();
		SourceUniformType info;
		if (typeName == "bool" || typeName == "bvec2" || typeName == "bvec3" || typeName == "bvec4") {
			info.is_bool = true;
			info.bool_vecsize = typeName == "bvec2" ? 2 : (typeName == "bvec3" ? 3 : (typeName == "bvec4" ? 4 : 1));
		} else if (typeName == "samplerBuffer") {
			info.sampler_gl_type = GL_SAMPLER_BUFFER;
		} else if (typeName == "isamplerBuffer") {
			info.sampler_gl_type = GL_INT_SAMPLER_BUFFER;
		} else if (typeName == "usamplerBuffer") {
			info.sampler_gl_type = GL_UNSIGNED_INT_SAMPLER_BUFFER;
		}
		types[uniformName] = info;
	}
}

static void applySourceUniformTypes(TranslatedProgram &out, const std::string &vertexSource, const std::string &fragmentSource) {
	std::unordered_map<std::string, SourceUniformType> sourceTypes;
	collectSourceUniformTypes(vertexSource, sourceTypes);
	collectSourceUniformTypes(fragmentSource, sourceTypes);
	for (TranslatedUniform &uniform : out.uniforms) {
		auto sourceType = sourceTypes.find(uniform.name);
		if (sourceType == sourceTypes.end()) continue;
		if (sourceType->second.is_bool) {
			uniform.kind = TranslatedUniform::Bool;
			uniform.vecsize = sourceType->second.bool_vecsize;
			uniform.columns = 1;
		} else if (sourceType->second.sampler_gl_type != 0 && uniform.kind == TranslatedUniform::Sampler) {
			uniform.sampler_gl_type = sourceType->second.sampler_gl_type;
			uniform.dim = TranslatedUniform::DimBuffer;
		}
	}
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
	static const std::regex outputRe(R"((^|\n)([ \t]*)(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bout[ \t]+(float|int|uint|bool|[biu]?vec[234]|mat[234](?:x[234])?)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:\[[ \t]*([0-9]+)[ \t]*\])?[ \t]*;)");
	static const std::regex inputRe(R"((^|\n)([ \t]*)(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bin[ \t]+(float|int|uint|bool|[biu]?vec[234]|mat[234](?:x[234])?)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:\[[ \t]*([0-9]+)[ \t]*\])?[ \t]*;)");
	const std::regex &re = storage[0] == 'o' ? outputRe : inputRe;
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
	static const std::regex outputRe(R"((^|[;\n])[ \t]*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bout\b[^;\n]*;)");
	static const std::regex inputRe(R"((^|[;\n])[ \t]*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bin\b[^;\n]*;)");
	const std::regex &re = storage[0] == 'o' ? outputRe : inputRe;
	return (size_t)std::distance(std::sregex_iterator(masked.begin(), masked.end(), re), std::sregex_iterator());
}

static bool hasLayoutQualifiedStorage(const std::string &masked, const char *storage) {
	static const std::regex outputRe(R"(layout[ \t]*\([^)]*\)[ \t]*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bout\b)");
	static const std::regex inputRe(R"(layout[ \t]*\([^)]*\)[ \t]*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)[ \t]+)*\bin\b)");
	return std::regex_search(masked, storage[0] == 'o' ? outputRe : inputRe);
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
		if (locationByName.find(v.name) != locationByName.end() || v.locationCount > UINT32_MAX - nextLocation) {
			return; // duplicate or unrepresentable interface: do not guess
		}
		const uint32_t location = nextLocation;
		nextLocation += v.locationCount;
		locationByName.emplace(v.name, location);
		interfaceByName.emplace(v.name, std::make_pair(v.type, v.locationCount));
		outputLocations.push_back(location);
	}
	for (const SimpleInterfaceVariable &v : inputs) {
		auto found = locationByName.find(v.name);
		auto interface = interfaceByName.find(v.name);
		if (found != locationByName.end() && interface != interfaceByName.end()) {
			if (interface->second.first != v.type || interface->second.second != v.locationCount) {
				return; // same-name type mismatch: do not silently connect incompatible varyings
			}
			inputLocations.push_back(found->second);
		} else {
			// A fragment input may be declared but not written by the vertex stage. Inactive
			// inputs are removed by SPIRV-Cross; assigning a private location here prevents
			// such a declaration from shifting every later, correctly matched varying.
			if (v.locationCount > UINT32_MAX - nextLocation) return;
			inputLocations.push_back(nextLocation);
			nextLocation += v.locationCount;
		}
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

static const std::unordered_set<std::string> &mslReservedIdentifiers() {
	static const std::unordered_set<std::string> reserved = {
		// C++ keywords used by Metal Shading Language.
		"alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case",
		"catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval",
		"constexpr", "constinit", "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default",
		"delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
		"for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not",
		"not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
		"reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast",
		"struct", "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
		"union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
		"module", "import", "final", "override",
		// Metal language qualifiers, stage keywords and Objective-C's id token.
		"kernel", "vertex", "fragment", "compute", "device", "constant", "threadgroup", "thread",
		"threadgroup_imageblock", "ray_data", "object_data", "visible_function_table", "intersection_function_table", "id",
		// Metal scalar/vector/matrix, resource and atomic type names.
		"half", "half2", "half3", "half4", "float2", "float3", "float4", "double2", "double3", "double4",
		"bool2", "bool3", "bool4", "char2", "char3", "char4", "uchar", "uchar2", "uchar3", "uchar4",
		"short2", "short3", "short4", "ushort", "ushort2", "ushort3", "ushort4", "int2", "int3", "int4",
		"uint", "uint2", "uint3", "uint4", "long2", "long3", "long4", "ulong", "ulong2", "ulong3", "ulong4",
		"float2x2", "float2x3", "float2x4", "float3x2", "float3x3", "float3x4", "float4x2", "float4x3", "float4x4",
		"half2x2", "half2x3", "half2x4", "half3x2", "half3x3", "half3x4", "half4x2", "half4x3", "half4x4",
		"packed_float2", "packed_float3", "packed_float4", "packed_half2", "packed_half3", "packed_half4",
		"atomic_int", "atomic_uint", "sampler", "texture1d", "texture2d", "texture3d", "texturecube",
		// MSL standard-library math, geometric, relational and packing functions.
		"abs", "acos", "acosh", "all", "any", "as_type", "asin", "asinh", "atan", "atan2", "atanh", "ceil",
		"clamp", "copysign", "cos", "cosh", "cospi", "cross", "degrees", "determinant", "distance", "distance_squared",
		"dot", "exp", "exp2", "exp10", "fabs", "faceforward", "fdim", "floor", "fma", "fmax", "fmin", "fmod",
		"fract", "frexp", "fwidth", "hypot", "ilogb", "inverse", "isfinite", "isinf", "isnan", "ldexp", "length",
		"length_squared", "log", "log2", "log10", "log1p", "logb", "mad", "max", "min", "mix", "modf", "normalize",
		"outer_product", "pow", "pown", "powr", "radians", "reflect", "refract", "remainder", "remquo", "rint",
		"round", "rsqrt", "select", "sign", "sin", "sincos", "sinh", "sinpi", "smoothstep", "sqrt", "step", "tan",
		"tanh", "tgamma", "transpose", "trunc", "bias"
	};
	return reserved;
}

static bool isReservedMslIdentifier(const std::string &name,
	const std::unordered_set<std::string> &reserved) {
	if (reserved.find(name) != reserved.end()) return true;
	// glslang may preserve a function's mangled signature in SPIR-V OpName,
	// e.g. "tanh(f1;". MSL needs the base identifier, not the mangled spelling.
	const size_t signatureStart = name.find('(');
	return signatureStart != std::string::npos &&
		reserved.find(name.substr(0, signatureStart)) != reserved.end();
}

static std::string mslIdentifierBase(const std::string &name) {
	const size_t signatureStart = name.find('(');
	return name.substr(0, signatureStart);
}

static std::string makeUniqueMslIdentifier(const std::string &oldName, uint32_t spirvSymbolId,
	std::unordered_set<std::string> &usedNames) {
	std::string baseName = "gl4metal_msl_" + oldName + "_" + std::to_string(spirvSymbolId);
	std::string candidate = baseName;
	uint32_t suffix = 1;
	while (usedNames.find(candidate) != usedNames.end()) {
		candidate = baseName + "_" + std::to_string(suffix++);
	}
	usedNames.insert(candidate);
	return candidate;
}

static void sanitizeMslStructTypeMembers(spirv_cross::CompilerMSL &compiler, spirv_cross::TypeID structTypeId,
	const std::unordered_set<std::string> &reserved, std::unordered_set<spirv_cross::TypeID> &visitedTypes,
	std::unordered_set<std::string> &usedNames, size_t &renamedCount) {
	if (!visitedTypes.insert(structTypeId).second) return;
	const spirv_cross::SPIRType &structType = compiler.get_type(structTypeId);
	if (structType.basetype != spirv_cross::SPIRType::Struct) return;
	for (uint32_t memberIndex = 0; memberIndex < structType.member_types.size(); ++memberIndex) {
		const std::string oldMemberName = compiler.get_member_name(structTypeId, memberIndex);
		if (!oldMemberName.empty() && reserved.find(oldMemberName) != reserved.end()) {
			const std::string newMemberName = makeUniqueMslIdentifier(oldMemberName,
				structTypeId ^ (memberIndex + 0x40000000u), usedNames);
			compiler.set_member_name(structTypeId, memberIndex, newMemberName);
			++renamedCount;
		}
		const spirv_cross::TypeID memberTypeId = structType.member_types[memberIndex];
		if (compiler.get_type(memberTypeId).basetype == spirv_cross::SPIRType::Struct) {
			sanitizeMslStructTypeMembers(compiler, memberTypeId, reserved, visitedTypes, usedNames, renamedCount);
		}
	}
}

template <typename ResourceList>
static void sanitizeMslStructResourceMembers(spirv_cross::CompilerMSL &compiler,
	const ResourceList &resourceList, const std::unordered_set<std::string> &reserved,
	std::unordered_set<spirv_cross::TypeID> &visitedTypes, std::unordered_set<std::string> &usedNames,
	size_t &renamedCount) {
	for (const spirv_cross::Resource &resource : resourceList) {
		const spirv_cross::SPIRType &resourceType = compiler.get_type(resource.base_type_id);
		if (resourceType.basetype == spirv_cross::SPIRType::Struct) {
			sanitizeMslStructTypeMembers(compiler, resource.base_type_id, reserved, visitedTypes, usedNames, renamedCount);
		}
	}
}

static size_t sanitizeMslIdentifiers(spirv_cross::CompilerMSL &compiler,
	const spirv_cross::ShaderResources &resources, const char *stageName, std::string &log) {
	const std::unordered_set<std::string> &reserved = mslReservedIdentifiers();
	std::unordered_set<std::string> usedNames;
	const uint32_t spirvIdBound = compiler.get_current_id_bound();
	for (uint32_t spirvSymbolId = 1; spirvSymbolId < spirvIdBound; ++spirvSymbolId) {
		const std::string symbolName = compiler.get_name(spirvSymbolId);
		if (!symbolName.empty()) usedNames.insert(symbolName);
	}

	size_t renamedCount = 0;
	for (uint32_t spirvSymbolId = 1; spirvSymbolId < spirvIdBound; ++spirvSymbolId) {
		const std::string symbolName = compiler.get_name(spirvSymbolId);
		if (symbolName.empty() || !isReservedMslIdentifier(symbolName, reserved)) continue;
		const std::string safeName = makeUniqueMslIdentifier(mslIdentifierBase(symbolName), spirvSymbolId, usedNames);
		compiler.set_name(spirvSymbolId, safeName);
		++renamedCount;
	}

	std::unordered_set<spirv_cross::TypeID> visitedTypes;
	// Resource reflection and GL-facing names have already been copied into TranslatedProgram.
	// Only SPIR-V OpMemberName values are changed here, immediately before MSL emission.
	sanitizeMslStructResourceMembers(compiler, resources.uniform_buffers, reserved, visitedTypes, usedNames, renamedCount);
	sanitizeMslStructResourceMembers(compiler, resources.storage_buffers, reserved, visitedTypes, usedNames, renamedCount);
	sanitizeMslStructResourceMembers(compiler, resources.stage_inputs, reserved, visitedTypes, usedNames, renamedCount);
	sanitizeMslStructResourceMembers(compiler, resources.stage_outputs, reserved, visitedTypes, usedNames, renamedCount);
	sanitizeMslStructResourceMembers(compiler, resources.push_constant_buffers, reserved, visitedTypes, usedNames, renamedCount);

	if (renamedCount != 0) {
		log += std::to_string(renamedCount) + " " + stageName +
			" MSL identifier(s) renamed to avoid Metal built-in/keyword collisions\n";
	}
	return renamedCount;
}

struct TranslationLookup {
	std::unordered_map<std::string, size_t> uniform_indices;
	std::unordered_map<std::string, size_t> block_indices;
	std::unordered_map<std::string, std::pair<size_t, size_t>> block_member_indices;
};

// SPIR-V locations emitted by glslang are not always stage-linked by name when a
// desktop shader contains compatibility-generated declarations plus conditional
// VSH/FSH blocks (OptiFine shaderpacks commonly do this). Metal requires the same
// [[user(locnN)]] index on a vertex output and its matching fragment input. Reflect
// the vertex interface once, then repair only the fragment-side decorations before
// SPIRV-Cross emits MSL. This avoids trying to parse preprocessor-inactive GLSL text.
struct StageVaryingInfo {
	uint32_t location = 0;
	uint32_t base_type = 0;
	uint32_t width = 0;
	uint32_t vecsize = 0;
	uint32_t columns = 0;
	spirv_cross::SmallVector<uint32_t> array;
	spirv_cross::SmallVector<bool> array_size_literal;
};

static std::string interfaceResourceName(const spirv_cross::CompilerMSL &compiler,
	const spirv_cross::Resource &resource) {
	if (!resource.name.empty()) return resource.name;
	return compiler.get_name(resource.id);
}

static bool collectVertexOutputLocations(const std::vector<unsigned int> &spirv,
	std::unordered_map<std::string, StageVaryingInfo> &locations, std::string &error) {
	try {
		spirv_cross::CompilerMSL reflection(spirv);
		const spirv_cross::ShaderResources resources = reflection.get_shader_resources();
		locations.reserve(resources.stage_outputs.size());
		for (const spirv_cross::Resource &resource : resources.stage_outputs) {
			if (reflection.has_decoration(resource.id, spv::DecorationBuiltIn) ||
				!reflection.has_decoration(resource.id, spv::DecorationLocation)) continue;
			const std::string name = interfaceResourceName(reflection, resource);
			if (name.empty()) continue;
			const spirv_cross::SPIRType &type = reflection.get_type(resource.type_id);
			if (type.basetype == spirv_cross::SPIRType::Struct) continue; // interface blocks require member-level matching
			StageVaryingInfo info;
			info.location = reflection.get_decoration(resource.id, spv::DecorationLocation);
			info.base_type = (uint32_t)type.basetype;
			info.width = type.width;
			info.vecsize = type.vecsize;
			info.columns = type.columns;
			info.array = type.array;
			info.array_size_literal = type.array_size_literal;
			const uint32_t outputLocation = info.location;
			auto inserted = locations.emplace(name, std::move(info));
			if (!inserted.second && inserted.first->second.location != outputLocation) {
				error = "duplicate vertex outputs named '" + name + "' have conflicting locations";
				return false;
			}
		}
	} catch (const std::exception &e) {
		error = std::string("vertex interface reflection failed: ") + e.what();
		return false;
	}
	return true;
}

static bool alignFragmentInputsToVertexOutputs(spirv_cross::CompilerMSL &compiler,
	const spirv_cross::ShaderResources &resources,
	const std::unordered_map<std::string, StageVaryingInfo> &vertexLocations,
	std::string &error, size_t &alignedCount) {
	alignedCount = 0;
	for (const spirv_cross::Resource &resource : resources.stage_inputs) {
		if (compiler.has_decoration(resource.id, spv::DecorationBuiltIn)) continue;
		const std::string name = interfaceResourceName(compiler, resource);
		auto vertex = vertexLocations.find(name);
		if (vertex == vertexLocations.end()) continue; // leave unused or unsupported inputs to normal link validation
		const spirv_cross::SPIRType &fragmentType = compiler.get_type(resource.type_id);
		if (fragmentType.basetype == spirv_cross::SPIRType::Struct) continue;
		const StageVaryingInfo &vertexInfo = vertex->second;
		
		auto isSameArray = [](const auto &a, const auto &b) {
			if (a.size() != b.size()) return false;
			for (size_t i = 0; i < a.size(); ++i) {
				if (static_cast<uint32_t>(a[i]) != static_cast<uint32_t>(b[i])) return false;
			}
			return true;
		};
		if ((uint32_t)fragmentType.basetype != vertexInfo.base_type ||
			fragmentType.width != vertexInfo.width || fragmentType.vecsize != vertexInfo.vecsize ||
			fragmentType.columns != vertexInfo.columns || !isSameArray(fragmentType.array, vertexInfo.array) ||
			!isSameArray(fragmentType.array_size_literal, vertexInfo.array_size_literal)) {
			error = "varying '" + name + "' has incompatible vertex/fragment types";
			return false;
		}
		const uint32_t currentLocation = compiler.has_decoration(resource.id, spv::DecorationLocation)
			? compiler.get_decoration(resource.id, spv::DecorationLocation) : UINT32_MAX;
		if (currentLocation != vertexInfo.location) {
			compiler.set_decoration(resource.id, spv::DecorationLocation, vertexInfo.location);
			++alignedCount;
		}
	}
	return true;
}

TranslatedUniform &findOrAdd(TranslatedProgram &out, const std::string &name, TranslationLookup &lookup) {
	auto found = lookup.uniform_indices.find(name);
	if (found != lookup.uniform_indices.end()) return out.uniforms[found->second];
	const size_t index = out.uniforms.size();
	out.uniforms.emplace_back();
	out.uniforms.back().name = name;
	lookup.uniform_indices.emplace(name, index);
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

// SPIRV-Cross identifies explicit MSL resource remaps by
// (execution stage, descriptor set, SPIR-V binding). Desktop GL sampler uniforms
// do not have to own unique descriptor bindings: their actual texture units are
// assigned later through glUniform1i. glslang can therefore emit multiple active
// resources with the same (set, binding), which makes SPIRV-Cross reuse one MSL
// texture/sampler location for otherwise distinct uniforms. Give only colliding
// resources a unique internal SPIR-V binding key before registering MSL remaps.
// This does not change GL texture units or the Metal slots stored in reflection.
static bool makeMslResourceBindingKeysUnique(
	spirv_cross::CompilerMSL &msl,
	const std::vector<const spirv_cross::Resource *> &mappedResources,
	const char *stageName,
	std::string &log) {
	auto makeKey = [](uint32_t descriptorSet, uint32_t binding) -> uint64_t {
		return (static_cast<uint64_t>(descriptorSet) << 32) | static_cast<uint64_t>(binding);
	};
	auto descriptorSetOf = [&msl](const spirv_cross::Resource &resource) -> uint32_t {
		return msl.has_decoration(resource.id, spv::DecorationDescriptorSet)
			? msl.get_decoration(resource.id, spv::DecorationDescriptorSet) : 0;
	};
	auto bindingOf = [&msl](const spirv_cross::Resource &resource) -> uint32_t {
		return msl.has_decoration(resource.id, spv::DecorationBinding)
			? msl.get_decoration(resource.id, spv::DecorationBinding) : 0;
	};

	std::unordered_set<uint32_t> seenResourceIds;
	std::unordered_set<uint64_t> seenKeys;
	uint32_t maximumBinding = 0;
	for (const spirv_cross::Resource *resource : mappedResources) {
		if (!resource || !seenResourceIds.insert(resource->id).second) continue;
		maximumBinding = std::max(maximumBinding, bindingOf(*resource));
	}

	uint64_t nextBinding = static_cast<uint64_t>(maximumBinding) + 1;
	seenResourceIds.clear();
	for (const spirv_cross::Resource *resource : mappedResources) {
		if (!resource || !seenResourceIds.insert(resource->id).second) continue;

		const uint32_t descriptorSet = descriptorSetOf(*resource);
		const uint32_t originalBinding = bindingOf(*resource);
		const uint64_t key = makeKey(descriptorSet, originalBinding);
		if (seenKeys.insert(key).second) continue;

		if (nextBinding > std::numeric_limits<uint32_t>::max()) {
			log += std::string(stageName) +
				": cannot create a unique internal SPIR-V resource binding (binding space exhausted)\n";
			return false;
		}

		const uint32_t remappedBinding = static_cast<uint32_t>(nextBinding++);
		msl.set_decoration(resource->id, spv::DecorationBinding, remappedBinding);
		seenKeys.insert(makeKey(descriptorSet, remappedBinding));

		const std::string resourceName = resource->name.empty() ? "<unnamed>" : resource->name;
		log += std::string(stageName) + ": duplicate SPIR-V resource key (set=" +
			std::to_string(descriptorSet) + ", binding=" + std::to_string(originalBinding) +
			") for '" + resourceName + "'; internal binding remapped to " +
			std::to_string(remappedBinding) + "\n";
	}
	return true;
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

struct RelaxedVaryingInfo {
	std::string name;
	uint32_t location = 0;
	uint32_t vecsize = 0;
};

static bool isHalfSafeVaryingName(const std::string &name) {
	std::string lower;
	lower.reserve(name.size());
	for (unsigned char ch : name) {
		lower.push_back((char)((ch >= 'A' && ch <= 'Z') ? ch - 'A' + 'a' : ch));
	}
	// Only compact normalized/interpolated data. Never narrow positions, depth,
	// distances, matrices, or arbitrary values that may contain HDR/world ranges.
	return lower.find("texcoord") != std::string::npos ||
		lower.find("lightcoord") != std::string::npos ||
		lower.find("lmcoord") != std::string::npos ||
		lower == "uv" || lower == "uv0" || lower == "uv1" || lower == "uv2" ||
		lower.find("normal") != std::string::npos ||
		lower.find("tangent") != std::string::npos;
}

static std::vector<RelaxedVaryingInfo> collectRelaxedVaryings(
	const std::vector<unsigned int> &spirv, bool collectInputs) {
	std::vector<RelaxedVaryingInfo> result;
	spirv_cross::CompilerMSL reflection(spirv);
	auto activeVariables = reflection.get_active_interface_variables();
	spirv_cross::ShaderResources resources = reflection.get_shader_resources(activeVariables);
	const auto &variables = collectInputs ? resources.stage_inputs : resources.stage_outputs;
	for (const spirv_cross::Resource &resource : variables) {
		const spirv_cross::SPIRType &type = reflection.get_type(resource.type_id);
		if (type.basetype != spirv_cross::SPIRType::Float || type.columns != 1 ||
			type.vecsize < 2 || type.vecsize > 4 || !isHalfSafeVaryingName(resource.name)) {
			continue;
		}
		if (!reflection.has_decoration(resource.id, spv::DecorationLocation)) continue;
		const bool relaxed =
			reflection.has_decoration(resource.id, spv::DecorationRelaxedPrecision) ||
			reflection.has_decoration(resource.type_id, spv::DecorationRelaxedPrecision) ||
			reflection.has_decoration(resource.base_type_id, spv::DecorationRelaxedPrecision);
		if (!relaxed) continue;
		RelaxedVaryingInfo info;
		info.name = resource.name;
		info.location = reflection.get_decoration(resource.id, spv::DecorationLocation);
		info.vecsize = type.vecsize;
		result.push_back(std::move(info));
	}
	return result;
}

static void buildHalfVaryingNameSets(
	const std::vector<unsigned int> &vertexSpirv,
	const std::vector<unsigned int> &fragmentSpirv,
	std::unordered_set<std::string> &vertexNames,
	std::unordered_set<std::string> &fragmentNames) {
	const auto vertexOutputs = collectRelaxedVaryings(vertexSpirv, false);
	const auto fragmentInputs = collectRelaxedVaryings(fragmentSpirv, true);
	std::unordered_map<std::string, const RelaxedVaryingInfo *> fragmentByName;
	fragmentByName.reserve(fragmentInputs.size());
	for (const RelaxedVaryingInfo &fragmentInput : fragmentInputs) {
		fragmentByName.emplace(fragmentInput.name, &fragmentInput);
	}
	for (const RelaxedVaryingInfo &vertexOutput : vertexOutputs) {
		auto matchingInput = fragmentByName.find(vertexOutput.name);
		if (matchingInput == fragmentByName.end()) continue;
		const RelaxedVaryingInfo &fragmentInput = *matchingInput->second;
		if (fragmentInput.location != vertexOutput.location || fragmentInput.vecsize != vertexOutput.vecsize) continue;
		vertexNames.insert(vertexOutput.name);
		fragmentNames.insert(fragmentInput.name);
	}
}

// A built-in redeclaration by itself is not a read. Ignore that declarator so only
// shaders that reference gl_FragCoord in an expression get the extra MSL parameter.
static bool usesFragmentFragCoord(const std::string &source) {
	std::string masked = maskShaderComments(source);
	static const std::regex redeclaration(
		R"((?:layout\s*\([^)]*\)\s*)?(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)\s+)*\bin\s+(?:(?:lowp|mediump|highp)\s+)?[A-Za-z_][A-Za-z0-9_]*\s+gl_FragCoord\s*;)");
	for (std::sregex_iterator it(masked.begin(), masked.end(), redeclaration), end; it != end; ++it) {
		const size_t start = (size_t)it->position();
		std::fill(masked.begin() + start, masked.begin() + start + it->length(), ' ');
	}
	const size_t identifierLength = sizeof("gl_FragCoord") - 1;
	for (size_t position = 0; (position = masked.find("gl_FragCoord", position)) != std::string::npos; ++position) {
		const bool leftBoundary = position == 0 || !isMslIdentifierChar(masked[position - 1]);
		const size_t end = position + identifierLength;
		const bool rightBoundary = end >= masked.size() || !isMslIdentifierChar(masked[end]);
		if (leftBoundary && rightBoundary) return true;
	}
	return false;
}

static bool declaresFragCoordOriginUpperLeft(const std::string &source) {
	const std::string masked = maskShaderComments(source);
	static const std::regex declaration(
		R"(\blayout\s*\([^)]*\borigin_upper_left\b[^)]*\)\s*(?:(?:flat|smooth|noperspective|centroid|sample|invariant|precise|highp|mediump|lowp)\s+)*\bin\b[^;]*\bgl_FragCoord\b)");
	return std::regex_search(masked, declaration);
}

// SPIRV-Cross maps the Vulkan fragment builtin to Metal's top-left [[position]] input.
// Rewrite only shaders that actually consume gl_FragCoord and use desktop GL's default
// lower-left convention. The render-target height and whether a conversion is needed
// for the current pass are supplied by the draw encoder, not baked into the shader.
static bool addFragmentFragCoordYFixup(std::string &source, std::string &error) {
	const std::string namespaceLine = "using namespace metal;";
	const std::string masked = maskShaderComments(source);
	const size_t namespacePosition = masked.find(namespaceLine);
	if (namespacePosition == std::string::npos) {
		error = "could not locate the MSL namespace declaration for fragment coordinate fixup";
		return false;
	}

	auto skipWhitespace = [&](size_t position) {
		while (position < masked.size() && (masked[position] == ' ' || masked[position] == '\t' ||
			masked[position] == '\r' || masked[position] == '\n')) ++position;
		return position;
	};

	size_t entryParen = std::string::npos;
	size_t fragmentPosition = 0;
	while ((fragmentPosition = masked.find("fragment", fragmentPosition)) != std::string::npos) {
		const size_t fragmentEnd = fragmentPosition + 8;
		const bool leftBoundary = fragmentPosition == 0 || !isMslIdentifierChar(masked[fragmentPosition - 1]);
		const bool rightBoundary = fragmentEnd >= masked.size() || !isMslIdentifierChar(masked[fragmentEnd]);
		if (!leftBoundary || !rightBoundary) {
			fragmentPosition = fragmentEnd;
			continue;
		}
		size_t tokenPosition = skipWhitespace(fragmentEnd);
		const size_t returnTypeStart = tokenPosition;
		while (tokenPosition < masked.size() && isMslIdentifierChar(masked[tokenPosition])) ++tokenPosition;
		if (tokenPosition == returnTypeStart) {
			fragmentPosition = fragmentEnd;
			continue;
		}
		tokenPosition = skipWhitespace(tokenPosition);
		const size_t functionNameStart = tokenPosition;
		while (tokenPosition < masked.size() && isMslIdentifierChar(masked[tokenPosition])) ++tokenPosition;
		const std::string functionName = masked.substr(functionNameStart, tokenPosition - functionNameStart);
		tokenPosition = skipWhitespace(tokenPosition);
		if (functionName == "main0" && tokenPosition < masked.size() && masked[tokenPosition] == '(') {
			entryParen = tokenPosition;
			break;
		}
		fragmentPosition = fragmentEnd;
	}
	if (entryParen == std::string::npos) {
		error = "could not locate the MSL fragment main0 entry point";
		return false;
	}

	const size_t signatureClose = matchingMslParen(masked, entryParen);
	if (signatureClose == std::string::npos) {
		error = "MSL fragment entry-point parameter list is unbalanced";
		return false;
	}
	const size_t positionAttribute = masked.find("[[position]]", entryParen);
	if (positionAttribute == std::string::npos || positionAttribute >= signatureClose) {
		error = "the active fragment entry point has no [[position]] input to rewrite";
		return false;
	}
	size_t nameEnd = positionAttribute;
	while (nameEnd > entryParen && (masked[nameEnd - 1] == ' ' || masked[nameEnd - 1] == '\t' ||
		masked[nameEnd - 1] == '\r' || masked[nameEnd - 1] == '\n')) --nameEnd;
	size_t nameStart = nameEnd;
	while (nameStart > entryParen && isMslIdentifierChar(masked[nameStart - 1])) --nameStart;
	if (nameStart == nameEnd) {
		error = "could not identify the MSL [[position]] fragment input";
		return false;
	}
	const std::string fragCoordName = masked.substr(nameStart, nameEnd - nameStart);

	size_t bodyOpen = signatureClose + 1;
	while (bodyOpen < masked.size() && masked[bodyOpen] != '{') ++bodyOpen;
	if (bodyOpen >= masked.size()) {
		error = "could not locate the MSL fragment entry-point body";
		return false;
	}

	std::string paramsType = "GL4MetalFragCoordParams";
	std::string paramsName = "gl4metal_FragCoordParams";
	for (uint32_t suffix = 1; source.find(paramsType) != std::string::npos || source.find(paramsName) != std::string::npos; ++suffix) {
		paramsType = "GL4MetalFragCoordParams" + std::to_string(suffix);
		paramsName = "gl4metal_FragCoordParams" + std::to_string(suffix);
	}

	size_t parameterStart = skipWhitespace(entryParen + 1);
	const bool hasParameters = parameterStart < signatureClose;
	const std::string parameter = std::string(hasParameters ? ", " : "") +
		"constant " + paramsType + "& " + paramsName + " [[buffer(" +
		std::to_string(GL4METAL_FRAGCOORD_BUFFER_INDEX) + ")]]";
	const std::string fixup = "\n    if (" + paramsName + ".flip_y != 0u) {\n" +
		"        " + fragCoordName + ".y = float(" + paramsName + ".target_height) - " + fragCoordName + ".y;\n" +
		"    }\n";
	const std::string paramsStruct = "\n\nstruct " + paramsType + " { uint target_height; uint flip_y; };\n";

	// Insert from right to left so the offsets obtained from the masked source stay valid.
	source.insert(bodyOpen + 1, fixup);
	source.insert(signatureClose, parameter);
	source.insert(namespacePosition + namespaceLine.size(), paramsStruct);
	return true;
}

static bool addVertexFlipFunctionConstant(std::string &source, std::string &error) {
	const std::string namespaceLine = "using namespace metal;";
	const size_t namespacePosition = source.find(namespaceLine);
	if (namespacePosition == std::string::npos) {
		error = "could not locate the MSL namespace declaration for vertex specialization";
		return false;
	}
	if (source.find("gl4metal_FlipY") != std::string::npos) {
		error = "MSL source already contains the reserved Y-flip specialization name";
		return false;
	}
	const size_t declarationPosition = namespacePosition + namespaceLine.size();
	source.insert(declarationPosition,
		"\n\nconstant bool gl4metal_FlipY [[function_constant(0)]];\n");

	// SPIRV-Cross emits a return-structure type between the MSL stage keyword and
	// the entry-point name, for example `vertex main0_out main0(...)`. Do not
	// assume the invalid `vertex main0(` spelling: simple Minecraft shaders use
	// a generated `main0_out` type and failed during v15 startup because of it.
	auto skipWhitespace = [&](size_t position) {
		while (position < source.size() && (source[position] == ' ' || source[position] == '\t' ||
			source[position] == '\r' || source[position] == '\n')) ++position;
		return position;
	};
	const std::string masked = maskShaderComments(source);
	size_t vertexPosition = 0;
	size_t entryParen = std::string::npos;
	while ((vertexPosition = masked.find("vertex", vertexPosition)) != std::string::npos) {
		const size_t vertexEnd = vertexPosition + 6;
		const bool leftBoundary = vertexPosition == 0 || !isMslIdentifierChar(masked[vertexPosition - 1]);
		const bool rightBoundary = vertexEnd >= masked.size() || !isMslIdentifierChar(masked[vertexEnd]);
		if (!leftBoundary || !rightBoundary) {
			vertexPosition = vertexEnd;
			continue;
		}
		size_t tokenPosition = skipWhitespace(vertexEnd);
		// Parse the generated MSL return type (main0_out, void, etc.).
		const size_t returnTypeStart = tokenPosition;
		while (tokenPosition < masked.size() && isMslIdentifierChar(masked[tokenPosition])) ++tokenPosition;
		if (tokenPosition == returnTypeStart) {
			vertexPosition = vertexEnd;
			continue;
		}
		tokenPosition = skipWhitespace(tokenPosition);
		const size_t functionNameStart = tokenPosition;
		while (tokenPosition < masked.size() && isMslIdentifierChar(masked[tokenPosition])) ++tokenPosition;
		const std::string functionName = masked.substr(functionNameStart, tokenPosition - functionNameStart);
		tokenPosition = skipWhitespace(tokenPosition);
		if (functionName == "main0" && tokenPosition < masked.size() && masked[tokenPosition] == '(') {
			entryParen = tokenPosition;
			break;
		}
		vertexPosition = vertexEnd;
	}
	if (entryParen == std::string::npos) {
		error = "could not locate the MSL vertex entry point (expected a vertex main0 function)";
		return false;
	}
	const size_t signatureClose = matchingMslParen(masked, entryParen);
	if (signatureClose == std::string::npos) {
		error = "MSL vertex entry-point parameter list is unbalanced";
		return false;
	}
	size_t bodyOpen = signatureClose + 1;
	while (bodyOpen < masked.size() && masked[bodyOpen] != '{') ++bodyOpen;
	if (bodyOpen >= masked.size()) {
		error = "could not locate the MSL vertex entry-point body";
		return false;
	}
	int depth = 0;
	size_t bodyClose = std::string::npos;
	for (size_t i = bodyOpen; i < masked.size(); ++i) {
		if (masked[i] == '{') ++depth;
		else if (masked[i] == '}' && --depth == 0) { bodyClose = i; break; }
	}
	if (bodyClose == std::string::npos) {
		error = "MSL vertex entry-point braces are unbalanced";
		return false;
	}

	// A vertex-stage output struct is normally returned through a local (usually
	// `out`). Parse the returned identifier instead of hard-coding its spelling.
	struct ReturnPatch { size_t position; size_t length; std::string expression; };
	std::vector<ReturnPatch> returnPatches;
	for (size_t searchPosition = bodyOpen + 1; searchPosition < bodyClose; ) {
		const size_t returnPosition = masked.find("return", searchPosition);
		if (returnPosition == std::string::npos || returnPosition >= bodyClose) break;
		const size_t returnEnd = returnPosition + 6;
		const bool leftBoundary = returnPosition == 0 || !isMslIdentifierChar(masked[returnPosition - 1]);
		const bool rightBoundary = returnEnd >= masked.size() || !isMslIdentifierChar(masked[returnEnd]);
		if (!leftBoundary || !rightBoundary) {
			searchPosition = returnEnd;
			continue;
		}
		size_t expressionStart = skipWhitespace(returnEnd);
		size_t expressionEnd = expressionStart;
		while (expressionEnd < bodyClose && isMslIdentifierChar(masked[expressionEnd])) ++expressionEnd;
		size_t semicolon = skipWhitespace(expressionEnd);
		if (expressionEnd == expressionStart || semicolon >= bodyClose || masked[semicolon] != ';') {
			searchPosition = returnEnd;
			continue;
		}
		returnPatches.push_back({returnPosition, semicolon + 1 - returnPosition,
			masked.substr(expressionStart, expressionEnd - expressionStart)});
		searchPosition = semicolon + 1;
	}
	if (returnPatches.empty()) {
		error = "could not locate a return of the MSL vertex output structure";
		return false;
	}
	// Process backwards so replacements do not invalidate remaining offsets.
	for (auto it = returnPatches.rbegin(); it != returnPatches.rend(); ++it) {
		const std::string &outputName = it->expression;
		const std::string specializedReturn = "if (gl4metal_FlipY) { " + outputName +
			".gl_Position.y = -" + outputName + ".gl_Position.y; }\n    return " + outputName + ";";
		source.replace(it->position, it->length, specializedReturn);
	}
	return true;
}

// Metal requires a fragment color output to provide enough components for the
// render-target pixel format. Desktop GL permits scalar/vec2/vec3 user outputs;
// widen those MSL fields to vec4 and explicitly supply the missing components.
// The missing-channel defaults are (0, 0, 0, 1), keeping alpha opaque for RGB
// outputs and preserving normal GUI/text blending.
static bool mslFragmentOutputScalarType(const spirv_cross::SPIRType &type, std::string &baseType) {
	switch (type.basetype) {
		case spirv_cross::SPIRType::Float: baseType = "float"; return type.width == 32;
		case spirv_cross::SPIRType::Half:  baseType = "half";  return type.width == 16;
		case spirv_cross::SPIRType::Int:   baseType = "int";   return type.width == 32;
		case spirv_cross::SPIRType::UInt:  baseType = "uint";  return type.width == 32;
		default: return false;
	}
}

static std::string mslOutputComponentConstant(const std::string &baseType, bool one) {
	if (baseType == "float") return one ? "1.0" : "0.0";
	if (baseType == "half") return one ? "half(1.0)" : "half(0.0)";
	if (baseType == "uint") return one ? "1u" : "0u";
	return one ? "1" : "0";
}

static size_t widenMslFragmentColorOutputs(std::string &source,
	const spirv_cross::CompilerMSL &compiler, const spirv_cross::ShaderResources &resources,
	std::string &notes) {
	struct OutputPatch {
		std::string fieldName;
		std::string baseType;
		uint32_t width = 0;
		uint32_t location = 0;
	};
	std::vector<OutputPatch> patches;
	patches.reserve(resources.stage_outputs.size());

	for (const spirv_cross::Resource &resource : resources.stage_outputs) {
		if (compiler.has_decoration(resource.id, spv::DecorationBuiltIn) ||
			!compiler.has_decoration(resource.id, spv::DecorationLocation)) continue;
		const spirv_cross::SPIRType &type = compiler.get_type(resource.type_id);
		if (type.columns != 1 || !type.array.empty() || type.vecsize == 0 || type.vecsize >= 4) continue;
		std::string baseType;
		if (!mslFragmentOutputScalarType(type, baseType)) continue;
		std::string fieldName = compiler.get_name(resource.id);
		if (fieldName.empty()) fieldName = resource.name;
		if (fieldName.empty()) continue;
		patches.push_back({fieldName, baseType, type.vecsize,
			compiler.get_decoration(resource.id, spv::DecorationLocation)});
	}

	if (patches.empty()) return 0;

	// SPIRV-Cross's MSL entry point returns a main0_out structure initialized as
	// `main0_out out = {};`. Seed each widened field with GL's missing-component
	// defaults before any whole-vector or component-wise stores. This also makes
	// partial stores deterministic without changing shader control flow.
	const size_t outputStructPosition = source.find("main0_out main0(");
	if (outputStructPosition == std::string::npos) return 0;
	const size_t functionBodyStart = source.find('{', outputStructPosition);
	if (functionBodyStart == std::string::npos) return 0;
	const std::string outputInit = "main0_out out = {};";
	const size_t outputInitPosition = source.find(outputInit, functionBodyStart);
	if (outputInitPosition == std::string::npos) return 0;

	std::vector<std::string> initializers;
	initializers.reserve(patches.size());
	size_t widened = 0;
	for (const OutputPatch &patch : patches) {
		const std::string colorTag = "[[color(" + std::to_string(patch.location) + ")";
		const size_t tagPosition = source.find(colorTag);
		if (tagPosition == std::string::npos) continue;
		const size_t lineStartFound = source.rfind('\n', tagPosition);
		const size_t declarationStart = lineStartFound == std::string::npos ? 0 : lineStartFound + 1;
		std::string declaration = source.substr(declarationStart, tagPosition - declarationStart);
		while (!declaration.empty() && (declaration.back() == ' ' || declaration.back() == '\t' || declaration.back() == '\r')) declaration.pop_back();

		const size_t nameEnd = declaration.size();
		size_t nameStart = nameEnd;
		while (nameStart > 0 && (std::isalnum((unsigned char)declaration[nameStart - 1]) || declaration[nameStart - 1] == '_')) --nameStart;
		if (nameStart == nameEnd) continue;
		const std::string emittedFieldName = declaration.substr(nameStart, nameEnd - nameStart);
		if (emittedFieldName != patch.fieldName) continue;
		size_t typeEnd = nameStart;
		while (typeEnd > 0 && std::isspace((unsigned char)declaration[typeEnd - 1])) --typeEnd;
		size_t typeStart = typeEnd;
		while (typeStart > 0 && (std::isalnum((unsigned char)declaration[typeStart - 1]) || declaration[typeStart - 1] == '_')) --typeStart;
		if (typeStart == typeEnd) continue;
		const std::string oldType = declaration.substr(typeStart, typeEnd - typeStart);
		const std::string expectedType = patch.width == 1
			? patch.baseType : patch.baseType + std::to_string(patch.width);
		if (oldType != expectedType) continue;

		// Replace only the MSL output field's type token.
		const size_t absoluteTypeStart = declarationStart + typeStart;
		source.replace(absoluteTypeStart, oldType.size(), patch.baseType + "4");

		// The output variable is readable in GLSL. Once its MSL storage is widened,
		// every whole-vector READ must still have the original width (for example,
		// `out.sky_map * float3` must become `out.sky_map.xyz * float3`). Writes are
		// handled separately: whole writes are widened below, while compound writes
		// target only the original-width swizzle and leave the default alpha intact.
		const size_t currentOutputInitPosition = source.find(outputInit);
		if (currentOutputInitPosition == std::string::npos) continue;
		const std::string assignmentPrefix = "out." + patch.fieldName;
		const std::string maskedForReads = maskShaderComments(source);
		const size_t refreshedOutputStructPosition = source.find("main0_out main0(");
		const size_t functionBodyOpen = refreshedOutputStructPosition == std::string::npos ?
			std::string::npos : maskedForReads.find('{', refreshedOutputStructPosition);
		if (functionBodyOpen == std::string::npos) continue;
		int bodyDepth = 0;
		size_t functionBodyClose = std::string::npos;
		for (size_t bodyPos = functionBodyOpen; bodyPos < maskedForReads.size(); ++bodyPos) {
			if (maskedForReads[bodyPos] == '{') ++bodyDepth;
			else if (maskedForReads[bodyPos] == '}' && --bodyDepth == 0) {
				functionBodyClose = bodyPos;
				break;
			}
		}
		std::vector<size_t> readWidthInsertions;
		const std::string originalWidthSwizzle = patch.width == 1 ? ".x" :
			(patch.width == 2 ? ".xy" : ".xyz");
		for (size_t readPos = currentOutputInitPosition + outputInit.size();
			functionBodyClose != std::string::npos &&
			(readPos = maskedForReads.find(assignmentPrefix, readPos)) != std::string::npos &&
			readPos < functionBodyClose; ) {
			const bool leftBoundary = readPos == 0 ||
				(!isMslIdentifierChar(maskedForReads[readPos - 1]) && maskedForReads[readPos - 1] != '.');
			const size_t afterName = readPos + assignmentPrefix.size();
			if (!leftBoundary || (afterName < functionBodyClose && isMslIdentifierChar(maskedForReads[afterName]))) {
				readPos = afterName;
				continue;
			}
			size_t nextToken = afterName;
			while (nextToken < functionBodyClose && std::isspace((unsigned char)maskedForReads[nextToken])) ++nextToken;
			if (nextToken < functionBodyClose && (maskedForReads[nextToken] == '.' || maskedForReads[nextToken] == '[')) {
				readPos = afterName; // already component/index access
				continue;
			}
			if (nextToken < functionBodyClose && maskedForReads[nextToken] == '=' &&
				(nextToken + 1 >= functionBodyClose || maskedForReads[nextToken + 1] != '=')) {
				readPos = afterName; // plain whole-field assignment; widen its RHS below
				continue;
			}
			// Compound assignment reads and writes the old-width value. Putting the
			// swizzle on its LHS keeps the fourth output component untouched. Ordinary
			// expression uses get the same swizzle as a read.
			readWidthInsertions.push_back(afterName);
			readPos = afterName;
		}
		for (auto insertion = readWidthInsertions.rbegin(); insertion != readWidthInsertions.rend(); ++insertion) {
			source.insert(*insertion, originalWidthSwizzle);
		}

		// Widen all whole-field stores. Component stores need no conversion; the
		// initializer inserted below supplies their untouched channels. Reads in RHS
		// expressions now carry the original-width swizzle, preventing float4/float3
		// mixtures such as `float4((out.sky_map * fog[1]) + fog[0], 1.0)`.
		const size_t refreshedOutputInitPosition = source.find(outputInit);
		size_t searchPosition = refreshedOutputInitPosition == std::string::npos ? 0 :
			refreshedOutputInitPosition + outputInit.size();
		bool wholeStoreFound = false;
		while ((searchPosition = source.find(assignmentPrefix, searchPosition)) != std::string::npos) {
			const size_t afterName = searchPosition + assignmentPrefix.size();
			if (afterName < source.size() && (std::isalnum((unsigned char)source[afterName]) ||
				source[afterName] == '_' || source[afterName] == '[' || source[afterName] == '.')) {
				searchPosition = afterName;
				continue;
			}
			size_t equalPosition = afterName;
			while (equalPosition < source.size() && std::isspace((unsigned char)source[equalPosition])) ++equalPosition;
			if (equalPosition >= source.size() || source[equalPosition] != '=' ||
				(equalPosition + 1 < source.size() && source[equalPosition + 1] == '=')) {
				searchPosition = afterName;
				continue;
			}
			const size_t expressionStart = equalPosition + 1;
			const size_t semicolon = source.find(';', expressionStart);
			if (semicolon == std::string::npos) break;
			std::string expression = source.substr(expressionStart, semicolon - expressionStart);
			while (!expression.empty() && std::isspace((unsigned char)expression.front())) expression.erase(expression.begin());
			while (!expression.empty() && std::isspace((unsigned char)expression.back())) expression.pop_back();
			if (expression.empty()) {
				searchPosition = semicolon + 1;
				continue;
			}
			std::string widenedExpression = patch.baseType + "4((" + expression + ")";
			for (uint32_t component = patch.width; component < 4; ++component) {
				widenedExpression += ", " + mslOutputComponentConstant(patch.baseType, component == 3);
			}
			widenedExpression += ")";
			source.replace(expressionStart, semicolon - expressionStart, " " + widenedExpression);
			searchPosition = expressionStart + widenedExpression.size() + 1;
			wholeStoreFound = true;
		}

		std::string defaultValue = patch.baseType + "4(";
		for (uint32_t component = 0; component < 4; ++component) {
			if (component != 0) defaultValue += ", ";
			defaultValue += mslOutputComponentConstant(patch.baseType, component == 3);
		}
		defaultValue += ")";
		initializers.push_back("\n    out." + patch.fieldName + " = " + defaultValue + ";");
		++widened;
		notes += "fragment output '" + patch.fieldName + "' at location " +
			std::to_string(patch.location) + " widened from " + expectedType + " to " +
			patch.baseType + "4; whole-vector stores=" + (wholeStoreFound ? "rewritten" : "none") +
			", partial stores retain GL defaults\n";
	}

	if (widened != 0) {
		std::string initializerBlock;
		for (const std::string &initializer : initializers) initializerBlock += initializer;
		const size_t finalOutputInitPosition = source.find(outputInit);
		if (finalOutputInitPosition != std::string::npos) {
			source.insert(finalOutputInitPosition + outputInit.size(), initializerBlock);
		}
	}
	return widened;
}

// The vertex stage is compiled once. A function constant supplies the render-to-texture
// Y-flip variant at function-specialization time, so no runtime branch is left in the shader.
bool compileStage(std::vector<unsigned int> spirv, spv::ExecutionModel model, int stage,
	const std::unordered_set<std::string> &halfVaryingNames, bool fixFragmentFragCoordY,
	const std::unordered_map<std::string, StageVaryingInfo> &vertexOutputLocations,
	bool alignFragmentInputs, TranslatedProgram &out, TranslationLookup &lookup) {
	try {
		spirv_cross::CompilerMSL msl(std::move(spirv));

		spirv_cross::CompilerMSL::Options options;
		options.platform = spirv_cross::CompilerMSL::Options::iOS;
		options.set_msl_version(2, 4);
		msl.set_msl_options(options);

		spirv_cross::CompilerGLSL::Options common = msl.get_common_options();
		common.vertex.fixup_clipspace = true; // GL clip space z is [-w, w], Metal's is [0, w]
		common.vertex.flip_vert_y = false;
		msl.set_common_options(common);

		// Only varyings explicitly marked RelaxedPrecision in BOTH stages are
		// lowered to half. Uniform blocks, positions, depth, and unqualified desktop
		// GLSL values retain their original 32-bit representation and std140 layout.
		if (!halfVaryingNames.empty()) {
			msl.set_variable_type_remap_callback([&halfVaryingNames](
				const spirv_cross::SPIRType &type, const std::string &name, std::string &replacementType) {
				bool isRelaxedVarying = halfVaryingNames.find(name) != halfVaryingNames.end();
				if (!isRelaxedVarying && name.compare(0, 13, "gl4metal_msl_") == 0) {
					const std::string renamedSuffix = name.substr(13);
					for (const std::string &originalName : halfVaryingNames) {
						if (renamedSuffix.compare(0, originalName.size(), originalName) == 0 &&
							renamedSuffix.size() > originalName.size() && renamedSuffix[originalName.size()] == '_') {
							isRelaxedVarying = true;
							break;
						}
					}
				}
				if (!isRelaxedVarying || type.basetype != spirv_cross::SPIRType::Float || type.columns != 1 ||
					type.vecsize < 2 || type.vecsize > 4) return;
				replacementType = type.vecsize == 2 ? "half2" : (type.vecsize == 3 ? "half3" : "half4");
			});
		}

		// Reflect only resources statically accessed by this entry point. Iris injects
		// shared sampler declarations into both stages, even when a stage does not use
		// them. Reflecting every declaration marks those inactive samplers as present,
		// which causes needless Metal bindings and can trigger false feedback-loop skips.
		auto activeVariables = msl.get_active_interface_variables();
		spirv_cross::ShaderResources resources = msl.get_shader_resources(activeVariables);

		// Uniform buffers and sampled images share SPIR-V's (set, binding) key
		// space even though Metal places them in different resource namespaces.
		// OpenGL sampler uniforms may carry the same default binding, so uniquify
		// all resources participating in the explicit SPIRV-Cross remap table.
		std::vector<const spirv_cross::Resource *> mappedResources;
		mappedResources.reserve(resources.uniform_buffers.size() + resources.sampled_images.size());
		for (const spirv_cross::Resource &resource : resources.uniform_buffers) mappedResources.push_back(&resource);
		for (const spirv_cross::Resource &resource : resources.sampled_images) mappedResources.push_back(&resource);
		if (!makeMslResourceBindingKeysUnique(msl, mappedResources,
			stage == VS ? "vertex" : "fragment", out.log)) return false;

		if (stage == FS && alignFragmentInputs && !vertexOutputLocations.empty()) {
			size_t alignedCount = 0;
			std::string interfaceError;
			if (!alignFragmentInputsToVertexOutputs(msl, resources, vertexOutputLocations, interfaceError, alignedCount)) {
				out.log += "stage interface: " + interfaceError + "\n";
				return false;
			}
			if (alignedCount != 0) {
				out.log += "stage interface: aligned " + std::to_string(alignedCount) +
					" fragment varying location(s) to vertex outputs by name\n";
			}
		}
		msl.set_enabled_interface_variables(std::move(activeVariables));

		// Uniform blocks: GL binding points are an independent namespace (36 on this backend).
		// Metal buffer indices are per-stage slots: 16...29 for up to 14 active UBOs, and 30
		// for GL's default uniform block. Do not derive the MSL slot from the GL binding point.
		uint32_t nextUniformBlockBufferSlot = 16;
		for (const spirv_cross::Resource &r : resources.uniform_buffers) {
			const spirv_cross::SPIRType &type = msl.get_type(r.base_type_id);
			bool isDefault = msl.get_name(r.base_type_id) == "gl_DefaultUniformBlock" || r.name == "gl_DefaultUniformBlock";
			uint32_t binding = msl.get_decoration(r.id, spv::DecorationBinding);

			spirv_cross::MSLResourceBinding b;
			b.stage = model;
			b.desc_set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
			b.binding = binding;
			if (isDefault) {
				b.msl_buffer = GL4METAL_UNIFORM_BUFFER_INDEX;
			} else {
				if (nextUniformBlockBufferSlot >= GL4METAL_UNIFORM_BUFFER_INDEX) {
					out.log += "shader stage uses more than 14 active uniform blocks; Metal buffer slots are exhausted\n";
					return false;
				}
				b.msl_buffer = nextUniformBlockBufferSlot++;
			}
			msl.add_msl_resource_binding(b);

			if (!isDefault) {
				std::string blockName = msl.get_name(r.base_type_id);
				if (blockName.empty()) {
					blockName = r.name;
				}
				TranslatedBlock *block = nullptr;
				auto blockFound = lookup.block_indices.find(blockName);
				if (blockFound == lookup.block_indices.end()) {
					const size_t blockIndex = out.blocks.size();
					out.blocks.emplace_back();
					out.blocks.back().name = blockName;
					lookup.block_indices.emplace(blockName, blockIndex);
					block = &out.blocks.back();
				} else {
					block = &out.blocks[blockFound->second];
				}
				size_t blockSize = msl.get_declared_struct_size(type);
				// GL_UNIFORM_BLOCK_DATA_SIZE includes the std140 structure's tail padding.
				// Keep that spec-visible size for glGetActiveUniformBlockiv, but also
				// retain the last-member extent. Some Minecraft 1.21.11 UBO writers bind
				// ranges ending exactly at the final field (140/40/12 bytes) rather than
				// including the trailing std140 padding (144/48/16). Metal binds the
				// backing MTLBuffer at an offset, so allowing that padding-only shortfall
				// is safe as long as the complete member extent remains in the allocation.
				const uint32_t memberExtent = (uint32_t)blockSize;
				const uint32_t std140Size = (uint32_t)((blockSize + 15) & ~(size_t)15);
				block->size = std::max(block->size, std140Size);
				block->required_size = std::max(block->required_size, memberExtent);
				block->stage[stage].present = true;
				block->stage[stage].slot = b.msl_buffer;

				// Reflect active members of named blocks for the GL 3.3 introspection API.
				const std::string memberPrefix = r.name.empty() ? blockName : r.name;
				for (uint32_t memberIndex = 0; memberIndex < type.member_types.size(); ++memberIndex) {
					const spirv_cross::SPIRType &memberType = msl.get_type(type.member_types[memberIndex]);
					const std::string memberLeaf = msl.get_member_name(r.base_type_id, memberIndex);
					if (memberType.basetype == spirv_cross::SPIRType::Struct) {
						out.log += "uniform-block member '" + memberLeaf + "' is a nested struct; nested member reflection is not supported\n";
						continue;
					}
					const std::string memberName = memberPrefix + "." + memberLeaf;
					auto memberFound = lookup.block_member_indices.find(memberName);
					TranslatedUniform *member = nullptr;
					if (memberFound == lookup.block_member_indices.end()) {
						const size_t blockIndex = (size_t)(block - out.blocks.data());
						const size_t memberIndex = block->members.size();
						block->members.emplace_back();
						block->members.back().name = memberName;
						lookup.block_member_indices.emplace(memberName, std::make_pair(blockIndex, memberIndex));
						member = &block->members.back();
					} else {
						member = &out.blocks[memberFound->second.first].members[memberFound->second.second];
					}
					member->kind = kindOf(memberType.basetype);
					member->vecsize = (int)memberType.vecsize;
					member->columns = (int)memberType.columns;
					member->array_size = arraySizeOf(memberType);
					TranslatedUniform::Stage &memberStage = member->stage[stage];
					memberStage.present = true;
					memberStage.offset = msl.type_struct_member_offset(type, memberIndex);
					memberStage.array_stride = memberType.array.empty() ? 0 : msl.type_struct_member_array_stride(type, memberIndex);
					memberStage.matrix_stride = memberType.columns > 1 ? msl.type_struct_member_matrix_stride(type, memberIndex) : 0;
					memberStage.row_major = msl.has_member_decoration(r.base_type_id, memberIndex, spv::DecorationRowMajor);
				}
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
				TranslatedUniform &u = findOrAdd(out, name, lookup);
				u.kind = kindOf(member.basetype);
				u.vecsize = (int)member.vecsize;
				u.columns = (int)member.columns;
				u.array_size = arraySizeOf(member);

				TranslatedUniform::Stage &st = u.stage[stage];
				st.present = true;
				st.offset = msl.type_struct_member_offset(type, i);
				st.array_stride = member.array.empty() ? 0 : msl.type_struct_member_array_stride(type, i);
				st.matrix_stride = member.columns > 1 ? msl.type_struct_member_matrix_stride(type, i) : 0;
				st.row_major = msl.has_member_decoration(r.base_type_id, i, spv::DecorationRowMajor);
			}
		}

		// Combined image samplers use one dense, stage-local texture/sampler slot.
		// The GL texture unit is only a lookup key into bound_textures; it is not a
		// Metal binding index. Keep the per-stage Metal slot budget at 16.
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
			if (count <= 0) {
				out.log += "sampler '" + r.name + "' has an invalid array size\n";
				return false;
			}

			spirv_cross::MSLResourceBinding b;
			b.stage = model;
			b.desc_set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
			b.binding = entry.first;
			b.msl_texture = slot;
			b.msl_sampler = slot;
			b.count = (uint32_t)count;
			msl.add_msl_resource_binding(b);

			TranslatedUniform &u = findOrAdd(out, r.name, lookup);
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
		const uint32_t stageTextureLimit = stage == VS
			? GL4METAL_MAX_VERTEX_TEXTURE_UNITS : GL4METAL_MAX_FRAGMENT_TEXTURE_UNITS;
		if (slot > stageTextureLimit) {
			out.log += "shader stage exceeds its Metal texture/sampler slot limit (" +
			std::to_string(stageTextureLimit) + ")\n";
			return false;
		}

		if (stage == VS) {
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

		sanitizeMslIdentifiers(msl, resources, stage == VS ? "vertex" : "fragment", out.log);
		std::string compiledMsl = msl.compile();
		if (stage == FS) {
			std::string outputNotes;
			const size_t widenedOutputs = widenMslFragmentColorOutputs(compiledMsl, msl, resources, outputNotes);
			if (widenedOutputs != 0) {
				out.log += "fragment output compatibility: widened " + std::to_string(widenedOutputs) +
					" sub-vec4 output(s) for Metal RGBA attachments\n" + outputNotes;
			}
		}
		const size_t clampedReads = clampMslTexture2DReadCoordinates(compiledMsl);
		if (clampedReads != 0) {
			out.log += std::to_string(clampedReads) + " out-of-range-safe texture2d.read coordinate guard(s) added to " +
				(stage == VS ? "vertex" : "fragment") + " MSL stage\n";
		}
		if (stage == FS && fixFragmentFragCoordY) {
			std::string fixupError;
			if (addFragmentFragCoordYFixup(compiledMsl, fixupError)) {
				out.uses_gl_frag_coord = true;
				out.log += "desktop GL gl_FragCoord Y conversion enabled for the fragment stage\n";
			} else {
				out.log += "gl_FragCoord Y conversion not injected: " + fixupError + "\n";
			}
		}
		if (stage == VS) {
			std::string specializationError;
			if (!addVertexFlipFunctionConstant(compiledMsl, specializationError)) {
				out.log += "vertex Y-flip specialization: " + specializationError + "\n";
				return false;
			}
		}
		out.msl[stage] = std::move(compiledMsl);
		return true;
	} catch (const std::exception &e) {
		out.log += std::string(stage == VS ? "vertex" : "fragment") + " stage, SPIRV-Cross: " + e.what() + "\n";
		return false;
	}
}

struct ComputeSourceBindings {
	std::unordered_map<std::string, uint32_t> uniformBlocks;
	std::unordered_map<std::string, uint32_t> storageBlocks;
	std::unordered_map<std::string, uint32_t> storageImages;
};

static bool parseLayoutBinding(const std::string &layout, uint32_t &bindingValue) {
	static const std::regex bindingPattern(R"(\bbinding\s*=\s*([0-9]+))");
	std::smatch match;
	if (!std::regex_search(layout, match, bindingPattern)) return false;
	try {
		bindingValue = (uint32_t)std::stoul(match[1].str());
		return true;
	} catch (...) {
		return false;
	}
}

static ComputeSourceBindings collectComputeSourceBindings(const std::string &source) {
	ComputeSourceBindings result;
	const std::string masked = maskShaderComments(source);
	static const std::regex uniformBlockPattern(
		R"(layout\s*\(([^)]*)\)\s*uniform\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{)");
	static const std::regex storageBlockPattern(
		R"(layout\s*\(([^)]*)\)\s*(?:(?:readonly|writeonly|coherent|volatile|restrict)\s+)*buffer\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{)");
	static const std::regex storageImagePattern(
		R"(layout\s*\(([^)]*)\)\s*(?:(?:readonly|writeonly|coherent|volatile|restrict)\s+)*uniform\s+(?:(?:readonly|writeonly|coherent|volatile|restrict)\s+)*[iu]?image[A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*))");
	for (std::sregex_iterator it(masked.begin(), masked.end(), uniformBlockPattern), end; it != end; ++it) {
		uint32_t bindingValue = 0;
		if (parseLayoutBinding((*it)[1].str(), bindingValue)) result.uniformBlocks[(*it)[2].str()] = bindingValue;
	}
	for (std::sregex_iterator it(masked.begin(), masked.end(), storageBlockPattern), end; it != end; ++it) {
		uint32_t bindingValue = 0;
		if (parseLayoutBinding((*it)[1].str(), bindingValue)) result.storageBlocks[(*it)[2].str()] = bindingValue;
	}
	for (std::sregex_iterator it(masked.begin(), masked.end(), storageImagePattern), end; it != end; ++it) {
		uint32_t bindingValue = 0;
		if (parseLayoutBinding((*it)[1].str(), bindingValue)) result.storageImages[(*it)[2].str()] = bindingValue;
	}
	return result;
}

static uint32_t glImageFormatForSpirv(spv::ImageFormat format) {
	switch (format) {
		case spv::ImageFormatRgba32f: return GL_RGBA32F;
		case spv::ImageFormatRgba16f: return GL_RGBA16F;
		case spv::ImageFormatR32f: return GL_R32F;
		case spv::ImageFormatRgba8: return GL_RGBA8;
		case spv::ImageFormatRgba8Snorm: return GL_RGBA8_SNORM;
		case spv::ImageFormatRgba16: return GL_RGBA16;
		case spv::ImageFormatRg16: return GL_RG16;
		case spv::ImageFormatRg8: return GL_RG8;
		case spv::ImageFormatR16: return GL_R16;
		case spv::ImageFormatR8: return GL_R8;
		case spv::ImageFormatRgba16Snorm: return GL_RGBA16_SNORM;
		case spv::ImageFormatRg16Snorm: return GL_RG16_SNORM;
		case spv::ImageFormatRg8Snorm: return GL_RG8_SNORM;
		case spv::ImageFormatR16Snorm: return GL_R16_SNORM;
		case spv::ImageFormatR8Snorm: return GL_R8_SNORM;
		case spv::ImageFormatRgba32i: return GL_RGBA32I;
		case spv::ImageFormatRgba16i: return GL_RGBA16I;
		case spv::ImageFormatRgba8i: return GL_RGBA8I;
		case spv::ImageFormatR32i: return GL_R32I;
		case spv::ImageFormatRgba32ui: return GL_RGBA32UI;
		case spv::ImageFormatRgba16ui: return GL_RGBA16UI;
		case spv::ImageFormatRgba8ui: return GL_RGBA8UI;
		case spv::ImageFormatR32ui: return GL_R32UI;
		case spv::ImageFormatR11fG11fB10f: return GL_R11F_G11F_B10F;
		case spv::ImageFormatRgb10A2: return GL_RGB10_A2;
		case spv::ImageFormatRg32f: return GL_RG32F;
		case spv::ImageFormatRg16f: return GL_RG16F;
		case spv::ImageFormatR16f: return GL_R16F;
		case spv::ImageFormatRg32i: return GL_RG32I;
		case spv::ImageFormatRg16i: return GL_RG16I;
		case spv::ImageFormatRg8i: return GL_RG8I;
		case spv::ImageFormatR16i: return GL_R16I;
		case spv::ImageFormatR8i: return GL_R8I;
		case spv::ImageFormatRg32ui: return GL_RG32UI;
		case spv::ImageFormatRg16ui: return GL_RG16UI;
		case spv::ImageFormatRg8ui: return GL_RG8UI;
		case spv::ImageFormatR16ui: return GL_R16UI;
		case spv::ImageFormatR8ui: return GL_R8UI;
		default: return 0;
	}
}

static bool compileComputeStage(std::vector<unsigned int> spirv, TranslatedProgram &out, TranslationLookup &lookup,
	const ComputeSourceBindings &sourceBindings) {
	try {
		spirv_cross::CompilerMSL msl(std::move(spirv));
		spirv_cross::CompilerMSL::Options options;
		options.platform = spirv_cross::CompilerMSL::Options::iOS;
		options.set_msl_version(2, 2);
		msl.set_msl_options(options);

		auto activeVariables = msl.get_active_interface_variables();
		spirv_cross::ShaderResources resources = msl.get_shader_resources(activeVariables);

		// All resource classes below are entered into SPIRV-Cross's explicit MSL
		// remap table. Avoid duplicate (set, binding) keys across buffer and
		// texture resources before assigning their independent Metal namespace slots.
		std::vector<const spirv_cross::Resource *> mappedResources;
		mappedResources.reserve(resources.storage_buffers.size() + resources.uniform_buffers.size() +
			resources.sampled_images.size() + resources.storage_images.size());
		for (const spirv_cross::Resource &resource : resources.storage_buffers) mappedResources.push_back(&resource);
		for (const spirv_cross::Resource &resource : resources.uniform_buffers) mappedResources.push_back(&resource);
		for (const spirv_cross::Resource &resource : resources.sampled_images) mappedResources.push_back(&resource);
		for (const spirv_cross::Resource &resource : resources.storage_images) mappedResources.push_back(&resource);
		if (!makeMslResourceBindingKeysUnique(msl, mappedResources, "compute", out.log)) return false;

		msl.set_enabled_interface_variables(std::move(activeVariables));

		// In Metal, buffer, texture and sampler namespaces are separate. Keep SSBOs
		// in slots 0...15, named UBOs in 16...29, and GL's loose uniforms in slot 30.
		std::vector<std::pair<uint32_t, const spirv_cross::Resource *>> storageBuffers;
		storageBuffers.reserve(resources.storage_buffers.size());
		for (const spirv_cross::Resource &resource : resources.storage_buffers) {
			storageBuffers.emplace_back(msl.get_decoration(resource.id, spv::DecorationBinding), &resource);
		}
		std::sort(storageBuffers.begin(), storageBuffers.end(), [](const auto &a, const auto &b) {
			return a.first < b.first;
		});
		if (storageBuffers.size() > 16) {
			out.log += "compute shader uses more than 16 active storage buffers; Metal buffer slots 0...15 are exhausted\n";
			return false;
		}
		for (uint32_t slot = 0; slot < storageBuffers.size(); ++slot) {
			const auto &resource = *storageBuffers[slot].second;
			const std::string blockTypeName = msl.get_name(resource.base_type_id);
			auto sourceBinding = sourceBindings.storageBlocks.find(blockTypeName);
			const uint32_t glBindingPoint = sourceBinding == sourceBindings.storageBlocks.end() ? 0 : sourceBinding->second;
			if (glBindingPoint >= GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS) {
				out.log += "compute storage-buffer binding exceeds GL4Metal's 16 SSBO binding points\n";
				return false;
			}
			spirv_cross::MSLResourceBinding binding;
			binding.stage = spv::ExecutionModelGLCompute;
			binding.desc_set = msl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			binding.binding = storageBuffers[slot].first;
			binding.msl_buffer = slot;
			msl.add_msl_resource_binding(binding);

			TranslatedStorageBuffer reflected;
			reflected.name = resource.name;
			reflected.binding = glBindingPoint;
			reflected.slot = slot;
			const spirv_cross::SPIRType &storageType = msl.get_type(resource.base_type_id);
			for (uint32_t memberTypeId : storageType.member_types) {
				const spirv_cross::SPIRType &memberType = msl.get_type(memberTypeId);
				if (!memberType.array.empty() && memberType.array.back() == 0) {
					reflected.runtime_sized = true;
				}
			}
			try {
				reflected.required_size = (uint32_t)msl.get_declared_struct_size(storageType);
			} catch (...) {
				// Runtime-sized SSBO tails have no fixed minimum size in SPIR-V reflection.
				reflected.required_size = 0;
			}
			out.storage_buffers.push_back(std::move(reflected));
		}

		uint32_t nextUniformBufferSlot = 16;
		for (const spirv_cross::Resource &resource : resources.uniform_buffers) {
			const spirv_cross::SPIRType &type = msl.get_type(resource.base_type_id);
			const bool isDefault = msl.get_name(resource.base_type_id) == "gl_DefaultUniformBlock" ||
				resource.name == "gl_DefaultUniformBlock";
			spirv_cross::MSLResourceBinding binding;
			binding.stage = spv::ExecutionModelGLCompute;
			binding.desc_set = msl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			binding.binding = msl.get_decoration(resource.id, spv::DecorationBinding);
			if (isDefault) {
				binding.msl_buffer = GL4METAL_UNIFORM_BUFFER_INDEX;
			} else {
				if (nextUniformBufferSlot >= GL4METAL_UNIFORM_BUFFER_INDEX) {
					out.log += "compute shader uses more than 14 active uniform blocks; Metal buffer slots are exhausted\n";
					return false;
				}
				binding.msl_buffer = nextUniformBufferSlot++;
			}
			msl.add_msl_resource_binding(binding);

			if (!isDefault) {
				std::string blockName = msl.get_name(resource.base_type_id);
				if (blockName.empty()) blockName = resource.name;
				auto blockFound = lookup.block_indices.find(blockName);
				size_t blockIndex = 0;
				if (blockFound == lookup.block_indices.end()) {
					blockIndex = out.blocks.size();
					out.blocks.emplace_back();
					out.blocks.back().name = blockName;
					lookup.block_indices.emplace(blockName, blockIndex);
				} else {
					blockIndex = blockFound->second;
				}
				TranslatedBlock &block = out.blocks[blockIndex];
				const size_t blockSize = msl.get_declared_struct_size(type);
				auto sourceBinding = sourceBindings.uniformBlocks.find(blockName);
				block.binding = sourceBinding == sourceBindings.uniformBlocks.end() ? 0 : sourceBinding->second;
				block.size = std::max(block.size, (uint32_t)((blockSize + 15) & ~(size_t)15));
				block.required_size = std::max(block.required_size, (uint32_t)blockSize);
				block.stage[2].present = true;
				block.stage[2].slot = binding.msl_buffer;

				const std::string memberPrefix = resource.name.empty() ? blockName : resource.name;
				for (uint32_t memberIndex = 0; memberIndex < type.member_types.size(); ++memberIndex) {
					const spirv_cross::SPIRType &memberType = msl.get_type(type.member_types[memberIndex]);
					if (memberType.basetype == spirv_cross::SPIRType::Struct) {
						out.log += "compute uniform-block member is a nested struct; nested reflection is not supported\n";
						continue;
					}
					const std::string memberName = memberPrefix + "." + msl.get_member_name(resource.base_type_id, memberIndex);
					TranslatedUniform *member = nullptr;
					auto memberFound = lookup.block_member_indices.find(memberName);
					if (memberFound == lookup.block_member_indices.end()) {
						const size_t memberPosition = block.members.size();
						block.members.emplace_back();
						block.members.back().name = memberName;
						lookup.block_member_indices.emplace(memberName, std::make_pair(blockIndex, memberPosition));
						member = &block.members.back();
					} else {
						member = &out.blocks[memberFound->second.first].members[memberFound->second.second];
					}
					member->kind = kindOf(memberType.basetype);
					member->vecsize = (int)memberType.vecsize;
					member->columns = (int)memberType.columns;
					member->array_size = arraySizeOf(memberType);
					TranslatedUniform::Stage &memberStage = member->stage[2];
					memberStage.present = true;
					memberStage.offset = msl.type_struct_member_offset(type, memberIndex);
					memberStage.array_stride = memberType.array.empty() ? 0 : msl.type_struct_member_array_stride(type, memberIndex);
					memberStage.matrix_stride = memberType.columns > 1 ? msl.type_struct_member_matrix_stride(type, memberIndex) : 0;
					memberStage.row_major = msl.has_member_decoration(resource.base_type_id, memberIndex, spv::DecorationRowMajor);
				}
				continue;
			}

			const size_t blockSize = msl.get_declared_struct_size(type);
			out.block_size[2] = (uint32_t)((blockSize + 15) & ~(size_t)15);
			for (uint32_t memberIndex = 0; memberIndex < type.member_types.size(); ++memberIndex) {
				const spirv_cross::SPIRType &memberType = msl.get_type(type.member_types[memberIndex]);
				if (memberType.basetype == spirv_cross::SPIRType::Struct) {
					out.log += "compute default uniform contains an unsupported nested struct\n";
					continue;
				}
				const std::string uniformName = msl.get_member_name(resource.base_type_id, memberIndex);
				TranslatedUniform &uniform = findOrAdd(out, uniformName, lookup);
				uniform.kind = kindOf(memberType.basetype);
				uniform.vecsize = (int)memberType.vecsize;
				uniform.columns = (int)memberType.columns;
				uniform.array_size = arraySizeOf(memberType);
				TranslatedUniform::Stage &stage = uniform.stage[2];
				stage.present = true;
				stage.offset = msl.type_struct_member_offset(type, memberIndex);
				stage.array_stride = memberType.array.empty() ? 0 : msl.type_struct_member_array_stride(type, memberIndex);
				stage.matrix_stride = memberType.columns > 1 ? msl.type_struct_member_matrix_stride(type, memberIndex) : 0;
				stage.row_major = msl.has_member_decoration(resource.base_type_id, memberIndex, spv::DecorationRowMajor);
			}
		}

		std::vector<std::pair<uint32_t, const spirv_cross::Resource *>> sampledImages;
		sampledImages.reserve(resources.sampled_images.size());
		for (const spirv_cross::Resource &resource : resources.sampled_images) {
			sampledImages.emplace_back(msl.get_decoration(resource.id, spv::DecorationBinding), &resource);
		}
		std::sort(sampledImages.begin(), sampledImages.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
		uint32_t nextTextureSlot = 0;
		for (const auto &entry : sampledImages) {
			const spirv_cross::Resource &resource = *entry.second;
			const int count = arraySizeOf(msl.get_type(resource.type_id));
			spirv_cross::MSLResourceBinding binding;
			binding.stage = spv::ExecutionModelGLCompute;
			binding.desc_set = msl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			binding.binding = entry.first;
			binding.msl_texture = nextTextureSlot;
			binding.msl_sampler = nextTextureSlot;
			binding.count = (uint32_t)count; // SPIRV-Cross reserves all array elements in Metal
			msl.add_msl_resource_binding(binding);

			TranslatedUniform &uniform = findOrAdd(out, resource.name, lookup);
			uniform.kind = TranslatedUniform::Sampler;
			const spirv_cross::SPIRType &samplerType = msl.get_type(resource.type_id);
			if (samplerType.image.dim == spv::DimBuffer) uniform.dim = TranslatedUniform::DimBuffer;
			else if (samplerType.image.dim == spv::DimCube) uniform.dim = TranslatedUniform::DimCube;
			else if (samplerType.image.dim == spv::Dim3D) uniform.dim = TranslatedUniform::Dim3D;
			else if (samplerType.image.arrayed) uniform.dim = TranslatedUniform::Dim2DArray;
			else uniform.dim = TranslatedUniform::Dim2D;
			uniform.vecsize = 1;
			uniform.columns = 1;
			uniform.array_size = count;
			uniform.stage[2].present = true;
			uniform.stage[2].slot = nextTextureSlot;
			nextTextureSlot += (uint32_t)count;
			if (nextTextureSlot > GL4METAL_MAX_COMPUTE_TEXTURE_UNITS) {
				out.log += "compute shader exceeds the 16 Metal sampler-state slots\n";
				return false;
			}
		}

		for (const spirv_cross::Resource &resource : resources.storage_images) {
			const spirv_cross::SPIRType &imageType = msl.get_type(resource.type_id);
			// Metal can expose a GL image3D as texture3d with read/write access. The
			// GL_TEXTURE_3D allocation path is already private + ShaderRead/ShaderWrite.
			// 2D-array/cube storage images remain explicitly unsupported until their GL
			// texture allocation and view-binding paths are implemented as well.
			const bool is2D = imageType.image.dim == spv::Dim2D && !imageType.image.arrayed;
			const bool is3D = imageType.image.dim == spv::Dim3D && !imageType.image.arrayed;
			if ((!is2D && !is3D) || imageType.image.ms) {
				out.log += "compute storage image '" + resource.name +
				"' has unsupported dimension (supported: image2D and image3D; no arrayed or multisample images)\n";
				return false;
			}
			const uint32_t resourceCount = (uint32_t)arraySizeOf(imageType);
			const uint32_t spirvBinding = msl.get_decoration(resource.id, spv::DecorationBinding);
			auto sourceBinding = sourceBindings.storageImages.find(resource.name);
			const uint32_t bindingPoint = sourceBinding == sourceBindings.storageImages.end() ? 0 : sourceBinding->second;
			if (resourceCount == 0 || resourceCount > GL4METAL_MAX_IMAGE_UNITS) {
				out.log += "compute storage image array has an invalid element count\n";
				return false;
			}
			if (bindingPoint >= GL4METAL_MAX_IMAGE_UNITS || resourceCount > GL4METAL_MAX_IMAGE_UNITS - bindingPoint) {
				out.log += "compute storage-image binding range exceeds GL4Metal's 32 image units\n";
				return false;
			}
			if (resourceCount > 32u - nextTextureSlot) {
				out.log += "compute shader exceeds the 32 Metal texture binding slots\n";
				return false;
			}
			spirv_cross::MSLResourceBinding binding;
			binding.stage = spv::ExecutionModelGLCompute;
			binding.desc_set = msl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			binding.binding = spirvBinding;
			binding.msl_texture = nextTextureSlot;
			binding.count = resourceCount;
			msl.add_msl_resource_binding(binding);

			for (uint32_t element = 0; element < resourceCount; ++element) {
				TranslatedImage image;
				image.name = resourceCount > 1 ? resource.name + "[" + std::to_string(element) + "]" : resource.name;
				image.binding = bindingPoint + element;
				image.slot = nextTextureSlot + element;
				image.format = glImageFormatForSpirv(imageType.image.format);
				image.is_3d = is3D;
				image.read_only = msl.has_decoration(resource.id, spv::DecorationNonWritable);
				image.write_only = msl.has_decoration(resource.id, spv::DecorationNonReadable);
				out.storage_images.push_back(std::move(image));
			}
			nextTextureSlot += resourceCount;
		}

		out.workgroup_size[0] = msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, 0);
		out.workgroup_size[1] = msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, 1);
		out.workgroup_size[2] = msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, 2);
		if (!out.workgroup_size[0] || !out.workgroup_size[1] || !out.workgroup_size[2]) {
			out.log += "compute shader has an invalid local workgroup size\n";
			return false;
		}

		sanitizeMslIdentifiers(msl, resources, "compute", out.log);
		out.msl[2] = msl.compile();
		return true;
	} catch (const std::exception &e) {
		out.log += std::string("compute stage, SPIRV-Cross: ") + e.what() + "\n";
		return false;
	}
}

} // namespace


bool compactMetalThreadgroupFloat3Arrays(std::string &mslSource, std::size_t maxThreadgroupBytes, std::string &notes, bool force) {
	if (maxThreadgroupBytes == 0 || mslSource.empty()) return false;

	// SPIRV-Cross represents GLSL shared vec3 arrays as nested spvUnsafeArray
	// wrappers. MSL float3 has a 16-byte array stride; packed_float3 is 12 bytes.
	// Workgroup storage is private to one threadgroup, so changing only this
	// storage representation does not alter an externally visible buffer ABI.
	static const std::regex declarationPattern(
		R"(threadgroup\s+spvUnsafeArray\s*<\s*spvUnsafeArray\s*<\s*float3\s*,\s*([0-9]+)\s*>\s*,\s*([0-9]+)\s*>\s*([A-Za-z_][A-Za-z0-9_]*)\s*;)" );

	struct Candidate {
		std::string declaration;
		std::string arrayName;
		std::size_t innerCount = 0;
		std::size_t outerCount = 0;
	};
	std::vector<Candidate> candidates;
	for (std::sregex_iterator it(mslSource.begin(), mslSource.end(), declarationPattern), end; it != end; ++it) {
		const std::smatch &match = *it;
		Candidate candidate;
		candidate.declaration = match.str();
		candidate.innerCount = (std::size_t)std::strtoull(match[1].str().c_str(), nullptr, 10);
		candidate.outerCount = (std::size_t)std::strtoull(match[2].str().c_str(), nullptr, 10);
		candidate.arrayName = match[3].str();
		if (candidate.innerCount == 0 || candidate.outerCount == 0 ||
			candidate.innerCount > std::numeric_limits<std::size_t>::max() / candidate.outerCount) continue;
		candidates.push_back(std::move(candidate));
	}
	if (candidates.empty()) return false;

	// Avoid changing storage layout for kernels that already fit. If Metal later
	// reports an aggregate overflow caused by additional declarations, the caller
	// retries with force=true and compacts every eligible float3 array.
	std::size_t estimatedCandidateBytes = 0;
	for (const Candidate &candidate : candidates) {
		const std::size_t elementCount = candidate.innerCount * candidate.outerCount;
		if (elementCount > std::numeric_limits<std::size_t>::max() / 16u ||
			elementCount * 16u > std::numeric_limits<std::size_t>::max() - estimatedCandidateBytes) return false;
		estimatedCandidateBytes += elementCount * 16u;
	}
	if (!force && estimatedCandidateBytes <= maxThreadgroupBytes) return false;

	std::string optimized = mslSource;
	std::size_t originalCandidateBytes = 0;
	std::size_t packedCandidateBytes = 0;
	std::size_t copyCounter = 0;
	std::vector<std::string> optimizedNames;

	for (const Candidate &candidate : candidates) {
		const std::size_t elementCount = candidate.innerCount * candidate.outerCount;
		if (elementCount > std::numeric_limits<std::size_t>::max() / 16u) continue;
		const std::size_t originalBytes = elementCount * 16u;
		const std::size_t packedBytes = elementCount * 12u;
		if (originalBytes > std::numeric_limits<std::size_t>::max() - originalCandidateBytes ||
			packedBytes > std::numeric_limits<std::size_t>::max() - packedCandidateBytes) continue;

		// Only compact arrays whose row values are copied into a normal MSL
		// spvUnsafeArray. Rewrite that aggregate copy to explicit element-wise
		// conversion, avoiding an invalid packed/unpacked aggregate assignment.
		const std::string copyPatternText =
			"spvUnsafeArray\\s*<\\s*float3\\s*,\\s*" + std::to_string(candidate.innerCount) +
			">\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*" + candidate.arrayName +
			R"(\s*\[\s*([^\]]+)\s*\]\s*;)";
		const std::regex copyPattern(copyPatternText);

		// Packed vectors support the same arithmetic as aligned vectors, but make
		// conversions explicit when SPIRV-Cross initializes a normal float3 local
		// directly from one shared-array element.
		const std::regex scalarCopyPattern(
			"(\\bfloat3\\s+[A-Za-z_][A-Za-z0-9_]*\\s*=\\s*)" + candidate.arrayName +
			R"(((?:\s*\[[^\]]+\]){2})\s*;)" );
		optimized = std::regex_replace(optimized, scalarCopyPattern,
			"$1float3(" + candidate.arrayName + "$2);");

		std::vector<std::smatch> copyMatches;
		for (std::sregex_iterator copyIt(optimized.begin(), optimized.end(), copyPattern), copyEnd; copyIt != copyEnd; ++copyIt) {
			copyMatches.push_back(*copyIt);
		}

		// If an aggregate row copy exists, all matching copies are rewritten into
		// element-wise float3 constructors. This keeps the helper-function argument
		// ABI identical to the original unpacked spvUnsafeArray<float3, N> type.
		std::string rewritten;
		if (!copyMatches.empty()) {
			std::size_t cursor = 0;
			for (const std::smatch &copyMatch : copyMatches) {
				const std::size_t matchPosition = (std::size_t)copyMatch.position();
				rewritten.append(optimized, cursor, matchPosition - cursor);
				const std::string localName = copyMatch[1].str();
				const std::string rowExpression = copyMatch[2].str();
				const std::size_t lineStart = optimized.rfind('\n', matchPosition);
				const std::size_t indentStart = lineStart == std::string::npos ? 0 : lineStart + 1;
				const std::string indentation = optimized.substr(indentStart, matchPosition - indentStart);
				std::string loopName;
				do {
					loopName = "gl4metal_threadgroup_copy_index_" + std::to_string(copyCounter++);
				} while (optimized.find(loopName) != std::string::npos || rewritten.find(loopName) != std::string::npos);
				rewritten += "spvUnsafeArray<float3, " + std::to_string(candidate.innerCount) + "> " + localName + ";\n";
				rewritten += indentation + "for (uint " + loopName + " = 0u; " + loopName + " < " +
					std::to_string(candidate.innerCount) + "u; ++" + loopName + ") {\n";
				rewritten += indentation + "    " + localName + "[" + loopName + "] = float3(" + candidate.arrayName +
					"[" + rowExpression + "][" + loopName + "]);\n";
				rewritten += indentation + "}\n";
				cursor = matchPosition + (std::size_t)copyMatch.length();
			}
			rewritten.append(optimized, cursor, std::string::npos);
			optimized.swap(rewritten);
		}

		std::string packedDeclaration = candidate.declaration;
		const std::size_t typePosition = packedDeclaration.find("float3");
		if (typePosition == std::string::npos) continue;
		packedDeclaration.replace(typePosition, 6, "packed_float3");
		const std::size_t declarationPosition = optimized.find(candidate.declaration);
		if (declarationPosition == std::string::npos) continue;
		optimized.replace(declarationPosition, candidate.declaration.size(), packedDeclaration);
		originalCandidateBytes += originalBytes;
		packedCandidateBytes += packedBytes;
		optimizedNames.push_back(candidate.arrayName);
	}

	if (optimizedNames.empty() || packedCandidateBytes >= originalCandidateBytes) return false;
	mslSource.swap(optimized);
	std::size_t savedBytes = originalCandidateBytes - packedCandidateBytes;
	std::ostringstream note;
	note << "packed " << optimizedNames.size() << " threadgroup float3 array(s) (" << originalCandidateBytes
		 << " -> " << packedCandidateBytes << " bytes for these arrays, saved " << savedBytes
		 << "); workgroup size and all barrier locations are unchanged; device limit=" << maxThreadgroupBytes << " bytes";
	if (packedCandidateBytes <= maxThreadgroupBytes) {
		note << "; compacted target arrays now fit within the reported limit (other declarations remain compiler-validated)";
	} else {
		note << "; other threadgroup declarations may still exceed the device limit";
	}
	notes = note.str();
	return true;
}

bool translateProgram(const std::string &vertexGLSL, const std::string &fragmentGLSL, TranslatedProgram &out) {
	out = TranslatedProgram();
	initGlslang();

	std::string vsText = patchVersion(vertexGLSL);
	std::string fsText = patchVersion(fragmentGLSL);
	const bool fragmentUsesFragCoord = usesFragmentFragCoord(fsText);
	const bool fragmentUsesUpperLeftOrigin = declaresFragCoordOriginUpperLeft(fsText);
	const bool fixFragmentFragCoordY = fragmentUsesFragCoord && !fragmentUsesUpperLeftOrigin;
	out.uses_gl_frag_coord = false;
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

	// mapIO() handles ordinary linked interfaces, but compatibility-generated
	// outputs such as OptiFine's unused outColor0 can still leave the two SPIR-V
	// stages with independent locations. Only repair shaders without explicit
	// location qualifiers; explicit GLSL interfaces must retain their declared ABI.
	const std::string vsInterfaceMasked = maskShaderComments(vsText);
	const std::string fsInterfaceMasked = maskShaderComments(fsText);
	const bool alignFragmentInputs =
		!hasLayoutQualifiedStorage(vsInterfaceMasked, "out") &&
		!hasLayoutQualifiedStorage(fsInterfaceMasked, "in");
	std::unordered_map<std::string, StageVaryingInfo> vertexOutputLocations;
	if (alignFragmentInputs) {
		std::string interfaceError;
		if (!collectVertexOutputLocations(spirv[VS], vertexOutputLocations, interfaceError)) {
			out.log += "stage interface: " + interfaceError + "\n";
			return false;
		}
	}

	// Narrow only inter-stage values that are explicitly RelaxedPrecision in both
	// GLSL stages and whose semantics are normalized coordinates/directions.
	std::unordered_set<std::string> vertexHalfVaryings;
	std::unordered_set<std::string> fragmentHalfVaryings;
	buildHalfVaryingNameSets(spirv[VS], spirv[FS], vertexHalfVaryings, fragmentHalfVaryings);
	if (!vertexHalfVaryings.empty()) {
		out.log += std::to_string(vertexHalfVaryings.size()) + " explicitly relaxed UV/normal varyings lowered to MSL half\n";
	}

	// Keep name-to-index maps for the complete linked program. SPIRV-Cross resource
	// reflection visits both stages; linear vector searches here used to become O(N^2)
	// for shader packs with many UBO members and samplers.
	TranslationLookup lookup;
	lookup.uniform_indices.reserve(64);
	lookup.block_indices.reserve(16);
	lookup.block_member_indices.reserve(128);

	// Compile each GLSL stage once. The vertex function constant produces normal and
	// texture-flipped variants from one Metal library at program-link time.
	const std::unordered_map<std::string, StageVaryingInfo> noVertexOutputLocations;
	if (!compileStage(std::move(spirv[VS]), spv::ExecutionModelVertex, VS, vertexHalfVaryings, false,
		noVertexOutputLocations, false, out, lookup) ||
		!compileStage(std::move(spirv[FS]), spv::ExecutionModelFragment, FS, fragmentHalfVaryings,
		fixFragmentFragCoordY, vertexOutputLocations, alignFragmentInputs, out, lookup)) {
		return false;
	}
	// Vulkan SPIR-V legalization may represent GLSL bool uniforms as 32-bit uints.
	// Preserve their original desktop-GL type for glGetActiveUniform / GL_UNIFORM_TYPE.
	// Likewise keep the exact samplerBuffer variant for integer/unsigned buffer samplers.
	applySourceUniformTypes(out, vertexGLSL, fragmentGLSL);
	finalizeGlslang();
	return true;
}


bool translateComputeProgram(const std::string &computeGLSL, TranslatedProgram &out) {
	initGlslang();
	out = TranslatedProgram();
	std::string computeText = patchVersion(computeGLSL, 430);
	const char *computeSource = computeText.c_str();
	glslang::TShader shader(EShLangCompute);
	setupShader(shader, EShLangCompute, &computeSource);
	const EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	const TBuiltInResource *resources = GetDefaultResources();
	if (!shader.parse(resources, 430, false, messages)) {
		out.log += std::string("compute shader:\n") + shader.getInfoLog() + shader.getInfoDebugLog();
		return false;
	}
	glslang::TProgram program;
	program.addShader(&shader);
	if (!program.link(messages)) {
		out.log += std::string("compute link:\n") + program.getInfoLog() + program.getInfoDebugLog();
		return false;
	}
	glslang::TIntermediate *intermediate = program.getIntermediate(EShLangCompute);
	if (!intermediate) {
		out.log += "no intermediate representation for compute stage\n";
		return false;
	}
	std::vector<unsigned int> spirv;
	spv::SpvBuildLogger logger;
	glslang::GlslangToSpv(*intermediate, spirv, &logger);
	if (spirv.empty()) {
		out.log += "compute SPIR-V generation failed: " + logger.getAllMessages() + "\n";
		return false;
	}
	TranslationLookup lookup;
	lookup.uniform_indices.reserve(64);
	lookup.block_indices.reserve(16);
	lookup.block_member_indices.reserve(128);
	const ComputeSourceBindings sourceBindings = collectComputeSourceBindings(computeGLSL);
	if (!compileComputeStage(std::move(spirv), out, lookup, sourceBindings)) return false;
	applySourceUniformTypes(out, computeGLSL, std::string());
	finalizeGlslang();
	return true;
}
