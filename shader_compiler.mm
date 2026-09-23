#include "shader_compiler.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv_cross_c.h>

namespace {

EShLanguage stageFor(bool vertex) {
    return vertex ? EShLangVertex : EShLangFragment;
}

bool compileStage(const char *source, bool vertex, std::vector<unsigned int> &spirv) {
    if (!source || !*source) return false;

    const char *strings[] = { source };
    glslang::TShader shader(stageFor(vertex));
    shader.setStrings(strings, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, stageFor(vertex), glslang::EShClientOpenGL, 330);
    shader.setEnvClient(glslang::EShClientOpenGL, glslang::EShTargetOpenGL_450);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

    if (!shader.parse(GetDefaultResources(), 330, false,
                      (EShMessages)(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules))) {
        return false;
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link((EShMessages)(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules))) {
        return false;
    }

    glslang::SpvOptions options;
    options.generateDebugInfo = false;
    options.stripDebugInfo = true;
    options.disableOptimizer = false;
    options.optimizeSize = true;
    options.validate = true;
    glslang::GlslangToSpv(*program.getIntermediate(stageFor(vertex)), spirv, &options);
    return !spirv.empty();
}

std::string crossToMSL(const std::vector<unsigned int> &spirv, bool vertex) {
    if (spirv.empty()) return {};

    spvc_context context = nullptr;
    if (spvc_context_create(&context) != SPVC_SUCCESS) return {};

    spvc_parsed_ir parsed = nullptr;
    spvc_compiler compiler = nullptr;
    spvc_compiler_options options = nullptr;
    const char *source = nullptr;
    std::string result;

    if (spvc_context_parse_spirv(context, spirv.data(), spirv.size(), &parsed) == SPVC_SUCCESS &&
        spvc_context_create_compiler(context, SPVC_BACKEND_MSL, parsed,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) == SPVC_SUCCESS &&
        spvc_compiler_create_compiler_options(compiler, &options) == SPVC_SUCCESS) {
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_VERSION,
                                       SPVC_TRUE);
        spvc_compiler_install_compiler_options(compiler, options);

        const spvc_entry_point *entryPoints = nullptr;
        size_t entryPointCount = 0;
        if (spvc_compiler_get_entry_points(compiler, &entryPoints, &entryPointCount) == SPVC_SUCCESS && entryPointCount > 0) {
            const char *oldName = entryPoints[0].name;
            const char *newName = vertex ? "gl4metal_program_vertex" : "gl4metal_program_fragment";
            spvc_compiler_rename_entry_point(compiler, oldName, newName, entryPoints[0].execution_model);
        }

        if (spvc_compiler_compile(compiler, &source) == SPVC_SUCCESS && source) {
            result = source;
        }
    }

    spvc_context_destroy(context);
    return result;
}

} // namespace

extern "C" char *gl4metalCompileGLSLToMSL(const char *vertexSource, const char *fragmentSource) {
    if (!vertexSource || !fragmentSource) return nullptr;

    static bool initialized = false;
    if (!initialized) {
        if (!glslang::InitializeProcess()) return nullptr;
        initialized = true;
    }

    std::vector<unsigned int> vertexSpirv;
    std::vector<unsigned int> fragmentSpirv;
    if (!compileStage(vertexSource, true, vertexSpirv) || !compileStage(fragmentSource, false, fragmentSpirv)) {
        return nullptr;
    }

    std::string vertexMSL = crossToMSL(vertexSpirv, true);
    std::string fragmentMSL = crossToMSL(fragmentSpirv, false);
    if (vertexMSL.empty() || fragmentMSL.empty()) return nullptr;

    std::string combined = vertexMSL + "\n" + fragmentMSL;
    char *result = static_cast<char *>(std::malloc(combined.size() + 1));
    if (!result) return nullptr;
    std::memcpy(result, combined.c_str(), combined.size() + 1);
    return result;
}

extern "C" void gl4metalFreeCompiledMSL(char *source) {
    std::free(source);
}
