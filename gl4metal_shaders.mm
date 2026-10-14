#import "gl4metal.h"
#import "gl4metal_shaders.h"
#import "gl4metal_runtime.h"

#include <regex>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>

NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao) {
    NSMutableString *key = [NSMutableString stringWithFormat:@"p%u:rtC%luD%luS%lu:B%d/%u/%u/%u/%u/%u/%u/%u/%u/%u/%u;",
        program,
        (unsigned long)gl4metalGetCurrentColorPixelFormat(),
        (unsigned long)gl4metalGetCurrentDepthPixelFormat(),
        (unsigned long)gl4metalGetCurrentStencilPixelFormat(),
        (int)ctx.blendEnabled,
        (unsigned)ctx.blendSrcRGB, (unsigned)ctx.blendDstRGB,
        (unsigned)ctx.blendSrcAlpha, (unsigned)ctx.blendDstAlpha,
        (unsigned)ctx.blendEquationRGB, (unsigned)ctx.blendEquationAlpha,
        (unsigned)ctx.colorMaskRed, (unsigned)ctx.colorMaskGreen,
        (unsigned)ctx.colorMaskBlue, (unsigned)ctx.colorMaskAlpha];
    if (!vao) {
        [key appendString:@"default"];
        return key;
    }

    for (NSUInteger index = 0; index < 16; index++) {
        const gl4metalVertexAttrib *attrib = &vao->attribs[index];
        if (!attrib->enabled) {
            [key appendFormat:@"a%lu:0;", (unsigned long)index];
            continue;
        }
        [key appendFormat:@"a%lu:%d:%u:%d:%d:%ld:%u:%u;",
         (unsigned long)index,
         attrib->size,
         attrib->type,
         attrib->normalized,
         attrib->integer,
         (long)attrib->stride,
         attrib->boundVBO,
         attrib->divisor];
    }
    return key;
}

void gl4metalCachePipelineForCurrentLayout(GLuint program) {
    if (!ctx || !ctx.pipelineState) return;
    NSString *cacheKey = gl4metalBuildPipelineCacheKey(program, gl4metalGetCurrentVAO());
    pipelineLayoutCache[cacheKey] = ctx.pipelineState;
}

BOOL gl4metalShaderUsesAttribute(NSString *source, NSString *name) {
    if (!source || !name || source.length == 0) return NO;
    NSArray *patterns = @[
        [NSString stringWithFormat:@"%@", name],
        [NSString stringWithFormat:@"a%@", name],
        [NSString stringWithFormat:@"in %@", name],
        [NSString stringWithFormat:@"attribute %@", name],
        [NSString stringWithFormat:@"varying %@", name]
    ];
    for (NSString *pattern in patterns) {
        if ([source rangeOfString:pattern options:NSCaseInsensitiveSearch].location != NSNotFound) {
            return YES;
        }
    }
    return NO;
}


static MTLVertexFormat gl4metalVertexFormatForShaderType(NSString *type) {
    if (!type) return MTLVertexFormatInvalid;
    if ([type isEqualToString:@"float"]) return MTLVertexFormatFloat;
    if ([type isEqualToString:@"vec2"]) return MTLVertexFormatFloat2;
    if ([type isEqualToString:@"vec3"]) return MTLVertexFormatFloat3;
    if ([type isEqualToString:@"vec4"]) return MTLVertexFormatFloat4;
    if ([type isEqualToString:@"int"]) return MTLVertexFormatInt;
    if ([type isEqualToString:@"ivec2"]) return MTLVertexFormatInt2;
    if ([type isEqualToString:@"ivec3"]) return MTLVertexFormatInt3;
    if ([type isEqualToString:@"ivec4"]) return MTLVertexFormatInt4;
    if ([type isEqualToString:@"uint"]) return MTLVertexFormatUInt;
    if ([type isEqualToString:@"uvec2"]) return MTLVertexFormatUInt2;
    if ([type isEqualToString:@"uvec3"]) return MTLVertexFormatUInt3;
    if ([type isEqualToString:@"uvec4"]) return MTLVertexFormatUInt4;
    return MTLVertexFormatInvalid;
}

static NSUInteger gl4metalShaderInputLocationAdvance(NSString *type) {
    if ([type isEqualToString:@"mat2"]) return 2;
    if ([type isEqualToString:@"mat3"]) return 3;
    if ([type isEqualToString:@"mat4"]) return 4;
    return 1;
}

BOOL gl4metalApplyVertexShaderInputFormats(NSString *vertexSource, GLuint program, MTLVertexDescriptor *descriptor) {
    if (!vertexSource || !descriptor) return NO;

    // We intentionally parse the original GLSL input declarations rather than
    // asking Metal to guess a format. The GLSL type is the authoritative type
    // for a disabled generic vertex attribute.
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:
        @"(?:layout\\s*\\(\\s*location\\s*=\\s*([0-9]+)[^)]*\\)\\s*)?"
         @"(?:flat\\s+|smooth\\s+|noperspective\\s+|centroid\\s+|sample\\s+)*"
         @"(?:in|attribute)\\s+(?:(?:lowp|mediump|highp)\\s+)?"
         @"([A-Za-z_][A-Za-z0-9_]*)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\[[^]]+\\])?\\s*;"
        options:0 error:nil];
    if (!regex) return NO;

    NSArray<NSTextCheckingResult *> *matches = [regex matchesInString:vertexSource options:0 range:NSMakeRange(0, vertexSource.length)];
    if (matches.count == 0) return NO;

    NSMutableDictionary<NSNumber *, NSString *> *explicitBindings = programAttribBindings[@(program)];
    NSMutableSet<NSNumber *> *usedLocations = [NSMutableSet set];
    NSUInteger nextImplicitLocation = 0;
    BOOL foundSupportedInput = NO;

    for (NSTextCheckingResult *match in matches) {
        NSString *locationText = nil;
        if (match.numberOfRanges > 1 && [match rangeAtIndex:1].location != NSNotFound) {
            locationText = [vertexSource substringWithRange:[match rangeAtIndex:1]];
        }
        NSString *type = [vertexSource substringWithRange:[match rangeAtIndex:2]];
        NSString *name = [vertexSource substringWithRange:[match rangeAtIndex:3]];
        MTLVertexFormat format = gl4metalVertexFormatForShaderType(type);

        NSUInteger location = NSNotFound;
        if (locationText) {
            location = (NSUInteger)locationText.integerValue;
        } else if (explicitBindings) {
            for (NSNumber *idx in explicitBindings) {
                if ([explicitBindings[idx] isEqualToString:name]) {
                    location = idx.unsignedIntegerValue;
                    break;
                }
            }
        }
        if (location == NSNotFound) {
            while ([usedLocations containsObject:@(nextImplicitLocation)]) nextImplicitLocation++;
            location = nextImplicitLocation;
        }

        if (location >= 16) {
            printf("[gl4metal] WARNING: shader vertex input %s assigned location %lu, outside supported 0..15\n",
                   name.UTF8String, (unsigned long)location);
            continue;
        }
        [usedLocations addObject:@(location)];
        nextImplicitLocation = MAX(nextImplicitLocation, location + gl4metalShaderInputLocationAdvance(type));

        if (format == MTLVertexFormatInvalid) {
            printf("[gl4metal] WARNING: unsupported vertex shader input type %s %s at location %lu; leaving descriptor unchanged\n",
                   type.UTF8String, name.UTF8String, (unsigned long)location);
            continue;
        }
        foundSupportedInput = YES;

        // Disabled GL arrays need a synthetic descriptor entry. Enabled
        // arrays keep the format derived from glVertexAttribPointer so GL
        // conversion/normalization semantics remain intact. If an enabled
        // attribute nevertheless produced an invalid Metal format, recover
        // from the GLSL input type so the strict real pipeline has a complete
        // vertex descriptor.
        gl4metalVertexArray *vao = gl4metalGetCurrentVAO();
        BOOL enabled = vao && vao->attribs[location].enabled;
        // The shader's input location is the authoritative Metal vertex
        // buffer index for this generated SPIR-V/MSL input. Always normalize
        // the descriptor mapping here, even when the GL attribute already had
        // a valid format. This fixes sparse layouts such as 0,1,2,3,4,5 where
        // an existing GL descriptor can otherwise retain a stale bufferIndex.
        descriptor.attributes[location].format =
            descriptor.attributes[location].format == MTLVertexFormatInvalid
                ? format : descriptor.attributes[location].format;
        descriptor.attributes[location].offset = 0;
        descriptor.attributes[location].bufferIndex = location;

        NSUInteger stride = 16;
        if (enabled && vao->attribs[location].stride > 0) {
            stride = (NSUInteger)vao->attribs[location].stride;
        }
        descriptor.layouts[location].stride = stride;
        descriptor.layouts[location].stepRate =
            (enabled && vao->attribs[location].divisor > 0)
                ? vao->attribs[location].divisor : 1;
        descriptor.layouts[location].stepFunction =
            (enabled && vao->attribs[location].divisor > 0)
                ? MTLVertexStepFunctionPerInstance
                : MTLVertexStepFunctionPerVertex;

        if (!enabled) {
            gl4metalRegisterConstantVertexAttribute(location, format);
        }

        printf("[gl4metal][ATTR] %s GL attr[%lu] normalized from GLSL input %s %s -> Metal format %lu buffer=%lu stride=%lu\n",
               enabled ? "Enabled" : "Disabled",
               (unsigned long)location,
               type.UTF8String, name.UTF8String, (unsigned long)format,
               (unsigned long)descriptor.attributes[location].bufferIndex,
               (unsigned long)descriptor.layouts[location].stride);
    }

    return foundSupportedInput;
}

static NSString *gl4metalExtractConstantFragmentColor(NSString *source) {
    if (!source) return nil;

    NSRegularExpression *expression = [NSRegularExpression regularExpressionWithPattern:
        @"gl_FragColor\\s*=\\s*vec4\\s*\\(\\s*([0-9eE+\\-.]+)\\s*,\\s*([0-9eE+\\-.]+)\\s*,\\s*([0-9eE+\\-.]+)\\s*,\\s*([0-9eE+\\-.]+)\\s*\\)"
        options:NSRegularExpressionCaseInsensitive error:nil];
    NSTextCheckingResult *match = [expression firstMatchInString:source options:0 range:NSMakeRange(0, source.length)];
    if (!match || match.numberOfRanges != 5) return nil;

    NSMutableArray<NSString *> *components = [NSMutableArray arrayWithCapacity:4];
    for (NSUInteger index = 1; index < 5; index++) {
        [components addObject:[source substringWithRange:[match rangeAtIndex:index]]];
    }
    return [NSString stringWithFormat:@"float4(%@, %@, %@, %@)", components[0], components[1], components[2], components[3]];
}

NSString *gl4metalBuildTranslatedMetalSource(NSString *vertexSource, NSString *fragmentSource, BOOL *hasColor, BOOL *hasTexCoord) {
    if (vertexSource == nil) {
        vertexSource = @"attribute vec4 aPosition; attribute vec4 aColor; varying vec4 vColor; void main() { vColor = aColor; gl_Position = aPosition; }";
    }
    if (fragmentSource == nil) {
        fragmentSource = @"varying vec4 vColor; void main() { gl_FragColor = vColor; }";
    }

    BOOL color = NO;
    BOOL texcoord = NO;
    NSString *constantColor = gl4metalExtractConstantFragmentColor(fragmentSource);
    color = gl4metalShaderUsesAttribute(vertexSource, @"aColor") ||
            gl4metalShaderUsesAttribute(vertexSource, @"color") ||
            gl4metalShaderUsesAttribute(fragmentSource, @"vColor");
    texcoord = gl4metalShaderUsesAttribute(vertexSource, @"aTexCoord") ||
               gl4metalShaderUsesAttribute(vertexSource, @"aUV") ||
               gl4metalShaderUsesAttribute(vertexSource, @"uv") ||
               gl4metalShaderUsesAttribute(vertexSource, @"texCoord");

    NSMutableString *metalSource = [[NSMutableString alloc] init];
    [metalSource appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"
     "struct VertexIn {\n"
     "    float4 position [[attribute(0)]];\n"];
    if (color) { [metalSource appendString:@"    float4 color [[attribute(1)]];\n"]; }
    if (texcoord) { [metalSource appendString:@"    float2 uv [[attribute(2)]];\n"]; }
    [metalSource appendString:@"};\n"
     "struct VertexOut {\n"
     "    float4 position [[position]];\n"];
    if (color) { [metalSource appendString:@"    float4 color;\n"]; }
    if (texcoord) { [metalSource appendString:@"    float2 uv;\n"]; }
    [metalSource appendString:@"};\n"
     "struct FallbackUniforms {\n"
     "    float4x4 mvp;\n"
     "};\n"
     "vertex VertexOut gl4metal_program_vertex(VertexIn in [[stage_in]], constant FallbackUniforms& uniforms [[buffer(16)]]) {\n"
     "    VertexOut out;\n"
     "    out.position = uniforms.mvp * in.position;\n"];
    if (color) { [metalSource appendString:@"    out.color = in.color;\n"]; }
    if (texcoord) { [metalSource appendString:@"    out.uv = in.uv;\n"]; }
    [metalSource appendString:@"    return out;\n}\n"
    "fragment half4 gl4metal_program_fragment(VertexOut in [[stage_in]]) {\n"];
    [metalSource appendFormat:@"    float4 color = %@;\n", constantColor ?: @"float4(1.0)"];
    if (color) { [metalSource appendString:@"    color = in.color;\n"]; }
    [metalSource appendString:@"    return half4(color);\n}\n"];

    if (hasColor) *hasColor = color;
    if (hasTexCoord) *hasTexCoord = texcoord;
    return metalSource;
}

BOOL gl4metalCreateDefaultProgramPipelineForProgram(GLuint program) {
    if (!ctx || !ctx.device) return NO;

    if (defaultFallbackPipeline) {
        ctx.pipelineState = defaultFallbackPipeline;
        return YES;
    }

    NSString *source = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct VertexIn {\n"
        "    float4 position [[attribute(0)]];\n"
        "    float4 color [[attribute(1)]];\n"
        "};\n"
        "struct VertexOut {\n"
        "    float4 position [[position]];\n"
        "    float4 color;\n"
        "};\n"
        "struct FallbackUniforms {\n"
        "    float4x4 mvp;\n"
        "};\n"
        "vertex VertexOut gl4metal_default_vertex(VertexIn in [[stage_in]], constant FallbackUniforms& uniforms [[buffer(16)]]) {\n"
        "    VertexOut out;\n"
        "    out.position = uniforms.mvp * in.position;\n"
        "    out.color = in.color;\n"
        "    return out;\n"
        "}\n"
        "fragment half4 gl4metal_default_fragment(VertexOut in [[stage_in]]) {\n"
        "    return half4(in.color);\n"
        "}\n";

    NSError *error = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        printf("[gl4metal] ERROR: Failed to compile default Metal library for program %u: %s\n", program, error.localizedDescription.UTF8String);
        return NO;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"gl4metal_default_vertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"gl4metal_default_fragment"];
    if (!vertexFunction || !fragmentFunction) {
        printf("[gl4metal] ERROR: Default Metal functions missing for program %u\n", program);
        return NO;
    }

    MTLVertexDescriptor *vertexDescriptor = gl4metalCreateVertexDescriptorForCurrentVAO();
    // The fallback shader explicitly expects float4 attributes 0 and 1.
    // Real pipelines do NOT get these defaults.
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat4;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.layouts[0].stride = sizeof(float) * 4;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
    vertexDescriptor.attributes[1].offset = 0;
    vertexDescriptor.attributes[1].bufferIndex = 1;
    vertexDescriptor.layouts[1].stride = sizeof(float) * 4;
    vertexDescriptor.layouts[1].stepRate = 1;
    vertexDescriptor.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;

    BOOL success = gl4metalCreatePipelineState(vertexFunction, fragmentFunction, vertexDescriptor);
    if (success) {
        defaultFallbackPipeline = ctx.pipelineState;
        gl4metalCachePipelineForCurrentLayout(program);
    }
    return success;
}

static void gl4metalDumpGLSLResourceHints(GLuint program, const GLShader *vertexShader, const GLShader *fragmentShader) {
    printf("[gl4metal][RESOURCES] program=%u GLSL resource declarations:\n", program);
    const GLShader *shaders[] = { vertexShader, fragmentShader };
    const char *stageNames[] = { "VERTEX", "FRAGMENT" };
    for (int si = 0; si < 2; ++si) {
        const GLShader *shader = shaders[si];
        if (!shader) continue;
        std::string src = shader->glslSource;
        size_t start = 0;
        bool found = false;
        while (start < src.size()) {
            size_t end = src.find('\n', start);
            if (end == std::string::npos) end = src.size();
            std::string line = src.substr(start, end - start);
            if (line.find("uniform") != std::string::npos ||
                line.find("sampler") != std::string::npos ||
                line.find("image") != std::string::npos ||
                line.find("buffer") != std::string::npos) {
                printf("[gl4metal][RESOURCES]   %s: %s\n", stageNames[si], line.c_str());
                found = true;
            }
            start = end + (end < src.size() ? 1 : 0);
        }
        if (!found) printf("[gl4metal][RESOURCES]   %s: no uniform/sampler/image/buffer declaration found\n", stageNames[si]);
    }
}


static void gl4metalSanitizeVertexDescriptor(MTLVertexDescriptor *descriptor, GLuint program) {
    if (!descriptor) return;

    // Metal rejects any vertex-buffer layout which has a non-zero stride but
    // is not referenced by at least one valid vertex attribute. Re-normalize
    // all valid attribute buffer indices and clear orphan layouts.
    BOOL referenced[16] = { NO };
    for (NSUInteger ai = 0; ai < 16; ++ai) {
        if (descriptor.attributes[ai].format == MTLVertexFormatInvalid) continue;
        NSUInteger bufferIndex = descriptor.attributes[ai].bufferIndex;
        if (bufferIndex >= 16) {
            descriptor.attributes[ai].bufferIndex = ai;
            bufferIndex = ai;
        }
        referenced[bufferIndex] = YES;
    }

    for (NSUInteger bi = 0; bi < 16; ++bi) {
        if (descriptor.layouts[bi].stride != 0 && !referenced[bi]) {
            if (getenv("GL4METAL_PIPELINE_TRACE")) {
                printf("[gl4metal][ATTR] program=%u clearing orphan vertex layout[%lu] stride=%lu\n",
                       program, (unsigned long)bi,
                       (unsigned long)descriptor.layouts[bi].stride);
            }
            descriptor.layouts[bi].stride = 0;
            descriptor.layouts[bi].stepRate = 1;
            descriptor.layouts[bi].stepFunction = MTLVertexStepFunctionPerVertex;
        }
    }
}

BOOL gl4metalCreateProgramPipelineForProgram(GLuint program) {
    if (!ctx || !ctx.device) return NO;

    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (attached == nil || [attached count] == 0) {
        printf("[gl4metal] ERROR: Program %u has no attached shaders; refusing fallback pipeline\n", program);
        return NO;
    }

    GLShader *vertexShader = nullptr;
    GLShader *fragmentShader = nullptr;

    for (NSNumber *shaderId in attached) {
        GLuint shader = shaderId.unsignedIntValue;
        auto it = shaderSources.find(shader);
        if (it == shaderSources.end()) {
            printf("[gl4metal] ERROR: Program %u references missing shader %u\n", program, shader);
            return NO;
        }

        GLShader &shaderData = it->second;
        if (!shaderData.compiled || shaderData.mtlFunction == nil) {
            printf("[gl4metal] ERROR: Program %u references uncompiled shader %u; refusing fallback pipeline\n", program, shader);
            return NO;
        }

        if (shaderData.type == GL_VERTEX_SHADER) {
            vertexShader = &shaderData;
        } else if (shaderData.type == GL_FRAGMENT_SHADER) {
            fragmentShader = &shaderData;
        }
    }

    if (!vertexShader || !fragmentShader) {
        printf("[gl4metal] ERROR: Program %u does not contain both vertex and fragment shaders; refusing fallback pipeline\n", program);
        return NO;
    }

    gl4metalDumpGLSLResourceHints(program, vertexShader, fragmentShader);

    // IMPORTANT: use the functions produced by transpileGLSLToMSL() above.
    // Do NOT regenerate a simplified shader here. The old implementation
    // ignored the real compiled shaders and substituted gl4metal_program_*.
    MTLVertexDescriptor *vertexDescriptor = gl4metalCreateVertexDescriptorForCurrentVAO();
    if (!vertexDescriptor) {
        printf("[gl4metal] ERROR: Failed to create vertex descriptor for program %u\n", program);
        return NO;
    }

    if (!gl4metalApplyVertexShaderInputFormats(
            [NSString stringWithUTF8String:vertexShader->glslSource.c_str()],
            program,
            vertexDescriptor)) {
        printf("[gl4metal] WARNING: No supported vertex input declarations were found in program %u; pipeline will use only GL attribute state\n", program);
    }

    gl4metalSanitizeVertexDescriptor(vertexDescriptor, program);

    BOOL success = gl4metalCreatePipelineStateForProgram(
        program,
        vertexShader->mtlFunction,
        fragmentShader->mtlFunction,
        vertexDescriptor
    );

    if (!success) {
        printf("[gl4metal] ERROR: Failed to create Metal pipeline from real shaders for program %u\n", program);
        return NO;
    }

    gl4metalCachePipelineForCurrentLayout(program);
    return YES;
}

static std::string stripGLSLComments(const std::string &source) {
    std::string out;
    out.reserve(source.size());

    bool inLineComment = false;
    bool inBlockComment = false;
    char stringDelimiter = '\0';
    bool escaped = false;

    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        const char next = (i + 1 < source.size()) ? source[i + 1] : '\0';

        if (stringDelimiter != '\0') {
            out.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == stringDelimiter) {
                stringDelimiter = '\0';
            }
            continue;
        }

        if (inLineComment) {
            // Keep newlines so glslang line/column diagnostics stay useful.
            if (c == '\n' || c == '\r') {
                inLineComment = false;
                out.push_back(c);
            }
            continue;
        }

        if (inBlockComment) {
            // Preserve line structure inside /* ... */ comments.
            if (c == '*' && next == '/') {
                inBlockComment = false;
                ++i;
                continue;
            }
            if (c == '\n' || c == '\r') {
                out.push_back(c);
            }
            continue;
        }

        if (c == '/' && next == '/') {
            if (!out.empty() && !std::isspace((unsigned char)out.back())) out.push_back(' ');
            inLineComment = true;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            if (!out.empty() && !std::isspace((unsigned char)out.back())) out.push_back(' ');
            inBlockComment = true;
            ++i;
            continue;
        }

        if (c == '"' || c == '\'') {
            stringDelimiter = c;
            out.push_back(c);
            continue;
        }

        out.push_back(c);
    }

    if (inBlockComment) out.append("/*");

    // Never feed an unterminated trailing line to glslang.
    if (out.empty() || (out.back() != '\n' && out.back() != '\r')) {
        out.push_back('\n');
    } else if (out.back() == '\r') {
        out.push_back('\n');
    }

    return out;
}

static void replaceGLSLIdentifier(std::string &source, const char *identifier,
                                  const char *replacement) {
    const std::regex token(std::string(R"(\b)") + identifier + R"(\b)");
    source = std::regex_replace(source, token, replacement);
}

static bool insertGLSLDeclarationBeforeCode(std::string &source,
                                             const std::string &declaration) {
    size_t lineStart = 0;
    bool continuingDirective = false;
    while (lineStart < source.size()) {
        size_t lineEnd = source.find('\n', lineStart);
        if (lineEnd == std::string::npos) lineEnd = source.size();
        size_t first = lineStart;
        while (first < lineEnd && (source[first] == ' ' || source[first] == '\t' || source[first] == '\r')) ++first;
        size_t last = lineEnd;
        while (last > first && (source[last - 1] == ' ' || source[last - 1] == '\t' || source[last - 1] == '\r')) --last;

        bool isDirective = first < lineEnd && source[first] == '#';
        bool isBlank = first == lineEnd;
        bool isConditional = isDirective &&
            (source.compare(first, 3, "#if") == 0 ||
             source.compare(first, 5, "#else") == 0 ||
             source.compare(first, 6, "#endif") == 0);
        if (isConditional && !continuingDirective) {
            source.insert(lineStart, declaration);
            return true;
        }
        if (!continuingDirective && !isDirective && !isBlank) {
            source.insert(lineStart, declaration);
            return true;
        }
        continuingDirective = last > first && source[last - 1] == '\\';
        lineStart = lineEnd < source.size() ? lineEnd + 1 : source.size();
    }
    source.append(declaration);
    return true;
}

static std::string gl4metalFlipDiffuseSamplerY(const std::string &source) {
    const std::string samplerName = "DiffuseSampler";
    const std::string textureName = "texture";
    std::string output;
    output.reserve(source.size() + 64);
    size_t cursor = 0;
    bool changed = false;

    while (cursor < source.size()) {
        size_t call = source.find(textureName, cursor);
        if (call == std::string::npos) {
            output.append(source, cursor, std::string::npos);
            break;
        }

        size_t afterName = call + textureName.size();
        const bool leftBoundary = call == 0 ||
            !(std::isalnum((unsigned char)source[call - 1]) || source[call - 1] == '_');
        const bool rightBoundary = afterName == source.size() ||
            !(std::isalnum((unsigned char)source[afterName]) || source[afterName] == '_');
        size_t open = afterName;
        while (open < source.size() && std::isspace((unsigned char)source[open])) ++open;
        if (!leftBoundary || !rightBoundary || open >= source.size() || source[open] != '(') {
            output.append(source, cursor, afterName - cursor);
            cursor = afterName;
            continue;
        }

        size_t samplerStart = open + 1;
        while (samplerStart < source.size() && std::isspace((unsigned char)source[samplerStart])) ++samplerStart;
        const size_t samplerEnd = samplerStart + samplerName.size();
        if (samplerEnd > source.size() || source.compare(samplerStart, samplerName.size(), samplerName) != 0 ||
            (samplerEnd < source.size() &&
             (std::isalnum((unsigned char)source[samplerEnd]) || source[samplerEnd] == '_'))) {
            output.append(source, cursor, afterName - cursor);
            cursor = afterName;
            continue;
        }

        size_t comma = samplerEnd;
        while (comma < source.size() && std::isspace((unsigned char)source[comma])) ++comma;
        if (comma >= source.size() || source[comma] != ',') {
            output.append(source, cursor, samplerEnd - cursor);
            cursor = samplerEnd;
            continue;
        }

        size_t coordinateStart = comma + 1;
        while (coordinateStart < source.size() && std::isspace((unsigned char)source[coordinateStart])) ++coordinateStart;
        size_t callEnd = coordinateStart;
        int depth = 1;
        char quote = '\0';
        bool escaped = false;
        for (; callEnd < source.size(); ++callEnd) {
            const char c = source[callEnd];
            if (quote != '\0') {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == quote) quote = '\0';
                continue;
            }
            if (c == '"' || c == '\'') quote = c;
            else if (c == '(') ++depth;
            else if (c == ')' && --depth == 0) break;
        }
        if (callEnd >= source.size()) {
            output.append(source, cursor, std::string::npos);
            break;
        }

        output.append(source, cursor, call);
        output.append("texture(DiffuseSampler, ((");
        output.append(source, coordinateStart, callEnd - coordinateStart);
        output.append(") * vec2(1.0, -1.0) + vec2(0.0, 1.0)))");
        cursor = callEnd + 1;
        changed = true;
    }

    if (changed) {
        printf("[gl4metal][ORIGIN] flipped framebuffer Y for DiffuseSampler\n");
        return output;
    }
    return source;
}

static bool normalizeGLSLForSPIRV(GLenum shaderType, std::string &source,
                                  std::string *diagnostics) {
    static const std::regex versionDirective(
        R"(#version[ \t]+([0-9]+)(?:[ \t]+(core|compatibility|es))?[^\r\n]*)");
    std::smatch versionMatch;
    const bool hasVersion = std::regex_search(source, versionMatch, versionDirective);
    int sourceVersion = 110;
    if (hasVersion) {
        try {
            sourceVersion = std::stoi(versionMatch[1].str());
        } catch (...) {
            if (diagnostics) *diagnostics = "Invalid GLSL #version directive.";
            return false;
        }
    }
    const std::string profile = hasVersion && versionMatch.size() > 2
        ? versionMatch[2].str() : std::string();

    if (profile == "compatibility" && sourceVersion >= 330) {
        if (diagnostics) *diagnostics = "GLSL compatibility-profile shaders cannot be represented by the OpenGL SPIR-V to Metal path.";
        return false;
    }

    // OpenGL SPIR-V requires desktop GLSL 3.30+. Older valid OpenGL source
    // is upgraded after translating its legacy interface and texture names.
    // Keep newer #version directives intact so their language features are
    // not silently erased by a downgrade to 330.
    const bool upgradeLegacy = !hasVersion || sourceVersion < 330 || profile == "es";
    if (upgradeLegacy) {
        if (hasVersion) source.replace((size_t)versionMatch.position(), (size_t)versionMatch.length(), "#version 330 core");
        else source.insert(0, "#version 330 core\n");

        // Precision qualifiers have no effect in desktop GLSL or Metal's
        // unified float model. Remove ESSL precision statements and tokens.
        source = std::regex_replace(source,
            std::regex(R"(\bprecision\s+(?:lowp|mediump|highp)\s+[A-Za-z_][A-Za-z0-9_]*\s*;\s*)"), "");
        source = std::regex_replace(source,
            std::regex(R"(\b(?:lowp|mediump|highp)\b\s*)"), "");

        if (shaderType == GL_VERTEX_SHADER) {
            replaceGLSLIdentifier(source, "attribute", "in");
            replaceGLSLIdentifier(source, "varying", "out");
        } else {
            replaceGLSLIdentifier(source, "varying", "in");
            replaceGLSLIdentifier(source, "gl_FragColor", "gl4metal_FragColor");
            replaceGLSLIdentifier(source, "gl_FragData", "gl4metal_FragData");
        }

        const char *legacyTextureFunctions[] = {
            "texture1D", "texture2D", "texture3D", "textureCube",
            "texture1DProj", "texture2DProj", "texture3DProj"
        };
        for (const char *name : legacyTextureFunctions) {
            replaceGLSLIdentifier(source, name,
                (std::string(name).find("Proj") != std::string::npos) ? "textureProj" : "texture");
        }
        replaceGLSLIdentifier(source, "shadow1D", "texture");
        replaceGLSLIdentifier(source, "shadow2D", "texture");
        replaceGLSLIdentifier(source, "shadow2DProj", "textureProj");

        if (shaderType == GL_FRAGMENT_SHADER) {
            const bool usesFragColor = source.find("gl4metal_FragColor") != std::string::npos;
            const bool usesFragData = source.find("gl4metal_FragData") != std::string::npos;
            if (usesFragData) {
                source = std::regex_replace(source,
                    std::regex(R"(\bgl4metal_FragData\s*\[\s*0\s*\])"), "gl4metal_FragData0");
                if (std::regex_search(source, std::regex(R"(\bgl4metal_FragData\b)"))) {
                    if (diagnostics) *diagnostics = "Legacy gl_FragData outputs above location 0 are not supported by this single-color Metal backend.";
                    return false;
                }
            }
            std::string outputs;
            if (usesFragColor) outputs += "layout(location = 0) out vec4 gl4metal_FragColor;\n";
            if (usesFragData) outputs += "layout(location = 0) out vec4 gl4metal_FragData0;\n";
            if (!outputs.empty()) insertGLSLDeclarationBeforeCode(source, outputs);
        }
    }

    return true;
}

    static void gl4metalDumpShaderSection(unsigned long long compileId,
                          GLenum shaderType,
                          const char *section,
                          const std::string &source) {
        fprintf(stderr, "\n[gl4metal][SHADER_DUMP][%llu][%s][%s] BEGIN (%zu bytes)\n",
            compileId, shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT",
            section, source.size());
        if (!source.empty()) fwrite(source.data(), 1, source.size(), stderr);
        if (source.empty() || source.back() != '\n') fputc('\n', stderr);
        fprintf(stderr, "[gl4metal][SHADER_DUMP][%llu][%s][%s] END\n",
            compileId, shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT",
            section);
        fflush(stderr);
    }

    static void gl4metalReserveVertexBufferSlots(spirv_cross::CompilerMSL &compiler,
                                                bool dumpShaders,
                                                unsigned long long compileId,
                                                GLenum shaderType) {
        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
        uint32_t nextBufferIndex = 16;
        uint32_t nextPlainUniformBinding = 0;

        const auto reserveExistingBinding = [&](const spirv_cross::Resource &resource) {
            if (!compiler.has_decoration(resource.id, spv::DecorationBinding)) return;
            uint32_t binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
            if (binding >= nextPlainUniformBinding && binding != UINT32_MAX) {
                nextPlainUniformBinding = binding + 1;
            }
        };
        for (const spirv_cross::Resource &resource : resources.uniform_buffers) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.storage_buffers) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.gl_plain_uniforms) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.subpass_inputs) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.storage_images) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.sampled_images) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.atomic_counters) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.acceleration_structures) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.tensors) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.push_constant_buffers) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.shader_record_buffers) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.separate_images) {
            reserveExistingBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.separate_samplers) {
            reserveExistingBinding(resource);
        }

        const auto addBufferBinding = [&](const spirv_cross::Resource &resource) {
            spirv_cross::MSLResourceBinding binding;
            binding.stage = compiler.get_execution_model();
            binding.desc_set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
            binding.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
            binding.msl_buffer = nextBufferIndex++;
            compiler.add_msl_resource_binding(binding);
            if (dumpShaders) {
                fprintf(stderr,
                    "[gl4metal][SHADER_DUMP][%llu][%s][RESOURCE_BINDING] %s set=%u binding=%u -> buffer(%u)\n",
                    compileId, shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT",
                    resource.name.c_str(), binding.desc_set, binding.binding, binding.msl_buffer);
            }
        };

        for (const spirv_cross::Resource &resource : resources.uniform_buffers) {
            addBufferBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.gl_plain_uniforms) {
            uint32_t descriptorSet = compiler.has_decoration(resource.id, spv::DecorationDescriptorSet)
                ? compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) : 0;
            compiler.set_decoration(resource.id, spv::DecorationDescriptorSet, descriptorSet);
            compiler.set_decoration(resource.id, spv::DecorationBinding, nextPlainUniformBinding++);
            addBufferBinding(resource);
        }
        for (const spirv_cross::Resource &resource : resources.storage_buffers) {
            addBufferBinding(resource);
        }
    }

std::string transpileGLSLToMSL(GLenum shaderType, const std::string &glslSource,
                               std::string *diagnostics) {
    if (diagnostics) diagnostics->clear();
    if (shaderType != GL_VERTEX_SHADER && shaderType != GL_FRAGMENT_SHADER) {
        if (diagnostics) *diagnostics = "Only vertex and fragment shaders can be translated to Metal.";
        return "";
    }
    if (glslSource.empty()) {
        if (diagnostics) *diagnostics = "Shader source is empty.";
        return "";
    }

    const char *dumpSetting = getenv("GL4METAL_DUMP_SHADERS");
    const bool dumpShaders = dumpSetting && dumpSetting[0] != '\0' &&
        strcmp(dumpSetting, "0") != 0 && strcmp(dumpSetting, "false") != 0;
    static unsigned long long nextCompileId = 0;
    const unsigned long long compileId = dumpShaders ? ++nextCompileId : 0;
    if (dumpShaders) {
        gl4metalDumpShaderSection(compileId, shaderType, "GLSL_SOURCE", glslSource);
    }

    // Minecraft frequently appends generated // [TAG] annotations. Strip
    // GLSL comments before parsing so glslang never sees an EOF comment.
    //std::string glsl = stripGLSLComments(glslSource);

    //if (!normalizeGLSLForSPIRV(shaderType, glsl, diagnostics)) {
        //return "";
    //}

    if (shaderType == GL_FRAGMENT_SHADER) {
        glsl = gl4metalFlipDiffuseSamplerY(glsl);
    }

    if (dumpShaders) {
        gl4metalDumpShaderSection(compileId, shaderType, "GLSL_TRANSLATION_INPUT", glsl);
    }

    EShLanguage stage = (shaderType == GL_VERTEX_SHADER) ? EShLangVertex : EShLangFragment;

    glslang::TShader shader(stage);
    const char *shaderStrings = glslSource.c_str();
    //if (glsl.size() > (size_t)INT_MAX) {
        //if (diagnostics) *diagnostics = "Shader source exceeds the maximum size supported by glslang.";
        //return "";
    //}
    const int shaderLength = (int)glslSource.size();
    shader.setStringsWithLengths(&shaderStrings, &shaderLength, 1);

    shader.setAutoMapLocations(true);
    shader.setAutoMapBindings(true);

    int defaultVersion = 460;
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientOpenGL, defaultVersion);
    shader.setEnvClient(glslang::EShClientOpenGL, glslang::EShTargetOpenGL_450);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

    const TBuiltInResource *resources = GetDefaultResources();
    EShMessages messages = (EShMessages)(EShMsgDefault | EShMsgSpvRules);

    if (!shader.parse(resources, defaultVersion, false, messages)) {
        std::string message = shader.getInfoLog() ? shader.getInfoLog() : "";
        std::string debug = shader.getInfoDebugLog() ? shader.getInfoDebugLog() : "";
        if (!debug.empty()) message += (message.empty() ? "" : "\n") + debug;
        if (diagnostics) *diagnostics = message.empty() ? "GLSL parsing failed." : message;
        printf("[gl4metal] ERROR: GLSL parsing failed: %s\n", message.c_str());
        return "";
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        std::string message = program.getInfoLog() ? program.getInfoLog() : "";
        std::string debug = program.getInfoDebugLog() ? program.getInfoDebugLog() : "";
        if (!debug.empty()) message += (message.empty() ? "" : "\n") + debug;
        if (diagnostics) *diagnostics = message.empty() ? "GLSL stage linking failed." : message;
        printf("[gl4metal] ERROR: GLSL linking failed: %s\n", message.c_str());
        return "";
    }

    program.mapIO();

    std::vector<unsigned int> spirv;
    const glslang::TIntermediate *intermediate = program.getIntermediate(stage);
    if (!intermediate) {
        if (diagnostics) *diagnostics = "glslang did not produce an intermediate shader representation.";
        return "";
    }
    spv::SpvBuildLogger spirvLogger;
    glslang::GlslangToSpv(*intermediate, spirv, &spirvLogger);

    if (spirv.empty()) {
        std::string message = spirvLogger.getAllMessages();
        if (diagnostics) *diagnostics = message.empty() ? "SPIR-V generation failed." : message;
        printf("[gl4metal] ERROR: SPIR-V generation failed: %s\n", message.c_str());
        return "";
    }

    try {
        spirv_cross::CompilerMSL mslCompiler(spirv);

        spirv_cross::CompilerMSL::Options mslOptions;
        mslOptions.platform = spirv_cross::CompilerMSL::Options::iOS;
        mslOptions.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(2, 1);
        mslCompiler.set_msl_options(mslOptions);
        gl4metalReserveVertexBufferSlots(mslCompiler, dumpShaders, compileId, shaderType);

        std::string mslSource = mslCompiler.compile();
        if (dumpShaders) {
            gl4metalDumpShaderSection(compileId, shaderType, "MSL_SPIRV_CROSS", mslSource);
        }

        // SPIRV-Cross can emit scalar helper parameters as references. Metal
        // does not allow a `thread const T&` parameter to bind to ordinary
        // function arguments. Read-only scalar references are equivalent to
        // pass-by-value here, so remove the reference entirely.
        std::regex constRefRegex(
            R"((?:\bthread\s+)?const\s+(float\d*(?:x\d+)?|int\d*|uint\d*|bool\d*)\s*&)"
        );
        mslSource = std::regex_replace(mslSource, constRefRegex, "$1");

        // Keep non-const scalar references (GLSL inout/out semantics), but
        // make sure Metal has an explicit thread address space. Do this with
        // a small scanner rather than regex look-behind/look-ahead so it also
        // works with Apple's C++ regex implementation.
        const char *scalarTypes[] = { "float", "float2", "float3", "float4", "int", "int2", "int3", "int4", "uint", "uint2", "uint3", "uint4", "bool", "bool2", "bool3", "bool4" };
        for (const char *type : scalarTypes) {
            std::string needle = std::string(type) + "&";
            size_t pos = 0;
            while ((pos = mslSource.find(needle, pos)) != std::string::npos) {
                static const char *addressSpacePrefixes[] = {
                    "constant ", "device ", "threadgroup ", "thread "
                };
                bool hasAddressSpace = false;
                for (const char *prefix : addressSpacePrefixes) {
                    size_t prefixLength = strlen(prefix);
                    if (pos >= prefixLength &&
                        mslSource.compare(pos - prefixLength, prefixLength, prefix) == 0) {
                        hasAddressSpace = true;
                        break;
                    }
                }
                if (!hasAddressSpace) {
                    // Avoid touching struct/resource declarations or pointer-like
                    // syntax; only normalize a type reference token.
                    mslSource.insert(pos, "thread ");
                    pos += 7 + needle.size();
                } else {
                    pos += needle.size();
                }
            }
        }

        if (dumpShaders) {
            gl4metalDumpShaderSection(compileId, shaderType, "MSL_FINAL", mslSource);
        }
        return mslSource;
    } catch (const std::exception &e) {
        if (diagnostics) *diagnostics = e.what();
        printf("[gl4metal] ERROR: SPIR-V to MSL transpilation failed: %s\n", e.what());
        return "";
    }
}
