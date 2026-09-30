#import "gl4metal.h"
#import "gl4metal_shaders.h"
#import "gl4metal_runtime.h"

#include <regex>
#include <algorithm>
#include <set>
#include <cctype>

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

// Rewrite framebuffer/post-process sampling without changing ordinary texture
// uploads. Minecraft's DiffuseSampler is used for compositor framebuffer
// textures; their Metal render-target origin is opposite to GL texture-space
// Y. Applying the flip at this exact sampling site avoids double-flipping
// normal asset textures such as the loading-screen texture.
static std::string gl4metalRewriteDiffuseSamplerY(const std::string &source) {
    if (source.find("DiffuseSampler") == std::string::npos) return source;
    if (getenv("GL4METAL_NO_FBO_Y_FLIP")) return source;

    std::string out;
    out.reserve(source.size() + 128);
    size_t cursor = 0;
    bool changed = false;

    while (cursor < source.size()) {
        size_t pos = source.find("texture", cursor);
        if (pos == std::string::npos) {
            out.append(source, cursor, std::string::npos);
            break;
        }

        // Avoid identifiers such as textureProj or mytexture.
        bool leftOk = (pos == 0) || !(std::isalnum((unsigned char)source[pos - 1]) || source[pos - 1] == '_');
        size_t afterName = pos + 7;
        bool exactTextureName = afterName <= source.size() &&
            (afterName == source.size() || std::isspace((unsigned char)source[afterName]) || source[afterName] == '(');
        size_t open = source.find('(', afterName);
        if (!leftOk || !exactTextureName || open == std::string::npos) {
            out.append(source, cursor, afterName - cursor);
            cursor = afterName;
            continue;
        }

        // Find the first argument and the top-level comma.
        size_t i = open + 1;
        while (i < source.size() && std::isspace((unsigned char)source[i])) ++i;
        const std::string samplerName = "DiffuseSampler";
        if (source.compare(i, samplerName.size(), samplerName) != 0) {
            out.append(source, cursor, open + 1 - cursor);
            cursor = open + 1;
            continue;
        }
        i += samplerName.size();
        while (i < source.size() && std::isspace((unsigned char)source[i])) ++i;
        if (i >= source.size() || source[i] != ',') {
            out.append(source, cursor, i - cursor);
            cursor = i;
            continue;
        }

        size_t exprStart = i + 1;
        while (exprStart < source.size() && std::isspace((unsigned char)source[exprStart])) ++exprStart;

        int depth = 1;
        bool inString = false;
        char stringQuote = '\0';
        size_t exprEnd = exprStart;
        for (; exprEnd < source.size(); ++exprEnd) {
            char c = source[exprEnd];
            if (inString) {
                if (c == stringQuote && (exprEnd == 0 || source[exprEnd - 1] != '\\')) inString = false;
                continue;
            }
            if (c == '"' || c == '\'') {
                inString = true;
                stringQuote = c;
            } else if (c == '(') {
                ++depth;
            } else if (c == ')') {
                --depth;
                if (depth == 0) break;
            }
        }
        if (exprEnd >= source.size()) {
            out.append(source, cursor, std::string::npos);
            cursor = source.size();
            break;
        }

        std::string coordExpr = source.substr(exprStart, exprEnd - exprStart);
        out.append(source, cursor, pos - cursor);
        out.append("texture(DiffuseSampler, vec2((");
        out.append(coordExpr);
        out.append(").x, 1.0 - (" );
        out.append(coordExpr);
        out.append(").y))");
        cursor = exprEnd + 1;
        changed = true;
    }

    if (changed) {
        printf("[gl4metal][ORIGIN] DiffuseSampler Y-flip enabled for framebuffer/post-process sampling\n");
        return out;
    }
    return source;
}

static std::string stripGLSLComments(const std::string &source) {
    std::string out;
    out.reserve(source.size());

    bool inLineComment = false;
    bool inBlockComment = false;

    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        const char next = (i + 1 < source.size()) ? source[i + 1] : '\0';

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
            inLineComment = true;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            inBlockComment = true;
            ++i;
            continue;
        }

        out.push_back(c);
    }

    // Never feed an unterminated trailing line to glslang.
    if (out.empty() || (out.back() != '\n' && out.back() != '\r')) {
        out.push_back('\n');
    } else if (out.back() == '\r') {
        out.push_back('\n');
    }

    return out;
}

std::string transpileGLSLToMSL(GLenum shaderType, const std::string &glslSource) {
    if (glslSource.empty()) return "";

    // Minecraft frequently appends generated // [TAG] annotations. Strip
    // GLSL comments before parsing so glslang never sees an EOF comment.
    std::string glsl = stripGLSLComments(glslSource);

    // Framebuffer/post-process textures use GL texture coordinates against
    // Metal render-target origin. Keep ordinary texture uploads untouched and
    // correct only DiffuseSampler reads here.
    glsl = gl4metalRewriteDiffuseSamplerY(glsl);

    std::regex versionRegex(R"(#version\s+\d+([^\r\n]*))");

    if (std::regex_search(glsl, versionRegex)) {
        glsl = std::regex_replace(glsl, versionRegex, "#version 330 core");
    } else {
        glsl = "#version 330 core\n" + glsl;
    }

    EShLanguage stage = (shaderType == GL_VERTEX_SHADER) ? EShLangVertex : EShLangFragment;

    glslang::TShader shader(stage);
    const char *shaderStrings = glsl.c_str();
    shader.setStrings(&shaderStrings, 1);

    shader.setAutoMapLocations(true);
    shader.setAutoMapBindings(true);

    int defaultVersion = 330;
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientOpenGL, defaultVersion);
    shader.setEnvClient(glslang::EShClientOpenGL, glslang::EShTargetOpenGL_450);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

    const TBuiltInResource *resources = GetDefaultResources();
    EShMessages messages = (EShMessages)(EShMsgDefault | EShMsgSpvRules);

    if (!shader.parse(resources, defaultVersion, false, messages)) {
        printf("[gl4metal] ERROR: GLSL parsing failed: %s\n", shader.getInfoLog());
        return "";
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        printf("[gl4metal] ERROR: GLSL linking failed: %s\n", program.getInfoLog());
        return "";
    }

    program.mapIO();

    std::vector<unsigned int> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);

    if (spirv.empty()) {
        printf("[gl4metal] ERROR: SPIR-V generation failed\n");
        return "";
    }

    try {
        spirv_cross::CompilerMSL mslCompiler(spirv);

        spirv_cross::CompilerMSL::Options mslOptions;
        mslOptions.platform = spirv_cross::CompilerMSL::Options::iOS;
        mslOptions.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(2, 1);
        mslCompiler.set_msl_options(mslOptions);

        std::string mslSource = mslCompiler.compile();

        // Keep Metal buffer(0..15) reserved for vertex-input resources.
        // SPIRV-Cross may place GLSL uniforms in the same low buffer slots,
        // which collides with the GL4Metal vertex-buffer namespace. Remap
        // known GLSL uniform declarations to a separate buffer namespace.
        // Vertex and fragment stages have independent Metal buffer namespaces.
        // We intentionally keep texture()/sampler() indices unchanged.
        {
            std::regex uniformDecl(
                R"(\buniform\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*;)"
            );
            std::vector<std::string> uniformNames;
            for (std::sregex_iterator it(glsl.begin(), glsl.end(), uniformDecl), end; it != end; ++it) {
                uniformNames.push_back((*it)[1].str());
            }

            // Also catch sampler/image uniforms. These are not remapped as
            // buffers, but are included here so the name scan stays explicit.
            std::regex uniformAny(
                R"(\buniform\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*;)"
            );
            for (std::sregex_iterator it(glsl.begin(), glsl.end(), uniformAny), end; it != end; ++it) {
                std::string n = (*it)[1].str();
                if (std::find(uniformNames.begin(), uniformNames.end(), n) == uniformNames.end())
                    uniformNames.push_back(n);
            }

            // Metal buffer namespaces are independent for vertex and fragment
            // stages, but resource indices must still be unique *within one
            // stage*. Do not give ColorModulator and ProjMat the same slot in
            // a fragment shader (both can be present there).
            //
            // Keep 0..15 reserved for vertex attributes and put uniforms at
            // 16+. Give the common Minecraft uniforms deterministic,
            // stage-local slots, then allocate the remaining uniforms after
            // them.
            const bool isFragmentStage = (shaderType == GL_FRAGMENT_SHADER);
            std::set<NSUInteger> usedUniformSlots;
            // Reserve every fixed slot before allocating generic uniforms.
            // This prevents declaration-order dependent collisions when a
            // generic uniform appears before ProjMat/ModelViewMat/etc.
            if (!isFragmentStage) {
                usedUniformSlots.insert(16); // ProjMat
                usedUniformSlots.insert(17); // ModelViewMat
                usedUniformSlots.insert(18); // ColorModulator, if present
            } else {
                usedUniformSlots.insert(16); // ColorModulator, if present
                usedUniformSlots.insert(17); // ProjMat, if present
                usedUniformSlots.insert(18); // ModelViewMat, if present
            }
            NSUInteger nextUniformSlot = 19;

            for (const std::string &name : uniformNames) {
                NSUInteger slot = 0;
                bool isFixedSlot = false;

                if (!isFragmentStage) {
                    // Vertex stage: matrices occupy the first uniform slots.
                    if (name == "ProjMat") { slot = 16; isFixedSlot = true; }
                    else if (name == "ModelViewMat") { slot = 17; isFixedSlot = true; }
                    else if (name == "ColorModulator") { slot = 18; isFixedSlot = true; }
                } else {
                    // Fragment stage: ColorModulator and ProjMat are both legal
                    // here, so they MUST have different Metal buffer indices.
                    if (name == "ColorModulator") { slot = 16; isFixedSlot = true; }
                    else if (name == "ProjMat") { slot = 17; isFixedSlot = true; }
                    else if (name == "ModelViewMat") { slot = 18; isFixedSlot = true; }
                }

                if (!isFixedSlot) {
                    while (usedUniformSlots.count(nextUniformSlot) != 0) ++nextUniformSlot;
                    slot = nextUniformSlot++;
                }
                usedUniformSlots.insert(slot);

                // Only remap a declaration that SPIRV-Cross emitted as a
                // Metal buffer. Samplers/textures use different attributes.
                std::string marker = name + " [[buffer(";
                size_t search = 0;
                bool changed = false;
                while ((search = mslSource.find(marker, search)) != std::string::npos) {
                    size_t numberStart = search + marker.size();
                    size_t numberEnd = mslSource.find(")]]", numberStart);
                    if (numberEnd == std::string::npos) break;
                    std::string replacement = name + " [[buffer(" + std::to_string(slot) + ")]]";
                    mslSource.replace(search, numberEnd + 3 - search, replacement);
                    printf("[gl4metal][RESOURCES] remapped uniform %s -> buffer(%lu) stage=%s\n",
                           name.c_str(), (unsigned long)slot, isFragmentStage ? "fragment" : "vertex");
                    search += replacement.size();
                    changed = true;
                }
                (void)changed;
            }
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
                bool alreadyThread = pos >= 7 && mslSource.compare(pos - 7, 7, "thread ") == 0;
                if (!alreadyThread) {
                    // Avoid touching struct/resource declarations or pointer-like
                    // syntax; only normalize a type reference token.
                    mslSource.insert(pos, "thread ");
                    pos += 7 + needle.size();
                } else {
                    pos += needle.size();
                }
            }
        }

        return mslSource;
    } catch (const std::exception &e) {
        printf("[gl4metal] ERROR: SPIR-V to MSL transpilation failed: %s\n", e.what());
        return "";
    }
}