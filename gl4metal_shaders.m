#import "gl4metal.h"
#import "gl4metal_shaders.h"
#import "gl4metal_runtime.h"
#import "shader_compiler.h"

NSString *gl4metalBuildPipelineCacheKey(GLuint program, gl4metalVertexArray *vao) {
    NSMutableString *key = [NSMutableString stringWithFormat:@"p%u:", program];
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
        [key appendFormat:@"a%lu:%d:%u:%d:%ld;",
         (unsigned long)index,
         attrib->size,
         attrib->type,
         attrib->normalized,
         (long)attrib->stride];
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

static NSString *gl4metalStripComments(NSString *source) {
    if (!source) return @"";
    NSString *withoutBlockComments = [source stringByReplacingOccurrencesOfString:@"/\\*.*?\\*/" withString:@"" options:NSRegularExpressionSearch range:NSMakeRange(0, source.length)];
    return [withoutBlockComments stringByReplacingOccurrencesOfString:@"//[^\\n]*" withString:@"" options:NSRegularExpressionSearch range:NSMakeRange(0, withoutBlockComments.length)];
}

static NSString *gl4metalTypeToMetal(NSString *type) {
    NSDictionary *types = @{
        @"float": @"float", @"vec2": @"float2", @"vec3": @"float3", @"vec4": @"float4",
        @"int": @"int", @"ivec2": @"int2", @"ivec3": @"int3", @"ivec4": @"int4",
        @"uint": @"uint", @"uvec2": @"uint2", @"uvec3": @"uint3", @"uvec4": @"uint4",
        @"bool": @"bool", @"bvec2": @"bool2", @"bvec3": @"bool3", @"bvec4": @"bool4",
        @"mat2": @"float2x2", @"mat3": @"float3x3", @"mat4": @"float4x4"
    };
    return types[type] ?: type;
}

static NSString *gl4metalExtractMainBody(NSString *source) {
    NSRange mainRange = [source rangeOfString:@"main\\s*\\(\\s*\\)\\s*\\{" options:NSRegularExpressionSearch];
    if (mainRange.location == NSNotFound) return nil;

    NSUInteger bodyStart = NSMaxRange(mainRange);
    NSUInteger depth = 1;
    for (NSUInteger index = bodyStart; index < source.length; index++) {
        unichar character = [source characterAtIndex:index];
        if (character == '{') depth++;
        else if (character == '}' && --depth == 0) {
            return [source substringWithRange:NSMakeRange(bodyStart, index - bodyStart)];
        }
    }
    return nil;
}

static NSArray<NSDictionary *> *gl4metalFindDeclarations(NSString *source, NSString *qualifier) {
    NSString *pattern = [NSString stringWithFormat:@"\\b%@\\s+(float|vec[234]|int|ivec[234]|uint|uvec[234]|bool|bvec[234]|mat[234]|sampler2D|samplerCube)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*;", qualifier];
    NSRegularExpression *expression = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
    NSMutableArray *result = [NSMutableArray array];
    for (NSTextCheckingResult *match in [expression matchesInString:source options:0 range:NSMakeRange(0, source.length)]) {
        [result addObject:@{
            @"type": [source substringWithRange:[match rangeAtIndex:1]],
            @"name": [source substringWithRange:[match rangeAtIndex:2]]
        }];
    }
    return result;
}

static NSString *gl4metalUniformExpression(NSDictionary *uniform) {
    NSString *name = uniform[@"name"];
    NSString *type = uniform[@"type"];
    if ([type hasPrefix:@"sampler"] || [type hasPrefix:@"mat"]) return [NSString stringWithFormat:@"uniforms.%@", name];
    if ([type hasPrefix:@"vec"] || [type isEqualToString:@"float"]) return [type isEqualToString:@"float"] ? [NSString stringWithFormat:@"uniforms.%@.x", name] : [NSString stringWithFormat:@"uniforms.%@", name];
    if ([type hasPrefix:@"ivec"] || [type isEqualToString:@"int"]) return [NSString stringWithFormat:@"int4(uniforms.%@).x", name];
    if ([type hasPrefix:@"uvec"] || [type isEqualToString:@"uint"]) return [NSString stringWithFormat:@"uint4(uniforms.%@).x", name];
    if ([type hasPrefix:@"bvec"] || [type isEqualToString:@"bool"]) return [NSString stringWithFormat:@"bool(uniforms.%@.x)", name];
    return [NSString stringWithFormat:@"uniforms.%@", name];
}

void gl4metalRegisterUniformLayout(GLuint program, NSString *vertexSource, NSString *fragmentSource) {
    NSMutableArray<NSDictionary *> *layout = [NSMutableArray array];
    NSArray *declarations = [gl4metalFindDeclarations(gl4metalStripComments(vertexSource ?: @""), @"uniform") arrayByAddingObjectsFromArray:gl4metalFindDeclarations(gl4metalStripComments(fragmentSource ?: @""), @"uniform")];
    NSMutableSet<NSString *> *seen = [NSMutableSet set];
    NSUInteger slot = 0;
    for (NSDictionary *declaration in declarations) {
        NSString *name = declaration[@"name"];
        if ([seen containsObject:name]) continue;
        [seen addObject:name];
        if ([declaration[@"type"] hasPrefix:@"sampler"]) {
            [layout addObject:@{ @"name": name, @"type": declaration[@"type"], @"textureIndex": @(slot) }];
            slot++;
            continue;
        }
        NSUInteger slots = [declaration[@"type"] hasPrefix:@"mat"] ? 4 : 1;
        [layout addObject:@{ @"name": name, @"type": declaration[@"type"], @"slot": @(slot) }];
        slot += slots;
    }
    uniformLayouts[@(program)] = layout;
}

static GLint gl4metalFindUniformLocation(NSString *name) {
    for (NSNumber *location in uniformLocationNames) {
        if ([uniformLocationNames[location] isEqualToString:name]) return location.intValue;
    }
    return -1;
}

NSData *gl4metalBuildUniformDataForProgram(GLuint program) {
    NSArray<NSDictionary *> *layout = uniformLayouts[@(program)];
    NSUInteger slotCount = 1;
    for (NSDictionary *item in layout) {
        if ([item[@"type"] hasPrefix:@"sampler"]) continue;
        slotCount = MAX(slotCount, [(NSNumber *)item[@"slot"] unsignedIntegerValue] + ([item[@"type"] hasPrefix:@"mat"] ? 4 : 1));
    }
    NSMutableData *data = [NSMutableData dataWithLength:(64 + slotCount * 16)];
    memcpy(data.mutableBytes, cachedFallbackMVP, sizeof(cachedFallbackMVP));
    for (NSDictionary *item in layout) {
        NSString *type = item[@"type"];
        if ([type hasPrefix:@"sampler"]) continue;
        GLint location = gl4metalFindUniformLocation(item[@"name"]);
        if (location < 0) continue;
        NSUInteger offset = 64 + [(NSNumber *)item[@"slot"] unsignedIntegerValue] * 16;
        void *destination = (char *)data.mutableBytes + offset;
        if ([type isEqualToString:@"mat4"]) {
            NSValue *value = uniformMatrix4fvValues[@(location)];
            if (value) [value getValue:destination];
        } else {
            GLfloat values[4] = {0, 0, 0, 0};
            NSValue *value = nil;
            if ([type isEqualToString:@"float"]) value = uniform1fValues[@(location)];
            else if ([type isEqualToString:@"vec2"]) value = uniform2fValues[@(location)];
            else if ([type isEqualToString:@"vec3"]) value = uniform3fValues[@(location)];
            else if ([type isEqualToString:@"vec4"]) value = uniform4fValues[@(location)];
            if (value) {
                [value getValue:values];
            } else if ([type hasPrefix:@"int"] || [type hasPrefix:@"uint"] || [type hasPrefix:@"bool"]) {
                NSNumber *integer = uniform1iValues[@(location)];
                values[0] = integer.floatValue;
            }
            memcpy(destination, values, sizeof(values));
        }
    }
    return data;
}

void gl4metalBindFragmentResources(id<MTLRenderCommandEncoder> encoder, GLuint program) {
    if (!encoder) return;
    NSUInteger fallbackIndex = 0;
    for (NSDictionary *item in uniformLayouts[@(program)]) {
        if (![item[@"type"] hasPrefix:@"sampler"]) continue;
        GLint location = gl4metalFindUniformLocation(item[@"name"]);
        NSUInteger unit = location >= 0 ? [uniform1iValues[@(location)] unsignedIntegerValue] : 0;
        id<MTLTexture> texture = textureObjects[@([boundTextureUnits[@(unit)] unsignedIntValue])];
        NSUInteger textureIndex = [(NSNumber *)item[@"textureIndex"] unsignedIntegerValue];
        if (!texture) texture = textureObjects[@([boundTextureUnits[@(fallbackIndex)] unsignedIntValue])];
        if (texture) [encoder setFragmentTexture:texture atIndex:textureIndex];
        if (ctx.defaultSamplerState) [encoder setFragmentSamplerState:ctx.defaultSamplerState atIndex:textureIndex];
        fallbackIndex++;
    }
}

static NSString *gl4metalRewriteBody(NSString *body, BOOL vertex, NSArray<NSDictionary *> *uniforms, NSArray<NSDictionary *> *inputs) {
    if (!body) return nil;
    NSString *result = [body copy];
    NSDictionary *replacements = @{
        @"vec2": @"float2", @"vec3": @"float3", @"vec4": @"float4",
        @"ivec2": @"int2", @"ivec3": @"int3", @"ivec4": @"int4",
        @"uvec2": @"uint2", @"uvec3": @"uint3", @"uvec4": @"uint4",
        @"mat2": @"float2x2", @"mat3": @"float3x3", @"mat4": @"float4x4",
        @"texture2D": @"texture", @"textureCube": @"texture"
    };
    for (NSString *glslType in replacements) {
        result = [result stringByReplacingOccurrencesOfString:glslType withString:replacements[glslType]];
    }
    result = [result stringByReplacingOccurrencesOfString:@"gl_Position" withString:@"out.position"];
    result = [result stringByReplacingOccurrencesOfString:@"gl_FragColor" withString:@"out.color"];
    result = [result stringByReplacingOccurrencesOfString:@"gl_FragDepth" withString:@"out.depth"];
    result = [result stringByReplacingOccurrencesOfString:@"discard" withString:@"discard_fragment()"];

    for (NSDictionary *uniform in uniforms) {
        NSString *name = uniform[@"name"];
        if ([uniform[@"type"] hasPrefix:@"sampler"]) continue;
        NSRegularExpression *expression = [NSRegularExpression regularExpressionWithPattern:[NSString stringWithFormat:@"\\b%@\\b", name] options:0 error:nil];
        result = [expression stringByReplacingMatchesInString:result options:0 range:NSMakeRange(0, result.length) withTemplate:gl4metalUniformExpression(uniform)];
    }
    NSRegularExpression *textureExpression = [NSRegularExpression regularExpressionWithPattern:@"texture\\s*\\(\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*,\\s*([^\\)]+)\\)" options:0 error:nil];
    result = [textureExpression stringByReplacingMatchesInString:result options:0 range:NSMakeRange(0, result.length) withTemplate:@"$1.sample($1Sampler, $2)"];
    for (NSDictionary *input in inputs) {
        NSString *name = input[@"name"];
        NSRegularExpression *expression = [NSRegularExpression regularExpressionWithPattern:[NSString stringWithFormat:@"(?<![.A-Za-z0-9_])%@\\b", name] options:0 error:nil];
        NSString *prefix = vertex ? @"in" : @"in";
        result = [expression stringByReplacingMatchesInString:result options:0 range:NSMakeRange(0, result.length) withTemplate:[NSString stringWithFormat:@"%@.%@", prefix, name]];
    }
    (void)vertex;
    return result;
}

NSString *gl4metalBuildTranslatedMetalSource(NSString *vertexSource, NSString *fragmentSource, BOOL *hasColor, BOOL *hasTexCoord) {
    if (vertexSource == nil) {
        vertexSource = @"attribute vec4 aPosition; attribute vec4 aColor; varying vec4 vColor; void main() { vColor = aColor; gl_Position = aPosition; }";
    }
    if (fragmentSource == nil) {
        fragmentSource = @"varying vec4 vColor; void main() { gl_FragColor = vColor; }";
    }

    char *compiledMSL = gl4metalCompileGLSLToMSL(vertexSource.UTF8String, fragmentSource.UTF8String);
    if (compiledMSL) {
        NSString *metalSource = [NSString stringWithUTF8String:compiledMSL];
        gl4metalFreeCompiledMSL(compiledMSL);
        if (hasColor) *hasColor = gl4metalShaderUsesAttribute(vertexSource, @"aColor") || gl4metalShaderUsesAttribute(fragmentSource, @"vColor");
        if (hasTexCoord) *hasTexCoord = gl4metalShaderUsesAttribute(vertexSource, @"aTexCoord") || gl4metalShaderUsesAttribute(vertexSource, @"aUV");
        return metalSource;
    }

    #if 0
        vertexSource = gl4metalStripComments(vertexSource);
        fragmentSource = gl4metalStripComments(fragmentSource);
        NSArray *attributes = [gl4metalFindDeclarations(vertexSource, @"attribute") arrayByAddingObjectsFromArray:gl4metalFindDeclarations(vertexSource, @"in")];
        NSArray *vertexVaryings = [gl4metalFindDeclarations(vertexSource, @"varying") arrayByAddingObjectsFromArray:gl4metalFindDeclarations(vertexSource, @"out")];
        NSArray *fragmentVaryings = [gl4metalFindDeclarations(fragmentSource, @"varying") arrayByAddingObjectsFromArray:gl4metalFindDeclarations(fragmentSource, @"in")];
        NSArray *uniforms = [gl4metalFindDeclarations(vertexSource, @"uniform") arrayByAddingObjectsFromArray:gl4metalFindDeclarations(fragmentSource, @"uniform")];
        NSString *vertexBody = gl4metalExtractMainBody(vertexSource);
        NSString *fragmentBody = gl4metalExtractMainBody(fragmentSource);
        if (!vertexBody || !fragmentBody) return nil;

        BOOL color = gl4metalShaderUsesAttribute(vertexSource, @"aColor") || gl4metalShaderUsesAttribute(vertexSource, @"color") || gl4metalShaderUsesAttribute(fragmentSource, @"vColor") || gl4metalShaderUsesAttribute(fragmentSource, @"color");
        BOOL texcoord = gl4metalShaderUsesAttribute(vertexSource, @"aTexCoord") || gl4metalShaderUsesAttribute(vertexSource, @"aUV") || gl4metalShaderUsesAttribute(vertexSource, @"uv") || gl4metalShaderUsesAttribute(vertexSource, @"texCoord");

    NSMutableString *metalSource = [[NSMutableString alloc] init];
    [metalSource appendString:@"#include <metal_stdlib>\nusing namespace metal;\nstruct VertexIn {\n"];
    NSUInteger attributeIndex = 0;
    NSMutableSet<NSNumber *> *usedAttributeLocations = [NSMutableSet set];
    for (NSDictionary *attribute in attributes) {
        NSString *type = gl4metalTypeToMetal(attribute[@"type"]);
        NSString *name = attribute[@"name"];
        NSUInteger location = attributeIndex++;
        NSString *lowerName = [name lowercaseString];
        if ([lowerName containsString:@"position"] || [lowerName isEqualToString:@"apos"] || [lowerName isEqualToString:@"vertex"]) location = 0;
        else if ([lowerName containsString:@"color"]) location = 1;
        else if ([lowerName containsString:@"tex"] || [lowerName containsString:@"uv"]) location = 2;
        while ([usedAttributeLocations containsObject:@(location)]) location = attributeIndex++;
        [usedAttributeLocations addObject:@(location)];
        [metalSource appendFormat:@"    %@ %@ [[attribute(%lu)]];\n", type, name, (unsigned long)location];
    }
    if (attributes.count == 0) [metalSource appendString:@"    float4 aPosition [[attribute(0)]];\n"];
    [metalSource appendString:@"};\n"
     "struct VertexOut {\n"
     "    float4 position [[position]];\n"];
    NSMutableArray *varyingNames = [NSMutableArray array];
    for (NSDictionary *varying in vertexVaryings) {
        NSString *name = varying[@"name"];
        [varyingNames addObject:name];
        [metalSource appendFormat:@"    %@ %@;\n", gl4metalTypeToMetal(varying[@"type"]), name];
    }
    if (color && ![varyingNames containsObject:@"vColor"]) [metalSource appendString:@"    float4 vColor;\n"];
    if (texcoord && ![varyingNames containsObject:@"vTexCoord"]) [metalSource appendString:@"    float2 vTexCoord;\n"];
    [metalSource appendString:@"    float4 color;\n"];
    [metalSource appendString:@"};\nstruct Uniforms {\n    float4x4 mvp;\n"];
    for (NSDictionary *uniform in uniforms) {
        if (![uniform[@"name"] isEqualToString:@"mvp"] && ![uniform[@"type"] hasPrefix:@"sampler"]) {
            NSString *fieldType = [uniform[@"type"] hasPrefix:@"mat"] ? gl4metalTypeToMetal(uniform[@"type"]) : @"float4";
            [metalSource appendFormat:@"    %@ %@;\n", fieldType, uniform[@"name"]];
        }
    }
    [metalSource appendString:@"};\n"];
    [metalSource appendString:@"vertex VertexOut gl4metal_program_vertex(VertexIn in [[stage_in]], constant Uniforms& uniforms [[buffer(16)]]) {\n    VertexOut out;\n"];
    NSString *rewrittenVertexBody = gl4metalRewriteBody(vertexBody, YES, uniforms, attributes);
    [metalSource appendString:rewrittenVertexBody];
    if ([rewrittenVertexBody rangeOfString:@"out.position"].location == NSNotFound) [metalSource appendString:@"    out.position = uniforms.mvp * float4(in.aPosition.xyz, 1.0);\n"];
    [metalSource appendString:@"    return out;\n}\nfragment half4 gl4metal_program_fragment(VertexOut in [[stage_in]], constant Uniforms& uniforms [[buffer(16)]]"];
    NSUInteger textureIndex = 0;
    for (NSDictionary *uniform in uniforms) {
        if ([uniform[@"type"] hasPrefix:@"sampler"]) {
            NSUInteger currentTextureIndex = textureIndex++;
            [metalSource appendFormat:@", texture2d<float> %@ [[texture(%lu)]], sampler %@Sampler [[sampler(%lu)]]", uniform[@"name"], (unsigned long)currentTextureIndex, uniform[@"name"], (unsigned long)currentTextureIndex];
        }
    }
    [metalSource appendString:@") {\n    VertexOut out = in;\n"];
    NSString *rewrittenFragmentBody = gl4metalRewriteBody(fragmentBody, NO, uniforms, fragmentVaryings);
    [metalSource appendString:rewrittenFragmentBody];
    if ([rewrittenFragmentBody rangeOfString:@"out.color"].location == NSNotFound) [metalSource appendString:@"    out.color = in.vColor;\n"];
    [metalSource appendString:@"    return half4(out.color);\n}\n"];

    if (hasColor) *hasColor = color;
    if (hasTexCoord) *hasTexCoord = texcoord;
    return metalSource;
#endif
    if (hasColor) *hasColor = NO;
    if (hasTexCoord) *hasTexCoord = NO;
    return nil;
}

#if 0
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
        NSLog(@"[gl4metal] ERROR: Failed to compile default Metal library for program %u: %@", program, error.localizedDescription);
        return NO;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"gl4metal_default_vertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"gl4metal_default_fragment"];
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"[gl4metal] ERROR: Default Metal functions missing for program %u", program);
        return NO;
    }

    MTLVertexDescriptor *vertexDescriptor = gl4metalCreateVertexDescriptorForCurrentVAO();
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[1].bufferIndex = 1;

    BOOL success = gl4metalCreatePipelineState(vertexFunction, fragmentFunction, vertexDescriptor);
    if (success) {
        defaultFallbackPipeline = ctx.pipelineState;
        gl4metalCachePipelineForCurrentLayout(program);
    }
    return success;
}
#endif

BOOL gl4metalCreateProgramPipelineForProgram(GLuint program) {
    if (!ctx || !ctx.device) return NO;

    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (attached == nil || [attached count] == 0) {
        return NO;
    }

    NSString *vertexSource = nil;
    NSString *fragmentSource = nil;
    for (NSNumber *shaderId in attached) {
        GLenum type = [shaderTypes[shaderId] unsignedIntValue];
        NSString *source = shaderSources[shaderId];
        if (type == GL_VERTEX_SHADER && source != nil) {
            vertexSource = source;
        } else if (type == GL_FRAGMENT_SHADER && source != nil) {
            fragmentSource = source;
        }
    }

    BOOL hasColor = NO;
    BOOL hasTexCoord = NO;
    gl4metalRegisterUniformLayout(program, vertexSource, fragmentSource);
    NSString *metalSource = gl4metalBuildTranslatedMetalSource(vertexSource, fragmentSource, &hasColor, &hasTexCoord);
    if (metalSource == nil || metalSource.length == 0) {
        return NO;
    }

    id<MTLLibrary> library = programMetalLibraries[@(program)];
    if (!library) {
        NSError *error = nil;
        library = [ctx.device newLibraryWithSource:metalSource options:nil error:&error];
        if (!library) {
            NSLog(@"[gl4metal] ERROR: Failed to compile translated Metal library for program %u: %@", program, error.localizedDescription);
            return NO;
        }
        programMetalLibraries[@(program)] = library;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"gl4metal_program_vertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"gl4metal_program_fragment"];
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"[gl4metal] ERROR: translated shader entry points missing for program %u", program);
        return NO;
    }

    MTLVertexDescriptor *vertexDescriptor = gl4metalCreateVertexDescriptorForCurrentVAO();
    vertexDescriptor.attributes[0].bufferIndex = 0;
    if (hasColor) { vertexDescriptor.attributes[1].bufferIndex = 1; }
    if (hasTexCoord) { vertexDescriptor.attributes[2].bufferIndex = 2; }

    BOOL success = gl4metalCreatePipelineState(vertexFunction, fragmentFunction, vertexDescriptor);
    if (success) {
        gl4metalCachePipelineForCurrentLayout(program);
    }
    return success;
}
