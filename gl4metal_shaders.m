#import "gl4metal.h"
#import "gl4metal_shaders.h"
#import "gl4metal_runtime.h"

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
        [key appendFormat:@"a%lu:%d:%u:%d:%ld:%u;",
         (unsigned long)index,
         attrib->size,
         attrib->type,
         attrib->normalized,
         (long)attrib->stride,
         attrib->boundVBO];
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

BOOL gl4metalCreateProgramPipelineForProgram(GLuint program) {
    if (!ctx || !ctx.device) return NO;

    NSMutableArray<NSNumber *> *attached = programShaders[@(program)];
    if (attached == nil || [attached count] == 0) {
        return gl4metalCreateDefaultProgramPipelineForProgram(program);
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
    NSString *metalSource = gl4metalBuildTranslatedMetalSource(vertexSource, fragmentSource, &hasColor, &hasTexCoord);
    if (metalSource == nil || metalSource.length == 0) {
        return gl4metalCreateDefaultProgramPipelineForProgram(program);
    }

    NSError *error = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:metalSource options:nil error:&error];
    if (!library) {
        NSLog(@"[gl4metal] ERROR: Failed to compile translated Metal library for program %u: %@", program, error.localizedDescription);
        return gl4metalCreateDefaultProgramPipelineForProgram(program);
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"gl4metal_program_vertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"gl4metal_program_fragment"];
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"[gl4metal] WARNING: program %u used translated shader source but Metal entry points missing; falling back.", program);
        return gl4metalCreateDefaultProgramPipelineForProgram(program);
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
