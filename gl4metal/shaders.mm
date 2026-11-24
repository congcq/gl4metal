#import "shaders.h"
#include <GL/glcorearb.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <new>
#include <stdexcept>

GLuint glCreateShader(GLenum type) {
	if (type != GL_VERTEX_SHADER && type != GL_FRAGMENT_SHADER && type != GL_COMPUTE_SHADER) {
		gl4metalSetError(GL_INVALID_ENUM);
		GL4METAL_LOG("glCreateShader: unsupported shader type 0x%X (Metal has no geometry/tessellation stage in this layer)", type);
		return 0;
	}
	GLuint shader = ++ctx.next_shader_id;
	ctx.shader_objects[shader].type = type;
	return shader;
}

static bool shaderIsAttached(GLuint shader) {
	for (const auto &program : ctx.program_objects) {
		for (GLuint attached : program.second.shaders) {
			if (attached == shader) {
				return true;
			}
		}
	}
	return false;
}

void releaseShaderIfUnused(GLuint shader) {
	auto it = ctx.shader_objects.find(shader);
	if (it != ctx.shader_objects.end() && it->second.deleted && !shaderIsAttached(shader)) {
		ctx.shader_objects.erase(it);
	}
}

void glDeleteShader(GLuint shader) {
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		return;
	}
	it->second.deleted = GL_TRUE; // the object lives on while a program still has it attached
	releaseShaderIfUnused(shader);
}

GLboolean glIsShader(GLuint shader) {
	return ctx.shader_objects.find(shader) != ctx.shader_objects.end();
}

void glShaderSource(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length) {
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glShaderSource: invalid shader %u", shader);
		return;
	}
	if (count < 0 || (count > 0 && !string)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	std::string source;
	try {
		for (GLsizei stringIndex = 0; stringIndex < count; ++stringIndex) {
			if (!string[stringIndex]) {
				gl4metalSetError(GL_INVALID_VALUE);
				return;
			}
			if (length && length[stringIndex] >= 0) {
				source.append(string[stringIndex], (size_t)length[stringIndex]);
			} else {
				source.append(string[stringIndex]);
			}
		}
	} catch (const std::bad_alloc &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	} catch (const std::length_error &) {
		gl4metalSetError(GL_OUT_OF_MEMORY);
		return;
	}
	ShaderObject &shaderObject = it->second;
	shaderObject.source.swap(source);
	// Replacing source invalidates only the shader object's compile result. A
	// previously linked program executable remains valid until it is relinked.
	shaderObject.compiled = GL_FALSE;
	shaderObject.compiled_stage.reset();
	shaderObject.info_log.clear();
}

void glCompileShader(GLuint shader) {
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glCompileShader: invalid shader %u", shader);
		return;
	}
	ShaderObject &shaderObject = it->second;
	shaderObject.compiled = GL_FALSE;
	shaderObject.compiled_stage.reset();
	shaderObject.info_log.clear();
	if (shaderObject.source.empty()) {
		shaderObject.info_log = "shader source is empty\n";
		return;
	}

	// glCompileShader performs the real GLSL stage parse and validation. The
	// parsed glslang intermediate is retained for the first program link, where
	// cross-stage validation, SPIR-V generation and MSL translation take place.
	shaderObject.compiled_stage = compileShaderStage((uint32_t)shaderObject.type,
		shaderObject.source, shaderObject.info_log);
	shaderObject.compiled = shaderObject.compiled_stage ? GL_TRUE : GL_FALSE;
	if (!shaderObject.compiled) {
		GL4METAL_LOG("glCompileShader(%u) failed:\n%s", shader, shaderObject.info_log.c_str());
		return;
	}

	const char *dump = getenv("GL4METAL_DUMP_SHADERS");
	if (dump && dump[0] == '1') {
		GL4METAL_LOG("---- %s shader %u ----\n%s\n---- end shader %u ----",
			shaderObject.type == GL_VERTEX_SHADER ? "vertex" : (shaderObject.type == GL_FRAGMENT_SHADER ? "fragment" : "compute"),
			shader, shaderObject.source.c_str(), shader);
	}
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
	if (!params) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		*params = 0;
		return;
	}
	const ShaderObject &shaderObject = it->second;
	switch (pname) {
		case GL_SHADER_TYPE:
			*params = (GLint)shaderObject.type;
			break;
		case GL_DELETE_STATUS:
			*params = shaderObject.deleted;
			break;
		case GL_COMPILE_STATUS:
			*params = shaderObject.compiled;
			break;
		case GL_INFO_LOG_LENGTH:
			*params = shaderObject.info_log.empty() ? 0 : (GLint)shaderObject.info_log.size() + 1;
			break;
		case GL_SHADER_SOURCE_LENGTH:
			*params = shaderObject.source.empty() ? 0 : (GLint)shaderObject.source.size() + 1;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			*params = 0;
			break;
	}
}

void copyGLString(const std::string &src, GLsizei bufSize, GLsizei *length, GLchar *dst) {
	GLsizei n = 0;
	if (bufSize > 0 && dst) {
		n = std::min<GLsizei>(bufSize - 1, (GLsizei)src.size());
		memcpy(dst, src.data(), (size_t)n);
		dst[n] = '\0';
	}
	if (length) {
		*length = n; // characters written, without the NUL terminator
	}
}

void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
	if (bufSize < 0 || (bufSize > 0 && !infoLog)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		return;
	}
	copyGLString(it->second.info_log, bufSize, length, infoLog);
}

void glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source) {
	if (bufSize < 0 || (bufSize > 0 && !source)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		return;
	}
	copyGLString(it->second.source, bufSize, length, source);
}
