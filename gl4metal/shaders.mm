#import "shaders.h"
#include <GL/glcorearb.h>

GLuint glCreateShader(GLenum type) {
	if (type != GL_VERTEX_SHADER && type != GL_FRAGMENT_SHADER) {
		GL4METAL_LOG("glCreateShader: unsupported shader type 0x%X (Metal has no geometry/tessellation stage in this layer)", type);
		return 0;
	}
	GLuint shader = ++ctx.next_shader_id; // not named `id`: that shadows the Objective-C type
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
		GL4METAL_LOG("glShaderSource: invalid shader %u", shader);
		return;
	}
	std::string source;
	for (GLsizei i = 0; i < count; i++) {
		if (!string[i]) {
			continue;
		}
		if (length && length[i] >= 0) {
			source.append(string[i], (size_t)length[i]);
		} else {
			source.append(string[i]);
		}
	}
	it->second.source = source;
}

void glCompileShader(GLuint shader) {
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		GL4METAL_LOG("glCompileShader: invalid shader %u", shader);
		return;
	}
	ShaderObject &s = it->second;
	s.info_log.clear();
	if (s.source.empty()) {
		s.compiled = GL_FALSE;
		s.info_log = "shader source is empty\n";
		return;
	}

	// GLSL is only stored here. The GLSL -> MSL translation needs both stages and their interface,
	// so it belongs to glLinkProgram.
	s.compiled = GL_TRUE;

	const char *dump = getenv("GL4METAL_DUMP_SHADERS");
	if (dump && dump[0] == '1') {
		GL4METAL_LOG("---- %s shader %u ----\n%s\n---- end shader %u ----",
			s.type == GL_VERTEX_SHADER ? "vertex" : "fragment", shader, s.source.c_str(), shader);
	}
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
	auto it = ctx.shader_objects.find(shader);
	if (it == ctx.shader_objects.end()) {
		GL4METAL_LOG("glGetShaderiv: invalid shader %u", shader);
		*params = 0;
		return;
	}
	const ShaderObject &s = it->second;
	switch (pname) {
		case GL_SHADER_TYPE:
			*params = (GLint)s.type;
			break;
		case GL_DELETE_STATUS:
			*params = s.deleted;
			break;
		case GL_COMPILE_STATUS:
			*params = s.compiled;
			break;
		case GL_INFO_LOG_LENGTH:
			*params = s.info_log.empty() ? 0 : (GLint)s.info_log.size() + 1; // includes the NUL terminator
			break;
		case GL_SHADER_SOURCE_LENGTH:
			*params = s.source.empty() ? 0 : (GLint)s.source.size() + 1;
			break;
		default:
			GL4METAL_LOG("glGetShaderiv: unhandled pname 0x%X", pname);
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
	auto it = ctx.shader_objects.find(shader);
	copyGLString(it == ctx.shader_objects.end() ? std::string() : it->second.info_log, bufSize, length, infoLog);
}

void glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source) {
	auto it = ctx.shader_objects.find(shader);
	copyGLString(it == ctx.shader_objects.end() ? std::string() : it->second.source, bufSize, length, source);
}
