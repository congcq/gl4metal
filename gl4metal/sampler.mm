#import "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

// Sampler objects belong to the thread-local MTLContext. A GL context is current
// on its owning thread, so a process-wide mutex would serialize unrelated devices.
static SamplerObject *findSamplerUnlocked(GLuint samplerName) {
	auto it = ctx.sampler_objects.find(samplerName);
	return it == ctx.sampler_objects.end() ? nullptr : &it->second;
}

static bool samplerParameteriUnlocked(GLuint samplerName, GLenum pname, GLint param);
static bool samplerParameterfUnlocked(GLuint samplerName, GLenum pname, GLfloat param);
static bool getSamplerParameterivUnlocked(GLuint samplerName, GLenum pname, GLint *params);

void glGenSamplers(GLsizei count, GLuint *samplers) {
	if (count < 0 || (count > 0 && !samplers)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei i = 0; i < count; i++) {
		GLuint samplerName = ++ctx.next_sampler_id;
		ctx.sampler_objects[samplerName] = SamplerObject();
		samplers[i] = samplerName;
	}
}

void glDeleteSamplers(GLsizei count, const GLuint *samplers) {
	if (count < 0 || (count > 0 && !samplers)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei i = 0; i < count; i++) {
		GLuint samplerName = samplers[i];
		if (samplerName == 0) continue;
		for (GLuint &bound : ctx.bound_samplers) {
			if (bound == samplerName) bound = 0;
		}
		ctx.sampler_objects.erase(samplerName);
	}
}

GLboolean glIsSampler(GLuint samplerName) {
	return samplerName != 0 && findSamplerUnlocked(samplerName) != nullptr;
}

void glBindSampler(GLuint unit, GLuint samplerName) {
	if (unit >= (GLuint)GL4METAL_MAX_TEXTURE_UNITS ||
		(samplerName != 0 && !findSamplerUnlocked(samplerName))) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindSampler: invalid unit %u or sampler %u (GL_INVALID_VALUE)", unit, samplerName);
		return;
	}
	ctx.bound_samplers[unit] = samplerName;
}

// MARK: - Sampler parameters (unlocked helpers)

static bool validSamplerMinFilter(GLint value) {
	return value == GL_NEAREST || value == GL_LINEAR || value == GL_NEAREST_MIPMAP_NEAREST ||
		value == GL_LINEAR_MIPMAP_NEAREST || value == GL_NEAREST_MIPMAP_LINEAR || value == GL_LINEAR_MIPMAP_LINEAR;
}

static bool validSamplerMagFilter(GLint value) {
	return value == GL_NEAREST || value == GL_LINEAR;
}

static bool validSamplerWrap(GLint value) {
	return value == GL_REPEAT || value == GL_CLAMP_TO_EDGE || value == GL_MIRRORED_REPEAT || value == GL_CLAMP_TO_BORDER;
}

static bool validSamplerCompareFunc(GLint value) {
	switch (value) {
		case GL_NEVER: case GL_LESS: case GL_EQUAL: case GL_LEQUAL:
		case GL_GREATER: case GL_NOTEQUAL: case GL_GEQUAL: case GL_ALWAYS: return true;
		default: return false;
	}
}

static GLint samplerFloatToInt(GLfloat value) {
	if (std::isnan(value)) return 0;
	if (value >= (GLfloat)std::numeric_limits<GLint>::max()) return std::numeric_limits<GLint>::max();
	if (value <= (GLfloat)std::numeric_limits<GLint>::min()) return std::numeric_limits<GLint>::min();
	return (GLint)value;
}

static bool samplerParameteriUnlocked(GLuint samplerName, GLenum pname, GLint param) {
	SamplerObject *sampler = findSamplerUnlocked(samplerName);
	if (!sampler) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glSamplerParameteri: invalid sampler %u", samplerName);
		return false;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:
			if (!validSamplerMinFilter(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->min_filter = (GLenum)param; return true;
		case GL_TEXTURE_MAG_FILTER:
			if (!validSamplerMagFilter(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->mag_filter = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_S:
			if (!validSamplerWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->wrap_s = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_T:
			if (!validSamplerWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->wrap_t = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_R:
			if (!validSamplerWrap(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->wrap_r = (GLenum)param; return true;
		case GL_TEXTURE_COMPARE_MODE:
			if (param != GL_NONE && param != GL_COMPARE_REF_TO_TEXTURE) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->compare_mode = (GLenum)param; return true;
		case GL_TEXTURE_COMPARE_FUNC:
			if (!validSamplerCompareFunc(param)) { gl4metalSetError(GL_INVALID_ENUM); return false; }
			sampler->compare_func = (GLenum)param; return true;
		case GL_TEXTURE_MIN_LOD:
		case GL_TEXTURE_MAX_LOD:
		case GL_TEXTURE_LOD_BIAS:
		case GL_TEXTURE_MAX_ANISOTROPY_EXT:
			return samplerParameterfUnlocked(samplerName, pname, (GLfloat)param);
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glSamplerParameteri: unhandled pname 0x%X", pname);
			return false;
	}
}

static bool samplerParameterfUnlocked(GLuint samplerName, GLenum pname, GLfloat param) {
	SamplerObject *sampler = findSamplerUnlocked(samplerName);
	if (!sampler) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glSamplerParameterf: invalid sampler %u", samplerName);
		return false;
	}
	if (!std::isfinite(param)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return false;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:  sampler->min_lod = param; return true;
		case GL_TEXTURE_MAX_LOD:  sampler->max_lod = param; return true;
		case GL_TEXTURE_LOD_BIAS: sampler->lod_bias = param; return true;
		case 0x84FE:              sampler->max_anisotropy = std::min(std::max(param, 1.0f), 16.0f); return true;
		default:
			if (param > (GLfloat)std::numeric_limits<GLint>::max() || param < (GLfloat)std::numeric_limits<GLint>::min()) {
				gl4metalSetError(GL_INVALID_VALUE);
				return false;
			}
			// Enum-valued parameters may legally arrive through the float overload.
			return samplerParameteriUnlocked(samplerName, pname, (GLint)param);
	}
}

void glSamplerParameteri(GLuint samplerName, GLenum pname, GLint param) {
	samplerParameteriUnlocked(samplerName, pname, param);
}

void glSamplerParameteriv(GLuint samplerName, GLenum pname, const GLint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!findSamplerUnlocked(samplerName)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = param[0] == 0 && param[1] == 0 && param[2] == 0 && param[3] == 0;
		const bool opaqueBlack = param[0] == 0 && param[1] == 0 && param[2] == 0 && param[3] == 1;
		const bool opaqueWhite = param[0] == 1 && param[1] == 1 && param[2] == 1 && param[3] == 1;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		SamplerObject *sampler = findSamplerUnlocked(samplerName);
		for (int component = 0; component < 4; ++component) sampler->border_color[component] = (GLfloat)param[component];
		return;
	}
	samplerParameteriUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterf(GLuint samplerName, GLenum pname, GLfloat param) {
	samplerParameterfUnlocked(samplerName, pname, param);
}

void glSamplerParameterfv(GLuint samplerName, GLenum pname, const GLfloat *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!findSamplerUnlocked(samplerName)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = param[0] == 0.0f && param[1] == 0.0f && param[2] == 0.0f && param[3] == 0.0f;
		const bool opaqueBlack = param[0] == 0.0f && param[1] == 0.0f && param[2] == 0.0f && param[3] == 1.0f;
		const bool opaqueWhite = param[0] == 1.0f && param[1] == 1.0f && param[2] == 1.0f && param[3] == 1.0f;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		SamplerObject *sampler = findSamplerUnlocked(samplerName);
		for (int component = 0; component < 4; ++component) sampler->border_color[component] = param[component];
		return;
	}
	samplerParameterfUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterIiv(GLuint samplerName, GLenum pname, const GLint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!findSamplerUnlocked(samplerName)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = param[0] == 0 && param[1] == 0 && param[2] == 0 && param[3] == 0;
		const bool opaqueBlack = param[0] == 0 && param[1] == 0 && param[2] == 0 && param[3] == 1;
		const bool opaqueWhite = param[0] == 1 && param[1] == 1 && param[2] == 1 && param[3] == 1;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		SamplerObject *sampler = findSamplerUnlocked(samplerName);
		for (int component = 0; component < 4; ++component) sampler->border_color[component] = (GLfloat)param[component];
		return;
	}
	samplerParameteriUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterIuiv(GLuint samplerName, GLenum pname, const GLuint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (!findSamplerUnlocked(samplerName)) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		const bool transparentBlack = param[0] == 0u && param[1] == 0u && param[2] == 0u && param[3] == 0u;
		const bool opaqueBlack = param[0] == 0u && param[1] == 0u && param[2] == 0u && param[3] == 1u;
		const bool opaqueWhite = param[0] == 1u && param[1] == 1u && param[2] == 1u && param[3] == 1u;
		if (!transparentBlack && !opaqueBlack && !opaqueWhite) { gl4metalSetError(GL_INVALID_OPERATION); return; }
		SamplerObject *sampler = findSamplerUnlocked(samplerName);
		for (int component = 0; component < 4; ++component) sampler->border_color[component] = (GLfloat)param[component];
		return;
	}
	if (param[0] > (GLuint)std::numeric_limits<GLint>::max()) { gl4metalSetError(GL_INVALID_VALUE); return; }
	samplerParameteriUnlocked(samplerName, pname, (GLint)param[0]);
}

// MARK: - Sampler queries

static bool getSamplerParameterivUnlocked(GLuint samplerName, GLenum pname, GLint *params) {
	SamplerObject *sampler = findSamplerUnlocked(samplerName);
	if (!sampler) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetSamplerParameteriv: invalid sampler %u (GL_INVALID_VALUE)", samplerName);
		*params = 0;
		return false;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:   *params = (GLint)sampler->min_filter; break;
		case GL_TEXTURE_MAG_FILTER:   *params = (GLint)sampler->mag_filter; break;
		case GL_TEXTURE_WRAP_S:       *params = (GLint)sampler->wrap_s; break;
		case GL_TEXTURE_WRAP_T:       *params = (GLint)sampler->wrap_t; break;
		case GL_TEXTURE_WRAP_R:       *params = (GLint)sampler->wrap_r; break;
		case GL_TEXTURE_COMPARE_MODE: *params = (GLint)sampler->compare_mode; break;
		case GL_TEXTURE_COMPARE_FUNC: *params = (GLint)sampler->compare_func; break;
		case GL_TEXTURE_MIN_LOD:      *params = samplerFloatToInt(sampler->min_lod); break;
		case GL_TEXTURE_MAX_LOD:      *params = samplerFloatToInt(sampler->max_lod); break;
		case GL_TEXTURE_LOD_BIAS:     *params = samplerFloatToInt(sampler->lod_bias); break;
		case GL_TEXTURE_MAX_ANISOTROPY_EXT: *params = samplerFloatToInt(sampler->max_anisotropy); break;
		case GL_TEXTURE_BORDER_COLOR:
			for (int component = 0; component < 4; ++component) params[component] = (GLint)sampler->border_color[component];
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetSamplerParameteriv: unhandled pname 0x%X", pname);
			*params = 0;
			return false;
	}
	return true;
}

void glGetSamplerParameteriv(GLuint samplerName, GLenum pname, GLint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	getSamplerParameterivUnlocked(samplerName, pname, params);
}

void glGetSamplerParameterfv(GLuint samplerName, GLenum pname, GLfloat *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	SamplerObject *sampler = findSamplerUnlocked(samplerName);
	if (!sampler) {
		gl4metalSetError(GL_INVALID_VALUE);
		*params = 0.0f;
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:  *params = sampler->min_lod; return;
		case GL_TEXTURE_MAX_LOD:  *params = sampler->max_lod; return;
		case GL_TEXTURE_LOD_BIAS: *params = sampler->lod_bias; return;
		case GL_TEXTURE_MAX_ANISOTROPY_EXT: *params = sampler->max_anisotropy; return;
		case GL_TEXTURE_BORDER_COLOR:
			for (int component = 0; component < 4; ++component) params[component] = sampler->border_color[component];
			return;
		default: {
			GLint value = 0;
			if (getSamplerParameterivUnlocked(samplerName, pname, &value)) *params = (GLfloat)value;
			else *params = 0.0f;
			return;
		}
	}
}

void glGetSamplerParameterIiv(GLuint samplerName, GLenum pname, GLint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	getSamplerParameterivUnlocked(samplerName, pname, params);
}

void glGetSamplerParameterIuiv(GLuint samplerName, GLenum pname, GLuint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	if (pname == GL_TEXTURE_BORDER_COLOR) {
		GLint values[4] = {};
		if (getSamplerParameterivUnlocked(samplerName, pname, values)) {
			for (int component = 0; component < 4; ++component) params[component] = (GLuint)values[component];
		} else {
			params[0] = params[1] = params[2] = params[3] = 0;
		}
		return;
	}
	GLint value = 0;
	if (getSamplerParameterivUnlocked(samplerName, pname, &value)) *params = (GLuint)value;
	else *params = 0;
}
