#import "sampler.h"

// Sampler-object state uses one mutex. Public entry points acquire it once
// and use unlocked helpers below, avoiding nested locking in parameter queries.
std::mutex &gl4metalSamplerObjectMutex = *new std::mutex();

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
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
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
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
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
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	return samplerName != 0 && findSamplerUnlocked(samplerName) != nullptr;
}

void glBindSampler(GLuint unit, GLuint samplerName) {
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	if (unit >= (GLuint)GL4METAL_MAX_TEXTURE_UNITS ||
		(samplerName != 0 && !findSamplerUnlocked(samplerName))) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindSampler: invalid unit %u or sampler %u (GL_INVALID_VALUE)", unit, samplerName);
		return;
	}
	ctx.bound_samplers[unit] = samplerName;
}

// MARK: - Sampler parameters (unlocked helpers)

static bool samplerParameteriUnlocked(GLuint samplerName, GLenum pname, GLint param) {
	SamplerObject *sampler = findSamplerUnlocked(samplerName);
	if (!sampler) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glSamplerParameteri: invalid sampler %u", samplerName);
		return false;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:   sampler->min_filter = (GLenum)param; return true;
		case GL_TEXTURE_MAG_FILTER:   sampler->mag_filter = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_S:       sampler->wrap_s = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_T:       sampler->wrap_t = (GLenum)param; return true;
		case GL_TEXTURE_WRAP_R:       sampler->wrap_r = (GLenum)param; return true;
		case GL_TEXTURE_COMPARE_MODE: sampler->compare_mode = (GLenum)param; return true;
		case GL_TEXTURE_COMPARE_FUNC: sampler->compare_func = (GLenum)param; return true;
		case GL_TEXTURE_MIN_LOD:
		case GL_TEXTURE_MAX_LOD:
		case GL_TEXTURE_LOD_BIAS:
		case 0x84FE: // GL_TEXTURE_MAX_ANISOTROPY_EXT
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
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:  sampler->min_lod = param; return true;
		case GL_TEXTURE_MAX_LOD:  sampler->max_lod = param; return true;
		case GL_TEXTURE_LOD_BIAS: sampler->lod_bias = param; return true;
		case 0x84FE:              sampler->max_anisotropy = std::max(param, 1.0f); return true;
		default:
			// Enum-valued parameters may legally arrive through the float overload.
			return samplerParameteriUnlocked(samplerName, pname, (GLint)param);
	}
}

void glSamplerParameteri(GLuint samplerName, GLenum pname, GLint param) {
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	samplerParameteriUnlocked(samplerName, pname, param);
}

void glSamplerParameteriv(GLuint samplerName, GLenum pname, const GLint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	// Vector-valued pnames (for example border colour) are not supported.
	samplerParameteriUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterf(GLuint samplerName, GLenum pname, GLfloat param) {
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	samplerParameterfUnlocked(samplerName, pname, param);
}

void glSamplerParameterfv(GLuint samplerName, GLenum pname, const GLfloat *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	samplerParameterfUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterIiv(GLuint samplerName, GLenum pname, const GLint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	samplerParameteriUnlocked(samplerName, pname, param[0]);
}

void glSamplerParameterIuiv(GLuint samplerName, GLenum pname, const GLuint *param) {
	if (!param) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
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
		case GL_TEXTURE_MIN_LOD:      *params = (GLint)sampler->min_lod; break;
		case GL_TEXTURE_MAX_LOD:      *params = (GLint)sampler->max_lod; break;
		case GL_TEXTURE_LOD_BIAS:     *params = (GLint)sampler->lod_bias; break;
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
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	getSamplerParameterivUnlocked(samplerName, pname, params);
}

void glGetSamplerParameterfv(GLuint samplerName, GLenum pname, GLfloat *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
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
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	getSamplerParameterivUnlocked(samplerName, pname, params);
}

void glGetSamplerParameterIuiv(GLuint samplerName, GLenum pname, GLuint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	std::lock_guard<std::mutex> lock(gl4metalSamplerObjectMutex);
	GLint value = 0;
	if (getSamplerParameterivUnlocked(samplerName, pname, &value)) *params = (GLuint)value;
	else *params = 0;
}
