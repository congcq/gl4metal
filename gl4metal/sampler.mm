#import "sampler.h"

static SamplerObject *findSampler(GLuint sampler) {
	auto it = ctx.sampler_objects.find(sampler);
	return it == ctx.sampler_objects.end() ? nullptr : &it->second;
}

void glGenSamplers(GLsizei count, GLuint *samplers) {
	for (GLsizei i = 0; i < count; i++) {
		GLuint name = ++ctx.next_sampler_id;
		ctx.sampler_objects[name] = SamplerObject();
		samplers[i] = name;
	}
}

void glDeleteSamplers(GLsizei count, const GLuint *samplers) {
	for (GLsizei i = 0; i < count; i++) {
		GLuint name = samplers[i];
		if (name == 0) {
			continue;
		}
		for (GLuint &bound : ctx.bound_samplers) {
			if (bound == name) {
				bound = 0;
			}
		}
		ctx.sampler_objects.erase(name);
	}
}

GLboolean glIsSampler(GLuint sampler) {
	return sampler != 0 && findSampler(sampler) != nullptr;
}

void glBindSampler(GLuint unit, GLuint sampler) {
	if (unit >= GL4METAL_MAX_TEXTURE_UNITS || (sampler != 0 && !findSampler(sampler))) {
		GL4METAL_LOG("glBindSampler: invalid unit %u or sampler %u (GL_INVALID_VALUE)", unit, sampler);
		return;
	}
	ctx.bound_samplers[unit] = sampler;
}

// MARK: - Sampler parameters

void glSamplerParameteri(GLuint sampler, GLenum pname, GLint param) {
	SamplerObject *so = findSampler(sampler);
	if (!so) {
		GL4METAL_LOG("glSamplerParameteri: invalid sampler %u", sampler);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_FILTER:   so->min_filter = (GLenum)param; break;
		case GL_TEXTURE_MAG_FILTER:   so->mag_filter = (GLenum)param; break;
		case GL_TEXTURE_WRAP_S:       so->wrap_s = (GLenum)param; break;
		case GL_TEXTURE_WRAP_T:       so->wrap_t = (GLenum)param; break;
		case GL_TEXTURE_WRAP_R:       so->wrap_r = (GLenum)param; break;
		case GL_TEXTURE_COMPARE_MODE: so->compare_mode = (GLenum)param; break;
		case GL_TEXTURE_COMPARE_FUNC: so->compare_func = (GLenum)param; break;
		case GL_TEXTURE_MIN_LOD:
		case GL_TEXTURE_MAX_LOD:
		case GL_TEXTURE_LOD_BIAS:
		case 0x84FE: // GL_TEXTURE_MAX_ANISOTROPY_EXT
			glSamplerParameterf(sampler, pname, (GLfloat)param);
			break;
		default:
			GL4METAL_LOG("glSamplerParameteri: unhandled pname 0x%X", pname);
			break;
	}
}

void glSamplerParameteriv(GLuint sampler, GLenum pname, const GLint *param) {
	glSamplerParameteri(sampler, pname, param[0]); // vector-valued pnames (border colour) are not supported
}

void glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param) {
	SamplerObject *so = findSampler(sampler);
	if (!so) {
		GL4METAL_LOG("glSamplerParameterf: invalid sampler %u", sampler);
		return;
	}
	switch (pname) {
		case GL_TEXTURE_MIN_LOD:  so->min_lod = param; break;
		case GL_TEXTURE_MAX_LOD:  so->max_lod = param; break;
		case GL_TEXTURE_LOD_BIAS: so->lod_bias = param; break;
		case 0x84FE:              so->max_anisotropy = std::max(param, 1.0f); break; // GL_TEXTURE_MAX_ANISOTROPY_EXT
		default:
			glSamplerParameteri(sampler, pname, (GLint)param); // enum-valued parameters passed as float
			break;
	}
}

void glSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat *param) {
	glSamplerParameterf(sampler, pname, param[0]);
}

void glSamplerParameterIiv(GLuint sampler, GLenum pname, const GLint *param) {
	GL4METAL_LOG("glSamplerParameterIiv: Unimplemented");
}

void glSamplerParameterIuiv(GLuint sampler, GLenum pname, const GLuint *param) {
	GL4METAL_LOG("glSamplerParameterIuiv: Unimplemented");
}

// MARK: - Sampler queries

void glGetSamplerParameteriv(GLuint sampler, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetSamplerParameteriv: Unimplemented");
}

void glGetSamplerParameterfv(GLuint sampler, GLenum pname, GLfloat *params) {
	GL4METAL_LOG("glGetSamplerParameterfv: Unimplemented");
}

void glGetSamplerParameterIiv(GLuint sampler, GLenum pname, GLint *params) {
	GL4METAL_LOG("glGetSamplerParameterIiv: Unimplemented");
}

void glGetSamplerParameterIuiv(GLuint sampler, GLenum pname, GLuint *params) {
	GL4METAL_LOG("glGetSamplerParameterIuiv: Unimplemented");
}
