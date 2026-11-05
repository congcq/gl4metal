#import "program.h"
#import "shaders.h"

// MARK: - Helpers

static ProgramObject *findProgram(GLuint program) {
	auto it = ctx.program_objects.find(program);
	return it == ctx.program_objects.end() ? nullptr : &it->second;
}

static void destroyProgram(GLuint program) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end()) {
		return;
	}
	std::vector<GLuint> shaders = it->second.shaders;
	ctx.program_objects.erase(it);
	for (GLuint shader : shaders) {
		releaseShaderIfUnused(shader); // shaders deleted while attached go away with their last program
	}
}

static bool dumpEnabled() {
	const char *dump = getenv("GL4METAL_DUMP_SHADERS");
	return dump && dump[0] == '1';
}

// MARK: - Program objects

GLuint glCreateProgram(void) {
	GLuint program = ++ctx.next_shader_id; // GL uses one name space for shaders and programs
	ctx.program_objects[program] = ProgramObject();
	return program;
}

void glDeleteProgram(GLuint program) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		return;
	}
	if (ctx.current_program == program) {
		p->deleted = GL_TRUE; // destroyed when another program is made current
		return;
	}
	destroyProgram(program);
}

GLboolean glIsProgram(GLuint program) {
	return findProgram(program) != nullptr;
}

void glAttachShader(GLuint program, GLuint shader) {
	ProgramObject *p = findProgram(program);
	if (!p || ctx.shader_objects.find(shader) == ctx.shader_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glAttachShader: invalid program %u or shader %u", program, shader);
		return;
	}
	if (std::find(p->shaders.begin(), p->shaders.end(), shader) != p->shaders.end()) {
		GL4METAL_LOG("glAttachShader: shader %u is already attached to program %u", shader, program);
		return;
	}
	p->shaders.push_back(shader);
}

void glDetachShader(GLuint program, GLuint shader) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glDetachShader: invalid program %u", program);
		return;
	}
	auto it = std::find(p->shaders.begin(), p->shaders.end(), shader);
	if (it == p->shaders.end()) {
		GL4METAL_LOG("glDetachShader: shader %u is not attached to program %u", shader, program);
		return;
	}
	p->shaders.erase(it);
	releaseShaderIfUnused(shader);
}

void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders) {
	ProgramObject *p = findProgram(program);
	GLsizei n = 0;
	if (p) {
		for (GLuint shader : p->shaders) {
			if (n >= maxCount) {
				break;
			}
			shaders[n++] = shader;
		}
	}
	if (count) {
		*count = n;
	}
}

void glLinkProgram(GLuint program) {
	ProgramObject *pp = findProgram(program);
	if (!pp) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glLinkProgram: invalid program %u", program);
		return;
	}
	ProgramObject &p = *pp;

	// A failed link must not keep anything from an earlier successful one.
	p.linked = GL_FALSE;
	p.info_log.clear();
	p.vertex_function = nil;
	p.fragment_function = nil;
	p.vertex_function_flipped = nil;
	p.attribs.clear();
	p.uniforms.clear();
	p.locations.clear();
	p.uniform_data[0].clear();
	p.uniform_data[1].clear();
	p.frag_outputs.clear();
	p.blocks.clear();

	const ShaderObject *vs = nullptr;
	const ShaderObject *fs = nullptr;
	for (GLuint shader : p.shaders) {
		auto it = ctx.shader_objects.find(shader);
		if (it == ctx.shader_objects.end()) {
			continue;
		}
		if (it->second.type == GL_VERTEX_SHADER) {
			vs = &it->second;
		} else if (it->second.type == GL_FRAGMENT_SHADER) {
			fs = &it->second;
		}
	}
	if (!vs || !fs) {
		p.info_log = "a program needs one vertex and one fragment shader attached\n";
		return;
	}
	if (!vs->compiled || !fs->compiled) {
		p.info_log = "an attached shader is not compiled\n";
		return;
	}

	// GLSL -> SPIR-V -> MSL for both stages at once (see shadertranslator.cpp).
	TranslatedProgram tp;
	if (!translateProgram(vs->source, fs->source, tp)) {
		p.info_log = tp.log;
		GL4METAL_LOG("glLinkProgram(%u): translation failed:\n%s", program, tp.log.c_str());
		return;
	}
	if (dumpEnabled()) {
		GL4METAL_LOG("---- MSL vertex, program %u ----\n%s\n---- MSL fragment, program %u ----\n%s\n---- end MSL ----",
					 program, tp.msl[0].c_str(), program, tp.msl[1].c_str());
	}

	// MSL -> MTLFunction. The Metal compiler runs here, so its errors become the link log like on a GL driver.
	auto buildFunction = [&](const std::string &msl, const char *stageName) -> id<MTLFunction> {
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		options.languageVersion = MTLLanguageVersion2_2; // matches set_msl_version(2, 2) in the translator
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:msl.c_str()];
		id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:options error:&error];
		id<MTLFunction> function = [library newFunctionWithName:@"main0"]; // SPIRV-Cross renames main() to main0()
		if (!function) {
			p.info_log += std::string(stageName) + " stage failed to compile as Metal:\n" +
						  (error ? error.localizedDescription.UTF8String : "no main0 entry point") + "\n";
		}
		return function;
	};
	id<MTLFunction> vertexFunction = buildFunction(tp.msl[0], "vertex");
	id<MTLFunction> fragmentFunction = buildFunction(tp.msl[1], "fragment");
	id<MTLFunction> flippedFunction = buildFunction(tp.msl_vs_flipped, "vertex (Y flipped)"); // for draws into textures
	if (!vertexFunction || !fragmentFunction || !flippedFunction) {
		GL4METAL_LOG("glLinkProgram(%u): %s", program, p.info_log.c_str());
		return;
	}

	// Vertex attributes: honour glBindAttribLocation first, then give the rest the lowest free index.
	bool used[16] = {};
	for (const TranslatedAttrib &a : tp.attribs) {
		auto bound = p.attrib_bindings.find(a.name);
		if (bound != p.attrib_bindings.end() && bound->second < 16) {
			used[bound->second] = true;
		}
	}
	for (const TranslatedAttrib &a : tp.attribs) {
		ProgramAttrib pa;
		pa.name = a.name;
		pa.msl_location = a.location;
		pa.vecsize = a.vecsize;
		pa.is_integer = a.is_integer;
		auto bound = p.attrib_bindings.find(a.name);
		if (bound != p.attrib_bindings.end() && bound->second < 16) {
			pa.gl_index = (GLint)bound->second;
		} else {
			int slot = 0;
			while (slot < 16 && used[slot]) {
				slot++;
			}
			if (slot == 16) {
				p.info_log = "too many active vertex attributes\n";
				return;
			}
			used[slot] = true;
			pa.gl_index = slot;
		}
		p.attribs.push_back(pa);
	}

	// Uniform locations: an array uniform of N elements takes N consecutive locations.
	GLint nextLocation = 0;
	for (const TranslatedUniform &u : tp.uniforms) {
		ProgramUniform pu;
		pu.info = u;
		pu.base_location = nextLocation;
		pu.units.assign((size_t)u.array_size, 0);
		for (int element = 0; element < u.array_size; element++) {
			p.locations.emplace_back((GLint)p.uniforms.size(), element);
		}
		nextLocation += u.array_size;
		p.uniforms.push_back(pu);
	}
	p.uniform_data[0].assign(tp.block_size[0], 0); // GL uniforms start at 0
	p.uniform_data[1].assign(tp.block_size[1], 0);
	for (const auto &output : tp.frag_outputs) {
		p.frag_outputs[output.first] = (GLint)output.second;
	}

	for (const TranslatedBlock &tb : tp.blocks) {
		ProgramBlock pb;
		pb.info = tb;
		pb.binding_point = 0; // GL's default until glUniformBlockBinding
		p.blocks.push_back(pb);
	}

	p.vertex_function = vertexFunction;
	p.fragment_function = fragmentFunction;
	p.vertex_function_flipped = flippedFunction;
	p.linked = GL_TRUE;

	if (dumpEnabled()) {
		for (const ProgramAttrib &a : p.attribs) {
			GL4METAL_LOG("program %u attrib '%s': GL index %d, [[attribute(%u)]]", program, a.name.c_str(), a.gl_index, a.msl_location);
		}
		for (const ProgramUniform &u : p.uniforms) {
			GL4METAL_LOG("program %u uniform '%s': location %d x%d, vs offset %d, fs offset %d", program, u.info.name.c_str(),
						 u.base_location, u.info.array_size, u.info.stage[0].present ? (int)u.info.stage[0].offset : -1,
						 u.info.stage[1].present ? (int)u.info.stage[1].offset : -1);
		}
	}
}

void glUseProgram(GLuint program) {
	if (program != 0) {
		ProgramObject *p = findProgram(program);
		if (!p || !p->linked) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("glUseProgram: program %u does not exist or is not linked (GL_INVALID_OPERATION)", program);
			return;
		}
	}
	GLuint previous = ctx.current_program;
	ctx.current_program = program;
	if (previous != program) {
		ProgramObject *old = findProgram(previous);
		if (old && old->deleted) {
			destroyProgram(previous); // glDeleteProgram was waiting for it to stop being current
		}
	}
}

void glValidateProgram(GLuint program) {
	ProgramObject *p = findProgram(program);
	if (p) {
		p->validated = p->linked;
	}
}

void glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
	ProgramObject *p = findProgram(program);
	if (!p) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetProgramiv: invalid program %u", program);
		*params = 0;
		return;
	}
	switch (pname) {
		case GL_DELETE_STATUS:
			*params = p->deleted;
			break;
		case GL_LINK_STATUS:
			*params = p->linked;
			break;
		case GL_VALIDATE_STATUS:
			*params = p->validated;
			break;
		case GL_INFO_LOG_LENGTH:
			*params = p->info_log.empty() ? 0 : (GLint)p->info_log.size() + 1; // includes the NUL terminator
			break;
		case GL_ATTACHED_SHADERS:
			*params = (GLint)p->shaders.size();
			break;
		case GL_ACTIVE_ATTRIBUTES:
			*params = (GLint)p->attribs.size();
			break;
		case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: {
			size_t longest = 0;
			for (const ProgramAttrib &a : p->attribs) {
				longest = std::max(longest, a.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		case GL_ACTIVE_UNIFORMS:
			*params = (GLint)p->uniforms.size();
			break;
		case GL_ACTIVE_UNIFORM_MAX_LENGTH: {
			size_t longest = 0;
			for (const ProgramUniform &u : p->uniforms) {
				longest = std::max(longest, u.info.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		case GL_ACTIVE_UNIFORM_BLOCKS:
			*params = (GLint)p->blocks.size();
			break;
		case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: {
			size_t longest = 0;
			for (const ProgramBlock &b : p->blocks) {
				longest = std::max(longest, b.info.name.size() + 1);
			}
			*params = (GLint)longest;
			break;
		}
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetProgramiv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
	ProgramObject *p = findProgram(program);
	copyGLString(p ? p->info_log : std::string(), bufSize, length, infoLog);
}

// MARK: - Vertex attribute locations

void glBindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || index >= 16 || !name || strncmp(name, "gl_", 3) == 0) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glBindAttribLocation: invalid program %u, index %u or name", program, index);
		return;
	}
	p->attrib_bindings[name] = index; // takes effect at the next glLinkProgram
}

GLint glGetAttribLocation(GLuint program, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || !p->linked || !name) {
		return -1;
	}
	for (const ProgramAttrib &a : p->attribs) {
		if (a.name == name) {
			return a.gl_index;
		}
	}
	return -1;
}

void glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || index >= it->second.attribs.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveAttrib: invalid program %u or attribute index %u (GL_INVALID_VALUE)", program, index);
		if (length) {
			*length = 0;
		}
		return;
	}
	const ProgramAttrib &a = it->second.attribs[index];
	static const GLenum floats[4] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum ints[4]   = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	int c = std::min(std::max((int)a.vecsize, 1), 4) - 1;
	if (size) {
		*size = 1;
	}
	if (type) {
		*type = a.is_integer ? ints[c] : floats[c];
	}
	copyGLString(a.name, bufSize, length, name);
}

// MARK: - Fragment output locations

void glBindFragDataLocation(GLuint program, GLuint color, const GLchar *name) {
	// one colour output: the translator numbers it 0
}

void glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber, GLuint index, const GLchar *name) {
	// dual-source blending is not supported
}

GLint glGetFragDataLocation(GLuint program, const GLchar *name) {
	ProgramObject *p = findProgram(program);
	if (!p || !p->linked || !name) {
		return -1;
	}
	auto it = p->frag_outputs.find(name);
	return it == p->frag_outputs.end() ? -1 : it->second;
}

GLint glGetFragDataIndex(GLuint program, const GLchar *name) {
	return glGetFragDataLocation(program, name) >= 0 ? 0 : -1;
}
