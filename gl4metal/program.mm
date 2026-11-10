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

// Active-uniform type mapping for GL reflection. The runtime uniform-location table
// remains limited to default-block uniforms; named-block members are introspection-only.
static GLenum activeUniformGLType(const TranslatedUniform &u) {
	if (u.kind == TranslatedUniform::Sampler) {
		if (u.sampler_gl_type != 0) return (GLenum)u.sampler_gl_type;
		switch (u.dim) {
			case TranslatedUniform::DimCube: return GL_SAMPLER_CUBE;
			case TranslatedUniform::Dim3D: return GL_SAMPLER_3D;
			case TranslatedUniform::Dim2DArray: return GL_SAMPLER_2D_ARRAY;
			default: return GL_SAMPLER_2D;
		}
	}
	const int rows = std::max(1, std::min(u.vecsize, 4));
	if (u.columns > 1) {
		if (u.columns == rows) {
			static const GLenum squareTypes[] = { GL_FLOAT_MAT2, GL_FLOAT_MAT3, GL_FLOAT_MAT4 };
			return (u.columns >= 2 && u.columns <= 4) ? squareTypes[u.columns - 2] : GL_FLOAT_MAT4;
		}
		static const GLenum matrixTypes[4][4] = {
			{ 0, 0, 0, 0 },
			{ 0, 0, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4 },
			{ 0, GL_FLOAT_MAT3x2, 0, GL_FLOAT_MAT3x4 },
			{ 0, GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, 0 }
		};
		if (u.columns >= 2 && u.columns <= 4 && rows >= 2 && rows <= 4)
			return matrixTypes[u.columns][rows];
	}
	static const GLenum floatTypes[] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum intTypes[] = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	static const GLenum uintTypes[] = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	static const GLenum boolTypes[] = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
	const size_t componentIndex = (size_t)rows - 1;
	switch (u.kind) {
		case TranslatedUniform::Int: return intTypes[componentIndex];
		case TranslatedUniform::UInt: return uintTypes[componentIndex];
		case TranslatedUniform::Bool: return boolTypes[componentIndex];
		default: return floatTypes[componentIndex];
	}
}

static std::string activeUniformName(const TranslatedUniform &u) {
	return u.array_size > 1 ? u.name + "[0]" : u.name;
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
	p.active_uniforms.clear();
	p.locations.clear();
	p.uniform_data[0].clear();
	p.uniform_data[1].clear();
	p.frag_outputs.clear();
	p.blocks.clear();
	p.uses_gl_frag_coord = false;

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
		if (!tp.log.empty()) {
			GL4METAL_LOG("program %u translation notes:\n%s", program, tp.log.c_str());
		}
		GL4METAL_LOG("---- MSL vertex, program %u ----\n%s\n---- MSL fragment, program %u ----\n%s\n---- end MSL ----",
					 program, tp.msl[0].c_str(), program, tp.msl[1].c_str());
	}

	// MSL -> MTLFunction. Compile one vertex library and specialize its static Y-flip
	// function constant into two functions. The old path compiled the same vertex
	// shader twice as separate Metal libraries for default-FB and texture-FB draws.
	auto makeCompileOptions = []() -> MTLCompileOptions * {
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		options.languageVersion = MTLLanguageVersion2_2; // matches set_msl_version(2, 2) in the translator
		// Leave Metal's default fast-math mode enabled, as in the previous compilation path.
		return options;
	};

	id<MTLLibrary> vertexLibrary = nil;
	id<MTLFunction> vertexFunction = nil;
	id<MTLFunction> flippedFunction = nil;
	{
		MTLCompileOptions *options = makeCompileOptions();
		NSError *libraryError = nil;
		NSString *source = [NSString stringWithUTF8String:tp.msl[0].c_str()];
		vertexLibrary = [ctx.device newLibraryWithSource:source options:options error:&libraryError];
		if (vertexLibrary) {
			MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
			bool flipY = false;
			[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
			NSError *normalError = nil;
			vertexFunction = [vertexLibrary newFunctionWithName:@"main0" constantValues:constants error:&normalError];
			flipY = true;
			[constants setConstantValue:&flipY type:MTLDataTypeBool atIndex:0];
			NSError *flippedError = nil;
			flippedFunction = [vertexLibrary newFunctionWithName:@"main0" constantValues:constants error:&flippedError];
			if (!vertexFunction || !flippedFunction) {
				NSString *details = normalError ? normalError.localizedDescription :
					(flippedError ? flippedError.localizedDescription : @"Metal did not create both specialized vertex functions");
				p.info_log += std::string("vertex stage function specialization failed:\n") + details.UTF8String + "\n";
			}
		} else {
			p.info_log += std::string("vertex stage failed to compile as Metal:\n") +
				(libraryError ? libraryError.localizedDescription.UTF8String : "no Metal library") + "\n";
		}
	}

	id<MTLFunction> fragmentFunction = nil;
	{
		MTLCompileOptions *options = makeCompileOptions();
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:tp.msl[1].c_str()];
		id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:options error:&error];
		fragmentFunction = [library newFunctionWithName:@"main0"]; // SPIRV-Cross renames main() to main0()
		if (!fragmentFunction) {
			p.info_log += std::string("fragment stage failed to compile as Metal:\n") +
				(error ? error.localizedDescription.UTF8String : "no main0 entry point") + "\n";
		}
	}
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
		pa.is_unsigned = a.is_unsigned;
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

	// GL_ACTIVE_UNIFORMS includes both default-block uniforms and named-block members.
	// Only the former have locations and can be modified through glUniform*.
	for (const ProgramUniform &uniform : p.uniforms) {
		ActiveUniformInfo active;
		active.name = activeUniformName(uniform.info);
		active.type = activeUniformGLType(uniform.info);
		active.size = uniform.info.array_size;
		active.block_index = -1;
		const TranslatedUniform::Stage &stage = uniform.info.stage[uniform.info.stage[0].present ? 0 : 1];
		active.offset = (GLint)stage.offset;
		active.array_stride = (GLint)stage.array_stride;
		active.matrix_stride = (GLint)stage.matrix_stride;
		active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
		p.active_uniforms.push_back(std::move(active));
	}
	for (size_t blockIndex = 0; blockIndex < p.blocks.size(); ++blockIndex) {
		ProgramBlock &block = p.blocks[blockIndex];
		for (const TranslatedUniform &member : block.info.members) {
			ActiveUniformInfo active;
			active.name = activeUniformName(member);
			active.type = activeUniformGLType(member);
			active.size = member.array_size;
			active.block_index = (GLint)blockIndex;
			const TranslatedUniform::Stage &stage = member.stage[member.stage[0].present ? 0 : 1];
			active.offset = (GLint)stage.offset;
			active.array_stride = (GLint)stage.array_stride;
			active.matrix_stride = (GLint)stage.matrix_stride;
			active.row_major = stage.row_major ? GL_TRUE : GL_FALSE;
			block.active_uniform_indices.push_back((GLuint)p.active_uniforms.size());
			p.active_uniforms.push_back(std::move(active));
		}
	}

	p.vertex_function = vertexFunction;
	p.fragment_function = fragmentFunction;
	p.vertex_function_flipped = flippedFunction;
	p.uses_gl_frag_coord = tp.uses_gl_frag_coord;
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
			*params = (GLint)p->active_uniforms.size();
			break;
		case GL_ACTIVE_UNIFORM_MAX_LENGTH: {
			size_t longest = 0;
			for (const ActiveUniformInfo &uniform : p->active_uniforms) {
				longest = std::max(longest, uniform.name.size() + 1);
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
	static const GLenum uints[4]  = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	int c = std::min(std::max((int)a.vecsize, 1), 4) - 1;
	if (size) {
		*size = 1;
	}
	if (type) {
		*type = a.is_integer ? (a.is_unsigned ? uints[c] : ints[c]) : floats[c];
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
