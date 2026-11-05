#import "uniform.h"

// MARK: - Helpers
// glUniform* writes into the CPU copy of the current program's default uniform block (std140 layout taken from the
// SPIR-V reflection). The copy is uploaded when a draw is issued: setVertexBytes/setFragmentBytes at
// GL4METAL_UNIFORM_BUFFER_INDEX.

enum SrcKind { SrcFloat, SrcInt, SrcUInt };

static ProgramObject *currentProgram() {
	if (ctx.current_program == 0) {
		return nullptr;
	}
	auto it = ctx.program_objects.find(ctx.current_program);
	return it == ctx.program_objects.end() ? nullptr : &it->second;
}

static ProgramUniform *uniformAt(ProgramObject &p, GLint location, GLint &element) {
	if (location < 0 || (size_t)location >= p.locations.size()) {
		return nullptr;
	}
	element = p.locations[(size_t)location].second;
	return &p.uniforms[(size_t)p.locations[(size_t)location].first];
}

static double srcValue(SrcKind kind, const void *data, size_t index) {
	switch (kind) {
		case SrcFloat: return ((const GLfloat *)data)[index];
		case SrcInt:   return ((const GLint *)data)[index];
		default:       return ((const GLuint *)data)[index];
	}
}

static void storeComponent(uint8_t *dst, TranslatedUniform::Kind kind, double value) {
	switch (kind) {
		case TranslatedUniform::Float: {
			float f = (float)value;
			memcpy(dst, &f, 4);
			break;
		}
		case TranslatedUniform::Int: {
			int32_t n = (int32_t)value;
			memcpy(dst, &n, 4);
			break;
		}
		case TranslatedUniform::Bool: {
			uint32_t b = value != 0.0 ? 1u : 0u;
			memcpy(dst, &b, 4);
			break;
		}
		default: {
			uint32_t u = (uint32_t)value;
			memcpy(dst, &u, 4);
			break;
		}
	}
}

// glUniform{1,2,3,4}{f,i,ui}[v]: `count` consecutive array elements starting at `location`.
static void writeVector(GLint location, GLsizei count, int comps, SrcKind kind, const void *data, const char *caller) {
	if (location == -1) {
		return; // GL silently ignores location -1 (inactive uniform)
	}
	ProgramObject *p = currentProgram();
	GLint element = 0;
	ProgramUniform *u = p ? uniformAt(*p, location, element) : nullptr;
	if (!u) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: no current program or invalid location %d (GL_INVALID_OPERATION)", caller, location);
		return;
	}

	if (u->info.kind == TranslatedUniform::Sampler) { // glUniform1i(sampler, textureUnit)
		for (GLsizei i = 0; i < count && element + i < u->info.array_size; i++) {
			u->units[(size_t)(element + i)] = (GLint)srcValue(kind, data, (size_t)i);
		}
		return;
	}
	if (comps != u->info.vecsize || u->info.columns != 1) {
		GL4METAL_LOG("%s: uniform '%s' is %dx%d, call writes %d components (GL_INVALID_OPERATION)",
					 caller, u->info.name.c_str(), u->info.vecsize, u->info.columns, comps);
		return;
	}

	for (GLsizei i = 0; i < count && element + i < u->info.array_size; i++) {
		for (int s = 0; s < 2; s++) { // the same uniform may live in both stages
			const TranslatedUniform::Stage &st = u->info.stage[s];
			if (!st.present) {
				continue;
			}
			size_t base = st.offset + (size_t)(element + i) * st.array_stride;
			if (base + (size_t)comps * 4 > p->uniform_data[s].size()) {
				continue;
			}
			for (int c = 0; c < comps; c++) {
				storeComponent(p->uniform_data[s].data() + base + (size_t)c * 4, u->info.kind,
							   srcValue(kind, data, (size_t)i * comps + c));
			}
		}
	}
}

// glUniformMatrix{C}x{R}fv: GL data is column-major unless `transpose`.
static void writeMatrix(GLint location, GLsizei count, int cols, int rows, GLboolean transpose, const GLfloat *value, const char *caller) {
	if (location == -1) {
		return;
	}
	ProgramObject *p = currentProgram();
	GLint element = 0;
	ProgramUniform *u = p ? uniformAt(*p, location, element) : nullptr;
	if (!u) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("%s: no current program or invalid location %d (GL_INVALID_OPERATION)", caller, location);
		return;
	}
	if (u->info.columns != cols || u->info.vecsize != rows) {
		GL4METAL_LOG("%s: uniform '%s' is %dx%d, call writes %dx%d (GL_INVALID_OPERATION)",
					 caller, u->info.name.c_str(), u->info.columns, u->info.vecsize, cols, rows);
		return;
	}

	for (GLsizei i = 0; i < count && element + i < u->info.array_size; i++) {
		const GLfloat *m = value + (size_t)i * cols * rows;
		for (int s = 0; s < 2; s++) {
			const TranslatedUniform::Stage &st = u->info.stage[s];
			if (!st.present) {
				continue;
			}
			size_t base = st.offset + (size_t)(element + i) * st.array_stride;
			if (base + (size_t)(cols - 1) * st.matrix_stride + (size_t)rows * 4 > p->uniform_data[s].size()) {
				continue;
			}
			for (int c = 0; c < cols; c++) {
				for (int r = 0; r < rows; r++) {
					float f = transpose ? m[r * cols + c] : m[c * rows + r];
					memcpy(p->uniform_data[s].data() + base + (size_t)c * st.matrix_stride + (size_t)r * 4, &f, 4);
				}
			}
		}
	}
}

// MARK: - Uniform reflection helpers

static GLenum glTypeOf(const TranslatedUniform &u) {
	if (u.kind == TranslatedUniform::Sampler) {
		switch (u.dim) {
			case TranslatedUniform::DimCube:    return GL_SAMPLER_CUBE;
			case TranslatedUniform::Dim3D:      return GL_SAMPLER_3D;
			case TranslatedUniform::Dim2DArray: return GL_SAMPLER_2D_ARRAY;
			default:                            return GL_SAMPLER_2D;
		}
	}
	int c = std::min(std::max(u.vecsize, 1), 4) - 1;
	if (u.columns > 1) {
		static const GLenum square[3] = { GL_FLOAT_MAT2, GL_FLOAT_MAT3, GL_FLOAT_MAT4 };
		if (u.columns == u.vecsize) {
			return square[u.columns - 2];
		}
		// glUniformMatrix{columns}x{rows}
		static const GLenum rect[3][3] = { { GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4, 0 }, { GL_FLOAT_MAT3x2, GL_FLOAT_MAT3x4, 0 }, { GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, 0 } };
		int col = u.columns - 2;
		int row = u.vecsize == 2 ? 0 : (u.vecsize == 3 ? (u.columns == 2 ? 0 : 1) : 1);
		if (u.columns == 2) row = u.vecsize == 3 ? 0 : 1;
		if (u.columns == 3) row = u.vecsize == 2 ? 0 : 1;
		if (u.columns == 4) row = u.vecsize == 2 ? 0 : 1;
		return rect[col][row];
	}
	static const GLenum floats[4] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum ints[4]   = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	static const GLenum uints[4]  = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	static const GLenum bools[4]  = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
	switch (u.kind) {
		case TranslatedUniform::Int:  return ints[c];
		case TranslatedUniform::UInt: return uints[c];
		case TranslatedUniform::Bool: return bools[c];
		default:                      return floats[c];
	}
}

static std::string activeName(const TranslatedUniform &u) {
	return u.array_size > 1 ? u.name + "[0]" : u.name; // arrays are reported as "name[0]"
}

// Reads `count` components of a uniform back from the CPU copy of its block.
static void readUniform(GLuint program, GLint location, int count, double *out) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || location < 0 || (size_t)location >= it->second.locations.size()) {
		gl4metalSetError(GL_INVALID_OPERATION);
		GL4METAL_LOG("glGetUniform*: invalid program %u or location %d (GL_INVALID_OPERATION)", program, location);
		return;
	}
	ProgramObject &p = it->second;
	const ProgramUniform &u = p.uniforms[(size_t)p.locations[(size_t)location].first];
	GLint element = p.locations[(size_t)location].second;
	if (u.info.kind == TranslatedUniform::Sampler) {
		out[0] = u.units[(size_t)element];
		return;
	}
	for (int s = 0; s < 2; s++) {
		const TranslatedUniform::Stage &st = u.info.stage[s];
		if (!st.present) {
			continue;
		}
		size_t base = st.offset + (size_t)element * st.array_stride;
		int rows = u.info.vecsize, cols = u.info.columns;
		for (int i = 0; i < count; i++) {
			size_t at = u.info.columns > 1 ? base + (size_t)(i / rows) * st.matrix_stride + (size_t)(i % rows) * 4 : base + (size_t)i * 4;
			if (at + 4 > p.uniform_data[s].size() || i >= rows * cols) {
				break;
			}
			if (u.info.kind == TranslatedUniform::Float) {
				float f; memcpy(&f, p.uniform_data[s].data() + at, 4); out[i] = f;
			} else if (u.info.kind == TranslatedUniform::Int) {
				int32_t n; memcpy(&n, p.uniform_data[s].data() + at, 4); out[i] = n;
			} else {
				uint32_t n; memcpy(&n, p.uniform_data[s].data() + at, 4); out[i] = n;
			}
		}
		return;
	}
}

static int componentsAt(GLuint program, GLint location) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || location < 0 || (size_t)location >= it->second.locations.size()) {
		return 0;
	}
	const TranslatedUniform &u = it->second.uniforms[(size_t)it->second.locations[(size_t)location].first].info;
	return u.kind == TranslatedUniform::Sampler ? 1 : u.vecsize * u.columns;
}

// MARK: - Uniform locations and queries

GLint glGetUniformLocation(GLuint program, const GLchar *name) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || !it->second.linked || !name) {
		return -1;
	}
	// "name" and "name[0]" address the first element, "name[n]" element n.
	std::string base(name);
	GLint element = 0;
	size_t bracket = base.find('[');
	if (bracket != std::string::npos) {
		element = atoi(base.c_str() + bracket + 1);
		base.resize(bracket);
	}
	for (const ProgramUniform &u : it->second.uniforms) {
		if (u.info.name == base) {
			return element >= 0 && element < u.info.array_size ? u.base_location + element : -1;
		}
	}
	return -1; // inactive (not referenced by the shaders)
}

void glGetUniformfv(GLuint program, GLint location, GLfloat *params) {
	double values[16] = { 0 };
	int n = componentsAt(program, location);
	readUniform(program, location, n, values);
	for (int i = 0; i < n; i++) {
		params[i] = (GLfloat)values[i];
	}
}

void glGetUniformiv(GLuint program, GLint location, GLint *params) {
	double values[16] = { 0 };
	int n = componentsAt(program, location);
	readUniform(program, location, n, values);
	for (int i = 0; i < n; i++) {
		params[i] = (GLint)values[i];
	}
}

void glGetUniformuiv(GLuint program, GLint location, GLuint *params) {
	double values[16] = { 0 };
	int n = componentsAt(program, location);
	readUniform(program, location, n, values);
	for (int i = 0; i < n; i++) {
		params[i] = (GLuint)values[i];
	}
}

void glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || index >= it->second.uniforms.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveUniform: invalid program %u or uniform index %u (GL_INVALID_VALUE)", program, index);
		if (length) {
			*length = 0;
		}
		return;
	}
	const TranslatedUniform &u = it->second.uniforms[index].info;
	if (size) {
		*size = u.array_size;
	}
	if (type) {
		*type = glTypeOf(u);
	}
	copyGLString(activeName(u), bufSize, length, name);
}

void glGetUniformIndices(GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices) {
	auto it = ctx.program_objects.find(program);
	for (GLsizei i = 0; i < uniformCount; i++) {
		uniformIndices[i] = GL_INVALID_INDEX;
		if (it == ctx.program_objects.end()) {
			continue;
		}
		std::string wanted(uniformNames[i]);
		if (wanted.size() > 3 && wanted.compare(wanted.size() - 3, 3, "[0]") == 0) {
			wanted.resize(wanted.size() - 3);
		}
		for (size_t k = 0; k < it->second.uniforms.size(); k++) {
			if (it->second.uniforms[k].info.name == wanted) {
				uniformIndices[i] = (GLuint)k;
				break;
			}
		}
	}
}

void glGetActiveUniformsiv(GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params) {
	auto it = ctx.program_objects.find(program);
	for (GLsizei i = 0; i < uniformCount; i++) {
		params[i] = 0;
		if (it == ctx.program_objects.end() || uniformIndices[i] >= it->second.uniforms.size()) {
			continue;
		}
		const TranslatedUniform &u = it->second.uniforms[uniformIndices[i]].info;
		const TranslatedUniform::Stage &st = u.stage[u.stage[0].present ? 0 : 1];
		switch (pname) {
			case GL_UNIFORM_TYPE:          params[i] = (GLint)glTypeOf(u); break;
			case GL_UNIFORM_SIZE:          params[i] = u.array_size; break;
			case GL_UNIFORM_NAME_LENGTH:   params[i] = (GLint)activeName(u).size() + 1; break;
			case GL_UNIFORM_BLOCK_INDEX:   params[i] = -1; break; // members of the default block
			case GL_UNIFORM_OFFSET:        params[i] = (GLint)st.offset; break;
			case GL_UNIFORM_ARRAY_STRIDE:  params[i] = (GLint)st.array_stride; break;
			case GL_UNIFORM_MATRIX_STRIDE: params[i] = (GLint)st.matrix_stride; break;
			case GL_UNIFORM_IS_ROW_MAJOR:  params[i] = 0; break;
			default:
				gl4metalSetError(GL_INVALID_ENUM);
				GL4METAL_LOG("glGetActiveUniformsiv: unhandled pname 0x%X", pname);
				break;
		}
	}
}

void glGetActiveUniformName(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || uniformIndex >= it->second.uniforms.size()) {
		if (length) {
			*length = 0;
		}
		return;
	}
	copyGLString(activeName(it->second.uniforms[uniformIndex].info), bufSize, length, uniformName);
}

// MARK: - Uniform blocks

GLuint glGetUniformBlockIndex(GLuint program, const GLchar *uniformBlockName) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || !it->second.linked || !uniformBlockName) {
		return GL_INVALID_INDEX;
	}
	for (size_t i = 0; i < it->second.blocks.size(); i++) {
		if (it->second.blocks[i].info.name == uniformBlockName) {
			return (GLuint)i;
		}
	}
	return GL_INVALID_INDEX;
}

void glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || uniformBlockIndex >= it->second.blocks.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveUniformBlockiv: invalid program %u or block %u", program, uniformBlockIndex);
		*params = 0;
		return;
	}
	const ProgramBlock &b = it->second.blocks[uniformBlockIndex];
	switch (pname) {
		case GL_UNIFORM_BLOCK_BINDING:                     *params = (GLint)b.binding_point; break;
		case GL_UNIFORM_BLOCK_DATA_SIZE:                   *params = (GLint)b.info.size; break;
		case GL_UNIFORM_BLOCK_NAME_LENGTH:                 *params = (GLint)b.info.name.size() + 1; break;
		case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS:             *params = 0; break; // members are not reflected yet
		case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:   *params = b.info.stage[0].present; break;
		case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *params = b.info.stage[1].present; break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			GL4METAL_LOG("glGetActiveUniformBlockiv: unhandled pname 0x%X", pname);
			*params = 0;
			break;
	}
}

void glGetActiveUniformBlockName(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || uniformBlockIndex >= it->second.blocks.size()) {
		if (length) {
			*length = 0;
		}
		return;
	}
	copyGLString(it->second.blocks[uniformBlockIndex].info.name, bufSize, length, uniformBlockName);
}

void glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || uniformBlockIndex >= it->second.blocks.size() ||
		uniformBlockBinding >= GL4METAL_MAX_UNIFORM_BUFFER_BINDINGS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glUniformBlockBinding: invalid program %u, block %u or binding %u", program, uniformBlockIndex, uniformBlockBinding);
		return;
	}
	it->second.blocks[uniformBlockIndex].binding_point = uniformBlockBinding;
}

// MARK: - Setting uniforms

void glUniform1f(GLint location, GLfloat v0) {
	GLfloat v[1] = { v0 };
	writeVector(location, 1, 1, SrcFloat, v, "glUniform1f");
}

void glUniform2f(GLint location, GLfloat v0, GLfloat v1) {
	GLfloat v[2] = { v0, v1 };
	writeVector(location, 1, 2, SrcFloat, v, "glUniform2f");
}

void glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
	GLfloat v[3] = { v0, v1, v2 };
	writeVector(location, 1, 3, SrcFloat, v, "glUniform3f");
}

void glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
	GLfloat v[4] = { v0, v1, v2, v3 };
	writeVector(location, 1, 4, SrcFloat, v, "glUniform4f");
}

void glUniform1i(GLint location, GLint v0) {
	GLint v[1] = { v0 };
	writeVector(location, 1, 1, SrcInt, v, "glUniform1i");
}

void glUniform2i(GLint location, GLint v0, GLint v1) {
	GLint v[2] = { v0, v1 };
	writeVector(location, 1, 2, SrcInt, v, "glUniform2i");
}

void glUniform3i(GLint location, GLint v0, GLint v1, GLint v2) {
	GLint v[3] = { v0, v1, v2 };
	writeVector(location, 1, 3, SrcInt, v, "glUniform3i");
}

void glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
	GLint v[4] = { v0, v1, v2, v3 };
	writeVector(location, 1, 4, SrcInt, v, "glUniform4i");
}

void glUniform1ui(GLint location, GLuint v0) {
	GLuint v[1] = { v0 };
	writeVector(location, 1, 1, SrcUInt, v, "glUniform1ui");
}

void glUniform2ui(GLint location, GLuint v0, GLuint v1) {
	GLuint v[2] = { v0, v1 };
	writeVector(location, 1, 2, SrcUInt, v, "glUniform2ui");
}

void glUniform3ui(GLint location, GLuint v0, GLuint v1, GLuint v2) {
	GLuint v[3] = { v0, v1, v2 };
	writeVector(location, 1, 3, SrcUInt, v, "glUniform3ui");
}

void glUniform4ui(GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3) {
	GLuint v[4] = { v0, v1, v2, v3 };
	writeVector(location, 1, 4, SrcUInt, v, "glUniform4ui");
}

void glUniform1fv(GLint location, GLsizei count, const GLfloat *value) {
	writeVector(location, count, 1, SrcFloat, value, "glUniform1fv");
}

void glUniform2fv(GLint location, GLsizei count, const GLfloat *value) {
	writeVector(location, count, 2, SrcFloat, value, "glUniform2fv");
}

void glUniform3fv(GLint location, GLsizei count, const GLfloat *value) {
	writeVector(location, count, 3, SrcFloat, value, "glUniform3fv");
}

void glUniform4fv(GLint location, GLsizei count, const GLfloat *value) {
	writeVector(location, count, 4, SrcFloat, value, "glUniform4fv");
}

void glUniform1iv(GLint location, GLsizei count, const GLint *value) {
	writeVector(location, count, 1, SrcInt, value, "glUniform1iv");
}

void glUniform2iv(GLint location, GLsizei count, const GLint *value) {
	writeVector(location, count, 2, SrcInt, value, "glUniform2iv");
}

void glUniform3iv(GLint location, GLsizei count, const GLint *value) {
	writeVector(location, count, 3, SrcInt, value, "glUniform3iv");
}

void glUniform4iv(GLint location, GLsizei count, const GLint *value) {
	writeVector(location, count, 4, SrcInt, value, "glUniform4iv");
}

void glUniform1uiv(GLint location, GLsizei count, const GLuint *value) {
	writeVector(location, count, 1, SrcUInt, value, "glUniform1uiv");
}

void glUniform2uiv(GLint location, GLsizei count, const GLuint *value) {
	writeVector(location, count, 2, SrcUInt, value, "glUniform2uiv");
}

void glUniform3uiv(GLint location, GLsizei count, const GLuint *value) {
	writeVector(location, count, 3, SrcUInt, value, "glUniform3uiv");
}

void glUniform4uiv(GLint location, GLsizei count, const GLuint *value) {
	writeVector(location, count, 4, SrcUInt, value, "glUniform4uiv");
}

void glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 2, 2, transpose, value, "glUniformMatrix2fv");
}

void glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 3, 3, transpose, value, "glUniformMatrix3fv");
}

void glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 4, 4, transpose, value, "glUniformMatrix4fv");
}

void glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 2, 3, transpose, value, "glUniformMatrix2x3fv");
}

void glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 3, 2, transpose, value, "glUniformMatrix3x2fv");
}

void glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 2, 4, transpose, value, "glUniformMatrix2x4fv");
}

void glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 4, 2, transpose, value, "glUniformMatrix4x2fv");
}

void glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 3, 4, transpose, value, "glUniformMatrix3x4fv");
}

void glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
	writeMatrix(location, count, 4, 3, transpose, value, "glUniformMatrix4x3fv");
}
