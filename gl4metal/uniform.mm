#import "uniform.h"

#include <atomic>

// MARK: - Helpers
// glUniform* writes into the CPU copy of the current program's default uniform block (std140 layout taken from
// SPIR-V reflection). Per-stage revisions let the draw/compute paths skip unchanged encoder uploads and create
// immutable or command-buffer-scoped Metal buffer snapshots for blocks larger than Metal's 4 KiB setBytes limit.

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

static inline void bumpUniformDataRevision(ProgramObject &program, int stage) {
	uint64_t &revision = program.uniform_data_revision[stage];
	++revision;
	if (revision == 0) ++revision; // reserve zero for an uninitialized GPU snapshot
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
	if (u->is_image) {
		if (comps != 1 || kind != SrcInt) {
			gl4metalSetError(GL_INVALID_OPERATION);
			GL4METAL_LOG("%s: image uniforms require glUniform1i/glUniform1iv", caller);
			return;
		}
		for (GLsizei i = 0; i < count && element + i < u->info.array_size; ++i) {
			const size_t imageUnitIndex = (size_t)u->image_index + (size_t)(element + i);
			const GLint imageUnit = (GLint)srcValue(kind, data, (size_t)i);
			if (imageUnit < 0 || imageUnit >= GL4METAL_MAX_IMAGE_UNITS || imageUnitIndex >= p->image_units.size()) {
				gl4metalSetError(GL_INVALID_VALUE);
				GL4METAL_LOG("%s: image uniform '%s' references invalid image unit %d (limit=%d)",
					caller, u->info.name.c_str(), imageUnit, GL4METAL_MAX_IMAGE_UNITS);
				return;
			}
			p->image_units[imageUnitIndex] = imageUnit;
		}
		return;
	}
	if (comps != u->info.vecsize || u->info.columns != 1) {
		GL4METAL_LOG("%s: uniform '%s' is %dx%d, call writes %d components (GL_INVALID_OPERATION)",
					 caller, u->info.name.c_str(), u->info.vecsize, u->info.columns, comps);
		return;
	}

	bool stageDirty[3] = {false, false, false};
	for (GLsizei i = 0; i < count && element + i < u->info.array_size; i++) {
		for (int s = 0; s < 3; s++) { // the same uniform may live in vertex, fragment or compute
			const TranslatedUniform::Stage &st = u->info.stage[s];
			if (!st.present) continue;
			size_t base = st.offset + (size_t)(element + i) * st.array_stride;
			if (base + (size_t)comps * 4 > p->uniform_data[s].size()) continue;
			for (int c = 0; c < comps; c++) {
				uint8_t encoded[4];
				storeComponent(encoded, u->info.kind, srcValue(kind, data, (size_t)i * comps + c));
				uint8_t *destination = p->uniform_data[s].data() + base + (size_t)c * 4;
				if (memcmp(destination, encoded, sizeof(encoded)) != 0) {
					memcpy(destination, encoded, sizeof(encoded));
					stageDirty[s] = true;
				}
			}
		}
	}
	for (int s = 0; s < 3; ++s) {
		if (stageDirty[s]) bumpUniformDataRevision(*p, s);
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

	bool stageDirty[3] = {false, false, false};
	for (GLsizei i = 0; i < count && element + i < u->info.array_size; i++) {
		const GLfloat *m = value + (size_t)i * cols * rows;
		for (int s = 0; s < 3; s++) {
			const TranslatedUniform::Stage &st = u->info.stage[s];
			if (!st.present) continue;
			size_t base = st.offset + (size_t)(element + i) * st.array_stride;
			if (base + (size_t)(cols - 1) * st.matrix_stride + (size_t)rows * 4 > p->uniform_data[s].size()) continue;
			for (int c = 0; c < cols; c++) {
				for (int r = 0; r < rows; r++) {
					float f = transpose ? m[r * cols + c] : m[c * rows + r];
					uint8_t *destination = p->uniform_data[s].data() + base + (size_t)c * st.matrix_stride + (size_t)r * 4;
					if (memcmp(destination, &f, sizeof(f)) != 0) {
						memcpy(destination, &f, sizeof(f));
						stageDirty[s] = true;
					}
				}
			}
		}
	}
	for (int s = 0; s < 3; ++s) {
		if (stageDirty[s]) bumpUniformDataRevision(*p, s);
	}
}

// MARK: - Uniform reflection helpers

static GLenum glTypeOf(const TranslatedUniform &u) {
	if (u.kind == TranslatedUniform::Sampler) {
		if (u.sampler_gl_type != 0) return (GLenum)u.sampler_gl_type;
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
	if (u.is_image) {
		const size_t imageUnitIndex = (size_t)u.image_index + (size_t)element;
		if (imageUnitIndex < p.image_units.size()) out[0] = p.image_units[imageUnitIndex];
		return;
	}
	if (u.info.kind == TranslatedUniform::Sampler) {
		out[0] = u.units[(size_t)element];
		return;
	}
	for (int s = 0; s < 3; s++) {
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
	const ProgramUniform &programUniform = it->second.uniforms[(size_t)it->second.locations[(size_t)location].first];
	const TranslatedUniform &u = programUniform.info;
	return (programUniform.is_image || u.kind == TranslatedUniform::Sampler) ? 1 : u.vecsize * u.columns;
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
	const size_t bracket = base.find('[');
	if (bracket != std::string::npos) {
		element = atoi(base.c_str() + bracket + 1);
		base.resize(bracket);
	}
	for (const ProgramUniform &uniform : it->second.uniforms) {
		if (uniform.info.name == base) {
			const GLint result = element >= 0 && element < uniform.info.array_size ? uniform.base_location + element : -1;
			return result;
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
	if (it == ctx.program_objects.end() || index >= it->second.active_uniforms.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveUniform: invalid program %u or uniform index %u (GL_INVALID_VALUE)", program, index);
		if (length) *length = 0;
		return;
	}
	const ActiveUniformInfo &uniform = it->second.active_uniforms[index];
	if (size) *size = uniform.size;
	if (type) *type = uniform.type;
	copyGLString(uniform.name, bufSize, length, name);
}

void glGetUniformIndices(GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices) {
	if (uniformCount < 0 || (uniformCount > 0 && (!uniformNames || !uniformIndices))) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || !it->second.linked) {
		gl4metalSetError(GL_INVALID_VALUE);
		for (GLsizei i = 0; i < uniformCount; ++i) uniformIndices[i] = GL_INVALID_INDEX;
		return;
	}
	for (GLsizei i = 0; i < uniformCount; i++) {
		uniformIndices[i] = GL_INVALID_INDEX;
		if (!uniformNames[i]) continue;
		std::string wanted(uniformNames[i]);
		const size_t bracket = wanted.find('[');
		if (bracket != std::string::npos && wanted.back() == ']') wanted.resize(bracket);
		for (size_t k = 0; k < it->second.active_uniforms.size(); k++) {
			const ActiveUniformInfo &active = it->second.active_uniforms[k];
			std::string activeBase = active.name;
			const size_t activeBracket = activeBase.find('[');
			if (activeBracket != std::string::npos) activeBase.resize(activeBracket);
			if (activeBase == wanted) {
				uniformIndices[i] = (GLuint)k;
				break;
			}
		}
	}
}

void glGetActiveUniformsiv(GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params) {
	if (uniformCount < 0 || (uniformCount > 0 && (!uniformIndices || !params))) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || !it->second.linked) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (params) for (GLsizei i = 0; i < uniformCount; ++i) params[i] = 0;
		return;
	}
	for (GLsizei i = 0; i < uniformCount; i++) {
		if (uniformIndices[i] >= it->second.active_uniforms.size()) {
			gl4metalSetError(GL_INVALID_VALUE);
			params[i] = 0;
			continue;
		}
		const ActiveUniformInfo &uniform = it->second.active_uniforms[uniformIndices[i]];
		switch (pname) {
			case GL_UNIFORM_TYPE: params[i] = (GLint)uniform.type; break;
			case GL_UNIFORM_SIZE: params[i] = uniform.size; break;
			case GL_UNIFORM_NAME_LENGTH: params[i] = (GLint)uniform.name.size() + 1; break;
			case GL_UNIFORM_BLOCK_INDEX: params[i] = uniform.block_index; break;
			case GL_UNIFORM_OFFSET: params[i] = uniform.offset; break;
			case GL_UNIFORM_ARRAY_STRIDE: params[i] = uniform.array_stride; break;
			case GL_UNIFORM_MATRIX_STRIDE: params[i] = uniform.matrix_stride; break;
			case GL_UNIFORM_IS_ROW_MAJOR: params[i] = uniform.row_major ? GL_TRUE : GL_FALSE; break;
			default:
				gl4metalSetError(GL_INVALID_ENUM);
				GL4METAL_LOG("glGetActiveUniformsiv: unhandled pname 0x%X", pname);
				params[i] = 0;
				break;
		}
	}
}

void glGetActiveUniformName(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || uniformIndex >= it->second.active_uniforms.size()) {
		if (length) *length = 0;
		return;
	}
	copyGLString(it->second.active_uniforms[uniformIndex].name, bufSize, length, uniformName);
}

// MARK: - Uniform blocks

GLuint glGetUniformBlockIndex(GLuint program, const GLchar *uniformBlockName) {
	auto it = ctx.program_objects.find(program);
	if (it == ctx.program_objects.end() || !it->second.linked || !uniformBlockName) return GL_INVALID_INDEX;
	for (size_t i = 0; i < it->second.blocks.size(); i++) {
		if (it->second.blocks[i].info.name == uniformBlockName) return (GLuint)i;
	}
	return GL_INVALID_INDEX;
}

void glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params) {
	auto it = ctx.program_objects.find(program);
	if (!params) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (it == ctx.program_objects.end() || uniformBlockIndex >= it->second.blocks.size()) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glGetActiveUniformBlockiv: invalid program %u or block %u", program, uniformBlockIndex);
		*params = 0;
		return;
	}
	const ProgramBlock &block = it->second.blocks[uniformBlockIndex];
	switch (pname) {
		case GL_UNIFORM_BLOCK_BINDING: *params = (GLint)block.binding_point; break;
		case GL_UNIFORM_BLOCK_DATA_SIZE: *params = (GLint)block.info.size; break;
		case GL_UNIFORM_BLOCK_NAME_LENGTH: *params = (GLint)block.info.name.size() + 1; break;
		case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: *params = (GLint)block.active_uniform_indices.size(); break;
		case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
			for (size_t i = 0; i < block.active_uniform_indices.size(); ++i)
				params[i] = (GLint)block.active_uniform_indices[i];
			break;
		case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *params = block.info.stage[0].present; break;
		case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *params = block.info.stage[1].present; break;
		case GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER: *params = block.info.stage[2].present; break;
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


// MARK: - Shader-storage block / program-resource introspection

static GLenum storageVariableGLType(const TranslatedUniform &uniform) {
	const int rows = std::max(1, std::min(uniform.vecsize, 4));
	if (uniform.columns > 1) {
		if (uniform.columns == rows) {
			static const GLenum squareTypes[] = { GL_FLOAT_MAT2, GL_FLOAT_MAT3, GL_FLOAT_MAT4 };
			return (uniform.columns >= 2 && uniform.columns <= 4) ? squareTypes[uniform.columns - 2] : GL_FLOAT_MAT4;
		}
		static const GLenum matrixTypes[4][4] = {
			{ 0, 0, 0, 0 },
			{ 0, 0, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4 },
			{ 0, GL_FLOAT_MAT3x2, 0, GL_FLOAT_MAT3x4 },
			{ 0, GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, 0 }
		};
		if (uniform.columns >= 2 && uniform.columns <= 4 && rows >= 2 && rows <= 4)
			return matrixTypes[uniform.columns][rows];
	}
	static const GLenum floatTypes[] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
	static const GLenum intTypes[] = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
	static const GLenum uintTypes[] = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
	static const GLenum boolTypes[] = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
	switch (uniform.kind) {
		case TranslatedUniform::Int: return intTypes[rows - 1];
		case TranslatedUniform::UInt: return uintTypes[rows - 1];
		case TranslatedUniform::Bool: return boolTypes[rows - 1];
		default: return floatTypes[rows - 1];
	}
}

static std::string storageVariableResourceName(const TranslatedUniform &member) {
	return member.array_size != 1 ? member.name + "[0]" : member.name;
}

static bool getStorageVariableByIndex(const ProgramObject &program, GLuint resourceIndex,
	GLuint &blockIndex, const TranslatedStorageBuffer *&block, const TranslatedUniform *&member) {
	GLuint currentIndex = 0;
	for (size_t candidateBlock = 0; candidateBlock < program.storage_buffers.size(); ++candidateBlock) {
		const TranslatedStorageBuffer &candidate = program.storage_buffers[candidateBlock];
		for (const TranslatedUniform &candidateMember : candidate.members) {
			if (currentIndex == resourceIndex) {
				blockIndex = (GLuint)candidateBlock;
				block = &candidate;
				member = &candidateMember;
				return true;
			}
			++currentIndex;
		}
	}
	return false;
}

void glGetProgramInterfaceiv(GLuint program, GLenum programInterface, GLenum pname, GLint *params) {
	if (!params) { gl4metalSetError(GL_INVALID_VALUE); return; }
	auto programIt = ctx.program_objects.find(program);
	if (programIt == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		*params = 0;
		return;
	}
	if (!programIt->second.linked) {
		gl4metalSetError(GL_INVALID_OPERATION);
		*params = 0;
		return;
	}
	const ProgramObject &object = programIt->second;
	if (programInterface != GL_SHADER_STORAGE_BLOCK && programInterface != GL_BUFFER_VARIABLE) {
		gl4metalSetError(GL_INVALID_ENUM);
		*params = 0;
		return;
	}
	size_t resourceCount = 0;
	size_t maxNameLength = 0;
	size_t maxVariables = 0;
	if (programInterface == GL_SHADER_STORAGE_BLOCK) {
		resourceCount = object.storage_buffers.size();
		for (const TranslatedStorageBuffer &block : object.storage_buffers) {
			maxNameLength = std::max(maxNameLength, block.name.size() + 1);
			maxVariables = std::max(maxVariables, block.members.size());
		}
	} else {
		for (const TranslatedStorageBuffer &block : object.storage_buffers) {
			resourceCount += block.members.size();
			for (const TranslatedUniform &member : block.members)
				maxNameLength = std::max(maxNameLength, storageVariableResourceName(member).size() + 1);
		}
	}
	switch (pname) {
		case GL_ACTIVE_RESOURCES: *params = (GLint)resourceCount; break;
		case GL_MAX_NAME_LENGTH: *params = (GLint)maxNameLength; break;
		case GL_MAX_NUM_ACTIVE_VARIABLES:
			if (programInterface != GL_SHADER_STORAGE_BLOCK) {
				gl4metalSetError(GL_INVALID_OPERATION); *params = 0;
			} else *params = (GLint)maxVariables;
			break;
		default:
			gl4metalSetError(GL_INVALID_ENUM);
			*params = 0;
			break;
	}
}

GLuint glGetProgramResourceIndex(GLuint program, GLenum programInterface, const GLchar *name) {
	if (!name) { gl4metalSetError(GL_INVALID_VALUE); return GL_INVALID_INDEX; }
	auto programIt = ctx.program_objects.find(program);
	if (programIt == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return GL_INVALID_INDEX;
	}
	if (!programIt->second.linked) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return GL_INVALID_INDEX;
	}
	const ProgramObject &object = programIt->second;
	if (programInterface == GL_SHADER_STORAGE_BLOCK) {
		for (size_t blockIndex = 0; blockIndex < object.storage_buffers.size(); ++blockIndex)
			if (object.storage_buffers[blockIndex].name == name) return (GLuint)blockIndex;
	} else if (programInterface == GL_BUFFER_VARIABLE) {
		GLuint resourceIndex = 0;
		for (const TranslatedStorageBuffer &block : object.storage_buffers) {
			for (const TranslatedUniform &member : block.members) {
				const std::string resourceName = storageVariableResourceName(member);
				if (resourceName == name || (member.array_size != 1 && member.name == name)) return resourceIndex;
				++resourceIndex;
			}
		}
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
	}
	return GL_INVALID_INDEX;
}

void glGetProgramResourceName(GLuint program, GLenum programInterface, GLuint index,
	GLsizei bufSize, GLsizei *length, GLchar *name) {
	if (bufSize < 0 || (bufSize > 0 && !name)) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		return;
	}
	auto programIt = ctx.program_objects.find(program);
	if (programIt == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		if (length) *length = 0;
		return;
	}
	if (!programIt->second.linked) {
		gl4metalSetError(GL_INVALID_OPERATION);
		if (length) *length = 0;
		return;
	}
	const ProgramObject &object = programIt->second;
	if (programInterface == GL_SHADER_STORAGE_BLOCK) {
		if (index >= object.storage_buffers.size()) { gl4metalSetError(GL_INVALID_VALUE); if (length) *length = 0; return; }
		copyGLString(object.storage_buffers[index].name, bufSize, length, name);
		return;
	}
	if (programInterface == GL_BUFFER_VARIABLE) {
		GLuint blockIndex = 0;
		const TranslatedStorageBuffer *block = nullptr;
		const TranslatedUniform *member = nullptr;
		if (!getStorageVariableByIndex(object, index, blockIndex, block, member)) {
			gl4metalSetError(GL_INVALID_VALUE); if (length) *length = 0; return;
		}
		copyGLString(storageVariableResourceName(*member), bufSize, length, name);
		return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
	if (length) *length = 0;
}

void glGetProgramResourceiv(GLuint program, GLenum programInterface, GLuint index,
	GLsizei propCount, const GLenum *props, GLsizei bufSize, GLsizei *length, GLint *params) {
	if (length) *length = 0;
	if (propCount <= 0 || bufSize < 0 || !props || (bufSize != 0 && !params)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	auto programIt = ctx.program_objects.find(program);
	if (programIt == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!programIt->second.linked) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const ProgramObject &object = programIt->second;
	const TranslatedStorageBuffer *storageBlock = nullptr;
	const TranslatedUniform *storageMember = nullptr;
	GLuint parentBlockIndex = 0;
	if (programInterface == GL_SHADER_STORAGE_BLOCK) {
		if (index >= object.storage_buffers.size()) { gl4metalSetError(GL_INVALID_VALUE); return; }
		storageBlock = &object.storage_buffers[index];
	} else if (programInterface == GL_BUFFER_VARIABLE) {
		if (!getStorageVariableByIndex(object, index, parentBlockIndex, storageBlock, storageMember)) {
			gl4metalSetError(GL_INVALID_VALUE); return;
		}
	} else {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}

	auto stageReferenced = [](bool vertexPresent, bool fragmentPresent, bool computePresent, GLenum property) -> GLint {
		switch (property) {
			case GL_REFERENCED_BY_VERTEX_SHADER: return vertexPresent ? GL_TRUE : GL_FALSE;
			case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
			case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
			case GL_REFERENCED_BY_GEOMETRY_SHADER: return GL_FALSE;
			case GL_REFERENCED_BY_FRAGMENT_SHADER: return fragmentPresent ? GL_TRUE : GL_FALSE;
			case GL_REFERENCED_BY_COMPUTE_SHADER: return computePresent ? GL_TRUE : GL_FALSE;
			default: return -1;
		}
	};
	auto memberStage = [](const TranslatedUniform &member) -> const TranslatedUniform::Stage & {
		return member.stage[member.stage[0].present ? 0 : (member.stage[1].present ? 1 : 2)];
	};
	auto isReferencedStageProperty = [](GLenum property) {
		return property == GL_REFERENCED_BY_VERTEX_SHADER ||
			property == GL_REFERENCED_BY_TESS_CONTROL_SHADER ||
			property == GL_REFERENCED_BY_TESS_EVALUATION_SHADER ||
			property == GL_REFERENCED_BY_GEOMETRY_SHADER ||
			property == GL_REFERENCED_BY_FRAGMENT_SHADER ||
			property == GL_REFERENCED_BY_COMPUTE_SHADER;
	};
	auto isKnownResourceProperty = [](GLenum property) {
		switch (property) {
			case GL_NAME_LENGTH:
			case GL_TYPE:
			case GL_ARRAY_SIZE:
			case GL_OFFSET:
			case GL_BLOCK_INDEX:
			case GL_ARRAY_STRIDE:
			case GL_MATRIX_STRIDE:
			case GL_IS_ROW_MAJOR:
			case GL_BUFFER_BINDING:
			case GL_BUFFER_DATA_SIZE:
			case GL_NUM_ACTIVE_VARIABLES:
			case GL_ACTIVE_VARIABLES:
			case GL_REFERENCED_BY_VERTEX_SHADER:
			case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
			case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
			case GL_REFERENCED_BY_GEOMETRY_SHADER:
			case GL_REFERENCED_BY_FRAGMENT_SHADER:
			case GL_REFERENCED_BY_COMPUTE_SHADER:
			case GL_TOP_LEVEL_ARRAY_SIZE:
			case GL_TOP_LEVEL_ARRAY_STRIDE:
				return true;
			default:
				return false;
		}
	};
	GLsizei written = 0;
	auto appendValue = [&](GLint value) {
		if (written < bufSize) params[written++] = value;
	};
	for (GLsizei propIndex = 0; propIndex < propCount; ++propIndex) {
		const GLenum property = props[propIndex];
		if (!isKnownResourceProperty(property)) {
			gl4metalSetError(GL_INVALID_ENUM);
			continue;
		}
		const bool allowedForInterface = programInterface == GL_SHADER_STORAGE_BLOCK
			? (property == GL_NAME_LENGTH || property == GL_BUFFER_BINDING || property == GL_BUFFER_DATA_SIZE ||
				property == GL_NUM_ACTIVE_VARIABLES || property == GL_ACTIVE_VARIABLES || isReferencedStageProperty(property))
			: (property == GL_NAME_LENGTH || property == GL_TYPE || property == GL_ARRAY_SIZE ||
				property == GL_OFFSET || property == GL_BLOCK_INDEX || property == GL_ARRAY_STRIDE ||
				property == GL_MATRIX_STRIDE || property == GL_IS_ROW_MAJOR ||
				property == GL_TOP_LEVEL_ARRAY_SIZE || property == GL_TOP_LEVEL_ARRAY_STRIDE ||
				isReferencedStageProperty(property));
		if (!allowedForInterface) {
			gl4metalSetError(GL_INVALID_OPERATION);
			continue;
		}
		if (programInterface == GL_SHADER_STORAGE_BLOCK) {
			const GLint referenced = stageReferenced(storageBlock->stage[0].present,
				storageBlock->stage[1].present, storageBlock->stage[2].present, property);
			if (referenced >= 0) { appendValue(referenced); continue; }
			switch (property) {
				case GL_BUFFER_BINDING: appendValue((GLint)storageBlock->binding); break;
				case GL_BUFFER_DATA_SIZE: appendValue((GLint)storageBlock->required_size); break;
				case GL_NUM_ACTIVE_VARIABLES: appendValue((GLint)storageBlock->members.size()); break;
				case GL_ACTIVE_VARIABLES: {
					GLuint baseIndex = 0;
					for (size_t blockIndex = 0; blockIndex < index; ++blockIndex)
						baseIndex += (GLuint)object.storage_buffers[blockIndex].members.size();
					for (size_t memberIndex = 0; memberIndex < storageBlock->members.size() && written < bufSize; ++memberIndex)
						appendValue((GLint)(baseIndex + (GLuint)memberIndex));
					break;
				}
				case GL_NAME_LENGTH: appendValue((GLint)storageBlock->name.size() + 1); break;
				default: gl4metalSetError(GL_INVALID_ENUM); break;
			}
		} else {
			const TranslatedUniform::Stage &stage = memberStage(*storageMember);
			const GLint referenced = stageReferenced(storageMember->stage[0].present,
				storageMember->stage[1].present, storageMember->stage[2].present, property);
			if (referenced >= 0) { appendValue(referenced); continue; }
			switch (property) {
				case GL_TYPE: appendValue((GLint)storageVariableGLType(*storageMember)); break;
				case GL_ARRAY_SIZE: appendValue(storageMember->array_size); break;
				case GL_OFFSET: appendValue((GLint)stage.offset); break;
				case GL_ARRAY_STRIDE: appendValue((GLint)stage.array_stride); break;
				case GL_MATRIX_STRIDE: appendValue((GLint)stage.matrix_stride); break;
				case GL_IS_ROW_MAJOR: appendValue(stage.row_major ? GL_TRUE : GL_FALSE); break;
				case GL_BLOCK_INDEX: appendValue((GLint)parentBlockIndex); break;
				case GL_TOP_LEVEL_ARRAY_SIZE: appendValue(storageMember->array_size); break;
				case GL_TOP_LEVEL_ARRAY_STRIDE: appendValue((GLint)stage.array_stride); break;
				case GL_NAME_LENGTH: appendValue((GLint)storageVariableResourceName(*storageMember).size() + 1); break;
				default: gl4metalSetError(GL_INVALID_ENUM); break;
			}
		}
	}
	if (length) *length = written;
}

void glShaderStorageBlockBinding(GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding) {
	auto programIt = ctx.program_objects.find(program);
	if (programIt == ctx.program_objects.end()) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	if (!programIt->second.linked) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (storageBlockIndex >= programIt->second.storage_buffers.size() ||
		storageBlockBinding >= GL4METAL_MAX_SHADER_STORAGE_BUFFER_BINDINGS) {
		gl4metalSetError(GL_INVALID_VALUE);
		GL4METAL_LOG("glShaderStorageBlockBinding: invalid block %u or binding point %u",
			storageBlockIndex, storageBlockBinding);
		return;
	}
	programIt->second.storage_buffers[storageBlockIndex].binding = storageBlockBinding;
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
