#import "query.h"

// Metal has no GL-style queries. They answer at once ("available") and report that everything is visible, so a game
// that waits for a result or culls with occlusion queries keeps drawing instead of hanging or hiding the world.
static std::unordered_map<GLuint, GLenum> sQueries; // name -> target of the last glBeginQuery (0 = never used)
static GLuint sNextQuery = 0;

static GLuint64 queryResult(GLuint objectName) {
	auto it = sQueries.find(objectName);
	if (it == sQueries.end()) {
		return 0;
	}
	return (it->second == GL_SAMPLES_PASSED || it->second == GL_ANY_SAMPLES_PASSED) ? 1 : 0;
}

void glGenQueries(GLsizei n, GLuint *ids) {
	for (GLsizei i = 0; i < n; i++) {
		ids[i] = ++sNextQuery;
		sQueries[ids[i]] = 0;
	}
}

void glDeleteQueries(GLsizei n, const GLuint *ids) {
	for (GLsizei i = 0; i < n; i++) {
		sQueries.erase(ids[i]);
	}
}

GLboolean glIsQuery(GLuint objectName) {
	return sQueries.find(objectName) != sQueries.end();
}

void glBeginQuery(GLenum target, GLuint objectName) {
	auto it = sQueries.find(objectName);
	if (it != sQueries.end()) {
		it->second = target;
	}
}

void glEndQuery(GLenum target) {
}

void glQueryCounter(GLuint objectName, GLenum target) {
	auto it = sQueries.find(objectName);
	if (it != sQueries.end()) {
		it->second = target;
	}
}

void glGetQueryiv(GLenum target, GLenum pname, GLint *params) {
	*params = (pname == GL_QUERY_COUNTER_BITS) ? 64 : 0; // GL_CURRENT_QUERY: no query is ever active
}

void glGetQueryObjectiv(GLuint objectName, GLenum pname, GLint *params) {
	*params = (pname == GL_QUERY_RESULT_AVAILABLE) ? GL_TRUE : (GLint)queryResult(objectName);
}

void glGetQueryObjectuiv(GLuint objectName, GLenum pname, GLuint *params) {
	*params = (pname == GL_QUERY_RESULT_AVAILABLE) ? GL_TRUE : (GLuint)queryResult(objectName);
}

void glGetQueryObjecti64v(GLuint objectName, GLenum pname, GLint64 *params) {
	*params = (pname == GL_QUERY_RESULT_AVAILABLE) ? GL_TRUE : (GLint64)queryResult(objectName);
}

void glGetQueryObjectui64v(GLuint objectName, GLenum pname, GLuint64 *params) {
	*params = (pname == GL_QUERY_RESULT_AVAILABLE) ? GL_TRUE : queryResult(objectName);
}

// MARK: - Conditional rendering

void glBeginConditionalRender(GLuint objectName, GLenum mode) {
	// everything is drawn
}

void glEndConditionalRender(void) {
}
