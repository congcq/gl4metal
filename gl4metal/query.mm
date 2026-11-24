#import "query.h"

#include <limits>

static bool validOcclusionQueryTarget(GLenum target) {
	return target == GL_SAMPLES_PASSED || target == GL_ANY_SAMPLES_PASSED ||
		target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE;
}

static bool validQueryObjectPname(GLenum pname) {
	return pname == GL_QUERY_RESULT || pname == GL_QUERY_RESULT_AVAILABLE ||
		pname == GL_QUERY_RESULT_NO_WAIT;
}

static GLQueryObject *findQueryObject(GLuint queryName) {
	auto it = ctx.query_objects.find(queryName);
	return it == ctx.query_objects.end() ? nullptr : &it->second;
}

static bool validateQueryOutput(const void *params) {
	if (params) return true;
	gl4metalSetError(GL_INVALID_VALUE);
	return false;
}

static void deleteQueryObjectIfInactive(GLuint queryName) {
	auto it = ctx.query_objects.find(queryName);
	if (it != ctx.query_objects.end() && it->second.delete_pending && !it->second.active) {
		ctx.query_objects.erase(it);
	}
}

void glGenQueries(GLsizei n, GLuint *queries) {
	if (n < 0 || (n > 0 && !queries)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei queryIndex = 0; queryIndex < n; ++queryIndex) {
		GLuint queryName = 0;
		do {
			++ctx.next_query_name;
			if (ctx.next_query_name == 0) ++ctx.next_query_name;
			queryName = ctx.next_query_name;
		} while (ctx.query_names.find(queryName) != ctx.query_names.end() ||
			ctx.query_objects.find(queryName) != ctx.query_objects.end());
		ctx.query_names.insert(queryName);
		queries[queryIndex] = queryName;
	}
}

void glDeleteQueries(GLsizei n, const GLuint *queries) {
	if (n < 0 || (n > 0 && !queries)) {
		gl4metalSetError(GL_INVALID_VALUE);
		return;
	}
	for (GLsizei queryIndex = 0; queryIndex < n; ++queryIndex) {
		const GLuint queryName = queries[queryIndex];
		ctx.query_names.erase(queryName);
		auto objectIt = ctx.query_objects.find(queryName);
		if (objectIt == ctx.query_objects.end()) continue;
		if (objectIt->second.active) {
			objectIt->second.delete_pending = true;
		} else {
			ctx.query_objects.erase(objectIt);
		}
	}
}

GLboolean glIsQuery(GLuint queryName) {
	auto it = ctx.query_objects.find(queryName);
	return it != ctx.query_objects.end() && !it->second.delete_pending ? GL_TRUE : GL_FALSE;
}

void glBeginQuery(GLenum target, GLuint queryName) {
	if (!validOcclusionQueryTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (queryName == 0 || (ctx.query_names.find(queryName) == ctx.query_names.end() &&
		ctx.query_objects.find(queryName) == ctx.query_objects.end())) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (ctx.active_queries.find(target) != ctx.active_queries.end()) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	GLQueryObject *queryObject = findQueryObject(queryName);
	if (queryObject && (queryObject->active || queryObject->delete_pending ||
		(queryObject->target != 0 && queryObject->target != target))) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	GLQueryObject &newQueryObject = ctx.query_objects[queryName];
	newQueryObject.target = target;
	newQueryObject.result = 0;
	newQueryObject.active = true;
	newQueryObject.available = false;
	ctx.active_queries[target] = queryName;
}

void glEndQuery(GLenum target) {
	if (!validOcclusionQueryTarget(target)) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	auto activeIt = ctx.active_queries.find(target);
	if (activeIt == ctx.active_queries.end()) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	const GLuint queryName = activeIt->second;
	ctx.active_queries.erase(activeIt);
	auto objectIt = ctx.query_objects.find(queryName);
	if (objectIt == ctx.query_objects.end()) return;
	GLQueryObject &queryObject = objectIt->second;
	queryObject.active = false;
	// Conservative iOS fallback: Metal occlusion results are not currently wired
	// into the GL query path. Report visible rather than incorrectly culling a
	// world region; this is deterministic but not a hardware-accurate sample count.
	queryObject.result = 1;
	queryObject.available = true;
	deleteQueryObjectIfInactive(queryName);
}

void glQueryCounter(GLuint queryName, GLenum target) {
	if (target != GL_TIMESTAMP) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	if (queryName == 0 || (ctx.query_names.find(queryName) == ctx.query_names.end() &&
		ctx.query_objects.find(queryName) == ctx.query_objects.end())) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	gl4metalSetError(GL_INVALID_OPERATION); // GPU timestamp queries require Metal counter sampling, not CPU wall time.
}

void glGetQueryiv(GLenum target, GLenum pname, GLint *params) {
	if (!validateQueryOutput(params)) return;
	if (!validOcclusionQueryTarget(target) && target != GL_TIMESTAMP) {
		gl4metalSetError(GL_INVALID_ENUM);
		*params = 0;
		return;
	}
	if (pname == GL_QUERY_COUNTER_BITS) {
		// No GPU occlusion/timestamp counter is currently exposed by this backend.
		*params = 0;
		return;
	}
	if (pname == GL_CURRENT_QUERY) {
		auto activeIt = ctx.active_queries.find(target);
		*params = activeIt == ctx.active_queries.end() ? 0 : (GLint)activeIt->second;
		return;
	}
	gl4metalSetError(GL_INVALID_ENUM);
	*params = 0;
}

template <typename ValueType>
static void getQueryObjectValue(GLuint queryName, GLenum pname, ValueType *params) {
	if (!validateQueryOutput(params)) return;
	GLQueryObject *queryObject = findQueryObject(queryName);
	if (!queryObject || queryObject->delete_pending) {
		gl4metalSetError(GL_INVALID_OPERATION);
		*params = (ValueType)0;
		return;
	}
	if (!validQueryObjectPname(pname)) {
		gl4metalSetError(GL_INVALID_ENUM);
		*params = (ValueType)0;
		return;
	}
	if (pname == GL_QUERY_RESULT_AVAILABLE) {
		*params = queryObject->available ? (ValueType)GL_TRUE : (ValueType)GL_FALSE;
		return;
	}
	if (pname == GL_QUERY_RESULT_NO_WAIT && !queryObject->available) {
		*params = (ValueType)0;
		return;
	}
	if (queryObject->active || !queryObject->available) {
		gl4metalSetError(GL_INVALID_OPERATION);
		*params = (ValueType)0;
		return;
	}
	*params = (ValueType)queryObject->result;
}

void glGetQueryObjectiv(GLuint queryName, GLenum pname, GLint *params) {
	getQueryObjectValue(queryName, pname, params);
}

void glGetQueryObjectuiv(GLuint queryName, GLenum pname, GLuint *params) {
	getQueryObjectValue(queryName, pname, params);
}

void glGetQueryObjecti64v(GLuint queryName, GLenum pname, GLint64 *params) {
	getQueryObjectValue(queryName, pname, params);
}

void glGetQueryObjectui64v(GLuint queryName, GLenum pname, GLuint64 *params) {
	getQueryObjectValue(queryName, pname, params);
}

// MARK: - Conditional rendering

void glBeginConditionalRender(GLuint queryName, GLenum mode) {
	const bool validMode = mode == GL_QUERY_WAIT || mode == GL_QUERY_NO_WAIT ||
		mode == GL_QUERY_BY_REGION_WAIT || mode == GL_QUERY_BY_REGION_NO_WAIT ||
		mode == GL_QUERY_WAIT_INVERTED || mode == GL_QUERY_NO_WAIT_INVERTED ||
		mode == GL_QUERY_BY_REGION_WAIT_INVERTED || mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
	if (!validMode) {
		gl4metalSetError(GL_INVALID_ENUM);
		return;
	}
	GLQueryObject *queryObject = findQueryObject(queryName);
	if (!queryObject || queryObject->active || !queryObject->available || queryObject->delete_pending) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	if (ctx.conditional_render_active) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	ctx.conditional_render_active = true;
	ctx.conditional_render_query = queryName;
	const bool inverted = mode == GL_QUERY_WAIT_INVERTED || mode == GL_QUERY_NO_WAIT_INVERTED ||
		mode == GL_QUERY_BY_REGION_WAIT_INVERTED || mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
	ctx.conditional_render_should_draw = inverted ? queryObject->result == 0 : queryObject->result != 0;
	// Conservative occlusion results report visible (result=1); inverted modes
	// therefore suppress GL draw commands instead of accidentally rendering them.
}

void glEndConditionalRender(void) {
	if (!ctx.conditional_render_active) {
		gl4metalSetError(GL_INVALID_OPERATION);
		return;
	}
	ctx.conditional_render_active = false;
	ctx.conditional_render_should_draw = true;
	ctx.conditional_render_query = 0;
}
