#import "gl4metal.h"

#include <mutex>

MTLContext ctx; // context for Metal device

static std::mutex gl4metalInitMutex;
static bool gl4metalInitialized = false;

void initDevice() {
	// Keep the device identity stable. Replacing it after resources have been created
	// would leave textures, buffers and pipeline states belonging to the old device.
	if (!ctx.device) {
		ctx.device = MTLCreateSystemDefaultDevice();
		if (ctx.device) {
			GL4METAL_LOG("Initialized context on device: %s", ctx.device.name.UTF8String ?: "(unnamed)");
		}
	}
	if (ctx.device && !ctx.commandQueue) {
		ctx.commandQueue = [ctx.device newCommandQueue];
	}
}

GLboolean gl4metalInit() {
	std::lock_guard<std::mutex> initLock(gl4metalInitMutex);
	if (gl4metalInitialized) return GL_TRUE;
	initDevice();
	if (!ctx.device || !ctx.commandQueue) {
		GL4METAL_LOG("gl4metalInit: Metal device or command queue creation failed");
		return GL_FALSE;
	}
	const char *check = getenv("GL4METAL_CHECK_EXPORTS");
	if (check && check[0] == '1') {
		gl4metalCheckExports();
	}

	// OpenGL initial state
	ctx.depth_write_enabled = GL_TRUE;
	ctx.depth_func = GL_LESS;
	ctx.clear_color = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
	ctx.clear_depth = 1.0;
	ctx.clear_stencil = 0;
	ctx.unpack_alignment = 4;
	ctx.pack_alignment = 4;
	ctx.caps[GL_DITHER] = GL_TRUE;
	ctx.caps[GL_MULTISAMPLE] = GL_TRUE;

	ctx.blend_src_rgb = ctx.blend_src_alpha = GL_ONE;
	ctx.blend_dst_rgb = ctx.blend_dst_alpha = GL_ZERO;
	ctx.blend_eq_rgb = ctx.blend_eq_alpha = GL_FUNC_ADD;
	for (int i = 0; i < 4; i++) {
		ctx.blend_color[i] = 0.0f;
		ctx.color_mask[i] = GL_TRUE;
	}
	ctx.cull_face_mode = GL_BACK;
	ctx.front_face = GL_CCW;
	ctx.polygon_mode = GL_FILL;
	ctx.polygon_offset_factor = ctx.polygon_offset_units = 0.0f;
	ctx.depth_range_near = 0.0f;
	ctx.depth_range_far = 1.0f;

	// Depth-stencil state for passes without a depth attachment: always pass, never write.
	MTLDepthStencilDescriptor *noDepthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
	noDepthDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
	noDepthDescriptor.depthWriteEnabled = NO;
	ctx.no_depth_state = [ctx.device newDepthStencilStateWithDescriptor:noDepthDescriptor];
	if (!ctx.no_depth_state) {
		GL4METAL_LOG("gl4metalInit: failed to create no-depth-stencil state");
		return GL_FALSE;
	}

	// GL's default value of a generic vertex attribute that has no array enabled is (0, 0, 0, 1).
	const float f[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const int32_t n[4] = { 0, 0, 0, 1 };
	uint8_t bytes[32];
	memcpy(bytes, f, 16);
	memcpy(bytes + 16, n, 16);
	ctx.constant_attrib_buffer = [ctx.device newBufferWithBytes:bytes length:sizeof(bytes) options:MTLResourceStorageModeShared];
	if (!ctx.constant_attrib_buffer) {
		GL4METAL_LOG("gl4metalInit: failed to create constant vertex-attribute buffer");
		return GL_FALSE;
	}

	updateDepthStencilState();
	if (!ctx.depthStencilState) {
		GL4METAL_LOG("gl4metalInit: failed to create initial depth-stencil state");
		return GL_FALSE;
	}
	gl4metalInitialized = true;
	return GL_TRUE;
}

void* GL4MetalCreateContext(GLenum format, MTLContext sharelist) {
	return NULL; // don't need yet
}

MTLCompareFunction getMetalCompareFunc(GLenum func) {
	switch(func) {
		case GL_NEVER:
			return MTLCompareFunctionNever;
		case GL_LESS:
			return MTLCompareFunctionLess;
		case GL_EQUAL:
			return MTLCompareFunctionEqual;
		case GL_LEQUAL:
			return MTLCompareFunctionLessEqual;
		case GL_GREATER:
			return MTLCompareFunctionGreater;
		case GL_NOTEQUAL:
			return MTLCompareFunctionNotEqual;
		case GL_GEQUAL:
			return MTLCompareFunctionGreaterEqual;
		case GL_ALWAYS:
			return MTLCompareFunctionAlways;
		default:
			GL4METAL_LOG("Missing compare function: 0x%X, defaulting to GL_ALWAYS", func);
			return MTLCompareFunctionAlways;
	}
}

void updateDepthStencilState() {
	// Depth-stencil states are immutable Metal objects, so cache one per (test, write, func) combination.
	static std::unordered_map<uint32_t, id<MTLDepthStencilState>> cache;
	static std::mutex cacheMutex;
	std::lock_guard<std::mutex> lock(cacheMutex);

	uint32_t key = 0; // depth test disabled: always pass, never write
	if (ctx.depth_test_enabled) {
		key = 1u | (ctx.depth_write_enabled ? 2u : 0u) | ((uint32_t)(ctx.depth_func & 7u) << 2);
	}

	auto it = cache.find(key);
	if (it == cache.end() || !it->second) {
		MTLDepthStencilDescriptor *dc = [[MTLDepthStencilDescriptor alloc] init];
		if (ctx.depth_test_enabled) {
			dc.depthCompareFunction = getMetalCompareFunc(ctx.depth_func);
			dc.depthWriteEnabled = ctx.depth_write_enabled;
		} else {
			dc.depthCompareFunction = MTLCompareFunctionAlways;
			dc.depthWriteEnabled = NO;
		}
		id<MTLDepthStencilState> created = [ctx.device newDepthStencilStateWithDescriptor:dc];
		if (it == cache.end()) {
			it = cache.emplace(key, created).first;
		} else {
			it->second = created; // allow retry after a transient Metal allocation failure
		}
	}

	ctx.depthStencilState = it->second;
}
