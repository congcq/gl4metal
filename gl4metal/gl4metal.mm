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
	for (int attachment = 0; attachment < GL4METAL_MAX_COLOR_ATTACHMENTS; ++attachment) {
		ctx.blend_enabled[attachment] = GL_FALSE;
		for (int component = 0; component < 4; ++component)
			ctx.color_masks[attachment][component] = GL_TRUE;
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

static MTLStencilOperation getMetalStencilOperation(GLenum operation) {
	switch (operation) {
		case GL_KEEP: return MTLStencilOperationKeep;
		case GL_ZERO: return MTLStencilOperationZero;
		case GL_REPLACE: return MTLStencilOperationReplace;
		case GL_INCR: return MTLStencilOperationIncrementClamp;
		case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
		case GL_DECR: return MTLStencilOperationDecrementClamp;
		case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
		case GL_INVERT: return MTLStencilOperationInvert;
		default: return MTLStencilOperationKeep;
	}
}

static MTLStencilDescriptor *makeStencilDescriptor(const MTLContext::StencilFaceState &state) {
	MTLStencilDescriptor *descriptor = [[MTLStencilDescriptor alloc] init];
	descriptor.stencilCompareFunction = getMetalCompareFunc(state.func);
	descriptor.stencilFailureOperation = getMetalStencilOperation(state.fail);
	descriptor.depthFailureOperation = getMetalStencilOperation(state.depth_fail);
	descriptor.depthStencilPassOperation = getMetalStencilOperation(state.depth_pass);
	descriptor.readMask = state.value_mask;
	descriptor.writeMask = state.write_mask;
	return descriptor;
}

void updateDepthStencilState() {
	static std::unordered_map<std::string, id<MTLDepthStencilState>> cache;
	static std::mutex cacheMutex;
	std::lock_guard<std::mutex> lock(cacheMutex);

	const bool depthEnabled = ctx.depth_test_enabled != GL_FALSE;
	const bool stencilEnabled = ctx.caps[GL_STENCIL_TEST] != GL_FALSE;
	const auto &front = ctx.stencil_front;
	const auto &back = ctx.stencil_back;
	std::string key;
	auto append = [&key](uint64_t value) {
		key.append(reinterpret_cast<const char *>(&value), sizeof(value));
	};
	append(depthEnabled); append(ctx.depth_write_enabled); append(ctx.depth_func);
	append(stencilEnabled);
	append(front.func); append((uint32_t)front.ref); append(front.value_mask); append(front.write_mask);
	append(front.fail); append(front.depth_fail); append(front.depth_pass);
	append(back.func); append((uint32_t)back.ref); append(back.value_mask); append(back.write_mask);
	append(back.fail); append(back.depth_fail); append(back.depth_pass);

	auto it = cache.find(key);
	if (it == cache.end() || !it->second) {
		MTLDepthStencilDescriptor *descriptor = [[MTLDepthStencilDescriptor alloc] init];
		descriptor.depthCompareFunction = depthEnabled ? getMetalCompareFunc(ctx.depth_func) : MTLCompareFunctionAlways;
		descriptor.depthWriteEnabled = depthEnabled && ctx.depth_write_enabled;
		if (stencilEnabled) {
			descriptor.frontFaceStencil = makeStencilDescriptor(front);
			descriptor.backFaceStencil = makeStencilDescriptor(back);
		} else {
			// A disabled stencil test must not mutate the stencil attachment.
			MTLStencilDescriptor *disabled = [[MTLStencilDescriptor alloc] init];
			disabled.stencilCompareFunction = MTLCompareFunctionAlways;
			disabled.stencilFailureOperation = MTLStencilOperationKeep;
			disabled.depthFailureOperation = MTLStencilOperationKeep;
			disabled.depthStencilPassOperation = MTLStencilOperationKeep;
			disabled.readMask = 0xFFFFFFFFu;
			disabled.writeMask = 0;
			descriptor.frontFaceStencil = disabled;
			descriptor.backFaceStencil = disabled;
		}
		id<MTLDepthStencilState> created = [ctx.device newDepthStencilStateWithDescriptor:descriptor];
		if (it == cache.end()) it = cache.emplace(key, created).first;
		else it->second = created;
	}
	ctx.depthStencilState = it->second;
}
