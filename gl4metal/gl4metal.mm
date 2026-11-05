#import "gl4metal.h"

MTLContext ctx; // context for Metal device

void initDevice() {
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();

	if (device) {
		GL4METAL_LOG("Initialized context on device: %s", device.name.UTF8String);
	}

	ctx.device = device;
	ctx.commandQueue = [ctx.device newCommandQueue];
}

GLboolean gl4metalInit() {
	if (ctx.device) {
		return GL_TRUE;
	}
	initDevice();
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
	ctx.no_depth_state = [ctx.device newDepthStencilStateWithDescriptor:[[MTLDepthStencilDescriptor alloc] init]];

	// GL's default value of a generic vertex attribute that has no array enabled is (0, 0, 0, 1).
	const float f[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const int32_t n[4] = { 0, 0, 0, 1 };
	uint8_t bytes[32];
	memcpy(bytes, f, 16);
	memcpy(bytes + 16, n, 16);
	ctx.constant_attrib_buffer = [ctx.device newBufferWithBytes:bytes length:sizeof(bytes) options:MTLResourceStorageModeShared];

	updateDepthStencilState();
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

	uint32_t key = 0; // depth test disabled: always pass, never write
	if (ctx.depth_test_enabled) {
		key = 1u | (ctx.depth_write_enabled ? 2u : 0u) | ((uint32_t)(ctx.depth_func & 7u) << 2);
	}

	auto it = cache.find(key);
	if (it == cache.end()) {
		MTLDepthStencilDescriptor *dc = [[MTLDepthStencilDescriptor alloc] init];
		if (ctx.depth_test_enabled) {
			dc.depthCompareFunction = getMetalCompareFunc(ctx.depth_func);
			dc.depthWriteEnabled = ctx.depth_write_enabled;
		} else {
			dc.depthCompareFunction = MTLCompareFunctionAlways;
			dc.depthWriteEnabled = NO;
		}
		it = cache.emplace(key, [ctx.device newDepthStencilStateWithDescriptor:dc]).first;
	}

	ctx.depthStencilState = it->second;
}
