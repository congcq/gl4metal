# gl4metal

gl4metal is an OpenGL compatibility library for iOS that implements an OpenGL 3.3 Core Profile over Apple's Metal API. It is intended for applications that load GL entry points at runtime and render to a `CAMetalLayer`, without ANGLE, EGL, OpenGL ES, or MoltenVK.

The library translates GLSL shader source to SPIR-V with glslang, converts SPIR-V to Metal Shading Language with SPIRV-Cross, and manages GL objects, state, command submission, and presentation.

## Features

- exports OpenGL 3.3 Core Profile
- Custom `glMakeCurrent(void *metalLayerPtr)` and `glSwapBuffers()` entry points
- Shader objects and linked programs, including vertex, fragment, and compute stages
- Buffers, textures, samplers, framebuffers, renderbuffers, and vertex arrays
- Uniforms, uniform and shader storage buffer bindings, queries, sync objects, and transform feedback
- GL draw APIs, multisample rendering, blending, depth/stencil state, and render-pass management
- GLSL 330 compatibility preprocessing and SPIR-V to MSL translation
- ARB extensions included in the implementation:
  - `GL_ARB_clear_texture`
  - `GL_ARB_compute_shader`
  - `GL_ARB_draw_buffers_blend`
  - `GL_ARB_invalidate_subdata`
  - `GL_ARB_shader_image_load_store`
- Classic Metal command queues and a Metal 4 compute path on supported iOS devices
- Automatic fallback from Metal 4 to the classic Metal command path

Metal 4 is used only on iOS 26 and newer devices that expose both `MTLGPUFamilyApple7` and `MTLGPUFamilyMetal4`. Older or unsupported hardware uses classic Metal.

## Project status

This project is work in progress. It does not implement the complete OpenGL API.

The reported GL version is `3.3 Core Profile MetalGL`, not a statement that every 3.3 Core API is implemented. Applications should handle missing entry points and unsupported device capabilities.

## Architecture

```text
OpenGL-like application
        |
        | C GL entry points
        v
gl4metal shared library
        |
        +-- Object and GL-state management
        +-- glslang: GLSL -> SPIR-V
        +-- SPIRV-Cross: SPIR-V -> Metal Shading Language
        +-- Classic Metal command encoder path or Metal 4 compute/command path on supported hardware
        |
        v
CAMetalLayer
```

The main source areas are:

| Path | Responsibility |
| --- | --- |
| `gl4metal/exports.h` | Explicit C declarations for exported GL entry points |
| `gl4metal/gl4metal.mm` | Metal device initialization, context state, Metal 4 backend |
| `gl4metal/context.mm` | Layer binding, frame presentation, swap/flush operations |
| `gl4metal/shadertranslator.mm` | GLSL preprocessing, validation, SPIR-V generation, MSL conversion |
| `gl4metal/program.mm` | Shader linking, pipeline creation, uniforms, compute dispatch |
| `gl4metal/buffers.mm` | Buffer objects and mapped/invalidated buffer storage |
| `gl4metal/texture.mm` | Textures, uploads, mipmaps, swizzle views, and sampling |
| `gl4metal/framebuffer.mm` | Framebuffers, render passes, blits, clearing, and invalidation |
| `gl4metal/draw.mm` | Indexed, instanced, and indirect draw submission |
| `gl4metal/extensions/ARB/` | Implementations of the bundled ARB extensions |
| `include/GL/` | Public GL headers bundled with the project |
| `external/glslang/` | GLSL parser and SPIR-V generator submodule |
| `external/SPIRV-Cross/` | SPIR-V reflection and MSL conversion submodule |

The implementation keeps state and Metal resources in a thread-local context. A separate context state is therefore maintained for each calling thread.

## Requirements

To build:

- iOS 26 SDK or 
- CMake 3.10 or newer

To run:

- iOS arm64
- A Metal-capable device
- A `CAMetalLayer` for presentation
- iOS 26 or newer for the Metal 4 path; classic Metal is used elsewhere

## Getting the source

Initialize the bundled submodules:

```bash
git clone --recursive https://github.com/congcq/gl4metal.git
cd gl4metal
```

If the repository has already been cloned:

```bash
git submodule update --init --recursive
```

## Building

### macOS

```bash
mkdir -p build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Darwin \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk iphoneos --show-sdk-path)" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DCMAKE_C_FLAGS="-arch arm64" \
  -DCMAKE_CXX_FLAGS="-arch arm64" \
  ..
cmake --build . --config Release --target gl4metal
```

The main build artifact is:

```text
build/libgl4metal.dylib
```

### Linux cross-build

`build-on-linux.sh` is provided for a Linux environment with [cctools-port](https://github.com/tpoechtrager/cctools-port) and an iPhoneOS 26 SDK at `/usr/share/SDKs/iPhoneOS.sdk`:

```bash
./build-on-linux.sh
```

The script creates `build/` and cross-compiles `gl4metal` for `arm64-apple-ios26.0`. Adjust `SDKPATH` and the deployment target if your SDK is stored elsewhere.

## Runtime integration

The host application must load or link `libgl4metal.dylib`, initialize the backend, attach a `CAMetalLayer`, and present each frame.

Call initialization:

```c
GLboolean gl4metalInit(void);
```

Call `glMakeCurrent()` with a pointer to the application's `CAMetalLayer`:

```c
void glMakeCurrent(void *metalLayerPtr);
```

Call `glSwapBuffers()` once per frame after issuing GL work:

```c
void glSwapBuffers(void);
```

Only GL entry points exported by the library should be resolved dynamically. `gl4metalCheckExports()` can be invoked with `GL4METAL_CHECK_EXPORTS=1` to report unresolved names while initializing.

## Debug environment variables

Set debug variables before initialization or frame execution.

| Variable | Purpose |
| --- | --- |
| `GL4METAL_DISABLE_METAL4=1` | Force the classic Metal command path |
| `GL4METAL_CHECK_EXPORTS=1` | Check the dynamically exported GL entry-point names |
| `GL4METAL_TRACE_METAL4=1` | Report Metal 4 state and in-flight resource information |
| `GL4METAL_TRACE_MEMORY=1` | Report periodically tracked Metal memory and object counts |
| `GL4METAL_TRACE_RENDER_PASSES=1` | Report frame render-pass reuse and pass counts |
| `GL4METAL_TRACE_RENDER_PASS_DETAILS=1` | Report individual render-pass details |
| `GL4METAL_TRACE_RENDER_PASS_TRANSITIONS=1` | Report render-pass resource transitions |
| `GL4METAL_TRACE_DRAW_LABELS=1` | Attach draw labels to Metal debug output |
| `GL4METAL_TRACE_DRAW_BUFFERS=1` | Trace draw-buffer state |
| `GL4METAL_TRACE_TEXTURE_COPIES=1` | Trace texture copy operations |
| `GL4METAL_TRACE_TEXTURE_SWIZZLE=1` | Trace texture swizzle view creation |
| `GL4METAL_TRACE_FBO_MAP=1` | Trace framebuffer object lookup and mapping |
| `GL4METAL_TRACE_VERTEX_FETCH=1` | Trace vertex attribute fetch setup |
| `GL4METAL_TRACE_VERTEX_SAMPLERS=1` | Trace vertex texture/sampler bindings |
| `GL4METAL_TRACE_FRAGMENT_SAMPLERS=1` | Trace fragment texture/sampler bindings |

Debug logging in the source is compiled behind `GL4METAL_DEBUG`; set it to `1` in `gl4metal/gl4metal.h` when the library needs additional internal logs.

## Limitations

- Not every OpenGL 3.3 Core function is implemented.
- Metal 4 APIs require iOS 26 and compatible Apple GPU families.
- Host applications need custom integration for loading the `.dylib` and binding a `CAMetalLayer`.
- `glSwapInterval()` is currently not implemented by a Metal-specific present path.
- Debug trace and compatibility behavior are active areas of development.

## License

This project is licensed under the GNU Lesser General Public License, version 2.1. See [LICENSE](LICENSE).

Bundled subprojects have their own source and license information under `external/glslang/` and `external/SPIRV-Cross/`.
