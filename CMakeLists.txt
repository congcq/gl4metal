cmake_minimum_required(VERSION 3.10)
project(gl4metal)

set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glslang)

set(SPIRV_CROSS_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
add_subdirectory(external/SPIRV-Cross)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target arm64-apple-ios -fcommon -fobjc-arc -ObjC -F'${CMAKE_OSX_SYSROOT}/System/Cryptexes/OS/System/Library/Frameworks'")
set(CMAKE_C_FLAGS_DEBUG "-funwind-tables -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc -fPIC -stdlib=libc++")

include_directories(
    "."
    "include"
    "external"
    "external/SPIRV-Cross"
)

add_compile_options(-flto=thin)
add_link_options(-flto=thin)

add_compile_options(-O2)

add_library(${CMAKE_PROJECT_NAME} SHARED
    gl4metal/gl4metal.mm

    gl4metal/buffers.mm
    gl4metal/context.mm
    gl4metal/draw.mm
    gl4metal/enable.mm
    gl4metal/exports.mm
    gl4metal/framebuffer.mm
    gl4metal/getter.mm
    gl4metal/program.mm
    gl4metal/query.mm
    gl4metal/renderbuffer.mm
    gl4metal/sampler.mm
    gl4metal/shaders.mm
    gl4metal/texture.mm
    gl4metal/transformfeedback.mm
    gl4metal/uniform.mm
    gl4metal/vertexarray.mm

    gl4metal/shadertranslator.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    glslang::glslang
    glslang-default-resource-limits
    spirv-cross-c
)

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
)