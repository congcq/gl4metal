cmake_minimum_required(VERSION 3.10)
project(gl4metal)

set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glslang EXCLUDE_FROM_ALL)

set(SPIRV_CROSS_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
add_subdirectory(external/SPIRV-Cross EXCLUDE_FROM_ALL)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target arm64-apple-ios -fcommon -fobjc-arc -ObjC -F'${CMAKE_OSX_SYSROOT}/System/Cryptexes/OS/System/Library/Frameworks'")
set(CMAKE_C_FLAGS_DEBUG "-funwind-tables -g")

include_directories(
    "."
    "include"
)

add_compile_options(-O2)

add_library(${CMAKE_PROJECT_NAME} SHARED
    getter.m
    gl4metal.m
    gl4metal_runtime.m
    gl4metal_shaders.m
    objects.m
    shader_compiler.mm
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    glslang
    glslang-default-resource-limits
    SPIRV
    spirv-cross-c
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
)