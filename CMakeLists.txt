cmake_minimum_required(VERSION 3.10)
project(gl4metal)

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
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
)