cmake_minimum_required(VERSION 3.10)
project(gl4metal)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target arm64-apple-ios -mcpu=apple-a8 -fcommon -fobjc-arc -ObjC -F'${CMAKE_OSX_SYSROOT}/System/Cryptexes/OS/System/Library/Frameworks'")
set(CMAKE_C_FLAGS_DEBUG "-funwind-tables -g")

include_directories(
    "."
    "include"
)

add_library(${CMAKE_PROJECT_NAME} SHARED
    gl4metal.m
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
)