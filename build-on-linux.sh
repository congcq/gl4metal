#!/bin/bash

set -e
SDKPATH="/usr/share/SDKs/iPhoneOS.sdk"

# build
rm -rf build
mkdir -p build
cd build && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Darwin \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_OSX_SYSROOT=$SDKPATH \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DCMAKE_C_FLAGS="-arch arm64 -isysroot $SDKPATH -target arm64-apple-ios14" \
    -DCMAKE_CXX_FLAGS="-arch arm64 -isysroot $SDKPATH -target arm64-apple-ios14" \
    ..
cmake --build . --config Release --target gl4metal