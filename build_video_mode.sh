#!/bin/bash

echo "Building fls-marker-localization in VIDEO_INPUT_MODE (no libcamera required)..."
mkdir -p build_video
cd build_video
cmake .. -DVIDEO_INPUT_MODE=ON
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo ""
echo "Copying config to build directory..."
cp ../src/dfrobot_gs_camera_config.json ./camera_config.json

echo ""
echo "Build complete. Run with:"
echo "  ./build_video/eye --video-input <path/to/video.mp4> --aruco -v -p --config camera_config.json"
