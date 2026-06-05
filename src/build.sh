#!/bin/bash

echo "Building fls-marker-localization..."
mkdir -p build
cd build
cmake ..
make

echo "Copying config to build directory..."
cp ../src/dfrobot_gs_camera_config.json ./camera_config.json

echo "Building blinker..."
g++ -O2 ../src/blinker.cpp -o blinker -llgpio