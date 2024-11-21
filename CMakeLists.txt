cmake_minimum_required(VERSION 3.10)

# Set project name
project(Eye)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)
#find_package(Ceres REQUIRED) ${CERES_INCLUDE_DIRS} ${CERES_LIBRARIES}
find_package(nlohmann_json REQUIRED)

# Add ED_Lib library path
#add_subdirectory(libs/ED_Lib)

add_subdirectory(libs/LibCamera)

# Find libcamera package
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Find Eigen (assuming it's in a folder like libs/Eigen)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories for OpenCV, ED_Lib, and Eigen ${EIGEN3_INCLUDE_DIR}, libs/ED_Lib
include_directories(${OpenCV_INCLUDE_DIRS} libs/LibCamera ${LIBCAMERA_INCLUDE_DIRS})

# Add your executable
add_executable(eye src/main.cpp)

# Link ED_Lib, OpenCV, and Eigen to your project Eigen3::Eigen, EDLib
target_link_libraries(eye ${OpenCV_LIBS} LibCamera ${LIBCAMERA_LIBRARIES} nlohmann_json::nlohmann_json Eigen3::Eigen)