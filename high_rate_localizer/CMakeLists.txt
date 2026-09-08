cmake_minimum_required(VERSION 3.20)
project(fls_high_rate_localizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLS_LOCALIZER_BUILD_TESTS "Build unit and video tests" ON)
option(FLS_LOCALIZER_ENABLE_LIBCAMERA "Build the production libcamera executable" ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d videoio)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Threads REQUIRED)

add_library(fls_localizer_core
  src/config.cpp
  src/grid_map.cpp
  src/blob_detector.cpp
  src/blink_decoder.cpp
  src/pose_solver.cpp
  src/hypergrid.cpp
  src/trajectory.cpp
  src/pipeline.cpp
  src/shared_memory.cpp
  src/debug_output.cpp)
target_include_directories(fls_localizer_core PUBLIC include)
target_link_libraries(fls_localizer_core
  PUBLIC
    nlohmann_json::nlohmann_json
    opencv_core
    opencv_imgproc
    opencv_calib3d
  PRIVATE
    opencv_videoio
    Threads::Threads)
target_compile_options(fls_localizer_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fls_localizer_video src/video_main.cpp)
target_link_libraries(fls_localizer_video PRIVATE fls_localizer_core opencv_videoio)

if(FLS_LOCALIZER_ENABLE_LIBCAMERA)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBCAMERA QUIET IMPORTED_TARGET libcamera)
  endif()
  if(LIBCAMERA_FOUND)
    add_executable(fls_localizer src/main.cpp src/libcamera_source.cpp)
    target_link_libraries(fls_localizer
      PRIVATE fls_localizer_core PkgConfig::LIBCAMERA)
    target_compile_definitions(fls_localizer PRIVATE FLS_WITH_LIBCAMERA=1)
  else()
    message(STATUS "libcamera not found; production target is skipped")
  endif()
endif()

if(FLS_LOCALIZER_BUILD_TESTS)
  enable_testing()
  add_executable(fls_localizer_tests tests/localizer_tests.cpp)
  target_link_libraries(fls_localizer_tests PRIVATE fls_localizer_core)
  target_compile_definitions(fls_localizer_tests PRIVATE
    FLS_TEST_GRID_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/hypergrid-mygrid-normal.json"
    FLS_TEST_TRAJECTORY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/trajectory.json")
  add_test(NAME fls_localizer_tests COMMAND fls_localizer_tests)
endif()
