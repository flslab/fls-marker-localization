cmake_minimum_required(VERSION 3.10)

# Set project name
project(Eye)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Option: build in video-input-only mode (no libcamera dependency)
option(VIDEO_INPUT_MODE "Build without libcamera; use video file input only" OFF)

# Find OpenCV package (aruco may be in contrib or bundled in objdetect on 4.7+)
find_package(OpenCV REQUIRED)
find_package(OpenCV QUIET COMPONENTS aruco)
#find_package(Ceres REQUIRED) ${CERES_INCLUDE_DIRS} ${CERES_LIBRARIES}
find_package(nlohmann_json REQUIRED)

# Find Eigen (assuming it's in a folder like libs/Eigen)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(NOT VIDEO_INPUT_MODE)
    # Add ED_Lib library path
    #add_subdirectory(libs/ED_Lib)

    add_subdirectory(libs/LibCamera)

    # Find libcamera package
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBCAMERA REQUIRED libcamera)

    # Include directories for OpenCV, ED_Lib, and Eigen ${EIGEN3_INCLUDE_DIR}, libs/ED_Lib
    include_directories(${OpenCV_INCLUDE_DIRS} libs/LibCamera ${LIBCAMERA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
else()
    message(STATUS "VIDEO_INPUT_MODE enabled — skipping libcamera, building video-input-only binary")
    include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
endif()

# Get git commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "unknown")
endif()

# Add your executable
add_executable(eye
    src/main.cpp
    src/application.cpp
    src/application_options.cpp
    src/background_saver.cpp
    src/camera_config.cpp
    src/frame_outputs.cpp
    src/frame_pacer.cpp
    src/frame_processor.cpp
    src/frame_source.cpp
    src/localization_pipeline.cpp
    src/marker_tracker.cpp
    src/pose_estimator.cpp
    src/pose_math.cpp
    src/pose_publisher.cpp
    src/session_log.cpp
)
target_compile_definitions(eye PRIVATE GIT_COMMIT_HASH=${GIT_COMMIT_HASH})

add_executable(marker_state_evaluator
    src/marker_state_evaluator.cpp
    src/frame_pacer.cpp
    src/frame_source.cpp
)

if(VIDEO_INPUT_MODE)
    target_compile_definitions(eye PRIVATE VIDEO_INPUT_MODE)
    target_compile_definitions(marker_state_evaluator PRIVATE VIDEO_INPUT_MODE)
    # Link ED_Lib, OpenCV, and Eigen to your project Eigen3::Eigen, EDLib
    target_link_libraries(eye ${OpenCV_LIBS} nlohmann_json::nlohmann_json Eigen3::Eigen)
    target_link_libraries(marker_state_evaluator ${OpenCV_LIBS})
else()
    # Link ED_Lib, OpenCV, and Eigen to your project Eigen3::Eigen, EDLib
    target_link_libraries(eye ${OpenCV_LIBS} LibCamera ${LIBCAMERA_LIBRARIES} nlohmann_json::nlohmann_json Eigen3::Eigen)
    target_link_libraries(marker_state_evaluator ${OpenCV_LIBS} LibCamera ${LIBCAMERA_LIBRARIES})
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(localization_pipeline_tests
        tests/localization_pipeline_test.cpp
        src/localization_pipeline.cpp
        src/pose_estimator.cpp
        src/pose_math.cpp
    )
    target_include_directories(localization_pipeline_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(localization_pipeline_tests PRIVATE
        TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    target_link_libraries(localization_pipeline_tests PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
    )
    add_test(NAME localization_pipeline_tests COMMAND localization_pipeline_tests)

    add_executable(marker_tracker_tests
        tests/marker_tracker_test.cpp
        src/marker_tracker.cpp
    )
    target_include_directories(marker_tracker_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(marker_tracker_tests PRIVATE ${OpenCV_LIBS})
    add_test(NAME marker_tracker_tests COMMAND marker_tracker_tests)
    add_test(NAME marker_state_evaluator_self_test
        COMMAND marker_state_evaluator --self-test)

    add_executable(pose_publisher_tests
        tests/pose_publisher_test.cpp
        src/pose_publisher.cpp
        src/pose_math.cpp
    )
    target_include_directories(pose_publisher_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(pose_publisher_tests PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
    )
    add_test(NAME pose_publisher_tests COMMAND pose_publisher_tests)
    set_tests_properties(pose_publisher_tests PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME localization_cli_rejects_invalid_window
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--window-size
            -DCLI_VALUE=2junk
            "-DEXPECTED_PATTERN=--window-size requires an integer"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_invalid_marker_age
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--grid-max-marker-age
            -DCLI_VALUE=1junk
            "-DEXPECTED_PATTERN=--grid-max-marker-age requires a number"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_invalid_attitude_age
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--max-attitude-age
            -DCLI_VALUE=0
            "-DEXPECTED_PATTERN=--max-attitude-age must be a positive finite number"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_removed_attitude_flag
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--use-shared-attitude
            "-DEXPECTED_PATTERN=unknown option: --use-shared-attitude"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_invalid_dark_blob_intensity
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--dark-blob-intensity
            -DCLI_VALUE=0.8
            "-DEXPECTED_PATTERN=--dark-blob-intensity must be in the range"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_missing_map_value
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--grid-map
            "-DEXPECTED_PATTERN=--grid-map requires a value"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_center_ap3p_without_map
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--grid-center-ap3p
            "-DEXPECTED_PATTERN=--grid-center-ap3p requires --grid-map"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
    add_test(NAME localization_cli_rejects_invalid_payload_size
        COMMAND ${CMAKE_COMMAND}
            -DEYE_EXECUTABLE=$<TARGET_FILE:eye>
            -DCLI_OPTION=--payload-size
            -DCLI_VALUE=-1
            "-DEXPECTED_PATTERN=--payload-size must be between 1 and 16 bits"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_rejected.cmake)
endif()
