#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "aruco_tracker.h"

bool readConfigFile(const std::string &filename, cv::Mat &camera_matrix,
                    cv::Mat &distortion_coefficients,
                    std::vector<cv::Point3f> &marker_points);

bool readArucoConfig(
    const std::string &filename, std::string &dictionary, double &marker_size,
    std::map<int, ArucoTracker::MarkerWorldPose> &known_markers);
