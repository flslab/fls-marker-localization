#include "camera_config.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "pose_math.h"

using json = nlohmann::json;

bool readConfigFile(const std::string &filename, cv::Mat &camera_matrix,
                    cv::Mat &distortion_coefficients,
                    std::vector<cv::Point3f> &marker_points) {
  try {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open config file: " << filename << std::endl;
      return false;
    }
    const json config = json::parse(file);

    camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        camera_matrix.at<double>(row, column) =
            config["/camera_matrix"_json_pointer][row][column];
      }
    }

    const std::vector<double> distortion = config["dist_coeffs"];
    distortion_coefficients = cv::Mat(distortion, true);

    marker_points.clear();
    if (config.contains("marker_points")) {
      for (const auto &point : config["marker_points"]) {
        marker_points.emplace_back(point[0], point[1], point[2]);
      }
    }
    return true;
  } catch (const std::exception &error) {
    std::cerr << "Error reading config file: " << error.what() << std::endl;
    return false;
  }
}

bool readArucoConfig(
    const std::string &filename, std::string &dictionary, double &marker_size,
    std::map<int, ArucoTracker::MarkerWorldPose> &known_markers) {
  try {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open config file: " << filename << std::endl;
      return false;
    }
    const json config = json::parse(file);
    if (!config.contains("aruco_markers")) {
      std::cerr << "Config file does not contain 'aruco_markers' section"
                << std::endl;
      return false;
    }

    const auto &aruco = config["aruco_markers"];
    dictionary = aruco.value("dictionary", "DICT_4X4_50");
    marker_size = aruco.value("marker_size", 0.20);
    known_markers.clear();

    if (aruco.contains("markers")) {
      for (auto it = aruco["markers"].begin(); it != aruco["markers"].end();
           ++it) {
        const int marker_id = std::stoi(it.key());
        const auto &marker = it.value();
        const double x = marker["position"][0].get<double>();
        const double y = marker["position"][1].get<double>();
        const double z = marker["position"][2].get<double>();
        const cv::Mat rotation = pose_math::rotationFromRpyDegrees(
            {marker["rotation_deg"][0].get<double>(),
             marker["rotation_deg"][1].get<double>(),
             marker["rotation_deg"][2].get<double>()});
        const cv::Mat translation = (cv::Mat_<double>(3, 1) << x, y, z);

        known_markers[marker_id] = {
            marker_id, pose_math::makeTransform(rotation, translation)};
        std::cout << "  ArUco marker " << marker_id << " at [" << x << ", " << y
                  << ", " << z << "]" << std::endl;
      }
    }

    std::cout << "ArUco config: dictionary=" << dictionary
              << ", marker_size=" << marker_size << "m, "
              << known_markers.size() << " known markers" << std::endl;
    return true;
  } catch (const std::exception &error) {
    std::cerr << "Error reading ArUco config: " << error.what() << std::endl;
    return false;
  }
}
