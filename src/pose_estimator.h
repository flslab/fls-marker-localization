#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace pose_estimation {

struct PnpEstimate {
  bool valid = false;
  std::string message;
  // OpenCV PnP convention: object coordinates transformed into camera space.
  cv::Mat object_to_camera_rvec;
  cv::Mat object_to_camera_rotation;
  cv::Mat object_to_camera_translation;
  // The inverse pose, exposed explicitly so callers do not repeat the math.
  cv::Mat camera_to_object_rotation;
  cv::Mat camera_position_object;
  double mean_reprojection_error = -1.0;
  double rms_reprojection_error = -1.0;
  int inlier_count = 0;
};

struct PlanarPosePrior {
  double distance = -1.0;
  cv::Mat expected_object_to_camera_rotation;
};

PnpEstimate solveAp3p(const std::vector<cv::Point3f> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const cv::Mat &camera_matrix,
                      const cv::Mat &distortion_coefficients);
PnpEstimate solveAp3p(const std::vector<cv::Point3f> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const cv::Mat &camera_matrix,
                      const cv::Mat &distortion_coefficients,
                      const PlanarPosePrior &prior);

PnpEstimate solveAp3pRansac(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients);

PnpEstimate solvePlanarIppe(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
    const PlanarPosePrior &prior);

} // namespace pose_estimation
