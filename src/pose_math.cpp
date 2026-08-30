#include "pose_math.h"

#include <algorithm>
#include <cmath>

namespace pose_math {

std::optional<cv::Vec4d>
normalizeQuaternionXyzw(const cv::Vec4d &quaternion_xyzw) {
  double scale = 0.0;
  for (double component : quaternion_xyzw.val) {
    if (!std::isfinite(component)) {
      return std::nullopt;
    }
    scale = std::max(scale, std::abs(component));
  }
  if (scale == 0.0) {
    return std::nullopt;
  }

  const cv::Vec4d scaled = quaternion_xyzw * (1.0 / scale);
  const double scaled_norm = std::sqrt(scaled.dot(scaled));
  if (!std::isfinite(scaled_norm) || scaled_norm == 0.0) {
    return std::nullopt;
  }
  return scaled * (1.0 / scaled_norm);
}

std::optional<cv::Matx33d>
rotationFromQuaternionXyzw(const cv::Vec4d &quaternion_xyzw) {
  const auto normalized = normalizeQuaternionXyzw(quaternion_xyzw);
  if (!normalized) {
    return std::nullopt;
  }

  const double x = (*normalized)[0];
  const double y = (*normalized)[1];
  const double z = (*normalized)[2];
  const double w = (*normalized)[3];
  return cv::Matx33d(
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
      2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
      1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
      1.0 - 2.0 * (x * x + y * y));
}

const cv::Matx33d &cameraToDroneRotation() {
  // OpenCV camera (+X right, +Y down, +Z forward) to drone/world FLU
  // (+X forward, +Y left, +Z up) for the downward-facing camera mount.
  static const cv::Matx33d rotation(0.0, -1.0, 0.0, -1.0, 0.0, 0.0,
                                    0.0, 0.0, -1.0);
  return rotation;
}

std::optional<cv::Matx33d>
gridToCameraRotationFromDroneQuaternionXyzw(
    const cv::Vec4d &quaternion_xyzw) {
  const auto drone_to_world = rotationFromQuaternionXyzw(quaternion_xyzw);
  if (!drone_to_world) {
    return std::nullopt;
  }
  // Grid and world align. q maps drone to world, while PnP and the grid
  // homography need grid/world to camera.
  return cameraToDroneRotation().t() * drone_to_world->t();
}

cv::Mat rotationFromRpyRadians(const cv::Vec3d &rpy_radians) {
  const double roll = rpy_radians[0];
  const double pitch = rpy_radians[1];
  const double yaw = rpy_radians[2];
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  return cv::Mat(cv::Matx33d(
      cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp,
      sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr,
      cp * cr));
}

cv::Mat rotationFromRpyDegrees(const cv::Vec3d &rpy_degrees) {
  return rotationFromRpyRadians(rpy_degrees * (CV_PI / 180.0));
}

cv::Vec3d rpyFromRotation(const cv::Mat &rotation) {
  cv::Mat rotation_64f;
  rotation.convertTo(rotation_64f, CV_64F);
  const double sy = std::hypot(rotation_64f.at<double>(0, 0),
                               rotation_64f.at<double>(1, 0));
  if (sy < 1e-9) {
    return {std::atan2(-rotation_64f.at<double>(1, 2),
                       rotation_64f.at<double>(1, 1)),
            std::atan2(-rotation_64f.at<double>(2, 0), sy), 0.0};
  }
  return {std::atan2(rotation_64f.at<double>(2, 1),
                     rotation_64f.at<double>(2, 2)),
          std::atan2(-rotation_64f.at<double>(2, 0), sy),
          std::atan2(rotation_64f.at<double>(1, 0),
                     rotation_64f.at<double>(0, 0))};
}

cv::Mat makeTransform(const cv::Mat &rotation, const cv::Mat &translation) {
  cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat rotation_64f;
  cv::Mat translation_64f;
  rotation.convertTo(rotation_64f, CV_64F);
  translation.reshape(1, 3).convertTo(translation_64f, CV_64F);
  rotation_64f.copyTo(transform(cv::Rect(0, 0, 3, 3)));
  translation_64f.copyTo(transform(cv::Rect(3, 0, 1, 3)));
  return transform;
}

cv::Mat invertTransform(const cv::Mat &transform) {
  const cv::Mat rotation = transform(cv::Rect(0, 0, 3, 3));
  const cv::Mat translation = transform(cv::Rect(3, 0, 1, 3));
  const cv::Mat inverse_rotation = rotation.t();
  const cv::Mat inverse_translation = -inverse_rotation * translation;
  return makeTransform(inverse_rotation, inverse_translation);
}

} // namespace pose_math
