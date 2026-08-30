#pragma once

#include <opencv2/core.hpp>
#include <optional>

namespace pose_math {

std::optional<cv::Vec4d>
normalizeQuaternionXyzw(const cv::Vec4d &quaternion_xyzw);
std::optional<cv::Matx33d>
rotationFromQuaternionXyzw(const cv::Vec4d &quaternion_xyzw);
const cv::Matx33d &cameraToDroneRotation();
std::optional<cv::Matx33d>
gridToCameraRotationFromDroneQuaternionXyzw(
    const cv::Vec4d &quaternion_xyzw);

cv::Mat rotationFromRpyRadians(const cv::Vec3d &rpy_radians);
cv::Mat rotationFromRpyDegrees(const cv::Vec3d &rpy_degrees);
cv::Vec3d rpyFromRotation(const cv::Mat &rotation);
cv::Mat makeTransform(const cv::Mat &rotation, const cv::Mat &translation);
cv::Mat invertTransform(const cv::Mat &transform);

} // namespace pose_math
