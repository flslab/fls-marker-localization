#pragma once

#include "fls_localizer/config.hpp"
#include "fls_localizer/types.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace flsloc {

std::optional<cv::Vec4d> normalizeQuaternion(const cv::Vec4d &xyzw);
std::optional<cv::Matx33d> rotationFromQuaternion(const cv::Vec4d &xyzw);
cv::Vec4d quaternionFromRotation(const cv::Matx33d &rotation);
cv::Vec3d rpyFromRotation(const cv::Matx33d &rotation);

class PoseSolver {
public:
  explicit PoseSolver(const ApplicationConfig &config);

  PoseSolution solveInitial(const std::vector<MatchedPoint> &matches,
                            double expected_distance) const;
  PoseSolution solveWithPnp(const std::vector<MatchedPoint> &matches,
                            double expected_distance = -1.0) const;
  PoseSolution solveWithAttitude(const std::vector<MatchedPoint> &matches,
                                 const cv::Vec4d &drone_quaternion_xyzw) const;

  cv::Matx33d
  worldToCameraFromDrone(const cv::Vec4d &drone_quaternion_xyzw) const;
  const cv::Matx33d &cameraToDroneRotation() const { return camera_to_drone_; }

private:
  struct IppeCandidate {
    bool valid = false;
    cv::Vec3d rvec{};
    cv::Vec3d tvec{};
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Vec3d camera_position{};
    double reprojection_rms = 0.0;
    double cost = 0.0;
  };

  std::optional<IppeCandidate>
  selectIppe(const std::vector<cv::Point3f> &object_points,
             const std::vector<cv::Point2f> &image_points,
             double expected_distance) const;
  bool translationForRotation(const std::vector<cv::Point3f> &object_points,
                              const std::vector<cv::Point2f> &image_points,
                              const cv::Matx33d &rotation,
                              cv::Vec3d &translation) const;
  double reprojectionRms(const std::vector<cv::Point3f> &object_points,
                         const std::vector<cv::Point2f> &image_points,
                         const cv::Matx33d &rotation,
                         const cv::Vec3d &translation) const;
  PoseSolution makeSolution(const cv::Matx33d &world_to_camera,
                            const cv::Vec3d &translation,
                            const cv::Matx33d &drone_to_world,
                            std::string solver, double reprojection_rms) const;

  cv::Mat camera_matrix_;
  cv::Mat distortion_;
  cv::Matx33d camera_to_drone_;
  cv::Vec3d camera_position_drone_;
};

} // namespace flsloc
