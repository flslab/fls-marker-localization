#include "fls_localizer/pose_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

namespace flsloc {
namespace {

cv::Matx33d matx(const cv::Mat &value) {
  cv::Mat converted;
  value.convertTo(converted, CV_64F);
  cv::Matx33d result;
  std::copy(converted.ptr<double>(), converted.ptr<double>() + 9, result.val);
  return result;
}

void correspondences(const std::vector<MatchedPoint> &matches,
                     std::vector<cv::Point3f> &object_points,
                     std::vector<cv::Point2f> &image_points) {
  object_points.clear();
  image_points.clear();
  object_points.reserve(matches.size());
  image_points.reserve(matches.size());
  for (const MatchedPoint &match : matches) {
    object_points.push_back(match.world);
    image_points.push_back(match.image);
  }
}

} // namespace

std::optional<cv::Vec4d> normalizeQuaternion(const cv::Vec4d &xyzw) {
  double norm_squared = 0.0;
  for (double value : xyzw.val) {
    if (!std::isfinite(value)) {
      return std::nullopt;
    }
    norm_squared += value * value;
  }
  if (norm_squared < 1e-18) {
    return std::nullopt;
  }
  return xyzw * (1.0 / std::sqrt(norm_squared));
}

std::optional<cv::Matx33d> rotationFromQuaternion(const cv::Vec4d &xyzw) {
  const auto q = normalizeQuaternion(xyzw);
  if (!q) {
    return std::nullopt;
  }
  const double x = (*q)[0];
  const double y = (*q)[1];
  const double z = (*q)[2];
  const double w = (*q)[3];
  return cv::Matx33d(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
                     2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
                     1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                     2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
                     1.0 - 2.0 * (x * x + y * y));
}

cv::Vec4d quaternionFromRotation(const cv::Matx33d &rotation) {
  cv::Vec4d quaternion{};
  const double trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    quaternion[3] = 0.25 * scale;
    quaternion[0] = (rotation(2, 1) - rotation(1, 2)) / scale;
    quaternion[1] = (rotation(0, 2) - rotation(2, 0)) / scale;
    quaternion[2] = (rotation(1, 0) - rotation(0, 1)) / scale;
  } else if (rotation(0, 0) > rotation(1, 1) &&
             rotation(0, 0) > rotation(2, 2)) {
    const double scale =
        std::sqrt(1.0 + rotation(0, 0) - rotation(1, 1) - rotation(2, 2)) * 2.0;
    quaternion[3] = (rotation(2, 1) - rotation(1, 2)) / scale;
    quaternion[0] = 0.25 * scale;
    quaternion[1] = (rotation(0, 1) + rotation(1, 0)) / scale;
    quaternion[2] = (rotation(0, 2) + rotation(2, 0)) / scale;
  } else if (rotation(1, 1) > rotation(2, 2)) {
    const double scale =
        std::sqrt(1.0 + rotation(1, 1) - rotation(0, 0) - rotation(2, 2)) * 2.0;
    quaternion[3] = (rotation(0, 2) - rotation(2, 0)) / scale;
    quaternion[0] = (rotation(0, 1) + rotation(1, 0)) / scale;
    quaternion[1] = 0.25 * scale;
    quaternion[2] = (rotation(1, 2) + rotation(2, 1)) / scale;
  } else {
    const double scale =
        std::sqrt(1.0 + rotation(2, 2) - rotation(0, 0) - rotation(1, 1)) * 2.0;
    quaternion[3] = (rotation(1, 0) - rotation(0, 1)) / scale;
    quaternion[0] = (rotation(0, 2) + rotation(2, 0)) / scale;
    quaternion[1] = (rotation(1, 2) + rotation(2, 1)) / scale;
    quaternion[2] = 0.25 * scale;
  }
  return *normalizeQuaternion(quaternion);
}

cv::Vec3d rpyFromRotation(const cv::Matx33d &rotation) {
  const double sy = std::hypot(rotation(0, 0), rotation(1, 0));
  if (sy < 1e-9) {
    return {std::atan2(-rotation(1, 2), rotation(1, 1)),
            std::atan2(-rotation(2, 0), sy), 0.0};
  }
  return {std::atan2(rotation(2, 1), rotation(2, 2)),
          std::atan2(-rotation(2, 0), sy),
          std::atan2(rotation(1, 0), rotation(0, 0))};
}

PoseSolver::PoseSolver(const ApplicationConfig &config)
    : camera_matrix_(config.calibration.camera_matrix.clone()),
      distortion_(config.calibration.distortion.clone()),
      camera_to_drone_(config.camera_to_drone_rotation),
      camera_position_drone_(config.camera_position_drone_flu) {}

cv::Matx33d PoseSolver::worldToCameraFromDrone(
    const cv::Vec4d &drone_quaternion_xyzw) const {
  const auto drone_to_world = rotationFromQuaternion(drone_quaternion_xyzw);
  return camera_to_drone_.t() * drone_to_world->t();
}

std::optional<PoseSolver::IppeCandidate>
PoseSolver::selectIppe(const std::vector<cv::Point3f> &object_points,
                       const std::vector<cv::Point2f> &image_points,
                       double expected_distance) const {
  if (object_points.size() < 4 || object_points.size() != image_points.size()) {
    return std::nullopt;
  }
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  const int count =
      cv::solvePnPGeneric(object_points, image_points, camera_matrix_,
                          distortion_, rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
  if (count <= 0) {
    return std::nullopt;
  }

  std::optional<IppeCandidate> best;
  for (int index = 0; index < count; ++index) {
    cv::Mat refined_rvec;
    cv::Mat refined_tvec;
    rvecs[index].convertTo(refined_rvec, CV_64F);
    tvecs[index].convertTo(refined_tvec, CV_64F);
    cv::solvePnPRefineLM(object_points, image_points, camera_matrix_,
                         distortion_, refined_rvec, refined_tvec);

    cv::Mat rotation_mat;
    cv::Rodrigues(refined_rvec, rotation_mat);
    const cv::Matx33d rotation = matx(rotation_mat);
    const cv::Vec3d translation{refined_tvec.at<double>(0),
                                refined_tvec.at<double>(1),
                                refined_tvec.at<double>(2)};
    const cv::Vec3d camera_position = -(rotation.t() * translation);
    if (!std::isfinite(camera_position[2]) ||
        camera_position[2] <= object_points.front().z) {
      continue;
    }
    bool positive_depth = true;
    for (const cv::Point3f &point : object_points) {
      const cv::Vec3d camera =
          rotation * cv::Vec3d(point.x, point.y, point.z) + translation;
      positive_depth = positive_depth && camera[2] > 1e-6;
    }
    if (!positive_depth) {
      continue;
    }
    IppeCandidate candidate;
    candidate.valid = true;
    candidate.rotation = rotation;
    candidate.camera_position = camera_position;
    candidate.tvec = translation;
    candidate.rvec = {refined_rvec.at<double>(0),
                      refined_rvec.at<double>(1),
                      refined_rvec.at<double>(2)};
    candidate.reprojection_rms =
        reprojectionRms(object_points, image_points, rotation, translation);
    candidate.cost = candidate.reprojection_rms;
    if (expected_distance > 0.0) {
      candidate.cost +=
          10.0 *
          std::abs((camera_position[2] - object_points.front().z) -
                   expected_distance) /
          expected_distance;
    }
    if (!best || candidate.cost < best->cost) {
      best = candidate;
    }
  }
  return best;
}

bool PoseSolver::translationForRotation(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points, const cv::Matx33d &rotation,
    cv::Vec3d &translation) const {
  if (object_points.size() < 2 || object_points.size() != image_points.size()) {
    return false;
  }
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(image_points, normalized, camera_matrix_, distortion_);
  cv::Mat coefficients(static_cast<int>(object_points.size() * 2), 3, CV_64F);
  cv::Mat values(static_cast<int>(object_points.size() * 2), 1, CV_64F);
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const cv::Point3f &point = object_points[index];
    const cv::Vec3d rotated = rotation * cv::Vec3d(point.x, point.y, point.z);
    const double u = normalized[index].x;
    const double v = normalized[index].y;
    const int row = static_cast<int>(index * 2);
    coefficients.at<double>(row, 0) = 1.0;
    coefficients.at<double>(row, 1) = 0.0;
    coefficients.at<double>(row, 2) = -u;
    values.at<double>(row) = u * rotated[2] - rotated[0];
    coefficients.at<double>(row + 1, 0) = 0.0;
    coefficients.at<double>(row + 1, 1) = 1.0;
    coefficients.at<double>(row + 1, 2) = -v;
    values.at<double>(row + 1) = v * rotated[2] - rotated[1];
  }
  cv::Mat solved;
  if (!cv::solve(coefficients, values, solved, cv::DECOMP_SVD)) {
    return false;
  }
  translation = {solved.at<double>(0), solved.at<double>(1),
                 solved.at<double>(2)};
  return cv::checkRange(solved);
}

double
PoseSolver::reprojectionRms(const std::vector<cv::Point3f> &object_points,
                            const std::vector<cv::Point2f> &image_points,
                            const cv::Matx33d &rotation,
                            const cv::Vec3d &translation) const {
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, translation, camera_matrix_,
                    distortion_, projected);
  double sum = 0.0;
  for (std::size_t index = 0; index < projected.size(); ++index) {
    const cv::Point2f delta = projected[index] - image_points[index];
    sum += delta.dot(delta);
  }
  return projected.empty()
             ? std::numeric_limits<double>::infinity()
             : std::sqrt(sum / static_cast<double>(projected.size()));
}

PoseSolution PoseSolver::makeSolution(const cv::Matx33d &world_to_camera,
                                      const cv::Vec3d &translation,
                                      const cv::Matx33d &drone_to_world,
                                      std::string solver,
                                      double reprojection_rms) const {
  PoseSolution result;
  result.solver = std::move(solver);
  result.tvec_world_to_camera = translation;
  result.world_to_camera_rotation = world_to_camera;
  result.marker_rpy_camera = rpyFromRotation(world_to_camera);
  result.camera_position_world = -(world_to_camera.t() * translation);
  result.drone_position_world =
      result.camera_position_world - drone_to_world * camera_position_drone_;
  const cv::Matx33d camera_to_world = world_to_camera.t();
  result.camera_rpy = rpyFromRotation(camera_to_world);
  result.drone_rpy = rpyFromRotation(drone_to_world);
  result.drone_quaternion_xyzw = quaternionFromRotation(drone_to_world);
  result.camera_to_plane_distance = result.camera_position_world[2];
  result.reprojection_rms = reprojection_rms;
  result.valid = std::isfinite(reprojection_rms) &&
                 cv::checkRange(cv::Mat(result.camera_position_world)) &&
                 result.camera_to_plane_distance > 0.0;
  return result;
}

PoseSolution PoseSolver::solveInitial(const std::vector<MatchedPoint> &matches,
                                      double expected_distance) const {
  return solveWithPnp(matches, expected_distance);
}

PoseSolution
PoseSolver::solveWithPnp(const std::vector<MatchedPoint> &matches,
                         double expected_distance) const {
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  correspondences(matches, object_points, image_points);
  if (object_points.size() < 4) {
    return {};
  }
  const auto ippe =
      selectIppe(object_points, image_points, expected_distance);
  if (ippe) {
    const cv::Matx33d camera_to_world = ippe->rotation.t();
    const cv::Matx33d drone_to_world = camera_to_world * camera_to_drone_.t();
    return makeSolution(ippe->rotation, ippe->tvec, drone_to_world,
                        "ippe_refined_lm", ippe->reprojection_rms);
  }

  // IPPE is singular for a perfectly fronto-parallel plane. Preserve the
  // IPPE path above, but recover that exact geometry with OpenCV's planar
  // homography initialization instead of losing the initial pose.
  cv::Vec3d rvec;
  cv::Vec3d translation;
  if (!cv::solvePnP(object_points, image_points, camera_matrix_, distortion_,
                    rvec, translation, false, cv::SOLVEPNP_ITERATIVE)) {
    return {};
  }
  cv::Matx33d rotation;
  cv::Rodrigues(rvec, rotation);
  const cv::Vec3d camera_position = -(rotation.t() * translation);
  if (camera_position[2] <= object_points.front().z) {
    return {};
  }
  const cv::Matx33d camera_to_world = rotation.t();
  const cv::Matx33d drone_to_world = camera_to_world * camera_to_drone_.t();
  return makeSolution(
      rotation, translation, drone_to_world, "ippe_iterative_degeneracy",
      reprojectionRms(object_points, image_points, rotation, translation));
}

PoseSolution
PoseSolver::solveWithAttitude(const std::vector<MatchedPoint> &matches,
                              const cv::Vec4d &drone_quaternion_xyzw) const {
  const auto drone_to_world = rotationFromQuaternion(drone_quaternion_xyzw);
  if (!drone_to_world) {
    return {};
  }
  const cv::Matx33d expected_rotation =
      camera_to_drone_.t() * drone_to_world->t();
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  correspondences(matches, object_points, image_points);

  cv::Vec3d translation;
  if (!translationForRotation(object_points, image_points, expected_rotation,
                              translation)) {
    return {};
  }
  const double error = reprojectionRms(object_points, image_points,
                                       expected_rotation, translation);
  return makeSolution(expected_rotation, translation, *drone_to_world,
                      "known_rotation", error);
}

} // namespace flsloc
