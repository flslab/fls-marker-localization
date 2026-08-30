#include "pose_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <utility>

namespace pose_estimation {
namespace {

bool finite(double value) { return std::isfinite(value); }

bool validCorrespondences(const std::vector<cv::Point3f> &object_points,
                          const std::vector<cv::Point2f> &image_points,
                          std::size_t minimum_count) {
  return object_points.size() == image_points.size() &&
         object_points.size() >= minimum_count;
}

PnpEstimate makeEstimate(const std::vector<cv::Point3f> &object_points,
                         const std::vector<cv::Point2f> &image_points,
                         const cv::Mat &camera_matrix,
                         const cv::Mat &distortion_coefficients,
                         const cv::Mat &rvec, const cv::Mat &tvec) {
  PnpEstimate result;
  if (rvec.empty() || tvec.empty()) {
    result.message = "PnP returned an empty pose";
    return result;
  }

  rvec.convertTo(result.object_to_camera_rvec, CV_64F);
  tvec.convertTo(result.object_to_camera_translation, CV_64F);
  if (!cv::checkRange(result.object_to_camera_rvec) ||
      !cv::checkRange(result.object_to_camera_translation)) {
    result.message = "PnP returned a non-finite pose";
    return result;
  }

  cv::Rodrigues(result.object_to_camera_rvec,
                result.object_to_camera_rotation);
  result.object_to_camera_rotation.convertTo(
      result.object_to_camera_rotation, CV_64F);
  result.camera_to_object_rotation = result.object_to_camera_rotation.t();
  result.camera_position_object =
      -result.camera_to_object_rotation * result.object_to_camera_translation;

  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, result.object_to_camera_rvec,
                    result.object_to_camera_translation, camera_matrix,
                    distortion_coefficients, projected);
  if (projected.size() != image_points.size() || projected.empty()) {
    result.message = "PnP reprojection returned no points";
    return result;
  }

  double distance_sum = 0.0;
  double squared_distance_sum = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i) {
    const cv::Point2f delta = projected[i] - image_points[i];
    const double squared_distance = delta.dot(delta);
    distance_sum += std::sqrt(squared_distance);
    squared_distance_sum += squared_distance;
  }
  result.mean_reprojection_error =
      distance_sum / static_cast<double>(projected.size());
  result.rms_reprojection_error =
      std::sqrt(squared_distance_sum / static_cast<double>(projected.size()));
  result.valid = cv::checkRange(result.camera_position_object) &&
                 finite(result.mean_reprojection_error) &&
                 finite(result.rms_reprojection_error);
  result.message = result.valid ? "pose solved" : "PnP returned invalid values";
  return result;
}

double rotationDifferenceRadians(const cv::Mat &actual,
                                 const cv::Mat &expected) {
  cv::Mat expected_64f;
  expected.convertTo(expected_64f, CV_64F);
  const cv::Mat relative = actual * expected_64f.t();
  const double cosine =
      std::clamp((cv::trace(relative)[0] - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine);
}

struct PlanarCandidate {
  PnpEstimate pose;
  double geometry_error = std::numeric_limits<double>::infinity();
};

bool evaluatePlanarCandidate(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
    const PlanarPosePrior &prior, const cv::Mat &rvec, const cv::Mat &tvec,
    PlanarCandidate &candidate) {
  candidate.pose = makeEstimate(object_points, image_points, camera_matrix,
                                distortion_coefficients, rvec, tvec);
  if (!candidate.pose.valid) {
    return false;
  }

  for (const auto &point : object_points) {
    const cv::Mat &rotation = candidate.pose.object_to_camera_rotation;
    const cv::Mat &translation = candidate.pose.object_to_camera_translation;
    const double depth = rotation.at<double>(2, 0) * point.x +
                         rotation.at<double>(2, 1) * point.y +
                         rotation.at<double>(2, 2) * point.z +
                         translation.at<double>(2, 0);
    if (!finite(depth) || depth <= 1e-6) {
      return false;
    }
  }

  candidate.geometry_error = rotationDifferenceRadians(
      candidate.pose.object_to_camera_rotation,
      prior.expected_object_to_camera_rotation);
  if (prior.distance > 0.0) {
    const double plane_z = object_points.front().z;
    const double estimated_distance = std::abs(
        candidate.pose.camera_position_object.at<double>(2, 0) - plane_z);
    candidate.geometry_error +=
        std::abs(estimated_distance - prior.distance) / prior.distance;
  }
  return finite(candidate.geometry_error);
}

bool translationForKnownRotation(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
    const cv::Mat &object_to_camera_rotation, cv::Mat &translation) {
  std::vector<cv::Point2f> normalized_points;
  cv::undistortPoints(image_points, normalized_points, camera_matrix,
                      distortion_coefficients);
  if (normalized_points.size() != object_points.size()) {
    return false;
  }

  cv::Mat rotation_64f;
  object_to_camera_rotation.convertTo(rotation_64f, CV_64F);
  cv::Mat coefficients = cv::Mat::zeros(
      static_cast<int>(object_points.size() * 2), 3, CV_64F);
  cv::Mat values = cv::Mat::zeros(
      static_cast<int>(object_points.size() * 2), 1, CV_64F);
  for (std::size_t i = 0; i < object_points.size(); ++i) {
    const cv::Point3f &point = object_points[i];
    const cv::Mat object =
        (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
    const cv::Mat rotated = rotation_64f * object;
    const double u = normalized_points[i].x;
    const double v = normalized_points[i].y;
    const int row = static_cast<int>(i * 2);
    coefficients.at<double>(row, 0) = 1.0;
    coefficients.at<double>(row, 2) = -u;
    values.at<double>(row, 0) =
        u * rotated.at<double>(2, 0) - rotated.at<double>(0, 0);
    coefficients.at<double>(row + 1, 1) = 1.0;
    coefficients.at<double>(row + 1, 2) = -v;
    values.at<double>(row + 1, 0) =
        v * rotated.at<double>(2, 0) - rotated.at<double>(1, 0);
  }
  return cv::solve(coefficients, values, translation, cv::DECOMP_SVD) &&
         cv::checkRange(translation);
}

} // namespace

PnpEstimate solveAp3p(const std::vector<cv::Point3f> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const cv::Mat &camera_matrix,
                      const cv::Mat &distortion_coefficients) {
  if (!validCorrespondences(object_points, image_points, 4) ||
      object_points.size() != 4) {
    return {false, "AP3P requires exactly four point correspondences"};
  }
  try {
    cv::Mat rvec;
    cv::Mat tvec;
    if (!cv::solvePnP(object_points, image_points, camera_matrix,
                      distortion_coefficients, rvec, tvec, false,
                      cv::SOLVEPNP_AP3P)) {
      return {false, "OpenCV AP3P could not solve the pose"};
    }
    return makeEstimate(object_points, image_points, camera_matrix,
                        distortion_coefficients, rvec, tvec);
  } catch (const cv::Exception &error) {
    return {false, "AP3P failed: " + std::string(error.what())};
  }
}

PnpEstimate solveAp3p(const std::vector<cv::Point3f> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const cv::Mat &camera_matrix,
                      const cv::Mat &distortion_coefficients,
                      const PlanarPosePrior &prior) {
  if (!validCorrespondences(object_points, image_points, 4) ||
      object_points.size() != 4) {
    return {false, "AP3P requires exactly four point correspondences"};
  }
  if (prior.expected_object_to_camera_rotation.rows != 3 ||
      prior.expected_object_to_camera_rotation.cols != 3) {
    return {false, "AP3P prior requires a 3x3 expected rotation"};
  }

  try {
    std::vector<cv::Mat> candidate_rvecs;
    std::vector<cv::Mat> candidate_tvecs;
    const int solution_count = cv::solvePnPGeneric(
        object_points, image_points, camera_matrix, distortion_coefficients,
        candidate_rvecs, candidate_tvecs, false, cv::SOLVEPNP_AP3P);
    if (solution_count <= 0 ||
        candidate_rvecs.size() != candidate_tvecs.size()) {
      return {false, "OpenCV AP3P returned no pose candidates"};
    }

    std::vector<PlanarCandidate> valid_candidates;
    for (std::size_t i = 0; i < candidate_rvecs.size(); ++i) {
      PlanarCandidate candidate;
      if (evaluatePlanarCandidate(
              object_points, image_points, camera_matrix,
              distortion_coefficients, prior, candidate_rvecs[i],
              candidate_tvecs[i], candidate)) {
        valid_candidates.push_back(std::move(candidate));
      }
    }

    if (valid_candidates.empty()) {
      return {false, "AP3P returned no physically valid pose"};
    }

    const auto best = std::min_element(
        valid_candidates.begin(), valid_candidates.end(),
        [](const PlanarCandidate &left, const PlanarCandidate &right) {
          if (std::abs(left.geometry_error - right.geometry_error) > 1e-9) {
            return left.geometry_error < right.geometry_error;
          }
          return left.pose.rms_reprojection_error <
                 right.pose.rms_reprojection_error;
        });
    return best->pose;
  } catch (const cv::Exception &error) {
    return {false, "AP3P failed: " + std::string(error.what())};
  }
}

PnpEstimate solveAp3pRansac(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients) {
  if (!validCorrespondences(object_points, image_points, 4)) {
    return {false, "AP3P RANSAC requires at least four correspondences"};
  }
  try {
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    if (!cv::solvePnPRansac(object_points, image_points, camera_matrix,
                            distortion_coefficients, rvec, tvec, false, 100,
                            4.0F, 0.99, inliers, cv::SOLVEPNP_AP3P) ||
        inliers.total() < 4) {
      return {false, "OpenCV AP3P RANSAC could not solve the pose"};
    }
    PnpEstimate result = makeEstimate(
        object_points, image_points, camera_matrix, distortion_coefficients,
        rvec, tvec);
    result.inlier_count = static_cast<int>(inliers.total());
    return result;
  } catch (const cv::Exception &error) {
    return {false, "AP3P RANSAC failed: " + std::string(error.what())};
  }
}

PnpEstimate solvePlanarIppe(
    const std::vector<cv::Point3f> &object_points,
    const std::vector<cv::Point2f> &image_points,
    const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
    const PlanarPosePrior &prior) {
  if (!validCorrespondences(object_points, image_points, 4)) {
    return {false, "planar IPPE requires at least four correspondences"};
  }
  if (prior.expected_object_to_camera_rotation.rows != 3 ||
      prior.expected_object_to_camera_rotation.cols != 3) {
    return {false, "planar IPPE requires a 3x3 expected rotation"};
  }

  try {
    const float plane_z = object_points.front().z;
    std::vector<cv::Point3f> ippe_object_points;
    ippe_object_points.reserve(object_points.size());
    for (const cv::Point3f &point : object_points) {
      if (std::abs(point.z - plane_z) > 1e-5F) {
        return {false, "planar IPPE object points are not coplanar"};
      }
      // OpenCV's IPPE implementation expects the object plane at Z=0.
      // Convert each returned translation back to the original world frame
      // before scoring or refining it.
      ippe_object_points.emplace_back(point.x, point.y, 0.0F);
    }

    std::vector<cv::Mat> candidate_rvecs;
    std::vector<cv::Mat> candidate_tvecs;
    const int solution_count = cv::solvePnPGeneric(
        ippe_object_points, image_points, camera_matrix,
        distortion_coefficients,
        candidate_rvecs, candidate_tvecs, false, cv::SOLVEPNP_IPPE);
    if (solution_count <= 0 ||
        candidate_rvecs.size() != candidate_tvecs.size()) {
      return {false, "OpenCV IPPE returned no pose candidates"};
    }

    std::vector<PlanarCandidate> valid_candidates;
    for (std::size_t i = 0; i < candidate_rvecs.size(); ++i) {
      cv::Mat candidate_rotation;
      cv::Rodrigues(candidate_rvecs[i], candidate_rotation);
      const cv::Mat plane_offset =
          (cv::Mat_<double>(3, 1) << 0.0, 0.0, plane_z);
      const cv::Mat world_translation =
          candidate_tvecs[i] - candidate_rotation * plane_offset;
      PlanarCandidate candidate;
      if (evaluatePlanarCandidate(
              object_points, image_points, camera_matrix,
              distortion_coefficients, prior, candidate_rvecs[i],
              world_translation, candidate)) {
        valid_candidates.push_back(std::move(candidate));
      }
    }
    cv::Mat prior_rotation;
    prior.expected_object_to_camera_rotation.convertTo(prior_rotation,
                                                        CV_64F);
    cv::Mat prior_rvec;
    cv::Rodrigues(prior_rotation, prior_rvec);
    cv::Mat prior_tvec;
    if (translationForKnownRotation(
            object_points, image_points, camera_matrix,
            distortion_coefficients, prior_rotation, prior_tvec) &&
        cv::solvePnP(object_points, image_points, camera_matrix,
                     distortion_coefficients, prior_rvec, prior_tvec, true,
                     cv::SOLVEPNP_ITERATIVE)) {
      PlanarCandidate prior_seeded;
      if (evaluatePlanarCandidate(
              object_points, image_points, camera_matrix,
              distortion_coefficients, prior, prior_rvec, prior_tvec,
              prior_seeded)) {
        valid_candidates.push_back(std::move(prior_seeded));
      }
    }
    if (valid_candidates.empty()) {
      return {false, "IPPE returned no physically valid pose"};
    }

    const auto best = std::min_element(
        valid_candidates.begin(), valid_candidates.end(),
        [](const PlanarCandidate &left, const PlanarCandidate &right) {
          if (std::abs(left.geometry_error - right.geometry_error) > 1e-9) {
            return left.geometry_error < right.geometry_error;
          }
          return left.pose.rms_reprojection_error <
                 right.pose.rms_reprojection_error;
        });
    PnpEstimate result = best->pose;

    cv::Mat refined_rvec = result.object_to_camera_rvec.clone();
    cv::Mat refined_tvec = result.object_to_camera_translation.clone();
    cv::solvePnPRefineLM(
        object_points, image_points, camera_matrix, distortion_coefficients,
        refined_rvec, refined_tvec,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100,
                         1e-12));
    PlanarCandidate refined;
    if (evaluatePlanarCandidate(
            object_points, image_points, camera_matrix,
            distortion_coefficients, prior, refined_rvec, refined_tvec,
            refined) &&
        refined.geometry_error <= best->geometry_error + 1e-9 &&
        refined.pose.rms_reprojection_error <=
            result.rms_reprojection_error + 1e-9) {
      result = std::move(refined.pose);
    }
    return result;
  } catch (const cv::Exception &error) {
    return {false, "planar IPPE failed: " + std::string(error.what())};
  }
}

} // namespace pose_estimation
