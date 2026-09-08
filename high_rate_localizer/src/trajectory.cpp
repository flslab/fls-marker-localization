#include "fls_localizer/trajectory.hpp"

#include "fls_localizer/pose_solver.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace flsloc {
namespace {

using json = nlohmann::json;

cv::Vec3d vec3(const json &value, const char *field) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(std::string(field) + " must have three values");
  }
  cv::Vec3d result{value[0].get<double>(), value[1].get<double>(),
                   value[2].get<double>()};
  if (!cv::checkRange(cv::Mat(result))) {
    throw std::runtime_error(std::string(field) +
                             " contains a non-finite value");
  }
  return result;
}

cv::Vec4d quaternion(const json &value) {
  if (!value.is_array() || value.size() != 4) {
    throw std::runtime_error("quaternion_xyzw must have four values");
  }
  const auto normalized =
      normalizeQuaternion({value[0].get<double>(), value[1].get<double>(),
                           value[2].get<double>(), value[3].get<double>()});
  if (!normalized) {
    throw std::runtime_error("quaternion_xyzw is invalid");
  }
  return *normalized;
}

} // namespace

GroundTruthTrajectory
GroundTruthTrajectory::load(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open ground-truth trajectory: " +
                             path.string());
  }
  const json root = json::parse(input);
  if (root.value("schema", "") != "fls-drone-trajectory" ||
      root.value("version", 0) != 1 ||
      root.value("coordinate_frame", "") != "world_FLU" ||
      root.value("quaternion_order", "") != "xyzw") {
    throw std::runtime_error("unsupported ground-truth trajectory schema");
  }

  GroundTruthTrajectory trajectory;
  trajectory.frame_rate_ = root.at("fps").get<double>();
  if (!std::isfinite(trajectory.frame_rate_) || trajectory.frame_rate_ <= 0.0) {
    throw std::runtime_error("trajectory fps must be positive");
  }

  const auto &frames = root.at("frames");
  if (!frames.is_array() || frames.empty()) {
    throw std::runtime_error("trajectory contains no frames");
  }
  trajectory.samples_.reserve(frames.size());
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto &frame = frames[index];
    TrajectorySample sample;
    sample.video_frame = frame.at("video_frame").get<std::uint64_t>();
    sample.blender_frame = frame.at("frame").get<std::int64_t>();
    sample.timestamp = frame.at("time").get<double>();
    sample.position_world_flu =
        vec3(frame.at("position"), "trajectory position");
    sample.quaternion_xyzw = quaternion(frame.at("quaternion_xyzw"));
    if (sample.video_frame != index || !std::isfinite(sample.timestamp) ||
        sample.timestamp < 0.0) {
      throw std::runtime_error(
          "trajectory frames must be finite, ordered, and zero-based");
    }
    const double expected_time =
        static_cast<double>(index) / trajectory.frame_rate_;
    if (std::abs(sample.timestamp - expected_time) >
        0.25 / trajectory.frame_rate_) {
      throw std::runtime_error("trajectory frame time does not match its fps");
    }
    trajectory.samples_.push_back(sample);
  }
  if (root.contains("frame_count") &&
      root.at("frame_count").get<std::size_t>() != trajectory.samples_.size()) {
    throw std::runtime_error("trajectory frame_count does not match frames");
  }
  return trajectory;
}

const TrajectorySample &
GroundTruthTrajectory::sample(std::uint64_t video_frame) const {
  if (video_frame >= samples_.size()) {
    throw std::out_of_range("video has more frames than its trajectory");
  }
  return samples_[video_frame];
}

void TrajectoryEvaluator::evaluate(const TrajectorySample &truth,
                                   FrameResult &result) {
  GroundTruthEvaluation &evaluation = result.ground_truth;
  evaluation.available = true;
  evaluation.video_frame = truth.video_frame;
  evaluation.blender_frame = truth.blender_frame;
  evaluation.position_world_flu = truth.position_world_flu;
  evaluation.quaternion_xyzw = truth.quaternion_xyzw;

  if (result.pose.valid) {
    evaluation.pose_evaluated = true;
    evaluation.position_error_xyz =
        result.pose.drone_position_world - truth.position_world_flu;
    const double squared_error =
        evaluation.position_error_xyz.dot(evaluation.position_error_xyz);
    evaluation.position_rmse_frame_m = std::sqrt(squared_error);
    squared_error_sum_ += squared_error;
    ++evaluated_pose_count_;
  }
  evaluation.evaluated_pose_count = evaluated_pose_count_;
  evaluation.position_rmse_cumulative_m = cumulativePositionRmse();
}

double TrajectoryEvaluator::cumulativePositionRmse() const {
  if (evaluated_pose_count_ == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(squared_error_sum_ /
                   static_cast<double>(evaluated_pose_count_));
}

} // namespace flsloc
