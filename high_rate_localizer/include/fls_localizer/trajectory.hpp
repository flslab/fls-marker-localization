#pragma once

#include "fls_localizer/types.hpp"

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

namespace flsloc {

struct TrajectorySample {
  std::uint64_t video_frame = 0;
  std::int64_t blender_frame = 0;
  double timestamp = 0.0;
  cv::Vec3d position_world_flu{};
  cv::Vec4d quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
};

struct OrientationErrorConfig {
  cv::Vec3d bias_rpy_rad{};
  cv::Vec3d noise_stddev_rpy_rad{};
  std::uint64_t seed = 0;
};

class OrientationErrorModel {
public:
  explicit OrientationErrorModel(OrientationErrorConfig config);

  cv::Vec4d apply(const cv::Vec4d &quaternion_xyzw);

private:
  OrientationErrorConfig config_;
  bool enabled_ = false;
  std::mt19937_64 random_;
  std::normal_distribution<double> standard_normal_{0.0, 1.0};
};

class GroundTruthTrajectory {
public:
  static GroundTruthTrajectory load(const std::filesystem::path &path);

  const TrajectorySample &sample(std::uint64_t video_frame) const;
  std::size_t size() const { return samples_.size(); }
  double frameRate() const { return frame_rate_; }

private:
  double frame_rate_ = 0.0;
  std::vector<TrajectorySample> samples_;
};

class TrajectoryEvaluator {
public:
  void evaluate(const TrajectorySample &truth, FrameResult &result);

  double cumulativePositionRmse() const;
  std::uint64_t evaluatedPoseCount() const { return evaluated_pose_count_; }

private:
  double squared_error_sum_ = 0.0;
  std::uint64_t evaluated_pose_count_ = 0;
};

} // namespace flsloc
