#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

struct ApplicationOptions;

struct PoseOutput {
  // Grid-mode shared-memory contract: position is the drone origin in world
  // coordinates and orientation is shared-attitude drone RPY in radians.
  bool valid = false;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

struct AttitudeSample {
  bool valid = false;
  std::string status = "disabled";
  cv::Vec4d quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
  double host_timestamp = 0.0;
  double age_seconds = 0.0;
  std::uint32_t sequence = 0;
};

struct FrameProcessingResult {
  nlohmann::json log;
  PoseOutput pose;
};

class FrameProcessor {
public:
  virtual ~FrameProcessor() = default;

  virtual FrameProcessingResult process(cv::Mat &image, double timestamp,
                                        const AttitudeSample &attitude) = 0;
  virtual void appendConfiguration(nlohmann::json &configuration,
                                   cv::Size image_size) const = 0;
};

std::unique_ptr<FrameProcessor>
createFrameProcessor(const ApplicationOptions &options,
                     const cv::Mat &camera_matrix,
                     const cv::Mat &distortion_coefficients,
                     const std::vector<cv::Point3f> &marker_points);
