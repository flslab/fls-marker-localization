#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

namespace flsloc {

struct CameraConfig {
  int width = 640;
  int height = 400;
  double frame_rate = 120.0;
  int buffer_count = 4;
  std::string pixel_format = "YUV420";
  int exposure_time_us = 3000;
  double analogue_gain = 1.0;
  double brightness = 0.0;
  double contrast = 1.0;
};

struct CalibrationConfig {
  cv::Mat camera_matrix;
  cv::Mat distortion;
};

struct DetectorConfig {
  int intensity_threshold = 160;
  int minimum_area = 8;
  int maximum_area = 6500;
  float minimum_fill_ratio = 0.45F;
  std::size_t maximum_candidates = 64;
};

struct TrackingConfig {
  std::size_t maximum_pose_points = 16;
  double initial_distance_m = 0.045;
  double maximum_reprojection_error_px = 5.0;
  double lattice_tolerance_cells = 0.45;
  double projection_gate_px = 35.0;
  int hypergrid_confirmation_frames = 3;
  int lost_after_frames = 30;
  double maximum_attitude_age_s = 0.1;
};

struct OutputConfig {
  std::filesystem::path directory = "logs/high_rate_localizer";
  double annotated_video_fps = 30.0;
  std::string annotated_video_name = "video.mp4";
  std::string json_name = "log.json";
};

void applyOutputTag(OutputConfig &output, const std::string &tag);

struct ApplicationConfig {
  CameraConfig camera;
  CalibrationConfig calibration;
  DetectorConfig detector;
  TrackingConfig tracking;
  OutputConfig output;
  cv::Matx33d camera_to_drone_rotation{0.0, -1.0, 0.0, -1.0, 0.0,
                                       0.0, 0.0,  0.0, -1.0};
  cv::Vec3d camera_position_drone_flu{0.0, 0.0, 0.0};
  std::filesystem::path grid_file;
  std::string shared_memory_name = "/fls_localizer_v2";
  int default_landing_tile_i = 0;
  int default_landing_tile_j = 0;
  double video_test_landing_time_s = 16.0;
};

ApplicationConfig loadApplicationConfig(const std::filesystem::path &path);

} // namespace flsloc
