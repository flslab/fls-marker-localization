#pragma once

#include <array>
#include <string>

struct ApplicationOptions {
  bool print_logs = false;
  bool preview = false;
  double distance = -1.0;
  int execution_time = 0;
  double save_rate = 1.0;
  bool save_frames = false;
  std::string save_frames_path;
  bool save_video = false;
  int video_fps = 30;
  std::string video_path;
  std::string config_file = "camera_config.json";
  std::string json_path;
  bool raw_preview = false;
  bool raw_stream = false;
  bool raw_save_frame = false;
  bool raw_save_video = false;
  bool aruco_mode = false;
  std::string video_input_path;
  bool enable_streaming = false;
  int stream_port = 8080;
  std::string stream_type = "http";
  double stream_rate = 10.0;
  double contrast = -2.0;
  double brightness = -2.0;
  int exposure_time = -2;
  int frame_rate = 120;
  int encoder_frame_rate = 50;
  int cam_width = 640;
  int cam_height = 400;
  double blob_area_threshold = 3.0;
  double dark_blob_intensity = 0.0;
  int payload_size = 4;
  int target_id = -1;
  double tracking_threshold = 30.0;
  double sync_threshold = 4.5;
  bool static_markers_mode = false;
  bool validate_mode = false;
  std::string grid_map_file;
  int grid_window_size = 2;
  bool grid_center_ap3p = false;
  double grid_rounding_tolerance = 0.30;
  double grid_max_marker_age = 0.0;
  double max_attitude_age = 0.1;
  std::array<double, 3> camera_offset_drone{0.0, 0.0, 0.0};
  bool enable_kalman_filter = false;
  double kf_process_noise = 0.5;
  double kf_measurement_noise = 0.02;
};

ApplicationOptions parseApplicationOptions(int argc, char **argv);
