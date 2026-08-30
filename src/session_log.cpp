#include "session_log.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "application_options.h"
#include "frame_processor.h"

using json = nlohmann::json;

namespace {

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH unknown
#endif

#define STRINGIFY_VALUE(value) #value
#define STRINGIFY(value) STRINGIFY_VALUE(value)

std::string gitCommitHash() { return STRINGIFY(GIT_COMMIT_HASH); }

json optionsJson(const ApplicationOptions &options) {
  return {{"print_logs", options.print_logs},
          {"preview", options.preview},
          {"initial_distance", options.distance},
          {"execution_time", options.execution_time},
          {"save_rate", options.save_rate},
          {"save_frames", options.save_frames},
          {"save_frames_path", options.save_frames_path},
          {"save_video", options.save_video},
          {"video_fps", options.video_fps},
          {"video_path", options.video_path},
          {"config_file", options.config_file},
          {"json_path", options.json_path},
          {"raw_preview", options.raw_preview},
          {"raw_stream", options.raw_stream},
          {"raw_save_frame", options.raw_save_frame},
          {"raw_save_video", options.raw_save_video},
          {"aruco_mode", options.aruco_mode},
          {"enable_streaming", options.enable_streaming},
          {"stream_port", options.stream_port},
          {"stream_type", options.stream_type},
          {"stream_rate", options.stream_rate},
          {"contrast", options.contrast},
          {"brightness", options.brightness},
          {"exposure_time", options.exposure_time},
          {"frame_rate", options.frame_rate},
          {"encoder_frame_rate", options.encoder_frame_rate},
          {"cam_width", options.cam_width},
          {"cam_height", options.cam_height},
          {"blob_area_threshold", options.blob_area_threshold},
          {"dark_blob_intensity", options.dark_blob_intensity},
          {"payload_size", options.payload_size},
          {"target_id", options.target_id},
          {"tracking_threshold", options.tracking_threshold},
          {"sync_threshold", options.sync_threshold},
          {"static_markers_mode", options.static_markers_mode},
          {"validate_mode", options.validate_mode},
          {"grid_map_file", options.grid_map_file},
          {"grid_window_size", options.grid_window_size},
          {"grid_center_ap3p", options.grid_center_ap3p},
          {"grid_rounding_tolerance", options.grid_rounding_tolerance},
          {"grid_max_marker_age", options.grid_max_marker_age},
          {"max_attitude_age", options.max_attitude_age},
          {"camera_offset_drone", options.camera_offset_drone},
          {"enable_kalman_filter", options.enable_kalman_filter},
          {"kf_process_noise", options.kf_process_noise},
          {"kf_measurement_noise", options.kf_measurement_noise},
          {"video_input_path", options.video_input_path}};
}

} // namespace

std::string createSessionDirectory(const ApplicationOptions &options) {
  const std::time_t now = std::time(nullptr);
  const std::tm local_time = *std::localtime(&now);
  std::ostringstream name;
  name << "logs/" << std::put_time(&local_time, "%H_%M_%S_%m_%d_%Y");

  std::error_code error;
  std::filesystem::create_directories(name.str(), error);
  if (error) {
    throw std::runtime_error("Unable to create directory " + name.str());
  }
  if (!options.save_frames_path.empty()) {
    std::filesystem::create_directories(options.save_frames_path, error);
    if (error) {
      throw std::runtime_error("Unable to create directory " +
                               options.save_frames_path);
    }
  }
  return name.str();
}

SessionLog::SessionLog(const ApplicationOptions &options) : options(options) {}

void SessionLog::addFrame(int frame_id, double timestamp, json frame_log) {
  frame_log["time"] = timestamp;
  frame_log["frame_id"] = frame_id;
  frames.push_back(std::move(frame_log));
}

void SessionLog::save(const std::string &log_directory,
                      const FrameProcessor &processor, cv::Size image_size,
                      double video_start_time) const {
  json log;
  log["args"] = optionsJson(options);
  log["config"] = {{"initial_distance", options.distance},
                   {"git_version", gitCommitHash()},
                   {"kalman_filter_enabled", options.enable_kalman_filter}};
  processor.appendConfiguration(log["config"], image_size);
  if (video_start_time >= 0.0) {
    log["config"]["video_start_time"] = video_start_time;
  }
  if (options.enable_kalman_filter) {
    log["config"]["kf_process_noise"] = options.kf_process_noise;
    log["config"]["kf_measurement_noise"] = options.kf_measurement_noise;
  }
  log["frames"] = frames;

  const std::string filename = options.json_path.empty()
                                   ? log_directory + "/log.json"
                                   : options.json_path;
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to write logs to file." << std::endl;
    return;
  }
  file << log.dump(4);
  std::cout << "Logs saved to " << filename << std::endl;
}
