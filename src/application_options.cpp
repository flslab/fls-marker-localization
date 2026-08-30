#include "application_options.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

const char *requireValue(const std::string &option, int &index, int argc,
                         char **argv) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(option + " requires a value");
  }
  return argv[++index];
}

int parseInteger(const std::string &option, int &index, int argc, char **argv) {
  const std::string text = requireValue(option, index, argc, argv);
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(option + " requires an integer, got '" + text +
                                "'");
  }
  return value;
}

double parseDouble(const std::string &option, int &index, int argc,
                   char **argv) {
  const std::string text = requireValue(option, index, argc, argv);
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(option + " requires a number, got '" + text +
                                "'");
  }
  return value;
}

} // namespace

ApplicationOptions parseApplicationOptions(int argc, char **argv) {
  ApplicationOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if (argument == "--verbose" || argument == "-v") {
      options.print_logs = true;
    } else if (argument == "--preview" || argument == "-p") {
      options.preview = true;
    } else if (argument == "--distance" || argument == "-d") {
      options.distance = parseDouble(argument, i, argc, argv);
      if (!std::isfinite(options.distance) || options.distance <= 0.0) {
        throw std::invalid_argument(argument +
                                    " must be a positive finite number");
      }
    } else if (argument == "--time" || argument == "-t") {
      options.execution_time = parseInteger(argument, i, argc, argv);
    } else if (argument == "--save-frames") {
      options.save_frames = true;
    } else if (argument == "--save-frames-path") {
      options.save_frames_path = requireValue(argument, i, argc, argv);
    } else if (argument == "--raw-preview") {
      options.raw_preview = true;
    } else if (argument == "--raw-stream") {
      options.raw_stream = true;
    } else if (argument == "--raw-save-frame") {
      options.raw_save_frame = true;
    } else if (argument == "--raw-save-video") {
      options.raw_save_video = true;
    } else if (argument == "--save-video" || argument == "-s") {
      options.save_video = true;
    } else if (argument == "--video-fps") {
      options.video_fps = parseInteger(argument, i, argc, argv);
    } else if (argument == "--video-path") {
      options.video_path = requireValue(argument, i, argc, argv);
    } else if (argument == "--json-path") {
      options.json_path = requireValue(argument, i, argc, argv);
    } else if (argument == "--config") {
      options.config_file = requireValue(argument, i, argc, argv);
    } else if (argument == "--save-rate") {
      options.save_rate = parseDouble(argument, i, argc, argv);
    } else if (argument == "--grid-map") {
      options.grid_map_file = requireValue(argument, i, argc, argv);
    } else if (argument == "--window-size" ||
               argument == "--grid-window-size" || argument == "-w") {
      options.grid_window_size = parseInteger(argument, i, argc, argv);
    } else if (argument == "--grid-center-ap3p") {
      options.grid_center_ap3p = true;
    } else if (argument == "--grid-rounding-tolerance") {
      options.grid_rounding_tolerance = parseDouble(argument, i, argc, argv);
    } else if (argument == "--grid-max-marker-age") {
      options.grid_max_marker_age = parseDouble(argument, i, argc, argv);
    } else if (argument == "--max-attitude-age") {
      options.max_attitude_age = parseDouble(argument, i, argc, argv);
    } else if (argument == "--camera-offset") {
      for (double &component : options.camera_offset_drone) {
        component = parseDouble(argument, i, argc, argv);
      }
    } else if (argument == "--contrast") {
      options.contrast = parseDouble(argument, i, argc, argv);
    } else if (argument == "--brightness") {
      options.brightness = parseDouble(argument, i, argc, argv);
    } else if (argument == "--exposure") {
      options.exposure_time = parseInteger(argument, i, argc, argv);
    } else if (argument == "--fps") {
      options.frame_rate = parseInteger(argument, i, argc, argv);
    } else if (argument == "--encoder-fps") {
      options.encoder_frame_rate = parseInteger(argument, i, argc, argv);
    } else if (argument == "--stream" || argument == "--streaming") {
      options.enable_streaming = true;
    } else if (argument == "--stream-port") {
      options.stream_port = parseInteger(argument, i, argc, argv);
    } else if (argument == "--stream-type") {
      options.stream_type = requireValue(argument, i, argc, argv);
      if (options.stream_type != "http" && options.stream_type != "udp") {
        throw std::invalid_argument(
            "Invalid stream type. Use 'http' or 'udp'.");
      }
    } else if (argument == "--stream-rate") {
      options.stream_rate = parseDouble(argument, i, argc, argv);
    } else if (argument == "--blob-area-threshold") {
      options.blob_area_threshold = parseDouble(argument, i, argc, argv);
    } else if (argument == "--dark-blob-intensity") {
      options.dark_blob_intensity = parseDouble(argument, i, argc, argv);
    } else if (argument == "--payload-size") {
      options.payload_size = parseInteger(argument, i, argc, argv);
    } else if (argument == "--target-id") {
      options.target_id = parseInteger(argument, i, argc, argv);
    } else if (argument == "--tracking-threshold") {
      options.tracking_threshold = parseDouble(argument, i, argc, argv);
    } else if (argument == "--sync-threshold") {
      options.sync_threshold = parseDouble(argument, i, argc, argv);
    } else if (argument == "--static-markers" || argument == "--static-blobs") {
      options.static_markers_mode = true;
    } else if (argument == "--validate") {
      options.validate_mode = true;
    } else if (argument == "--aruco") {
      options.aruco_mode = true;
    } else if (argument == "--video-input") {
      options.video_input_path = requireValue(argument, i, argc, argv);
    } else if (argument == "--width") {
      options.cam_width = parseInteger(argument, i, argc, argv);
    } else if (argument == "--height") {
      options.cam_height = parseInteger(argument, i, argc, argv);
    } else if (argument == "--kalman-filter" || argument == "--kf") {
      options.enable_kalman_filter = true;
    } else if (argument == "--kf-process-noise") {
      options.kf_process_noise = parseDouble(argument, i, argc, argv);
    } else if (argument == "--kf-measurement-noise") {
      options.kf_measurement_noise = parseDouble(argument, i, argc, argv);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if (!std::isfinite(options.grid_max_marker_age) ||
      options.grid_max_marker_age < 0.0) {
    throw std::invalid_argument(
        "--grid-max-marker-age must be a non-negative finite number");
  }
  if (!std::isfinite(options.max_attitude_age) ||
      options.max_attitude_age <= 0.0) {
    throw std::invalid_argument(
        "--max-attitude-age must be a positive finite number");
  }
  for (double component : options.camera_offset_drone) {
    if (!std::isfinite(component)) {
      throw std::invalid_argument(
          "--camera-offset values must be finite numbers");
    }
  }
  if (options.payload_size < 1 || options.payload_size > 16) {
    throw std::invalid_argument("--payload-size must be between 1 and 16 bits");
  }
  if (!std::isfinite(options.dark_blob_intensity) ||
      options.dark_blob_intensity < 0.0 ||
      options.dark_blob_intensity >= 0.8) {
    throw std::invalid_argument(
        "--dark-blob-intensity must be in the range [0, 0.8)");
  }
  if (options.grid_center_ap3p && options.grid_map_file.empty()) {
    throw std::invalid_argument("--grid-center-ap3p requires --grid-map");
  }
  if (options.grid_center_ap3p && options.grid_window_size != 2) {
    throw std::invalid_argument(
        "--grid-center-ap3p requires --window-size 2");
  }
  if (!options.grid_map_file.empty() && options.aruco_mode) {
    throw std::invalid_argument(
        "--grid-map cannot be combined with --aruco");
  }
  if (!options.grid_map_file.empty() &&
      (!std::isfinite(options.distance) || options.distance <= 0.0)) {
    throw std::invalid_argument(
        "--grid-map requires a positive --distance");
  }
  return options;
}
