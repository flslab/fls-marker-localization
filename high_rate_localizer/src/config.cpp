#include "fls_localizer/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace flsloc {
namespace {

using json = nlohmann::json;

cv::Mat readMatrix(const json &value, int rows, int columns,
                   const char *field) {
  if (!value.is_array() || static_cast<int>(value.size()) != rows) {
    throw std::runtime_error(std::string(field) + " has the wrong shape");
  }
  cv::Mat output(rows, columns, CV_64F);
  for (int row = 0; row < rows; ++row) {
    if (!value[row].is_array() ||
        static_cast<int>(value[row].size()) != columns) {
      throw std::runtime_error(std::string(field) + " has the wrong shape");
    }
    for (int column = 0; column < columns; ++column) {
      output.at<double>(row, column) = value[row][column].get<double>();
    }
  }
  return output;
}

cv::Vec3d readVec3(const json &value, const char *field) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(std::string(field) + " must have three values");
  }
  return {value[0].get<double>(), value[1].get<double>(),
          value[2].get<double>()};
}

} // namespace

void applyOutputTag(OutputConfig &output, const std::string &tag) {
  if (tag.empty()) {
    return;
  }
  const auto tagged_name = [&tag](const std::string &name) {
    std::filesystem::path path(name);
    path.replace_filename(path.stem().string() + "_" + tag +
                          path.extension().string());
    return path.string();
  };
  output.annotated_video_name = tagged_name(output.annotated_video_name);
  output.json_name = tagged_name(output.json_name);
}

ApplicationConfig loadApplicationConfig(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open configuration: " + path.string());
  }
  const json root = json::parse(input);
  ApplicationConfig config;

  const auto &camera = root.at("camera");
  config.camera.width = camera.value("width", config.camera.width);
  config.camera.height = camera.value("height", config.camera.height);
  config.camera.frame_rate =
      camera.value("frame_rate", config.camera.frame_rate);
  config.camera.buffer_count =
      camera.value("buffer_count", config.camera.buffer_count);
  config.camera.pixel_format =
      camera.value("pixel_format", config.camera.pixel_format);
  config.camera.exposure_time_us =
      camera.value("exposure_time_us", config.camera.exposure_time_us);
  config.camera.analogue_gain =
      camera.value("analogue_gain", config.camera.analogue_gain);
  config.camera.brightness =
      camera.value("brightness", config.camera.brightness);
  config.camera.contrast = camera.value("contrast", config.camera.contrast);

  const auto &calibration = root.at("calibration");
  config.calibration.camera_matrix =
      readMatrix(calibration.at("camera_matrix"), 3, 3, "camera_matrix");
  const std::vector<double> distortion =
      calibration.at("distortion_coefficients").get<std::vector<double>>();
  config.calibration.distortion = cv::Mat(distortion, true);

  if (root.contains("detector")) {
    const auto &detector = root["detector"];
    config.detector.intensity_threshold = detector.value(
        "intensity_threshold", config.detector.intensity_threshold);
    config.detector.minimum_area =
        detector.value("minimum_area", config.detector.minimum_area);
    config.detector.maximum_area =
        detector.value("maximum_area", config.detector.maximum_area);
    config.detector.minimum_fill_ratio = detector.value(
        "minimum_fill_ratio", config.detector.minimum_fill_ratio);
    config.detector.maximum_candidates = detector.value(
        "maximum_candidates", config.detector.maximum_candidates);
  }

  if (root.contains("tracking")) {
    const auto &tracking = root["tracking"];
    config.tracking.maximum_pose_points = tracking.value(
        "maximum_pose_points", config.tracking.maximum_pose_points);
    config.tracking.initial_distance_m = tracking.value(
        "initial_distance_m", config.tracking.initial_distance_m);
    config.tracking.maximum_reprojection_error_px =
        tracking.value("maximum_reprojection_error_px",
                       config.tracking.maximum_reprojection_error_px);
    config.tracking.lattice_tolerance_cells = tracking.value(
        "lattice_tolerance_cells", config.tracking.lattice_tolerance_cells);
    config.tracking.projection_gate_px = tracking.value(
        "projection_gate_px", config.tracking.projection_gate_px);
    config.tracking.hypergrid_confirmation_frames =
        tracking.value("hypergrid_confirmation_frames",
                       config.tracking.hypergrid_confirmation_frames);
    config.tracking.lost_after_frames =
        tracking.value("lost_after_frames", config.tracking.lost_after_frames);
    config.tracking.maximum_attitude_age_s = tracking.value(
        "maximum_attitude_age_s", config.tracking.maximum_attitude_age_s);
  }

  if (root.contains("output")) {
    const auto &output = root["output"];
    config.output.directory =
        output.value("directory", config.output.directory.string());
    config.output.annotated_video_fps =
        output.value("annotated_video_fps", config.output.annotated_video_fps);
    config.output.annotated_video_name = output.value(
        "annotated_video_name", config.output.annotated_video_name);
    config.output.json_name =
        output.value("json_name", config.output.json_name);
  }

  const auto &mount = root.at("camera_mount");
  cv::Mat mount_rotation = readMatrix(mount.at("camera_to_drone_rotation"), 3,
                                      3, "camera_to_drone_rotation");
  config.camera_to_drone_rotation = mount_rotation;
  config.camera_position_drone_flu =
      readVec3(mount.at("position_drone_flu_m"), "position_drone_flu_m");

  config.grid_file = root.at("grid_file").get<std::string>();
  if (config.grid_file.is_relative()) {
    config.grid_file = path.parent_path() / config.grid_file;
  }
  config.shared_memory_name =
      root.value("shared_memory_name", config.shared_memory_name);
  if (root.contains("landing_tile")) {
    config.default_landing_tile_i = root["landing_tile"].at(0).get<int>();
    config.default_landing_tile_j = root["landing_tile"].at(1).get<int>();
  }
  config.video_test_landing_time_s =
      root.value("video_test_landing_time_s", config.video_test_landing_time_s);

  if (config.camera.width <= 0 || config.camera.height <= 0 ||
      config.camera.frame_rate <= 0.0 || config.detector.minimum_area <= 0 ||
      config.detector.maximum_area < config.detector.minimum_area ||
      config.detector.maximum_candidates < 4 ||
      config.detector.maximum_candidates > 64 ||
      config.tracking.maximum_pose_points < 4 ||
      config.tracking.maximum_pose_points > 16) {
    throw std::runtime_error("configuration contains an invalid bound");
  }
  return config;
}

} // namespace flsloc
