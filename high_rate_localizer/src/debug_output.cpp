#include "fls_localizer/debug_output.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <stdexcept>

namespace flsloc {
namespace {

using json = nlohmann::json;

double rounded(double value, double scale = 1'000'000.0) {
  return std::round(value * scale) / scale;
}

json vec3(const cv::Vec3d &value) {
  return {rounded(value[0]), rounded(value[1]), rounded(value[2])};
}
json vec4(const cv::Vec4d &value) {
  return {rounded(value[0]), rounded(value[1]), rounded(value[2]),
          rounded(value[3])};
}

std::string webGridType(PoseSource source) {
  return source == PoseSource::MyGrid ? "short_range" : "main";
}

} // namespace

DebugOutput::DebugOutput(const ApplicationConfig &config, const GridMap &map,
                         std::string input_description,
                         std::string trajectory_description)
    : config_(config), map_(map),
      input_description_(std::move(input_description)),
      trajectory_description_(std::move(trajectory_description)) {
  std::filesystem::create_directories(config_.output.directory);
  log_path_ = config_.output.directory / config_.output.json_name;
  temporary_log_path_ = log_path_;
  temporary_log_path_ += ".tmp";
  video_path_ = config_.output.directory / config_.output.annotated_video_name;
  worker_ = std::thread(&DebugOutput::worker, this);
}

DebugOutput::~DebugOutput() {
  try {
    finish();
  } catch (...) {
  }
}

bool DebugOutput::wantsVideoFrame(double timestamp) {
  if (config_.output.annotated_video_fps <= 0.0) {
    return false;
  }
  if (next_video_timestamp_ < 0.0 ||
      timestamp + 1e-9 >= next_video_timestamp_) {
    next_video_timestamp_ =
        timestamp + 1.0 / config_.output.annotated_video_fps;
    return true;
  }
  return false;
}

cv::Mat DebugOutput::annotate(const cv::Mat &image,
                              const FrameResult &result) const {
  cv::Mat annotated;
  if (image.channels() == 1) {
    cv::cvtColor(image, annotated, cv::COLOR_GRAY2BGR);
  } else {
    annotated = image.clone();
  }
  for (const Blob &blob : result.blobs) {
    cv::Scalar color(0, 0, 255);
    if (blob.classification == PoseSource::HyperGrid) {
      color = cv::Scalar(0, 255, 0);
    } else if (blob.classification == PoseSource::MyGrid) {
      color = cv::Scalar(0, 165, 255);
    }
    cv::circle(annotated, blob.center, 8, color, 2, cv::LINE_AA);
    if (blob.used_for_pose) {
      cv::line(annotated, blob.center + cv::Point2f(-5, 0),
               blob.center + cv::Point2f(5, 0), cv::Scalar(255, 255, 255), 1,
               cv::LINE_AA);
      cv::line(annotated, blob.center + cv::Point2f(0, -5),
               blob.center + cv::Point2f(0, 5), cv::Scalar(255, 255, 255), 1,
               cv::LINE_AA);
    }
  }
  const std::string line =
      std::string(toString(result.state)) + "  " + toString(result.source) +
      "  points=" + std::to_string(result.matched.size()) + "  " +
      std::to_string(result.processing_ms).substr(0, 4) + " ms";
  cv::putText(annotated, line, {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(80, 255, 80), 1, cv::LINE_AA);
  if (result.pose.valid) {
    std::ostringstream pose;
    pose << std::fixed << std::setprecision(3) << "FLU ["
         << result.pose.drone_position_world[0] << ", "
         << result.pose.drone_position_world[1] << ", "
         << result.pose.drone_position_world[2]
         << "]  rms=" << result.pose.reprojection_rms;
    cv::putText(annotated, pose.str(), {12, 46}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(80, 255, 80), 1, cv::LINE_AA);
  }
  if (result.ground_truth.available) {
    std::ostringstream rmse;
    rmse << std::fixed << std::setprecision(4) << "position RMSE xyz  frame=";
    if (result.ground_truth.pose_evaluated) {
      rmse << result.ground_truth.position_rmse_frame_m << " m";
    } else {
      rmse << "n/a";
    }
    rmse << "  cumulative=";
    if (std::isfinite(result.ground_truth.position_rmse_cumulative_m)) {
      rmse << result.ground_truth.position_rmse_cumulative_m << " m";
    } else {
      rmse << "n/a";
    }
    cv::putText(annotated, rmse.str(), {12, 68}, cv::FONT_HERSHEY_SIMPLEX, 0.43,
                cv::Scalar(80, 255, 255), 1, cv::LINE_AA);

    std::ostringstream truth;
    truth << std::fixed << std::setprecision(4) << "GT FLU xyz ["
          << result.ground_truth.position_world_flu[0] << ", "
          << result.ground_truth.position_world_flu[1] << ", "
          << result.ground_truth.position_world_flu[2] << "] m";
    cv::putText(annotated, truth.str(), {12, 90}, cv::FONT_HERSHEY_SIMPLEX,
                0.43, cv::Scalar(80, 255, 255), 1, cv::LINE_AA);

    std::ostringstream absolute_error;
    absolute_error << std::fixed << std::setprecision(4) << "abs error xyz ";
    if (result.ground_truth.pose_evaluated) {
      absolute_error << "["
                     << std::abs(result.ground_truth.position_error_xyz[0])
                     << ", "
                     << std::abs(result.ground_truth.position_error_xyz[1])
                     << ", "
                     << std::abs(result.ground_truth.position_error_xyz[2])
                     << "] m";
    } else {
      absolute_error << "[n/a]";
    }
    cv::putText(annotated, absolute_error.str(), {12, 112},
                cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(80, 180, 255), 1,
                cv::LINE_AA);
  }
  return annotated;
}

void DebugOutput::submit(FrameResult result, cv::Mat annotated) {
  std::unique_lock lock(mutex_);
  if (queue_.size() >= 192) {
    annotated.release();
  }
  space_.wait(lock, [&] { return queue_.size() < 256 || stopping_; });
  if (stopping_) {
    return;
  }
  queue_.push_back({std::move(result), std::move(annotated)});
  ready_.notify_one();
}

void DebugOutput::finish() {
  {
    std::lock_guard lock(mutex_);
    if (finished_) {
      return;
    }
    finished_ = true;
    stopping_ = true;
  }
  ready_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (!worker_error_.empty()) {
    throw std::runtime_error(worker_error_);
  }
}

json DebugOutput::metadata() const {
  json short_range_tiles = json::array();
  for (const MyGridTile &tile : map_.tiles()) {
    json markers = json::array();
    for (const MyGridMarker &marker : tile.markers) {
      markers.push_back({{"id", marker.id},
                         {"local_i", marker.local_i},
                         {"local_j", marker.local_j},
                         {"global_x", marker.world.x},
                         {"global_y", marker.world.y},
                         {"global_z", marker.world.z}});
    }
    short_range_tiles.push_back(
        {{"i", tile.i}, {"j", tile.j}, {"markers", markers}});
  }
  json args = {{"frame_rate", config_.camera.frame_rate},
               {"cam_width", config_.camera.width},
               {"cam_height", config_.camera.height},
               {"save_video", true},
               {"video_fps", config_.output.annotated_video_fps},
               {"grid_map_file", map_.path().string()},
               {"video_input_path", input_description_}};
  if (!trajectory_description_.empty()) {
    args["trajectory_input_path"] = trajectory_description_;
  }
  return {
      {"args", std::move(args)},
      {"config",
       {{"initial_distance", config_.tracking.initial_distance_m},
        {"aruco_mode", false},
        {"blob_grid_localization_enabled", true},
        {"coordinate_frame", "world_FLU"},
        {"marker_grid",
         {{"map_file", map_.path().string()},
          {"rows", 0},
          {"cols", 0},
          {"infinite", true},
          {"cell_spacing", map_.hypergridSpacing()},
          {"grid_origin", {map_.origin().x, map_.origin().y, map_.origin().z}},
          {"pnp_solver", "ippe+shared_attitude"},
          {"camera_offset_drone",
           {config_.camera_position_drone_flu[0],
            config_.camera_position_drone_flu[1],
            config_.camera_position_drone_flu[2]}},
          {"shared_memory_position", "drone_position_world_FLU"},
          {"maximum_pose_points", config_.tracking.maximum_pose_points},
          {"hypergrid_acquisition_height_m",
           2.0 * map_.hypergridSpacing() *
               std::max(config_.calibration.camera_matrix.at<double>(0, 0) /
                            config_.camera.width,
                        config_.calibration.camera_matrix.at<double>(1, 1) /
                            config_.camera.height) *
               1.10},
          {"short_range", {{"tiles", short_range_tiles}}}}}}}};
}

json DebugOutput::frameJson(const FrameResult &result) {
  json blobs = json::array();
  for (const Blob &blob : result.blobs) {
    blobs.push_back({{"x", rounded(blob.center.x, 1000.0)},
                     {"y", rounded(blob.center.y, 1000.0)},
                     {"id", blob.decoded_id}});
  }
  json matched = json::array();
  json used_ids = json::array();
  for (const MatchedPoint &point : result.matched) {
    json marker = {{"id", point.id},
                   {"image_x", rounded(point.image.x, 1000.0)},
                   {"image_y", rounded(point.image.y, 1000.0)},
                   {"grid_type", webGridType(result.source)},
                   {"map_row", point.grid_x},
                   {"map_col", point.grid_y},
                   {"global_position",
                    {rounded(point.world.x), rounded(point.world.y),
                     rounded(point.world.z)}}};
    if (result.source == PoseSource::MyGrid) {
      marker["tile_i"] = point.tile_i;
      marker["tile_j"] = point.tile_j;
      marker["local_i"] = point.local_i;
      marker["local_j"] = point.local_j;
    }
    matched.push_back(std::move(marker));
    used_ids.push_back(point.id);
  }

  json frame = {{"time", rounded(result.timestamp)},
                {"frame_id", result.frame_id},
                {"blobs", blobs},
                {"poses", json::array()},
                {"blob_grid_localization",
                 {{"status", result.status},
                  {"state", toString(result.state)},
                  {"grid_type", webGridType(result.source)},
                  {"pose_valid", result.pose.valid},
                  {"accepted_marker_count", result.matched.size()},
                  {"required_marker_count", 4},
                  {"candidate_count", result.blobs.size()},
                  {"matched_markers", matched},
                  {"mygrid_request", toString(result.mygrid_request)},
                  {"processing_ms", rounded(result.processing_ms, 1000.0)}}}};
  if (result.source == PoseSource::MyGrid) {
    frame["blob_grid_localization"]["tile"] = {{"i", result.tile_i},
                                               {"j", result.tile_j}};
  }
  if (result.pose.valid) {
    frame["blob_grid_localization"]["reprojection_error"] =
        rounded(result.pose.reprojection_rms);
    frame["blob_grid_localization"]["camera_to_plane_distance"] =
        rounded(result.pose.camera_to_plane_distance);
    frame["poses"].push_back(
        {{"camera_pose", true},
         {"source", "blob_grid"},
         {"grid_type", webGridType(result.source)},
         {"camera_position", vec3(result.pose.camera_position_world)},
         {"camera_orientation", vec3(result.pose.camera_rpy)},
         {"drone_position", vec3(result.pose.drone_position_world)},
         {"drone_orientation", vec3(result.pose.drone_rpy)},
         {"drone_quaternion_xyzw", vec4(result.pose.drone_quaternion_xyzw)},
         {"marker_position", vec3(result.pose.tvec_world_to_camera)},
         {"camera_to_plane_distance",
          rounded(result.pose.camera_to_plane_distance)},
         {"pnp_solver", result.pose.solver},
         {"markers_used", result.matched.size()},
         {"used_marker_ids", used_ids},
         {"reprojection_error", rounded(result.pose.reprojection_rms)}});
  }
  if (result.ground_truth.available) {
    json truth = {
        {"video_frame", result.ground_truth.video_frame},
        {"blender_frame", result.ground_truth.blender_frame},
        {"position", vec3(result.ground_truth.position_world_flu)},
        {"quaternion_xyzw", vec4(result.ground_truth.quaternion_xyzw)},
        {"pose_evaluated", result.ground_truth.pose_evaluated},
        {"evaluated_pose_count", result.ground_truth.evaluated_pose_count}};
    if (result.ground_truth.pose_evaluated) {
      truth["position_error_xyz"] =
          vec3(result.ground_truth.position_error_xyz);
      truth["position_rmse_frame_m"] =
          rounded(result.ground_truth.position_rmse_frame_m);
    } else {
      truth["position_error_xyz"] = nullptr;
      truth["position_rmse_frame_m"] = nullptr;
    }
    if (std::isfinite(result.ground_truth.position_rmse_cumulative_m)) {
      truth["position_rmse_cumulative_m"] =
          rounded(result.ground_truth.position_rmse_cumulative_m);
    } else {
      truth["position_rmse_cumulative_m"] = nullptr;
    }
    frame["ground_truth"] = std::move(truth);
  }
  return frame;
}

void DebugOutput::worker() {
  try {
    std::ofstream log(temporary_log_path_);
    if (!log) {
      throw std::runtime_error("unable to open " +
                               temporary_log_path_.string());
    }
    const json root = metadata();
    log << "{\n  \"args\": " << root["args"].dump()
        << ",\n  \"config\": " << root["config"].dump()
        << ",\n  \"frames\": [\n";
    bool first = true;
    cv::VideoWriter video;
    while (true) {
      Item item;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) {
          break;
        }
        item = std::move(queue_.front());
        queue_.pop_front();
        space_.notify_one();
      }
      if (!first) {
        log << ",\n";
      }
      first = false;
      log << frameJson(item.result).dump();
      if (!item.annotated.empty()) {
        if (!video.isOpened()) {
          const int avc1 = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
          video.open(video_path_.string(), avc1,
                     config_.output.annotated_video_fps, item.annotated.size(),
                     true);
          if (!video.isOpened()) {
            const int mp4v = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            video.open(video_path_.string(), mp4v,
                       config_.output.annotated_video_fps,
                       item.annotated.size(), true);
          }
          if (!video.isOpened()) {
            throw std::runtime_error("unable to open annotated video " +
                                     video_path_.string());
          }
        }
        if (video.isOpened()) {
          video.write(item.annotated);
        }
      }
    }
    log << "\n  ]\n}\n";
    log.close();
    video.release();
    std::filesystem::rename(temporary_log_path_, log_path_);
  } catch (const std::exception &error) {
    {
      std::lock_guard lock(mutex_);
      worker_error_ = error.what();
      stopping_ = true;
    }
    space_.notify_all();
    ready_.notify_all();
  }
}

} // namespace flsloc
