#include "frame_processor.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "application_options.h"
#include "aruco_tracker.h"
#include "camera_config.h"
#include "localization_pipeline.h"
#include "marker_tracker.h"
#include "pose_estimator.h"
#include "pose_math.h"
#include "pose_publisher.h"
#include "position_kalman_filter.h"

using json = nlohmann::json;

namespace {

constexpr uint16_t ARUCO_CAMERA_FILTER_ID = 0xFFFF;
constexpr uint16_t GRID_DRONE_FILTER_ID = 0xFFFE;
json workingRangeJson(const std::optional<WorkingRange> &range) {
  if (!range) {
    return nullptr;
  }
  return {{"min_distance", range->min_distance},
          {"max_distance", range->max_distance}};
}

const char *blobAnnotationState(const json &blob, bool rejected) {
  const std::string grid_type = blob.value("grid_type", "");
  const std::string id_source = blob.value("id_source", "");
  if (rejected || (blob.value("id", -1) >= 0 && grid_type == "unknown")) {
    return "rejected";
  }
  if (!id_source.empty() && id_source != "decoder") {
    return "inferred";
  }
  if (blob.value("decoder_suppressed", false)) {
    return "decoder_suppressed";
  }
  if (grid_type == "short_range") {
    return "decoded_short_range";
  }
  if (grid_type == "main") {
    return "decoded_main";
  }
  return "anonymous";
}

cv::Scalar blobAnnotationColor(const std::string &state) {
  if (state == "decoded_main") {
    return {0, 255, 0};
  }
  if (state == "decoded_short_range") {
    return {0, 165, 255};
  }
  if (state == "inferred") {
    return {255, 255, 0};
  }
  if (state == "decoder_suppressed") {
    return {128, 128, 128};
  }
  if (state == "rejected") {
    return {255, 0, 255};
  }
  return {0, 0, 255};
}

bool sameBlob(const json &blob, const MarkerDetection &detection) {
  const std::uint64_t track_id = blob.value("track_id", std::uint64_t{0});
  if (track_id != 0 && detection.track_id != 0) {
    return track_id == detection.track_id;
  }
  return std::abs(blob.value("x", 0.0F) - detection.x) < 0.5F &&
         std::abs(blob.value("y", 0.0F) - detection.y) < 0.5F;
}

void annotateGridBlobs(cv::Mat &image, json &blobs,
                       const std::vector<MarkerDetection> &detections,
                       const std::vector<MarkerDetection> &main_blobs,
                       const GridIdAssignmentResult &assignment,
                       bool assignment_attempted,
                       const LocalizationResult &localization) {
  std::vector<MarkerDetection> rejected;
  if (assignment_attempted && assignment.rejected_blob_count > 0) {
    for (const MarkerDetection &candidate : main_blobs) {
      const bool accepted = std::any_of(
          assignment.detections.begin(), assignment.detections.end(),
          [&](const MarkerDetection &detection) {
            return detection.track_id == candidate.track_id;
          });
      if (!accepted) {
        rejected.push_back(candidate);
      }
    }
  }
  for (const RelativeMarker &marker : localization.relative_markers) {
    if (!marker.accepted && marker.detection_index < detections.size()) {
      rejected.push_back(detections[marker.detection_index]);
    }
  }

  std::vector<MarkerDetection> used_for_pnp;
  for (const GlobalMarker &marker : localization.pose_markers) {
    if (marker.detection_index < detections.size()) {
      used_for_pnp.push_back(detections[marker.detection_index]);
    }
  }

  for (auto &blob : blobs) {
    const bool was_rejected =
        std::any_of(rejected.begin(), rejected.end(),
                    [&](const MarkerDetection &item) {
                      return sameBlob(blob, item);
                    });
    const bool used =
        std::any_of(used_for_pnp.begin(), used_for_pnp.end(),
                    [&](const MarkerDetection &item) {
                      return sameBlob(blob, item);
                    });
    const std::string state = blobAnnotationState(blob, was_rejected);
    blob["annotation_state"] = state;
    blob["used_for_pnp"] = used;
    const cv::Point center(cvRound(blob.at("x").get<float>()),
                           cvRound(blob.at("y").get<float>()));
    cv::circle(image, center, 10, blobAnnotationColor(state), 2, cv::LINE_AA);
    if (used) {
      cv::line(image, center + cv::Point(-5, 0), center + cv::Point(5, 0),
               cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      cv::line(image, center + cv::Point(0, -5), center + cv::Point(0, 5),
               cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      cv::line(image, center + cv::Point(-5, 0), center + cv::Point(5, 0),
               cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::line(image, center + cv::Point(0, -5), center + cv::Point(0, 5),
               cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
  }
}

std::vector<double> matVec3ToVector(const cv::Mat &value) {
  cv::Mat value_64f;
  value.convertTo(value_64f, CV_64F);
  return {value_64f.at<double>(0, 0), value_64f.at<double>(1, 0),
          value_64f.at<double>(2, 0)};
}

std::vector<double> vec3dToVector(const cv::Vec3d &value) {
  return {value[0], value[1], value[2]};
}

cv::Point2f centroid(const std::vector<cv::Point2f> &points) {
  cv::Point2f center;
  for (const auto &point : points) {
    center += point;
  }
  return center * (1.0F / static_cast<float>(points.size()));
}

void sortClockwise(std::vector<cv::Point2f> &points) {
  const cv::Point2f center = centroid(points);
  std::sort(points.begin(), points.end(), [&](const cv::Point2f &left,
                                              const cv::Point2f &right) {
    return std::atan2(left.y - center.y, left.x - center.x) <
           std::atan2(right.y - center.y, right.x - center.x);
  });
}

void annotateMarkerGroup(cv::Mat &image, int id,
                         const std::vector<cv::Point2f> &points,
                         const cv::Scalar &color, bool complete) {
  for (std::size_t i = 0; i < points.size(); ++i) {
    cv::circle(image, points[i], 8, color, -1);
    cv::putText(image, std::to_string(i),
                {static_cast<int>(points[i].x + 12),
                 static_cast<int>(points[i].y - 12)},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv::LINE_AA);
  }
  const cv::Point2f center = centroid(points);
  const std::string label =
      complete ? "ID: " + std::to_string(id)
               : "ID: " + std::to_string(id) + " (" +
                     std::to_string(points.size()) + "/4)";
  cv::putText(image, label,
              {static_cast<int>(center.x - 20),
               static_cast<int>(center.y - 20)},
              cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv::LINE_AA);
}

class PositionFilters {
public:
  explicit PositionFilters(const ApplicationOptions &options)
      : is_enabled(options.enable_kalman_filter),
        process_noise(options.kf_process_noise),
        measurement_noise(options.kf_measurement_noise) {}

  bool enabled() const { return is_enabled; }

  std::vector<double> apply(uint16_t id, const std::vector<double> &position,
                            double timestamp) {
    if (!is_enabled) {
      return position;
    }
    auto filter =
        filters.try_emplace(id, process_noise, measurement_noise).first;
    const Eigen::Vector3d measurement(position[0], position[1], position[2]);
    const Eigen::Vector3d filtered =
        filter->second.update(measurement, timestamp);
    return {filtered[0], filtered[1], filtered[2]};
  }

private:
  bool is_enabled;
  double process_noise;
  double measurement_noise;
  std::map<uint16_t, PositionKalmanFilter> filters;
};

PoseOutput makePose(const std::vector<double> &position,
                    const std::vector<double> &orientation) {
  return {true,           position[0],    position[1],   position[2],
          orientation[0], orientation[1], orientation[2]};
}

class ArucoFrameProcessor final : public FrameProcessor {
public:
  ArucoFrameProcessor(const ApplicationOptions &options,
                      const cv::Mat &camera_matrix,
                      const cv::Mat &distortion_coefficients)
      : options(options), camera_matrix(camera_matrix),
        distortion_coefficients(distortion_coefficients), filters(options) {
    std::string dictionary;
    double marker_size = 0.0;
    if (!readArucoConfig(options.config_file, dictionary, marker_size,
                         known_markers)) {
      throw std::runtime_error("Failed to read ArUco configuration");
    }
    if (known_markers.empty()) {
      throw std::runtime_error("No ArUco markers defined in config");
    }
    tracker =
        std::make_unique<ArucoTracker>(dictionary, marker_size, known_markers);
    std::cout << "ArUco detection mode ENABLED" << std::endl;
  }

  FrameProcessingResult process(cv::Mat &image, double timestamp,
                                const AttitudeSample &) override {
    FrameProcessingResult output;
    output.log = {{"poses", json::array()}};
    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const auto detection =
        tracker->processFrame(image, camera_matrix, distortion_coefficients);
    if (!detection.valid) {
      return output;
    }

    const std::vector<double> camera_position =
        matVec3ToVector(detection.tvec_world);
    const std::vector<double> camera_orientation =
        vec3dToVector(detection.roll_pitch_yaw);
    if (options.print_logs) {
      std::cout << "[ArUco] Camera world pos: [" << camera_position[0] << ", "
                << camera_position[1] << ", " << camera_position[2]
                << "]  RPY: [" << camera_orientation[0] << ", "
                << camera_orientation[1] << ", " << camera_orientation[2]
                << "]  markers: " << detection.markers_used
                << "  reproj_err: " << detection.reprojection_error
                << std::endl;
    }

    json camera_pose = {{"camera_pose", true},
                        {"camera_position", camera_position},
                        {"camera_orientation", camera_orientation},
                        {"markers_used", detection.markers_used},
                        {"detected_ids", detection.detected_ids},
                        {"reprojection_error", detection.reprojection_error}};
    std::vector<double> published_position = camera_position;
    if (filters.enabled()) {
      published_position =
          filters.apply(ARUCO_CAMERA_FILTER_ID, camera_position, timestamp);
      camera_pose["camera_position_filtered"] = published_position;
    }
    output.log["poses"].push_back(std::move(camera_pose));

    for (int marker_id : detection.detected_ids) {
      const auto marker = known_markers.find(marker_id);
      if (marker == known_markers.end()) {
        continue;
      }
      const cv::Mat &transform = marker->second.T_world_marker;
      const cv::Mat rotation = transform(cv::Rect(0, 0, 3, 3));
      const cv::Mat translation = transform(cv::Rect(3, 0, 1, 3));
      output.log["poses"].push_back(
          {{"marker_pose", true},
           {"marker_id", marker_id},
           {"marker_position", matVec3ToVector(translation)},
           {"marker_orientation",
            vec3dToVector(pose_math::rpyFromRotation(rotation))}});
    }

    output.pose = makePose(published_position, camera_orientation);
    return output;
  }

  void appendConfiguration(json &configuration, cv::Size) const override {
    configuration["aruco_mode"] = true;
    configuration["blob_grid_localization_enabled"] = false;
  }

private:
  const ApplicationOptions &options;
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  std::map<int, ArucoTracker::MarkerWorldPose> known_markers;
  std::unique_ptr<ArucoTracker> tracker;
  PositionFilters filters;
};

class BlobFrameProcessor : public FrameProcessor {
public:
  BlobFrameProcessor(const ApplicationOptions &options,
                     const cv::Mat &camera_matrix,
                     const cv::Mat &distortion_coefficients)
      : options(options), camera_matrix(camera_matrix),
        distortion_coefficients(distortion_coefficients),
        tracker(3000.0 / options.encoder_frame_rate, options.payload_size,
                options.tracking_threshold, options.sync_threshold,
                options.static_markers_mode, options.validate_mode,
                options.dark_blob_intensity,
                !options.grid_map_file.empty()),
        filters(options) {}

protected:
  struct TrackedFrame {
    MarkerTracker::Result markers;
    json blobs = json::array();
  };

  TrackedFrame track(
      cv::Mat &image, double timestamp,
      const std::set<std::uint64_t> &decode_ignored_track_ids = {}) {
    TrackedFrame frame;
    frame.markers = tracker.processFrame(
        image, timestamp, options.blob_area_threshold,
        decode_ignored_track_ids);
    for (const auto &blob : frame.markers.current_blobs) {
      frame.blobs.push_back({{"x", blob.x},
                             {"y", blob.y},
                             {"id", blob.id},
                             {"decoder_id", blob.id},
                             {"track_id", blob.track_id}});
    }
    return frame;
  }

  const ApplicationOptions &options;
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  MarkerTracker tracker;
  PositionFilters filters;
};

class LegacyBlobFrameProcessor final : public BlobFrameProcessor {
public:
  LegacyBlobFrameProcessor(const ApplicationOptions &options,
                           const cv::Mat &camera_matrix,
                           const cv::Mat &distortion_coefficients,
                           std::vector<cv::Point3f> marker_points)
      : BlobFrameProcessor(options, camera_matrix, distortion_coefficients),
        marker_points(std::move(marker_points)) {}

  FrameProcessingResult process(cv::Mat &image, double timestamp,
                                const AttitudeSample &) override {
    FrameProcessingResult output;
    TrackedFrame frame = track(image, timestamp);
    output.log = {{"poses", json::array()}, {"blobs", std::move(frame.blobs)}};

    if (options.validate_mode) {
      return output;
    }

    std::map<int, std::vector<cv::Point2f>> groups;
    for (const auto &marker : frame.markers.decoded_markers) {
      if (marker.id >= 0) {
        groups[marker.id].emplace_back(marker.x, marker.y);
      }
    }

    for (auto &[marker_id, points] : groups) {
      sortClockwise(points);
      if (points.size() < 4) {
        annotateMarkerGroup(image, marker_id, points,
                            cv::Scalar(0, 165, 255), false);
        continue;
      }
      if (points.size() != 4 || marker_points.size() != 4) {
        continue;
      }

      annotateMarkerGroup(image, marker_id, points, cv::Scalar(0, 255, 0),
                          true);
      const pose_estimation::PnpEstimate pose = pose_estimation::solveAp3p(
          marker_points, points, camera_matrix, distortion_coefficients);
      if (!pose.valid) {
        continue;
      }

      const cv::Vec3d camera_rpy =
          pose_math::rpyFromRotation(pose.camera_to_object_rotation);
      const cv::Vec3d marker_rpy =
          pose_math::rpyFromRotation(pose.object_to_camera_rotation);
      const std::vector<double> camera_position =
          matVec3ToVector(pose.camera_position_object);
      const std::vector<double> camera_orientation =
          vec3dToVector(camera_rpy);
      const std::vector<double> marker_position =
          matVec3ToVector(pose.object_to_camera_translation);
      const std::vector<double> marker_orientation =
          vec3dToVector(marker_rpy);

      if (options.print_logs) {
        std::cout << "ID: " << marker_id << " Camera Position: "
                  << pose.camera_position_object.t() << " Camera RPY: ["
                  << camera_orientation[0] << ", " << camera_orientation[1]
                  << ", " << camera_orientation[2]
                  << "] Marker Position: "
                  << pose.object_to_camera_translation.t()
                  << " Marker RPY: [" << marker_orientation[0] << ", "
                  << marker_orientation[1] << ", " << marker_orientation[2]
                  << "]" << std::endl;
      }

      json pose_log = {{"marker_id", marker_id},
                       {"camera_position", camera_position},
                       {"camera_orientation", camera_orientation},
                       {"marker_position", marker_position},
                       {"marker_orientation", marker_orientation}};
      std::vector<double> published_position = marker_position;
      if (filters.enabled()) {
        published_position = filters.apply(
            static_cast<uint16_t>(marker_id), marker_position, timestamp);
        pose_log["marker_position_filtered"] = published_position;
      }
      output.log["poses"].push_back(std::move(pose_log));

      if (!output.pose.valid &&
          (options.target_id == -1 || marker_id == options.target_id)) {
        output.pose = makePose(published_position, marker_orientation);
      }
    }
    return output;
  }

  void appendConfiguration(json &configuration, cv::Size) const override {
    configuration["aruco_mode"] = false;
    configuration["blob_grid_localization_enabled"] = false;
  }

private:
  std::vector<cv::Point3f> marker_points;
};

class GridBlobFrameProcessor final : public BlobFrameProcessor {
public:
  GridBlobFrameProcessor(const ApplicationOptions &options,
                         const cv::Mat &camera_matrix,
                         const cv::Mat &distortion_coefficients)
      : BlobFrameProcessor(options, camera_matrix, distortion_coefficients),
        pipeline(options.grid_map_file, options.grid_window_size,
                 makeGeometry(options), options.grid_center_ap3p),
        id_assigner(pipeline.grid(), pipeline.geometry()),
        cross_grid_assigner(pipeline.grid(), pipeline.shortRangeGrid(),
                            pipeline.geometry()),
        main_marker_ids(pipeline.grid().markerIds()),
        short_range_marker_ids(pipeline.shortRangeGrid().markerIds()),
        current_distance(options.distance) {
    const MarkerGrid &grid = pipeline.grid();
    const uint32_t encodable_id_count = uint32_t{1} << options.payload_size;
    int maximum_marker_id = *main_marker_ids.rbegin();
    if (!short_range_marker_ids.empty()) {
      maximum_marker_id =
          std::max(maximum_marker_id, *short_range_marker_ids.rbegin());
    }
    if (maximum_marker_id < 0 ||
        static_cast<uint32_t>(maximum_marker_id) >= encodable_id_count) {
      throw std::runtime_error(
          "Marker map uses ID " + std::to_string(maximum_marker_id) +
          ", but --payload-size " + std::to_string(options.payload_size) +
          " represents IDs [0, " +
          std::to_string(encodable_id_count - 1) + "]");
    }
    if (grid.uniqueWindowCount() != grid.totalWindowCount()) {
      throw std::runtime_error(
          "Marker map has only " + std::to_string(grid.uniqueWindowCount()) +
          "/" + std::to_string(grid.totalWindowCount()) + " unique " +
          std::to_string(options.grid_window_size) + "x" +
          std::to_string(options.grid_window_size) +
          " windows; localization would be ambiguous");
    }
    std::cout << "Grid map localization ENABLED: " << options.grid_map_file
              << " (" << grid.rows() << 'x' << grid.cols() << ", window "
              << options.grid_window_size << 'x' << options.grid_window_size
              << ", spacing " << grid.cellSpacing() << " m, PnP "
              << (options.grid_center_ap3p ? "center 2x2 AP3P"
                                           : "all-marker IPPE+iterative")
              << ')' << std::endl;
    const ShortRangeMarkerGrid &short_range = pipeline.shortRangeGrid();
    if (short_range.enabled()) {
      std::cout << "Short-range marker localization ENABLED: "
                << short_range.tiles().size() << " tiles, window "
                << short_range.windowSize() << 'x'
                << short_range.windowSize() << ", spacing "
                << short_range.cellSpacing() << " m" << std::endl;
    }
    if (short_range.enabled()) {
      std::cout << "Shared-memory grid selection ENABLED" << std::endl;
    }
  }

  FrameProcessingResult process(cv::Mat &image, double timestamp,
                                const AttitudeSample &attitude) override {
    FrameProcessingResult output;
    const GridSelectionDecision grid_selection = selectGrid(attitude);
    const bool main_grid_selected =
        grid_selection.grid_type == "main";
    const bool short_grid_selected =
        grid_selection.grid_type == "short_range";
    const bool visible_selected_anchor = std::any_of(
        visible_tracks.begin(), visible_tracks.end(),
        [&](const std::uint64_t track_id) {
          return cross_grid_assigner.hasTrack(track_id,
                                              short_grid_selected) ||
                 (!short_grid_selected && id_assigner.hasTrack(track_id));
        });
    const bool selected_grid_locked =
        short_grid_selected ? cross_grid_assigner.hasGridTracks(true)
                            : id_assigner.mapLocked() ||
                                  cross_grid_assigner.hasGridTracks(false);
    // A cached but no-longer-visible anchor must not prevent undecoded tracks
    // from recovering through the normal decoder.
    const bool map_locked_before_tracking =
        selected_grid_locked && visible_selected_anchor;
    const std::set<std::uint64_t> decode_ignored_tracks =
        decoderIgnoredTracksForGridSelection(
            short_grid_selected, map_locked_before_tracking, seen_tracks,
            known_main_tracks, known_short_range_tracks);
    const std::size_t decoder_suppressed_track_count =
        decode_ignored_tracks.size();
    TrackedFrame frame =
        track(image, timestamp, decode_ignored_tracks);
    id_assigner.forgetTracks(frame.markers.retired_track_ids);
    cross_grid_assigner.forgetTracks(frame.markers.retired_track_ids);
    for (const std::uint64_t track_id : frame.markers.retired_track_ids) {
      seen_tracks.erase(track_id);
      visible_tracks.erase(track_id);
      known_main_tracks.erase(track_id);
      known_short_range_tracks.erase(track_id);
    }
    visible_tracks.clear();
    for (const auto &blob : frame.markers.current_blobs) {
      seen_tracks.insert(blob.track_id);
      visible_tracks.insert(blob.track_id);
    }
    for (const auto &marker : frame.markers.decoded_markers) {
      seen_tracks.insert(marker.track_id);
    }
    output.log = {{"poses", json::array()}, {"blobs", std::move(frame.blobs)}};

    std::vector<MarkerDetection> decoded_main_detections;
    std::vector<MarkerDetection> decoded_short_range_detections;
    std::size_t tracked_main_marker_count = 0;
    std::size_t tracked_short_range_marker_count = 0;
    std::size_t visible_short_range_marker_count = 0;
    json track_logs = json::array();
    for (const auto &marker : frame.markers.decoded_markers) {
      const bool eligible = isMarkerObservationEligible(
          marker.visible, marker.last_seen_age, options.grid_max_marker_age);
      const bool is_main = main_marker_ids.count(marker.id) != 0;
      const bool is_short_range =
          short_range_marker_ids.count(marker.id) != 0;
      const bool decoder_suppressed =
          decode_ignored_tracks.count(marker.track_id) != 0;
      if (is_main) {
        known_main_tracks.insert(marker.track_id);
        known_short_range_tracks.erase(marker.track_id);
      }
      if (is_short_range) {
        known_short_range_tracks.insert(marker.track_id);
        known_main_tracks.erase(marker.track_id);
      }
      const char *grid_type = is_main          ? "main"
                              : is_short_range ? "short_range"
                                               : "unknown";
      tracked_main_marker_count += is_main ? 1 : 0;
      tracked_short_range_marker_count += is_short_range ? 1 : 0;
      visible_short_range_marker_count +=
          is_short_range && marker.visible ? 1 : 0;
      track_logs.push_back({{"id", marker.id},
                            {"track_id", marker.track_id},
                            {"grid_type", grid_type},
                            {"image_x", marker.x},
                            {"image_y", marker.y},
                            {"visible", marker.visible},
                            {"last_seen_age", marker.last_seen_age},
                            {"decoder_suppressed", decoder_suppressed},
                            {"eligible_for_localization",
                             eligible && (is_main || is_short_range)}});
      if (eligible && (is_main || is_short_range)) {
        MarkerDetection detection;
        detection.x = marker.x;
        detection.y = marker.y;
        detection.id = marker.id;
        detection.track_id = marker.track_id;
        detection.visible = marker.visible;
        detection.last_seen_age = marker.last_seen_age;
        (is_short_range ? decoded_short_range_detections
                        : decoded_main_detections)
            .push_back(detection);
      }
    }
    std::vector<MarkerDetection> current_grid_blobs;
    std::vector<MarkerDetection> current_main_blobs;
    current_grid_blobs.reserve(frame.markers.current_blobs.size());
    current_main_blobs.reserve(frame.markers.current_blobs.size());
    for (const auto &blob : frame.markers.current_blobs) {
      MarkerDetection detection;
      detection.x = blob.x;
      detection.y = blob.y;
      detection.id = blob.id;
      detection.track_id = blob.track_id;
      current_grid_blobs.push_back(detection);
      const bool is_main = main_marker_ids.count(blob.id) != 0;
      const bool is_short_range = short_range_marker_ids.count(blob.id) != 0;
      const bool known_short_range =
          known_short_range_tracks.count(blob.track_id) != 0;
      if (is_main ||
          (blob.id < 0 && !known_short_range &&
           (main_grid_selected || visible_short_range_marker_count == 0))) {
        current_main_blobs.push_back(detection);
      }
      for (auto &blob_log : output.log["blobs"]) {
        if (blob_log.at("track_id").get<std::uint64_t>() != blob.track_id) {
          continue;
        }
        blob_log["decoder_suppressed"] =
            decode_ignored_tracks.count(blob.track_id) != 0;
        if (is_main) {
          blob_log["grid_type"] = "main";
        } else if (is_short_range) {
          blob_log["grid_type"] = "short_range";
        } else if (blob.id >= 0) {
          blob_log["grid_type"] = "unknown";
        }
        break;
      }
    }
    const std::size_t decoded_main_marker_count =
        decoded_main_detections.size();
    const std::size_t decoded_short_range_marker_count =
        decoded_short_range_detections.size();
    GridIdAssignmentResult cross_main_assignment;
    GridIdAssignmentResult cross_short_assignment;
    const auto merge_cross_assignments = [](
        std::vector<MarkerDetection> &into,
        const GridIdAssignmentResult &from) {
      for (const MarkerDetection &detection : from.detections) {
        const auto existing = std::find_if(
            into.begin(), into.end(), [&](const MarkerDetection &candidate) {
              return candidate.track_id == detection.track_id;
            });
        if (existing == into.end()) {
          into.push_back(detection);
        } else if (existing->id == detection.id) {
          *existing = detection;
        }
      }
    };
    const auto record_cross_assignments = [&](bool short_range,
                                               const GridIdAssignmentResult &result) {
      for (const MarkerDetection &detection : result.detections) {
        std::set<std::uint64_t> &selected =
            short_range ? known_short_range_tracks : known_main_tracks;
        std::set<std::uint64_t> &other =
            short_range ? known_main_tracks : known_short_range_tracks;
        selected.insert(detection.track_id);
        other.erase(detection.track_id);
        for (auto &blob_log : output.log["blobs"]) {
          if (blob_log.at("track_id").get<std::uint64_t>() !=
              detection.track_id) {
            continue;
          }
          blob_log["id"] = detection.id;
          blob_log["grid_type"] = short_range ? "short_range" : "main";
          blob_log["id_source"] =
              detection.inferred ? "cross_grid" : "decoder";
          if (detection.hasMapCell()) {
            blob_log["map_row"] = detection.map_row;
            blob_log["map_col"] = detection.map_col;
          }
          break;
        }
      }
    };
    std::vector<MarkerDetection> detections =
        short_grid_selected ? decoded_short_range_detections
                            : decoded_main_detections;
    GridIdAssignmentResult assignment;
    assignment.map_locked = id_assigner.mapLocked();
    bool assignment_attempted = false;

    LocalizationResult localization;
    if (short_grid_selected) {
      localization.grid_type = "short_range";
    }
    std::string attitude_status = attitude.status;
    std::optional<cv::Matx33d> drone_to_world;
    if (attitude.valid) {
      drone_to_world =
          pose_math::rotationFromQuaternionXyzw(attitude.quaternion_xyzw);
      if (drone_to_world) {
        const cv::Matx33d grid_to_camera =
            pose_math::cameraToDroneRotation().t() * drone_to_world->t();
        // Existing target identities may use the latest range estimate. A new
        // cross-grid bootstrap waits below for this frame's selected-grid PnP.
        if (cross_grid_assigner.hasGridTracks(false)) {
          cross_main_assignment = cross_grid_assigner.assign(
              false, current_grid_blobs, camera_matrix,
              distortion_coefficients, grid_to_camera, current_distance);
          merge_cross_assignments(decoded_main_detections,
                                  cross_main_assignment);
          record_cross_assignments(false, cross_main_assignment);
        }
        if (cross_grid_assigner.hasGridTracks(true)) {
          cross_short_assignment = cross_grid_assigner.assign(
              true, current_grid_blobs, camera_matrix,
              distortion_coefficients, grid_to_camera, current_distance);
          merge_cross_assignments(decoded_short_range_detections,
                                  cross_short_assignment);
          record_cross_assignments(true, cross_short_assignment);
        }
        detections = short_grid_selected ? decoded_short_range_detections
                                         : decoded_main_detections;
        LocalizationResult short_range_localization;
        if (pipeline.shortRangeGrid().enabled() && !main_grid_selected) {
          short_range_localization = pipeline.localizeShortRange(
              decoded_short_range_detections, camera_matrix,
              distortion_coefficients, grid_to_camera, current_distance,
              image.size());
        }
        if (short_grid_selected ||
            short_range_localization.lookup.status ==
                GridLookupStatus::UNIQUE) {
          localization = std::move(short_range_localization);
          detections = decoded_short_range_detections;
        } else if (!short_grid_selected) {
          assignment_attempted = true;
          assignment = id_assigner.assign(
              decoded_main_detections, current_main_blobs, camera_matrix,
              distortion_coefficients, grid_to_camera, current_distance);
          detections = assignment.detections;
          std::map<std::uint64_t, const MarkerDetection *> assigned_by_track;
          for (const auto &detection : detections) {
            if (main_marker_ids.count(detection.id) != 0) {
              known_main_tracks.insert(detection.track_id);
            }
            assigned_by_track[detection.track_id] = &detection;
          }
          for (auto &blob_log : output.log["blobs"]) {
            const auto assigned = assigned_by_track.find(
                blob_log.at("track_id").get<std::uint64_t>());
            if (assigned == assigned_by_track.end()) {
              continue;
            }
            const MarkerDetection &detection = *assigned->second;
            blob_log["id"] = detection.id;
            blob_log["grid_type"] = "main";
            blob_log["id_source"] = detection.inferred ? "map" : "decoder";
            if (detection.hasMapCell()) {
              blob_log["map_row"] = detection.map_row;
              blob_log["map_col"] = detection.map_col;
            }
          }
          localization = pipeline.localize(
              detections, camera_matrix, distortion_coefficients,
              grid_to_camera, current_distance, image.size());
        }
        if (localization.pose_valid &&
            cross_grid_assigner.rememberUnique(localization.lookup,
                                               detections)) {
          const double refined_distance =
              localization.camera_to_plane_distance;
          if (short_grid_selected) {
            cross_main_assignment = cross_grid_assigner.assign(
                false, current_grid_blobs, camera_matrix,
                distortion_coefficients, grid_to_camera, refined_distance);
            record_cross_assignments(false, cross_main_assignment);
          } else {
            cross_short_assignment = cross_grid_assigner.assign(
                true, current_grid_blobs, camera_matrix,
                distortion_coefficients, grid_to_camera, refined_distance);
            record_cross_assignments(true, cross_short_assignment);
          }
        }
      } else {
        attitude_status = "invalid";
      }
    }
    if (localization.pnp_solver.empty()) {
      const int active_window_size =
          localization.grid_type == "short_range"
              ? pipeline.shortRangeGrid().windowSize()
              : pipeline.grid().windowSize();
      localization.pnp_solver = options.grid_center_ap3p &&
                                        active_window_size == 2
                                    ? "ap3p"
                                    : "ippe_iterative";
      localization.lookup.required_marker_count =
          active_window_size * active_window_size;
      localization.distance_used = current_distance;
      if (detections.empty()) {
        localization.status = LocalizationStatus::NO_DETECTIONS;
        localization.message = "no eligible marker detections";
      } else {
        localization.status = LocalizationStatus::NORMALIZATION_FAILED;
        localization.message = "shared attitude is " + attitude_status;
      }
    }
    if (localization.grid_type == "short_range") {
      for (const auto &marker : localization.lookup.markers) {
        if (marker.detection_index >= detections.size()) {
          continue;
        }
        const std::uint64_t track_id =
            detections[marker.detection_index].track_id;
        for (auto &blob_log : output.log["blobs"]) {
          if (blob_log.at("track_id").get<std::uint64_t>() != track_id) {
            continue;
          }
          blob_log["grid_type"] = "short_range";
          blob_log["tile_i"] = marker.tile_i;
          blob_log["tile_j"] = marker.tile_j;
          blob_log["local_i"] = marker.local_i;
          blob_log["local_j"] = marker.local_j;
          break;
        }
      }
    }
    const bool short_range_selected =
        localization.grid_type == "short_range";
    output.log["blob_grid_localization"] = makeLocalizationLog(
        localization, detections,
        short_range_selected ? decoded_short_range_marker_count
                             : decoded_main_marker_count,
        short_range_selected ? tracked_short_range_marker_count
                             : tracked_main_marker_count,
        assignment, assignment_attempted, std::move(track_logs));
    const auto cross_assignment_log = [](const GridIdAssignmentResult &value) {
      return json{{"map_locked", value.map_locked},
                  {"alignment_valid", value.alignment_valid},
                  {"inferred_marker_count", value.inferred_marker_count},
                  {"rejected_blob_count", value.rejected_blob_count},
                  {"message", value.message}};
    };
    output.log["blob_grid_localization"]["cross_grid_id_assignment"] = {
        {"main", cross_assignment_log(cross_main_assignment)},
        {"short_range", cross_assignment_log(cross_short_assignment)}};
    output.log["blob_grid_localization"]["grid_selection"] = {
        {"enabled", pipeline.shortRangeGrid().enabled()},
        {"source", "shared_memory"},
        {"selected_grid", grid_selection.grid_type},
        {"resolved_grid", localization.grid_type},
        {"reason", grid_selection.reason},
        {"use_short_range", attitude.use_short_range},
        {"distance", current_distance},
        {"decoder_suppressed_track_count",
         decoder_suppressed_track_count},
        {"decoder_gate_mode",
         map_locked_before_tracking ? "non_selected_and_unknown"
                                    : "known_non_selected"},
        {"map_locked_before_tracking", map_locked_before_tracking},
        {"selected_grid_has_cached_identity", selected_grid_locked},
        {"working_ranges",
         {{"main", workingRangeJson(pipeline.grid().workingRange())},
          {"short_range",
           workingRangeJson(pipeline.shortRangeGrid().workingRange())}}}};
    json attitude_log = {{"source", "shared_memory"},
                         {"status", attitude_status},
                         {"valid", attitude.valid},
                         {"use_short_range", attitude.use_short_range},
                         {"sequence", attitude.sequence},
                         {"host_timestamp", attitude.host_timestamp},
                         {"age_seconds", attitude.age_seconds}};
    if (attitude.valid) {
      attitude_log["quaternion_xyzw"] = {
          attitude.quaternion_xyzw[0], attitude.quaternion_xyzw[1],
          attitude.quaternion_xyzw[2], attitude.quaternion_xyzw[3]};
    }
    output.log["blob_grid_localization"]["attitude"] =
        std::move(attitude_log);

    annotateGridBlobs(image, output.log["blobs"], detections,
                      current_main_blobs, assignment, assignment_attempted,
                      localization);

    if (options.print_logs &&
        localization.status != LocalizationStatus::NO_DETECTIONS) {
      std::cout << "[Blob grid] " << localizationStatusName(localization.status)
                << ": " << localization.message;
      if (localization.reprojection_error >= 0.0) {
        std::cout << "  reproj_err=" << localization.reprojection_error;
      }
      std::cout << std::endl;
    }
    if (!localization.pose_valid) {
      return output;
    }
    current_distance = localization.camera_to_plane_distance;

    const cv::Vec3d camera_position_vector(
        localization.camera_position_world.at<double>(0, 0),
        localization.camera_position_world.at<double>(1, 0),
        localization.camera_position_world.at<double>(2, 0));
    const std::vector<double> camera_position =
        vec3dToVector(camera_position_vector);
    const std::vector<double> camera_orientation =
        vec3dToVector(localization.camera_roll_pitch_yaw);
    const cv::Vec3d camera_offset_drone(options.camera_offset_drone[0],
                                        options.camera_offset_drone[1],
                                        options.camera_offset_drone[2]);
    const cv::Vec3d drone_position_vector =
        camera_position_vector - *drone_to_world * camera_offset_drone;
    const std::vector<double> drone_position =
        vec3dToVector(drone_position_vector);
    const std::vector<double> drone_orientation = vec3dToVector(
        pose_math::rpyFromRotation(cv::Mat(*drone_to_world)));
    const std::vector<double> marker_position =
        matVec3ToVector(localization.tvec_world_to_camera);
    std::vector<int> used_marker_ids;
    std::vector<json> used_map_cells;
    for (const auto &marker : localization.pose_markers) {
      used_marker_ids.push_back(marker.id);
      json cell = {{"row", marker.map_row},
                   {"col", marker.map_col},
                   {"grid_type", marker.grid_type}};
      if (marker.grid_type == "short_range") {
        cell["tile_i"] = marker.tile_i;
        cell["tile_j"] = marker.tile_j;
        cell["local_i"] = marker.local_i;
        cell["local_j"] = marker.local_j;
      }
      used_map_cells.push_back(std::move(cell));
    }

    json pose_log = {{"camera_pose", true},
                     {"source", "blob_grid"},
                     {"grid_type", localization.grid_type},
                     {"camera_position", camera_position},
                     {"camera_orientation", camera_orientation},
                     {"drone_position", drone_position},
                     {"drone_orientation", drone_orientation},
                     {"marker_position", marker_position},
                     {"camera_to_plane_distance",
                      localization.camera_to_plane_distance},
                     {"pnp_solver", localization.pnp_solver},
                     {"markers_used", localization.pose_markers.size()},
                     {"used_marker_ids", used_marker_ids},
                     {"used_map_cells", used_map_cells},
                     {"reprojection_error", localization.reprojection_error}};
    if (localization.grid_type == "short_range") {
      pose_log["tile"] =
          {{"i", localization.tile_i}, {"j", localization.tile_j}};
    }
    std::vector<double> published_position = drone_position;
    if (filters.enabled()) {
      published_position =
          filters.apply(GRID_DRONE_FILTER_ID, drone_position, timestamp);
      pose_log["drone_position_filtered"] = published_position;
    }
    output.log["poses"].push_back(std::move(pose_log));
    output.pose = makePose(published_position, drone_orientation);
    return output;
  }

  void appendConfiguration(json &configuration,
                           cv::Size image_size) const override {
    configuration["aruco_mode"] = false;
    configuration["blob_grid_localization_enabled"] = true;
    const MarkerGrid &grid = pipeline.grid();
    const cv::Point3f origin = grid.gridOrigin();
    json marker_grid = {
        {"map_file", options.grid_map_file},
        {"rows", grid.rows()},
        {"cols", grid.cols()},
        {"num_ids", grid.numIds()},
        {"min_k", grid.minK()},
        {"cell_spacing", grid.cellSpacing()},
        {"grid_origin", {origin.x, origin.y, origin.z}},
        {"window_size", grid.windowSize()},
        {"total_windows", grid.totalWindowCount()},
        {"unique_windows", grid.uniqueWindowCount()},
        {"image_resolution_pixels", {image_size.width, image_size.height}},
        {"focal_length_pixels",
         {camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1)}},
        {"principal_point_pixels",
         {camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2)}},
        {"initial_distance", options.distance},
        {"latest_distance", current_distance},
        {"attitude_source", "shared_memory"},
        {"shared_attitude_memory", pose_shared_memory::name},
        {"max_attitude_age_seconds", options.max_attitude_age},
        {"camera_offset_drone", options.camera_offset_drone},
        {"shared_memory_position", "drone_position_world"},
        {"camera_to_drone_rotation",
         {{0.0, -1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}}},
        {"orientation_convention",
         "R_c_g=R_d_c^T*R_w_d(q)^T from shared attitude"},
        {"rounding_tolerance_cells", options.grid_rounding_tolerance},
        {"max_marker_age_seconds", options.grid_max_marker_age},
        {"pnp_solver", options.grid_center_ap3p ? "ap3p" : "ippe_iterative"},
        {"center_window_ap3p", options.grid_center_ap3p},
        {"grid_selection_source", "shared_memory"},
        {"blob_annotation_legend",
         {{"anonymous",
           {{"color_rgb", "#ff0000"},
            {"meaning", "detected, ID not assigned"}}},
          {"decoder_suppressed",
           {{"color_rgb", "#808080"},
            {"meaning", "decoder intentionally idle"}}},
          {"decoded_main",
           {{"color_rgb", "#00ff00"},
            {"meaning", "decoded main-grid marker"}}},
          {"decoded_short_range",
           {{"color_rgb", "#ffa500"},
            {"meaning", "decoded short-range marker"}}},
          {"inferred",
           {{"color_rgb", "#00ffff"},
            {"meaning", "ID inferred from grid alignment"}}},
          {"rejected",
           {{"color_rgb", "#ff00ff"},
            {"meaning", "excluded from localization"}}},
          {"pnp_used",
           {{"symbol", "+"}, {"meaning", "passed to the PnP solver"}}}}}};
    if (grid.workingRange()) {
      marker_grid["working_range"] = workingRangeJson(grid.workingRange());
    }
    const ShortRangeMarkerGrid &short_range = pipeline.shortRangeGrid();
    if (short_range.enabled()) {
      json short_range_config = {
          {"window_size", short_range.windowSize()},
          {"cell_spacing", short_range.cellSpacing()},
          {"marker_size", short_range.markerSize()},
          {"marker_ids", short_range.markerIds()},
          {"tiles", json::array()}};
      if (short_range.workingRange()) {
        short_range_config["working_range"] =
            workingRangeJson(short_range.workingRange());
      }
      for (const ShortRangeTile &tile : short_range.tiles()) {
        json tile_config = {{"i", tile.i},
                            {"j", tile.j},
                            {"signature", tile.signature},
                            {"markers", json::array()}};
        for (const ShortRangeMarker &marker : tile.markers) {
          tile_config["markers"].push_back(
              {{"local_i", marker.local_i},
               {"local_j", marker.local_j},
               {"id", marker.id},
               {"global_x", marker.global_position.x},
               {"global_y", marker.global_position.y},
               {"global_z", marker.global_position.z}});
        }
        short_range_config["tiles"].push_back(std::move(tile_config));
      }
      marker_grid["short_range"] = std::move(short_range_config);
    }
    configuration["marker_grid"] = std::move(marker_grid);
  }

private:
  GridSelectionDecision selectGrid(const AttitudeSample &attitude) const {
    if (!pipeline.shortRangeGrid().enabled()) {
      return {"main", "short_range_not_configured"};
    }
    if (!attitude.valid) {
      return {"main", "shared_memory_flag_unavailable"};
    }
    return attitude.use_short_range
               ? GridSelectionDecision{"short_range", "shared_memory_flag"}
               : GridSelectionDecision{"main", "shared_memory_flag"};
  }

  static CameraPlaneGeometry makeGeometry(const ApplicationOptions &options) {
    CameraPlaneGeometry geometry;
    geometry.rounding_tolerance = options.grid_rounding_tolerance;
    return geometry;
  }

  json makeLocalizationLog(
      const LocalizationResult &localization,
      const std::vector<MarkerDetection> &detections,
      std::size_t decoded_marker_count, std::size_t tracked_marker_count,
      const GridIdAssignmentResult &assignment, bool assignment_attempted,
      json track_logs) const {
    const bool map_assignments_used =
        !detections.empty() &&
        std::all_of(detections.begin(), detections.end(),
                    [](const MarkerDetection &detection) {
                      return detection.hasMapCell();
                    });
    const bool lookup_attempted =
        !map_assignments_used &&
        localization.status != LocalizationStatus::NO_DETECTIONS &&
        localization.status != LocalizationStatus::NORMALIZATION_FAILED;
    json log = {
        {"status", localizationStatusName(localization.status)},
        {"message", localization.message},
        {"grid_type", localization.grid_type},
        {"tracked_decoded_marker_count", tracked_marker_count},
        {"decoded_marker_count", decoded_marker_count},
        {"localization_marker_count", detections.size()},
        {"decoded_tracks", std::move(track_logs)},
        {"grid_id_assignment",
         {{"map_locked", assignment.map_locked},
          {"attempted", assignment_attempted},
          {"alignment_valid", assignment.alignment_valid},
          {"inferred_marker_count", assignment.inferred_marker_count},
          {"rejected_blob_count", assignment.rejected_blob_count},
          {"message", assignment.message}}},
        {"max_marker_age", options.grid_max_marker_age},
        {"required_marker_count", localization.lookup.required_marker_count},
        {"accepted_marker_count", localization.lookup.accepted_marker_count},
        {"complete_window_count", localization.lookup.complete_window_count},
        {"candidate_count", localization.lookup.candidate_count},
        {"best_match_count", localization.lookup.best_match_count},
        {"lookup_attempted", lookup_attempted},
        {"association_mode", map_assignments_used
                                 ? "map_aligned"
                                 : lookup_attempted ? "lookup" : "none"},
        {"lookup_status", lookup_attempted || map_assignments_used
                              ? gridLookupStatusName(localization.lookup.status)
                              : "not_attempted"},
        {"pose_valid", localization.pose_valid},
        {"pnp_solver", localization.pnp_solver},
        {"pnp_marker_count", localization.pose_markers.size()},
        {"distance_used", localization.distance_used},
        {"relative_markers", json::array()},
        {"matched_markers", json::array()}};
    if (localization.grid_type == "short_range" && localization.tile_i >= 0) {
      log["tile"] =
          {{"i", localization.tile_i}, {"j", localization.tile_j}};
    }

    for (const auto &marker : localization.relative_markers) {
      json marker_log = {{"id", marker.id},
                         {"grid_type", localization.grid_type},
                         {"image_x", marker.image_x},
                         {"image_y", marker.image_y},
                         {"relative_row", marker.row},
                         {"relative_col", marker.col},
                         {"row_coordinate", marker.row_coordinate},
                         {"col_coordinate", marker.col_coordinate},
                         {"row_rounding_error", marker.row_rounding_error},
                         {"col_rounding_error", marker.col_rounding_error},
                         {"accepted", marker.accepted}};
      if (marker.detection_index < detections.size()) {
        const auto &detection = detections[marker.detection_index];
        marker_log["track_id"] = detection.track_id;
        marker_log["id_source"] = detection.inferred ? "map" : "decoder";
        marker_log["visible"] = detection.visible;
        marker_log["last_seen_age"] = detection.last_seen_age;
      }
      log["relative_markers"].push_back(std::move(marker_log));
    }
    for (const auto &marker : localization.lookup.markers) {
      json marker_log = {
          {"id", marker.id},
          {"grid_type", marker.grid_type},
          {"image_x", marker.image_x},
          {"image_y", marker.image_y},
          {"relative_row", marker.relative_row},
          {"relative_col", marker.relative_col},
          {"map_row", marker.map_row},
          {"map_col", marker.map_col},
          {"global_position",
           {marker.global_x, marker.global_y, marker.global_z}}};
      if (marker.grid_type == "short_range") {
        marker_log["tile_i"] = marker.tile_i;
        marker_log["tile_j"] = marker.tile_j;
        marker_log["local_i"] = marker.local_i;
        marker_log["local_j"] = marker.local_j;
      }
      log["matched_markers"].push_back(std::move(marker_log));
    }
    if (localization.lookup.relative_window_row >= 0) {
      log["window_match"] = {
          {"grid_type", localization.grid_type},
          {"window_size",
           localization.grid_type == "short_range"
               ? pipeline.shortRangeGrid().windowSize()
               : options.grid_window_size},
          {"relative_origin",
           {localization.lookup.relative_window_row,
            localization.lookup.relative_window_col}},
          {"map_origin",
           {localization.lookup.map_window_row,
            localization.lookup.map_window_col}},
          {"signature", localization.lookup.window_signature}};
      if (localization.grid_type == "short_range") {
        log["window_match"]["tile"] =
            {{"i", localization.tile_i}, {"j", localization.tile_j}};
      }
    }
    if (localization.reprojection_error >= 0.0) {
      log["reprojection_error"] = localization.reprojection_error;
    }
    if (localization.camera_to_plane_distance > 0.0) {
      log["camera_to_plane_distance"] =
          localization.camera_to_plane_distance;
    }
    return log;
  }

  LocalizationPipeline pipeline;
  GridIdAssigner id_assigner;
  CrossGridIdAssigner cross_grid_assigner;
  std::set<int> main_marker_ids;
  std::set<int> short_range_marker_ids;
  std::set<std::uint64_t> seen_tracks;
  std::set<std::uint64_t> visible_tracks;
  std::set<std::uint64_t> known_main_tracks;
  std::set<std::uint64_t> known_short_range_tracks;
  double current_distance;
};

} // namespace

std::unique_ptr<FrameProcessor>
createFrameProcessor(const ApplicationOptions &options,
                     const cv::Mat &camera_matrix,
                     const cv::Mat &distortion_coefficients,
                     const std::vector<cv::Point3f> &marker_points) {
  if (options.aruco_mode) {
    return std::make_unique<ArucoFrameProcessor>(options, camera_matrix,
                                                 distortion_coefficients);
  }
  if (!options.grid_map_file.empty()) {
    return std::make_unique<GridBlobFrameProcessor>(options, camera_matrix,
                                                    distortion_coefficients);
  }
  return std::make_unique<LegacyBlobFrameProcessor>(
      options, camera_matrix, distortion_coefficients, marker_points);
}
