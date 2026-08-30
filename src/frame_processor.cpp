#include "frame_processor.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
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

  TrackedFrame track(cv::Mat &image, double timestamp) {
    TrackedFrame frame;
    frame.markers =
        tracker.processFrame(image, timestamp, options.blob_area_threshold);
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
        current_distance(options.distance) {
    const MarkerGrid &grid = pipeline.grid();
    const uint32_t encodable_id_count = uint32_t{1} << options.payload_size;
    if (static_cast<uint32_t>(grid.numIds()) > encodable_id_count) {
      throw std::runtime_error(
          "Marker map needs " + std::to_string(grid.numIds()) +
          " IDs, but --payload-size " + std::to_string(options.payload_size) +
          " represents " + std::to_string(encodable_id_count));
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
  }

  FrameProcessingResult process(cv::Mat &image, double timestamp,
                                const AttitudeSample &attitude) override {
    FrameProcessingResult output;
    TrackedFrame frame = track(image, timestamp);
    id_assigner.forgetTracks(frame.markers.retired_track_ids);
    output.log = {{"poses", json::array()}, {"blobs", std::move(frame.blobs)}};

    std::vector<MarkerDetection> decoded_detections;
    json track_logs = json::array();
    for (const auto &marker : frame.markers.decoded_markers) {
      const bool eligible = isMarkerObservationEligible(
          marker.visible, marker.last_seen_age, options.grid_max_marker_age);
      track_logs.push_back({{"id", marker.id},
                            {"track_id", marker.track_id},
                            {"image_x", marker.x},
                            {"image_y", marker.y},
                            {"visible", marker.visible},
                            {"last_seen_age", marker.last_seen_age},
                            {"eligible_for_localization", eligible}});
      if (eligible) {
        MarkerDetection detection;
        detection.x = marker.x;
        detection.y = marker.y;
        detection.id = marker.id;
        detection.track_id = marker.track_id;
        detection.visible = marker.visible;
        detection.last_seen_age = marker.last_seen_age;
        decoded_detections.push_back(detection);
      }
    }
    std::vector<MarkerDetection> current_blobs;
    current_blobs.reserve(frame.markers.current_blobs.size());
    for (const auto &blob : frame.markers.current_blobs) {
      MarkerDetection detection;
      detection.x = blob.x;
      detection.y = blob.y;
      detection.id = blob.id;
      detection.track_id = blob.track_id;
      current_blobs.push_back(detection);
    }
    std::vector<MarkerDetection> detections = decoded_detections;
    GridIdAssignmentResult assignment;
    assignment.map_locked = id_assigner.mapLocked();
    bool assignment_attempted = false;

    LocalizationResult localization;
    std::string attitude_status = attitude.status;
    std::optional<cv::Matx33d> drone_to_world;
    if (attitude.valid) {
      drone_to_world =
          pose_math::rotationFromQuaternionXyzw(attitude.quaternion_xyzw);
      if (drone_to_world) {
        const cv::Matx33d grid_to_camera =
            pose_math::cameraToDroneRotation().t() * drone_to_world->t();
        assignment_attempted = true;
        assignment = id_assigner.assign(
            decoded_detections, current_blobs, camera_matrix,
            distortion_coefficients, grid_to_camera, current_distance);
        detections = assignment.detections;
        std::map<std::uint64_t, const MarkerDetection *> assigned_by_track;
        for (const auto &detection : detections) {
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
          blob_log["id_source"] = detection.inferred ? "map" : "decoder";
          if (detection.hasMapCell()) {
            blob_log["map_row"] = detection.map_row;
            blob_log["map_col"] = detection.map_col;
          }
        }
        localization = pipeline.localize(
            detections, camera_matrix, distortion_coefficients,
            grid_to_camera, current_distance, image.size());
      } else {
        attitude_status = "invalid";
      }
    }
    if (localization.pnp_solver.empty()) {
      localization.pnp_solver =
          options.grid_center_ap3p ? "ap3p" : "ippe_iterative";
      localization.lookup.required_marker_count =
          pipeline.grid().windowSize() * pipeline.grid().windowSize();
      localization.distance_used = current_distance;
      if (detections.empty()) {
        localization.status = LocalizationStatus::NO_DETECTIONS;
        localization.message = "no eligible marker detections";
      } else {
        localization.status = LocalizationStatus::NORMALIZATION_FAILED;
        localization.message = "shared attitude is " + attitude_status;
      }
    }
    output.log["blob_grid_localization"] = makeLocalizationLog(
        localization, detections, decoded_detections.size(),
        frame.markers.decoded_markers.size(), assignment,
        assignment_attempted,
        std::move(track_logs));
    json attitude_log = {{"source", "shared_memory"},
                         {"status", attitude_status},
                         {"valid", attitude.valid},
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
      used_map_cells.push_back(
          {{"row", marker.map_row}, {"col", marker.map_col}});
    }

    json pose_log = {{"camera_pose", true},
                     {"source", "blob_grid"},
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
    configuration["marker_grid"] = {
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
        {"center_window_ap3p", options.grid_center_ap3p}};
  }

private:
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

    for (const auto &marker : localization.relative_markers) {
      json marker_log = {{"id", marker.id},
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
      log["matched_markers"].push_back(
          {{"id", marker.id},
           {"image_x", marker.image_x},
           {"image_y", marker.image_y},
           {"relative_row", marker.relative_row},
           {"relative_col", marker.relative_col},
           {"map_row", marker.map_row},
           {"map_col", marker.map_col},
           {"global_position",
            {marker.global_x, marker.global_y, marker.global_z}}});
    }
    if (localization.lookup.relative_window_row >= 0) {
      log["window_match"] = {
          {"window_size", options.grid_window_size},
          {"relative_origin",
           {localization.lookup.relative_window_row,
            localization.lookup.relative_window_col}},
          {"map_origin",
           {localization.lookup.map_window_row,
            localization.lookup.map_window_col}},
          {"signature", localization.lookup.window_signature}};
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
